package simdvec

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"testing"
)

func roundTrip(t *testing.T, ix *Index) *Index {
	t.Helper()
	var buf bytes.Buffer
	n, err := ix.WriteTo(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if n != int64(buf.Len()) {
		t.Fatalf("WriteTo reported %d bytes, wrote %d", n, buf.Len())
	}
	got, err := Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func TestPersistRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, m := range []Metric{Cosine, DotProduct, Euclidean} {
		for _, dim := range []int{4, 768} {
			for _, n := range []int{1, 1000} {
				ix := New(dim, m)
				q := make([]float32, dim)
				for i := 0; i < n; i++ {
					v := make([]float32, dim)
					for j := range v {
						v[j] = rng.Float32()*2 - 1
					}
					if i == 0 {
						copy(q, v)
					}
					if err := ix.Add("id"+itoa(i), v); err != nil {
						t.Fatal(err)
					}
				}
				got := roundTrip(t, ix)

				if got.Dim() != ix.Dim() || got.Len() != ix.Len() || got.metric != ix.metric {
					t.Fatalf("header: dim %d/%d n %d/%d metric %v/%v",
						got.Dim(), ix.Dim(), got.Len(), ix.Len(), got.metric, ix.metric)
				}
				// The loaded index answers identically, which is the property
				// that matters: the file is a means, the search is the end.
				a, err := ix.Search(q, 10)
				if err != nil {
					t.Fatal(err)
				}
				b, err := got.Search(q, 10)
				if err != nil {
					t.Fatal(err)
				}
				if len(a) != len(b) {
					t.Fatalf("%v dim=%d n=%d: %d results before, %d after", m, dim, n, len(a), len(b))
				}
				for i := range a {
					if a[i].ID != b[i].ID || a[i].Score != b[i].Score {
						t.Fatalf("%v dim=%d n=%d result %d: %v before, %v after", m, dim, n, i, a[i], b[i])
					}
				}
			}
		}
	}
}

// The format's byte order is little-endian by decision, not by inheritance.
// This builds the bytes by hand with an explicit order and requires the
// encoder to agree, so a host of the other endianness cannot quietly write a
// different file -- which would not be an error anywhere, just different
// floats coming back.
func TestPersistByteOrderIsPinned(t *testing.T) {
	ix := New(2, DotProduct)
	if err := ix.Add("ab", []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if _, err := ix.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}

	var want bytes.Buffer
	want.WriteString("SIMDVEC1")
	le := binary.LittleEndian
	var u32 [4]byte
	var u64 [8]byte
	le.PutUint32(u32[:], 1)
	want.Write(u32[:]) // version
	le.PutUint32(u32[:], uint32(DotProduct))
	want.Write(u32[:]) // metric
	le.PutUint64(u64[:], 2)
	want.Write(u64[:]) // dim
	le.PutUint64(u64[:], 1)
	want.Write(u64[:]) // rows
	for _, f := range []float32{1, 2} {
		le.PutUint32(u32[:], math.Float32bits(f))
		want.Write(u32[:])
	}
	le.PutUint32(u32[:], math.Float32bits(5)) // norm: 1*1 + 2*2
	want.Write(u32[:])
	le.PutUint32(u32[:], 2) // id length
	want.Write(u32[:])
	want.WriteString("ab")

	if !bytes.Equal(buf.Bytes(), want.Bytes()) {
		t.Fatalf("format changed:\n got  %x\n want %x", buf.Bytes(), want.Bytes())
	}
}

// Every truncation of a valid file is an error, never a panic and never a
// half-loaded index. A file that stops mid-matrix must not leave the caller
// with rows that were never read.
func TestPersistTruncationIsCleanAtEveryLength(t *testing.T) {
	ix := New(4, Cosine)
	for i := 0; i < 20; i++ {
		if err := ix.Add("id"+itoa(i), []float32{float32(i), 1, 2, 3}); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if _, err := ix.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	full := buf.Bytes()
	for cut := 0; cut < len(full); cut++ {
		target := New(4, Cosine)
		if err := target.Add("keep", []float32{9, 9, 9, 9}); err != nil {
			t.Fatal(err)
		}
		_, err := target.ReadFrom(bytes.NewReader(full[:cut]))
		if err == nil {
			t.Fatalf("a %d-byte prefix of a %d-byte file loaded cleanly", cut, len(full))
		}
		// The failed load left the index exactly as it was.
		if target.Len() != 1 {
			t.Fatalf("cut %d: a failed load changed the index (Len=%d)", cut, target.Len())
		}
		got, err := target.Search([]float32{9, 9, 9, 9}, 1)
		if err != nil || len(got) != 1 || got[0].ID != "keep" {
			t.Fatalf("cut %d: the index is unusable after a failed load: %v %v", cut, got, err)
		}
	}
}

func TestPersistRejectsBadHeaders(t *testing.T) {
	good := func() []byte {
		ix := New(2, Cosine)
		ix.Add("a", []float32{1, 0})
		var b bytes.Buffer
		ix.WriteTo(&b)
		return b.Bytes()
	}
	for _, c := range []struct {
		name string
		mut  func([]byte) []byte
		want error
	}{
		{"empty", func([]byte) []byte { return nil }, ErrFormat},
		{"wrong magic", func(b []byte) []byte { c := append([]byte(nil), b...); copy(c, "NOTVEC01"); return c }, ErrFormat},
		{"future version", func(b []byte) []byte {
			c := append([]byte(nil), b...)
			binary.LittleEndian.PutUint32(c[8:12], 99)
			return c
		}, ErrVersion},
		{"unknown metric", func(b []byte) []byte {
			c := append([]byte(nil), b...)
			binary.LittleEndian.PutUint32(c[12:16], 77)
			return c
		}, ErrFormat},
		{"zero dimension", func(b []byte) []byte {
			c := append([]byte(nil), b...)
			binary.LittleEndian.PutUint64(c[16:24], 0)
			return c
		}, ErrFormat},
		// A header claiming more than the file can hold must be refused
		// rather than believed: this is the allocation a corrupt file uses to
		// take the process down.
		{"absurd dimension", func(b []byte) []byte {
			c := append([]byte(nil), b...)
			binary.LittleEndian.PutUint64(c[16:24], 1<<40)
			return c
		}, ErrFormat},
		{"absurd row count", func(b []byte) []byte {
			c := append([]byte(nil), b...)
			binary.LittleEndian.PutUint64(c[24:32], 1<<40)
			return c
		}, ErrFormat},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := Load(bytes.NewReader(c.mut(good())))
			if !errors.Is(err, c.want) {
				t.Fatalf("err %v, want %v", err, c.want)
			}
		})
	}
}

// An id may hold anything, including bytes a NUL-terminated format would eat.
func TestPersistIDsAreLengthPrefixed(t *testing.T) {
	ix := New(1, DotProduct)
	ids := []string{"", "a\x00b", "\xff\xfe", "with spaces", string(make([]byte, 300))}
	for _, id := range ids {
		if err := ix.Add(id, []float32{1}); err != nil {
			t.Fatal(err)
		}
	}
	got := roundTrip(t, ix)
	if got.Len() != len(ids) {
		t.Fatalf("Len %d, want %d", got.Len(), len(ids))
	}
	for i, want := range ids {
		if got.ids[i] != want {
			t.Fatalf("id %d round-tripped as %q, want %q", i, got.ids[i], want)
		}
	}
}

// Loading over a populated index replaces it rather than appending to it.
func TestPersistLoadReplaces(t *testing.T) {
	src := New(2, DotProduct)
	src.Add("new", []float32{1, 1})
	var buf bytes.Buffer
	src.WriteTo(&buf)

	dst := New(2, DotProduct)
	for i := 0; i < 5; i++ {
		dst.Add("old"+itoa(i), []float32{float32(i), 0})
	}
	if _, err := dst.ReadFrom(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
	if dst.Len() != 1 {
		t.Fatalf("Len %d after loading a one-row index over five rows", dst.Len())
	}
	got, err := dst.Search([]float32{1, 1}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != "new" {
		t.Fatalf("loaded index holds %v", got)
	}
}

// An empty index round-trips, since "save before anything is added" is a
// perfectly ordinary thing for a caller to do.
func TestPersistEmptyIndex(t *testing.T) {
	ix := New(16, Euclidean)
	got := roundTrip(t, ix)
	if got.Len() != 0 || got.Dim() != 16 || got.metric != Euclidean {
		t.Fatalf("empty round trip: n=%d dim=%d metric=%v", got.Len(), got.Dim(), got.metric)
	}
	if err := got.Add("a", make([]float32, 16)); err != nil {
		t.Fatalf("the loaded empty index does not accept adds: %v", err)
	}
}
