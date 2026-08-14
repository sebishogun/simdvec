package simdvec

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// The shipped contract, frozen. Every promise the README and the docs make is
// a test here, so a change that breaks one is a failing test rather than a
// surprise for whoever upgrades.
//
// This file was written to pass against the code as shipped. That is the
// point: it proves the documentation and the implementation already agree,
// and it is the baseline any later change is measured against.

func mustPanic(t *testing.T, why string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: no panic", why)
		}
	}()
	f()
}

func TestContractNewRejectsNonPositiveDim(t *testing.T) {
	mustPanic(t, "New(0)", func() { New(0, Cosine) })
	mustPanic(t, "New(-1)", func() { New(-1, Cosine) })
}

// An unknown metric is rejected at construction, for the same reason a
// non-positive dimension is: it is a programming error, it cannot come from
// run-time input, and the alternative was an index that silently answered a
// different question than the caller asked.
func TestContractNewRejectsUnknownMetric(t *testing.T) {
	for _, m := range []Metric{Metric(99), Metric(-1), Metric(3)} {
		mustPanic(t, "New with metric "+m.String(), func() { New(4, m) })
	}
	// The three real ones are accepted.
	for _, m := range []Metric{Cosine, DotProduct, Euclidean} {
		if ix := New(4, m); ix == nil {
			t.Fatalf("New rejected %v", m)
		}
	}
	if got := Metric(99).String(); got != "unknown" {
		t.Fatalf("Metric(99).String() = %q", got)
	}
}

func TestContractDimErrors(t *testing.T) {
	ix := New(4, Cosine)
	if err := ix.Add("a", []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	before := ix.Len()
	raw := append([]float32(nil), ix.data...)

	err := ix.Add("b", []float32{1, 2, 3})
	if !errors.Is(err, ErrDim) {
		t.Fatalf("Add wrong length: %v, want ErrDim", err)
	}
	// A failed Add appends nothing: the index is exactly as it was.
	if ix.Len() != before {
		t.Fatalf("Len %d after a failed Add, was %d", ix.Len(), before)
	}
	if len(ix.data) != len(raw) {
		t.Fatalf("a failed Add changed the matrix length: %d, was %d", len(ix.data), len(raw))
	}
	for i := range raw {
		if ix.data[i] != raw[i] {
			t.Fatalf("a failed Add modified the matrix at %d", i)
		}
	}

	if _, err := ix.Search([]float32{1, 2, 3}, 1); !errors.Is(err, ErrDim) {
		t.Fatalf("Search wrong length: %v, want ErrDim", err)
	}
}

func TestContractEmptyAndK(t *testing.T) {
	ix := New(3, Cosine)
	// An empty index returns nothing, not an error.
	got, err := ix.Search([]float32{1, 0, 0}, 5)
	if err != nil || got != nil {
		t.Fatalf("empty index: %v %v", got, err)
	}
	for i := 0; i < 3; i++ {
		if err := ix.Add(string(rune('a'+i)), []float32{float32(i + 1), 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	// k <= 0 returns nothing.
	for _, k := range []int{0, -1, -100} {
		got, err := ix.Search([]float32{1, 0, 0}, k)
		if err != nil || got != nil {
			t.Fatalf("k=%d: %v %v", k, got, err)
		}
	}
	// k > n clamps to n.
	got, err = ix.Search([]float32{1, 0, 0}, 99)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("k=99 over 3 vectors returned %d", len(got))
	}
}

func TestContractDuplicateIDsAndZeroVectors(t *testing.T) {
	ix := New(2, Cosine)
	for i := 0; i < 3; i++ {
		if err := ix.Add("same", []float32{1, 0}); err != nil {
			t.Fatal(err)
		}
	}
	if ix.Len() != 3 {
		t.Fatalf("duplicate ids collapsed: Len=%d", ix.Len())
	}
	got, err := ix.Search([]float32{1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("%d results for 3 duplicate rows", len(got))
	}
	// A zero vector is accepted; under cosine it has no direction, and the
	// contract is that it does not error or panic.
	zero := New(2, Cosine)
	if err := zero.Add("z", []float32{0, 0}); err != nil {
		t.Fatalf("zero vector rejected: %v", err)
	}
	if _, err := zero.Search([]float32{0, 0}, 1); err != nil {
		t.Fatalf("zero query rejected: %v", err)
	}
}

// Ownership: Search must not modify the caller's slice. Cosine normalises the
// query, and normalising in place would silently rewrite the caller's data.
func TestContractSearchDoesNotTouchTheQuery(t *testing.T) {
	for _, m := range []Metric{Cosine, DotProduct, Euclidean} {
		ix := New(4, m)
		for i := 0; i < 5; i++ {
			if err := ix.Add("v", []float32{float32(i), 1, 2, 3}); err != nil {
				t.Fatal(err)
			}
		}
		q := []float32{3, 4, 0, 12}
		before := append([]float32(nil), q...)
		if _, err := ix.Search(q, 3); err != nil {
			t.Fatal(err)
		}
		for i := range q {
			if q[i] != before[i] {
				t.Fatalf("%v: Search modified the query at %d: %v, was %v", m, i, q, before)
			}
		}
	}
}

// Ordering is best-first, and best means the largest score for the similarity
// metrics and the smallest distance for Euclidean.
func TestContractOrderingIsBestFirst(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, m := range []Metric{Cosine, DotProduct, Euclidean} {
		ix := New(8, m)
		for i := 0; i < 50; i++ {
			v := make([]float32, 8)
			for j := range v {
				v[j] = rng.Float32()*2 - 1
			}
			if err := ix.Add("v", v); err != nil {
				t.Fatal(err)
			}
		}
		q := make([]float32, 8)
		for j := range q {
			q[j] = rng.Float32()
		}
		got, err := ix.Search(q, 10)
		if err != nil {
			t.Fatal(err)
		}
		for i := 1; i < len(got); i++ {
			if m == Euclidean {
				if got[i].Score < got[i-1].Score {
					t.Fatalf("%v: result %d scores %v after %v", m, i, got[i].Score, got[i-1].Score)
				}
				continue
			}
			if got[i].Score > got[i-1].Score {
				t.Fatalf("%v: result %d scores %v after %v", m, i, got[i].Score, got[i-1].Score)
			}
		}
	}
}

// The scores agree with a straightforward scalar computation. This is the
// property the kernels have to keep, and it is checked at a tolerance rather
// than exactly because the kernel sums in a different order.
func TestContractScoresMatchScalarReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const dim = 16
	for _, m := range []Metric{Cosine, DotProduct, Euclidean} {
		ix := New(dim, m)
		// Unique ids, because Result carries no row index: with duplicate ids
		// there is no way to say which row a hit came from.
		byID := map[string][]float32{}
		for i := 0; i < 40; i++ {
			v := make([]float32, dim)
			for j := range v {
				v[j] = rng.Float32()*2 - 1
			}
			id := "v" + itoa(i)
			byID[id] = v
			if err := ix.Add(id, v); err != nil {
				t.Fatal(err)
			}
		}
		q := make([]float32, dim)
		for j := range q {
			q[j] = rng.Float32()*2 - 1
		}
		got, err := ix.Search(q, len(byID))
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range got {
			want := scalarScore(m, byID[r.ID], q)
			if math.Abs(float64(r.Score-want)) > 1e-4 {
				t.Errorf("%v: %s scored %v, scalar %v", m, r.ID, r.Score, want)
			}
		}
	}
}

func scalarScore(m Metric, v, q []float32) float32 {
	var dot, nv, nq float64
	for i := range v {
		dot += float64(v[i]) * float64(q[i])
		nv += float64(v[i]) * float64(v[i])
		nq += float64(q[i]) * float64(q[i])
	}
	switch m {
	case DotProduct:
		return float32(dot)
	case Euclidean:
		return float32(math.Sqrt(math.Max(0, nv+nq-2*dot)))
	}
	if nv == 0 || nq == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(nv) * math.Sqrt(nq)))
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// Ties resolve by insertion order, in the selection as well as the ordering.
//
// Before this, twelve identical vectors with k=5 returned rows 5, 6, 0, 7, 2:
// deterministic for a given input, but nothing a caller could predict, and
// changing an unrelated part of the index moved it. Worse than the ordering
// was the selection -- with ties spanning the k boundary, *which* rows came
// back was arbitrary, so a partial tie over three equal top scores returned
// the second and third rather than the first and second.
func TestContractTiesResolveByInsertionOrder(t *testing.T) {
	t.Run("all tied", func(t *testing.T) {
		ix := New(4, DotProduct)
		for i := 0; i < 12; i++ {
			if err := ix.Add(itoa(i), []float32{1, 0, 0, 0}); err != nil {
				t.Fatal(err)
			}
		}
		got, err := ix.Search([]float32{1, 0, 0, 0}, 5)
		if err != nil {
			t.Fatal(err)
		}
		for i, r := range got {
			if r.ID != itoa(i) {
				t.Fatalf("result %d is %s, want %s (got %s)", i, r.ID, itoa(i), ids(got))
			}
		}
	})
	t.Run("tie spanning the k boundary", func(t *testing.T) {
		ix := New(4, DotProduct)
		for i := 0; i < 3; i++ {
			if err := ix.Add("hi"+itoa(i), []float32{2, 0, 0, 0}); err != nil {
				t.Fatal(err)
			}
		}
		for i := 0; i < 5; i++ {
			if err := ix.Add("lo"+itoa(i), []float32{1, 0, 0, 0}); err != nil {
				t.Fatal(err)
			}
		}
		got, err := ix.Search([]float32{1, 0, 0, 0}, 2)
		if err != nil {
			t.Fatal(err)
		}
		if got[0].ID != "hi0" || got[1].ID != "hi1" {
			t.Fatalf("got %s, want hi0 hi1", ids(got))
		}
	})
	t.Run("euclidean ties too", func(t *testing.T) {
		ix := New(4, Euclidean)
		for i := 0; i < 6; i++ {
			if err := ix.Add(itoa(i), []float32{1, 1, 1, 1}); err != nil {
				t.Fatal(err)
			}
		}
		got, err := ix.Search([]float32{1, 1, 1, 1}, 3)
		if err != nil {
			t.Fatal(err)
		}
		for i, r := range got {
			if r.ID != itoa(i) {
				t.Fatalf("euclidean result %d is %s, want %s", i, r.ID, itoa(i))
			}
		}
	})
	t.Run("stable across repeated searches", func(t *testing.T) {
		ix := New(4, Cosine)
		for i := 0; i < 20; i++ {
			if err := ix.Add(itoa(i), []float32{1, 0, 0, 0}); err != nil {
				t.Fatal(err)
			}
		}
		first := ids(mustSearch(t, ix, []float32{1, 0, 0, 0}, 7))
		for run := 0; run < 5; run++ {
			if got := ids(mustSearch(t, ix, []float32{1, 0, 0, 0}, 7)); got != first {
				t.Fatalf("run %d returned %s, first run %s", run, got, first)
			}
		}
	})
}

func mustSearch(t *testing.T, ix *Index, q []float32, k int) []Result {
	t.Helper()
	got, err := ix.Search(q, k)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func ids(rs []Result) string {
	s := ""
	for i, r := range rs {
		if i > 0 {
			s += " "
		}
		s += r.ID
	}
	return s
}
