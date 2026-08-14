package simdvec

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// The on-disk format.
//
// An index is a matrix, a norm per row, and an id per row, so the file is
// those three in that order behind a header that says how to read them. Two
// decisions are worth stating because both are easy to get wrong once and
// live with forever:
//
// Byte order is little-endian, always, chosen rather than inherited. A format
// that writes host order produces files that a machine of the other endianness
// silently misreads -- the floats come back as different floats, not as an
// error -- and the test that would catch it cannot run on the host that wrote
// them. So the encoder names the order explicitly and a test builds the same
// bytes by hand.
//
// The header carries a version. Refusing an unknown one is the difference
// between a clear error today and an unexplainable one after the format
// changes.

const (
	magic         = "SIMDVEC1"
	formatVersion = 1
	// headerSize is magic(8) + version(4) + metric(4) + dim(8) + n(8).
	headerSize = 8 + 4 + 4 + 8 + 8
)

var (
	// ErrFormat means the bytes are not a simdvec index.
	ErrFormat = errors.New("simdvec: not an index file")
	// ErrVersion means the file was written by an incompatible version.
	ErrVersion = errors.New("simdvec: unsupported format version")
)

// WriteTo writes the index in the documented format.
//
// It writes what the index holds, which for Cosine means the normalised
// vectors: those are what Add stored, and reproducing the caller's originals
// would need a copy the index deliberately does not keep.
func (ix *Index) WriteTo(w io.Writer) (int64, error) {
	var head [headerSize]byte
	copy(head[0:8], magic)
	binary.LittleEndian.PutUint32(head[8:12], formatVersion)
	binary.LittleEndian.PutUint32(head[12:16], uint32(ix.metric))
	binary.LittleEndian.PutUint64(head[16:24], uint64(ix.dim))
	binary.LittleEndian.PutUint64(head[24:32], uint64(ix.n))
	written := int64(0)
	n, err := w.Write(head[:])
	written += int64(n)
	if err != nil {
		return written, err
	}

	// The matrix and the norms, little-endian float32 either way.
	buf := make([]byte, 0, 4096)
	flush := func() error {
		if len(buf) == 0 {
			return nil
		}
		m, err := w.Write(buf)
		written += int64(m)
		buf = buf[:0]
		return err
	}
	putFloat := func(f float32) error {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(f))
		if len(buf) >= 4096 {
			return flush()
		}
		return nil
	}
	for _, f := range ix.data[:ix.n*ix.dim] {
		if err := putFloat(f); err != nil {
			return written, err
		}
	}
	for _, f := range ix.norms[:ix.n] {
		if err := putFloat(f); err != nil {
			return written, err
		}
	}
	if err := flush(); err != nil {
		return written, err
	}

	// Ids: a length then the bytes, so an id may hold anything including a NUL.
	for i := 0; i < ix.n; i++ {
		var l [4]byte
		binary.LittleEndian.PutUint32(l[:], uint32(len(ix.ids[i])))
		m, err := w.Write(l[:])
		written += int64(m)
		if err != nil {
			return written, err
		}
		m, err = io.WriteString(w, ix.ids[i])
		written += int64(m)
		if err != nil {
			return written, err
		}
	}
	return written, nil
}

// ReadFrom replaces the index's contents with what r holds.
//
// A truncated or corrupt file is an error, never a panic and never a
// half-loaded index: the new contents are built beside the old ones and only
// swapped in once the whole file has been read.
func (ix *Index) ReadFrom(r io.Reader) (int64, error) {
	var head [headerSize]byte
	read := int64(0)
	n, err := io.ReadFull(r, head[:])
	read += int64(n)
	if err != nil {
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			return read, fmt.Errorf("%w: shorter than a header", ErrFormat)
		}
		return read, err
	}
	if string(head[0:8]) != magic {
		return read, ErrFormat
	}
	if v := binary.LittleEndian.Uint32(head[8:12]); v != formatVersion {
		return read, fmt.Errorf("%w: file is version %d, this build reads %d", ErrVersion, v, formatVersion)
	}
	metric := Metric(binary.LittleEndian.Uint32(head[12:16]))
	switch metric {
	case Cosine, DotProduct, Euclidean:
	default:
		return read, fmt.Errorf("%w: unknown metric %d", ErrFormat, metric)
	}
	dim64 := binary.LittleEndian.Uint64(head[16:24])
	rows64 := binary.LittleEndian.Uint64(head[24:32])
	// A header can claim any size; believing it is how a corrupt file becomes
	// an allocation the machine cannot serve.
	if dim64 == 0 || dim64 > 1<<20 {
		return read, fmt.Errorf("%w: dimension %d", ErrFormat, dim64)
	}
	if rows64 > 1<<32 {
		return read, fmt.Errorf("%w: %d rows", ErrFormat, rows64)
	}
	dim, rows := int(dim64), int(rows64)

	data := make([]float32, rows*dim)
	norms := make([]float32, rows)
	fbuf := make([]byte, 4*len(data))
	m, err := io.ReadFull(r, fbuf)
	read += int64(m)
	if err != nil {
		return read, fmt.Errorf("%w: matrix truncated", ErrFormat)
	}
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(fbuf[4*i:]))
	}
	nbuf := make([]byte, 4*rows)
	m, err = io.ReadFull(r, nbuf)
	read += int64(m)
	if err != nil {
		return read, fmt.Errorf("%w: norms truncated", ErrFormat)
	}
	for i := range norms {
		norms[i] = math.Float32frombits(binary.LittleEndian.Uint32(nbuf[4*i:]))
	}
	ids := make([]string, rows)
	var l [4]byte
	for i := 0; i < rows; i++ {
		m, err := io.ReadFull(r, l[:])
		read += int64(m)
		if err != nil {
			return read, fmt.Errorf("%w: ids truncated", ErrFormat)
		}
		ln := binary.LittleEndian.Uint32(l[:])
		if ln > 1<<20 {
			return read, fmt.Errorf("%w: id %d claims %d bytes", ErrFormat, i, ln)
		}
		b := make([]byte, ln)
		m, err = io.ReadFull(r, b)
		read += int64(m)
		if err != nil {
			return read, fmt.Errorf("%w: id %d truncated", ErrFormat, i)
		}
		ids[i] = string(b)
	}

	// Swapped in only now: an error above leaves the index as it was.
	ix.dim, ix.metric, ix.n = dim, metric, rows
	ix.data, ix.norms, ix.ids = data, norms, ids
	ix.scores = nil
	return read, nil
}

// Load returns an index read from r.
func Load(r io.Reader) (*Index, error) {
	ix := &Index{}
	if _, err := ix.ReadFrom(r); err != nil {
		return nil, err
	}
	return ix, nil
}
