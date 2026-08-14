package simdvec

import (
	"fmt"

	"github.com/sebishogun/simd"
)

// Delete, Replace and Reset.
//
// All three keep the one invariant the whole design rests on: the vectors are
// a single contiguous N×D matrix, so a search is one matrix-vector product. A
// tombstone scheme would leave holes in that matrix and turn the scan into
// something that has to skip, which is a different and slower shape. Delete
// therefore compacts, at O(n·dim) copy for the rows after the first removal.
//
// Every one of them acts on *all* rows carrying the id. Add accepts duplicate
// ids, so a mutation that touched only the first would mean something
// different depending on data the caller cannot see -- and deleting once
// would not be enough, with nothing to say how many times is.

// Delete removes every row indexed under id and returns how many were
// removed. Deleting an id that is not present removes nothing and is not an
// error: the caller asked for a state that already holds.
func (ix *Index) Delete(id string) int {
	if ix.n == 0 {
		return 0
	}
	// One pass, compacting in place. The first surviving row before any
	// deletion does not move, so an index with nothing to delete copies
	// nothing.
	w := 0
	removed := 0
	for r := 0; r < ix.n; r++ {
		if ix.ids[r] == id {
			removed++
			continue
		}
		if w != r {
			copy(ix.data[w*ix.dim:(w+1)*ix.dim], ix.data[r*ix.dim:(r+1)*ix.dim])
			ix.ids[w] = ix.ids[r]
			ix.norms[w] = ix.norms[r]
		}
		w++
	}
	if removed == 0 {
		return 0
	}
	// The slices shrink together, because a length that disagrees with its
	// matrix is the defect this is easiest to introduce.
	ix.data = ix.data[:w*ix.dim]
	ix.ids = ix.ids[:w]
	ix.norms = ix.norms[:w]
	ix.n = w
	return removed
}

// Replace overwrites the vector of every row indexed under id and returns how
// many were updated. The vector is copied, and normalised under Cosine exactly
// as Add does it, so a replaced row is indistinguishable from one added with
// the same vector.
//
// A wrong-length vector is rejected before anything is written, so a failed
// Replace leaves the index untouched.
func (ix *Index) Replace(id string, vec []float32) (int, error) {
	if len(vec) != ix.dim {
		return 0, fmt.Errorf("%w: got %d, want %d", ErrDim, len(vec), ix.dim)
	}
	updated := 0
	for r := 0; r < ix.n; r++ {
		if ix.ids[r] != id {
			continue
		}
		row := ix.data[r*ix.dim : (r+1)*ix.dim]
		copy(row, vec)
		if ix.metric == Cosine {
			simd.Normalize(row)
		}
		ix.norms[r] = simd.SumSquares(row)
		updated++
	}
	return updated, nil
}

// Reset empties the index and keeps its memory.
//
// That is the whole reason it exists rather than the caller building a new
// Index: refilling an index of the same size afterwards allocates nothing.
func (ix *Index) Reset() {
	ix.data = ix.data[:0]
	ix.ids = ix.ids[:0]
	ix.norms = ix.norms[:0]
	ix.n = 0
}
