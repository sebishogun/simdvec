package simdvec

import (
	"fmt"

	"github.com/sebishogun/simd"
)

// Filtered search: top-k among the rows a predicate admits.
//
// Two shapes are possible and they trade opposite ways. Masking scores the
// whole matrix with the one matrix-vector product the design is built around
// and then ignores the rows the predicate rejected -- full scan cost, no
// copying. Compaction gathers the admitted rows into a scratch matrix and
// scores only those -- less arithmetic, but a copy of every admitted row
// first, and a Gemv over a matrix that has to be rebuilt per search.
//
// Which wins depends entirely on selectivity, so both are implemented and
// SearchFiltered picks by measurement rather than by preference. The
// crossover is in docs/wrong.md with the numbers that put it there.

// Filter reports whether the row with this id and position is a candidate.
//
// It takes the row index as well as the id because ids may repeat: a caller
// filtering on something it stores alongside the index needs to know which
// row it is being asked about.
type Filter func(row int, id string) bool

// SearchFiltered returns the k best matches among the rows the filter admits.
//
// A nil filter admits everything and costs what Search costs.
func (ix *Index) SearchFiltered(query []float32, k int, keep Filter) ([]Result, error) {
	if len(query) != ix.dim {
		return nil, fmt.Errorf("%w: got %d, want %d", ErrDim, len(query), ix.dim)
	}
	if keep == nil {
		return ix.Search(query, k)
	}
	if ix.n == 0 || k <= 0 {
		return nil, nil
	}
	// The admitted set is needed either way, and it is cheap next to the
	// scoring: one predicate call per row.
	rows := ix.admitted(keep)
	if len(rows) == 0 {
		return nil, nil
	}
	if k > len(rows) {
		k = len(rows)
	}
	// The crossover, measured at 100k x 768 rather than guessed: below about
	// a tenth of the index admitted, gathering the rows and scoring the
	// gathered matrix wins; above it the copy costs more than the arithmetic
	// it saves. At 1% compaction is 18.9x faster, at 10% they are level, and
	// at 100% compaction is 6.2x slower. The numbers are in docs/wrong.md.
	//
	// The first version of this line used a fifth, which would have chosen
	// compaction at 15% where masking is 1.9x better. The benchmark moved it.
	if len(rows)*10 < ix.n {
		return ix.searchCompacted(query, k, rows)
	}
	return ix.searchMasked(query, k, rows)
}

// admitted collects the rows the filter accepts.
func (ix *Index) admitted(keep Filter) []int {
	rows := ix.filtered[:0]
	for r := 0; r < ix.n; r++ {
		if keep(r, ix.ids[r]) {
			rows = append(rows, r)
		}
	}
	ix.filtered = rows
	return rows
}

// searchMasked scores every row and selects among the admitted ones. It pays
// the full scan and no copy, which is the right trade when most rows are
// admitted anyway.
func (ix *Index) searchMasked(query []float32, k int, rows []int) ([]Result, error) {
	scores, err := ix.scoreAll(query)
	if err != nil {
		return nil, err
	}
	return ix.topKOf(scores, rows, k, ix.metric == Euclidean), nil
}

// searchCompacted gathers the admitted rows and scores only those. It pays a
// copy per admitted row and scores far less, which wins when the filter is
// selective.
func (ix *Index) searchCompacted(query []float32, k int, rows []int) ([]Result, error) {
	need := len(rows) * ix.dim
	if cap(ix.gather) < need {
		ix.gather = make([]float32, need)
	}
	sub := ix.gather[:need]
	for i, r := range rows {
		copy(sub[i*ix.dim:(i+1)*ix.dim], ix.data[r*ix.dim:(r+1)*ix.dim])
	}
	if cap(ix.subScores) < len(rows) {
		ix.subScores = make([]float32, len(rows))
	}
	scores := ix.subScores[:len(rows)]

	q := ix.queryFor(query)
	simd.GemvParallelInto(scores, sub, q, len(rows), ix.dim)
	if ix.metric == Euclidean {
		qn := simd.SumSquares(q)
		for i, r := range rows {
			d := ix.norms[r] - 2*scores[i] + qn
			if d < 0 {
				d = 0
			}
			scores[i] = sqrt32(d)
		}
	}
	// The scores are indexed by position in rows, so the comparator ties on
	// that position -- which is the row order, so insertion order survives.
	local := make([]int, len(rows))
	for i := range local {
		local[i] = i
	}
	asc := ix.metric == Euclidean
	cmp := func(a, b int) bool {
		if scores[a] == scores[b] {
			return a < b
		}
		if asc {
			return scores[a] < scores[b]
		}
		return scores[a] > scores[b]
	}
	quickSelect(local, k, cmp)
	sel := local[:k]
	sortInts(sel, cmp)
	out := make([]Result, k)
	for i, j := range sel {
		out[i] = Result{ID: ix.ids[rows[j]], Score: scores[j]}
	}
	return out, nil
}
