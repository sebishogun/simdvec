package simdvec

import (
	"fmt"

	"github.com/sebishogun/simd"
)

// Batch search: many queries at once.
//
// The obvious candidate is one matrix product instead of B matrix-vector
// products -- the N×D matrix against a D×B block of queries. It is the shape
// a BLAS library is built for, and it should win by reusing each row of the
// matrix across every query in the block rather than re-reading the matrix B
// times.
//
// "Should" is doing work in that sentence, which is why both are here and
// SearchBatch picks by measurement. The scan is memory-bound, not
// arithmetic-bound, and a blocked kernel that assumes a wide result can lose
// to a loop of narrow ones when the block is small.

// SearchBatch returns the k best matches for each query.
//
// Every query must have the index's dimension. The results are per query, in
// the order the queries were given, and each is what Search would have
// returned for that query alone.
func (ix *Index) SearchBatch(queries [][]float32, k int) ([][]Result, error) {
	for i, q := range queries {
		if len(q) != ix.dim {
			return nil, fmt.Errorf("%w: query %d got %d, want %d", ErrDim, i, len(q), ix.dim)
		}
	}
	if len(queries) == 0 {
		return nil, nil
	}
	if ix.n == 0 || k <= 0 {
		return make([][]Result, len(queries)), nil
	}
	// Measured: the matrix product wins from a block of about eight queries
	// upward, and loses below it where the per-call setup outweighs the reuse.
	// docs/wrong.md carries the numbers.
	if len(queries) >= 8 {
		return ix.batchMatMul(queries, k)
	}
	return ix.batchSerial(queries, k)
}

// batchSerial is B independent searches. The baseline, and the right answer
// for a small block.
func (ix *Index) batchSerial(queries [][]float32, k int) ([][]Result, error) {
	out := make([][]Result, len(queries))
	for i, q := range queries {
		r, err := ix.Search(q, k)
		if err != nil {
			return nil, err
		}
		out[i] = r
	}
	return out, nil
}

// batchMatMul scores every query against every row with one product.
//
// The queries are packed into a D×B column-major block so the result is the
// N×B score matrix, one column per query. Under Cosine the pack normalises on
// the way in -- the caller's slices are theirs, and the copy is happening
// anyway.
func (ix *Index) batchMatMul(queries [][]float32, k int) ([][]Result, error) {
	b := len(queries)
	need := ix.dim * b
	if cap(ix.qblock) < need {
		ix.qblock = make([]float32, need)
	}
	block := ix.qblock[:need]
	for j, q := range queries {
		for d := 0; d < ix.dim; d++ {
			block[d*b+j] = q[d]
		}
	}
	if ix.metric == Cosine {
		// Column-wise normalisation, since each column is one query.
		for j := 0; j < b; j++ {
			var ss float32
			for d := 0; d < ix.dim; d++ {
				v := block[d*b+j]
				ss += v * v
			}
			if ss == 0 {
				continue
			}
			inv := 1 / sqrt32(ss)
			for d := 0; d < ix.dim; d++ {
				block[d*b+j] *= inv
			}
		}
	}

	need = ix.n * b
	if cap(ix.blockScores) < need {
		ix.blockScores = make([]float32, need)
	}
	scores := ix.blockScores[:need]
	simd.MatMulParallelInto(scores, ix.data[:ix.n*ix.dim], block, ix.n, ix.dim, b)

	if ix.metric == Euclidean {
		qn := make([]float32, b)
		for j := 0; j < b; j++ {
			var ss float32
			for d := 0; d < ix.dim; d++ {
				v := block[d*b+j]
				ss += v * v
			}
			qn[j] = ss
		}
		for r := 0; r < ix.n; r++ {
			base := r * b
			for j := 0; j < b; j++ {
				d := ix.norms[r] - 2*scores[base+j] + qn[j]
				if d < 0 {
					d = 0
				}
				scores[base+j] = sqrt32(d)
			}
		}
	}

	// One column at a time into the selection, which wants a contiguous score
	// per row. Gathering the column is a strided read of n values -- cheaper
	// than the product it follows, and the alternative is a transposed result
	// the kernel does not produce.
	kk := k
	if kk > ix.n {
		kk = ix.n
	}
	if cap(ix.colScores) < ix.n {
		ix.colScores = make([]float32, ix.n)
	}
	col := ix.colScores[:ix.n]
	out := make([][]Result, b)
	for j := 0; j < b; j++ {
		for r := 0; r < ix.n; r++ {
			col[r] = scores[r*b+j]
		}
		out[j] = ix.topK(col, kk, ix.metric == Euclidean)
	}
	return out, nil
}
