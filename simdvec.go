// Package simdvec is a vector index for embedding search, built on
// [simd.go](https://github.com/sebishogun/simd). No cgo, and the same code runs
// on amd64, arm64, riscv64, s390x, ppc64le and loong64.
//
//	ix := simdvec.New(768, simdvec.Cosine)
//	ix.Add("doc-1", embedding)
//	hits := ix.Search(query, 10)
//
// # Why this is fast
//
// The obvious way to search N embeddings is N dot products. That is N calls,
// and for a 768-dimension vector each one is over before the call overhead is
// amortised.
//
// The vectors are stored instead as one contiguous N×D matrix, which makes the
// entire scan a single matrix-vector product: [simd.GemvParallelInto] computes
// every score in one call, across every core. Searching a hundred thousand
// embeddings is one Gemv, not a hundred thousand dots.
//
// That is also why Add copies into the matrix rather than keeping a pointer.
// The layout is the optimisation.
//
// # There is no int8 index, and that was measured
//
// An int8 index was written, tested and deleted. Quantizing to int8 is a
// quarter of the memory and its recall was fine — 0.954 to 0.982 at k=10 — but
// it is slower, not faster, and by a lot.
//
// The scan becomes [simd.QMatMulInt8Into], and searching one query is an
// n×dim by dim×1 multiply. One output column is a degenerate shape for a
// matrix-multiply kernel, whose blocking assumes a wide result. Batching helps
// and does not rescue it, on 100,000 vectors of 768 dimensions:
//
//	int8, one query at a time   311.7 ms/query
//	int8, batches of 8           37.5 ms/query
//	int8, batches of 32           1.25 ms/query
//	int8, batches of 128          1.11 ms/query
//	float32, one query            0.21 ms/query
//
// The best int8 arrangement is five times slower than the float32 scan, because
// GemvParallelInto is parallel and reads memory in the order the prefetcher
// wants, and four times the elements per register does not make up for either.
//
// So this package stores float32. If the memory matters more than the latency,
// quantize before inserting — the index does not need to know.
package simdvec

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/sebishogun/simd"
)

// Metric is how two vectors are compared.
type Metric int

const (
	// Cosine compares direction and ignores magnitude. Vectors are normalised
	// on insert and on query, which turns the comparison into a dot product —
	// the division is done once per vector rather than once per comparison.
	Cosine Metric = iota
	// DotProduct compares without normalising, for models whose magnitude
	// carries meaning.
	DotProduct
	// Euclidean is straight-line distance. Computed from the dot product and
	// the precomputed norms rather than by subtracting, so it is the same one
	// matrix-vector product as the others.
	Euclidean
)

func (m Metric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case DotProduct:
		return "dot"
	case Euclidean:
		return "euclidean"
	}
	// Reachable only for a Metric value New would have rejected, which leaves
	// printing one in a test or a log as the case this serves.
	return "unknown"
}

// Result is one hit.
type Result struct {
	ID    string
	Score float32 // higher is better for Cosine and DotProduct; lower for Euclidean
}

// Index is a flat (brute-force) index over float32 embeddings.
//
// Every search scans every vector. That is the right structure up to a few
// hundred thousand embeddings, because one matrix-vector product over a
// contiguous block is bound by memory bandwidth rather than by arithmetic, and
// an approximate index only starts to win once the scan no longer fits in
// cache.
type Index struct {
	dim    int
	metric Metric

	data  []float32 // n*dim, row-major
	norms []float32 // squared norm per vector, for Euclidean
	ids   []string
	n     int

	scores []float32 // reused across searches

	// Scratch for filtered search: the admitted row list, the gathered
	// sub-matrix, and its scores. Reused for the same reason scores is.
	filtered  []int
	gather    []float32
	subScores []float32

	// Scratch for batch search: the packed query block, its N×B scores, and
	// one gathered column.
	qblock      []float32
	blockScores []float32
	colScores   []float32
}

// New returns an empty index for vectors of the given dimension.
//
// It panics on a non-positive dimension or an unrecognised metric. Both are
// programming errors that no run-time input can cause, and both are caught at
// construction rather than at the first search -- which is the whole reason to
// panic here instead of returning an error nobody checks at start-up.
//
// The unrecognised metric used to fall through and score like a dot product.
// Nothing said so at the call site, so a typo in a Metric constant produced a
// working index that answered the wrong question, silently and forever.
func New(dim int, metric Metric) *Index {
	if dim <= 0 {
		panic("simdvec: dimension must be positive")
	}
	switch metric {
	case Cosine, DotProduct, Euclidean:
	default:
		panic("simdvec: unknown metric")
	}
	return &Index{dim: dim, metric: metric}
}

// Dim returns the vector dimension.
func (ix *Index) Dim() int { return ix.dim }

// Len returns the number of indexed vectors.
func (ix *Index) Len() int { return ix.n }

// ErrDim is returned when a vector's length does not match the index.
var ErrDim = errors.New("simdvec: wrong vector dimension")

// Add indexes a vector under id.
//
// The vector is copied into the index's matrix; the caller's slice is not
// retained. For Cosine the copy is normalised on the way in, so the query-time
// comparison is a plain dot product.
func (ix *Index) Add(id string, vec []float32) error {
	if len(vec) != ix.dim {
		return fmt.Errorf("%w: got %d, want %d", ErrDim, len(vec), ix.dim)
	}
	ix.data = append(ix.data, vec...)
	row := ix.data[ix.n*ix.dim:]

	if ix.metric == Cosine {
		simd.Normalize(row)
	}
	ix.norms = append(ix.norms, simd.SumSquares(row))
	ix.ids = append(ix.ids, id)
	ix.n++
	return nil
}

// Search returns the k best matches for query.
//
// The whole index is scored with one matrix-vector product, then the k best are
// selected. Scoring is the expensive half and it is one call; selection is a
// partial sort over the scores.
func (ix *Index) Search(query []float32, k int) ([]Result, error) {
	if len(query) != ix.dim {
		return nil, fmt.Errorf("%w: got %d, want %d", ErrDim, len(query), ix.dim)
	}
	if ix.n == 0 || k <= 0 {
		return nil, nil
	}
	if k > ix.n {
		k = ix.n
	}

	scores, err := ix.scoreAll(query)
	if err != nil {
		return nil, err
	}

	return ix.topK(scores, k, ix.metric == Euclidean), nil
}

// queryFor returns the query the kernels should see: the caller's slice, or a
// normalised copy of it under Cosine. The copy exists because a caller's slice
// is theirs -- normalising in place would rewrite their data.
func (ix *Index) queryFor(query []float32) []float32 {
	if ix.metric != Cosine {
		return query
	}
	if cap(ix.scores) < ix.dim {
		ix.scores = make([]float32, ix.dim)
	}
	tmp := ix.scores[:ix.dim:ix.dim]
	copy(tmp, query)
	simd.Normalize(tmp)
	return tmp
}

// scoreAll scores every row against the query, in one matrix-vector product.
//
// Both the plain and the filtered searches come through here, so the metric
// arithmetic has one implementation. Two copies of the Euclidean conversion
// is precisely the shape of bug that survives review.
func (ix *Index) scoreAll(query []float32) ([]float32, error) {
	q := ix.queryFor(query)
	if cap(ix.scores) < ix.n+ix.dim {
		ix.scores = make([]float32, ix.n+ix.dim)
		if ix.metric == Cosine {
			// The grow moved the buffer the normalised query lives in.
			q = ix.queryFor(query)
		}
	}
	scores := ix.scores[ix.dim : ix.dim+ix.n]
	simd.GemvParallelInto(scores, ix.data, q, ix.n, ix.dim)
	if ix.metric == Euclidean {
		// |a-b|^2 = |a|^2 - 2a.b + |b|^2. The last term is the same for every
		// candidate, so it does not affect the ordering and is added back only
		// so the reported score is the real distance.
		qn := simd.SumSquares(q)
		for i := range scores {
			d := ix.norms[i] - 2*scores[i] + qn
			if d < 0 {
				d = 0 // rounding, not geometry
			}
			scores[i] = sqrt32(d)
		}
	}
	return scores, nil
}

func sqrt32(f float32) float32 { return float32(math.Sqrt(float64(f))) }

// topKOf selects the k best among a subset of rows, scoring by the full-length
// scores slice. It is topK with an explicit candidate list.
func (ix *Index) topKOf(scores []float32, rows []int, k int, ascending bool) []Result {
	idx := make([]int, len(rows))
	copy(idx, rows)
	cmp := func(a, b int) bool {
		if scores[a] == scores[b] {
			return a < b
		}
		if ascending {
			return scores[a] < scores[b]
		}
		return scores[a] > scores[b]
	}
	quickSelect(idx, k, cmp)
	sel := idx[:k]
	sortInts(sel, cmp)
	out := make([]Result, k)
	for i, j := range sel {
		out[i] = Result{ID: ix.ids[j], Score: scores[j]}
	}
	return out
}

// sortInts orders a selected set by the comparator.
func sortInts(sel []int, less func(a, b int) bool) {
	sort.Slice(sel, func(a, b int) bool { return less(sel[a], sel[b]) })
}

// topK selects the k best scores.
//
// A partial selection rather than a full sort: the scan produced n scores and
// only k are wanted, and for a large index n is very much larger than k.
func (ix *Index) topK(scores []float32, k int, ascending bool) []Result {
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	// A total order, not just a score comparison. Ties broken by row index
	// make both the selection and the ordering predictable: without it,
	// twelve identical vectors with k=5 returned rows 5, 6, 0, 7, 2, and a
	// tie spanning the k boundary returned the second and third of three
	// equal top scores rather than the first two. Which rows come back is a
	// harder problem than what order they come back in, and a partial
	// selection cannot be fixed by a stable sort afterwards -- by then the
	// wrong rows have been chosen.
	//
	// The extra comparison runs only when the scores are equal, so a search
	// over distinct scores pays for one branch that is never taken.
	cmp := func(a, b int) bool {
		if scores[a] == scores[b] {
			return a < b
		}
		if ascending {
			return scores[a] < scores[b]
		}
		return scores[a] > scores[b]
	}
	quickSelect(idx, k, cmp)
	sel := idx[:k]
	sort.Slice(sel, func(a, b int) bool { return cmp(sel[a], sel[b]) })

	out := make([]Result, k)
	for i, j := range sel {
		out[i] = Result{ID: ix.ids[j], Score: scores[j]}
	}
	return out
}

// quickSelect partitions idx so the first k are the k best under less.
func quickSelect(idx []int, k int, less func(a, b int) bool) {
	lo, hi := 0, len(idx)-1
	for lo < hi {
		p := partition(idx, lo, hi, less)
		switch {
		case p == k-1:
			return
		case p < k-1:
			lo = p + 1
		default:
			hi = p - 1
		}
	}
}

func partition(idx []int, lo, hi int, less func(a, b int) bool) int {
	// Median of three, so sorted input does not degenerate to quadratic.
	mid := lo + (hi-lo)/2
	if less(idx[mid], idx[lo]) {
		idx[lo], idx[mid] = idx[mid], idx[lo]
	}
	if less(idx[hi], idx[lo]) {
		idx[lo], idx[hi] = idx[hi], idx[lo]
	}
	if less(idx[hi], idx[mid]) {
		idx[mid], idx[hi] = idx[hi], idx[mid]
	}
	pivot := idx[mid]
	idx[mid], idx[hi] = idx[hi], idx[mid]
	store := lo
	for i := lo; i < hi; i++ {
		if less(idx[i], pivot) {
			idx[i], idx[store] = idx[store], idx[i]
			store++
		}
	}
	idx[store], idx[hi] = idx[hi], idx[store]
	return store
}
