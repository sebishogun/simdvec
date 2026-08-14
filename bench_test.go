package simdvec

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

// naiveIndex is what a Go program does today without a library: keep the
// vectors in a slice of slices and loop.
type naiveIndex struct {
	vecs [][]float32
	ids  []string
}

func (n *naiveIndex) Add(id string, v []float32) {
	c := append([]float32(nil), v...)
	var s float64
	for _, x := range c {
		s += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range c {
		c[i] *= inv
	}
	n.vecs = append(n.vecs, c)
	n.ids = append(n.ids, id)
}

func (n *naiveIndex) Search(q []float32, k int) []Result {
	var s float64
	for _, x := range q {
		s += float64(x) * float64(x)
	}
	inv := float32(1 / math.Sqrt(s))
	qn := make([]float32, len(q))
	for i := range q {
		qn[i] = q[i] * inv
	}
	scores := make([]float32, len(n.vecs))
	for i, v := range n.vecs {
		var d float32
		for j := range v {
			d += v[j] * qn[j]
		}
		scores[i] = d
	}
	idx := make([]int, len(scores))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return scores[idx[a]] > scores[idx[b]] })
	out := make([]Result, k)
	for i := 0; i < k; i++ {
		out[i] = Result{ID: n.ids[idx[i]], Score: scores[idx[i]]}
	}
	return out
}

func BenchmarkSearch(b *testing.B) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, dim := range []int{384, 768} {
		for _, n := range []int{10000, 100000} {
			vecs := randVecs(r, n, dim)
			q := randVecs(r, 1, dim)[0]

			nv := &naiveIndex{}
			ix := New(dim, Cosine)
			for i, v := range vecs {
				id := fmt.Sprint(i)
				nv.Add(id, v)
				ix.Add(id, append([]float32(nil), v...))
			}
			b.Run(fmt.Sprintf("dim=%d/n=%d/naive", dim, n), func(b *testing.B) {
				for b.Loop() {
					nv.Search(q, 10)
				}
			})
			b.Run(fmt.Sprintf("dim=%d/n=%d/simdvec", dim, n), func(b *testing.B) {
				for b.Loop() {
					ix.Search(q, 10)
				}
			})
		}
	}
}

// Delete compacts the matrix rather than tombstoning, because the contiguous
// N×D layout is what makes a search one matrix-vector product. This measures
// what that costs at scale: the compaction is a copy of the rows after the
// first removal, and the search that follows is unchanged in shape.
func BenchmarkDelete(b *testing.B) {
	const dim = 128
	build := func(n int) (*Index, []string) {
		ix := New(dim, Cosine)
		ids := make([]string, n)
		v := make([]float32, dim)
		for i := 0; i < n; i++ {
			for j := range v {
				v[j] = float32((i*j)%101) / 101
			}
			ids[i] = "v" + itoa(i)
			if err := ix.Add(ids[i], v); err != nil {
				b.Fatal(err)
			}
		}
		return ix, ids
	}
	for _, n := range []int{10000, 100000} {
		b.Run("first-of-"+itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				ix, ids := build(n)
				b.StartTimer()
				// The worst case: removing the first row moves every other.
				if got := ix.Delete(ids[0]); got != 1 {
					b.Fatalf("removed %d", got)
				}
			}
		})
		b.Run("last-of-"+itoa(n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				ix, ids := build(n)
				b.StartTimer()
				// The best case: nothing after it to move.
				if got := ix.Delete(ids[n-1]); got != 1 {
					b.Fatalf("removed %d", got)
				}
			}
		})
		b.Run("absent-of-"+itoa(n), func(b *testing.B) {
			ix, _ := build(n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if got := ix.Delete("nothere"); got != 0 {
					b.Fatalf("removed %d", got)
				}
			}
		})
	}
}

// Masking against compaction across selectivity. Masking pays the full scan
// and no copy; compaction pays a copy per admitted row and scores only those.
// Which wins is entirely a function of how much the filter admits, so the
// crossover is measured rather than guessed.
func BenchmarkFilterStrategies(b *testing.B) {
	const dim = 768
	const n = 100000
	r := rand.New(rand.NewPCG(4, 5))
	ix := New(dim, Cosine)
	vecs := randVecs(r, n, dim)
	for i, v := range vecs {
		if err := ix.Add(itoa(i), append([]float32(nil), v...)); err != nil {
			b.Fatal(err)
		}
	}
	q := randVecs(r, 1, dim)[0]

	for _, pct := range []int{1, 5, 10, 20, 50, 100} {
		every := 100 / pct
		var rows []int
		for i := 0; i < n; i++ {
			if i%every == 0 {
				rows = append(rows, i)
			}
		}
		b.Run(fmt.Sprintf("masked/%d%%", pct), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := ix.searchMasked(q, 10, rows); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("compacted/%d%%", pct), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := ix.searchCompacted(q, 10, rows); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
