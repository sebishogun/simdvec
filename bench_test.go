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
