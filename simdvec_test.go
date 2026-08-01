package simdvec

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

func randVecs(r *rand.Rand, n, dim int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(r.NormFloat64())
		}
		out[i] = v
	}
	return out
}

// naive is the implementation someone writes by hand, and the definition of
// correct: score every vector with a plain loop, sort, take k.
func naive(vecs [][]float32, q []float32, k int, metric Metric) []Result {
	type sc struct {
		i int
		v float64
	}
	all := make([]sc, len(vecs))
	for i, v := range vecs {
		var dot, na, nb float64
		for j := range v {
			dot += float64(v[j]) * float64(q[j])
			na += float64(v[j]) * float64(v[j])
			nb += float64(q[j]) * float64(q[j])
		}
		switch metric {
		case Cosine:
			all[i] = sc{i, dot / (math.Sqrt(na) * math.Sqrt(nb))}
		case DotProduct:
			all[i] = sc{i, dot}
		case Euclidean:
			all[i] = sc{i, math.Sqrt(na - 2*dot + nb)}
		}
	}
	asc := metric == Euclidean
	sort.SliceStable(all, func(a, b int) bool {
		if asc {
			return all[a].v < all[b].v
		}
		return all[a].v > all[b].v
	})
	out := make([]Result, k)
	for i := 0; i < k; i++ {
		out[i] = Result{ID: fmt.Sprint(all[i].i), Score: float32(all[i].v)}
	}
	return out
}

func TestMatchesNaive(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, metric := range []Metric{Cosine, DotProduct, Euclidean} {
		for _, dim := range []int{4, 64, 384, 768} {
			for _, n := range []int{1, 5, 100, 1000} {
				vecs := randVecs(r, n, dim)
				ix := New(dim, metric)
				for i, v := range vecs {
					// Add normalises in place for Cosine, so hand it a copy —
					// the naive reference needs the originals.
					c := append([]float32(nil), v...)
					if err := ix.Add(fmt.Sprint(i), c); err != nil {
						t.Fatal(err)
					}
				}
				q := randVecs(r, 1, dim)[0]
				k := min(10, n)
				got, err := ix.Search(q, k)
				if err != nil {
					t.Fatal(err)
				}
				want := naive(vecs, q, k, metric)
				if len(got) != len(want) {
					t.Fatalf("%v dim=%d n=%d: got %d results, want %d", metric, dim, n, len(got), len(want))
				}
				for i := range want {
					if got[i].ID != want[i].ID {
						t.Fatalf("%v dim=%d n=%d: rank %d is %s (%.6f), want %s (%.6f)",
							metric, dim, n, i, got[i].ID, got[i].Score, want[i].ID, want[i].Score)
					}
					if d := math.Abs(float64(got[i].Score - want[i].Score)); d > 1e-4 {
						t.Fatalf("%v dim=%d n=%d: rank %d score %v, want %v",
							metric, dim, n, i, got[i].Score, want[i].Score)
					}
				}
			}
		}
	}
}

func TestDimensionMismatch(t *testing.T) {
	ix := New(8, Cosine)
	if err := ix.Add("a", make([]float32, 4)); err == nil {
		t.Error("expected an error adding a wrong-size vector")
	}
	ix.Add("a", make([]float32, 8))
	if _, err := ix.Search(make([]float32, 4), 1); err == nil {
		t.Error("expected an error searching with a wrong-size vector")
	}
}

// Add must not retain or modify the caller's slice.
func TestAddCopies(t *testing.T) {
	ix := New(4, Cosine)
	v := []float32{3, 4, 0, 0}
	orig := append([]float32(nil), v...)
	ix.Add("a", v)
	for i := range v {
		if v[i] != orig[i] {
			t.Fatalf("Add modified the caller's slice: %v became %v", orig, v)
		}
	}
}

func TestEmptyAndOversizedK(t *testing.T) {
	ix := New(4, Cosine)
	if got, _ := ix.Search([]float32{1, 0, 0, 0}, 5); got != nil {
		t.Error("an empty index should return no results")
	}
	ix.Add("a", []float32{1, 0, 0, 0})
	got, _ := ix.Search([]float32{1, 0, 0, 0}, 100)
	if len(got) != 1 {
		t.Errorf("k larger than the index should return %d results, got %d", 1, len(got))
	}
}
