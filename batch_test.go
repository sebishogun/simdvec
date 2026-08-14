package simdvec

import (
	"math/rand"
	"testing"
)

// A batch search must return, for every query, exactly what a single search
// for that query returns. The serial path makes that true by construction;
// the matrix-product path has to earn it, and this is where it does.
func TestBatchMatchesSerial(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	for _, m := range []Metric{Cosine, DotProduct, Euclidean} {
		for _, dim := range []int{8, 96} {
			for _, b := range []int{1, 2, 7, 8, 32} {
				ix := New(dim, m)
				for i := 0; i < 300; i++ {
					v := make([]float32, dim)
					for j := range v {
						v[j] = rng.Float32()*2 - 1
					}
					if err := ix.Add("v"+itoa(i), v); err != nil {
						t.Fatal(err)
					}
				}
				queries := make([][]float32, b)
				for i := range queries {
					q := make([]float32, dim)
					for j := range q {
						q[j] = rng.Float32()*2 - 1
					}
					queries[i] = q
				}
				// Both implementations, and the single-query oracle.
				serial, err := ix.batchSerial(queries, 10)
				if err != nil {
					t.Fatal(err)
				}
				mm, err := ix.batchMatMul(queries, 10)
				if err != nil {
					t.Fatal(err)
				}
				for i, q := range queries {
					want, err := ix.Search(q, 10)
					if err != nil {
						t.Fatal(err)
					}
					if ids(serial[i]) != ids(want) {
						t.Fatalf("%v dim=%d b=%d query %d: serial %s, single %s",
							m, dim, b, i, ids(serial[i]), ids(want))
					}
					if ids(mm[i]) != ids(want) {
						t.Fatalf("%v dim=%d b=%d query %d: matmul %s, single %s",
							m, dim, b, i, ids(mm[i]), ids(want))
					}
					for r := range want {
						if d := mm[i][r].Score - want[r].Score; d > 1e-4 || d < -1e-4 {
							t.Fatalf("%v dim=%d b=%d query %d result %d: matmul %v, single %v",
								m, dim, b, i, r, mm[i][r].Score, want[r].Score)
						}
					}
				}
			}
		}
	}
}

func TestBatchEdges(t *testing.T) {
	ix := New(3, DotProduct)
	for i := 0; i < 10; i++ {
		if err := ix.Add("v"+itoa(i), []float32{float32(i), 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	t.Run("no queries", func(t *testing.T) {
		got, err := ix.SearchBatch(nil, 3)
		if err != nil || got != nil {
			t.Fatalf("%v %v", got, err)
		}
	})
	t.Run("wrong dimension names the query", func(t *testing.T) {
		_, err := ix.SearchBatch([][]float32{{1, 0, 0}, {1, 0}}, 3)
		if err == nil {
			t.Fatal("accepted a wrong-length query")
		}
	})
	t.Run("k clamps per query", func(t *testing.T) {
		qs := make([][]float32, 12)
		for i := range qs {
			qs[i] = []float32{1, 0, 0}
		}
		got, err := ix.SearchBatch(qs, 99)
		if err != nil {
			t.Fatal(err)
		}
		for i, r := range got {
			if len(r) != 10 {
				t.Fatalf("query %d returned %d of 10 rows", i, len(r))
			}
		}
	})
	t.Run("empty index", func(t *testing.T) {
		e := New(3, Cosine)
		got, err := e.SearchBatch([][]float32{{1, 0, 0}, {0, 1, 0}}, 3)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 2 || got[0] != nil || got[1] != nil {
			t.Fatalf("%v", got)
		}
	})
	t.Run("queries survive", func(t *testing.T) {
		c := New(3, Cosine)
		c.Add("a", []float32{1, 0, 0})
		qs := make([][]float32, 12)
		before := make([][]float32, 12)
		for i := range qs {
			qs[i] = []float32{3, 4, 12}
			before[i] = []float32{3, 4, 12}
		}
		if _, err := c.SearchBatch(qs, 1); err != nil {
			t.Fatal(err)
		}
		for i := range qs {
			for j := range qs[i] {
				if qs[i][j] != before[i][j] {
					t.Fatalf("SearchBatch modified query %d: %v", i, qs[i])
				}
			}
		}
	})
	t.Run("the threshold does not change the answer", func(t *testing.T) {
		// Seven queries take the serial path and eight take the product; both
		// must agree with the single-query search.
		rng := rand.New(rand.NewSource(9))
		big := New(16, Cosine)
		for i := 0; i < 100; i++ {
			v := make([]float32, 16)
			for j := range v {
				v[j] = rng.Float32()
			}
			big.Add("v"+itoa(i), v)
		}
		qs := make([][]float32, 8)
		for i := range qs {
			q := make([]float32, 16)
			for j := range q {
				q[j] = rng.Float32()
			}
			qs[i] = q
		}
		below, err := big.SearchBatch(qs[:7], 5)
		if err != nil {
			t.Fatal(err)
		}
		atOrAbove, err := big.SearchBatch(qs, 5)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 7; i++ {
			if ids(below[i]) != ids(atOrAbove[i]) {
				t.Fatalf("query %d: %s below the threshold, %s at it",
					i, ids(below[i]), ids(atOrAbove[i]))
			}
		}
	})
}
