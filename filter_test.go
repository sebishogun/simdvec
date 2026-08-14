package simdvec

import (
	"math/rand"
	"testing"
)

// A filtered search must return exactly what an unfiltered search over an
// index containing only the admitted rows would return -- same rows, same
// order, same scores. That is the oracle, and it is built by actually
// constructing that index rather than by reasoning about what it would say.
func TestFilterMatchesAnIndexOfTheSurvivors(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	const dim = 24
	for _, m := range []Metric{Cosine, DotProduct, Euclidean} {
		for _, keepEvery := range []int{1, 2, 7, 50} {
			ix := New(dim, m)
			oracle := New(dim, m)
			for i := 0; i < 300; i++ {
				v := make([]float32, dim)
				for j := range v {
					v[j] = rng.Float32()*2 - 1
				}
				id := "v" + itoa(i)
				if err := ix.Add(id, v); err != nil {
					t.Fatal(err)
				}
				if i%keepEvery == 0 {
					if err := oracle.Add(id, v); err != nil {
						t.Fatal(err)
					}
				}
			}
			q := make([]float32, dim)
			for j := range q {
				q[j] = rng.Float32()
			}
			got, err := ix.SearchFiltered(q, 10, func(row int, _ string) bool {
				return row%keepEvery == 0
			})
			if err != nil {
				t.Fatal(err)
			}
			want, err := oracle.Search(q, 10)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("%v keep 1/%d: %d results, oracle %d", m, keepEvery, len(got), len(want))
			}
			for i := range got {
				if got[i].ID != want[i].ID {
					t.Fatalf("%v keep 1/%d result %d: %s, oracle %s (got %s / want %s)",
						m, keepEvery, i, got[i].ID, want[i].ID, ids(got), ids(want))
				}
				if d := got[i].Score - want[i].Score; d > 1e-5 || d < -1e-5 {
					t.Fatalf("%v keep 1/%d result %d: score %v, oracle %v",
						m, keepEvery, i, got[i].Score, want[i].Score)
				}
			}
		}
	}
}

// Both implementations must agree with each other, at every selectivity. The
// public entry point picks one by measurement, so a divergence would show up
// only at whatever selectivity the crossover happens to sit at.
func TestFilterImplementationsAgree(t *testing.T) {
	rng := rand.New(rand.NewSource(33))
	const dim = 16
	for _, m := range []Metric{Cosine, DotProduct, Euclidean} {
		ix := New(dim, m)
		for i := 0; i < 200; i++ {
			v := make([]float32, dim)
			for j := range v {
				v[j] = rng.Float32()*2 - 1
			}
			if err := ix.Add("v"+itoa(i), v); err != nil {
				t.Fatal(err)
			}
		}
		q := make([]float32, dim)
		for j := range q {
			q[j] = rng.Float32()
		}
		for _, keepEvery := range []int{1, 2, 3, 5, 10, 40, 100} {
			var rows []int
			for r := 0; r < ix.n; r++ {
				if r%keepEvery == 0 {
					rows = append(rows, r)
				}
			}
			k := 5
			if k > len(rows) {
				k = len(rows)
			}
			masked, err := ix.searchMasked(q, k, rows)
			if err != nil {
				t.Fatal(err)
			}
			compacted, err := ix.searchCompacted(q, k, rows)
			if err != nil {
				t.Fatal(err)
			}
			if len(masked) != len(compacted) {
				t.Fatalf("%v 1/%d: masked %d, compacted %d", m, keepEvery, len(masked), len(compacted))
			}
			for i := range masked {
				if masked[i].ID != compacted[i].ID {
					t.Fatalf("%v 1/%d result %d: masked %s, compacted %s",
						m, keepEvery, i, masked[i].ID, compacted[i].ID)
				}
				if d := masked[i].Score - compacted[i].Score; d > 1e-5 || d < -1e-5 {
					t.Fatalf("%v 1/%d result %d: masked %v, compacted %v",
						m, keepEvery, i, masked[i].Score, compacted[i].Score)
				}
			}
		}
	}
}

func TestFilterEdges(t *testing.T) {
	ix := New(2, DotProduct)
	for i := 0; i < 10; i++ {
		if err := ix.Add("v"+itoa(i), []float32{float32(i), 0}); err != nil {
			t.Fatal(err)
		}
	}
	q := []float32{1, 0}

	t.Run("nil filter is a plain search", func(t *testing.T) {
		a, err := ix.SearchFiltered(q, 3, nil)
		if err != nil {
			t.Fatal(err)
		}
		b, err := ix.Search(q, 3)
		if err != nil {
			t.Fatal(err)
		}
		if ids(a) != ids(b) {
			t.Fatalf("%s vs %s", ids(a), ids(b))
		}
	})
	t.Run("nothing admitted", func(t *testing.T) {
		got, err := ix.SearchFiltered(q, 3, func(int, string) bool { return false })
		if err != nil || got != nil {
			t.Fatalf("%v %v", got, err)
		}
	})
	t.Run("k larger than the admitted set clamps", func(t *testing.T) {
		got, err := ix.SearchFiltered(q, 99, func(row int, _ string) bool { return row < 3 })
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 3 {
			t.Fatalf("%d results for 3 admitted rows", len(got))
		}
	})
	t.Run("wrong dimension", func(t *testing.T) {
		if _, err := ix.SearchFiltered([]float32{1}, 1, func(int, string) bool { return true }); err == nil {
			t.Fatal("accepted a wrong-length query")
		}
	})
	t.Run("the filter sees ids as well as rows", func(t *testing.T) {
		got, err := ix.SearchFiltered(q, 2, func(_ int, id string) bool { return id == "v9" || id == "v8" })
		if err != nil {
			t.Fatal(err)
		}
		if ids(got) != "v9 v8" {
			t.Fatalf("%s", ids(got))
		}
	})
	t.Run("the query survives", func(t *testing.T) {
		ixc := New(2, Cosine)
		ixc.Add("a", []float32{1, 0})
		ixc.Add("b", []float32{0, 1})
		query := []float32{3, 4}
		before := append([]float32(nil), query...)
		if _, err := ixc.SearchFiltered(query, 1, func(int, string) bool { return true }); err != nil {
			t.Fatal(err)
		}
		for i := range query {
			if query[i] != before[i] {
				t.Fatalf("SearchFiltered modified the query: %v, was %v", query, before)
			}
		}
	})
	t.Run("ties still resolve by row order", func(t *testing.T) {
		tie := New(2, DotProduct)
		for i := 0; i < 12; i++ {
			if err := tie.Add(itoa(i), []float32{1, 0}); err != nil {
				t.Fatal(err)
			}
		}
		// Admit the even rows; the answer must be the first three of those.
		got, err := tie.SearchFiltered([]float32{1, 0}, 3, func(row int, _ string) bool { return row%2 == 0 })
		if err != nil {
			t.Fatal(err)
		}
		if ids(got) != "0 2 4" {
			t.Fatalf("%s, want \"0 2 4\"", ids(got))
		}
	})
}
