package simdvec

import (
	"math/rand"
	"testing"
)

// Delete, Replace and Reset: the semantics pinned before the code, because
// every one of them has a choice in it that is easier to argue about than to
// change later.

func addN(t testing.TB, ix *Index, ids []string, vecs [][]float32) {
	t.Helper()
	for i := range ids {
		if err := ix.Add(ids[i], vecs[i]); err != nil {
			t.Fatal(err)
		}
	}
}

// Delete removes every row with the id. Duplicates are the interesting case:
// Add accepts them, so Delete has to say what it does about them, and removing
// only the first would leave an index where deleting once is not enough and
// nothing says how many times is.
func TestDeleteRemovesAllRowsWithTheID(t *testing.T) {
	ix := New(2, DotProduct)
	addN(t, ix, []string{"a", "dup", "b", "dup", "c"}, [][]float32{
		{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0},
	})
	n := ix.Delete("dup")
	if n != 2 {
		t.Fatalf("Delete removed %d rows, want 2", n)
	}
	if ix.Len() != 3 {
		t.Fatalf("Len %d after deleting two of five", ix.Len())
	}
	got, err := ix.Search([]float32{1, 0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if s := ids(got); s != "c b a" {
		t.Fatalf("survivors %q, want \"c b a\"", s)
	}
	// The scores still match the survivors' vectors, so the matrix and the
	// norms were compacted together rather than one of them.
	for _, r := range got {
		want := map[string]float32{"a": 1, "b": 3, "c": 5}[r.ID]
		if r.Score != want {
			t.Fatalf("%s scored %v, want %v -- matrix and norms disagree", r.ID, r.Score, want)
		}
	}
}

// Deleting an id that is not there is a no-op returning zero, not an error.
// The caller asked for a state -- no rows with this id -- and that state
// already holds.
func TestDeleteAbsentIsANoOp(t *testing.T) {
	ix := New(2, Cosine)
	addN(t, ix, []string{"a"}, [][]float32{{1, 0}})
	before := ix.Len()
	if n := ix.Delete("nothere"); n != 0 {
		t.Fatalf("Delete of an absent id removed %d", n)
	}
	if ix.Len() != before {
		t.Fatalf("Len changed to %d", ix.Len())
	}
	if n := ix.Delete("a"); n != 1 {
		t.Fatalf("Delete removed %d", n)
	}
	if ix.Len() != 0 {
		t.Fatal("the index is not empty after deleting its only row")
	}
	// An empty index still searches.
	got, err := ix.Search([]float32{1, 0}, 3)
	if err != nil || got != nil {
		t.Fatalf("search after emptying: %v %v", got, err)
	}
}

// Deleting everything, then adding again, has to leave a working index: the
// bug this catches is a length field compacted without its backing slice.
func TestDeleteThenAdd(t *testing.T) {
	ix := New(3, Euclidean)
	addN(t, ix, []string{"x", "y", "z"}, [][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
	ix.Delete("x")
	ix.Delete("y")
	ix.Delete("z")
	if ix.Len() != 0 {
		t.Fatalf("Len %d", ix.Len())
	}
	addN(t, ix, []string{"new"}, [][]float32{{1, 1, 1}})
	got, err := ix.Search([]float32{1, 1, 1}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != "new" {
		t.Fatalf("after refilling: %v", got)
	}
	if got[0].Score > 1e-6 {
		t.Fatalf("distance to itself is %v", got[0].Score)
	}
}

// Delete against a full rebuild from the survivors: the same index either way.
func TestDeleteMatchesRebuild(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const dim = 12
	for _, m := range []Metric{Cosine, DotProduct, Euclidean} {
		ix := New(dim, m)
		var keepIDs []string
		var keepVecs [][]float32
		for i := 0; i < 60; i++ {
			v := make([]float32, dim)
			for j := range v {
				v[j] = rng.Float32()*2 - 1
			}
			id := "v" + itoa(i%20) // ids repeat, so deletes hit several rows
			if err := ix.Add(id, v); err != nil {
				t.Fatal(err)
			}
			if id != "v3" && id != "v7" {
				keepIDs = append(keepIDs, id)
				keepVecs = append(keepVecs, v)
			}
		}
		ix.Delete("v3")
		ix.Delete("v7")

		rebuilt := New(dim, m)
		addN(t, rebuilt, keepIDs, keepVecs)

		q := make([]float32, dim)
		for j := range q {
			q[j] = rng.Float32()
		}
		a, err := ix.Search(q, 10)
		if err != nil {
			t.Fatal(err)
		}
		b, err := rebuilt.Search(q, 10)
		if err != nil {
			t.Fatal(err)
		}
		if len(a) != len(b) {
			t.Fatalf("%v: %d results after delete, %d after rebuild", m, len(a), len(b))
		}
		for i := range a {
			if a[i].ID != b[i].ID || a[i].Score != b[i].Score {
				t.Fatalf("%v: result %d is %v after delete, %v after rebuild", m, i, a[i], b[i])
			}
		}
	}
}

// Replace updates every row with the id, for the same reason Delete removes
// every row: leaving some rows on the old vector would make Replace mean
// something different depending on data the caller cannot see.
func TestReplaceUpdatesAllRowsWithTheID(t *testing.T) {
	ix := New(2, DotProduct)
	addN(t, ix, []string{"a", "dup", "dup"}, [][]float32{{1, 0}, {2, 0}, {3, 0}})
	n, err := ix.Replace("dup", []float32{10, 0})
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("Replace updated %d rows, want 2", n)
	}
	got, err := ix.Search([]float32{1, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range got {
		if r.ID == "dup" && r.Score != 10 {
			t.Fatalf("a dup row scores %v, want 10 -- not every row was replaced", r.Score)
		}
	}
}

func TestReplaceValidatesAndReportsAbsent(t *testing.T) {
	ix := New(3, Cosine)
	addN(t, ix, []string{"a"}, [][]float32{{1, 0, 0}})
	if _, err := ix.Replace("a", []float32{1, 0}); err == nil {
		t.Fatal("Replace accepted a wrong-length vector")
	}
	// The failed Replace changed nothing.
	got, _ := ix.Search([]float32{1, 0, 0}, 1)
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("index disturbed by a failed Replace: %v", got)
	}
	n, err := ix.Replace("absent", []float32{1, 0, 0})
	if err != nil {
		t.Fatalf("Replace of an absent id errored: %v", err)
	}
	if n != 0 {
		t.Fatalf("Replace of an absent id updated %d rows", n)
	}
}

// Replace under Cosine has to normalise the new vector, the same way Add does.
func TestReplaceNormalisesUnderCosine(t *testing.T) {
	ix := New(2, Cosine)
	addN(t, ix, []string{"a"}, [][]float32{{1, 0}})
	if _, err := ix.Replace("a", []float32{100, 0}); err != nil {
		t.Fatal(err)
	}
	got, err := ix.Search([]float32{1, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Cosine of a vector with itself is 1 whatever its magnitude.
	if d := got[0].Score - 1; d > 1e-5 || d < -1e-5 {
		t.Fatalf("cosine score %v, want 1 -- the replacement was not normalised", got[0].Score)
	}
}

// Reset empties the index and keeps the memory, which is the whole reason it
// exists rather than the caller making a new Index.
func TestResetKeepsCapacity(t *testing.T) {
	ix := New(8, Cosine)
	for i := 0; i < 500; i++ {
		v := make([]float32, 8)
		v[i%8] = 1
		if err := ix.Add(itoa(i), v); err != nil {
			t.Fatal(err)
		}
	}
	capBefore := cap(ix.data)
	if capBefore == 0 {
		t.Fatal("no matrix allocated")
	}
	ix.Reset()
	if ix.Len() != 0 {
		t.Fatalf("Len %d after Reset", ix.Len())
	}
	if cap(ix.data) != capBefore {
		t.Fatalf("Reset released the matrix: cap %d, was %d", cap(ix.data), capBefore)
	}
	if got, err := ix.Search([]float32{1, 0, 0, 0, 0, 0, 0, 0}, 3); err != nil || got != nil {
		t.Fatalf("search after Reset: %v %v", got, err)
	}
	// And it refills without reallocating, which is the point.
	for i := 0; i < 500; i++ {
		v := make([]float32, 8)
		v[i%8] = 1
		if err := ix.Add(itoa(i), v); err != nil {
			t.Fatal(err)
		}
	}
	if cap(ix.data) != capBefore {
		t.Fatalf("refilling after Reset reallocated: cap %d, was %d", cap(ix.data), capBefore)
	}
	if ix.Len() != 500 {
		t.Fatalf("Len %d after refilling", ix.Len())
	}
}
