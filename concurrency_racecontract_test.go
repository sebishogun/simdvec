//go:build racecontract

package simdvec

import (
	"sync"
	"testing"
)

// The documented contract, demonstrated rather than described.
//
// Index is not safe for concurrent use: Search reuses one score buffer across
// calls, so two concurrent searches write the same memory. The README and the
// architecture doc both say so, and this file is the proof -- run it under
// -race and the detector reports the write-write race on ix.scores.
//
// It is behind a build tag on purpose. A demonstration of unsafety belongs
// nowhere near the default suite: `go test -race ./...` must stay green, or
// the one signal that matters gets trained away. Run it deliberately:
//
//	go test -race -tags racecontract -run TestContractSearchIsNotConcurrencySafe ./...
//
// A detected race is the expected outcome. When a future change makes Search
// concurrency-safe, this file is deleted and replaced by an ordinary test that
// passes under -race -- not amended to expect safety, because a demonstration
// that no longer demonstrates anything is worse than none.
func TestContractSearchIsNotConcurrencySafe(t *testing.T) {
	ix := New(64, Cosine)
	for i := 0; i < 500; i++ {
		v := make([]float32, 64)
		for j := range v {
			v[j] = float32(i*j%97) / 97
		}
		if err := ix.Add(itoa(i), v); err != nil {
			t.Fatal(err)
		}
	}
	q := make([]float32, 64)
	for j := range q {
		q[j] = float32(j) / 64
	}

	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				// No synchronization, which is the point: the contract says a
				// caller must provide it, and this is what happens when one
				// does not.
				if _, err := ix.Search(q, 10); err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
	t.Log("no race detected; either the detector is off (run with -race) or Search became safe, in which case this file should be replaced rather than kept")
}

// The supported way to share an index: the caller's lock around every
// operation. This one is race-free and passes under -race, which is what the
// README's example has to be.
func TestContractExternalSyncIsRaceFree(t *testing.T) {
	ix := New(64, Cosine)
	for i := 0; i < 200; i++ {
		v := make([]float32, 64)
		for j := range v {
			v[j] = float32((i+j)%53) / 53
		}
		if err := ix.Add(itoa(i), v); err != nil {
			t.Fatal(err)
		}
	}
	var mu sync.Mutex
	q := make([]float32, 64)
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				mu.Lock()
				_, err := ix.Search(q, 5)
				mu.Unlock()
				if err != nil {
					t.Error(err)
					return
				}
			}
		}()
	}
	wg.Wait()
}
