# simdvec

`simdvec` is a flat in-memory index for exact float32 embedding search. It stores
vectors as one contiguous matrix and scores the full index with
[simd.go](https://github.com/sebishogun/simd). No cgo is required.

Requires Go 1.25 or later. The current main branch uses
`github.com/sebishogun/simd v1.20.0`; the published v0.1.0 release uses
`simd v1.2.0`.

```sh
go get github.com/sebishogun/simdvec
```

```go
ix := simdvec.New(768, simdvec.Cosine)

for id, embedding := range embeddings {
	if err := ix.Add(id, embedding); err != nil {
		return err
	}
}

hits, err := ix.Search(query, 10)
if err != nil {
	return err
}
```

## API and scores

`New(dim, metric)` creates an empty index. It panics on a non-positive
dimension and on an unrecognised metric: both are programming errors that no
run-time input can cause, and both are caught at construction rather than at
the first search.

| metric | `Result.Score` | ordering |
|---|---|---|
| `Cosine` | cosine similarity | highest first |
| `DotProduct` | raw dot product | highest first |
| `Euclidean` | Euclidean distance | lowest first |

`Index.Dim()` returns the configured dimension and `Index.Len()` returns the
number of vectors. `Add` and `Search` return errors wrapping `ErrDim` when a
vector length does not match; callers can use `errors.Is`.

`Search` returns `nil, nil` for an empty index or `k <= 0`. If `k` exceeds the
index length, every vector is returned. Results are sorted best-first according
to the selected metric, and **ties resolve by insertion order** -- in the
selection as well as the ordering, which matters when equal scores span the k
boundary.

`SearchFiltered(query, k, filter)` searches among the rows a predicate admits.
It scores the whole matrix and ignores the rejected rows, or gathers the
admitted rows and scores only those, choosing by selectivity: measured at
100k x 768, gathering is 18.9x faster at 1% admitted, level at 10%, and 6.2x
slower at 100%.

`SearchBatch(queries, k)` answers many queries at once. Below a block of eight
it runs independent searches; from eight upward it uses one matrix product,
which is 2.74x faster at 32 queries and 3x *slower* at two. Each result is
what `Search` would have returned for that query alone.

## Ownership and lifecycle

`Add` copies each vector into row-major storage and does not retain or modify
the caller's slice. Cosine normalization applies to that internal copy. Search
also leaves the query unchanged.

IDs are opaque strings, and duplicate IDs append separate vectors.

`Delete(id)` removes every row with that id and returns how many; `Replace(id,
vec)` updates every such row and returns how many; both act on all matching
rows, because a mutation touching only the first would mean something
different depending on data the caller cannot see. Deleting or replacing an id
that is not there is a no-op returning zero, not an error. `Reset()` empties
the index and keeps the matrix memory, so refilling it allocates nothing.

`WriteTo`/`ReadFrom` and `Load` save and restore an index. The format is
little-endian by decision rather than by host, and ids are length-prefixed so
one may contain any bytes. A truncated file is an error and leaves the index
untouched -- the new contents are built beside the old and swapped in only
once the whole file has been read.

`Index` is not safe for concurrent use. `Search` reuses score storage, so two
searches cannot overlap, and neither can `Add` overlap any search or add. Put
external synchronization around every operation when sharing an index:

```go
var mu sync.Mutex

// Every operation, not only the writes: two concurrent Searches race on the
// shared score buffer just as a Search and an Add race on the matrix.
mu.Lock()
hits, err := ix.Search(query, 10)
mu.Unlock()
```

That contract is demonstrated rather than asserted. A build-tagged test runs
four goroutines searching one index with no lock, and the race detector
reports the write-write race on the score buffer:

```
go test -race -tags racecontract -run TestContractSearchIsNotConcurrencySafe ./...
```

It is behind a tag so the default `go test -race ./...` stays green: a suite
that is red on purpose trains people to ignore the one signal that matters.

## Why the scan is fast

The usual exact implementation performs one dot-product call per vector and
then sorts all scores. `simdvec` stores an N x D matrix, so scoring is one
`GemvParallelInto` call followed by quickselect of the requested `k` and a sort
of only that selected set. Large scans may split across up to `GOMAXPROCS`
workers; smaller scans stay serial when worker overhead would cost more.

Cosine vectors are normalized once on insert and the query once per search.
Euclidean distance uses the matrix-vector dot products plus precomputed squared
norms, then reports the actual square-root distance.

## Performance

The v0.1.0 release compared cosine search with the hand-written implementation
in `bench_test.go`: vectors in `[][]float32`, one scalar dot product each, a full
sort, and the same generated inputs. These Zen 5 values are the slower of two
benchmark runs.

| index shape | naive | `simdvec` | ratio |
|---|---:|---:|---:|
| 100,000 x 768 | 66.3 ms | 3.68 ms | **18.0x** |
| 100,000 x 384 | 47.6 ms | 2.19 ms | **21.7x** |
| 10,000 x 768 | 5.79 ms | 0.15 ms | **38.4x** |
| 10,000 x 384 | 3.01 ms | 0.09 ms | **32.1x** |

```sh
go test -run '^$' -bench '^BenchmarkSearch$' -benchmem -count=6
```

The exact table is a historical release measurement, not a regression gate.
Performance is measured on amd64 only; no latency claim is made for the other
architectures supported by `simd`.

### The int8 experiment

The v0.1.0 README at the tag reports an int8 prototype with recall 0.954-0.982
at k=10 (the package comment, unchanged since the tag, carries the same
figures). On 100,000 vectors of 768 dimensions, its best batch measured 1.11 ms
per query versus 0.21 ms for one float32 query. The prototype, recall fixture,
and benchmark were deleted, so those figures are historical and cannot be
reproduced from the current tree.

The result explains the current API rather than promising a universal rule: an
N x D by D x 1 integer multiply is a narrow shape for a blocked matrix kernel,
while the float32 matrix-vector path is contiguous and parallel. This package
therefore stores float32 only. The full record, with the per-batch figures, is
in [docs/wrong.md](docs/wrong.md).

## Limits

Every query scans every vector. This is exact search, not HNSW, IVF, product
quantization, or another approximate index. The included benchmark covers up to
100,000 vectors; choose an approximate index when your measured scale or memory
budget makes a full float32 scan unsuitable.

The index has no persistence and no internal synchronization. It also does not
validate ID uniqueness or reject zero vectors. Normalization leaves a zero
vector unchanged, so its cosine score is 0.

## Verification

```sh
go test ./...
go test -race ./...
go vet ./...
```

The differential covers all three metrics, dimensions 4, 64, 384 and 768, and
index sizes from 1 to 1,000. It compares rank and score against a scalar
score-sort-take-k implementation, checks dimension errors and oversized `k`,
and verifies that `Add` does not mutate its input.

## Documentation

- [docs/architecture.md](docs/architecture.md) — the shipped design: layout, ownership, math, top-k, errors, concurrency, SIMD dependency.
- [docs/lld/index-and-search.md](docs/lld/index-and-search.md) — low-level design: the scratch protocol, Add/Search/top-k walk-throughs, allocation facts.
- [docs/roadmap.md](docs/roadmap.md) — what is evaluated next (safety, concurrency, mutation, persistence, filter, batch, scale); ANN and quantization stay non-goals unless evidence changes the scope.
- [docs/verification.md](docs/verification.md) — the gates and the benchmark methodology.
- [docs/wrong.md](docs/wrong.md) — the record: measurements that argued against changes, including the deleted int8 index.
- [docs/plans/2026-08-13-simdvec-production-design.md](docs/plans/2026-08-13-simdvec-production-design.md) and [docs/plans/2026-08-13-simdvec-production.md](docs/plans/2026-08-13-simdvec-production.md) — the future production work, tests-first, with an adopt-or-record outcome per task.
- [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) — the working rules for agents.

## Status

The latest tagged and published release is **v0.1.0**. The current main branch
uses `simd v1.20.0`; that dependency update is not yet tagged as a simdvec
release. Published v0.1.0 uses `simd v1.2.0`. The API is pre-1.0.

The maintained inventory of libraries built on `simd` is in the
[`simd` README](https://github.com/sebishogun/simd#built-on-this).

## License

MIT. See [LICENSE](LICENSE). `simd` is MIT; the indirect `golang.org/x/sys`
dependency is BSD-3-Clause.
