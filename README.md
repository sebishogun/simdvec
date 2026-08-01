# simdvec

**A vector index for embedding search.** Built on
[simd.go](https://github.com/sebishogun/simd). No cgo, and the same code runs on
amd64, arm64, riscv64, s390x, ppc64le and loong64.

```
go get github.com/sebishogun/simdvec
```

```go
ix := simdvec.New(768, simdvec.Cosine)

for id, emb := range embeddings {
	ix.Add(id, emb)
}

hits, err := ix.Search(query, 10)
```

## Numbers

Against the loop a Go program writes today — vectors in a `[][]float32`, a dot
product each, sort. Zen 5, worse of two runs:

| | naive | simdvec | |
|---|---|---|---|
| 100,000 × 768 | 66.3 ms | 3.68 ms | **18.0×** |
| 100,000 × 384 | 47.6 ms | 2.19 ms | **21.7×** |
| 10,000 × 768 | 5.79 ms | 0.15 ms | **38.4×** |
| 10,000 × 384 | 3.01 ms | 0.09 ms | **32.1×** |

## Why it is faster

The obvious implementation is N dot products. That is N calls, and for a
768-dimension vector each one is over before the call overhead is amortised.

The vectors are stored instead as one contiguous N×D matrix, so **the entire
scan is a single matrix-vector product** — one `GemvParallelInto` across every
core, not a hundred thousand dots. Selection is then a quickselect over the
scores, because only k of N are wanted and N is very much larger.

That is also why `Add` copies. The layout is the optimisation.

## Metrics

`Cosine`, `DotProduct` and `Euclidean`.

Cosine normalises on insert and on query, which turns the comparison into a
plain dot product — the division happens once per vector instead of once per
comparison. Euclidean is computed from the dot product and precomputed norms
rather than by subtracting, so it is the same single matrix-vector product as
the others.

## There is no int8 index, and that was measured

One was written, tested and deleted. Quantizing to int8 is a quarter of the
memory and the recall was fine — **0.954 to 0.982 at k=10** across 128, 384 and
768 dimensions. It is also slower, by a lot.

Searching one query becomes an n×dim by dim×1 integer multiply, and a single
output column is a degenerate shape for a matrix-multiply kernel whose blocking
assumes a wide result. On 100,000 vectors of 768 dimensions:

| | ms per query |
|---|---|
| int8, one query at a time | 311.7 |
| int8, batches of 8 | 37.5 |
| int8, batches of 32 | 1.25 |
| int8, batches of 128 | 1.11 |
| **float32, one query** | **0.21** |

Batching helps and does not rescue it. The best int8 arrangement is **five times
slower** than the float32 scan, because `GemvParallelInto` is parallel and reads
memory in the order the prefetcher wants, and four times the elements per
register does not make up for either.

So this stores float32. If memory matters more than latency, quantize before
inserting — the index does not need to know.

## What this is not

**Not approximate.** Every search scans every vector. That is the right
structure up to a few hundred thousand embeddings, because one pass over a
contiguous block is bound by memory bandwidth rather than arithmetic, and an
approximate index only starts to win once the scan stops fitting in cache. Above
that, use one.

**Not persistent, and not concurrent.** The index is memory, and `Add` and
`Search` are not safe to call at the same time. Both are deliberate for a first
release; say if you need them.

## Correctness

Every result is compared against a naive implementation — score everything with
a plain loop, sort, take k — across three metrics, four dimensions and four
index sizes. Ranks and scores must both match.

```
go test ./...
```

## Status

Early, and measured on amd64 only. The `simd` package underneath is verified on
amd64 and arm64 NEON and under emulation elsewhere.

## License

MIT — see [LICENSE](LICENSE). Depends on
[simd.go](https://github.com/sebishogun/simd) (MIT).
