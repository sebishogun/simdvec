# simdvec — architecture

Status: describes the current tree on branch `docs/v120-documentation`
(simd v1.20.0 dependency, tag v0.1.0 published earlier). Every claim is
backed by the source it cites; nothing here is a promise about future work
(that lives in docs/roadmap.md and docs/plans/).

## The product

An in-memory flat (brute-force) index for **exact** float32 embedding
search. `New(dim, metric)` + `Add` + `Search` is the whole API, plus
`Dim`, `Len`, `Result`, `Metric` and `ErrDim`. There is no persistence,
delete, replace, reset, filter, batch, or approximate-index surface.

The design has one idea: store the index as a contiguous N×D matrix so the
entire scan of N embeddings is **one matrix-vector product** —
`simd.GemvParallelInto` — instead of N dot-product calls (simdvec.go:15-21).
Searching a hundred thousand embeddings is one Gemv, not a hundred thousand
dots.

## Layout

`Index` (simdvec.go:99-108):

| field | holds |
|---|---|
| `dim`, `metric` | configuration, fixed at `New` |
| `data []float32` | n·dim floats, row-major: row i is vector i |
| `norms []float32` | squared norm per vector (Euclidean's |a|² term) |
| `ids []string` | parallel to the rows |
| `n int` | row count |
| `scores []float32` | scratch, reused across searches |

The matrix is the point: row i starts at `data[i*dim:]`, the scan walks the
rows in memory order, and the prefetcher sees one linear stream
(simdvec.go:15-21, 183).

## Growth and ownership

`Add` appends the caller's vector into `data`, `norms` and `ids` — plain
`append`, so amortized growth, no preallocation (simdvec.go:133-146). The
caller's slice is **copied, never retained and never modified**; cosine
normalization runs on the internal copy (`row := ix.data[ix.n*ix.dim:]`,
then `simd.Normalize(row)`, simdvec.go:138-142).

- IDs are opaque strings; duplicates simply append separate rows. There is
  no uniqueness check.
- Zero vectors are accepted: `simd.Normalize` leaves an all-zero vector
  unchanged (simd stats.go, "it has no direction to preserve"), so a zero
  vector's cosine score is 0. Euclidean and dot scores are unaffected.

## Normalization and math

All three metrics are served by the **same** Gemv; the differences are
pre- and post-processing:

- **Cosine** — vectors are normalized once on `Add` (internal copy) and the
  query once per `Search` (into a scratch copy, simdvec.go:166-175), turning
  the comparison into a plain dot product. The division happens once per
  vector instead of once per comparison (simdvec.go:62-64).
- **DotProduct** — nothing is normalized; magnitude carries meaning.
- **Euclidean** — distance computed from the dot products and precomputed
  norms, not by subtracting vectors: `d = norms[i] - 2·score[i] + qn` where
  `qn` is the query's squared norm, `d` clamped at 0 (rounding, not
  geometry) and square-rooted, so the reported score is the real distance
  (simdvec.go:185-197).

## Score sign and Result ordering

`Result.Score` is higher-is-better for Cosine and DotProduct, lower-is-better
for Euclidean (simdvec.go:89). Results come back sorted best-first by that
order.

**Ties are unspecified.** Selection is a quickselect partition followed by
`sort.Slice`, which is not stable (simdvec.go:206-226); among equal scores
any order is valid and callers must not rely on tie order, ID order, or
insertion order.

## The top-k path

After the Gemv: `topK` allocates an index array of n ints, `quickSelect`
partitions it so the k best are first (median-of-three pivot, so sorted
input does not degenerate to quadratic, simdvec.go:229-267), and
`sort.Slice` orders only those k (simdvec.go:217-219). The full score list
is never sorted: selection is O(n), the final sort is O(k log k), and for a
large index n ≫ k (simdvec.go:202-205).

## Errors and panics

- `New(dim <= 0, …)` **panics**: "simdvec: dimension must be positive"
  (simdvec.go:113-115). This is the only panic in the package.
- `Add`/`Search` with a mismatched vector length return an error wrapping
  `ErrDim` ("simdvec: wrong vector dimension", simdvec.go:126) — never a
  panic.
- `Search` on an empty index, or with `k <= 0`, returns `nil, nil`
  (simdvec.go:158-160). `k > n` is clamped to n (simdvec.go:161-163).
- An unknown `Metric` value is **not rejected** by `New`; only Cosine and
  Euclidean take special paths, so an invalid metric scores like a dot
  product. That fallback is not part of the contract (`Metric.String`
  reports "unknown"), and the README says so.

## Scratch and concurrency

`ix.scores` is one buffer doing two jobs in `Search` (simdvec.go:166-180):

- the cosine query copy lives in `scores[:dim]` (a normalized copy — a
  caller's query slice is theirs, simdvec.go:167);
- the per-candidate scores live in the window `scores[dim : dim+n]`.

The buffer is grown — reallocated — whenever `cap(scores) < n+dim`; a
growth does not invalidate the normalized query, which still references the
old backing array.

That reuse is exactly why the index is **not safe for concurrent use**: two
`Search` calls share the scratch, and `Add` mutates `data`/`norms`/`ids`/`n`
while a search is reading them. No `Search`/`Search` overlap, no `Add`
overlap with anything — external synchronization is required around every
operation on a shared index.

## Allocation profile

From the source, per operation:

- `New`: no allocations (struct literal).
- `Add`: amortized `append` growth of `data`, `norms`, `ids`; nothing else.
- `Search`: the cosine temp is scratch; `topK` allocates a fresh `[]int` of
  n per search and a `[]Result` of k. The scores window never allocates.

So a search allocates O(n + k) regardless of dimension; the Gemv itself is
the cost, not the scaffolding.

## SIMD dependency and platform

`github.com/sebishogun/simd` v1.20.0, Go 1.25 (go.mod). No cgo; the same
code runs on amd64, arm64, riscv64, s390x, ppc64le and loong64
(simdvec.go:1-3) — `simd` dispatches to committed per-architecture assembly
with portable fallbacks, and simdvec itself is portable Go.

`GemvParallelInto` divides the work by output row across up to
`GOMAXPROCS` workers and is bit-identical to `GemvInto` (simd parallel.go);
below `gemvParallelMinWork = 2^20` elements — or with fewer than two
workers — it runs serial (simd parallel.go:105-125). Consequence for the
index: small scans stay serial (worker overhead would cost more), large
scans split across cores. Scoring is memory-bandwidth-bound, not
arithmetic-bound (simd parallel.go:102-104).

## Scope and non-goals

The product is exact flat vector search over float32. **ANN** (HNSW, IVF,
product quantization) and **quantization** are non-goals: the int8 variant
was built, measured with recall 0.954-0.982 at k=10, and rejected on
latency (docs/wrong.md) — an N×D by D×1 integer multiply is a degenerate
shape for the blocked matrix kernel, while the float32 Gemv is parallel and
prefetcher-friendly. If memory matters more than latency, callers quantize
before inserting; the index does not need to know (simdvec.go:44-45).

Persistence, mutation, filtering, batching and internal concurrency do not
exist in the API today; each is evaluated — not promised — in
docs/roadmap.md and the production design in docs/plans/.
