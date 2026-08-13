# simdvec — roadmap

Evaluations, not promises. Each item names what would change, the evidence
that would settle it, and where the outcome lands (adopted, or recorded in
docs/wrong.md and dropped). Nothing in this file commits the package to a
feature.

## The scope that does not move without evidence

The product is **exact flat vector search over float32**, one
`GemvParallelInto` per query. Two things are non-goals unless future
measurement changes the case:

- **ANN** (HNSW, IVF, product quantization). The flat scan is
  bandwidth-bound and parallel; an approximate index only starts to win once
  the scan stops fitting in cache. The evaluation gate would be a measured
  recall/latency trade-off at a scale where the flat scan demonstrably
  loses — not an assumption that it does.
- **Quantization**. Measured once already: int8 gave recall 0.954-0.982 at
  k=10 but was 5.3x slower than the float32 scan at 100k×768
  (docs/wrong.md). The int8 path was deleted; the rejection stands unless a
  new shape (batch-heavy workloads, memory-bound deployments) changes the
  arithmetic.

Scope changes must clear the differential oracle and the measurement
methodology in docs/verification.md first.

## Evaluations

### Safety (API contract hardening)

Candidate changes, none adopted:

- `New` rejecting unknown `Metric` values (today they score like dot
  product, explicitly not contractual).
- Pinning tie order in `Result` (today unspecified: quickselect +
  unstable sort).
- Freezing the panic/error surface (`ErrDim` wrapping, the `New` panic).

Gate: a contract test asserting the chosen behavior, then README/LLD
sync. Outcome recorded either way.

### Concurrency

Today the index is not safe for concurrent use (shared `scores` scratch,
`Add` appends). Options to evaluate, in order of cost:

1. Keep the contract, document harder (current state).
2. Internal mutex on `Add`/`Search`.
3. Per-search scratch (allocation per call) so searches may overlap;
   `Add` still needs synchronization.

Gate: `-race` clean under the chosen contract, then an interleaved
benchmark A/B — the noise-floor methodology decides whether option 3's
allocations are worth the concurrency it buys.

### Mutation: delete / replace / reset

None exist. The design questions are semantic, not just mechanical:

- delete-by-id with **duplicate ids** (ids are opaque and may repeat today);
- replace semantics under the same duplication;
- reset = drop n to 0 (the cheap win: `data = data[:0]` — worth noting the
  matrix memory is retained, so reset does not release memory);
- whether deletion is tombstone (compaction later) or immediate row
  removal (O(n) shift or rebuild).

Gate: differential oracle still passes after the mutation; top-k semantics
after delete are pinned by test. Rejected shapes are recorded.

### Persistence

No save/load. The natural unit is the row-major matrix + norms + ids; the
questions are format versioning, endianness/architecture independence, and
crash safety (append-only vs rewrite). The index is memory-resident by
design; persistence would not change the query path.

Gate: a golden-file round-trip test + a truncation/reopen test. Not
promised.

### Filter and batch

- **Filter**: predicate search (e.g. metadata pre-filter before scoring)
  changes the Gemv shape — masked rows would need zeroing or a compacted
  matrix. The int8 record warns that shape changes are where this design
  pays.
- **Batch**: multiple queries as one matrix product (N×D by D×B). The
  int8 experiment showed batching rescues a bad shape but does not beat the
  single-query Gemv; the same measurement discipline applies before any
  batch API.

### Scale

The included benchmark covers up to 100,000 vectors (bench_test.go). The
flat scan is memory-bound; the honest ceiling is "fits in RAM and the
latency budget". Evaluations at larger n must be measured — interleaved,
quiet machine, minima — never extrapolated from the 100k table. The memory
question (float32 at 4 bytes/element, no compression) is part of this
evaluation, and the int8 trade-off above is its standing answer.

## Ground rules

- Every evaluation ends in one of two records: the change, or a
  docs/wrong.md entry saying why not.
- No claim smaller than the 8.3% layout noise floor is made from wall
  clock; sub-floor differences go through `perf stat -e
  instructions:u,cycles:u` and the disassembly.
- Releases keep the README's performance table historical — a release
  measurement, not a regression gate.
