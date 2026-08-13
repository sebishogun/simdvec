# simdvec production design

Design for the production hardening and extension evaluations. This
document is the "what and why"; docs/plans/2026-08-13-simdvec-production.md
is the "how, task by task". Nothing here promises a feature — every item
ends in adoption or a docs/wrong.md rejection.

## Goal

Take the exact flat float32 index through a production pass: evaluate the
candidate extensions against the package's one idea — the whole scan is one
matrix-vector product — and keep or reject each on measured evidence, never
on taste.

Release status follows the README's Status section, the model for release
claims: the published **v0.1.0 tag** uses `simd v1.2.0`; the **current
untagged main** uses `simd v1.20.0` (Go 1.25). The product-as-shipped
wording below describes the current tree; claims about the published
release stay with the tag.

## Audience

The primary reader is a Go developer evaluating simdvec for embedding search.
Secondary: contributors deciding what to build next, and maintainers
checking that a change preserves the contract.

## The product as shipped (source-backed)

Two states exist, per the README Status model: the published **v0.1.0 tag**
(b326499, `simd v1.2.0`) and the **current untagged main** (`simd v1.20.0`,
Go 1.25). `simdvec.go` is unchanged since the tag — the only product-adjacent
changes since are the dependency bump (815876a) and docs — so the bullets
below describe both, with the current tree's dependency and platform
wording:

- `New(dim, metric)` panics on `dim <= 0`; `Dim`, `Len`, `Add`, `Search`,
  `Result`, `Metric` (Cosine/DotProduct/Euclidean), `ErrDim` is the whole
  surface (simdvec.go).
- Contiguous row-major float32 matrix; `Add` copies (cosine normalizes the
  internal copy), the query is never modified; one
  `simd.GemvParallelInto` scores the index; exact top-k via quickselect +
  sort of the k selected (simdvec.go:154-226).
- Empty/k<=0 → nil,nil; k clamps to n; duplicate opaque ids append; zero
  vectors accepted (cosine score 0); `scores` scratch reused; not safe for
  concurrent use.
- No persistence, delete, replace, reset, filter, batch, or ANN surface.
- Invalid `Metric` values score like dot product — not contractual.
- Platform, current tree: Go 1.25, `simd v1.20.0`, amd64/arm64/riscv64/
  s390x/ppc64le/loong64, no cgo (simdvec.go:1-3). The six-architecture
  wording existed at the tag — simdvec.go is unchanged since b326499 and
  the v0.1.0 README says it verbatim — but the support each state promises
  differs. The v0.1.0 README qualified it precisely (its Status section):
  "measured on amd64 only. The `simd` package underneath is verified on
  amd64 and arm64 NEON and under emulation elsewhere." The current tree's
  statement sits on the simd v1.20.0 support matrix — per-architecture
  dispatch with portable fallbacks — which is distinct from what v1.2.0
  offered. The current README's platform caveat — measured on amd64 only —
  applies to both.

## Scope decisions

- **Exact flat vector search stays the product.** ANN and quantization are
  non-goals unless future evidence changes the case (docs/roadmap.md). The
  int8 rejection (recall 0.954-0.982 at k=10, 1.11 ms vs 0.21 ms at 100k×768)
  is the standing evidence; docs/wrong.md preserves it.
- **The Gemv shape is load-bearing.** Any feature that changes the scan
  shape (filtering, masking, batching, delete-compaction) must justify the
  shape change by measurement, because the int8 record shows exactly how a
  bad shape loses.
- **The differential oracle is the correctness floor.** The scalar
  score-sort-take-k reference in the tests is the forever conformance
  baseline for rank and score within 1e-4.

## The evaluations

Each has: what changes, the gate that settles it, where the outcome lands.
The plan file runs these as TDD tasks with an explicit adopt-or-record step.

### E1. API safety

Harden what is loose today without breaking callers:

- `New` rejecting unknown `Metric` (today: dot-product-like scoring,
  non-contractual — a documented gap, not a bug).
- Tie ordering in `Result` (today unspecified — quickselect + unstable
  sort).
- The panic surface (`New` on dim<=0) and the `ErrDim` wrapping stay as
  shipped; freezing them in a contract test is the change.

Gate: contract tests for the chosen behavior; README + LLD sync. Rejected
alternatives recorded.

### E2. Concurrency

Today: not safe (shared `scores` scratch; `Add` appends). Options:

1. Keep the contract and document harder — zero code, current state.
2. Internal mutex over Add/Search — simple, serializes searches.
3. Per-search scratch so searches may overlap (Add still excluded) — costs
   an allocation per search, the exact thing the scratch reuse exists to
   avoid.

Gate: `-race` clean under the chosen contract; interleaved A/B with the
noise-floor methodology decides whether option 3's allocation is worth its
concurrency. The race detector is the primary tool; latency only if a
parallel contract is chosen.

### E3. Mutation: delete / replace / reset

None exist. Design questions first:

- delete-by-id under **duplicate ids** (opaque, may repeat);
- replace semantics under duplication;
- reset semantics: `data = data[:0]` keeps the matrix memory (a reset is
  not a free; document it) vs full release;
- deletion mechanics: tombstone + compaction vs immediate removal (O(n)
  shift or matrix rebuild).

Gate: differential oracle still green after mutations; top-k semantics
after delete pinned by test; the rebuild path benchmarked, not assumed.
Any rejected semantics get a wrong.md entry.

### E4. Persistence

Save/load of the matrix + norms + ids. Questions: format versioning,
endianness and architecture independence (the matrix is plain float32, but
the format must not assume it), crash safety (append-only vs atomic
rewrite), and whether persistence implies a memory-mapped query path
(shape change — E2's gate applies).

Gate: golden-file round-trip, truncation/reopen test, version-mismatch
behavior pinned. Not promised; the memory-resident design may simply keep
persistence out of scope.

### E5. Filter and batch

- Filter: metadata predicates before scoring. The scan shape question is
  masked-rows (Gemv over the full matrix, results zeroed) vs compacted
  matrix (rebuild per filter) — measured, not assumed.
- Batch: multiple queries as one N×D by D×B product. The int8 record
  (batching rescued a bad shape but never beat the single-query Gemv)
  applies directly: a batch API must beat B single queries by the
  methodology, or it does not ship.

### E6. Scale

Benchmarks cover up to 100k vectors (bench_test.go). Evaluations beyond
that are measured on the quiet machine — interleaved, minima, load < 1 —
never extrapolated from the 100k table. The memory question (4 bytes/
element, no compression) is part of it; int8 remains the standing answer.

## Verification design

- All gates from docs/verification.md run before any adoption commit:
  `go test ./...`, `-race`, `vet`, cross-arch builds, benchmark
  methodology.
- The differential oracle extends with each adopted feature; the scalar
  reference never goes away.
- Every rejected evaluation produces a docs/wrong.md entry with the exact
  numbers — the entry is the deliverable.
- Documentation changes are .md-only commits; README facts stay
  source-backed.

## Sources of truth

- Exported names and behavior: Go declarations and tests in the tree.
- Scoring, layout, scratch protocol: simdvec.go, cited by line.
- Dependency behavior (GemvParallelInto threshold, Normalize zero-vector
  rule): the simd library at v1.20.0, cited by file.
- Performance: committed benchmark records measured by the methodology.
- History: git log, the v0.1.0 tag, docs/wrong.md.
