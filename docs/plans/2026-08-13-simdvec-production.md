# simdvec production implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task when it is scheduled. This plan is committed as documentation before any of its tasks run; it is the future work order, not a description of the current tree.

**Goal:** Take the exact flat float32 index through a production pass — API safety, concurrency, mutation, persistence, filter/batch, and scale — where every task ends in an explicit outcome: adopted and committed, or rejected with its measurement preserved in `docs/wrong.md`. Nothing is promised; everything is evaluated.

**Architecture:** The package keeps its one idea — the whole scan is one `simd.GemvParallelInto` matrix-vector product over a contiguous row-major float32 matrix — and any extension that changes the scan shape (masking, batching, compaction) must justify the shape by measurement, because the int8 rejection (docs/wrong.md) shows how a bad shape loses. The scalar score-sort-take-k reference in the tests is the forever conformance oracle (rank + score within 1e-4), and the README's performance table stays historical.

**Tech Stack:** Go 1.25, `github.com/sebishogun/simd` v1.20.0 (GemvParallelInto, Normalize, SumSquares), the noise-floor methodology from docs/verification.md (8.3% floor, `perf stat -e instructions:u,cycles:u`, interleaved minima, load < 1, bare gates or `pipefail`).

**Design doc:** docs/plans/2026-08-13-simdvec-production-design.md (evaluations E1-E6). **Discipline:** disassemble first, tests first, every task ends in a commit, every rejection ends in a wrong.md entry.

---

## Phase 0: The contract harness

### Task 0.1: Freeze the shipped contract in tests

**Files:**
- Create: `contract_test.go`
- Modify: `README.md`, `docs/verification.md`

**Step 1:** Write the failing contract tests first:

- `New(0)` and `New(-1)` panic with "dimension must be positive"; `New(1, Metric(99))` does not panic (documents the non-contractual fallback).
- Add/Search wrong-length returns an `ErrDim`-wrapped error (`errors.Is`), and the failed Add appends nothing (`Len()` unchanged, matrix byte-identical).
- Empty index and `k <= 0` return `nil, nil`; `k > n` clamps.
- Duplicate ids append distinct rows; zero vectors are accepted.
- Search leaves the query slice byte-identical for all three metrics (ownership).
- Result ordering is best-first per metric and the scores are within 1e-4 of the scalar reference (the existing differential, called from the new file).

**Step 2:** Run to verify fail: `go test ./...` — FAIL (nothing exists).

**Step 3:** Implement the tests (no library code changes — this task may pass with zero product changes; that is a success).

**Step 4:** `go test ./...`, `go test -race ./...`, `go vet ./...` PASS; cross-arch builds green (`GOOS=linux GOARCH=amd64|arm64|riscv64|s390x|ppc64le|loong64 go build ./...`).

**Step 5:** Commit: `test: freeze the shipped simdvec contract`.

## Phase 1: API safety (E1)

### Task 1.1: Metric validation

**Files:**
- Modify: `simdvec.go`, `simdvec_test.go`
- Modify: `README.md`, `docs/architecture.md`, `docs/lld/index-and-search.md`

**Step 1:** Tests first: `New(dim, Metric(99))` — decide the contract before writing it. Two candidates: (a) panic like `dim <= 0`; (b) documented error return. Choose (a) for symmetry with the existing panic surface, unless the review disagrees.

**Step 2:** Implement; `Metric.String()` stays "unknown" only if the fallback survives (it does not under (a) — delete the fallback text from the docs).

**Step 3:** Gates: full suite + race + vet + cross-arch; README/architecture/LLD updated in the same commit.

**Step 4: Evaluate.** Outcome: adopt → commit `api: New rejects unknown metrics`. Reject → revert and record the reasoning in docs/wrong.md. Either way the docs match the code.

### Task 1.2: Tie ordering

**Files:**
- Modify: `simdvec.go`, `simdvec_test.go`
- Modify: `docs/lld/index-and-search.md`

**Step 1:** Tests first: construct score arrays with exact ties (duplicated vectors) and pin the chosen order. Candidates: (a) keep unspecified (test that no order is required — contract stays loose); (b) pin descending-score/insertion-order via a stable sort.

**Step 2:** Implement (b) if adopted: `sort.SliceStable` on the selected k; note the quickselect partition still permutes ties, so (b) requires selection that preserves insertion order among ties — if that costs a measurable fraction of Search, the benchmark methodology decides.

**Step 3:** Gates as always.

**Step 4: Evaluate.** (a) is the default: ties are genuinely rare in embedding scores, and pinning them may cost more than it buys. Adopt or record in wrong.md.

## Phase 2: Concurrency (E2)

### Task 2.1: The documented contract is race-tested

**Files:**
- Create: `concurrency_test.go`
- Modify: `README.md`

**Step 1:** Tests first: a test that provably races under the current contract (two goroutines Search on a shared index; `-race` must FAIL) — the test documents what "not safe" means, and passes only when run serialized or with `-race` off.

**Step 2:** Add the external-sync usage example to the README (mutex around every operation).

**Step 3:** Gates: `go test -race ./...` — the race test fails by design; document the invocation (`go test -run TestConcurrentSearch -race` red, and why).

**Step 4: Evaluate.** Outcome: adopt the documented-external-sync contract as the default and record it in the README. Reject → wrong.md.

### Task 2.2: Per-search scratch (option 3), measured

**Files:**
- Create: `scratch_test.go`, benchmark additions to `bench_test.go`

**Step 1:** Tests first: two interleaved `Search` calls with per-call scratch produce identical results to the serial path (the oracle).

**Step 2:** Implement as a variant behind a build tag or a separate function (`SearchParallel` candidate), keeping the shipped `Search` untouched until the measurement.

**Step 3:** Interleaved A/B, quiet machine, minima: shipped vs per-scratch at 10k/100k × 384/768. If the delta is under the 8.3% floor, run `perf stat -e instructions:u,cycles:u` and read the disassembly before concluding.

**Step 4: Evaluate.** Adopt only if the allocation cost is inside the noise floor AND the concurrency win is real (race test passes for the new path). Otherwise: revert, record the numbers in wrong.md. The shipped `Search` changes only on adoption.

## Phase 3: Mutation (E3)

### Task 3.1: Delete

**Files:**
- Create: `mutation_test.go`
- Modify: `simdvec.go`, docs

**Step 1:** Tests first — semantics pinned before code: delete-by-id removes all rows with that id (duplicates!); the matrix stays contiguous (rebuild or mark-and-compact); top-k after delete matches the oracle on the survivor set; delete of an absent id is a no-op, not an error (decide and pin).

**Step 2:** Implement the cheapest correct option first: rebuild the matrix from survivors (O(n·dim) copy) — the scan stays one Gemv, the shape is unchanged.

**Step 3:** Bench the rebuild vs a tombstone variant at 100k; the Gemv shape is load-bearing, so a tombstone that breaks contiguity is presumed guilty until measured.

**Step 4: Evaluate.** Adopt the rebuild, or record why not. Commit: `api: Delete(id)` or the wrong.md entry.

### Task 3.2: Replace and reset

**Files:** as 3.1.

**Step 1:** Tests first: `Replace(id, vec)` semantics under duplicate ids (all rows? first row? error?) — decide by test; `Reset()` drops all rows and, per the design doc (E3), keeps the matrix memory (pin that in a test: `cap(ix.data)` unchanged after reset).

**Step 2:** Implement; reset is `data = data[:0]` etc. — trivial; replace follows the 3.1 decision.

**Step 3:** Gates.

**Step 4: Evaluate.** Each adopted API gets a commit; each rejected semantic gets a wrong.md entry.

## Phase 4: Persistence (E4)

### Task 4.1: Format design, golden files, crash safety

**Files:**
- Create: `persist_test.go`, `format.md` (or extend docs/architecture.md)
- Modify: `simdvec.go`, docs

**Step 1:** Tests first: round-trip a populated index (all metrics, dims 4/768, n 1/1000) through the format and compare against the oracle; golden bytes committed for a fixed fixture; truncated file → clean error, never a panic; version-mismatch → clean error.

**Step 2:** Implement save/load over the row-major matrix + norms + ids with a versioned header; endianness and architecture independence pinned by test (the matrix is plain float32 — write a big-endian byte-order test via `binary` even if the host is little-endian).

**Step 3:** Gates: full suite, race, vet, cross-arch builds (s390x is big-endian — the format must round-trip there).

**Step 4: Evaluate.** Persistence may legitimately stay out of scope (the design is memory-resident); if the round-trip tests expose a shape cost, record it. Adopt → commit; reject → wrong.md.

## Phase 5: Filter and batch (E5)

### Task 5.1: Filter evaluation

**Files:**
- Create: `filter_test.go`, benchmark additions

**Step 1:** Tests first: predicate → allowed row set; top-k over the filtered set matches the oracle filtered the same way (score semantics unchanged).

**Step 2:** Implement the masked-rows candidate (Gemv over the full matrix, non-members' scores zeroed/ignored) and the compacted candidate (rebuild the matrix per filter) — both behind the same interface, both differential-tested.

**Step 3:** Interleaved A/B at 100k × 768 with 1%, 10%, 50% selectivity; the int8 record says shape changes pay — measure before adopting either.

**Step 4: Evaluate.** Adopt the winner or record both rejections. Commit: `api: filter search` or the wrong.md entry.

### Task 5.2: Batch evaluation

**Files:**
- Create: `batch_test.go`, benchmark additions

**Step 1:** Tests first: batch search returns per-query results identical to the serial oracle.

**Step 2:** Implement the N×D by D×B product candidate (one Gemv per query is the baseline; a gemm is the candidate — the simd library's matmul shape is the same blocked kernel the int8 record blamed).

**Step 3:** A/B: B serial Gemvs vs one gemm, B ∈ {2, 8, 32}, 100k × 768. If the gemm loses, the int8 lesson repeats and the record lands in wrong.md.

**Step 4: Evaluate.** Adopt or record. Commit either way.

## Phase 6: Scale evaluation (E6)

### Task 6.1: The 1M measurement

**Files:**
- Create: `scale_test.go` or a scale benchmark (behind a flag — not part of `go test ./...`)

**Step 1:** Measure, don't extrapolate: BenchmarkSearch at n ∈ {200k, 500k, 1M} × dim 768, quiet machine, interleaved minima, the README table stays historical.

**Step 2:** Report the numbers in the README's Limits section (or in wrong.md if they embarrass — the discipline is the same either way).

**Step 3:** Memory: measure RSS at 1M×768 (3 GB of matrix); state the ceiling in the README.

**Step 4: Evaluate.** The flat scan's standing at 1M is the input to any future ANN discussion — if the scan wins, the non-goal stays; if it collapses, the scope decision is revisited with the numbers. Record.

## Phase 7: Release pass

### Task 7.1: The v0.2.0 documentation sweep

**Files:**
- Modify: `README.md`, docs as needed

**Step 1:** Verify every README fact against the tree (API surface from declarations, versions from go.mod, release claims from tags); every doc link resolves; the wrong.md sweep confirms every rejected evaluation has its entry.

**Step 2:** Decide versioning with the maintainer; the API is pre-1.0 and the contract tests from Phase 0 are the back-compat gate.

**Step 3:** Gates: full suite, race, vet, cross-arch, benchmark methodology.

**Step 4:** Commit: `docs: v0.2.0 sweep` (or the release commit). Tagging and publishing are outside this plan's scope unless scheduled.

---

## Decision points

| # | decision | open? | rule that settles it |
|---|---|---|---|
| 1 | Unknown `Metric`: panic vs error | open | Task 1.1; symmetry with the dim<=0 panic unless review objects |
| 2 | Tie order in Result | open | Task 1.2; unspecified is the default, pinned only if cost-free |
| 3 | Concurrency contract | open | Task 2.1/2.2; race detector + noise-floor A/B |
| 4 | Delete semantics under duplicate ids | open | Task 3.1; pin by test before code |
| 5 | Delete mechanics: rebuild vs tombstone | open | Task 3.1 Step 3; the Gemv shape is presumed guilty until measured |
| 6 | Reset memory retention | settled | Task 3.2; reset keeps the matrix (append growth is never shrunk) — documented, not changed |
| 7 | Persistence in scope? | open | Task 4.1; memory-resident design may keep it out, recorded either way |
| 8 | Filter shape: masked vs compacted | open | Task 5.1; interleaved A/B by selectivity |
| 9 | Batch API | open | Task 5.2; must beat B serial Gemvs by the methodology or it does not ship |
| 10 | ANN/quantization scope | settled | non-goals; the int8 record (docs/wrong.md) is the standing evidence, revisited only by Task 6.1-style measurement |
| 11 | Toolchain | settled | Go 1.25, simd v1.20.0; no toolchain-quirk dependencies |

When a gate measurement contradicts a settled decision, the decision changes and the entry lands in `docs/wrong.md` — that is the plan working, not failing.

## The plan's honesty clause

Nothing in these phases is promised to ship. Every task has an explicit
adopt-or-record step; a rejection with its numbers in `docs/wrong.md` is a
successful outcome. The shipped contract (Phase 0's tests) is the floor
nothing may break, and the scalar oracle is the forever definition of
correct.
