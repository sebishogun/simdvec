# simdvec — verification

What the gates are and how to run them. The methodology here is
mandatory: it is what separates a measured claim from a guess.


## The concurrency contract, demonstrated

`Index` is not safe for concurrent use, and that claim is a test rather than
a sentence. `concurrency_racecontract_test.go` runs four goroutines searching
one index with no synchronization; under the race detector it reports the
write-write race on the reused score buffer.

```
go test -race -tags racecontract -run TestContractSearchIsNotConcurrencySafe ./...   # expected: DATA RACE
go test -race ./...                                                                  # expected: green
```

The build tag is the important half. A demonstration of unsafety in the
default suite would make `-race` red permanently, and a suite that is red on
purpose trains people to stop reading it. The same file carries the
supported pattern -- a mutex around every operation -- as an ordinary test
that passes under `-race`, so both halves of the contract are executable.

When a change makes `Search` concurrency-safe, this file is deleted and
replaced by a normal test, not amended to expect safety: a demonstration
that no longer demonstrates anything is worse than none.

## The gates

```sh
go test ./...
go test -race ./...
go vet ./...
```

Run gates **bare** — never piped through `tail` or anything else without
`set -o pipefail`: the pipe reports the last command's status and a failure
vanishes. This has laundered red runs into green exits; it is the one rule
with a body count.

## Correctness coverage (the differential oracle)

`simdvec_test.go` pins part of the contract. The exact coverage is what is
listed here and nothing more; behavior that is true in the source but not
pinned by a current test is called out as such and scheduled for Task 0.1 of
the [production plan](plans/2026-08-13-simdvec-production.md):

- `TestMatchesNaive` — the index vs a scalar score-sort-take-k reference,
  on **rank and score within 1e-4**, across:
  - all three metrics (Cosine, DotProduct, Euclidean);
  - dimensions 4, 64, 384, 768;
  - index sizes 1, 5, 100, 1,000.
- `TestDimensionMismatch` — Add and Search return `ErrDim`-wrapped errors
  on wrong-length vectors. That the failed Add appends nothing is
  implementation behavior (the length check precedes all mutation,
  simdvec.go:135), not yet pinned by a test; plan Task 0.1 pins it.
- `TestAddCopies` — Add leaves the caller's slice byte-identical (Cosine
  normalization happens on the internal copy).
- `TestEmptyAndOversizedK` — an empty index returns `nil, nil`; `k > n` is
  clamped and returns every vector. `k <= 0` returning `nil, nil` is
  implementation behavior (simdvec.go:159), not covered by a current test;
  plan Task 0.1 pins it.

`TestMatchesNaive` is the definition of correct: a change that disagrees
with the oracle is wrong regardless of benchmark results. The oracle does
not pin tie order — today it is unspecified (quickselect + unstable sort);
whether to pin it is evaluated in the production plan (Task 1.2), not
decided here.

## Cross-architecture

The package runs on amd64, arm64, riscv64, s390x, ppc64le and loong64
(simdvec.go:1-3). For every docs or release claim about the other
architectures:

```sh
for arch in amd64 arm64 riscv64 s390x ppc64le loong64; do
  GOOS=linux GOARCH=$arch go build ./... || exit 1
done
```

A build is the floor, not a performance claim: `simd` dispatches to
per-architecture assembly, and latency is only claimed on the architecture
it was measured on (amd64, in the README table).

## Benchmarks and the noise floor

`BenchmarkSearch` (bench_test.go) compares the index against a hand-written
naive index (`[][]float32`, scalar dots, full sort) at dims 384/768 × sizes
10k/100k:

```sh
go test -run '^$' -bench '^BenchmarkSearch$' -benchmem -count=6
```

The code-layout noise floor here is **8.3%**. Anything smaller cannot be
told from nothing by wall-clock; more samples do not help (layout noise is
per-build, not per-run). When a change is expected to be worth less than
that:

- compare **instructions retired** and **cycles** with
  `perf stat -e instructions:u,cycles:u` — layout-independent;
- and **disassemble first, always**:

```sh
go test -c -o /tmp/x.test .
go tool objdump -s 'simdvec\.functionName' /tmp/x.test | less
```

Register pressure, eliminated bounds checks, inlined vs `memmove` appends,
and branch layout are only visible in the instructions.

A/B methodology: **interleaved** builds in one session, compared on the
**minimum**, never across sessions, machine quiet (**load average under
1**). The README's performance table is a historical release measurement,
not a regression gate; new numbers are published only by this methodology.

## Releases and docs

- The int8 figures in README/docs/wrong.md are historical: the prototype,
  recall fixture and benchmark were deleted, so they cannot be reproduced
  from the current tree. Do not re-measure them as if they were current.
- Local documentation links must resolve (manual pass over README, the
  docs/ tree, and the agent files).
- A docs branch touches **only .md files** — check `git diff --name-only`
  before committing.
- AGENTS.md and CLAUDE.md declare the AGENTS body verbatim in CLAUDE.md —
  keep them in sync. Manual check:
  `sed -n '/^# Working on simdvec$/,$p' CLAUDE.md | diff - AGENTS.md`
  (empty output means in sync).
- README facts are source-backed: API surface from Go declarations, go.mod
  for the Go version and `simd` version, the tag list for release claims.
