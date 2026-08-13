# CLAUDE.md — working on simdvec

This file is the working contract for Claude Code on this repository. It is
derived from AGENTS.md (below, verbatim) plus the architecture and the plans;
where this header and AGENTS.md ever disagree, AGENTS.md wins.

## Where everything is

- **Architecture:** docs/architecture.md — the shipped design, cited against
  the source it describes.
- **Low-level design:** docs/lld/index-and-search.md — layouts, the scratch
  buffer protocol, the top-k path, allocation and concurrency facts.
- **Roadmap:** docs/roadmap.md — what is evaluated next and what is
  deliberately not in scope.
- **Design and build plan:** docs/plans/2026-08-13-simdvec-production-design.md
  and docs/plans/2026-08-13-simdvec-production.md — the future production
  work, tests-first, with an evaluate/delete-record outcome per task.
- **Verification:** docs/verification.md — the gates and how to run them.
- **The record:** docs/wrong.md — every measurement that argued against a
  change, whether or not code changed.

## Consuming simd

`github.com/sebishogun/simd` v1.20.0 (go.mod), Go 1.25 minimum. Pure Go, no
cgo; the kernels are committed assembly dispatched per architecture with
portable fallbacks. The index's contract is scoring through
`simd.GemvParallelInto`, `simd.Normalize` and `simd.SumSquares`; a change
that replaces a call site must stay within the differential oracle's 1e-4
score tolerance or be reverted and recorded.

## The product, in one breath

Exact flat vector search over float32 embeddings, one matrix-vector product
per query. ANN and quantization are non-goals unless future evidence
changes the scope.

---

# Working on simdvec

## Disassemble first, always

Before proposing a cause for anything slow, before writing a variant, before
reading a profile delta — **build it and read the instructions**.

```
go test -c -o /tmp/x.test .
go tool objdump -s 'simdvec\.functionName' /tmp/x.test | less
```

Use gdb when a breakpoint or a live register is needed, and both together
when that helps. Go compiles in seconds; every guess that skips the
disassembly costs a build-measure-revert cycle and risks a wrong conclusion
landing in `docs/wrong.md` as fact.

What the disassembly says that nothing else does:

- **Register pressure.** A large stack frame with the loop counter or a flag
  spilled and reloaded per iteration. No performance counter reports this.
- **Whether a bounds check was eliminated**, and whether an index multiply
  is a shift or a multiply.
- **Whether a call was inlined**, and whether `append(b, s...)` became
  inline stores or a `memmove` call.
- **Which branch the compiler laid out as fallthrough.**

## The correctness oracle

The differential in `simdvec_test.go` is the definition of correct: `naive`
scores every vector with a plain scalar loop, sorts, and takes k, and the
index must match it on rank and score — across all three metrics, dimensions
4, 64, 384 and 768, and index sizes 1, 5, 100 and 1,000 — with scores within
1e-4. A change that disagrees with the oracle is wrong, whatever the
benchmark says. Add and Search must also keep the caller's slices untouched
(`TestAddCopies`), and dimension mismatches must be `ErrDim` errors, not
panics.

## Benchmarks

The code-layout noise floor here is **8.3%**. Anything smaller cannot be
told from nothing by wall-clock, and more samples do not help — layout
noise is per-build, not per-run. When a change is expected to be worth less
than that:

- compare **instructions retired** and **cycles** with
  `perf stat -e instructions:u,cycles:u`, which are layout-independent;
- and read the disassembly, which is the only thing that explains *why*.

A/B builds must be **interleaved** in one session and compared on the
minimum, never across sessions. Run the machine quiet: wait for load average
under 1.

**Never pipe a gate through `tail`** (or anything else) without `pipefail`:
the pipe reports the last command's status and the failure vanishes. Run
gates bare, or `set -o pipefail` first.

The numbers in the README's performance table are release measurements on
amd64 only; no latency claim is made for the other architectures `simd`
supports.

## The product, in one breath

Exact flat vector search over float32 embeddings: the whole index is one
`simd.GemvParallelInto` matrix-vector product per query. ANN and
quantization are non-goals unless future evidence changes the scope — the
int8 rejection in `docs/wrong.md` is the standing example of why.

## The record

`docs/wrong.md` holds measurements that argued against changes, including
changes that were then reverted. A finding that cost a measurement belongs
there whether or not any code changed — the entry is the deliverable. The
int8 index is the standing case: built, measured, deleted, and the exact
figures preserved.
