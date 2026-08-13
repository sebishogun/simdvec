# LLD — Index and Search

Low-level design of the shipped index, from the source (simdvec.go) and the
tests. The architecture-level rationale is docs/architecture.md.

## Data structures

```
type Index struct {
    dim    int            // vector dimension, fixed at New
    metric Metric         // Cosine | DotProduct | Euclidean
    data  []float32       // n*dim, row-major: row i at data[i*dim : (i+1)*dim]
    norms []float32       // squared norm per row, appended at Add time
    ids   []string        // ids[i] is row i's id
    n     int             // live row count
    scores []float32      // scratch: [dim] query copy + [n] score window
}
```

- Row i's elements are contiguous; the whole matrix is one `[]float32`, so
  the scan in `Search` is one linear stream of n·dim floats.
- `norms[i]` is `simd.SumSquares(row)` computed at Add time (simdvec.go:143)
  — for Cosine the row is already normalized there, so the norm is 1.

## New

```
New(dim, metric):
    dim <= 0  -> panic("simdvec: dimension must be positive")
    return &Index{dim, metric}       // no allocation beyond the struct
```

No data buffers are allocated until the first Add. The metric is not
validated: an unknown value takes no special path in Add/Search, so it
behaves like DotProduct — explicitly not contractual.

## Add

```
Add(id, vec):
    len(vec) != dim -> error wrapping ErrDim ("got %d, want %d")
    data  = append(data, vec...)          // copy; caller's slice untouched
    row   = data[n*dim : ]                // the just-appended row
    metric == Cosine -> simd.Normalize(row)     // on the internal copy
    norms = append(norms, SumSquares(row))
    ids   = append(ids, id)
    n++
```

- Zero vectors: `Normalize` leaves them unchanged (simd stats.go) — the row
  is stored as-is, norm 0, cosine score 0.
- Duplicate ids: appended like any other id; no check, no replacement.

## Search

```
Search(query, k):
    len(query) != dim   -> (nil, ErrDim-wrapped error)
    n == 0 || k <= 0    -> (nil, nil)
    k > n               -> k = n                       // clamp

    metric == Cosine:
        grow scratch if cap(scores) < dim
        tmp  = scores[:dim:dim]
        copy(tmp, query); Normalize(tmp); q = tmp     // query untouched
    else:
        q = query                                      // used read-only

    grow scratch if cap(scores) < n+dim
    scores = ix.scores[dim : dim+n]

    GemvParallelInto(scores, data, q, n, dim)          // the whole scan

    metric == Euclidean:
        qn = SumSquares(q)
        for i in scores:
            d  = norms[i] - 2*scores[i] + qn
            d  = max(d, 0)                             // rounding, not geometry
            scores[i] = sqrt(d)                        // real distance

    return topK(scores, k, metric == Euclidean)
```

### The scratch protocol

One buffer, two regions:

1. `scores[:dim]` — cosine's normalized query copy (only in the Cosine
   branch).
2. `scores[dim:dim+n]` — the Gemv's output window.

Growth rule: reallocate to `n+dim` whenever `cap < n+dim`. After a growth
the normalized query still lives in the old backing array — `q` holds a
reference to it, so it stays alive and correct. The alternative — a
separate query buffer — is unnecessary because the two regions never
overlap.

Consequence for concurrency: the scratch is shared mutable state. Two
overlapping `Search` calls clobber each other's query copy and score window;
an `Add` during a search can reallocate `data` (append growth) under the
Gemv's read. The contract is: not safe, synchronize externally.

### Why the query copy for Cosine

`GemvParallelInto` consumes `x` as a plain slice; normalizing it in place
would mutate the caller's query (simdvec.go:167: "a caller's slice is
theirs"). The copy costs one `dim`-length buffer reused across searches,
and the Normalize runs once per query, not once per comparison.

## Top-k

```
topK(scores, k, ascending):
    idx = make([]int, n)                    // fresh per search
    less(a,b) = scores[a] < scores[b]       // ascending: Euclidean
             or scores[a] > scores[b]       // descending: cosine, dot
    quickSelect(idx, k, less)               // k best land in idx[:k], unordered
    sort.Slice(idx[:k], less)               // order just the k selected
    out = make([]Result, k)
    out[i] = {ids[j], scores[j]}
```

`quickSelect` is the standard partition-based selection with a
**median-of-three** pivot (`lo`, `mid`, `hi`) so sorted input does not
degenerate to quadratic (simdvec.go:245-255). Partitioning moves equal
scores arbitrarily; the final `sort.Slice` is unstable. **Tie order is
unspecified** — the differential oracle only ever compares against
exact-score random data, and no test pins tie order.

Complexity per search: O(n) selection + O(k log k) final sort + O(n)
`idx` allocation.

## Errors and panics — complete list

| condition | behavior | where |
|---|---|---|
| `New(dim <= 0)` | panic "dimension must be positive" | simdvec.go:113 |
| `Add` wrong length | `ErrDim`-wrapped error, index unchanged | simdvec.go:135 |
| `Search` wrong length | `ErrDim`-wrapped error, `nil` results | simdvec.go:155 |
| empty index | `nil, nil` | simdvec.go:158 |
| `k <= 0` | `nil, nil` | simdvec.go:159 |
| `k > n` | clamp to n, all vectors returned | simdvec.go:161 |
| invalid metric | dot-product-like scoring, **not contractual** | simdvec.go:112-116 |

On the `Add` error path nothing is appended: the length check precedes all
mutations.

## Allocation per operation

| op | allocations |
|---|---|
| `New` | none |
| `Add` | amortized append growth of `data`/`norms`/`ids` |
| `Search` | `[]int` of n (topK) + `[]Result` of k; scratch reused |

Scores and the cosine temp never allocate after warm-up. Dimension
independence: a 768-dim search allocates the same as a 4-dim search at the
same n.

## Dependency call sites

| call | purpose | in |
|---|---|---|
| `simd.GemvParallelInto(scores, data, q, n, dim)` | the whole scan | simdvec.go:183 |
| `simd.Normalize(row)` | cosine, internal Add copy | simdvec.go:141 |
| `simd.Normalize(tmp)` | cosine, query copy | simdvec.go:173 |
| `simd.SumSquares(row)` | norm at Add | simdvec.go:143 |
| `simd.SumSquares(q)` | Euclidean |q|² | simdvec.go:189 |

`GemvParallelInto` runs serial below 2^20 elements of work or with
`GOMAXPROCS < 2`, parallel (by output row, bit-identical) above (simd
parallel.go:105-137) — the index inherits both regimes with no branching of
its own.

## Test coverage this design must satisfy

- `TestMatchesNaive`: rank AND score within 1e-4 of the scalar oracle, all
  metrics × dims {4, 64, 384, 768} × sizes {1, 5, 100, 1000}.
- `TestDimensionMismatch`: Add and Search return errors, not panics.
- `TestAddCopies`: Add leaves the caller's slice byte-identical.
- `TestEmptyAndOversizedK`: empty → nil results; k=100 on a 1-vector index
  → exactly 1 result.
