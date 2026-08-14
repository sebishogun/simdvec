# The record

Measurements that argued against a change, whether or not code changed.
The entry is the deliverable.

## The int8 index: recall was fine, latency was not — built, measured, deleted

An int8 quantized index was written, tested and deleted during the v0.1.0
development cycle. This is the rejection that explains why the shipped
index stores float32 and nothing else.

Measured, 100,000 vectors of 768 dimensions, cosine. Source of the exact
figures: the **v0.1.0 README at the tag (b326499)**; the package comment
(simdvec.go:23-45), unchanged since the tag, carries the same numbers:

    int8, one query at a time   311.7 ms/query
    int8, batches of 8           37.5 ms/query
    int8, batches of 32           1.25 ms/query
    int8, batches of 128          1.11 ms/query
    float32, one query            0.21 ms/query

Recall was not the problem: **0.954 to 0.982 at k=10**. The problem was
latency — the best int8 arrangement, batched 128 queries at a time, was
**1.11 ms vs 0.21 ms, about 5.3x slower** than one float32 query.

Why, per the v0.1.0 README: the scan becomes an n×dim by dim×1 matrix
multiply,
and one output column is a degenerate shape for a blocked matrix-multiply
kernel whose blocking assumes a wide result. Batching (up to 128) narrows
but does not close the gap, because `GemvParallelInto` is parallel and
reads memory in the order the prefetcher wants, and four times the
elements per register does not make up for either.

The verdict stood: quantization is a memory play that pays in latency, so
the index stays float32, and callers who need less memory quantize before
inserting — the index does not need to know.

**Historical, not reproducible:** the prototype, its recall fixture and its
benchmark are not in the current tree — the figures survive only as text in
the v0.1.0 README and the package comment, so they cannot be re-measured
from the current tree. They are preserved here and in the README as the
record of the decision, and any future quantization proposal must re-measure
from scratch against this baseline.

## Delete compacts rather than tombstones, and the cost says why

**The question the plan asked.** Delete could compact the matrix -- an
O(n·dim) copy of the rows after the first removal -- or leave tombstones
and skip them at search time. The contiguous N×D matrix is what makes a
search one matrix-vector product, so the plan called a tombstone
"presumed guilty until measured".

**Measured.** Delete, dim 128:

    n=10,000    first row  211 us   last row  25.9 us   absent   6.0 us
    n=100,000   first row  2.47 ms  last row   253 us   absent  97.3 us

Linear in the number of rows after the removal, as expected. For
comparison, one search at n=100,000 and dim=384 costs 2.33 ms.

**Consequence.** A worst-case delete costs about what a single search
costs, once, and every search afterwards runs at full speed on an
unbroken matrix. A tombstone scheme would move that cost onto every
search, permanently, in exchange for making the rare operation cheap.
That is the wrong way round for an index that is searched far more often
than it is edited, so Delete compacts.

The absent-id case is the one worth noting: 97 us at 100,000 rows, which
is the scan for a matching id and no copying at all. Deleting something
that is not there costs a walk of the id slice and nothing else.

## Filtered search: the crossover is a tenth, not a fifth

**The question.** A filtered search can score the whole matrix and ignore
the rows the predicate rejected (masking), or gather the admitted rows and
score only those (compaction). Masking pays the full scan and no copy;
compaction pays a copy per admitted row. Which wins depends on
selectivity, and the plan asked for both to be built and measured.

**Measured**, 100,000 rows x 768 dimensions, Cosine:

    admitted     masked      compacted
       1%       2.88 ms      0.15 ms     compaction 18.9x
       5%       2.67 ms      1.41 ms     compaction  1.9x
      10%       2.85 ms      2.82 ms     level
      20%       3.11 ms      6.02 ms     masking     1.9x
      50%       3.31 ms     12.56 ms     masking     3.8x
     100%       3.82 ms     23.56 ms     masking     6.2x

**Consequence.** `SearchFiltered` picks by selectivity, with the threshold
at a tenth. The first implementation used a fifth -- a guess -- which
would have chosen compaction at 15%, where masking is 1.9x faster. The
benchmark moved it, which is the entire reason the plan asked for two
implementations instead of one.

Both are kept and both are differential-tested against each other at seven
selectivities and against an index built from the survivors, because a
threshold that picks between two implementations hides a divergence in
whichever one the threshold does not choose.

The shape is worth stating: compaction at 100% costs 6.2x masking, so a
"gather then score" design that looked obviously better for filtering is
catastrophic for the unfiltered case. The copy is the whole cost, and it
scales with what is admitted while the scan does not.

## Batch search: the matrix product loses below a block of eight

**The question.** B queries can be B independent matrix-vector products,
or one N×D by D×B matrix product. The product should win by reusing each
row of the matrix across the whole block instead of re-reading it B times.
The plan asked for both, because the scan is memory-bound rather than
arithmetic-bound and a blocked kernel that assumes a wide result can lose
on a narrow one.

**Measured**, 100,000 rows x 768 dimensions, Cosine:

    B      serial      one product     per query, serial -> product
     2    7.39 ms      21.82 ms        3.70 ms -> 10.91 ms   product 3.0x worse
     8   29.62 ms      26.89 ms        3.70 ms ->  3.36 ms   product 1.10x better
    32  120.93 ms      44.17 ms        3.78 ms ->  1.38 ms   product 2.74x better

The serial cost per query is flat, as it must be: each query is an
independent scan of the same matrix. The product's per-query cost falls
from 10.9 ms to 1.38 ms as the block widens, which is the row reuse
arriving.

**Consequence.** `SearchBatch` takes the product from a block of eight
upward and the serial loop below it. Both are kept and both are tested
against the single-query search, at batch sizes on each side of the
threshold, because a threshold choosing between two implementations leaves
one of them unexercised otherwise.

The B=2 row is the finding worth keeping: the product is **three times
worse** there. A design that reached for the matrix kernel because batching
"obviously" amortises the scan would have made small batches -- which is
most batches, in a request-shaped workload -- three times slower.
