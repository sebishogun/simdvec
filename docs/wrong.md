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
