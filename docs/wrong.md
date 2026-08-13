# The record

Measurements that argued against a change, whether or not code changed.
The entry is the deliverable.

## The int8 index: recall was fine, latency was not — built, measured, deleted

An int8 quantized index was written, tested and deleted during the v0.1.0
development cycle. This is the rejection that explains why the shipped
index stores float32 and nothing else.

Measured, 100,000 vectors of 768 dimensions, cosine (from the v0.1.0
development record):

    int8, one query at a time   311.7 ms/query
    int8, batches of 8           37.5 ms/query
    int8, batches of 32           1.25 ms/query
    int8, batches of 128          1.11 ms/query
    float32, one query            0.21 ms/query

Recall was not the problem: **0.954 to 0.982 at k=10**. The problem was
latency — the best int8 arrangement, batched 128 queries at a time, was
**1.11 ms vs 0.21 ms, about 5.3x slower** than one float32 query.

Why, per the record: the scan becomes an n×dim by dim×1 matrix multiply,
and one output column is a degenerate shape for a blocked matrix-multiply
kernel whose blocking assumes a wide result. Batching (up to 128) narrows
but does not close the gap, because `GemvParallelInto` is parallel and
reads memory in the order the prefetcher wants, and four times the
elements per register does not make up for either.

The verdict stood: quantization is a memory play that pays in latency, so
the index stays float32, and callers who need less memory quantize before
inserting — the index does not need to know.

**Historical, not reproducible:** the prototype, its recall fixture and its
benchmark were deleted with the rejection, so the figures above cannot be
reproduced from the current tree. They are preserved here and in the README
as the record of the decision, and any future quantization proposal must
re-measure from scratch against this baseline.
