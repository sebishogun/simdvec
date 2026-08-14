//go:build scale

package simdvec

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"testing"
)

// The scale curve, behind a build tag because it allocates gigabytes.
//
//	go test -tags scale -run '^$' -bench BenchmarkScale -benchtime 5x .
//
// The plan budgeted about 9 GB of peak RSS, on the assumption the harness
// holds three copies of the corpus: the generated vectors, a naive index's
// per-vector copies, and simdvec's matrix. This holds one. Vectors are
// generated straight into the index and dropped, so peak is the matrix plus a
// row of scratch: 3 GB at 1M x 768 rather than 9. Nothing is lost by it -- the
// comparison against a naive index belongs in the small benchmarks, where it
// can run without a machine that has 9 GB to spare.
func BenchmarkScale(b *testing.B) {
	const dim = 768
	for _, n := range []int{200000, 500000, 1000000} {
		r := rand.New(rand.NewPCG(11, 12))
		ix := New(dim, Cosine)
		v := make([]float32, dim)
		for i := 0; i < n; i++ {
			for j := range v {
				v[j] = r.Float32()*2 - 1
			}
			if err := ix.Add(itoa(i), v); err != nil {
				b.Fatal(err)
			}
		}
		q := make([]float32, dim)
		for j := range q {
			q[j] = r.Float32()
		}

		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		b.Run(fmt.Sprintf("n=%d/dim=%d", n, dim), func(b *testing.B) {
			b.ReportMetric(float64(ms.HeapAlloc)/(1<<30), "GB-heap")
			b.ReportMetric(float64(n*dim*4)/(1<<30), "GB-matrix")
			for i := 0; i < b.N; i++ {
				if _, err := ix.Search(q, 10); err != nil {
					b.Fatal(err)
				}
			}
		})
		ix = nil
		runtime.GC()
	}
}
