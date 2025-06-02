#pragma once

#include <cuda/std/limits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cassert>

#include "cuda_util.cuh"
#include "spatial.cuh"

#include <faiss/gpu/utils/Select.cuh>

namespace { // anonymous
/*
auto constexpr NTHREADS_PER_WARP = 32u;
auto constexpr NLANES_PER_WARP = NTHREADS_PER_WARP - 1u;

// Change to 2048 on compute canada
auto constexpr NTHREADS_PER_BLOCK = 1024u;
auto constexpr NQUERIES_PER_BLOCK = NTHREADS_PER_BLOCK / NTHREADS_PER_WARP;
*/
template <class R, std::size_t WarpQ, std::size_t ThreadQ>
__global__ void brute_force_scan(idx_t n, R *data, idx_t q, idx_t *queries, idx_t *results, R *distances, idx_t k)
{
    faiss::gpu::WarpSelect<
            R,
            idx_t,
            false,
            faiss::gpu::Comparator<R>,
            WarpQ /* NumWarpQ */,
            ThreadQ /* NumThreadQ */,
            NTHREADS_PER_WARP * NQUERIES_PER_BLOCK /* TODO: active? ThreadsPerBlock */
	>
        heap(::cuda::std::numeric_limits<R>::max() /*initK*/, -1 /* initV */, k);

    idx_t lane = threadIdx.x % NTHREADS_PER_WARP;
    idx_t file = threadIdx.x / NTHREADS_PER_WARP;

    idx_t current_query_idx = (blockIdx.x * NQUERIES_PER_BLOCK) + file;
    if (current_query_idx < q) {
        R *q = data + (queries[current_query_idx] * dim); // Pick up query point

	idx_t limit = (n / NTHREADS_PER_WARP) * NTHREADS_PER_WARP;
	idx_t i = lane;
	data += lane * dim;

        for (; i < limit; i += NTHREADS_PER_WARP) {
            heap.add(l2dist(q, data), i);
            data += NTHREADS_PER_WARP * dim;
        }

    	// Handle non-warp multiple remainder
    	if (i < n) {
            heap.addThreadQ(l2dist(q, data), i);
    	}

	heap.reduce();

	//*(distances + (k * current_query_idx) + threadIdx.x) = 5;
	
	heap.writeOut(
            distances + (k * current_query_idx),
            results + (k * current_query_idx),
	    k
	);
	
    }
}

} // namespace anonymous

namespace faiss_warp_select {

template <class R>
void knn_gpu(std::size_t n, R *data, std::size_t q, idx_t *queries, std::size_t k, idx_t *results_knn, R *results_distances)
{
   // dim3 { NTHREADS_PER_WARP, NQUERIES_PER_BLOCK, 1 } 
    if (k == 1) {
    	brute_force_scan<R, 1, 1> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), NTHREADS_PER_WARP * NQUERIES_PER_BLOCK >>> (n, data, q, queries, results_knn, results_distances, k);
    } else if (k <= 32) {
    	brute_force_scan<R, 32, 2> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), NTHREADS_PER_WARP * NQUERIES_PER_BLOCK >>> (n, data, q, queries, results_knn, results_distances, k);
    } else if (k <= 64) {
    	brute_force_scan<R, 64, 3> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), NTHREADS_PER_WARP * NQUERIES_PER_BLOCK >>> (n, data, q, queries, results_knn, results_distances, k);
    } else if (k <= 128) {
    	brute_force_scan<R, 128, 3> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), NTHREADS_PER_WARP * NQUERIES_PER_BLOCK >>> (n, data, q, queries, results_knn, results_distances, k);
    } else if (k <= 256) {
    	brute_force_scan<R, 256, 4> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), NTHREADS_PER_WARP * NQUERIES_PER_BLOCK >>> (n, data, q, queries, results_knn, results_distances, k);
    } else if (k <= 512) {
    	brute_force_scan<R, 512, 8> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), NTHREADS_PER_WARP * NQUERIES_PER_BLOCK >>> (n, data, q, queries, results_knn, results_distances, k);
    }
    
    CHECK_ERROR("Running scan kernel.");
}

} // namespace faiss_warp_select
