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
template <class R, std::size_t WarpQ, std::size_t ThreadQ, std::size_t ThreadsPerBlock>
__global__ void brute_force_scan(idx_t n, R *data, idx_t q, idx_t *queries, idx_t *results, R *distances, idx_t k)
{
    constexpr idx_t ParallelWarps = ThreadsPerBlock / NTHREADS_PER_WARP;

    __shared__ R smemK[ParallelWarps * WarpQ];
    __shared__ idx_t smemV[ParallelWarps * WarpQ];

    faiss::gpu::BlockSelect<
            R,
            idx_t,
            false,
            faiss::gpu::Comparator<R>,
            WarpQ /* NumWarpQ */,
            ThreadQ /* NumThreadQ */,
            ThreadsPerBlock
	>
        heap(::cuda::std::numeric_limits<R>::max() /*initK*/, -1 /* initV */, smemK, smemV, k);

    R *query = data + (queries[blockIdx.x] * dim); // Pick up query point

    idx_t limit = (n / NTHREADS_PER_WARP) * NTHREADS_PER_WARP;
    idx_t i = threadIdx.x;
    data += i * dim;

    for (; i < limit; i += ThreadsPerBlock) {
        heap.add(l2dist(query, data), i);
        data += ThreadsPerBlock * dim;
    }

    // Handle non-warp multiple remainder
    if (i < n) {
        heap.addThreadQ(l2dist(query, data), i);
    }

    heap.reduce();

    for (idx_t j = threadIdx.x; j < k; j += ThreadsPerBlock) {
        distances[k * blockIdx.x + j] = smemK[j];
        results[k * blockIdx.x + j] = smemV[j];
    }
}

} // namespace anonymous

namespace faiss_block_select {

template <class R>
void knn_gpu(std::size_t n, R *data, std::size_t q, idx_t *queries, std::size_t k, idx_t *results_knn, R *results_distances)
{
    if (k == 1) {
    	brute_force_scan<R, 1, 1, 128> <<< q, 128 >>> (n, data, q, queries, results_knn, results_distances, k);
    } else if (k <= 32) {
    	brute_force_scan<R, 32, 2, 128> <<< q, 128 >>> (n, data, q, queries, results_knn, results_distances, k);
    } else if (k <= 64) {
    	brute_force_scan<R, 64, 3, 128> <<< q, 128 >>> (n, data, q, queries, results_knn, results_distances, k);
    } else if (k <= 128) {
    	brute_force_scan<R, 128, 3, 128> <<< q, 128 >>> (n, data, q, queries, results_knn, results_distances, k);
    } else if (k <= 256) {
    	brute_force_scan<R, 256, 4, 128> <<< q, 128 >>> (n, data, q, queries, results_knn, results_distances, k);
    } else if (k <= 512) {
    	brute_force_scan<R, 512, 8, 128> <<< q, 128 >>> (n, data, q, queries, results_knn, results_distances, k);
    }
    
    CHECK_ERROR("Running scan kernel.");
}

} // namespace faiss_block_select
