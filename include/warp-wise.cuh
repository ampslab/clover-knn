#pragma once

#include <cuda/std/limits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cassert>


#include "cuda_util.cuh"
#include "spatial.cuh"

namespace { // anonymous

auto constexpr NTHREADS_PER_WARP = 32u;
auto constexpr NLANES_PER_WARP = NTHREADS_PER_WARP - 1u;

// Change to 2048 on compute canada
auto constexpr NTHREADS_PER_BLOCK = 1024u;
auto constexpr NQUERIES_PER_BLOCK = NTHREADS_PER_BLOCK / NTHREADS_PER_WARP;


template <typename key_t, typename value_t>
__device__ void shuffle_if_greater(int pop, key_t & key, value_t & val )
{
    if ( threadIdx.x >= pop ) {
        key = __shfl_up_sync(__activemask(), key, 1);
        val = __shfl_up_sync(__activemask(), val, 1);
    }
}


template <typename key_t, typename value_t>
__device__ void insert_new_kv_pair(int pop, key_t & old_key, key_t new_key, value_t & old_val, value_t new_val )
{
    if (threadIdx.x == pop) {
        old_key = new_key;
        old_val = new_val;
    }
}


template <typename key_t, typename value_t>
__device__ void broadcast_last_lane(key_t & old_key, key_t new_key, value_t & old_val, value_t new_val )
{
    auto constexpr last_lane = NTHREADS_PER_WARP - 1;

    old_key = __shfl_sync(0xFFFFFFFF, new_key, last_lane);
    old_val = __shfl_sync(0xFFFFFFFF, new_val, last_lane);
}

template <class R, typename dist_t>
__device__ void single_point_update(R *query, R const* point_data, idx_t point_id, dist_t & dist, idx_t & p, int k )
{
    auto const di = l2dist(query, point_data);
    int const pop = __popc(__ballot_sync(0xFFFFFFFF, dist <= di));

    if (pop != k)
    {
        shuffle_if_greater(pop, dist, p);
        insert_new_kv_pair(pop, dist, di, p, point_id);
    }
}


template <class R>
__global__ void brute_force_scan_single(idx_t n, R *data, idx_t *queries, idx_t *results, idx_t k)
{
    R    *q = data + (queries[blockIdx.x] * dim);        // Pick up query point
    R     d = ::cuda::std::numeric_limits<R>::max();     // Distance to point
    idx_t p = ::cuda::std::numeric_limits<idx_t>::max(); // Point itself

    // Step through all data points
    for (idx_t i = 0; i < n; ++i) {
        single_point_update( q, data + (i * dim), i, d, p, k );
    }

    results[(blockDim.x * blockIdx.x) + threadIdx.x] = p;
}

template <class R>
__global__ void brute_force_scan(idx_t n, R *data, idx_t q, idx_t *queries, idx_t *results, idx_t k)
{
    idx_t current_query_idx = (blockIdx.x * NQUERIES_PER_BLOCK) + threadIdx.y;
    if ((threadIdx.x < k) && (current_query_idx < q)) {
        R    *q = data + (queries[current_query_idx] * dim); // Pick up query point
        R     d = ::cuda::std::numeric_limits<R>::max();     // Distance to point
        idx_t p = ::cuda::std::numeric_limits<idx_t>::max(); // Point itself

        // Step through all data points
        for (idx_t i = 0; i < n; ++i) {
            single_point_update( q, data + (i * dim), i, d, p, k );
        }

        results[(k * current_query_idx) + threadIdx.x] = p;
    }
}

template <class R, std::size_t ROUNDS>
__global__ void brute_force_scan_round(idx_t n, R *data, idx_t q, idx_t *queries, idx_t *results_knn, R *results_distances, idx_t k)
{
    // Enforce correct round structure
    // static_assert(ROUNDS >= 1 && "Must perform at least one round.");
    // static_assert(ROUNDS <= 4 && "Too many rounds will exceed thread register allotment.");

    // Calculate current query index
    idx_t current_query_idx = (blockIdx.x * NQUERIES_PER_BLOCK) + threadIdx.y;

    if (current_query_idx < q) {
        R *q = data + (queries[current_query_idx] * dim); // Pick up query point

        R d[ROUNDS]; // Distances to points
        #pragma unroll
        for (idx_t i = 0; i < ROUNDS; ++i) d[i] = ::cuda::std::numeric_limits<R>::max();

        idx_t p[ROUNDS]; // Point indices
        #pragma unroll
        for (idx_t i = 0; i < ROUNDS; ++i) p[i] = ::cuda::std::numeric_limits<idx_t>::max();

        // Step through all data points
        for (idx_t i = 0; i < n; ++i) {
            idx_t current_point = i;
            R di = l2dist(q, data + (current_point * dim));
            #pragma unroll
            for (idx_t r = 0; r < ROUNDS; ++r) {
                int pop = __popc(__ballot_sync(0xFFFFFFFF, d[r] <= di));
                if (pop < NLANES_PER_WARP) { // TODO: verify & move before popcount
                    shuffle_if_greater(pop, d[r], p[r]);
                    insert_new_kv_pair(pop, d[r], di, p[r], current_point);
                    broadcast_last_lane(di, d[r], current_point, p[r]);
                }
            }
            // TODO: Don't broadcast on last iteration, allow for early return (some threads must not run for shfl)
        }

        if (threadIdx.x != 31) {
            auto * query_result_knn = results_knn + (k * current_query_idx);
            auto * query_result_distances = results_distances + (k * current_query_idx);
            #pragma unroll
            for (idx_t r = 0; r < ROUNDS - 1; ++r) {
                query_result_knn[ threadIdx.x + (r * NLANES_PER_WARP)] = p[r];
                query_result_distances[ threadIdx.x + (r * NLANES_PER_WARP)] = d[r];
            }

            auto const last_round_offset = ( ROUNDS - 1 ) * NLANES_PER_WARP;
            if ( threadIdx.x + last_round_offset < k )
            {
                query_result_knn[threadIdx.x + last_round_offset] = p[ROUNDS - 1];
                query_result_distances[threadIdx.x + last_round_offset] = d[ROUNDS - 1];
            }
        }
    }
}


} // namespace anonymous

namespace warpwise {

template <class R>
void knn_gpu(std::size_t n, R *data, std::size_t q, idx_t *queries, std::size_t k, idx_t *results_knn, R *results_distances)
{
    switch (util::CEIL_DIV(k, NLANES_PER_WARP))
    {
        case 1: { brute_force_scan_round<R, 1> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), dim3 { NTHREADS_PER_WARP, NQUERIES_PER_BLOCK, 1 } >>> (n, data, q, queries, results_knn, results_distances, k); } break;
        case 2: { brute_force_scan_round<R, 2> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), dim3 { NTHREADS_PER_WARP, NQUERIES_PER_BLOCK, 1 } >>> (n, data, q, queries, results_knn, results_distances, k); } break;
        case 3: { brute_force_scan_round<R, 3> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), dim3 { NTHREADS_PER_WARP, NQUERIES_PER_BLOCK, 1 } >>> (n, data, q, queries, results_knn, results_distances, k); } break;
        case 4: { brute_force_scan_round<R, 4> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), dim3 { NTHREADS_PER_WARP, NQUERIES_PER_BLOCK, 1 } >>> (n, data, q, queries, results_knn, results_distances, k); } break;
        case 5: { brute_force_scan_round<R, 5> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), dim3 { NTHREADS_PER_WARP, NQUERIES_PER_BLOCK, 1 } >>> (n, data, q, queries, results_knn, results_distances, k); } break;
        case 6: { brute_force_scan_round<R, 6> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), dim3 { NTHREADS_PER_WARP, NQUERIES_PER_BLOCK, 1 } >>> (n, data, q, queries, results_knn, results_distances, k); } break;
        case 7: { brute_force_scan_round<R, 7> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), dim3 { NTHREADS_PER_WARP, NQUERIES_PER_BLOCK, 1 } >>> (n, data, q, queries, results_knn, results_distances, k); } break;
        case 8: { brute_force_scan_round<R, 8> <<< util::CEIL_DIV(q, NQUERIES_PER_BLOCK), dim3 { NTHREADS_PER_WARP, NQUERIES_PER_BLOCK, 1 } >>> (n, data, q, queries, results_knn, results_distances, k); } break;
        default: assert(false && "Rounds required to fulfill k value will exceed thread register allotment.");
    }
    CHECK_ERROR("Running scan kernel.");
}

} // namespace warpwise
