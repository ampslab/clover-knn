#pragma once

#include <cassert>
#include <cmath>   // for INFINITY; numeric limits doesn't work

#include "cuda_util.cuh"
#include "spatial.cuh"
#include "bitonic-shared.cuh"

namespace { // anonymous
} // namespace anonymous

namespace bitonic {

template < std::size_t ROUNDS >
__global__
void Query( idx_t * Qps, idx_t * solutions_knn, float *solutions_distances, int K, int Points_num, float* points)
{
    static_assert( "ROUNDS must be strictly positive" && ROUNDS > 0llu );

    int constexpr warp_size = 32;

    int const lane_id = threadIdx.x;
    int const local_query_id = threadIdx.y;
    int const query_batch_id = blockIdx.x;
    int const query_batch_size = blockDim.y;
    int const global_query_id = query_batch_id * query_batch_size + local_query_id;
    
    idx_t best_point_id[ROUNDS];
    #pragma unroll
    for( int r = 0; r < ROUNDS; ++r ) { best_point_id[ r ] = lane_id + r * warp_size; }

    // Be careful here! Either the whole warp must be on or the whole
    // warp must return, because it is needed for correctness in the
    // bitonic sort.

    if( global_query_id >= Points_num ) { return; }

    float const q_x = points[ global_query_id * dim ];
    float const q_y = points[ global_query_id * dim + 1 ];
    float const q_z = points[ global_query_id * dim + 2 ];

    float best_distance[ROUNDS];
    #pragma unroll
    for( int r = 0; r < ROUNDS; ++r )
    {
        int const global_lane_id = r * warp_size + lane_id;
        if( r < ROUNDS - 1 || global_lane_id < Points_num ) // r < ROUNDS-1 might be useful for compiler
        {
            best_distance[r] = spatial::l2dist( q_x, q_y, q_z, points + best_point_id[r] * dim );
        }
        else
        {
            best_distance[r] = INFINITY;
        }
    }

    sort<true, ROUNDS>( best_point_id, best_distance );

    float next_distance = INFINITY;

    for( idx_t i = warp_size * ROUNDS; i < Points_num; i += warp_size )
    {
        // Note that we need all threads to sort.
        // If next_point_id >= Points_num, this thread will use the value
        // for next_distance from the previous round, which is known
        // already not to be in the top-k. If there was no previous
        // round, it will be infinity.

        idx_t next_point_id = i + lane_id;
        if( next_point_id < Points_num )
        {
            next_distance = spatial::l2dist( q_x, q_y, q_z, points + next_point_id * dim );
        }

        float const kth_distance = __shfl_sync( 0xFFFFFFFF, best_distance[ ROUNDS - 1 ], K % warp_size );
        if( __any_sync( 0xFFFFFFFF, next_distance < kth_distance ) )
        {
            sort<false, 1>( &next_point_id, &next_distance );
            if( next_distance < best_distance[ROUNDS - 1] )
            {
                util::swap( next_distance, best_distance[ROUNDS - 1] );
                util::swap( next_point_id, best_point_id[ROUNDS - 1] );
            }
            sort<true, ROUNDS>( best_point_id, best_distance, warp_size * ROUNDS );
        }
    }

    // Copy result from registers to `solutions` vectors
    #pragma unroll
    for( int r = 0; r < ROUNDS; ++r )
    {
        int const global_lane_id = r * warp_size + lane_id;
        if( global_lane_id < K )
        {
            int const neighbour_location = global_query_id * K + global_lane_id;
            solutions_knn[ neighbour_location ] = best_point_id[r];
            solutions_distances[ neighbour_location ] = best_distance[r];
        }
    }
}

template <class R>
void knn_gpu(std::size_t n, R *data, std::size_t q, idx_t *queries, std::size_t k, idx_t *results_knn, R *results_distances)
{
    std::size_t constexpr block_size = 128;
    std::size_t constexpr warp_size = 32;
    std::size_t constexpr queries_per_block = block_size / warp_size;
    std::size_t const num_blocks = util::CEIL_DIV(n, queries_per_block);

    switch (util::CEIL_DIV(k, warp_size))
    {
        case 1: { Query<1> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data); } break;
        case 2: { Query<2> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data); } break;
        case 3: { Query<4> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data); } break;
        case 4: { Query<4> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data); } break;
        case 5: { Query<8> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data); } break;
        case 6: { Query<8> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data); } break;
        case 7: { Query<8> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data); } break;
        case 8: { Query<8> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data); } break;
        default: assert(false && "Rounds required to fulfill k value will exceed thread register allotment.");
    }
    CHECK_ERROR("Running scan kernel.");
}

} // namespace bitonic
