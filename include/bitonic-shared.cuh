#pragma once

#include <cassert>
#include <cmath>   // for INFINITY; numeric limits doesn't work

#include "cuda_util.cuh"
#include "spatial.cuh"

namespace bitonic {

__device__ inline
bool should_swap( int lane_id, int paired_lane
                , float distance, float paired_distance
                , bool ascending )
{
    bool const is_lower_of_pair = lane_id < paired_lane;

    // Be careful here! The paired threads have to return with the same
    // value from this function to perform a correct swap. Failure will
    // cause deadlock. Negating comparison operators like < is incorrect.

    if( ascending )
    {
        return ( is_lower_of_pair && paired_distance < distance )
            || (!is_lower_of_pair && paired_distance > distance );
    }
    else
    {
        return ( is_lower_of_pair && paired_distance > distance )
            || (!is_lower_of_pair && paired_distance < distance );
    }
}

template < bool ASCENDING, std::size_t ROUNDS >
__device__
void sort( idx_t point_id[ROUNDS], float distance[ROUNDS], int start_at_step = 2 )
{
    int constexpr warp_size = 32;
    int const lane_id = threadIdx.x;

    for( int step = start_at_step; step <= warp_size * ROUNDS; step *= 2 )
    {
        for( int sub_step = step/2; sub_step > 0; sub_step /= 2 )
        {
            #pragma unroll
            for( int r = 0; r < ROUNDS; ++r )
            {
                int const global_lane_id = r * warp_size + lane_id;
                bool const direction = (!ASCENDING) != !( global_lane_id & step );

                if( sub_step >= warp_size )
                {
                    int const paired_register = r ^ ( sub_step / warp_size );
                    if( r < paired_register )
                    {
                        float const paired_distance = distance[paired_register];

                        if( should_swap( r, paired_register, distance[r], paired_distance, direction ) )
                        {
                            distance[paired_register] = distance[r];
                            distance[r] = paired_distance;
                            util::swap( point_id[r], point_id[paired_register] );
                        }
                    }
                }
                else
                {
                    int const paired_lane = lane_id ^ sub_step;
                    float const paired_distance = __shfl_sync(0xFFFFFFFF, distance[r], paired_lane);

                    if( should_swap( lane_id, paired_lane, distance[r], paired_distance, direction ) )
                    {
                        distance[r] = paired_distance;
                        point_id[r] = __shfl_sync(__activemask(), point_id[r], paired_lane);
                    }
                }
            }
            __syncthreads(); // is this necessary? they're step-locked *and* we just shfl_sync'd.
        }
    }
}

} // namespace bitonic
