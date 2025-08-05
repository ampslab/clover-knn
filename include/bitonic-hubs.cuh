#pragma once

#include <cassert>
#include <cmath>   // for INFINITY; numeric limits doesn't work
#include <curand_kernel.h> 
#include <cfloat>
#include <fstream>
#include <string>
#include <sstream>

#include "cuda_util.cuh"
#include "spatial.cuh"
#include "bitonic-shared.cuh"

namespace { // anonymous

idx_t constexpr H = 2048;
idx_t constexpr warp_size = 32;

} // namespace anonymous

namespace bitonic_hubs {
__device__ int logPopCounts[33];

__global__ void Randomly_Select_Hubs(idx_t n, idx_t * dH){
    idx_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int seed = 1234 + idx;

    if( idx < H){
        unsigned int rand_num = (seed * 1103515245 + 12345) % 2147483648;
        dH[idx] = rand_num % n;
    }
}

/**
 * Produces column-major distance matrix for each point to each hub
 */
template <class R>
__global__ void Calculate_Distances(idx_t b_id, idx_t b_size, idx_t n, idx_t const* dH, R *distances, R const* points, idx_t *hub_counts, idx_t *dH_assignments)
{
    assert( "Must have at least one hub" && H > 0 );

    // TODO: Check if we can launch more threads for this kernel.
    // I don't think we need the for loop if we launch H-fold more threads

    idx_t idx = blockIdx.x * blockDim.x + threadIdx.x + b_id * b_size;
    idx_t idx_within_b = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < n && idx_within_b < b_size)
    {
        float q_x = points[ idx * dim ];
        float q_y = points[ idx * dim + 1];
        float q_z = points[ idx * dim + 2];

        float minimal_dist = FLT_MAX;
        idx_t assigned_H   = H + 1;       // should be impossible

        for(idx_t h = 0; h < H; h++)
        {
            // Steps column-major, i.e., increment by num points
            float next_hub_distance = sqrt( spatial::l2dist( q_x, q_y, q_z, &points[ dim * dH[h] ]) );
            distances[ h * b_size + idx_within_b ] = next_hub_distance;
            if( next_hub_distance < minimal_dist )
            {
                assigned_H = h;
                minimal_dist = next_hub_distance;
            }
        }

        dH_assignments[idx] = assigned_H;
        atomicAdd( &hub_counts[assigned_H], 1 );
    }
}

template < typename T >
__device__ __forceinline__
void prefix_sum_warp( T & my_val )
{
    int constexpr FULL_MASK = 0xFFFFFFFF;
    int constexpr warp_size = 32;

    for( int stride = 1; stride < warp_size; stride = stride << 1 )
    {
        __syncwarp();
        T const paired_val = __shfl_up_sync( FULL_MASK, my_val, stride );
        if( threadIdx.x >= stride )
        {
            my_val += paired_val;
        }
    }
}

template < typename T >
__global__
void fused_prefix_sum_copy( T *arr, T * copy )
{
    // Expected grid size: 1 x  1 x 1
    // Expected CTA size: 32 x 32 x 1

    // lazy implementation for now. Not even close to a hot spot.
    // just iterate H with one thread block and revisit if we start
    // using *very* large H, e.g., H > 8096, or it shows up in profile.

    assert( "H is a power of 2."  && __popc( H ) == 1 );
    assert( "H uses a full warp." && H >= 32 );

    int const lane_id = threadIdx.x;
    int const warp_id = threadIdx.y;
    int const th_id   = warp_id * blockDim.x + lane_id;

    if( th_id >= H ) { return; } // guard clause for syncthreads later

    // the first location of smem will contain the sum of all the
    // size-1024 chunks so far. The remaining 32 are a staging site
    // to propagate warp-level results across warps.
    int const shared_memory_size = 32 + 1;
    __shared__ T smem[ shared_memory_size ];
    if( th_id == 0 ) { smem[ 0 ] = 0; }

    // iterate in chunks of 1024 at a time
    for( int i = th_id; i < H ; i = i + blockDim.x * blockDim.y )
    {

        T my_val = arr[ i ];

        prefix_sum_warp( my_val );

        // compute partial sums over warp-level results
        // first, last lane in each warp copies result to smem for sharing
        if( lane_id == ( blockDim.x - 1) )
        {
            smem[ warp_id + 1 ] = my_val;
        }
        __syncthreads(); // safe because H is a power of 2 & guard clause earlier

        T sum_of_chunk_sofar = 0;

        // first warp computes prefix scan over 32 warp-level sums
        if( warp_id == 0 )
        {
            // fetch other warps' data from smem
            T warp_level_sum = smem[ lane_id + 1 ]
				            + smem[ 0 ] * ( lane_id == 0 );
            prefix_sum_warp( warp_level_sum );

            // write results back out to smem to broadcast to other warps
            // also update smem[ 0 ] to be first sum for next chunk
            smem[ lane_id + 1 ] = warp_level_sum;
            if( lane_id == ( blockDim.x - 1 ) )
            {
                sum_of_chunk_sofar = warp_level_sum;
            }
        }

        // propagate partial results across all threads
        // each thread only needs the partial sum for its warp
        __syncthreads(); // safe for same reasons as previous sync

        my_val += smem[ warp_id ];

        arr [ i ] = my_val;
        copy[ i ] = my_val;

        if(warp_id == 0 && lane_id == ( blockDim.x - 1 )) { smem[0] = sum_of_chunk_sofar; }
    }
}

/**
 * Physically resorts an array with a small domain of V unique values using O(nV) work using an out-of-place
 * struct-of-arrays decomposition.
 */
template <class R>
__global__
void BucketSort( idx_t n, R * arr_x, R *arr_y, R *arr_z, idx_t * arr_idx, R const* points, idx_t const* dH_assignments, idx_t * dH_psum )
{
    
    idx_t const idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < n )
    {
        idx_t const hub_idx = dH_assignments[idx];
        idx_t const loc = atomicAdd(&dH_psum[hub_idx], 1);

        arr_x[loc] = points[idx*dim+0];
        arr_y[loc] = points[idx*dim+1];
        arr_z[loc] = points[idx*dim+2];
        arr_idx[loc] = idx;
    }
}

__global__ void set_max_float(float *D, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        D[idx] = __FLT_MAX__;
    }
}

/**
 * Builds the HxH distance matrix, D, in which the asymmetric distance from hub H_i to hub H_j
 * is the distance from H_i to the closest point in H_j.
 */
__global__
void Construct_D( float const * distances, idx_t const * assignments, idx_t b_id, idx_t b_size, idx_t n, float * D )
{
    
    int constexpr shared_memory_size = 2048;

    assert( "Array fits in shared memory" && H <= shared_memory_size );

    // Each thread block will work on one row of the HxH matrix
    // unless the hub is empty in which case this thread block will just return.
    int const hub_id = blockIdx.x;

    float const * this_hubs_distances = &distances[ hub_id * b_size ];

    int const block_level_lane_id = threadIdx.x;
    assert( "Expect to have one __shared__ lane per thread" && block_level_lane_id < shared_memory_size );

    // Initialise row to a sequence of really large values in shared memory
    __shared__ int s_dists[shared_memory_size];

    int R = int (( H + blockDim.x  -1 ) / blockDim.x );

    for ( int r = 0; r < R; r ++)
    {
        if( r * blockDim.x + block_level_lane_id < H )
        {
            // IEEE-754 max exponent, no mantissa, no sign bit
            // This value sorts to back as both int and as float
            // compared to our domain
            s_dists[ r * blockDim.x + block_level_lane_id ] = 0x7f000000;
        }
    }

    __syncwarp(); __syncthreads();

    for( idx_t p = block_level_lane_id; p < b_size; p += blockDim.x )
    {
        idx_t idx = b_id * b_size + p;
        if( idx < n)
        {
            idx_t const this_H = assignments[ idx ];
            assert( "Retrieved a valid hub id" && this_H < H );
            atomicMin( &s_dists[this_H], __float_as_int( this_hubs_distances[ p ] ) );
            // Note: the reinterpret cast is necessary because there is no atomic min
            // defined for floats. Still, it would be nice to find a better solution
            // than this. Per nvidia forums, it seems that CUDA might not follow the IEEE
            // standard exactly? If this works, it should be a more mainstream hack?
        }
    }

    __syncwarp(); __syncthreads();

    for ( int r = 0; r < R; r ++)
    {
        if( r * blockDim.x + block_level_lane_id < H )
        {
            atomicMin(&s_dists[ r * blockDim.x + block_level_lane_id ],  __float_as_int( D[ H * hub_id + r * blockDim.x + block_level_lane_id ] ));
            D[ H * hub_id + r * blockDim.x + block_level_lane_id ] = __int_as_float( s_dists[ r * blockDim.x + block_level_lane_id ] );
        } 
    }
}

/**
 * Sorts the HxH D matrix and decomposes it into a separate matrix of ids and distances
 */
template < typename R, int ROUNDS >
__global__
void fused_transform_sort_D( R     const * D // square matrix with lower dist bound from hub i to j
                           , idx_t       * sorted_hub_ids  // square matrix where (i,j) is id of j'th closest hub to i
                           , R           * sorted_hub_dist // square matrix where (i,j) is dist of j'th closest hub to i
                           )
{
    // NOTE: this currently uses a lot of smem, reducing warp occupancy by 2x at H=1024.
    __shared__ R smem[ 2 * H ];

    // each block will sort one row of size H.
    // each thread is responsible for determining the final contents of one cell
    auto  const     warp_size = 32u;
    auto  const     block_size = 1024u;
    idx_t const     lane_id   = threadIdx.x;
    idx_t const     warp_id   = threadIdx.y;
    idx_t const     sort_id   = warp_id * warp_size + lane_id;
    idx_t const     hub_id    = blockIdx.x;

    if(sort_id >= H || hub_id >=H) {return;}

    // each thread grabs the contents of its cell in the input distance matrix
    R     dist[ROUNDS] ;
    idx_t hub[ROUNDS] ;

    for (int r=0; r< ROUNDS; r++)
    {
        dist[r] = D[ H * hub_id + sort_id + block_size * r ];
        hub[r]   = sort_id + block_size * r;
    }

    // create num_hubs >> 5 sorted runs in registers
    //bitonic::sort<warp_id % 2, 1>( &hub, &dist );
    //branch divergence here

    for ( int r = 0; r < ROUNDS; r ++)
    {
        if ( warp_id % 2 == 0  ) {
            bitonic::sort<true, 1>( &hub[r], &dist[r] );  
        } else {
            bitonic::sort<false, 1>( &hub[r], &dist[r] ); 
        }
    }

    // perform repeated merges with a given number of cooperating threads
    for( idx_t coop = warp_size << 1; coop <= H; coop = coop << 1 )
    {
        // do first steps of merge in shared memory 
        for( idx_t stride = coop >> 1; stride >= warp_size; stride = stride >> 1 )
        {
            for ( int r = 0; r < ROUNDS; r ++)
            {
                int const global_lane_id = r * block_size + sort_id;
                smem[ global_lane_id ] = dist[r];
                smem[ global_lane_id + H ] = float(hub[r]);
            }

            __syncthreads();

            for ( int r = 0; r < ROUNDS; r ++)
            {
                int const global_lane_id = r * block_size + sort_id;
                // TODO: optimise this part to reduce trips to smem somehow
                // something more tiled? each thread only reads two vals per sync
                // TODO: this is a guaranteed bank conflict (BC) followed immediately
                // by a sync to force *all* threads to wait for the BC

                // TODO: this is a guaranteed bank conflict, too.
                // but maybe these are inevitable anyway due
                // to pigeon hole principle?
                idx_t paired_thread = (global_lane_id)  ^ stride;
                R     const paired_dist = smem[ paired_thread ];
                idx_t const paired_hub  = int(smem[ paired_thread + H ]);
        
                if( ( paired_thread > global_lane_id && ( global_lane_id & coop ) == 0 && ( paired_dist < dist[r] ) )
                || ( paired_thread < global_lane_id && ( global_lane_id & coop ) != 0 && ( paired_dist < dist[r] ) )
                || ( paired_thread > global_lane_id && ( global_lane_id & coop ) != 0 && ( paired_dist > dist[r] ) )
                || ( paired_thread < global_lane_id && ( global_lane_id & coop ) == 0 && ( paired_dist > dist[r] ) ) )
                {
                    dist[r] = paired_dist;
                    hub[r]  = paired_hub;
                }
                __syncthreads();

            }
        }

        for ( int r = 0; r < ROUNDS; r ++)
        {
            int const global_lane_id = r * block_size + sort_id;
            if ( ( global_lane_id & coop ) == 0 ){
                bitonic::sort<true, 1>( &hub[r], &dist[r] );  
            } else {
                bitonic::sort<false, 1>( &hub[r], &dist[r] ); 
            }
        }

    }

    __syncthreads();

    for (int r = 0 ; r < ROUNDS ; r ++)
    {
        sorted_hub_ids [ hub_id * H + sort_id + r * block_size ] = hub[r];
        sorted_hub_dist[ hub_id * H + sort_id + r * block_size ] = dist[r];
    }

}


template < std::size_t ROUNDS >
__global__
__launch_bounds__(128, 16)
void Query( idx_t const * Qps, idx_t * solutions_knn, float *solutions_distances, int K, int Points_num, float const * points, idx_t * dH, idx_t const * arr_idx, float const * arr_x, float const * arr_y, float const * arr_z, idx_t const * iD, float const * dD, idx_t const * dH_psum, idx_t const * assignments,  int * hubs_scanned, int *pointsScanned)
{
    
    static_assert( "ROUNDS must be strictly positive" && ROUNDS > 0llu );

    int const lane_id           = threadIdx.x;
    int const query_id_in_block = threadIdx.y;
    int const queries_per_block = blockDim.y;
    int const query_sequence_id = blockIdx.x * queries_per_block + query_id_in_block;
    
    if( query_sequence_id >= Points_num ) { return; }

    int const qp = arr_idx[ query_sequence_id ];

    // Set up iteration counters
    int const hub_containing_qp = assignments[qp];
    int current_H = hub_containing_qp;
    int hubs_processed = 0;
    int poitns_scanned = 0;

    int scan_hub_from = dH_psum[ hub_containing_qp + hubs_processed ];
    int scan_hub_to   = dH_psum[ current_H + 1 ];


    // Initialise top-k registers with first bit_ceil(k) point ids,
    // i.e., the "values" for the key-value sort,
    // or up to end of this hub, whichever is smaller
    idx_t best_point_id[ROUNDS];

    #pragma unroll
    for( int r = 0; r < ROUNDS; ++r ) 
    {
        idx_t const   idx        = lane_id + r * warp_size;
        idx_t const * hub_points = arr_idx + scan_hub_from;
        
        if( scan_hub_from + idx < scan_hub_to ) 
        {
            best_point_id[r] = hub_points[idx];
        }
        // else doesn't matter; it will sort to the back.
    }

    poitns_scanned += ((ROUNDS * warp_size) > (scan_hub_to - scan_hub_from))?(scan_hub_to - scan_hub_from):(ROUNDS * warp_size);
 
    // Be careful here! Either the whole warp must be on or the whole
    // warp must return, because it is needed for correctness in the
    // bitonic sort.

    // Move query point to registers
    float const q_x = points[ qp * dim ];
    float const q_y = points[ qp * dim + 1 ];
    float const q_z = points[ qp * dim + 2 ];

    // Initialise top-k registers with first bit_ceil(k) distances,
    // i.e., the "keys" for the key-value sort,
    // or up to end of this hub, whichever is smaller
    float best_distance[ROUNDS];

    #pragma unroll
    for( int r = 0; r < ROUNDS; ++r )
    {
        idx_t const idx = lane_id + r * warp_size;

        if( scan_hub_from + idx < scan_hub_to )
        {
            float const next_x = arr_x[ scan_hub_from + idx ];
            float const next_y = arr_y[ scan_hub_from + idx ];
            float const next_z = arr_z[ scan_hub_from + idx ];

            best_distance[r] = spatial::l2dist( q_x, q_y, q_z, next_x, next_y, next_z );//squared
        }
        else
        {
            best_distance[r] = FLT_MAX;
        }
    }

    // Sort initialised values to create a new top-k and extract k'th best score
    idx_t const lane_K  = (K - 1) % warp_size;
    idx_t const round_K = ROUNDS - 1;

    bitonic::sort<true, ROUNDS>( best_point_id, best_distance );
    float kth_distance = __shfl_sync( 0xFFFFFFFF, best_distance[ round_K ], lane_K );

    float h_x = points[ dH[hub_containing_qp] * dim ];
    float h_y = points[ dH[hub_containing_qp] * dim + 1 ];
    float h_z = points[ dH[hub_containing_qp] * dim + 2 ];


    // Set up triangle inequality parameters for early termination
    float dist_to_my_hub   = sqrt(spatial::l2dist( q_x, q_y, q_z, h_x, h_y, h_z ));
    float dist_to_this_hub = dD[ hub_containing_qp * H + hubs_processed ];

    // Iterate until we have the top-k!!
    scan_hub_from += ROUNDS * warp_size; // already done

    if( scan_hub_from >= scan_hub_to )   // go to next hub
    {
        if( ++hubs_processed < H )
        {
            current_H        = iD[ hub_containing_qp * H + hubs_processed ];
            dist_to_this_hub = dD[ hub_containing_qp * H + hubs_processed ];
            scan_hub_from    = dH_psum[ current_H ];
            scan_hub_to      = dH_psum[ current_H + 1 ];
        }
    }

    while( hubs_processed < H && sqrt( kth_distance ) > dist_to_this_hub - dist_to_my_hub )
    {
        // Get next 32 values and batch insert them into the top-k
        // Note that we need all threads to sort.
        // If there are fewer than 32 points left in the hub,
        // they will get sentinel values that sort to the back.

        idx_t next_point_id = Points_num;
        float next_distance = FLT_MAX;

        if( scan_hub_from + lane_id < scan_hub_to )
        {
            next_point_id      = arr_idx[ scan_hub_from + lane_id ];
            float const next_x =   arr_x[ scan_hub_from + lane_id ];
            float const next_y =   arr_y[ scan_hub_from + lane_id ];
            float const next_z =   arr_z[ scan_hub_from + lane_id ];

            next_distance = spatial::l2dist( q_x, q_y, q_z, next_x, next_y, next_z );
        }
        
        if( __any_sync( 0xFFFFFFFF, next_distance < kth_distance ) )
        {
            bitonic::sort<false, 1>( &next_point_id, &next_distance );
            if( next_distance < best_distance[ ROUNDS - 1 ] )
            {
                util::swap( next_distance, best_distance[ ROUNDS - 1 ] );
                util::swap( next_point_id, best_point_id[ ROUNDS - 1 ] );
            }
            bitonic::sort<true, ROUNDS>( best_point_id, best_distance, warp_size * ROUNDS );
            kth_distance = __shfl_sync( 0xFFFFFFFF, best_distance[ round_K ], lane_K );
        }

        // Advance iterators
        scan_hub_from += warp_size;
        poitns_scanned += (warp_size > (scan_hub_to - scan_hub_from))?(scan_hub_to - scan_hub_from) : warp_size;

        if( scan_hub_from >= scan_hub_to )
        {
            if( ++hubs_processed < H )
            {
                current_H        = iD[ hub_containing_qp * H + hubs_processed ];
                dist_to_this_hub = dD[ hub_containing_qp * H + hubs_processed ];
                scan_hub_from    = dH_psum[ current_H ];
                scan_hub_to      = dH_psum[ current_H + 1 ];
            }
        }
    }

    //hubs_scanned[query_sequence_id] = hubs_processed;
    //pointsScanned[query_sequence_id] = poitns_scanned;
    // Copy result from registers to `solutions` vectors
    #pragma unroll
    for( int r = 0; r < ROUNDS; ++r )
    {
        int const global_lane_id = r * warp_size + lane_id;
        if( global_lane_id < K )
        {
            int const neighbour_location = qp * K + global_lane_id;
            solutions_knn[ neighbour_location ] = best_point_id[r];
            solutions_distances[ neighbour_location ] = best_distance[r];
        }
    }
}

template <class R>
void C_and_Q(std::size_t n, R *data, std::size_t q, idx_t *queries, std::size_t k, idx_t *results_knn, R *results_distances)
{
    idx_t constexpr block_size = 1024;
    
    idx_t * dH;
    CUDA_CALL(cudaMalloc((void **) &dH, sizeof(idx_t) * H));

    idx_t * dH_psum, * dH_psum_copy, * dH_assignments, * d_psum_placeholder;
    CUDA_CALL(cudaMalloc((void **) &dH_psum,        sizeof(idx_t) * ( H + 1 )));
    CUDA_CALL(cudaMalloc((void **) &dH_psum_copy,   sizeof(idx_t) * ( H + 1 )));
    CUDA_CALL(cudaMalloc((void **) &d_psum_placeholder,   sizeof(idx_t) * ( H + 1)));
    CUDA_CALL(cudaMalloc((void **) &dH_assignments, sizeof(idx_t) * n));
    cudaMemset(dH_psum, 0, sizeof(idx_t) * (1+H));
    cudaMemset(dH_psum_copy, 0, sizeof(idx_t) * (1+H));
    cudaMemset(d_psum_placeholder, 0, sizeof(idx_t) * (1+H));
    cudaMemset(dH_assignments, 0, sizeof(idx_t) * n);

    float * distances;
    idx_t constexpr batch_size = 10000;
    idx_t batch_number = (n + batch_size -1) / batch_size;
    CUDA_CALL(cudaMalloc((void **) &distances, sizeof(R) * H * batch_size));

    float * arr_x, *arr_y, *arr_z;
    idx_t * arr_idx;
    CUDA_CALL(cudaMalloc((void **) &arr_x, sizeof(float) * n));
    CUDA_CALL(cudaMalloc((void **) &arr_y, sizeof(float) * n));
    CUDA_CALL(cudaMalloc((void **) &arr_z, sizeof(float) * n));
    CUDA_CALL(cudaMalloc((void **) &arr_idx, sizeof(idx_t) * n));

    float * D;
    CUDA_CALL(cudaMalloc((void **) &D, sizeof(float) * H * H));

    idx_t *iD;
    float * dD;
    CUDA_CALL(cudaMalloc((void **) &iD, sizeof(idx_t) * H * H));
    CUDA_CALL(cudaMalloc((void **) &dD, sizeof(float) * H * H));

    std::size_t num_blocks = (H + block_size - 1) / block_size;
    Randomly_Select_Hubs<<<num_blocks, block_size>>>(n, dH);
    CHECK_ERROR("Randomly_Select_Hubs.");

    num_blocks = (batch_size + block_size - 1) / block_size;

    idx_t batch_id;

    set_max_float<<<( H * H + block_size - 1 ) / block_size, block_size>>>(D, H * H);

    for (batch_id = 0; batch_id < batch_number; batch_id++)
    {
        Calculate_Distances<<<num_blocks, block_size>>>(batch_id, batch_size, n, dH, distances, data, dH_psum, dH_assignments);
        Construct_D<<<H, block_size>>>(distances, dH_assignments, batch_id, batch_size, n, D);
    }
    cudaFree( distances );
    
    fused_prefix_sum_copy<<<1, dim3( warp_size,  warp_size, 1)  >>>(dH_psum, dH_psum_copy);
    cudaMemcpy(d_psum_placeholder, dH_psum_copy, (H + 1 )* sizeof(idx_t), cudaMemcpyDeviceToDevice);
    
    cudaMemcpy(dH_psum_copy + 1, d_psum_placeholder, H * sizeof(idx_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dH_psum + 1, d_psum_placeholder, H * sizeof(idx_t), cudaMemcpyDeviceToDevice);
    cudaMemset(dH_psum, 0, sizeof(idx_t));
    cudaMemset(dH_psum_copy, 0, sizeof(idx_t));
    CHECK_ERROR("Fused_prefix_sum_copy.");
    cudaFree(d_psum_placeholder);

    num_blocks = (n + block_size - 1) / block_size;

    BucketSort<<<num_blocks,  block_size>>>(n, arr_x, arr_y, arr_z, arr_idx, data, dH_assignments, dH_psum_copy);
    CHECK_ERROR("BucketSort.");
    cudaFree(dH_psum_copy);

    fused_transform_sort_D<float, (H + block_size - 1) / block_size> <<<H, dim3 { warp_size, block_size/warp_size, 1 }>>> (D, iD, dD);
    CHECK_ERROR("Sort_D.");
    cudaFree(D); 

    int hubsScanned[n], pointsScanned[n];
    int * d_hubsScanned, * d_pointsScanned;
    CUDA_CALL(cudaMalloc((void **) &d_hubsScanned, sizeof(int)* 1));//change here if want to log
    CUDA_CALL(cudaMalloc((void **) &d_pointsScanned, sizeof(int)* 1));

    std::size_t constexpr queries_per_block = 128 / warp_size;
    num_blocks = util::CEIL_DIV(n, queries_per_block);

    switch (util::CEIL_DIV(k, warp_size))
    {
        case 1: { Query<1> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data, 
                dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, d_hubsScanned, d_pointsScanned); } break;
        case 2: { Query<2> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data, 
                dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, d_hubsScanned, d_pointsScanned); } break;
        case 3: { Query<3> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data, 
                dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, d_hubsScanned, d_pointsScanned); } break;
        case 4: { Query<4> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data, 
                dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, d_hubsScanned, d_pointsScanned); } break;
        case 5: { Query<5> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data, 
                dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, d_hubsScanned, d_pointsScanned); } break;
        case 6: { Query<6> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data, 
                dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, d_hubsScanned, d_pointsScanned); } break;
        case 7: { Query<7> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data, 
                dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, d_hubsScanned, d_pointsScanned); } break;
        case 8: { Query<8> <<<num_blocks, dim3 { warp_size, queries_per_block, 1 }>>>(queries, results_knn, results_distances, k, n, data, 
                dH, arr_idx, arr_x, arr_y, arr_z, iD, dD, dH_psum, dH_assignments, d_hubsScanned, d_pointsScanned); } break;
        default: assert(false && "Rounds required to fulfill k value will exceed thread register allotment.");
    }

    //cudaMemcpy(hubsScanned, d_hubsScanned, n * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(pointsScanned, d_pointsScanned, n * sizeof(int), cudaMemcpyDeviceToHost);

    //logHubsScanned(hubsScanned, pointsScanned, n);

    CHECK_ERROR("Running scan kernel.");
    cudaFree( iD );
    cudaFree( dD );
    cudaFree( dH_psum );
    cudaFree( dH_assignments );
    cudaFree( distances );
    cudaFree( arr_idx );
    cudaFree( arr_x );
    cudaFree( arr_y );
    cudaFree( arr_z );
}

} // namespace bitonic
