#pragma once

#include <cassert>
#include <thrust/host_vector.h>

#include "spatial.cuh"

namespace { // anonymous

using score_t = std::pair< float, idx_t >; 

inline
bool operator < ( score_t const& l, score_t const& r )
{
    if( l.first == r.first ){ return l.second < r.second; }
    else                    { return l.first  < r.first ; }
}

inline
bool operator == ( score_t const& l, score_t const& r )
{
    return l.first == r.first && l.second == r.second;
}

} // namespace anonymous

inline
auto cpu_brute_force( float * data
                    , idx_t * Q
                    , std::size_t num_points
                    , std::size_t num_queries
                    , std::size_t num_dimensions
                    , std::size_t k )                      
    -> std::pair< thrust::host_vector<idx_t>, thrust::host_vector<float> >
{
    assert( "Please choose a meaningfully large test case" && k < num_points );

    thrust::host_vector<idx_t> all_results_knn( num_queries * k );
    thrust::host_vector<float> all_results_distances( num_queries * k );

    for( idx_t i = 0, q = Q[i]; i < num_queries; q = Q[++i] )
    {
        thrust::host_vector< score_t > q_result;
        q_result.reserve( k + 1 );

        for( idx_t p = 0; p < k + 1; ++p )
        {
            float distance = l2dist( data + q * num_dimensions, data + p * num_dimensions );
            q_result.push_back( { distance, p } );
        }
        std::sort( q_result.begin(), q_result.end() );

        for( idx_t p = k + 1; p < num_points; ++p )
        {
            q_result.back() = { l2dist( data + q * num_dimensions, data + p * num_dimensions ), p };
            std::sort( q_result.begin(), q_result.end() );
        }
        
        for( int result_index = 0; result_index < k; ++result_index )
        {
            auto const [ distance, point_id ] = q_result[ result_index ];
            all_results_knn[ i * k + result_index ] = point_id;
            all_results_distances[ i * k + result_index ] = distance;
        }
    }

    return std::make_pair( all_results_knn, all_results_distances);
}
