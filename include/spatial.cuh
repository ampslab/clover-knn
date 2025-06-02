#pragma once

using idx_t = unsigned int;
auto const dim = 3u;

// 3D L2 Distance Squared
template <typename T>
__device__ __host__
T l2dist(T *p1, T *p2)
{
    T const x = p1[0] - p2[0];
    T const y = p1[1] - p2[1];
    T const z = p1[2] - p2[2];
    return (x * x) + (y * y) + (z * z);
    // return norm3df(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]); // Not generic
}

namespace spatial {

// Same as l2dist(), but takes three registers and a pointer as input (i.e., overload).
template < typename T >
__device__ __host__
T l2dist( T q_x, T q_y, T q_z, T const* p )
{
    T const squared_distance = (q_x - p[0]) * (q_x - p[0])
                             + (q_y - p[1]) * (q_y - p[1])
                             + (q_z - p[2]) * (q_z - p[2]);
    
    return squared_distance;
    // return sqrt( squared_distance );
}

// Same as l2dist(), but takes six registers as input (i.e., overload).
template < typename T >
__device__ __host__
T l2dist( T p_x, T p_y, T p_z, T q_x, T q_y, T q_z )
{
    T const squared_distance = (p_x - q_x) * (p_x - q_x)
                             + (p_y - q_y) * (p_y - q_y)
                             + (p_z - q_z) * (p_z - q_z);

    return squared_distance;
    // return sqrt( squared_distance );
}

template <typename T>
__device__ __host__
T l2dist(T const *p1, T *p2)
{
    T const x = p1[0] - p2[0];
    T const y = p1[1] - p2[1];
    T const z = p1[2] - p2[2];
    return (x * x) + (y * y) + (z * z);
    // return norm3df(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]); // Not generic
}
} // namespace spatial
