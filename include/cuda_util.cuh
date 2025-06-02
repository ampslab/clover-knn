#pragma once

#define CUDA_CALL(exp)                                       \
    do {                                                     \
        cudaError res = (exp);                               \
        if(res != cudaSuccess) {                             \
            printf("Error at %s:%d\n %s\n",                  \
                __FILE__,__LINE__, cudaGetErrorString(res)); \
           exit(EXIT_FAILURE);                               \
        }                                                    \
    } while(0)
    
#define CHECK_ERROR(msg)                                             \
    do {                                                             \
        cudaError_t err = cudaGetLastError();                        \
        if(cudaSuccess != err) {                                     \
            printf("Error (%s) at %s:%d\n %s\n",                     \
                (msg), __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

namespace util {

constexpr int CEIL_DIV( int x, int y )
{
    return 1 + (((x) - 1) / (y));
}

/** Swap the contents of two variables using copy-assignment. */
template < typename T >
__device__
void swap( T & left, T & right )
{
    // Note: no std::swap() on the device.

    T left_temp = left;
    left = right;
    right = left_temp;
}

} // namespace util
