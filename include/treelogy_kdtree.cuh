#include "treelogy/GPU/knearestneighbor/kdtree/gpu_non_lockstep/nn_gpu.h"
#include "treelogy/GPU/common/util_common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace treelogy{

template <class R>
void treelogy_kd_tree(std::size_t n, R *data, std::size_t q, idx_t *queries, std::size_t k, int *results_knn, R *results_distances)
{
    node *points;
    node *search_points;

    unsigned int npoints;
    unsigned int nsearchpoints;

    float* nearest_distance;
    unsigned int* nearest_point_index;

    float* nearest_distance_brute;
    unsigned int* nearest_point_index_brute;

    node * tree;

    float min = FLT_MAX;
    float max = FLT_MIN;
    int c;

    npoints = n;
    nsearchpoints = npoints;
    
    SAFE_CALLOC(points, npoints, sizeof(node));
    //SAFE_MALLOC(search_points, sizeof(float)*nsearchpoints*DIM);
    SAFE_CALLOC(search_points, nsearchpoints, sizeof(node));
    SAFE_MALLOC(nearest_distance, sizeof(float)*nsearchpoints*k);
    SAFE_MALLOC(nearest_point_index, sizeof(unsigned int)*nsearchpoints*k);

    for (int i = 0; i < n; i++) {
        points[i].point_index = i;
        cudaMemcpy(points[i].point, &data[i * dim], dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(search_points[i].point, &data[i * dim], dim * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaEvent_t start, stop, start_construction, stop_construction, start_query, stop_query;
    float elapsedTime, query_time, construction_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_construction);
    cudaEventCreate(&stop_construction);
    cudaEventCreate(&start_query);
    cudaEventCreate(&stop_query);


    cudaEventRecord(start, 0);

    cudaEventRecord(start_construction, 0);
    tree = construct_tree(points, 0, n-1, 0);
    cudaEventRecord(stop_construction, 0);
    cudaEventSynchronize(stop_construction);

    // *** GPU Kerel Call *** //
    gpu_tree * h_tree = gpu_transform_tree(tree);
    gpu_tree * d_tree = gpu_copy_to_dev(h_tree);

    // Allocate variables to store results of each thread
    node * d_search_points;
    float * d_nearest_distance = results_distances;
    int * d_nearest_point_index = results_knn;
/*
    #ifdef TRACK_TRAVERSALS	
    int *h_nodes_accessed;
    int *d_nodes_accessed;
    SAFE_CALLOC(h_nodes_accessed, nsearchpoints, sizeof(int));
    CUDA_SAFE_CALL(cudaMalloc(&d_nodes_accessed, sizeof(int)*nsearchpoints));
    CUDA_SAFE_CALL(cudaMemcpy(d_nodes_accessed, h_nodes_accessed, sizeof(int)*nsearchpoints, cudaMemcpyHostToDevice));
    #endif
*/
    // Read from but not written to
    CUDA_SAFE_CALL(cudaMalloc(&d_search_points, sizeof(node)*nsearchpoints));
    CUDA_SAFE_CALL(cudaMemcpy(d_search_points, search_points, sizeof(node)*nsearchpoints, cudaMemcpyHostToDevice));

    cudaEventRecord(start_query, 0);

    //gpu_print_tree_host(h_tree);
    dim3 grid(NUM_THREAD_BLOCKS, 1, 1);
    dim3 block(NUM_THREADS_PER_BLOCK, 1, 1);
    unsigned int smem_bytes = DIM*NUM_THREADS_PER_BLOCK*sizeof(float) + k*NUM_THREADS_PER_BLOCK*sizeof(int) + k*NUM_THREADS_PER_BLOCK*sizeof(float);
    nearest_neighbor_search<<<grid, block, smem_bytes>>>(*d_tree, nsearchpoints, d_search_points, d_nearest_distance, d_nearest_point_index, k
#ifdef TRACK_TRAVERSALS
                                                                                                            , d_nodes_accessed																											 
#endif
    );
    cudaEventRecord(stop_query, 0);        
    cudaEventSynchronize(stop_query);

    cudaEventRecord(stop, 0);  
    cudaEventSynchronize(stop);

    // Calculate elapsed timed
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventElapsedTime(&query_time, start_query, stop_query);
    cudaEventElapsedTime(&construction_time, start_construction, stop_construction);

    printf("Total execution time: %f ms\n", elapsedTime);
    printf("Construction time: %f ms\n", construction_time);
    printf("Query time: %f ms\n", query_time);
                            
    gpu_free_tree_dev(d_tree);
    gpu_free_tree_host(h_tree);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);      
    cudaEventDestroy(start_construction);
    cudaEventDestroy(stop_construction);   
    cudaEventDestroy(start_query);
    cudaEventDestroy(stop_query);   
}
}
