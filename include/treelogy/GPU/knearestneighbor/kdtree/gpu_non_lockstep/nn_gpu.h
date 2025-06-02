/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __NN_GPU_H
#define __NN_GPU_H

#include "nn.h"
#include "gpu_tree.h"

#define DOWN 0
#define UP 1

#define WARP_SIZE (32)

#define NUM_THREAD_BLOCKS (1024)
#define NUM_THREADS_PER_BLOCK (WARP_SIZE*6)
#define NUM_THREADS_PER_GRID (NUM_THREAD_BLOCKS * NUM_THREADS_PER_BLOCK)

#define NUM_WARPS_PER_BLOCK (NUM_THREADS_PER_BLOCK / WARP_SIZE)
#define NUM_WARPS_PER_GRID (NUM_THREADS_PER_GRID / WARP_SIZE)

#define WARP_IDX (threadIdx.x / WARP_SIZE)
#define GLOBAL_WARP_IDX (WARP_IDX + (blockIdx.x*NUM_WARPS_PER_BLOCK))
#define THREAD_IDX_IN_WARP (threadIdx.x - (WARP_SIZE * WARP_IDX))
#define IS_FIRST_THREAD_IN_WARP (threadIdx.x == (WARP_IDX * WARP_SIZE))

// top levelfunctions
__global__ void init_kernel(void);
__global__ void nearest_neighbor_search(gpu_tree gpu_tree, int nsearchpoints, node * search_points, float * nearest_distance, int * nearest_point_index, int K
#ifdef TRACK_TRAVERSALS
																				,int *d_nodes_accessed
#endif
);

gpu_tree * gpu_transform_tree(node *root);
gpu_tree * gpu_copy_to_dev(gpu_tree *h_tree);
void gpu_copy_tree_to_host(gpu_tree *d_tree, gpu_tree *h_tree);
void gpu_free_tree_dev(gpu_tree *d_tree);
void gpu_free_tree_host(gpu_tree *h_tree);
void gpu_print_tree_host(gpu_tree *h_tree);

#endif
