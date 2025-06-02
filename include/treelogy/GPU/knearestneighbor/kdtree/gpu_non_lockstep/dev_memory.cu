/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <cuda.h>
#include "../../../common/util_common.h"
#include "nn_gpu.h"

static gpu_tree* gpu_alloc_tree_dev(gpu_tree *h_tree);

gpu_tree * gpu_copy_to_dev(gpu_tree *h_tree) {

	gpu_tree * d_tree = gpu_alloc_tree_dev(h_tree);
	
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes0, h_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes1, h_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->nnodes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_tree->nodes2, h_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->nnodes, cudaMemcpyHostToDevice));
	
	return d_tree;
}

void gpu_copy_tree_to_host(gpu_tree *d_tree, gpu_tree *h_tree) {
	// Nothing in the tree is modified to there is nothing to do here!
}

void gpu_free_tree_dev(gpu_tree *d_tree) {
	CHECK_PTR(d_tree);
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes0));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes1));
	CUDA_SAFE_CALL(cudaFree(d_tree->nodes2));
	CUDA_SAFE_CALL(cudaFree(d_tree->stk_node));
	CUDA_SAFE_CALL(cudaFree(d_tree->stk_axis_dist));
}

static gpu_tree* gpu_alloc_tree_dev(gpu_tree *h_tree) {
	
	CHECK_PTR(h_tree);
	
	gpu_tree * d_tree;
	SAFE_MALLOC(d_tree, sizeof(gpu_tree));
	
	// copy tree value params:
	d_tree->nnodes = h_tree->nnodes;
	d_tree->depth = h_tree->depth;
	
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->nnodes));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->stk_node, sizeof(int)*h_tree->depth*2*NUM_THREADS_PER_BLOCK*NUM_THREAD_BLOCKS));
	CUDA_SAFE_CALL(cudaMalloc(&d_tree->stk_axis_dist, sizeof(float)*h_tree->depth*2*NUM_THREADS_PER_BLOCK*NUM_THREAD_BLOCKS));

	return d_tree;
}
