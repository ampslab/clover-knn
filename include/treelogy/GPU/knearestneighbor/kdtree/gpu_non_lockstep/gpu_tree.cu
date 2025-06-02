/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "../../../common/util_common.h"
#include "nn_gpu.h"
#include "gpu_tree.h"

static void gpu_alloc_tree_host(gpu_tree * h_tree);
static void gpu_init_tree_properties(gpu_tree *h_tree, node *root, int depth);
static int gpu_build_tree(node *root, gpu_tree *h_tree, int *index, int depth, int parent_index);

gpu_tree * gpu_transform_tree(node *root) {

	CHECK_PTR(root);
	
	gpu_tree *tree;
	SAFE_MALLOC(tree, sizeof(gpu_tree));

	tree->nnodes = 0;
	tree->depth = 0;

	gpu_init_tree_properties(tree, root, 0);
	gpu_alloc_tree_host(tree);
	
	int index = 0;
	gpu_build_tree(root, tree, &index, 0, NULL_NODE);

	return tree;
}

void gpu_free_tree_host(gpu_tree *h_tree) {
	CHECK_PTR(h_tree);
	free(h_tree->nodes0);
	free(h_tree->nodes1);
	free(h_tree->nodes2);
}

static void gpu_alloc_tree_host(gpu_tree * h_tree) {
	SAFE_MALLOC(h_tree->nodes0, sizeof(gpu_tree_node_0)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes1, sizeof(gpu_tree_node_1)*h_tree->nnodes);
	SAFE_MALLOC(h_tree->nodes2, sizeof(gpu_tree_node_2)*h_tree->nnodes);
}

static void gpu_init_tree_properties(gpu_tree * h_tree, node * root, int depth) {

	h_tree->nnodes++;

	if(depth > h_tree->depth) 
		h_tree->depth = depth;

	if(root->left != NULL)
		gpu_init_tree_properties(h_tree, root->left, depth + 1);

	if(root->right != NULL)
		gpu_init_tree_properties(h_tree, root->right, depth + 1);
}

static int gpu_build_tree(node *root, gpu_tree *h_tree, int *index, int depth, int parent_index) {
	// add node to tree
	gpu_tree_node_0 node0;
	gpu_tree_node_1 node1;
	gpu_tree_node_2 node2;
	int i;
	int my_index = *index; *index += 1;

	node0.items.axis = root->axis;
	node0.items.point_index = root->point_index;
	for(i = 0; i < DIM; i++) {
		node2.point[i] = root->point[i];
	}

	//node1.parent = parent_index;
	if(root->left != NULL)
		node1.left = gpu_build_tree(root->left, h_tree, index, depth + 1, my_index);
	else
		node1.left = NULL_NODE;
	
	if(root->right != NULL) {
		node1.right = gpu_build_tree(root->right, h_tree, index, depth + 1, my_index);
	} else {
		node1.right = NULL_NODE;
	}
	
	h_tree->nodes0[my_index] =  node0;
	h_tree->nodes1[my_index] =  node1;
	h_tree->nodes2[my_index] =  node2;
	return my_index;
}

void gpu_print_tree_host(gpu_tree *h_tree) {
	
}

