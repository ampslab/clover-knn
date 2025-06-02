/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "../../../common/util_common.h"
#include "nn.h"

static int sort_split;

static int cmpfn_float(const void *a, const void *b);
static int cmpfn_node(const void *a, const void *b);

node * construct_tree(node *points, int start_idx, int end_idx, int depth) {
	int i;

	if (start_idx > end_idx)
		return NULL;

	if (start_idx == end_idx) {

		node * cell = &points[start_idx];
		cell->axis = DIM; // leaf nodes get axis = DIM
		cell->point_index = start_idx;
		cell->left = NULL;
		cell->right = NULL;
		return cell;

	} else {

		// this is not a single point:
		int split = depth % DIM;

		// find the median and partition points into left and right sets:
		int median_index;
		int j;

		// partition to get some sort of split around the median
		sort_split = split;
		qsort(&points[start_idx], end_idx - start_idx + 1, sizeof(node), cmpfn_node);
		median_index = (start_idx + end_idx) / 2;
		
		node * cell = &points[median_index];
		cell->axis = split;
		cell->point_index = median_index;
		cell->left = construct_tree(points, start_idx, median_index - 1, depth + 1);
		cell->right = construct_tree(points, median_index + 1, end_idx, depth + 1);
		return cell;
	}

}

void sort_search_points(float *points, int start_idx, int end_idx, int depth) {
	int i;

	if (start_idx >= end_idx)
		return;

		// this is not a single point:
		int split = depth % DIM;

		// find the median and partition points into left and right sets:
		int median_index;

		// partition to get some sort of split around the median
		sort_split = split;
		qsort(&points[start_idx], end_idx - start_idx + 1, sizeof(float)*DIM, cmpfn_float);
		median_index = (start_idx + end_idx) / 2;
		
		sort_search_points(points, start_idx, median_index - 1, depth + 1);
		sort_search_points(points, median_index + 1, end_idx, depth + 1);
}

static int cmpfn_float(const void *a, const void *b) {
	
	float *fa = (float*)a;
	float *fb = (float*)b;

	if(fa[sort_split] < fb[sort_split]) {
		return -1;
	} else if(fa[sort_split] > fb[sort_split]) {
		return 1;
	} else {
		return 0;
	}
}

static int cmpfn_node(const void *a, const void *b) {
	
	node *fa = (node*)a;
	node *fb = (node*)b;

	if(fa->point[sort_split] < fb->point[sort_split]) {
		return -1;
	} else if(fa->point[sort_split] > fb->point[sort_split]) {
		return 1;
	} else {
		return 0;
	}
}


void print_tree(node * root) {
	int i;

	printf("dom: ( ");
	for (i = 0; i < DIM; i++)
		printf("%f ", root->point[i]);
	printf(")");

	if(root->axis == DIM)
		printf(" (leaf)\n");
	else
		printf(" ax = %d\n", root->axis);

	if (root->left != NULL)
		print_tree(root->left);

	if (root->right != NULL)
		print_tree(root->right);
}
