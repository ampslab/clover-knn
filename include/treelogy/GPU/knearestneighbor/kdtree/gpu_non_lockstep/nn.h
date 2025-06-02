/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef NN_H_
#define NN_H_

#ifndef DIM
#define DIM 3
#endif

#define KNEIGHBORS 30

typedef struct node_ {
	int axis;
	int point_index;
	float point[DIM];

	struct node_ *left;
	struct node_ *right;

} node;

#define distance_axis(a, b, axis) ((a[axis]-b[axis])*(a[axis]-b[axis]))

node *construct_tree(node *points, int start_idx, int end_idx, int depth);
void sort_search_points(float *points, int start_idx, int end_idx, int depth);
void print_tree(node * root);

#endif /* NN_H_ */
