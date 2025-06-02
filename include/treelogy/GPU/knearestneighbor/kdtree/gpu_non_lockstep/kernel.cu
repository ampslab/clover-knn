/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <float.h>
#include "nn.h"
#include "nn_gpu.h"
#include <stdio.h>

__global__ void init_kernel(void) {

}

__global__ void
nearest_neighbor_search (gpu_tree gpu_tree, int nsearchpoints, node *d_search_points, float *d_nearest_distance,
												 int *d_nearest_point_index, int K
												 #ifdef TRACK_TRAVERSALS
												 , int *d_nodes_accessed
												 #endif
)
{
	extern __shared__ int smem[];
  float *search_points;
	float *nearest_distance;
	int *nearest_point_index;

  int pidx;
  int i, j;

	// get cached to registers
  gpu_tree_node_0 cur_node0;
  gpu_tree_node_1 cur_node1;

  int cur_node_index;
  int sp;

  int current_split;
  float axis_dist;
  float dist;
	float t;

  // Get the position of the 1st item
	int *stk_node;
	int stk_node_top;
	float *stk_axis_dist;
	float stk_axis_dist_top;

	float tmpdist;
	int tmpidx;
	int n;

	#ifdef TRACK_TRAVERSALS
	int nodes_accessed;
	#endif

#include "nn_kernel_macros.inc"

	// setup shared memory pointers:
	// It seems like these all need to be alligned to 32? byte boundry?
	unsigned int off=0;
	search_points = (float*)(&smem[off]);
	
	off = NUM_THREADS_PER_BLOCK*DIM;
	nearest_point_index = (int*)(&smem[off]);

	off = off + NUM_THREADS_PER_BLOCK*K;
	nearest_distance = (float*)(&smem[off]);	
	
  for (pidx = blockIdx.x * blockDim.x + threadIdx.x; pidx < nsearchpoints;
       pidx += blockDim.x * gridDim.x)
    {

      for (j = 0; j < DIM; j++)
				search_points[NUM_THREADS_PER_BLOCK*j+threadIdx.x] = d_search_points[pidx].point[j];

			for(i = 0; i < K; i++) {
				nearest_point_index[i*NUM_THREADS_PER_BLOCK+threadIdx.x] = -1;
				nearest_distance[i*NUM_THREADS_PER_BLOCK+threadIdx.x] = FLT_MAX;
			}

//            printf("i*NUM_THREADS_PER_BLOCK+threadIdx.x = %d\n", threadIdx.x);

			#ifdef TRACK_TRAVERSALS
			nodes_accessed = 0;
			#endif

      // run this for some number of iterations until done...
      STACK_INIT();
			while(sp >= 0) {
				// get top of stack
				cur_node_index = STACK_NODE;
				axis_dist = STACK_AXIS_DIST;

				if(axis_dist > nearest_distance[threadIdx.x]) {
					STACK_POP();
					continue;
				}
				
				#ifdef TRACK_TRAVERSALS
				nodes_accessed++;
				#endif

				cur_node0 = gpu_tree.nodes0[cur_node_index];
				current_split = AXIS;
										
//				if(pidx == 0) {
//					printf("sp = %d, ZERO is visiting %d node, point_index = %d.\n",
//							sp, cur_node_index, POINT_INDEX);
//				}

				// Swap it if our point is closer
				//dist = distance (POINT, &search_points[threadIdx.x * DIM]);
				dist = 0.0;
				for (i = 0; i < DIM; i++) {
					t = (gpu_tree.nodes2[cur_node_index].point[i] - search_points[NUM_THREADS_PER_BLOCK*i + threadIdx.x]);
					dist +=  t*t;
				}

				// update closest point:
				if(dist < nearest_distance[threadIdx.x]) {
					nearest_distance[threadIdx.x] = dist;
					nearest_point_index[threadIdx.x] = POINT_INDEX;
							
					// push the value back to maintain sorted order
					for(n = 0; n < K-1 && nearest_distance[n*NUM_THREADS_PER_BLOCK+threadIdx.x] < nearest_distance[((n+1)*NUM_THREADS_PER_BLOCK)+threadIdx.x]; n++) {
						tmpdist = nearest_distance[(n+1)*NUM_THREADS_PER_BLOCK+threadIdx.x];
						tmpidx = nearest_point_index[(n+1)*NUM_THREADS_PER_BLOCK+threadIdx.x];
						nearest_distance[(n+1)*NUM_THREADS_PER_BLOCK+threadIdx.x] = nearest_distance[n*NUM_THREADS_PER_BLOCK+threadIdx.x];
						nearest_point_index[(n+1)*NUM_THREADS_PER_BLOCK+threadIdx.x] = nearest_point_index[n*NUM_THREADS_PER_BLOCK+threadIdx.x];
						nearest_distance[n*NUM_THREADS_PER_BLOCK+threadIdx.x] = tmpdist;
						nearest_point_index[n*NUM_THREADS_PER_BLOCK+threadIdx.x] = tmpidx;
					}
				}							

				cur_node1 = gpu_tree.nodes1[cur_node_index];
				if (LEFT != NULL_NODE && search_points[current_split*NUM_THREADS_PER_BLOCK + threadIdx.x] <= POINT_SPLIT (current_split)) {
					axis_dist =	(search_points[current_split*NUM_THREADS_PER_BLOCK + threadIdx.x] - POINT_SPLIT (current_split));
					if(RIGHT != NULL_NODE) {
						STACK_NODE = RIGHT;
						STACK_AXIS_DIST = axis_dist * axis_dist;						
						STACK_PUSH();
					}
					
					STACK_NODE = LEFT;
					STACK_AXIS_DIST = FLT_MIN;					
				} else if (RIGHT != NULL_NODE) {
					axis_dist =	(search_points[current_split*NUM_THREADS_PER_BLOCK + threadIdx.x] - POINT_SPLIT (current_split));
					
					if(LEFT != NULL_NODE) {											
						STACK_NODE = LEFT;
						STACK_AXIS_DIST = axis_dist * axis_dist;
						STACK_PUSH();
					}

					STACK_NODE = RIGHT;
					STACK_AXIS_DIST = FLT_MIN;
				} else {
					STACK_POP();
				}
			}
			
      // Save to global memory
			for(i = 0; i < K; i++) {
				d_nearest_point_index[K*pidx+i] = nearest_point_index[i*NUM_THREADS_PER_BLOCK+threadIdx.x];
				d_nearest_distance[K*pidx+i] = nearest_distance[i*NUM_THREADS_PER_BLOCK+threadIdx.x];
			}
			#ifdef TRACK_TRAVERSALS
			d_nodes_accessed[pidx] = nodes_accessed;
			#endif
    }
}

__device__ float
distance (float *a, float *b)
{
  int i;
  float d = 0;
  // returns distance squared
#pragma unroll
  for (i = 0; i < DIM; i++)
    {
      d += distance_axis (a, b, i);
    }

  return d;
}
