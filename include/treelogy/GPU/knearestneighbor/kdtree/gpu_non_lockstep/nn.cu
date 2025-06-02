/* -*- mode: c -*- */
/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <unistd.h>
#include <time.h>

#include "../../../common/util_common.h"
#include "nn.h"
#include "nn_gpu.h"

node *points;
node *search_points;
//float *search_points;
unsigned int npoints;
unsigned int nsearchpoints;

float* nearest_distance;
unsigned int* nearest_point_index;

float* nearest_distance_brute;
unsigned int* nearest_point_index_brute;

node * tree;

int K;

int sort_flag = 0;
int verbose_flag = 0;
int check_flag = 0;
int ratio_flag = 0;
int warp_flag = 0;
/*
TIME_INIT(construct_tree);
TIME_INIT(gpu_build_tree);
TIME_INIT(init_kernel);
TIME_INIT(gpu_copy_tree_to);
TIME_INIT(kernel);
TIME_INIT(sort);
TIME_INIT(gpu_copy_tree_from);
TIME_INIT(traversal_time);
TIME_INIT(runtime);

int main(int argc, char **argv) {

	int i, j, k;
	float min = FLT_MAX;
	float max = FLT_MIN;
	int c;
	char *input_file;

	if(argc < 4) {
		fprintf(stderr, "usage: nn [-c] [-v] [-s] <k> <input_file> <npoints> [<nsearchpoints>]\n");
		exit(1);
	}

	while((c = getopt(argc, argv, "cvt:srw")) != -1) {
		switch(c) {
		case 'c':
			check_flag = 1;
			break;

		case 'v':
			verbose_flag = 1;
			break;

		case 's':
			sort_flag = 1;
			break;

        case 'r':
            ratio_flag = 1;
            break;

        case 'w':
            warp_flag = 1;
            break;

		case '?':
			fprintf(stderr, "Error: unknown option.\n");
			exit(1);
			break;

		default:
			abort();
		}
	}
	
	for(i = optind; i < argc; i++) {
		switch(i - optind) {
		case 0:
			K = atoi(argv[i]);
			if(K <= 0) {
				fprintf(stderr, "Invalid number of neighbors.\n");
				exit(1);
			}
			break;

		case 1:
			input_file = argv[i];
			break;

		case 2:
				npoints = atoi(argv[i]);
				nsearchpoints = npoints;
				if(npoints <= 0) {
					fprintf(stderr, "Not enough points.\n");
					exit(1);
				}
				break;

		case 3:
			nsearchpoints = atoi(argv[i]);
			if(nsearchpoints <= 0) {
				fprintf(stderr, "Not enough search points.");
				exit(1);
			}
			break;
		}
	}
	
	printf("configuration: sort_flag=%d, check_flag = %d, verbose_flag=%d, K=%d, input_file=%s, npoints=%d, nsearchpoints=%d\n", sort_flag, check_flag, verbose_flag, K, input_file, npoints, nsearchpoints);
	
	SAFE_CALLOC(points, npoints, sizeof(node));
	//SAFE_MALLOC(search_points, sizeof(float)*nsearchpoints*DIM);
	SAFE_CALLOC(search_points, nsearchpoints, sizeof(node));
	SAFE_MALLOC(nearest_distance, sizeof(float)*nsearchpoints*K);
	SAFE_MALLOC(nearest_point_index, sizeof(unsigned int)*nsearchpoints*K);

	if(check_flag) {
		SAFE_MALLOC(nearest_distance_brute, sizeof(float)*nsearchpoints*K);
		SAFE_MALLOC(nearest_point_index_brute, sizeof(unsigned int)*nsearchpoints*K);
	}

	if(strcmp(input_file, "random") != 0) {
		FILE * in = fopen(input_file, "r");
		if(in == NULL) {
			fprintf(stderr, "Could not open %s\n", input_file);
			exit(1);
		}
		
		int junk;
		float data;

		for(i = 0; i < npoints; i++) {
			points[i].point_index = i;
			if(fscanf(in, "%d", &junk) != 1) {
				fprintf(stderr, "Input file not large enough.\n");
				exit(1);
			} 
			for(j = 0; j < DIM; j++) {
				if(fscanf(in, "%f", &data) != 1) {
					fprintf(stderr, "Input file not large enough.\n");
					exit(1);
				}
				points[i].point[j] = data;
			}
		}
	
		for(i = 0; i < nsearchpoints; i++) {
			if(fscanf(in, "%d", &junk) != 1) {
				fprintf(stderr, "Input file not large enough.\n");
				exit(1);
			}
			for(j = 0; j < DIM; j++) {
				if(fscanf(in, "%f", &data) != 1) {
					fprintf(stderr, "Input file not large enough.\n");
					exit(1);
				}
				search_points[i].point[j] = data;
				//search_points[i*DIM + j] = data;
			}
		}
		
		fclose(in);
		
	} else {
		for(i = 0; i < npoints; i++) {
			points[i].point_index = i;			
			for(j = 0; j < DIM; j++) {
				points[i].point[j] = 1.0 + (float)rand() / RAND_MAX;			
			}
		}
	
		for(i = 0; i < nsearchpoints; i++) {
			for(j = 0; j < DIM; j++) {
				//search_points[i*DIM + j] = 1.0 + (float)rand() / RAND_MAX;			
				search_points[i].point[j] = 1.0 + (float)rand() / RAND_MAX;
			}
		}
	}

	TIME_START(runtime);
	TIME_START(construct_tree);

	if(sort_flag) {
		TIME_START(sort);
		construct_tree(search_points, 0, nsearchpoints - 1, 0);
		TIME_END(sort);
	}
	
	tree = construct_tree(points, 0, npoints - 1, 0);
	
	TIME_END(construct_tree);

	TIME_START(init_kernel);
	init_kernel<<<1, 1>>>();
	TIME_END(init_kernel);

	TIME_START(traversal_time);

	TIME_START(gpu_build_tree);
	// *** GPU Kerel Call *** //
	gpu_tree * h_tree = gpu_transform_tree(tree);
	TIME_END(gpu_build_tree);
	
	TIME_START(gpu_copy_tree_to);
	gpu_tree * d_tree = gpu_copy_to_dev(h_tree);

	// Allocate variables to store results of each thread
	node * d_search_points;
	float * d_nearest_distance;
	int * d_nearest_point_index;

	#ifdef TRACK_TRAVERSALS	
	int *h_nodes_accessed;
	int *d_nodes_accessed;
	SAFE_CALLOC(h_nodes_accessed, nsearchpoints, sizeof(int));
	CUDA_SAFE_CALL(cudaMalloc(&d_nodes_accessed, sizeof(int)*nsearchpoints));
	CUDA_SAFE_CALL(cudaMemcpy(d_nodes_accessed, h_nodes_accessed, sizeof(int)*nsearchpoints, cudaMemcpyHostToDevice));
	#endif

	// Read from but not written to
	CUDA_SAFE_CALL(cudaMalloc(&d_search_points, sizeof(node)*nsearchpoints));
	CUDA_SAFE_CALL(cudaMemcpy(d_search_points, search_points, sizeof(node)*nsearchpoints, cudaMemcpyHostToDevice));

	// Immediatly written to at kernel
	CUDA_SAFE_CALL(cudaMalloc(&d_nearest_distance, sizeof(float)*nsearchpoints*K));
	CUDA_SAFE_CALL(cudaMalloc(&d_nearest_point_index, sizeof(int)*nsearchpoints*K));
	
	TIME_END(gpu_copy_tree_to);
	TIME_START(kernel);

	//gpu_print_tree_host(h_tree);
	dim3 grid(NUM_THREAD_BLOCKS, 1, 1);
	dim3 block(NUM_THREADS_PER_BLOCK, 1, 1);
	unsigned int smem_bytes = DIM*NUM_THREADS_PER_BLOCK*sizeof(float) + K*NUM_THREADS_PER_BLOCK*sizeof(int) + K*NUM_THREADS_PER_BLOCK*sizeof(float);
	nearest_neighbor_search<<<grid, block, smem_bytes>>>(*d_tree, nsearchpoints, d_search_points, d_nearest_distance, d_nearest_point_index, K
  #ifdef TRACK_TRAVERSALS
																											 , d_nodes_accessed																											 
#endif
																											 );
																											 
	cudaError_t err = cudaThreadSynchronize();
	if(err != cudaSuccess) {
		fprintf(stderr,"Kernel failed with error: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	
	TIME_END(kernel);
	TIME_START(gpu_copy_tree_from);

	// Copy results back
	CUDA_SAFE_CALL(cudaMemcpy(nearest_point_index, d_nearest_point_index, sizeof(int)*nsearchpoints*K, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(nearest_distance, d_nearest_distance, sizeof(int)*nsearchpoints*K, cudaMemcpyDeviceToHost));
	#ifdef TRACK_TRAVERSALS
	CUDA_SAFE_CALL(cudaMemcpy(h_nodes_accessed, d_nodes_accessed, sizeof(int)*nsearchpoints, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_nodes_accessed));
	#endif
	CUDA_SAFE_CALL(cudaFree(d_nearest_point_index));
	CUDA_SAFE_CALL(cudaFree(d_nearest_distance));
	CUDA_SAFE_CALL(cudaFree(d_search_points));

	TIME_END(gpu_copy_tree_from);
	TIME_END(traversal_time);
	TIME_END(runtime);

		if(verbose_flag) {
		for(j = 0; j < nsearchpoints; j++) {
			printf("\n%d: ", j);
			for(i = 0; i < K; i++) {*/
/*				if(i == K-1)
					printf("%d", nearest_point_index[j*K + i]);
				else
					printf("%d,", nearest_point_index[j*K + i]);*/
			    /*if(i == K-1)
                    printf("%d (%1.3f)", nearest_point_index[j*K + i], nearest_distance[j*K + i]);
                else
                    printf("%d (%1.3f),", nearest_point_index[j*K + i], nearest_distance[j*K + i]);
            }
		}
		printf("\n");
	}*/

	/*
	#ifdef TRACK_TRAVERSALS
	unsigned long long nodes_accessed_sum = 0;
    int maximum = 0, all = 0;
    unsigned long long maximum_sum = 0, all_sum = 0;

	for(i = 0; i < nsearchpoints + (nsearchpoints % 32); i+=32) {
		int na = h_nodes_accessed[i];
		nodes_accessed_sum += na;
        maximum = na;
        all = na;

		for(j = i + 1; j < i + 32 && j < nsearchpoints; j++) {
			if(h_nodes_accessed[j] > na)
				na = h_nodes_accessed[j];
			nodes_accessed_sum += h_nodes_accessed[j];
            
            if (warp_flag) {
                if(h_nodes_accessed[j] > maximum)
                    maximum = h_nodes_accessed[j];
                all += h_nodes_accessed[j];
            }
	    }
        if (warp_flag) {
            printf("%d\n", maximum);
            maximum_sum += maximum;
            all_sum += all;
        }
//			printf("nodes warp %d: %d\n", i/32, na);
	}
	printf("num nodes: %f\n", (float)nodes_accessed_sum);
    printf("avg nodes: %f\n", (float)nodes_accessed_sum/nsearchpoints);
    #endif
	
	TIME_ELAPSED_PRINT(construct_tree, stdout);
	TIME_ELAPSED_PRINT(gpu_build_tree, stdout);
	TIME_ELAPSED_PRINT(init_kernel, stdout);
	TIME_ELAPSED_PRINT(gpu_copy_tree_to, stdout);
	TIME_ELAPSED_PRINT(kernel, stdout);
	TIME_ELAPSED_PRINT(sort, stdout);
	TIME_ELAPSED_PRINT(gpu_copy_tree_from, stdout);
	TIME_ELAPSED_PRINT(traversal_time, stdout);
	TIME_ELAPSED_PRINT(runtime, stdout);

	gpu_free_tree_dev(d_tree);
	gpu_free_tree_host(h_tree);
	
	return 0;
}
*/