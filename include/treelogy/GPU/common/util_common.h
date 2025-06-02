/************************************************************************************************
 #* Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 #* Purdue University. All Rights Reserved. See Copyright.txt
************************************************************************************************/
#ifndef __UTIL_COMMON_H_
#define __UTIL_COMMON_H_

// Pointer Checking
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <memory>
#include <vector>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <ctype.h>
#include <assert.h>
#include <string>
#include <cstring>
#include <iostream>
#include <queue>
#include <cmath>
#include <fstream>
#include <sstream>
#include <limits>
using namespace std;

#define SAFE_MALLOC(p, n) { p = (typeof(p))malloc(n); if(!p) { fprintf(stderr, "Error: malloc failed to allcate %d bytes: %s in %s at line %d!\n", n, #p, __FILE__, __LINE__); exit(-1); } }

#define SAFE_CALLOC(p, n, elemsize) { p = (typeof(p))calloc(n, elemsize); if(!p) { fprintf(stderr, "Error: calloc failed to allcate %d bytes: %s in %s at line %d!\n", n*elemsize, #p, __FILE__, __LINE__); exit(-1); } }

#define CHECK_PTR(p) { if(!p) { fprintf(stderr, "Error: NULL pointer: %s in %s at line %d!\n", #p, __FILE__, __LINE__); exit(-1); } }

// Code Timing

#include <sys/time.h>

#define TIME_INIT(timeName) struct timeval timeName ## _start; \
        struct timeval timeName ## _end; \
	struct timeval timeName ## _temp;

#define TIME_START(timeName) gettimeofday(&(timeName ## _start), NULL);

#define TIME_END(timeName) gettimeofday(&(timeName ## _end), NULL); 

#define TIME_RESTART(timeName) timeName ## _temp.tv_sec = timeName ## _end.tv_sec - timeName ## _start.tv_sec; \
timeName ## _temp.tv_usec = timeName ## _end.tv_usec - timeName ## _start.tv_usec; \
TIME_START(timeName); \
timeName ## _start.tv_sec -= timeName ## _temp.tv_sec; \
timeName ## _start.tv_usec -= timeName ## _temp.tv_usec;


#define TIME_ELAPSED(timeName) float timeName ## _elapsed = (timeName ## _end.tv_sec - timeName ## _start.tv_sec); \
timeName ## _elapsed += ((timeName ## _end.tv_usec - timeName ## _start.tv_usec) / (1.0e6)); \
timeName ## _elapsed *= 1000; 

#define TIME_ELAPSED_PRINT(timeName, stream) TIME_ELAPSED(timeName) \
fprintf(stream, "@ %s: %2.0f ms\n", #timeName, timeName ## _elapsed); 

// CUDA Utils

#ifdef __CUDACC__
#include <cuda.h>

#define CUDA_SAFE_CALL(call) {																					\
	 cudaError_t err = call;																							\
	 if(err != cudaSuccess) {																							\
	  fprintf(stderr, "CUDA Error: %s: %s (%d) in %s at line %d!\n", #call, cudaGetErrorString(err), err, __FILE__, __LINE__);  \
		exit(-1);																														\
	 }																																		\
}

#endif

#endif
