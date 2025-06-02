#************************************************************************************************
 #* Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 #* Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
#CUDA_20 = 1
#DEBUG = 1

ifeq ($(TRACK_TRAVERSALS),1)
	COMMON_COMPILE_FLAGS += -DTRACK_TRAVERSALS=$(TRACK_TRAVERSALS)
endif

ifeq ($(TRACK_TRAVERSALS_EXT),1)
    COMMON_COMPILE_FLAGS += -DTRACK_TRAVERSALS_EXT=$(TRACK_TRAVERSALS_EXT)
endif

ifdef RADIUS
ifneq ($(RADIUS),0)
        COMMON_COMPILE_FLAGS += -DRADIUS=$(RADIUS)
endif
endif

ifneq ($(DEBUG),1)
	COMMON_COMPILE_FLAGS += -O2
else
	COMMON_COMPILE_FLAGS += -g
endif

ifdef SPLICE_DEPTH
ifneq ($(SPLICE_DEPTH),100000)
	COMMON_COMPILE_FLAGS += -DSPLICE_DEPTH=$(SPLICE_DEPTH)
endif
endif

ifdef DIM
	COMMON_COMPILE_FLAGS += -DDIM=$(DIM)
endif

ifdef NUM_OF_BLOCKS
	COMMON_COMPILE_FLAGS += -DNUM_OF_BLOCKS=$(NUM_OF_BLOCKS)
endif

ifdef NUM_OF_WARPS_PER_BLOCK
	COMMON_COMPILE_FLAGS += -DNUM_OF_WARPS_PER_BLOCK=$(NUM_OF_WARPS_PER_BLOCK)
endif

COMMON_LINK_FLAGS += -lm -lpthread

CUDA_PATH = /usr/local
NVCC = nvcc

ifdef CUDA_20
	NVCC_OPTIONS += -arch sm_20 --ptxas-options=-v $(COMMON_COMPILE_FLAGS)
else
	NVCC_OPTIONS += -arch sm_35 --ptxas-options=-v $(COMMON_COMPILE_FLAGS)
endif

NVCC_LINK_OPTIONS = $(COMMON_LINK_FLAGS)

