#*************************************************************************************************
# * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
# * Purdue University. All Rights Reserved. See Copyright.txt
#************************************************************************************************/
include ../../../common/cuda.mk

all: nn

nn: nn.o kdtree.o dev_memory.o gpu_tree.o kernel.o
	$(NVCC) $(NVCC_LINK_OPTIONS) -o nn nn.o kdtree.o gpu_tree.o dev_memory.o kernel.o

nn.o: nn.cu nn.h
	$(NVCC) $(NVCC_OPTIONS) -c nn.cu

kdtree.o: kdtree.cu nn.h
	$(NVCC) $(NVCC_OPTIONS) -c kdtree.cu

gpu_tree.o: gpu_tree.cu gpu_tree.h nn.h
	$(NVCC) $(NVCC_OPTIONS) -c gpu_tree.cu

dev_memory.o: dev_memory.cu nn.h
	$(NVCC) $(NVCC_OPTIONS) -c dev_memory.cu

kernel.o: kernel.cu nn.h nn_gpu.h nn_kernel_macros.inc
	$(NVCC) $(NVCC_OPTIONS) -c kernel.cu

clean:
	rm -f *.o nn *~


