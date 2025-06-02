/*************************************************************************************************
 * Copyright (C) 2017, Nikhil Hegde, Jianqiao Liu, Kirshanthan Sundararajah, Milind Kulkarni, and
 * Purdue University. All Rights Reserved. See Copyright.txt
*************************************************************************************************/
#ifndef __BLOCK_H
#define __BLOCK_H

#include "kd_cell.h"
#include "block_size.h"


#define NQ_SIZE (BLOCK_SIZE+1)

typedef struct _node_queue_entry {
	kd_cell * node;
	int index;
	int parent_index;
	int depth;
} node_queue_entry;

typedef struct _node_queue {
	
	int f;
	int b;
	node_queue_entry entries[NQ_SIZE];

} node_queue;

#endif
