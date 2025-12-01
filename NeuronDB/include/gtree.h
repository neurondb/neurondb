/*-------------------------------------------------------------------------
 *
 * gtree.h
 *    Simple growable tree container used by ML components.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_GTREE_H
#define NEURONDB_GTREE_H

#include "postgres.h"

#define GTREE_MAX_NODES (1 << 24)
#define GTREE_MAX_DEPTH 64

typedef struct GTreeNode
{
	int feature_idx;
	double threshold;
	int left;
	int right;
	int is_leaf;
	double value;
} GTreeNode;

typedef struct GTree
{
	MemoryContext ctx;
	GTreeNode *nodes;
	int count;
	int capacity;
	int root;
	int max_depth;
} GTree;

extern GTree *gtree_create(const char *name, Size initial_cap);
extern void gtree_free(GTree *t);
extern void gtree_reset(GTree *t);
extern int gtree_add_leaf(GTree *t, double value);
extern int gtree_add_split(GTree *t, int feature_idx, double threshold);
extern void gtree_set_child(GTree *t, int parent, int child, bool is_left);
extern void gtree_set_children(GTree *t, int parent, int left, int right);
extern void gtree_set_root(GTree *t, int node_idx);
extern void gtree_validate(const GTree *t);
extern const GTreeNode *gtree_nodes(const GTree *t);
extern int gtree_count(const GTree *t);

#endif /* NEURONDB_GTREE_H */
