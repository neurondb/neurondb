#include "postgres.h"
#include "utils/memutils.h"
#include "gtree.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * Expand the array of nodes for the given GTree to ensure
 * it can hold at least (count + need_extra) nodes.
 */
static void
gtree_grow(GTree *t, int need_extra)
{
	int32		need;
	int32		newcap;

	if (!t)
		ereport(ERROR, (errmsg("gtree_grow: NULL tree")));
	if (need_extra <= 0)
		ereport(ERROR,
				(errmsg("gtree_grow: need_extra must be positive")));

	if (t->count < 0 || t->count > GTREE_MAX_NODES)
		ereport(ERROR, (errmsg("gtree_grow: tree node count corrupt")));
	if (t->count > GTREE_MAX_NODES - need_extra)
		ereport(ERROR, (errmsg("gtree_grow: node limit reached")));

	need = t->count + need_extra;

	newcap = (t->capacity > 0) ? t->capacity : 32;
	while (newcap < need)
	{
		if (newcap > GTREE_MAX_NODES / 2)
			newcap = GTREE_MAX_NODES;
		else
			newcap *= 2;
	}

	if (newcap < t->count || newcap > GTREE_MAX_NODES)
		ereport(ERROR, (errmsg("gtree_grow: node cap overflow")));

	t->nodes = (GTreeNode *) repalloc(
									  t->nodes, (Size) newcap * sizeof(GTreeNode));
	if (!t->nodes)
		ereport(ERROR, (errmsg("gtree_grow: out of memory")));

	t->capacity = newcap;
}

/*
 * Allocate and initialize a new GTree object with requested initial capacity.
 */
GTree *
gtree_create(const char *name, Size initial_cap)
{
	MemoryContext parent = CurrentMemoryContext;
	MemoryContext ctx;
	GTree	   *t;

	ctx = AllocSetContextCreate(parent, "gtree", ALLOCSET_DEFAULT_SIZES);
	if (name != NULL)
	{
		MemoryContextSetIdentifier(ctx, MemoryContextStrdup(ctx, name));
	}

	MemoryContextSwitchTo(ctx);

	t = (GTree *) palloc0(sizeof(GTree));
	if (!t)
		ereport(ERROR, (errmsg("gtree_create: out of memory")));

	t->ctx = ctx;
	t->nodes = NULL;
	t->count = 0;
	t->capacity = 0;
	t->root = -1;
	t->max_depth = 0;

	if (initial_cap > 0)
	{
		if (initial_cap > (Size) GTREE_MAX_NODES)
			ereport(ERROR,
					(errmsg("gtree: initial cap too large")));
		t->nodes =
			(GTreeNode *) palloc0(initial_cap * sizeof(GTreeNode));
		if (!t->nodes)
			ereport(ERROR,
					(errmsg("gtree_create: node array OOM")));
		t->capacity = (int32) initial_cap;
	}

	MemoryContextSwitchTo(parent);
	return t;
}

/*
 * Free all memory associated with a GTree (including context).
 */
void
gtree_free(GTree *t)
{
	if (!t)
		return;
	if (t->ctx)
		MemoryContextDelete(t->ctx);
}

/*
 * Add a new leaf node to the tree with the given value.
 */
int
gtree_add_leaf(GTree *t, double value)
{
	int			idx;

	if (!t)
		ereport(ERROR, (errmsg("gtree_add_leaf: NULL tree")));

	if (t->count < 0 || t->count > t->capacity || t->capacity < 0)
		ereport(ERROR, (errmsg("gtree_add_leaf: invalid tree state")));

	if (t->count == t->capacity)
		gtree_grow(t, 1);

	idx = t->count++;
	t->nodes[idx].feature_idx = -1;
	t->nodes[idx].threshold = 0.0;
	t->nodes[idx].left = -1;
	t->nodes[idx].right = -1;
	t->nodes[idx].is_leaf = 1;
	t->nodes[idx].value = value;
	return idx;
}

/*
 * Add a new split (internal) node to the tree for the given feature and threshold.
 */
int
gtree_add_split(GTree *t, int feature_idx, double threshold)
{
	int			idx;

	if (!t)
		ereport(ERROR, (errmsg("gtree_add_split: NULL tree")));

	if (feature_idx < 0)
		ereport(ERROR, (errmsg("gtree: feature_idx must be >= 0")));
	if (t->count < 0 || t->count > t->capacity || t->capacity < 0)
		ereport(ERROR, (errmsg("gtree_add_split: invalid tree state")));

	if (t->count == t->capacity)
		gtree_grow(t, 1);

	idx = t->count++;
	t->nodes[idx].feature_idx = feature_idx;
	t->nodes[idx].threshold = threshold;
	t->nodes[idx].left = -1;
	t->nodes[idx].right = -1;
	t->nodes[idx].is_leaf = 0;
	t->nodes[idx].value = 0.0;
	return idx;
}

static int
gtree_depth_dfs(const GTree *t, int node, int depth, bool *seen)
{
	int			l;
	int			r;
	int			dl;
	int			dr;

	if (!t || !seen)
		ereport(ERROR, (errmsg("gtree_depth_dfs: NULL input")));

	if (node < 0 || node >= t->count)
		ereport(ERROR, (errmsg("gtree: node %d out of bounds", node)));

	if (depth > GTREE_MAX_DEPTH)
		ereport(ERROR, (errmsg("gtree: depth limit exceeded")));

	if (seen[node])
		ereport(ERROR, (errmsg("gtree: cycle detected at %d", node)));
	seen[node] = true;

	if (t->nodes[node].is_leaf)
		return depth;

	l = t->nodes[node].left;
	r = t->nodes[node].right;

	if (l < 0 || r < 0)
		ereport(ERROR,
				(errmsg("gtree: internal node %d missing child",
						node)));
	if (l >= t->count || r >= t->count)
		ereport(ERROR, (errmsg("gtree: child index OOB")));

	dl = gtree_depth_dfs(t, l, depth + 1, seen);
	dr = gtree_depth_dfs(t, r, depth + 1, seen);
	return (dl > dr) ? dl : dr;
}

/*
 * Set either left or right child for the parent node.
 */
void
gtree_set_child(GTree *t, int parent, int child, bool is_left)
{
	if (!t)
		ereport(ERROR, (errmsg("gtree: NULL tree")));
	if (parent < 0 || parent >= t->count)
		ereport(ERROR, (errmsg("gtree: parent OOB")));
	if (child < 0 || child >= t->count)
		ereport(ERROR, (errmsg("gtree: child OOB")));
	if (t->nodes[parent].is_leaf)
		ereport(ERROR, (errmsg("gtree: parent is leaf")));

	if (is_left)
		t->nodes[parent].left = child;
	else
		t->nodes[parent].right = child;
}

/*
 * Set both left and right children for this parent node.
 */
void
gtree_set_children(GTree *t, int parent, int left, int right)
{
	gtree_set_child(t, parent, left, true);
	gtree_set_child(t, parent, right, false);
}

/*
 * Set the root node index of the tree.
 */
void
gtree_set_root(GTree *t, int node_idx)
{
	if (!t)
		ereport(ERROR, (errmsg("gtree: NULL tree")));
	if (node_idx < 0 || node_idx >= t->count)
		ereport(ERROR, (errmsg("gtree: root index OOB")));
	t->root = node_idx;
}

/*
 * Validate the integrity of the tree structure and check for cycles,
 * missing root, and excessive depth.
 */
void
gtree_validate(const GTree *t)
{
	bool	   *seen;
	int			depth;

	if (!t)
		ereport(ERROR, (errmsg("gtree_validate: NULL tree")));
	if (t->count <= 0)
		ereport(ERROR, (errmsg("gtree: empty tree")));
	if (t->root < 0 || t->root >= t->count)
		ereport(ERROR, (errmsg("gtree: invalid root")));

	seen = (bool *) palloc0((Size) t->count);
	if (!seen)
		ereport(ERROR,
				(errmsg("gtree_validate: out of memory for seen "
						"array")));
	depth = gtree_depth_dfs(t, t->root, 0, seen);
	NDB_FREE(seen);

	if (depth > GTREE_MAX_DEPTH)
		ereport(ERROR,
				(errmsg("gtree: depth %d exceeds limit", depth)));
}

/*
 * Reset the tree to empty. Memory stays allocated.
 */
void
gtree_reset(GTree *t)
{
	if (!t)
		return;
	t->count = 0;
	t->root = -1;
	t->max_depth = 0;
}

/*
 * Return the pointer to the node array, or NULL on NULL tree.
 */
const GTreeNode *
gtree_nodes(const GTree *t)
{
	if (!t)
		return NULL;
	return t->nodes;
}

/*
 * Return the number of nodes in the tree.
 */
int
gtree_count(const GTree *t)
{
	if (!t)
		return 0;
	return t->count;
}
