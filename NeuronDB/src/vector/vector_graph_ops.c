/*-------------------------------------------------------------------------
 *
 * vector_graph_ops.c
 *	  Graph-based operations for vgraph type
 *
 * This file implements graph traversal and analysis algorithms including:
 * - BFS (Breadth-First Search)
 * - DFS (Depth-First Search)
 * - PageRank algorithm
 * - Community detection (Louvain algorithm)
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/vector/vector_graph_ops.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_types.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "lib/stringinfo.h"
#include "utils/array.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Queue structure for BFS */
typedef struct QueueNode
{
	int32		node_idx;
	struct QueueNode *next;
}			QueueNode;

typedef struct Queue
{
	QueueNode  *front;
	QueueNode  *rear;
	int			size;
}			Queue;

static Queue *
queue_create(void)
{
	NDB_DECLARE(Queue *, q);
	NDB_ALLOC(q, Queue, 1);

	q->front = NULL;
	q->rear = NULL;
	q->size = 0;
	return q;
}

static void
queue_enqueue(Queue * q, int32 node_idx)
{
	NDB_DECLARE(QueueNode *, node);
	NDB_ALLOC(node, QueueNode, 1);

	node->node_idx = node_idx;
	node->next = NULL;

	if (q->rear == NULL)
	{
		q->front = node;
		q->rear = node;
	}
	else
	{
		q->rear->next = node;
		q->rear = node;
	}
	q->size++;
}

static int32
queue_dequeue(Queue * q)
{
	QueueNode  *node;
	int32		node_idx;

	if (q->front == NULL)
		return -1;

	node = q->front;
	node_idx = node->node_idx;
	q->front = node->next;

	if (q->front == NULL)
		q->rear = NULL;

	NDB_FREE(node);
	q->size--;
	return node_idx;
}

static void
queue_free(Queue * q)
{
	while (q->front != NULL)
		queue_dequeue(q);
	NDB_FREE(q);
}

/* Stack structure for DFS */
typedef struct StackNode
{
	int32		node_idx;
	struct StackNode *next;
}			StackNode;

typedef struct Stack
{
	StackNode  *top;
	int			size;
}			Stack;

static Stack *
stack_create(void)
{
	NDB_DECLARE(Stack *, s);
	NDB_ALLOC(s, Stack, 1);

	s->top = NULL;
	s->size = 0;
	return s;
}

static void
stack_push(Stack * s, int32 node_idx)
{
	NDB_DECLARE(StackNode *, node);
	NDB_ALLOC(node, StackNode, 1);

	node->node_idx = node_idx;
	node->next = s->top;
	s->top = node;
	s->size++;
}

static int32
stack_pop(Stack * s)
{
	StackNode  *node;
	int32		node_idx;

	if (s->top == NULL)
		return -1;

	node = s->top;
	node_idx = node->node_idx;
	s->top = node->next;

	NDB_FREE(node);
	s->size--;
	return node_idx;
}

static void
stack_free(Stack * s)
{
	while (s->top != NULL)
		stack_pop(s);
	NDB_FREE(s);
}

/*
 * BFS: Breadth-First Search from a starting node
 * Returns: SETOF (node_idx int, depth int, parent_idx int)
 */
PG_FUNCTION_INFO_V1(vgraph_bfs);
Datum
vgraph_bfs(PG_FUNCTION_ARGS)
{
	VectorGraph *graph = (VectorGraph *) PG_GETARG_POINTER(0);
	int32		start_node_idx = PG_GETARG_INT32(1);
	int32		max_depth = PG_ARGISNULL(2) ? -1 : PG_GETARG_INT32(2);
	FuncCallContext *funcctx;
	typedef struct bfs_fctx
	{
		int32	   *visited;
		int32	   *depth;
		int32	   *parent;
		Queue	   *queue;
		int32	   *result_nodes;
		int32	   *result_depths;
		int32	   *result_parents;
		int			result_count;
		int			current_idx;
	}			bfs_fctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		bfs_fctx   *fctx;
		GraphEdge  *edges;
		int32		i;
		int32		current_node;
		int32		max_result_size;

		NDB_DECLARE(int32 *, visited);
		NDB_DECLARE(int32 *, depth);
		NDB_DECLARE(int32 *, parent);
		NDB_DECLARE(int32 *, result_nodes);
		NDB_DECLARE(int32 *, result_depths);
		NDB_DECLARE(int32 *, result_parents);

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Validate start node */
		if (start_node_idx < 0 || start_node_idx >= graph->num_nodes)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("start_node_idx out of range: %d (max: %d)",
							start_node_idx,
							graph->num_nodes - 1)));

		fctx = NULL;
		NDB_ALLOC(fctx, bfs_fctx, 1);
		NDB_ALLOC(visited, int32, graph->num_nodes);
		NDB_ALLOC(depth, int32, graph->num_nodes);
		NDB_ALLOC(parent, int32, graph->num_nodes);
		/* Initialize arrays to zero */
		memset(visited, 0, sizeof(int32) * graph->num_nodes);
		memset(depth, 0, sizeof(int32) * graph->num_nodes);
		memset(parent, 0, sizeof(int32) * graph->num_nodes);
		fctx->visited = visited;
		fctx->depth = depth;
		fctx->parent = parent;

		/* Initialize depth and parent arrays */
		for (i = 0; i < graph->num_nodes; i++)
		{
			fctx->depth[i] = -1;
			fctx->parent[i] = -1;
		}

		fctx->queue = queue_create();
		edges = VGRAPH_EDGES(graph);

		/* Initialize BFS */
		fctx->visited[start_node_idx] = 1;
		fctx->depth[start_node_idx] = 0;
		fctx->parent[start_node_idx] = -1;
		queue_enqueue(fctx->queue, start_node_idx);

		max_result_size = graph->num_nodes;
		NDB_ALLOC(result_nodes, int32, max_result_size);
		NDB_ALLOC(result_depths, int32, max_result_size);
		NDB_ALLOC(result_parents, int32, max_result_size);
		fctx->result_nodes = result_nodes;
		fctx->result_depths = result_depths;
		fctx->result_parents = result_parents;
		fctx->result_count = 0;

		/* BFS traversal */
		while (fctx->queue->size > 0)
		{
			current_node = queue_dequeue(fctx->queue);

			/* Add to results */
			fctx->result_nodes[fctx->result_count] = current_node;
			fctx->result_depths[fctx->result_count] = fctx->depth[current_node];
			fctx->result_parents[fctx->result_count] = fctx->parent[current_node];
			fctx->result_count++;

			/* Check max depth */
			if (max_depth >= 0 && fctx->depth[current_node] >= max_depth)
				continue;

			/* Explore neighbors */
			for (i = 0; i < graph->num_edges; i++)
			{
				if (edges[i].src_idx == current_node)
				{
					int32		neighbor = edges[i].dst_idx;

					if (!fctx->visited[neighbor])
					{
						fctx->visited[neighbor] = 1;
						fctx->depth[neighbor] = fctx->depth[current_node] + 1;
						fctx->parent[neighbor] = current_node;
						queue_enqueue(fctx->queue, neighbor);
					}
				}
			}
		}

		queue_free(fctx->queue);
		fctx->current_idx = 0;

		funcctx->user_fctx = fctx;
		MemoryContextSwitchTo(oldcontext);

		/* Build tuple descriptor */
		if (get_call_result_type(fcinfo, NULL, &funcctx->tuple_desc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_FUNCTION_DEFINITION),
					 errmsg("return type must be a composite type")));
		BlessTupleDesc(funcctx->tuple_desc);
	}

	funcctx = SRF_PERCALL_SETUP();
	{
		bfs_fctx   *fctx;
		HeapTuple	tuple;
		Datum		values[3];
		bool		nulls[3] = {false, false, false};

		fctx = (bfs_fctx *) funcctx->user_fctx;

		if (fctx->current_idx < fctx->result_count)
		{
			values[0] = Int32GetDatum(fctx->result_nodes[fctx->current_idx]);
			values[1] = Int32GetDatum(fctx->result_depths[fctx->current_idx]);
			values[2] = Int32GetDatum(fctx->result_parents[fctx->current_idx]);

			tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
			fctx->current_idx++;
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
	}

	SRF_RETURN_DONE(funcctx);
}

/*
 * DFS: Depth-First Search from a starting node
 * Returns: SETOF (node_idx int, discovery_time int, finish_time int, parent_idx int)
 */
PG_FUNCTION_INFO_V1(vgraph_dfs);
Datum
vgraph_dfs(PG_FUNCTION_ARGS)
{
	VectorGraph *graph = (VectorGraph *) PG_GETARG_POINTER(0);
	int32		start_node_idx = PG_GETARG_INT32(1);
	FuncCallContext *funcctx;
	typedef struct dfs_fctx
	{
		int32	   *visited;
		int32	   *discovery_time;
		int32	   *finish_time;
		int32	   *parent;
		int32	   *result_nodes;
		int32	   *result_discovery;
		int32	   *result_finish;
		int32	   *result_parents;
		int			result_count;
		int			current_idx;
		int			time_counter;
	}			dfs_fctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		dfs_fctx   *fctx;
		GraphEdge  *edges;
		Stack	   *stack;
		int32		current_node;
		int32		i;
		int32		max_result_size;

		NDB_DECLARE(int32 *, visited);
		NDB_DECLARE(int32 *, discovery_time);
		NDB_DECLARE(int32 *, finish_time);
		NDB_DECLARE(int32 *, parent);
		NDB_DECLARE(int32 *, result_nodes);
		NDB_DECLARE(int32 *, result_discovery);
		NDB_DECLARE(int32 *, result_finish);
		NDB_DECLARE(int32 *, result_parents);

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Validate start node */
		if (start_node_idx < 0 || start_node_idx >= graph->num_nodes)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("start_node_idx out of range: %d (max: %d)",
							start_node_idx,
							graph->num_nodes - 1)));

		fctx = NULL;
		NDB_ALLOC(fctx, dfs_fctx, 1);
		NDB_ALLOC(visited, int32, graph->num_nodes);
		NDB_ALLOC(discovery_time, int32, graph->num_nodes);
		NDB_ALLOC(finish_time, int32, graph->num_nodes);
		NDB_ALLOC(parent, int32, graph->num_nodes);
		/* Initialize arrays to zero */
		memset(visited, 0, sizeof(int32) * graph->num_nodes);
		memset(discovery_time, 0, sizeof(int32) * graph->num_nodes);
		memset(finish_time, 0, sizeof(int32) * graph->num_nodes);
		memset(parent, 0, sizeof(int32) * graph->num_nodes);
		fctx->visited = visited;
		fctx->discovery_time = discovery_time;
		fctx->finish_time = finish_time;
		fctx->parent = parent;

		/* Initialize arrays */
		for (i = 0; i < graph->num_nodes; i++)
		{
			fctx->discovery_time[i] = -1;
			fctx->finish_time[i] = -1;
			fctx->parent[i] = -1;
		}

		stack = stack_create();
		edges = VGRAPH_EDGES(graph);
		fctx->time_counter = 0;

		max_result_size = graph->num_nodes;
		NDB_ALLOC(result_nodes, int32, max_result_size);
		NDB_ALLOC(result_discovery, int32, max_result_size);
		NDB_ALLOC(result_finish, int32, max_result_size);
		NDB_ALLOC(result_parents, int32, max_result_size);
		fctx->result_nodes = result_nodes;
		fctx->result_discovery = result_discovery;
		fctx->result_finish = result_finish;
		fctx->result_parents = result_parents;
		fctx->result_count = 0;

		/* DFS traversal using iterative approach */
		stack_push(stack, start_node_idx);
		fctx->parent[start_node_idx] = -1;

		while (stack->size > 0)
		{
			current_node = stack_pop(stack);

			if (fctx->visited[current_node])
			{
				/* Finish time */
				if (fctx->finish_time[current_node] == -1)
				{
					fctx->finish_time[current_node] = fctx->time_counter++;
					fctx->result_nodes[fctx->result_count] = current_node;
					fctx->result_discovery[fctx->result_count] =
						fctx->discovery_time[current_node];
					fctx->result_finish[fctx->result_count] =
						fctx->finish_time[current_node];
					fctx->result_parents[fctx->result_count] =
						fctx->parent[current_node];
					fctx->result_count++;
				}
				continue;
			}

			/* Discovery time */
			fctx->visited[current_node] = 1;
			fctx->discovery_time[current_node] = fctx->time_counter++;

			/* Push back to process finish time later */
			stack_push(stack, current_node);

			/* Explore neighbors (in reverse order for correct DFS order) */
			for (i = graph->num_edges - 1; i >= 0; i--)
			{
				if (edges[i].src_idx == current_node)
				{
					int32		neighbor = edges[i].dst_idx;

					if (!fctx->visited[neighbor])
					{
						fctx->parent[neighbor] = current_node;
						stack_push(stack, neighbor);
					}
				}
			}
		}

		stack_free(stack);
		fctx->current_idx = 0;

		funcctx->user_fctx = fctx;
		MemoryContextSwitchTo(oldcontext);

		/* Build tuple descriptor */
		if (get_call_result_type(fcinfo, NULL, &funcctx->tuple_desc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_FUNCTION_DEFINITION),
					 errmsg("return type must be a composite type")));
		BlessTupleDesc(funcctx->tuple_desc);
	}

	funcctx = SRF_PERCALL_SETUP();
	{
		dfs_fctx   *fctx;
		HeapTuple	tuple;
		Datum		values[4];
		bool		nulls[4] = {false, false, false, false};

		fctx = (dfs_fctx *) funcctx->user_fctx;

		if (fctx->current_idx < fctx->result_count)
		{
			values[0] = Int32GetDatum(fctx->result_nodes[fctx->current_idx]);
			values[1] = Int32GetDatum(fctx->result_discovery[fctx->current_idx]);
			values[2] = Int32GetDatum(fctx->result_finish[fctx->current_idx]);
			values[3] = Int32GetDatum(fctx->result_parents[fctx->current_idx]);

			tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
			fctx->current_idx++;
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
	}

	SRF_RETURN_DONE(funcctx);
}

/*
 * PageRank: Compute PageRank scores for all nodes
 * Returns: SETOF (node_idx int, pagerank_score double precision)
 */
PG_FUNCTION_INFO_V1(vgraph_pagerank);
Datum
vgraph_pagerank(PG_FUNCTION_ARGS)
{
	VectorGraph *graph = (VectorGraph *) PG_GETARG_POINTER(0);
	float8		damping_factor = PG_ARGISNULL(1) ? 0.85 : PG_GETARG_FLOAT8(1);
	int32		max_iterations = PG_ARGISNULL(2) ? 100 : PG_GETARG_INT32(2);
	float8		tolerance = PG_ARGISNULL(3) ? 1e-6 : PG_GETARG_FLOAT8(3);
	FuncCallContext *funcctx;
	typedef struct pagerank_fctx
	{
		double	   *scores;
		double	   *new_scores;
		int32	   *out_degree;
		int32	   *result_nodes;
		double	   *result_scores;
		int			result_count;
		int			current_idx;
	}			pagerank_fctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		pagerank_fctx *fctx;
		GraphEdge  *edges;
		int32		i;
		int32		iter;
		double		diff;
		double		initial_score;
		double		damping_sum;

		NDB_DECLARE(double *, scores);
		NDB_DECLARE(double *, new_scores);
		NDB_DECLARE(int32 *, out_degree);
		NDB_DECLARE(int32 *, result_nodes);
		NDB_DECLARE(double *, result_scores);

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (damping_factor < 0.0 || damping_factor > 1.0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("damping_factor must be between 0 and 1")));

		fctx = NULL;
		NDB_ALLOC(fctx, pagerank_fctx, 1);
		NDB_ALLOC(scores, double, graph->num_nodes);
		NDB_ALLOC(new_scores, double, graph->num_nodes);
		NDB_ALLOC(out_degree, int32, graph->num_nodes);
		/* Initialize arrays to zero */
		memset(scores, 0, sizeof(double) * graph->num_nodes);
		memset(new_scores, 0, sizeof(double) * graph->num_nodes);
		memset(out_degree, 0, sizeof(int32) * graph->num_nodes);
		fctx->scores = scores;
		fctx->new_scores = new_scores;
		fctx->out_degree = out_degree;
		edges = VGRAPH_EDGES(graph);

		/* Compute out-degrees */
		for (i = 0; i < graph->num_edges; i++)
		{
			if (edges[i].src_idx >= 0 && edges[i].src_idx < graph->num_nodes)
				fctx->out_degree[edges[i].src_idx]++;
		}

		/* Initialize PageRank scores */
		initial_score = 1.0 / (double) graph->num_nodes;
		for (i = 0; i < graph->num_nodes; i++)
		{
			fctx->scores[i] = initial_score;
			fctx->new_scores[i] = 0.0;
		}

		/* PageRank iteration */
		for (iter = 0; iter < max_iterations; iter++)
		{
			/* Reset new scores */
			damping_sum = (1.0 - damping_factor) / (double) graph->num_nodes;
			for (i = 0; i < graph->num_nodes; i++)
				fctx->new_scores[i] = damping_sum;

			/* Distribute PageRank */
			for (i = 0; i < graph->num_edges; i++)
			{
				int32		src = edges[i].src_idx;
				int32		dst = edges[i].dst_idx;

				if (src >= 0 && src < graph->num_nodes
					&& dst >= 0 && dst < graph->num_nodes)
				{
					if (fctx->out_degree[src] > 0)
					{
						fctx->new_scores[dst] +=
							damping_factor * fctx->scores[src] /
							(double) fctx->out_degree[src];
					}
				}
			}

			/* Check convergence */
			diff = 0.0;
			for (i = 0; i < graph->num_nodes; i++)
			{
				double		change = fabs(fctx->new_scores[i] - fctx->scores[i]);

				if (change > diff)
					diff = change;
			}

			/* Swap scores */
			for (i = 0; i < graph->num_nodes; i++)
			{
				double		temp = fctx->scores[i];

				fctx->scores[i] = fctx->new_scores[i];
				fctx->new_scores[i] = temp;
			}

			if (diff < tolerance)
				break;
		}

		/* Prepare results */
		NDB_ALLOC(result_nodes, int32, graph->num_nodes);
		NDB_ALLOC(result_scores, double, graph->num_nodes);
		fctx->result_nodes = result_nodes;
		fctx->result_scores = result_scores;
		fctx->result_count = graph->num_nodes;

		for (i = 0; i < graph->num_nodes; i++)
		{
			fctx->result_nodes[i] = i;
			fctx->result_scores[i] = fctx->scores[i];
		}

		fctx->current_idx = 0;

		funcctx->user_fctx = fctx;
		MemoryContextSwitchTo(oldcontext);

		/* Build tuple descriptor */
		if (get_call_result_type(fcinfo, NULL, &funcctx->tuple_desc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_FUNCTION_DEFINITION),
					 errmsg("return type must be a composite type")));
		BlessTupleDesc(funcctx->tuple_desc);
	}

	funcctx = SRF_PERCALL_SETUP();
	{
		pagerank_fctx *fctx;
		HeapTuple	tuple;
		Datum		values[2];
		bool		nulls[2] = {false, false};

		fctx = (pagerank_fctx *) funcctx->user_fctx;

		if (fctx->current_idx < fctx->result_count)
		{
			values[0] = Int32GetDatum(fctx->result_nodes[fctx->current_idx]);
			values[1] = Float8GetDatum(fctx->result_scores[fctx->current_idx]);

			tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
			fctx->current_idx++;
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
	}

	SRF_RETURN_DONE(funcctx);
}

/*
 * Community Detection: Simplified Louvain algorithm
 * Returns: SETOF (node_idx int, community_id int, modularity double precision)
 */
PG_FUNCTION_INFO_V1(vgraph_community_detection);
Datum
vgraph_community_detection(PG_FUNCTION_ARGS)
{
	VectorGraph *graph = (VectorGraph *) PG_GETARG_POINTER(0);
	int32		max_iterations = PG_ARGISNULL(1) ? 10 : PG_GETARG_INT32(1);
	FuncCallContext *funcctx;
	typedef struct community_fctx
	{
		int32	   *communities;
		int32	   *result_nodes;
		int32	   *result_communities;
		double	   *result_modularity;
		int			result_count;
		int			current_idx;
	}			community_fctx;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		community_fctx *fctx;
		GraphEdge  *edges;
		int32		i;
		int32		iter;
		double		total_edges;
		double		modularity;

		NDB_DECLARE(int32 *, communities);
		NDB_DECLARE(int32 *, result_nodes);
		NDB_DECLARE(int32 *, result_communities);
		NDB_DECLARE(double *, result_modularity);

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		fctx = NULL;
		NDB_ALLOC(fctx, community_fctx, 1);
		NDB_ALLOC(communities, int32, graph->num_nodes);
		fctx->communities = communities;
		edges = VGRAPH_EDGES(graph);

		/* Initialize: each node in its own community */
		for (i = 0; i < graph->num_nodes; i++)
			fctx->communities[i] = i;

		total_edges = (double) graph->num_edges;

		/* Simplified community detection: merge communities greedily */
		for (iter = 0; iter < max_iterations; iter++)
		{
			bool		changed = false;
			int32		j;

			/* For each node, try moving to neighbor's community */
			for (i = 0; i < graph->num_nodes; i++)
			{
				int32		best_community = fctx->communities[i];
				double		best_modularity = 0.0;
				int32		neighbor_community;

				/* Check neighbors */
				for (j = 0; j < graph->num_edges; j++)
				{
					if (edges[j].src_idx == i)
					{
						neighbor_community = fctx->communities[edges[j].dst_idx];

						/* Simplified: move if neighbor has higher degree */
						if (neighbor_community != fctx->communities[i])
						{
							/* Try this community */
							int32		old_comm = fctx->communities[i];

							fctx->communities[i] = neighbor_community;

							/* Compute modularity (simplified) */
							{
								double		mod = 0.0;
								int32		k;

								for (k = 0; k < graph->num_edges; k++)
								{
									if (fctx->communities[edges[k].src_idx]
										== fctx->communities[edges[k].dst_idx])
										mod += 1.0;
								}
								mod = mod / total_edges;

								if (mod > best_modularity)
								{
									best_modularity = mod;
									best_community = neighbor_community;
								}
							}

							fctx->communities[i] = old_comm;
						}
					}
				}

				if (best_community != fctx->communities[i])
				{
					fctx->communities[i] = best_community;
					changed = true;
				}
			}

			if (!changed)
				break;
		}

		/* Compute final modularity */
		{
			double		mod = 0.0;
			int32		j;

			for (j = 0; j < graph->num_edges; j++)
			{
				if (fctx->communities[edges[j].src_idx]
					== fctx->communities[edges[j].dst_idx])
					mod += 1.0;
			}
			modularity = mod / total_edges;
		}

		/* Prepare results */
		NDB_ALLOC(result_nodes, int32, graph->num_nodes);
		NDB_ALLOC(result_communities, int32, graph->num_nodes);
		NDB_ALLOC(result_modularity, double, graph->num_nodes);
		fctx->result_nodes = result_nodes;
		fctx->result_communities = result_communities;
		fctx->result_modularity = result_modularity;
		fctx->result_count = graph->num_nodes;

		for (i = 0; i < graph->num_nodes; i++)
		{
			fctx->result_nodes[i] = i;
			fctx->result_communities[i] = fctx->communities[i];
			fctx->result_modularity[i] = modularity;
		}

		fctx->current_idx = 0;

		funcctx->user_fctx = fctx;
		MemoryContextSwitchTo(oldcontext);

		/* Build tuple descriptor */
		if (get_call_result_type(fcinfo, NULL, &funcctx->tuple_desc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_FUNCTION_DEFINITION),
					 errmsg("return type must be a composite type")));
		BlessTupleDesc(funcctx->tuple_desc);
	}

	funcctx = SRF_PERCALL_SETUP();
	{
		community_fctx *fctx;
		HeapTuple	tuple;
		Datum		values[3];
		bool		nulls[3] = {false, false, false};

		fctx = (community_fctx *) funcctx->user_fctx;

		if (fctx->current_idx < fctx->result_count)
		{
			values[0] = Int32GetDatum(fctx->result_nodes[fctx->current_idx]);
			values[1] = Int32GetDatum(fctx->result_communities[fctx->current_idx]);
			values[2] = Float8GetDatum(fctx->result_modularity[fctx->current_idx]);

			tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
			fctx->current_idx++;
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
	}

	SRF_RETURN_DONE(funcctx);
}
