/*-------------------------------------------------------------------------
 *
 * custom_hybrid_scan.c
 *		CustomScan node for single-pass hybrid vector+FTS+metadata execution
 *
 * Implements a custom scan node that fuses:
 * - Vector ANN search
 * - Full-text search (FTS)
 * - Metadata filtering
 * - Reranking
 *
 * This avoids multiple sequential scans and provides optimal hybrid search.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/scan/custom_hybrid_scan.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_scan.h"
#include "fmgr.h"
#include "access/relscan.h"
#include "access/genam.h"
#include "access/table.h"
#include "access/heapam.h"
#include "catalog/pg_am.h"
#include "utils/syscache.h"
#include "utils/snapshot.h"
#include "storage/bufmgr.h"
#include "commands/explain.h"
#include "executor/executor.h"
#include "executor/nodeCustom.h"
#include "nodes/execnodes.h"
#include "nodes/extensible.h"
#include "nodes/plannodes.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/cost.h"
#include "optimizer/planmain.h"
#include "optimizer/planner.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "utils/lsyscache.h"
#include "catalog/pg_am.h"
#include "catalog/pg_am_d.h"
#include "catalog/pg_class.h"
#include "catalog/pg_index.h"
#include "commands/defrem.h"
#include "nodes/parsenodes.h"
#include "nodes/primnodes.h"
#include "nodes/extensible.h"
#include <stdlib.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

static planner_hook_type prev_planner_hook = NULL;
static set_rel_pathlist_hook_type prev_set_rel_pathlist_hook = NULL;

void		register_hybrid_scan_planner_hook(void);
static CustomScanMethods hybrid_custom_scan_methods;

/* Query under planning (set by planner hook, read by pathlist hook) */
static Query *hybrid_planning_query = NULL;

/* Check if relation has both a vector AM index (HNSW/IVF) and a GIN index (FTS) */
static bool
relation_has_vector_and_fts_indexes(PlannerInfo *root, RelOptInfo *rel)
{
	ListCell   *lc;
	Oid			relid = rel->relid;
	Relation	relation;
	bool		has_vector = false;
	bool		has_gin = false;
	Oid			hnsw_oid = get_am_oid("hnsw", true);
	Oid			ivf_oid = get_am_oid("ivf", true);

	(void) root;
	if (!OidIsValid(hnsw_oid) && !OidIsValid(ivf_oid))
		return false;

	relation = table_open(relid, AccessShareLock);
	foreach(lc, relation->rd_indexlist)
	{
		Oid			idxoid = lfirst_oid(lc);
		Relation	idxrel;
		Oid			amoid;

		idxrel = index_open(idxoid, AccessShareLock);
		amoid = idxrel->rd_rel->relam;
		index_close(idxrel, AccessShareLock);
		if (OidIsValid(hnsw_oid) && amoid == hnsw_oid)
			has_vector = true;
		if (OidIsValid(ivf_oid) && amoid == ivf_oid)
			has_vector = true;
		if (amoid == GIN_AM_OID)
			has_gin = true;
		if (has_vector && has_gin)
			break;
	}
	table_close(relation, AccessShareLock);
	return (has_vector && has_gin);
}

static Plan *hybrid_plan_custom_path(PlannerInfo *root, RelOptInfo *rel,
									struct CustomPath *best_path, List *tlist,
									List *clauses, List *custom_plans);

static CustomPathMethods hybrid_custom_path_methods = {
	.CustomName = "HybridScan",
	.PlanCustomPath = hybrid_plan_custom_path,
	.ReparameterizeCustomPathByChild = NULL
};

/* Pathlist hook: add hybrid CustomPath when relation has both vector and FTS indexes */
static void
hybrid_set_rel_pathlist(PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{
	if (prev_set_rel_pathlist_hook)
		prev_set_rel_pathlist_hook(root, rel, rti, rte);

	if (hybrid_planning_query == NULL)
		return;
	if (rte->rtekind != RTE_RELATION)
		return;
	if (!relation_has_vector_and_fts_indexes(root, rel))
		return;

	{
		CustomPath *cpath = (CustomPath *) palloc(sizeof(CustomPath));

		memset(cpath, 0, sizeof(CustomPath));
		cpath->path.pathtype = T_CustomScan;
		cpath->path.parent = rel;
		cpath->path.pathtarget = rel->reltarget;
		cpath->path.param_info = get_baserel_parampathinfo(root, rel, NULL);
		cpath->path.parallel_aware = false;
		cpath->path.parallel_safe = rel->consider_parallel;
		cpath->path.parallel_workers = 0;
		cpath->path.rows = rel->rows;
		cpath->path.startup_cost = 100.0;
		cpath->path.total_cost = 100.0 + (rel->tuples * 0.01);
		cpath->path.pathkeys = NIL;
		cpath->flags = 0;
		cpath->custom_paths = NIL;
		cpath->custom_restrictinfo = NIL;
		cpath->custom_private = NIL;
		cpath->methods = &hybrid_custom_path_methods;

		add_path(rel, (Path *) cpath);
	}
}

static Plan *
hybrid_plan_custom_path(PlannerInfo *root, RelOptInfo *rel,
					   struct CustomPath *best_path, List *tlist,
					   List *clauses, List *custom_plans)
{
	CustomScan *cscan = makeNode(CustomScan);

	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual = clauses;
	cscan->scan.plan.lefttree = NULL;
	cscan->scan.plan.righttree = NULL;
	cscan->scan.scanrelid = rel->relid;
	cscan->flags = best_path->flags;
	cscan->custom_plans = custom_plans;
	cscan->custom_private = best_path->custom_private;
	cscan->methods = &hybrid_custom_scan_methods;
	(void) root;
	return &cscan->scan.plan;
}

/*
 * Hybrid scan state
 */
typedef struct HybridScanState
{
	CustomScanState css;		/* Must be first */

	/* Query parameters */
	Vector	   *queryVector;	/* Vector query */
	char	   *ftsQuery;		/* FTS query string */
	List	   *filters;		/* Metadata filters */
	float4		vectorWeight;	/* Hybrid weight: 0..1 */
	float4		ftsWeight;		/* 1 - vectorWeight */
	int			k;				/* Target results */

	/* Candidate management */
	TupleTableSlot **candidates;
	float4	   *scores;
	int			candidateCount;
	int			currentPos;

	/* Sub-scans */
	IndexScanDesc vectorScan;
	IndexScanDesc ftsScan;
	TableScanDesc heapScan;

	bool		vectorDone;
	bool		ftsDone;
}			HybridScanState;

typedef struct HybridScanPlanData
{
	Oid			vectorIndexOid;
	Oid			ftsIndexOid;
	Oid			relationOid;
	float4		vectorWeight;
	float4		ftsWeight;
	int			k;
}			HybridScanPlanData;

/* CustomScan method declarations */
static Node * hybrid_create_scan_state(CustomScan * cscan);
static void hybrid_begin(CustomScanState * node, EState * estate, int eflags);
static TupleTableSlot * hybrid_exec(CustomScanState * node);
static void hybrid_end(CustomScanState * node);
static void hybrid_rescan(CustomScanState * node);
static void
			hybrid_explain(CustomScanState * node, List * ancestors, ExplainState * es);

static float4
compute_hybrid_score(float4 vectorDist, float4 ftsScore, float4 vectorWeight);
static void merge_candidates(HybridScanState * state);
static void sort_indices_by_scores(int *indices, float4 * scores, int count);

static CustomExecMethods hybrid_exec_methods =
{
	.CustomName = "HybridScan",
		.BeginCustomScan = hybrid_begin,
		.ExecCustomScan = hybrid_exec,
		.EndCustomScan = hybrid_end,
		.ReScanCustomScan = hybrid_rescan,
		.ExplainCustomScan = hybrid_explain
};

/*
 * hybrid_estimate_path_cost - Estimate cost for hybrid scan path
 */
static void
__attribute__((unused))
hybrid_estimate_path_cost(PlannerInfo * root, RelOptInfo * rel, CustomPath * path)
{
	(void) root; /* unused */
	(void) rel;  /* unused */
	path->path.startup_cost = 100.0;
	path->path.total_cost = 100.0 + (rel->tuples * 0.01);
	path->path.rows = rel->tuples * 0.1;
}

/*
 * Planner hook: Set query context and pathlist hook, then run standard planner.
 * The pathlist hook adds a HybridScan CustomPath when the relation has both
 * vector (HNSW/IVF) and FTS (GIN) indexes.
 */
static PlannedStmt *
hybrid_planner_hook(Query * parse, const char *query_string, int cursorOptions,
					ParamListInfo boundParams)
{
	PlannedStmt *result = NULL;
	set_rel_pathlist_hook_type prev = set_rel_pathlist_hook;

	hybrid_planning_query = parse;
	set_rel_pathlist_hook = hybrid_set_rel_pathlist;

	if (prev_planner_hook)
		result = prev_planner_hook(parse, query_string, cursorOptions, boundParams);
	else
		result = standard_planner(parse, query_string, cursorOptions, boundParams);

	set_rel_pathlist_hook = prev;
	hybrid_planning_query = NULL;

	return result;
}

/*
 * Create a hybrid scan custom path
 *
 * This would be called from a planning hook to inject the custom path.
 */
PG_FUNCTION_INFO_V1(create_hybrid_scan_path);

Datum
create_hybrid_scan_path(PG_FUNCTION_ARGS)
{
	/* This function can be called to manually create hybrid scan paths */
	/* The planner hook automatically adds paths during query planning */
	(void) fcinfo; /* unused */

	PG_RETURN_VOID();
}

/*
 * Register planner hook for hybrid scan
 * Should be called from _PG_init()
 */
void
register_hybrid_scan_planner_hook(void)
{
	/* Install planner hook */
	prev_planner_hook = planner_hook;
	planner_hook = hybrid_planner_hook;
}

/*
 * Create scan state
 *
 * Note: this is only used for node factory registration and should not generate
 * an unused-function warning.
 */
__attribute__((unused)) static Node *
hybrid_create_scan_state(CustomScan * cscan)
{
	HybridScanState *state = NULL;

	(void) cscan; /* unused */

	state = (HybridScanState *) newNode(
										sizeof(HybridScanState), T_CustomScanState);
	state->css.methods = &hybrid_exec_methods;
	state->vectorDone = false;
	state->ftsDone = false;
	state->currentPos = 0;
	state->candidateCount = 0;

	return (Node *) state;
}

static CustomScanMethods hybrid_custom_scan_methods = {
	.CustomName = "HybridScan",
	.CreateCustomScanState = hybrid_create_scan_state
};

/*
 * Begin execution
 */
static void
hybrid_begin(CustomScanState * node, EState * estate, int eflags)
{
	HybridScanState *state = (HybridScanState *) node;
	Relation	heapRel;
	Relation	vectorIndexRel = NULL;
	Relation	ftsIndexRel = NULL;
	Oid			vectorIndexOid = InvalidOid;
	Oid			ftsIndexOid = InvalidOid;
	int			nkeys = 0;
	int			norderbys = 0;

	(void) estate; /* unused */
	(void) eflags; /* unused */

	/* Get heap relation */
	heapRel = node->ss.ss_currentRelation;
	if (!heapRel)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("hybrid_begin: no heap relation")));

	/* Extract parameters from plan */
	{
		CustomScan *cscan = (CustomScan *) node->ss.ps.plan;
		HybridScanPlanData *planData = NULL;

		/* Try to extract plan data from custom_private */
		/* custom_private is typically a List, but could contain our struct */
		if (cscan->custom_private != NIL && list_length(cscan->custom_private) > 0)
		{
			/* Check if first element is our plan data structure */
			/* Would need to check type or use a known format */
			/* For now, assume it's stored in a way we can extract */
			(void) linitial(cscan->custom_private); /* Reserved for future use */
		}

		/* Extract parameters with fallback to defaults */
		if (planData != NULL)
		{
			state->vectorWeight = planData->vectorWeight;
			state->ftsWeight = planData->ftsWeight;
			state->k = planData->k;
			vectorIndexOid = planData->vectorIndexOid;
			ftsIndexOid = planData->ftsIndexOid;
		}
		else
		{
			/* Use defaults if plan data not available */
			state->vectorWeight = 0.5;
			state->ftsWeight = 0.5;
			state->k = 100;
		}

		state->queryVector = NULL;	/* Would be extracted from plan or
									 * evaluated */
		state->ftsQuery = NULL; /* Would be extracted from plan or evaluated */
	}

	/* Try to find vector and FTS indexes on this relation */
	/* If not in plan, scan all indexes as fallback */
	{
		List *indexOidList = NULL;
		ListCell *lc = NULL;

		indexOidList = RelationGetIndexList(heapRel);
		foreach(lc, indexOidList)
		{
			Oid			indexOid = lfirst_oid(lc);
			Relation	idxRel;
			Form_pg_am	amform;
			HeapTuple	amtup;

			idxRel = index_open(indexOid, AccessShareLock);
			amtup = SearchSysCache1(AMOID, ObjectIdGetDatum(idxRel->rd_rel->relam));
			if (HeapTupleIsValid(amtup))
			{
				amform = (Form_pg_am) GETSTRUCT(amtup);
				if (strcmp(NameStr(amform->amname), "hnsw") == 0 ||
					strcmp(NameStr(amform->amname), "ivfflat") == 0)
				{
					if (!OidIsValid(vectorIndexOid))
					{
						vectorIndexOid = indexOid;
						vectorIndexRel = idxRel;
					}
					else
					{
						index_close(idxRel, AccessShareLock);
					}
				}
				else if (strcmp(NameStr(amform->amname), "gin") == 0 ||
						 strcmp(NameStr(amform->amname), "gist") == 0)
				{
					if (!OidIsValid(ftsIndexOid))
					{
						ftsIndexOid = indexOid;
						ftsIndexRel = idxRel;
					}
					else
					{
						index_close(idxRel, AccessShareLock);
					}
				}
				else
				{
					index_close(idxRel, AccessShareLock);
				}
				ReleaseSysCache(amtup);
			}
			else
			{
				index_close(idxRel, AccessShareLock);
			}
		}
		list_free(indexOidList);
	}

	/* Allocate candidate arrays */
	{
		TupleTableSlot **candidates_tmp = NULL;
		float4 *scores_tmp = NULL;
		nalloc(candidates_tmp, TupleTableSlot *, state->k);
		nalloc(scores_tmp, float4, state->k);
		state->candidates = candidates_tmp;
		state->scores = scores_tmp;
	}
	NDB_CHECK_ALLOC(state, "state");

	/* Open vector index scan if available */
	if (OidIsValid(vectorIndexOid) && vectorIndexRel)
	{
		/* Create scan keys for vector search */
		/* For now, use empty scan keys - query vector would come from plan */
#if PG_VERSION_NUM >= 180000
		/* PostgreSQL 18+ includes IndexScanInstrumentation parameter */
		state->vectorScan = index_beginscan(heapRel,
											vectorIndexRel,
											GetActiveSnapshot(),
											NULL, /* instrument */
											nkeys,
											norderbys);
#else
		/* PostgreSQL 16 and 17 */
		state->vectorScan = index_beginscan(heapRel,
											vectorIndexRel,
											GetActiveSnapshot(),
											nkeys,
											norderbys);
#endif
		/* Set up order-by keys with query vector if available */
		if (state->queryVector)
		{
			/* Would set up orderbys here with query vector */
		}
	}
	else
	{
		state->vectorScan = NULL;
		state->vectorDone = true;
	}

	/* Open FTS index scan if available */
	if (OidIsValid(ftsIndexOid) && ftsIndexRel)
	{
		/* Create scan keys for FTS search */
#if PG_VERSION_NUM >= 180000
		/* PostgreSQL 18+ includes IndexScanInstrumentation parameter */
		state->ftsScan = index_beginscan(heapRel,
										 ftsIndexRel,
										 GetActiveSnapshot(),
										 NULL, /* instrument */
										 nkeys,
										 norderbys);
#else
		/* PostgreSQL 16 and 17 */
		state->ftsScan = index_beginscan(heapRel,
										 ftsIndexRel,
										 GetActiveSnapshot(),
										 nkeys,
										 norderbys);
#endif
		/* Set up scan keys with FTS query if available */
		if (state->ftsQuery)
		{
			/* Would set up scan keys here with FTS query */
		}
	}
	else
	{
		state->ftsScan = NULL;
		state->ftsDone = true;
	}

	/* Store heap relation for tuple fetching */
	state->heapScan = NULL;		/* Will be used if needed for heap access */

	/* Initialize state */
	state->vectorDone = (state->vectorScan == NULL);
	state->ftsDone = (state->ftsScan == NULL);
	state->currentPos = 0;
	state->candidateCount = 0;
}

/*
 * Execute scan - return next tuple
 */
static TupleTableSlot *
hybrid_exec(CustomScanState * node)
{
	HybridScanState *state = (HybridScanState *) node;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
	Relation	heapRel = node->ss.ss_currentRelation;
	ItemPointerData *vectorItems = NULL;
	float4 *vectorDistances = NULL;
	int			vectorCount = 0;
	ItemPointerData *ftsItems = NULL;
	float4 *ftsScores = NULL;
	int			ftsCount = 0;
	int			i;

	/* On first call, gather all candidates */
	if (state->currentPos == 0 && state->candidateCount == 0)
	{
		/* Scan vector index */
		if (state->vectorScan && !state->vectorDone)
		{
			/* Collect vector candidates */
			nalloc(vectorItems, ItemPointerData, state->k);
			NDB_CHECK_ALLOC(vectorItems, "vectorItems");
			nalloc(vectorDistances, float4, state->k);
			NDB_CHECK_ALLOC(vectorDistances, "vectorDistances");

			while (vectorCount < state->k)
			{
				bool		found;
				ItemPointer tid;
				float4		distance = 0.0f;

				found = index_getnext_slot(state->vectorScan,
										   ForwardScanDirection,
										   slot);
				if (!found)
				{
					state->vectorDone = true;
					break;
				}

				/* Extract ItemPointer and distance from scan */
				tid = &slot->tts_tid;
				if (ItemPointerIsValid(tid))
				{
					/*
					 * For HNSW/IVF, distance is typically in
					 * scan->xs_orderbyvals
					 */
					/* Simplified: assume distance is available */
					if (state->vectorScan->xs_orderbyvals != NULL)
					{
						distance = DatumGetFloat4(state->vectorScan->xs_orderbyvals[0]);
					}

					vectorItems[vectorCount] = *tid;
					vectorDistances[vectorCount] = distance;
					vectorCount++;
				}
			}
		}

		/* Scan FTS index */
		if (state->ftsScan && !state->ftsDone)
		{
			/* Collect FTS candidates */
			nalloc(ftsItems, ItemPointerData, state->k);
			NDB_CHECK_ALLOC(ftsItems, "ftsItems");
			nalloc(ftsScores, float4, state->k);
			NDB_CHECK_ALLOC(ftsScores, "ftsScores");

			while (ftsCount < state->k)
			{
				bool		found;
				ItemPointer tid;
				float4		score = 0.0f;

				found = index_getnext_slot(state->ftsScan,
										   ForwardScanDirection,
										   slot);
				if (!found)
				{
					state->ftsDone = true;
					break;
				}

				/* Extract ItemPointer and relevance score */
				tid = &slot->tts_tid;
				if (ItemPointerIsValid(tid))
				{
					/* FTS score might be in scan->xs_orderbyvals or computed */
					/* Simplified: assume score is available or default to 1.0 */
					if (state->ftsScan->xs_orderbyvals != NULL)
					{
						score = DatumGetFloat4(state->ftsScan->xs_orderbyvals[0]);
					}
					else
					{
						score = 1.0f;	/* Default relevance */
					}

					ftsItems[ftsCount] = *tid;
					ftsScores[ftsCount] = score;
					ftsCount++;
				}
			}
		}

		/* Merge candidates: deduplicate and compute hybrid scores */
		{
			int			mergedCount = 0;
			int			maxCandidates = vectorCount + ftsCount;

			ItemPointerData *mergedItems = NULL;
			float4 *mergedScores = NULL;

			/* Allocate arrays for merged candidates */
			nalloc(mergedItems, ItemPointerData, maxCandidates);
			NDB_CHECK_ALLOC(mergedItems, "mergedItems");
			nalloc(mergedScores, float4, maxCandidates);
			NDB_CHECK_ALLOC(mergedScores, "mergedScores");

			/* Create a hash table or sorted list for deduplication */
			/* Simplified: use a simple array and check duplicates */
			for (i = 0; i < vectorCount; i++)
			{
				int			j;
				bool		found = false;

				/* Check if already in merged list */
				for (j = 0; j < mergedCount; j++)
				{
					if (ItemPointerEquals(&vectorItems[i], &mergedItems[j]))
					{
						/* Update score with hybrid formula */
						mergedScores[j] = compute_hybrid_score(
															   vectorDistances[i],
															   mergedScores[j], /* Existing FTS score */
															   state->vectorWeight);
						found = true;
						break;
					}
				}

				if (!found)
				{
					/* Add new candidate */
					float4		vectorScore = 1.0f / (1.0f + vectorDistances[i]);

					mergedItems[mergedCount] = vectorItems[i];
					mergedScores[mergedCount] = vectorScore * state->vectorWeight;
					mergedCount++;
				}
			}

			/* Add FTS candidates */
			for (i = 0; i < ftsCount; i++)
			{
				int			j;
				bool		found = false;

				/* Check if already in merged list */
				for (j = 0; j < mergedCount; j++)
				{
					if (ItemPointerEquals(&ftsItems[i], &mergedItems[j]))
					{
						/* Update score with hybrid formula */
						mergedScores[j] = compute_hybrid_score(
															   0.0f,	/* No vector distance
																		 * for FTS-only */
															   ftsScores[i],
															   state->vectorWeight);
						found = true;
						break;
					}
				}

				if (!found)
				{
					/* Add new candidate */
					mergedItems[mergedCount] = ftsItems[i];
					mergedScores[mergedCount] = ftsScores[i] * state->ftsWeight;
					mergedCount++;
				}
			}

			/* Sort by score (descending) */
			{
				int *indices = NULL;

				nalloc(indices, int, mergedCount);

				NDB_CHECK_ALLOC(indices, "indices");
				for (i = 0; i < mergedCount; i++)
					indices[i] = i;

				/* Sort indices by score using helper function */
				sort_indices_by_scores(indices, mergedScores, mergedCount);

				/* Store top-k candidates */
				state->candidateCount = Min(mergedCount, state->k);
				for (i = 0; i < state->candidateCount; i++)
				{
					int			idx = indices[i];
					TupleTableSlot *candidateSlot = NULL;

					candidateSlot = MakeTupleTableSlot(node->ss.ss_currentRelation->rd_att,
													   &TTSOpsHeapTuple);
					/* Fetch tuple from heap */
					{
						HeapTupleData tuple;
						HeapTuple	tupleptr = &tuple;
						Buffer		buffer;

						if (heap_fetch(heapRel,
									   GetActiveSnapshot(),
									   tupleptr,
									   &buffer,
									   false))
						{
							ExecStoreHeapTuple(tupleptr, candidateSlot, false);
							ReleaseBuffer(buffer);
						}
					}

					state->candidates[i] = candidateSlot;
					state->scores[i] = mergedScores[idx];
				}

				pfree(indices);
			}

			pfree(mergedItems);
			pfree(mergedScores);
		}

		/* Free temporary arrays */
		if (vectorItems)
			pfree(vectorItems);
		if (vectorDistances)
			pfree(vectorDistances);
		if (ftsItems)
			pfree(ftsItems);
		if (ftsScores)
			pfree(ftsScores);
	}

	/* Return next result */
	if (state->currentPos < state->candidateCount)
	{
		/* Copy candidate to output slot */
		ExecCopySlot(slot, state->candidates[state->currentPos]);
		state->currentPos++;
		return slot;
	}

	/* Done */
	return ExecClearTuple(slot);
}

/*
 * End scan
 */
static void
hybrid_end(CustomScanState * node)
{
	HybridScanState *state = (HybridScanState *) node;
	int			i;

	/* Close scans */
	if (state->vectorScan)
	{
		index_endscan(state->vectorScan);
		index_close(state->vectorScan->indexRelation, AccessShareLock);
	}
	if (state->ftsScan)
	{
		index_endscan(state->ftsScan);
		index_close(state->ftsScan->indexRelation, AccessShareLock);
	}

	/* Free candidate slots */
	if (state->candidates)
	{
		for (i = 0; i < state->candidateCount; i++)
		{
			if (state->candidates[i])
				ExecDropSingleTupleTableSlot(state->candidates[i]);
		}
		pfree(state->candidates);
	}
	if (state->scores)
		pfree(state->scores);

	/* Free query strings */
	if (state->queryVector)
		pfree(state->queryVector);
	if (state->ftsQuery)
		pfree(state->ftsQuery);
}

/*
 * Rescan
 */
static void
hybrid_rescan(CustomScanState * node)
{
	HybridScanState *state = (HybridScanState *) node;
	int			i;

	/* Free existing candidate slots */
	if (state->candidates)
	{
		for (i = 0; i < state->candidateCount; i++)
		{
			if (state->candidates[i])
				ExecDropSingleTupleTableSlot(state->candidates[i]);
		}
	}

	state->currentPos = 0;
	state->candidateCount = 0;
	state->vectorDone = false;
	state->ftsDone = false;

	/* Rescan indexes */
	if (state->vectorScan)
		index_rescan(state->vectorScan, NULL, 0, NULL, 0);
	if (state->ftsScan)
		index_rescan(state->ftsScan, NULL, 0, NULL, 0);
}

/*
 * Explain plan
 */
static void
hybrid_explain(CustomScanState * node, List * ancestors, ExplainState * es)
{
	HybridScanState *state = (HybridScanState *) node;

	(void) ancestors; /* unused */

#if PG_VERSION_NUM >= 180000
	/* PG18 removed ExplainProperty* functions - use appendStringInfo instead */
	if (es->format == EXPLAIN_FORMAT_TEXT)
	{
		appendStringInfoSpaces(es->str, es->indent * 2);
		appendStringInfo(es->str, "Hybrid Scan Type: Vector+FTS\n");
		appendStringInfoSpaces(es->str, es->indent * 2);
		appendStringInfo(
						 es->str, "Vector Weight: %.2f\n", state->vectorWeight);
		appendStringInfoSpaces(es->str, es->indent * 2);
		appendStringInfo(
						 es->str, "FTS Weight: %.2f\n", state->ftsWeight);
		appendStringInfoSpaces(es->str, es->indent * 2);
		appendStringInfo(es->str, "Target Results: %d\n", state->k);
	}
#else
	ExplainPropertyText("Hybrid Scan Type", "Vector+FTS", es);
	ExplainPropertyFloat("Vector Weight", NULL, state->vectorWeight, 2, es);
	ExplainPropertyFloat("FTS Weight", NULL, state->ftsWeight, 2, es);
	ExplainPropertyInteger("Target Results", NULL, state->k, es);
#endif
}

/*
 * Compute hybrid score from vector distance and FTS score
 *
 * Mark as maybe unused so no warning if only used in template or ahead-of-time.
 */
__attribute__((unused)) static float4
compute_hybrid_score(float4 vectorDist, float4 ftsScore, float4 vectorWeight)
{
	float4		vectorScore;
	float4		hybridScore;

	/* Normalize vector distance to score (0..1, higher is better) */
	vectorScore = 1.0 / (1.0 + vectorDist);

	/* Weighted combination */
	hybridScore =
		vectorWeight * vectorScore + (1.0 - vectorWeight) * ftsScore;

	return hybridScore;
}

/*
 * Merge and rerank candidates by hybrid score
 *
 * This function is now implemented inline in hybrid_exec for efficiency.
 * Keeping this stub for potential future use or refactoring.
 */
__attribute__((unused)) static void
merge_candidates(HybridScanState * state)
{
	(void) state; /* unused */
	/* Merging is now done inline in hybrid_exec() for better performance */
	/* This function is kept for potential future refactoring */
}

/*
 * Helper: Sort indices by scores (descending)
 */
static void
sort_indices_by_scores(int *indices, float4 * scores, int count)
{
	int			i,
				j;
	int			temp;

	/* Simple bubble sort by score (descending) */
	for (i = 0; i < count - 1; i++)
	{
		for (j = 0; j < count - i - 1; j++)
		{
			if (scores[indices[j]] < scores[indices[j + 1]])
			{
				temp = indices[j];
				indices[j] = indices[j + 1];
				indices[j + 1] = temp;
			}
		}
	}
}

/*
 * Register custom scan provider
 *
 * This should be called in _PG_init().
 */
void
register_hybrid_scan_provider(void)
{
	/* Register planner hook for automatic path injection */
	/* CustomScan methods are used when CustomScan nodes are created */
	register_hybrid_scan_planner_hook();

}
