/*-------------------------------------------------------------------------
 *
 * temporal_integration.c
 *		Temporal scoring integration into ANN search
 *
 * Integrates time-decay scoring with vector similarity:
 * - Exponential time decay
 * - Recency boosting
 * - Time window filtering
 * - Temporal reranking
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/search/temporal_integration.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "access/htup_details.h"
#include "access/heapam.h"
#include "access/tableam.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/timestamp.h"
#include "utils/snapmgr.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include <math.h>

/* Temporal scoring parameters */
#define DEFAULT_DECAY_RATE 0.1	/* Per day */
#define DEFAULT_RECENCY_WEIGHT 0.3	/* 0..1 */

/*
 * Temporal scoring configuration
 */
typedef struct TemporalConfig
{
	float4		decayRate;		/* Exponential decay rate */
	float4		recencyWeight;	/* Weight for temporal component */
	TimestampTz referenceTime;	/* Reference timestamp (usually now) */
	Interval   *timeWindow;		/* Optional time window filter */
	bool		enabled;
}			TemporalConfig;

/*
 * Comparison function for temporal items (for qsort)
 */
typedef struct TemporalItem
{
	ItemPointer item;
	float4		distance;
	TimestampTz timestamp;
	float4		score;
}			TemporalItem;

static int
compare_temporal_items(const void *a, const void *b)
{
	const		TemporalItem *item_a = (const TemporalItem *) a;
	const		TemporalItem *item_b = (const TemporalItem *) b;

	/* Sort by decreasing score */
	if (item_b->score > item_a->score)
		return 1;
	if (item_b->score < item_a->score)
		return -1;
	return 0;
}

/*
 * Compute time decay factor
 *
 * Returns a multiplier in [0,1] based on how old the document is.
 * Uses exponential decay: exp(-lambda * age_days)
 */
static float4
compute_time_decay(TimestampTz docTime, TimestampTz refTime, float4 decayRate)
{
	float8		age_seconds;
	float8		age_days;
	float4		decay;

	age_seconds = (float8) (refTime - docTime) / USECS_PER_SEC;
	age_days = age_seconds / (24.0 * 3600.0);

	/* Exponential decay */
	decay = (float4) exp(-decayRate * age_days);

	return decay;
}

/*
 * Compute hybrid score with temporal component
 *
 * final_score = (1 - w) * vector_similarity + w * temporal_score
 */
static float4
temporal_compute_hybrid_score(float4 vectorDistance,
							  TimestampTz docTime,
							  TemporalConfig * config)
{
	float4		vectorScore;
	float4		temporalScore;
	float4		finalScore;

	if (!config->enabled)
		return 1.0 / (1.0 + vectorDistance);	/* Vector-only */

	/* Normalize vector distance to similarity score */
	vectorScore = 1.0 / (1.0 + vectorDistance);

	/* Compute temporal score */
	temporalScore = compute_time_decay(
									   docTime, config->referenceTime, config->decayRate);

	/* Weighted combination */
	finalScore = (1.0 - config->recencyWeight) * vectorScore
		+ config->recencyWeight * temporalScore;

	return finalScore;
}

/*
 * Check if document is within time window
 */
static bool
temporal_in_window(TimestampTz docTime, TemporalConfig * config)
{
	TimestampTz windowStart;

	if (!config->enabled || config->timeWindow == NULL)
		return true;

	/* Compute window start time */
	/* Proper interval subtraction: convert interval to microseconds */
	if (config->timeWindow != NULL)
	{
		int64		interval_usec;

		/* Convert interval to microseconds */
		interval_usec = config->timeWindow->time;
		interval_usec += (int64) config->timeWindow->day * USECS_PER_DAY;
		interval_usec += (int64) config->timeWindow->month * (30 * USECS_PER_DAY);

		windowStart = config->referenceTime - interval_usec;
	}
	else
	{
		/* No window specified, use default 7 days */
		windowStart = config->referenceTime
			- (7 * 24 * 3600 * USECS_PER_SEC);
	}

	return docTime >= windowStart;
}

/*
 * Rerank results by temporal score
 *
 * Takes a set of results and re-orders them by combined vector+temporal score.
 */
static void
temporal_rerank_results(ItemPointer * items,
						float4 * distances,
						TimestampTz * timestamps,
						int count,
						TemporalConfig * config)
{
	float4	   *scores;
	int			i,
				j;
	float4		temp_score,
				temp_dist;
	ItemPointer temp_item;
	TimestampTz temp_ts;

	if (!config->enabled || count == 0)
		return;

	/* Compute hybrid scores */
	scores = (float4 *) palloc(count * sizeof(float4));
	NDB_CHECK_ALLOC(scores, "scores");

	for (i = 0; i < count; i++)
	{
		scores[i] = temporal_compute_hybrid_score(
												  distances[i], timestamps[i], config);
	}

	/* Sort by score descending */
	if (count <= 50)
	{
		/* Bubble sort for small result sets */
		for (i = 0; i < count - 1; i++)
		{
			for (j = i + 1; j < count; j++)
			{
				if (scores[j] > scores[i])	/* Higher score = better */
				{
					/* Swap */
					temp_score = scores[i];
					scores[i] = scores[j];
					scores[j] = temp_score;

					temp_dist = distances[i];
					distances[i] = distances[j];
					distances[j] = temp_dist;

					temp_item = items[i];
					items[i] = items[j];
					items[j] = temp_item;

					temp_ts = timestamps[i];
					timestamps[i] = timestamps[j];
					timestamps[j] = temp_ts;
				}
			}
		}
	}
	else
	{
		/* Use qsort for larger sets */
		struct
		{
			ItemPointer item;
			float4		distance;
			TimestampTz timestamp;
			float4		score;
		}		   *sort_items;
		int			idx;

		sort_items = palloc(sizeof(*sort_items) * count);
		NDB_CHECK_ALLOC(sort_items, "allocation");
		for (idx = 0; idx < count; idx++)
		{
			sort_items[idx].item = items[idx];
			sort_items[idx].distance = distances[idx];
			sort_items[idx].timestamp = timestamps[idx];
			sort_items[idx].score = scores[idx];
		}

		qsort(sort_items,
			  count,
			  sizeof(*sort_items),
			  compare_temporal_items);

		for (idx = 0; idx < count; idx++)
		{
			items[idx] = sort_items[idx].item;
			distances[idx] = sort_items[idx].distance;
			timestamps[idx] = sort_items[idx].timestamp;
		}

		NDB_FREE(sort_items);
	}

	NDB_FREE(scores);

}

/*
 * SQL-callable function: temporal_score(vector_distance, timestamp)
 */
PG_FUNCTION_INFO_V1(neurondb_temporal_score);

Datum
neurondb_temporal_score(PG_FUNCTION_ARGS)
{
	float4		vectorDistance = PG_GETARG_FLOAT4(0);
	TimestampTz docTime = PG_GETARG_TIMESTAMPTZ(1);
	float4		decayRate = PG_GETARG_FLOAT4(2);
	float4		recencyWeight = PG_GETARG_FLOAT4(3);
	TemporalConfig config;
	float4		score;

	config.decayRate = decayRate;
	config.recencyWeight = recencyWeight;
	config.referenceTime = GetCurrentTimestamp();
	config.timeWindow = NULL;
	config.enabled = true;

	score = temporal_compute_hybrid_score(vectorDistance, docTime, &config);

	PG_RETURN_FLOAT4(score);
}

/*
 * SQL-callable function: temporal_filter(timestamp, window_interval)
 */
PG_FUNCTION_INFO_V1(neurondb_temporal_filter);

Datum
neurondb_temporal_filter(PG_FUNCTION_ARGS)
{
	TimestampTz docTime = PG_GETARG_TIMESTAMPTZ(0);
	Interval   *window = PG_GETARG_INTERVAL_P(1);
	TemporalConfig config;
	bool		inWindow;

	config.referenceTime = GetCurrentTimestamp();
	config.timeWindow = window;
	config.enabled = true;

	inWindow = temporal_in_window(docTime, &config);

	PG_RETURN_BOOL(inWindow);
}

/*
 * Create default temporal config
 */
static TemporalConfig *
temporal_create_config(float4 decayRate, float4 recencyWeight)
{
	TemporalConfig *config;

	config = (TemporalConfig *) palloc0(sizeof(TemporalConfig));
	NDB_CHECK_ALLOC(config, "config");
	config->decayRate = decayRate;
	config->recencyWeight = recencyWeight;
	config->referenceTime = GetCurrentTimestamp();
	config->timeWindow = NULL;
	config->enabled = true;

	return config;
}

/*
 * Integrate temporal scoring into HNSW search
 *
 * This would be called from the HNSW search function to apply
 * temporal boosting to results.
 */
void
temporal_integrate_hnsw_search(Relation heapRel,
							   ItemPointer * items,
							   float4 * distances,
							   int resultCount,
							   float4 decayRate,
							   float4 recencyWeight,
							   const char *timestampColumnName)
{
	TimestampTz *timestamps;
	TemporalConfig *config;
	Snapshot	snapshot;
	TupleDesc	tupdesc;
	int			timestamp_attnum = -1;
	int			i;

	if (resultCount == 0 || items == NULL)
		return;

	/* Allocate timestamp array */
	timestamps = (TimestampTz *) palloc(resultCount * sizeof(TimestampTz));
	NDB_CHECK_ALLOC(timestamps, "timestamps");

	/* Get snapshot and tuple descriptor */
	snapshot = GetActiveSnapshot();
	if (snapshot == NULL)
		snapshot = GetTransactionSnapshot();
	tupdesc = RelationGetDescr(heapRel);

	/* Find timestamp column */
	if (timestampColumnName != NULL && strlen(timestampColumnName) > 0)
	{
		timestamp_attnum = SPI_fnumber(tupdesc, timestampColumnName);
		if (timestamp_attnum == SPI_ERROR_NOATTRIBUTE)
		{
			/* Column not found - use default timestamp */
			timestamp_attnum = -1;
		}
	}

	/* Fetch timestamps from heap tuples */
	for (i = 0; i < resultCount; i++)
	{
		HeapTupleData tupleData;
		HeapTuple	tuple = &tupleData;
		Buffer		buffer;
		bool		isnull;
		Datum		datum;
		bool		found;

		if (!ItemPointerIsValid(items[i]))
		{
			timestamps[i] = GetCurrentTimestamp();
			continue;
		}

		/* Fetch heap tuple */
		ItemPointerCopy(items[i], &tupleData.t_self);
		found = heap_fetch(heapRel, snapshot, tuple, &buffer, false);
		if (!found || !HeapTupleIsValid(tuple))
		{
			/* Tuple not found or deleted - use current timestamp */
			timestamps[i] = GetCurrentTimestamp();
			continue;
		}

		/* Extract timestamp column */
		if (timestamp_attnum > 0)
		{
			Oid			atttype;

			datum = heap_getattr(tuple, timestamp_attnum, tupdesc, &isnull);
			if (!isnull)
			{
				atttype = TupleDescAttr(tupdesc, timestamp_attnum - 1)->atttypid;

				/* Convert to TimestampTz */
				if (atttype == TIMESTAMPTZOID)
				{
					timestamps[i] = DatumGetTimestampTz(datum);
				}
				else if (atttype == TIMESTAMPOID)
				{
					Timestamp	ts = DatumGetTimestamp(datum);

					timestamps[i] = DatumGetTimestampTz(DirectFunctionCall1(timestamp_timestamptz, TimestampGetDatum(ts)));
				}
				else
				{
					/* Try to convert */
					timestamps[i] = GetCurrentTimestamp();
				}
			}
			else
			{
				timestamps[i] = GetCurrentTimestamp();
			}
		}
		else
		{
			/* No timestamp column specified - use current timestamp */
			timestamps[i] = GetCurrentTimestamp();
		}
	}

	/* Create config */
	config = temporal_create_config(decayRate, recencyWeight);

	/* Rerank results with temporal scoring */
	temporal_rerank_results(items, distances, timestamps, resultCount, config);

	NDB_FREE(timestamps);
	NDB_FREE(config);
}

/*
 * Helper: Extract timestamp from heap tuple
 */
static TimestampTz
pg_attribute_unused() temporal_get_tuple_timestamp(HeapTuple tuple,
												   TupleDesc tupdesc,
												   const char *columnName)
{
	bool		isnull;
	Datum		datum;
	int			attnum;

	/* Find timestamp column */
	attnum = SPI_fnumber(tupdesc, columnName);
	if (attnum == SPI_ERROR_NOATTRIBUTE)
		return GetCurrentTimestamp();	/* Fallback to now */

	datum = SPI_getbinval(tuple, tupdesc, attnum, &isnull);
	if (isnull)
		return GetCurrentTimestamp();

	return DatumGetTimestampTz(datum);
}
