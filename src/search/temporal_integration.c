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
 * Copyright (c) 2024-2026, neurondb, Inc.
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

#define DEFAULT_DECAY_RATE 0.1
#define DEFAULT_RECENCY_WEIGHT 0.3

typedef struct TemporalConfig
{
	float4		decayRate;
	float4		recencyWeight;
	TimestampTz referenceTime;
	Interval   *timeWindow;
	bool		enabled;
}			TemporalConfig;

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

	if (item_b->score > item_a->score)
		return 1;
	if (item_b->score < item_a->score)
		return -1;
	return 0;
}

static float4
compute_time_decay(TimestampTz docTime, TimestampTz refTime, float4 decayRate)
{
	float8		age_seconds;
	float8		age_days;
	float4		decay;

	age_seconds = (float8) (refTime - docTime) / USECS_PER_SEC;
	age_days = age_seconds / (24.0 * 3600.0);

	decay = (float4) exp(-decayRate * age_days);

	return decay;
}

static float4
temporal_compute_hybrid_score(float4 vectorDistance,
							  TimestampTz docTime,
							  TemporalConfig * config)
{
	float4		vectorScore;
	float4		temporalScore;
	float4		finalScore;

	if (!config->enabled)
		return 1.0 / (1.0 + vectorDistance);

	vectorScore = 1.0 / (1.0 + vectorDistance);

	temporalScore = compute_time_decay(
									   docTime, config->referenceTime, config->decayRate);

	finalScore = (1.0 - config->recencyWeight) * vectorScore
		+ config->recencyWeight * temporalScore;

	return finalScore;
}

static bool
temporal_in_window(TimestampTz docTime, TemporalConfig * config)
{
	TimestampTz windowStart;

	if (!config->enabled || config->timeWindow == NULL)
		return true;

	if (config->timeWindow != NULL)
	{
		int64		interval_usec;

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

static void
temporal_rerank_results(ItemPointer * items,
						float4 * distances,
						TimestampTz * timestamps,
						int count,
						TemporalConfig * config)
{
	float4	   *scores = NULL;
	int			i,
				j;
	float4		temp_score,
				temp_dist;
	ItemPointer temp_item;
	TimestampTz temp_ts;

	if (!config->enabled || count == 0)
		return;

	nalloc(scores, float4, count);
	NDB_CHECK_ALLOC(scores, "scores");

	for (i = 0; i < count; i++)
	{
		scores[i] = temporal_compute_hybrid_score(
												  distances[i], timestamps[i], config);
	}

	if (count <= 50)
	{
		for (i = 0; i < count - 1; i++)
		{
			for (j = i + 1; j < count; j++)
			{
				if (scores[j] > scores[i])
				{
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
		TemporalItem *sort_items;
		int			idx;

		nalloc(sort_items, TemporalItem, count);
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

		pfree(sort_items);
	}

	pfree(scores);

}

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

static TemporalConfig *
temporal_create_config(float4 decayRate, float4 recencyWeight)
{
	TemporalConfig *config = NULL;

	nalloc(config, TemporalConfig, 1);
	MemSet(config, 0, sizeof(TemporalConfig));
	NDB_CHECK_ALLOC(config, "config");
	config->decayRate = decayRate;
	config->recencyWeight = recencyWeight;
	config->referenceTime = GetCurrentTimestamp();
	config->timeWindow = NULL;
	config->enabled = true;

	return config;
}

void
temporal_integrate_hnsw_search(Relation heapRel,
							   ItemPointer * items,
							   float4 * distances,
							   int resultCount,
							   float4 decayRate,
							   float4 recencyWeight,
							   const char *timestampColumnName)
{
	TimestampTz *timestamps = NULL;
	TemporalConfig *config = NULL;
	Snapshot	snapshot;
	TupleDesc	tupdesc;
	int			timestamp_attnum = -1;
	int			i;

	if (resultCount == 0 || items == NULL)
		return;

	nalloc(timestamps, TimestampTz, resultCount);
	NDB_CHECK_ALLOC(timestamps, "timestamps");

	snapshot = GetActiveSnapshot();
	if (snapshot == NULL)
		snapshot = GetTransactionSnapshot();
	tupdesc = RelationGetDescr(heapRel);

	if (timestampColumnName != NULL && strlen(timestampColumnName) > 0)
	{
		timestamp_attnum = SPI_fnumber(tupdesc, timestampColumnName);
		if (timestamp_attnum == SPI_ERROR_NOATTRIBUTE)
		{
			timestamp_attnum = -1;
		}
	}

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

		ItemPointerCopy(items[i], &tupleData.t_self);
		found = heap_fetch(heapRel, snapshot, tuple, &buffer, false);
		if (!found || !HeapTupleIsValid(tuple))
		{
			timestamps[i] = GetCurrentTimestamp();
			continue;
		}

		if (timestamp_attnum > 0)
		{
			Oid			atttype;

			datum = heap_getattr(tuple, timestamp_attnum, tupdesc, &isnull);
			if (!isnull)
			{
				atttype = TupleDescAttr(tupdesc, timestamp_attnum - 1)->atttypid;

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
			timestamps[i] = GetCurrentTimestamp();
		}
	}

	config = temporal_create_config(decayRate, recencyWeight);

	temporal_rerank_results(items, distances, timestamps, resultCount, config);

	pfree(timestamps);
	pfree(config);
}

static TimestampTz
pg_attribute_unused() temporal_get_tuple_timestamp(HeapTuple tuple,
												   TupleDesc tupdesc,
												   const char *columnName)
{
	bool		isnull;
	Datum		datum;
	int			attnum;

	attnum = SPI_fnumber(tupdesc, columnName);
	if (attnum == SPI_ERROR_NOATTRIBUTE)
		return GetCurrentTimestamp();

	datum = SPI_getbinval(tuple, tupdesc, attnum, &isnull);
	if (isnull)
		return GetCurrentTimestamp();

	return DatumGetTimestampTz(datum);
}
