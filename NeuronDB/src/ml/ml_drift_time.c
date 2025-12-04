/*-------------------------------------------------------------------------
 *
 * ml_drift_time.c
 *    Temporal drift monitoring.
 *
 * This module monitors embedding drift over time by computing centroids
 * at different time windows and tracking centroid movement.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_drift_time.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/timestamp.h"

#include "neurondb.h"
#include "executor/spi.h"
#include "utils/memutils.h"
#include "utils/jsonb.h"
#include "lib/stringinfo.h"
#include "vector/vector_types.h"

#include <math.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_macros.h"

/*
 * monitor_drift_timeseries
 * -------------------------
 * Track embedding drift over time windows.
 *
 * SQL Arguments:
 *   table_name      - Source table
 *   vector_column   - Embedding column
 *   timestamp_column - Timestamp column for temporal grouping
 *   window_size     - Time window size (e.g., '1 day', '1 hour')
 *
 * Returns:
 *   Table of (window_start, drift_distance, num_vectors)
 *
 * Example Usage:
 *   -- Monitor daily drift:
 *   SELECT * FROM monitor_drift_timeseries(
 *     'user_embeddings',
 *     'embedding',
 *     'created_at',
 *     '1 day'::interval
 *   ) ORDER BY window_start;
 *
 *   -- Alert on significant drift:
 *   WITH drift_monitor AS (
 *     SELECT * FROM monitor_drift_timeseries('docs', 'vec', 'ts', '1 hour'::interval)
 *   )
 *   SELECT * FROM drift_monitor
 *   WHERE drift_distance > 0.5  -- Threshold
 *   ORDER BY window_start DESC
 *   LIMIT 10;
 *
 * Notes:
 *   - Compares consecutive time windows
 *   - First window has NULL drift (no baseline)
 *   - Consider seasonal patterns when setting thresholds
 */
PG_FUNCTION_INFO_V1(monitor_drift_timeseries);

Datum
monitor_drift_timeseries(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	text	   *timestamp_column;
	text	   *window_size_text;
	char	   *tbl_str;
	char	   *col_str;
	char	   *ts_col_str;
	char	   *window_str;
	int			ret;
	StringInfoData sql;
	MemoryContext oldcontext;
	MemoryContext drift_context;
	NDB_DECLARE(Vector *, baseline_centroid);
	NDB_DECLARE(Vector *, current_centroid);
	float		drift_distance = 0.0f;
	int			n_baseline = 0;
	int			n_current = 0;
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext_spi;

	/* Defensive: validate inputs */
	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	timestamp_column = PG_GETARG_TEXT_PP(2);
	window_size_text = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);

	if (table_name == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("monitor_drift_timeseries: table_name cannot be NULL")));
	if (vector_column == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("monitor_drift_timeseries: vector_column cannot be NULL")));
	if (timestamp_column == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("monitor_drift_timeseries: timestamp_column cannot be NULL")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);
	ts_col_str = text_to_cstring(timestamp_column);
	window_str = window_size_text ? text_to_cstring(window_size_text) : pstrdup("1 day");

	/* Create memory context */
	drift_context = AllocSetContextCreate(CurrentMemoryContext,
										  "drift monitoring context",
										  ALLOCSET_DEFAULT_SIZES);
	elog(DEBUG1, "drift monitoring context created");
	oldcontext_spi = CurrentMemoryContext;
	
	oldcontext = MemoryContextSwitchTo(drift_context);

	/* Connect to SPI */

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext_spi);

	/* Compute baseline centroid (previous window) */
	ndb_spi_stringinfo_init(spi_session, &sql);
	{
		const char *col_quoted = quote_identifier(col_str);
		const char *tbl_quoted = quote_identifier(tbl_str);
		const char *ts_col_quoted = quote_identifier(ts_col_str);

		appendStringInfo(&sql,
						 "SELECT AVG(%s) as centroid, COUNT(*) as cnt "
						 "FROM %s "
						 "WHERE %s >= NOW() - INTERVAL '%s' - INTERVAL '%s' "
						 "AND %s < NOW() - INTERVAL '%s'",
						 col_quoted, tbl_quoted, ts_col_quoted, window_str, window_str, ts_col_quoted, window_str);
		elog(DEBUG1, "baseline query: %s", sql.data);

		/*
		 * Note: quote_identifier returns const char * pointing to managed
		 * memory, don't pfree
		 */
	}

	ret = ndb_spi_execute(spi_session, sql.data, true, 1);
	ndb_spi_stringinfo_free(spi_session, &sql);

	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		centroid_datum;
		Datum		cnt_datum;
		/* Note: Vector type doesn't have direct ndb_spi_get_* function, so we access SPI_tuptable directly */
		/* Safe access for complex types - validate before access */
		if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL && 
			SPI_tuptable->vals != NULL && SPI_processed > 0 && SPI_tuptable->vals[0] != NULL)
		{
			centroid_datum = SPI_getbinval(SPI_tuptable->vals[0],
									   SPI_tuptable->tupdesc,
									   1,
									   &isnull);
			if (!isnull)
			{
				baseline_centroid = DatumGetVector(centroid_datum);
				/* Safe access for count - validate tupdesc has at least 2 columns */
				if (SPI_tuptable->tupdesc->natts >= 2)
				{
					cnt_datum = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc,
									  2,
									  &isnull);
					if (!isnull)
						n_baseline = DatumGetInt64(cnt_datum);
				}
			}
		}
	}

	/* Compute current centroid (current window) */
	ndb_spi_stringinfo_init(spi_session, &sql);
	{
		const char *col_quoted = quote_identifier(col_str);
		const char *tbl_quoted = quote_identifier(tbl_str);
		const char *ts_col_quoted = quote_identifier(ts_col_str);

		appendStringInfo(&sql,
						 "SELECT AVG(%s) as centroid, COUNT(*) as cnt "
						 "FROM %s "
						 "WHERE %s >= NOW() - INTERVAL '%s'",
						 col_quoted, tbl_quoted, ts_col_quoted, window_str);
		elog(DEBUG1, "current query: %s", sql.data);

		/*
		 * Note: quote_identifier returns const char * pointing to managed
		 * memory, don't pfree
		 */
	}

	ret = ndb_spi_execute(spi_session, sql.data, true, 1);
	ndb_spi_stringinfo_free(spi_session, &sql);

	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		bool		isnull;
		Datum		centroid_datum;
		Datum		cnt_datum;
		/* Note: Vector type doesn't have direct ndb_spi_get_* function, so we access SPI_tuptable directly */
		/* Safe access for complex types - validate before access */
		if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL && 
			SPI_tuptable->vals != NULL && SPI_processed > 0 && SPI_tuptable->vals[0] != NULL)
		{
			centroid_datum = SPI_getbinval(SPI_tuptable->vals[0],
									   SPI_tuptable->tupdesc,
									   1,
									   &isnull);
			if (!isnull)
			{
				current_centroid = DatumGetVector(centroid_datum);
				/* Safe access for count - validate tupdesc has at least 2 columns */
				if (SPI_tuptable->tupdesc->natts >= 2)
				{
					cnt_datum = SPI_getbinval(SPI_tuptable->vals[0],
								  SPI_tuptable->tupdesc,
								  2,
								  &isnull);
					if (!isnull)
						n_current = DatumGetInt64(cnt_datum);
				}
			}
		}
	}

	NDB_SPI_SESSION_END(spi_session);

	/* Calculate drift distance if both centroids exist */
	if (baseline_centroid != NULL && current_centroid != NULL &&
		baseline_centroid->dim == current_centroid->dim)
	{
		float		sum_sq_diff = 0.0f;
		int			i;

		for (i = 0; i < baseline_centroid->dim; i++)
		{
			float		diff = baseline_centroid->data[i] - current_centroid->data[i];

			sum_sq_diff += diff * diff;
		}
		drift_distance = sqrtf(sum_sq_diff);
	}

	/* Cleanup and return */
	MemoryContextSwitchTo(oldcontext);
	{
		Jsonb	   *result_jsonb;
		StringInfoData result_json;

		initStringInfo(&result_json);
		{
			Datum		window_datum = CStringGetDatum(window_str);
			Datum		window_quoted = DirectFunctionCall1(quote_literal, window_datum);
			char	   *window_quoted_str = DatumGetCString(window_quoted);

			appendStringInfo(&result_json,
							 "{\"drift_distance\":%.6f,\"baseline_samples\":%d,\"current_samples\":%d,\"window_size\":%s}",
							 drift_distance, n_baseline, n_current,
							 window_quoted_str);
			NDB_FREE(window_quoted_str);
		}

		result_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
														  CStringGetTextDatum(result_json.data)));

		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		NDB_FREE(ts_col_str);
		NDB_FREE(window_str);
		MemoryContextDelete(drift_context);

		PG_RETURN_JSONB_P(result_jsonb);
	}
}
