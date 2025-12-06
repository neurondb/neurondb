/*-------------------------------------------------------------------------
 *
 * neurondb_spi_safe.c
 *    Safe SPI execution wrappers for NeuronDB crash prevention
 *
 * Provides safe SPI execution with automatic context management
 * to prevent crashes from:
 * - SPI_execute failures without error handling
 * - Accessing SPI_tuptable after context cleanup
 * - Not copying data out of SPI context before SPI_finish()
 * - SPI context cleanup issues
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/util/neurondb_spi_safe.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "utils/memutils.h"
#include "utils/elog.h"
#include "utils/builtins.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "utils/jsonb.h"
#include "utils/varlena.h"
#include "utils/array.h"
#include "utils/lsyscache.h"

#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

static Datum
datumCopy(Datum value, int typlen, bool typbyval)
{
	if (typbyval || typlen == -1)
		return value;
	else
		return PointerGetDatum(PG_DETOAST_DATUM_COPY(value));
}

int
ndb_spi_execute_safe(const char *query, bool read_only, long tcount)
{
	int			ret;
	MemoryContext save_context;

	if (query == NULL)
	{
		elog(ERROR,
			 "neurondb: ndb_spi_execute_safe: query cannot be NULL");
		return -1;
	}

	if (SPI_processed == -1)
	{
		int			connect_ret = SPI_connect();

		if (connect_ret != SPI_OK_CONNECT)
		{
			elog(ERROR,
				 "neurondb: SPI not connected and SPI_connect() failed with return code %d",
				 connect_ret);
			return -1;
		}
		elog(DEBUG1, "neurondb: ndb_spi_execute_safe auto-connected SPI");
	}

	/*
	 * Save current memory context before PG_TRY block. If an error occurs,
	 * the catch handler needs a valid context to allocate error data. The
	 * saved context must not be ErrorContext since that context cannot
	 * be used for allocations during error processing.
	 */
	save_context = CurrentMemoryContext;

	PG_TRY();
	{
		ret = SPI_execute(query, read_only ? 1 : 0, tcount);

		/*
		 * Handle SPI_ERROR_UNCONNECTED by attempting to connect and retry.
		 * This can occur if SPI connection was lost between calls or if
		 * the initial connection check above was insufficient. We retry
		 * the query after connecting to avoid failing on transient
		 * connection state issues.
		 */
		if (ret == SPI_ERROR_UNCONNECTED)
		{
			elog(DEBUG1, "neurondb: SPI_execute returned SPI_ERROR_UNCONNECTED, attempting to connect and retry");
			if (SPI_connect() == SPI_OK_CONNECT)
			{
				ret = SPI_execute(query, read_only ? 1 : 0, tcount);
				elog(DEBUG1, "neurondb: retry after SPI_connect returned %d", ret);
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: SPI_execute returned SPI_ERROR_UNCONNECTED and SPI_connect() failed"),
						 errdetail("Query: %s", query ? query : "(NULL)")));
			}
		}

		if (ret == SPI_OK_SELECT || ret == SPI_OK_SELINTO ||
			ret == SPI_OK_INSERT_RETURNING || ret == SPI_OK_UPDATE_RETURNING ||
			ret == SPI_OK_DELETE_RETURNING)
		{
			NDB_CHECK_SPI_TUPTABLE();
		}
		if (ret < 0)
		{
			const char *error_msg = "unknown SPI error";

			switch (ret)
			{
				case SPI_ERROR_UNCONNECTED:
					error_msg = "SPI not connected";
					break;
				case SPI_ERROR_COPY:
					error_msg = "COPY command in progress";
					break;
				case SPI_ERROR_TRANSACTION:
					error_msg = "transaction state error";
					break;
				case SPI_ERROR_ARGUMENT:
					error_msg = "invalid argument to SPI_execute";
					break;
				case SPI_ERROR_OPUNKNOWN:
					error_msg = "unknown operation";
					break;
			}
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: SPI_execute returned error code %d: %s", ret, error_msg),
					 errdetail("Query: %s", query ? query : "(NULL)")));
		}
	}
	PG_CATCH();
	{
		ErrorData  *edata;

		/*
		 * Select a safe memory context for error handling. ErrorContext
		 * cannot be used for allocations during error processing. If the
		 * saved context is ErrorContext or NULL, fall back to TopMemoryContext
		 * which is always valid and persists for the session lifetime.
		 */
		MemoryContext safe_context = save_context;

		if (safe_context == ErrorContext || safe_context == NULL)
		{
			safe_context = TopMemoryContext;
		}

		/*
		 * Switch to the safe context before allocating error data. This
		 * ensures CopyErrorData and subsequent allocations succeed even
		 * if the original context was destroyed or is invalid.
		 */
		MemoryContextSwitchTo(safe_context);

		if (CurrentMemoryContext != ErrorContext)
		{
			edata = CopyErrorData();
			FlushErrorState();

			ereport(ERROR,
					(errcode(edata->sqlerrcode),
					 errmsg("%s", edata->message ? edata->message : "SPI_execute failed"),
					 errdetail("Query: %s%s",
							   query ? query : "(NULL)",
							   edata->detail ? edata->detail : ""),
					 errhint("%s", edata->hint ? edata->hint : "")));
			FreeErrorData(edata);
		}
		else
		{
			FlushErrorState();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: SPI_execute failed for query: %s", query ? query : "(NULL)")));
		}
		ret = -1;
	}
	PG_END_TRY();

	return ret;
}

bool
ndb_spi_execute_and_validate(const char *query,
							 bool read_only,
							 long tcount,
							 int expected_ret,
							 long min_rows)
{
	int			ret;

	ret = ndb_spi_execute_safe(query, read_only, tcount);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret < 0)
		return false;

	if (ret != expected_ret)
	{
		const char *err_msg;

		switch (ret)
		{
			case SPI_ERROR_UNCONNECTED:
				err_msg = "SPI not connected";
				break;
			case SPI_ERROR_COPY:
				err_msg = "COPY command in progress";
				break;
			case SPI_ERROR_TRANSACTION:
				err_msg = "transaction state error";
				break;
			case SPI_ERROR_ARGUMENT:
				err_msg = "invalid argument";
				break;
			case SPI_ERROR_OPUNKNOWN:
				err_msg = "unknown operation";
				break;
			default:
				err_msg = "unknown SPI error";
				break;
		}

		elog(ERROR,
			 "neurondb: SPI operation failed: %s (got %d, expected %d)",
			 err_msg,
			 ret,
			 expected_ret);
		return false;
	}

	if (min_rows >= 0 && SPI_processed < min_rows)
	{
		elog(ERROR,
			 "neurondb: query returned %ld rows, expected at least %ld",
			 (long) SPI_processed,
			 min_rows);
		return false;
	}

	return true;
}

/*
 * ndb_spi_exec_select_one_row_safe - Execute SELECT and copy single row result
 */
bool
ndb_spi_exec_select_one_row_safe(const char *query,
								 bool read_only,
								 MemoryContext dest_context,
								 TupleDesc * out_tupdesc,
								 Datum * *out_datum,
								 bool **out_isnull,
								 int *out_natts)
{
	MemoryContext oldcontext;
	int			ret;
	int			i;
	TupleDesc	temp_tupdesc;
	int			natts;

	if (query == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: query cannot be NULL")));
		return false;
	}

	/* Check SPI connection */
	if (SPI_processed == -1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI not connected - call SPI_connect() first")));
		return false;
	}

	if (dest_context == NULL)
		dest_context = CurrentMemoryContext;
	else
		oldcontext = MemoryContextSwitchTo(dest_context);

	PG_TRY();
	{
		ret = ndb_spi_execute_safe(query, read_only, 0);
		if (ret == SPI_OK_SELECT)
			NDB_CHECK_SPI_TUPTABLE();
	}
	PG_CATCH();
	{
		if (dest_context != CurrentMemoryContext)
			MemoryContextSwitchTo(oldcontext);
		FlushErrorState();
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_execute failed for query: %s", query)));
		return false;
	}
	PG_END_TRY();

	if (ret != SPI_OK_SELECT)
	{
		if (dest_context != CurrentMemoryContext)
			MemoryContextSwitchTo(oldcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI query did not return SPI_OK_SELECT (got %d)", ret)));
		return false;
	}

	if (SPI_processed != 1)
	{
		if (dest_context != CurrentMemoryContext)
			MemoryContextSwitchTo(oldcontext);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: query returned %ld rows, expected exactly 1",
						(long) SPI_processed)));
		return false;
	}

	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL ||
		SPI_tuptable->vals == NULL || SPI_tuptable->vals[0] == NULL)
	{
		if (dest_context != CurrentMemoryContext)
			MemoryContextSwitchTo(oldcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: SPI_tuptable is NULL or invalid")));
		return false;
	}

	temp_tupdesc = SPI_tuptable->tupdesc;
	natts = temp_tupdesc->natts;

	/* Switch to destination context for copying */
	oldcontext = MemoryContextSwitchTo(dest_context);

	/* Copy tuple descriptor to destination context */
	if (out_tupdesc != NULL)
		*out_tupdesc = CreateTupleDescCopy(temp_tupdesc);

	if (out_datum != NULL || out_isnull != NULL)
	{
		if (out_datum != NULL)
			*out_datum = (Datum *) palloc(sizeof(Datum) * natts);
		if (out_isnull != NULL)
			*out_isnull = (bool *) palloc(sizeof(bool) * natts);

		for (i = 0; i < natts; i++)
		{
			Datum		temp_datum;
			bool		temp_isnull;
			Oid			type;

			temp_datum = SPI_getbinval(SPI_tuptable->vals[0],
									   temp_tupdesc,
									   i + 1,
									   &temp_isnull);

			if (!temp_isnull)
			{
				type = SPI_gettypeid(temp_tupdesc, i + 1);
				if (out_datum != NULL)
					(*out_datum)[i] = datumCopy(temp_datum,
												get_typlen(type),
												get_typbyval(type));
			}
			else
			{
				if (out_datum != NULL)
					(*out_datum)[i] = (Datum) 0;
			}
			if (out_isnull != NULL)
				(*out_isnull)[i] = temp_isnull;
		}
	}

	if (out_natts != NULL)
		*out_natts = natts;

	SPI_finish();

	if (dest_context != CurrentMemoryContext)
		MemoryContextSwitchTo(oldcontext);

	return true;
}

/*
 * ndb_spi_get_result_safe - Safely extract result from SPI_tuptable
 *
 * Validates SPI_tuptable before access and extracts a single value.
 * Returns true on success, false on failure.
 *
 * *out_datum and *out_isnull are set on success.
 */
bool
ndb_spi_get_result_safe(int row_idx,
						int col_idx,
						Oid * out_type,
						Datum * out_datum,
						bool *out_isnull)
{
	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL)
	{
		elog(ERROR,
			 "neurondb: SPI_tuptable is NULL or invalid");
		return false;
	}

	if (row_idx < 0 || row_idx >= SPI_processed)
	{
		elog(ERROR,
			 "neurondb: row index %d out of bounds (SPI_processed=%ld)",
			 row_idx,
			 (long) SPI_processed);
		return false;
	}

	if (col_idx < 1 || col_idx > SPI_tuptable->tupdesc->natts)
	{
		elog(ERROR,
			 "neurondb: column index %d out of bounds (natts=%d)",
			 col_idx,
			 SPI_tuptable->tupdesc->natts);
		return false;
	}

	if (SPI_tuptable->vals[row_idx] == NULL)
	{
		elog(ERROR,
			 "neurondb: SPI_tuptable->vals[%d] is NULL",
			 row_idx);
		return false;
	}

	if (out_type != NULL)
		*out_type = SPI_gettypeid(SPI_tuptable->tupdesc, col_idx);

	if (out_datum != NULL && out_isnull != NULL)
	{
		*out_datum = SPI_getbinval(SPI_tuptable->vals[row_idx],
								   SPI_tuptable->tupdesc,
								   col_idx,
								   out_isnull);
	}

	return true;
}

/*
 * ndb_spi_get_jsonb_safe - Safely extract JSONB from SPI result
 */
Jsonb *
ndb_spi_get_jsonb_safe(int row_idx, int col_idx, MemoryContext dest_context)
{
	MemoryContext oldcontext;
	Datum		result_datum;
	bool		result_isnull;
	Jsonb	   *temp_jsonb;
	Jsonb	   *result = NULL;
	bool		success;

	if (dest_context == NULL)
		dest_context = CurrentMemoryContext;

	oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);

	/* Get datum from SPI result */
	success = ndb_spi_get_result_safe(row_idx, col_idx, NULL, &result_datum, &result_isnull);
	if (!success)
	{
		MemoryContextSwitchTo(oldcontext);
		return NULL;
	}

	if (result_isnull)
	{
		MemoryContextSwitchTo(oldcontext);
		return NULL;
	}

	if (DatumGetPointer(result_datum) == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		elog(ERROR,
			 "neurondb: SPI result datum pointer is NULL");
		return NULL;
	}

	temp_jsonb = DatumGetJsonbP(result_datum);

	if (temp_jsonb == NULL || VARSIZE(temp_jsonb) < sizeof(Jsonb))
	{
		MemoryContextSwitchTo(oldcontext);
		elog(ERROR,
			 "neurondb: invalid JSONB structure in SPI result");
		return NULL;
	}

	MemoryContextSwitchTo(dest_context);
	result = (Jsonb *) PG_DETOAST_DATUM_COPY((Datum) temp_jsonb);

	if (result == NULL || VARSIZE(result) < sizeof(Jsonb))
	{
		if (result != NULL)
			NDB_FREE(result);
		MemoryContextSwitchTo(oldcontext);
		elog(ERROR,
			 "neurondb: JSONB copy validation failed");
		return NULL;
	}

	MemoryContextSwitchTo(oldcontext);
	return result;
}

/*
 * ndb_spi_get_text_safe - Safely extract text from SPI result
 */
text *
ndb_spi_get_text_safe(int row_idx, int col_idx, MemoryContext dest_context)
{
	MemoryContext oldcontext;
	Datum		result_datum;
	bool		result_isnull;
	text	   *temp_text;
	text	   *result = NULL;
	bool		success;

	if (dest_context == NULL)
		dest_context = CurrentMemoryContext;

	oldcontext = MemoryContextSwitchTo(CurrentMemoryContext);

	/* Get datum from SPI result */
	success = ndb_spi_get_result_safe(row_idx, col_idx, NULL, &result_datum, &result_isnull);
	if (!success)
	{
		MemoryContextSwitchTo(oldcontext);
		return NULL;
	}

	if (result_isnull)
	{
		MemoryContextSwitchTo(oldcontext);
		return NULL;
	}

	temp_text = DatumGetTextP(result_datum);

	if (temp_text == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		elog(ERROR,
			 "neurondb: text pointer is NULL in SPI result");
		return NULL;
	}

	MemoryContextSwitchTo(dest_context);
	result = (text *) PG_DETOAST_DATUM_COPY((Datum) temp_text);

	if (result == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		elog(ERROR,
			 "neurondb: text copy failed");
		return NULL;
	}

	MemoryContextSwitchTo(oldcontext);
	return result;
}

/*
 * ndb_spi_finish_safe - Safely finish SPI connection
 *
 * Validates SPI is connected before finishing, and ensures
 * we're not in SPI context before cleanup.
 */
void
ndb_spi_finish_safe(MemoryContext oldcontext)
{
	if (SPI_processed == -1)
	{
		return;
	}

	if (oldcontext != NULL)
		MemoryContextSwitchTo(oldcontext);

	SPI_finish();
}

/*
 * ndb_spi_cleanup_safe - Comprehensive SPI cleanup
 */
void
ndb_spi_cleanup_safe(MemoryContext oldcontext, MemoryContext callcontext, bool finish_spi)
{
	if (finish_spi)
		ndb_spi_finish_safe(oldcontext);

	if (oldcontext != NULL)
		MemoryContextSwitchTo(oldcontext);

	if (callcontext != NULL && callcontext != oldcontext)
		MemoryContextDelete(callcontext);
}

/*
 * ndb_spi_iterate_safe - Safely iterate over SPI results
 */
int
ndb_spi_iterate_safe(bool (*callback) (int row_idx, HeapTuple tuple, TupleDesc tupdesc, void *userdata),
					 void *userdata)
{
	int			i;
	int			processed = 0;

	if (callback == NULL)
	{
		elog(ERROR,
			 "neurondb: ndb_spi_iterate_safe: callback cannot be NULL");
		return -1;
	}

	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL)
	{
		elog(ERROR,
			 "neurondb: SPI_tuptable is NULL or invalid");
		return -1;
	}

	for (i = 0; i < SPI_processed; i++)
	{
		if (SPI_tuptable->vals[i] == NULL)
		{
			elog(WARNING,
				 "neurondb: SPI_tuptable->vals[%d] is NULL, skipping",
				 i);
			continue;
		}

		if (!callback(i, SPI_tuptable->vals[i], SPI_tuptable->tupdesc, userdata))
			break;

		processed++;
	}

	return processed;
}
