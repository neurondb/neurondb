/*-------------------------------------------------------------------------
 *
 * neurondb_spi.c
 *    Centralized SPI session management for NeuronDB
 *
 * Provides a unified interface for all SPI operations with automatic:
 * - Connection state tracking (nested SPI support)
 * - Memory context management
 * - Error handling
 * - StringInfoData management in correct contexts
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/util/neurondb_spi.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "lib/stringinfo.h"
#include "utils/memutils.h"
#include "utils/builtins.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "utils/jsonb.h"
#include "utils/varlena.h"

#include "neurondb_spi.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

struct NdbSpiSession
{
	bool		we_connected_spi;	/* Did we call SPI_connect()? */
	MemoryContext parent_context;	/* Context before SPI_connect() */
	MemoryContext spi_context;	/* SPI context (if connected) */
};

NdbSpiSession *
ndb_spi_session_begin(MemoryContext parent_context, bool assume_spi_connected)
{
	NDB_DECLARE(NdbSpiSession *, session);
	MemoryContext oldcontext;

	if (parent_context == NULL)
		parent_context = CurrentMemoryContext;

	oldcontext = MemoryContextSwitchTo(parent_context);
	NDB_ALLOC(session, NdbSpiSession, 1);

	MemoryContextSwitchTo(oldcontext);

	session->parent_context = parent_context;

	if (assume_spi_connected)
	{
		session->we_connected_spi = false;
		session->spi_context = CurrentMemoryContext;
		elog(DEBUG1, "neurondb: SPI session: assuming SPI already connected");
	}
	else
	{
		if (SPI_connect() != SPI_OK_CONNECT)
		{
			NDB_FREE(session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: SPI_connect failed in ndb_spi_session_begin")));
		}
		session->we_connected_spi = true;
		session->spi_context = CurrentMemoryContext;
		elog(DEBUG1, "neurondb: SPI session: connected SPI (we_connected=true)");
	}

	return session;
}

void
ndb_spi_session_end(NdbSpiSession **session)
{
	if (session == NULL || *session == NULL)
		return;

	if ((*session)->we_connected_spi)
	{
		if ((*session)->parent_context != NULL)
			MemoryContextSwitchTo((*session)->parent_context);
		SPI_finish();
		elog(DEBUG1, "neurondb: SPI session: finished SPI (we connected it)");
	}
	else
	{
		elog(DEBUG1, "neurondb: SPI session: not finishing SPI (caller connected it)");
	}

	NDB_FREE(*session);
}

bool
ndb_spi_session_controls_connection(NdbSpiSession *session)
{
	if (session == NULL)
		return false;
	return session->we_connected_spi;
}

MemoryContext
ndb_spi_session_get_context(NdbSpiSession *session)
{
	if (session == NULL)
		return CurrentMemoryContext;
	return session->spi_context;
}

int
ndb_spi_execute(NdbSpiSession *session,
				const char *query,
				bool read_only,
				long tcount)
{
	int			ret;
	MemoryContext oldcontext;

	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ndb_spi_execute: session is NULL")));

	if (query == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ndb_spi_execute: query is NULL")));

	oldcontext = MemoryContextSwitchTo(session->spi_context);

	PG_TRY();
	{
		ret = SPI_execute(query, read_only ? 1 : 0, tcount);

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
					 errmsg("neurondb: SPI_execute returned error code %d: %s",
							ret, error_msg),
					 errdetail("Query: %s (SPI code: %d)", query, ret)));
		}
	}
	PG_CATCH();
	{
		MemoryContextSwitchTo(oldcontext);
		PG_RE_THROW();
	}
	PG_END_TRY();

	MemoryContextSwitchTo(oldcontext);
	return ret;
}

/*
 * ndb_spi_execute_with_args - Execute a parameterized SQL query through SPI
 *
 * Executes a parameterized SQL query with arguments through the SPI interface.
 * Similar to ndb_spi_execute but accepts query parameters to avoid SQL injection
 * and improve performance through prepared statement-like behavior.
 *
 * Parameters:
 *   session - SPI session to execute query in
 *   src - SQL query string with parameter placeholders ($1, $2, etc.)
 *   nargs - Number of parameters
 *   argtypes - Array of parameter type OIDs
 *   values - Array of parameter values (as Datum)
 *   nulls - Array indicating which parameters are NULL ('n' for NULL, ' ' for not NULL)
 *   read_only - If true, query is read-only (optimization hint)
 *   tcount - Maximum number of tuples to process (0 = all)
 *
 * Returns:
 *   SPI return code (SPI_OK_* on success, negative on error)
 *
 * Notes:
 *   The function switches to the session's SPI context before execution.
 *   Validates SPI_tuptable for queries that return result sets. Errors
 *   are reported with detailed messages including the query string.
 */
int
ndb_spi_execute_with_args(NdbSpiSession *session,
						  const char *src,
						  int nargs,
						  Oid * argtypes,
						  Datum * values,
						  const char *nulls,
						  bool read_only,
						  long tcount)
{
	int			ret;
	MemoryContext oldcontext;

	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ndb_spi_execute_with_args: session is NULL")));

	if (src == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ndb_spi_execute_with_args: src is NULL")));

	oldcontext = MemoryContextSwitchTo(session->spi_context);

	PG_TRY();
	{
		ret = SPI_execute_with_args(src, nargs, argtypes, values, nulls,
									read_only ? 1 : 0, tcount);

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
					error_msg = "invalid argument to SPI_execute_with_args";
					break;
				case SPI_ERROR_OPUNKNOWN:
					error_msg = "unknown operation";
					break;
			}
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: SPI_execute_with_args returned error code %d: %s",
							ret, error_msg),
					 errdetail("Query: %s (SPI code: %d)", src, ret)));
		}
	}
	PG_CATCH();
	{
		MemoryContextSwitchTo(oldcontext);
		PG_RE_THROW();
	}
	PG_END_TRY();

	MemoryContextSwitchTo(oldcontext);
	return ret;
}

/*
 * ndb_spi_stringinfo_init - Initialize StringInfo in SPI context
 *
 * Initializes a StringInfo structure in the SPI memory context associated
 * with the session. This ensures the string buffer is allocated in the
 * correct context for SPI operations.
 *
 * Parameters:
 *   session - SPI session to use for context
 *   str - StringInfo structure to initialize
 *
 * Notes:
 *   The function switches to the session's SPI context before initialization
 *   and restores the previous context afterward. This ensures proper memory
 *   management for strings used in SPI operations.
 */
void
ndb_spi_stringinfo_init(NdbSpiSession *session, StringInfoData * str)
{
	MemoryContext oldcontext;

	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ndb_spi_stringinfo_init: session is NULL")));

	if (str == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ndb_spi_stringinfo_init: str is NULL")));

	oldcontext = MemoryContextSwitchTo(session->spi_context);
	initStringInfo(str);
	MemoryContextSwitchTo(oldcontext);
}

/*
 * ndb_spi_stringinfo_free - Free StringInfo buffer allocated in SPI context
 *
 * Frees the data buffer of a StringInfo structure that was allocated in
 * the SPI context. Safe to call with NULL pointers.
 *
 * Parameters:
 *   session - SPI session (unused, kept for API consistency)
 *   str - StringInfo structure to free
 *
 * Notes:
 *   The function uses NDB_FREE which is context-aware through chunk headers,
 *   so explicit context switching is not required. Safe to call even if
 *   str or str->data is NULL.
 */
void
ndb_spi_stringinfo_free(NdbSpiSession *session, StringInfoData * str)
{
	if (session == NULL || str == NULL || str->data == NULL)
		return;

	NDB_FREE(str->data);
}

/*
 * ndb_spi_stringinfo_reset - Reset StringInfo and reinitialize in SPI context
 *
 * Frees the existing StringInfo buffer and reinitializes the structure in
 * the SPI context. Useful for reusing a StringInfo structure for multiple
 * operations.
 *
 * Parameters:
 *   session - SPI session to use for context
 *   str - StringInfo structure to reset
 *
 * Notes:
 *   Safe to call with NULL pointers. The function first frees the old buffer,
 *   then reinitializes the structure in the SPI context.
 */
void
ndb_spi_stringinfo_reset(NdbSpiSession *session, StringInfoData * str)
{
	if (session == NULL || str == NULL)
		return;

	ndb_spi_stringinfo_free(session, str);
	ndb_spi_stringinfo_init(session, str);
}

/*
 * ndb_spi_get_int32 - Extract int32 value from SPI result set
 *
 * Extracts an int32 value from a specific row and column in the SPI result
 * set. Handles type conversion for various integer types (int2, int4, int8).
 *
 * Parameters:
 *   session - SPI session (unused, kept for API consistency)
 *   row_idx - Zero-based row index in result set
 *   col_idx - One-based column index in result set
 *   out_value - Output pointer to receive the extracted value
 *
 * Returns:
 *   true if value was successfully extracted, false if row/column out of
 *   bounds, value is NULL, or SPI_tuptable is not available
 *
 * Notes:
 *   The function validates row and column indices before extraction.
 *   Supports automatic type conversion from int2, int4, and int8 to int32.
 *   Returns false if the value is NULL.
 */
bool
ndb_spi_get_int32(NdbSpiSession *session,
				  int row_idx,
				  int col_idx,
				  int32 * out_value)
{
	Datum		datum;
	bool		isnull;
	Oid			type_oid;

	if (session == NULL || out_value == NULL)
		return false;

	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL)
		return false;

	if (row_idx < 0 || row_idx >= SPI_processed)
		return false;

	if (col_idx < 1 || col_idx > SPI_tuptable->tupdesc->natts)
		return false;

	datum = SPI_getbinval(SPI_tuptable->vals[row_idx],
						  SPI_tuptable->tupdesc,
						  col_idx,
						  &isnull);

	if (isnull)
		return false;

	type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, col_idx);

	if (type_oid == INT4OID || type_oid == INT2OID || type_oid == INT8OID)
	{
		if (type_oid == INT4OID)
			*out_value = DatumGetInt32(datum);
		else if (type_oid == INT2OID)
			*out_value = (int32) DatumGetInt16(datum);
		else if (type_oid == INT8OID)
			*out_value = (int32) DatumGetInt64(datum);
	}
	else
	{
		elog(WARNING,
			 "neurondb: ndb_spi_get_int32: unexpected type OID %u (expected integer type)",
			 type_oid);
		return false;
	}

	return true;
}

/*
 * ndb_spi_get_text - Extract text value from SPI result set
 *
 * Extracts a text value from a specific row and column in the SPI result
 * set and copies it to the specified destination memory context.
 *
 * Parameters:
 *   session - SPI session to get parent context from
 *   row_idx - Zero-based row index in result set
 *   col_idx - One-based column index in result set
 *   dest_context - Memory context to allocate result in (NULL uses session's parent context)
 *
 * Returns:
 *   Pointer to text value in dest_context, NULL if row/column out of bounds,
 *   value is NULL, SPI_tuptable is not available, or session is NULL
 *
 * Notes:
 *   The function copies the datum to the destination context to ensure it
 *   remains valid after SPI context cleanup. The result must be freed by
 *   the caller using pfree or NDB_FREE.
 */
text *
ndb_spi_get_text(NdbSpiSession *session,
				 int row_idx,
				 int col_idx,
				 MemoryContext dest_context)
{
	Datum		datum;
	bool		isnull;
	text	   *result = NULL;
	MemoryContext oldcontext;

	if (session == NULL)
		return NULL;

	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL)
		return NULL;

	if (row_idx < 0 || row_idx >= SPI_processed)
		return NULL;

	if (col_idx < 1 || col_idx > SPI_tuptable->tupdesc->natts)
		return NULL;

	if (dest_context == NULL)
		dest_context = session->parent_context;

	datum = SPI_getbinval(SPI_tuptable->vals[row_idx],
						  SPI_tuptable->tupdesc,
						  col_idx,
						  &isnull);

	if (isnull)
		return NULL;

	oldcontext = MemoryContextSwitchTo(dest_context);
	result = (text *) PG_DETOAST_DATUM_COPY(datum);
	MemoryContextSwitchTo(oldcontext);

	return result;
}

/*
 * ndb_spi_get_jsonb - Extract JSONB value from SPI result set
 *
 * Extracts a JSONB value from a specific row and column in the SPI result
 * set and copies it to the specified destination memory context.
 *
 * Parameters:
 *   session - SPI session to get parent context from
 *   row_idx - Zero-based row index in result set
 *   col_idx - One-based column index in result set
 *   dest_context - Memory context to allocate result in (NULL uses session's parent context)
 *
 * Returns:
 *   Pointer to JSONB value in dest_context, NULL if row/column out of bounds,
 *   value is NULL, SPI_tuptable is not available, or session is NULL
 *
 * Notes:
 *   The function copies the datum to the destination context to ensure it
 *   remains valid after SPI context cleanup. The result must be freed by
 *   the caller using pfree or NDB_FREE.
 */
Jsonb *
ndb_spi_get_jsonb(NdbSpiSession *session,
				  int row_idx,
				  int col_idx,
				  MemoryContext dest_context)
{
	Datum		datum;
	bool		isnull;
	Jsonb	   *result = NULL;
	MemoryContext oldcontext;

	if (session == NULL)
		return NULL;

	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL)
		return NULL;

	if (row_idx < 0 || row_idx >= SPI_processed)
		return NULL;

	if (col_idx < 1 || col_idx > SPI_tuptable->tupdesc->natts)
		return NULL;

	if (dest_context == NULL)
		dest_context = session->parent_context;

	datum = SPI_getbinval(SPI_tuptable->vals[row_idx],
						  SPI_tuptable->tupdesc,
						  col_idx,
						  &isnull);

	if (isnull)
		return NULL;

	oldcontext = MemoryContextSwitchTo(dest_context);
	result = (Jsonb *) PG_DETOAST_DATUM_COPY(datum);
	MemoryContextSwitchTo(oldcontext);

	return result;
}

/*
 * ndb_spi_get_bytea - Extract bytea value from SPI result set
 *
 * Extracts a bytea value from a specific row and column in the SPI result
 * set and copies it to the specified destination memory context.
 *
 * Parameters:
 *   session - SPI session to get parent context from
 *   row_idx - Zero-based row index in result set
 *   col_idx - One-based column index in result set
 *   dest_context - Memory context to allocate result in (NULL uses session's parent context)
 *
 * Returns:
 *   Pointer to bytea value in dest_context, NULL if row/column out of bounds,
 *   value is NULL, SPI_tuptable is not available, or session is NULL
 *
 * Notes:
 *   The function copies the datum to the destination context to ensure it
 *   remains valid after SPI context cleanup. The result must be freed by
 *   the caller using pfree or NDB_FREE.
 */
bytea *
ndb_spi_get_bytea(NdbSpiSession *session,
				  int row_idx,
				  int col_idx,
				  MemoryContext dest_context)
{
	Datum		datum;
	bool		isnull;
	bytea	   *result = NULL;
	MemoryContext oldcontext;

	if (session == NULL)
		return NULL;

	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL)
		return NULL;

	if (row_idx < 0 || row_idx >= SPI_processed)
		return NULL;

	if (col_idx < 1 || col_idx > SPI_tuptable->tupdesc->natts)
		return NULL;

	if (dest_context == NULL)
		dest_context = session->parent_context;

	datum = SPI_getbinval(SPI_tuptable->vals[row_idx],
						  SPI_tuptable->tupdesc,
						  col_idx,
						  &isnull);

	if (isnull)
		return NULL;

	oldcontext = MemoryContextSwitchTo(dest_context);
	result = (bytea *) PG_DETOAST_DATUM_COPY(datum);
	MemoryContextSwitchTo(oldcontext);

	return result;
}
