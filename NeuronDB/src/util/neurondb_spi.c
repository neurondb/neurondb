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

/*-------------------------------------------------------------------------
 * SPI Session Structure
 *-------------------------------------------------------------------------
 */

struct NdbSpiSession
{
	bool		we_connected_spi;	/* Did we call SPI_connect()? */
	MemoryContext parent_context;	/* Context before SPI_connect() */
	MemoryContext spi_context;	/* SPI context (if connected) */
};

/*-------------------------------------------------------------------------
 * SPI Session Management
 *-------------------------------------------------------------------------
 */

NdbSpiSession *
ndb_spi_session_begin(MemoryContext parent_context, bool assume_spi_connected)
{
	NDB_DECLARE(NdbSpiSession *, session);
	MemoryContext oldcontext;

	if (parent_context == NULL)
		parent_context = CurrentMemoryContext;

	/* Allocate session in parent context */
	oldcontext = MemoryContextSwitchTo(parent_context);
	NDB_ALLOC(session, NdbSpiSession, 1);

	MemoryContextSwitchTo(oldcontext);

	session->parent_context = parent_context;

	if (assume_spi_connected)
	{
		/* Caller says SPI is already connected - we don't manage it */
		session->we_connected_spi = false;
		session->spi_context = CurrentMemoryContext;
		elog(DEBUG1, "neurondb: SPI session: assuming SPI already connected");
	}
	else
	{
		/* Always connect SPI ourselves */
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

	/* Only finish SPI if we connected it ourselves */
	if ((*session)->we_connected_spi)
	{
		/* Switch to parent context before SPI_finish() */
		if ((*session)->parent_context != NULL)
			MemoryContextSwitchTo((*session)->parent_context);
		SPI_finish();
		elog(DEBUG1, "neurondb: SPI session: finished SPI (we connected it)");
	}
	else
	{
		elog(DEBUG1, "neurondb: SPI session: not finishing SPI (caller connected it)");
	}

	/* Free session structure */
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

/*-------------------------------------------------------------------------
 * SPI Query Execution
 *-------------------------------------------------------------------------
 */

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

	/* Ensure we're in SPI context */
	oldcontext = MemoryContextSwitchTo(session->spi_context);

	PG_TRY();
	{
		ret = SPI_execute(query, read_only ? 1 : 0, tcount);

		/* Only check SPI_tuptable for queries that return result sets */
		if (ret == SPI_OK_SELECT || ret == SPI_OK_SELINTO ||
			ret == SPI_OK_INSERT_RETURNING || ret == SPI_OK_UPDATE_RETURNING ||
			ret == SPI_OK_DELETE_RETURNING)
		{
			NDB_CHECK_SPI_TUPTABLE();
		}

		/* Check for SPI error codes */
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

	/* Ensure we're in SPI context */
	oldcontext = MemoryContextSwitchTo(session->spi_context);

	PG_TRY();
	{
		ret = SPI_execute_with_args(src, nargs, argtypes, values, nulls,
									read_only ? 1 : 0, tcount);

		/* Only check SPI_tuptable for queries that return result sets */
		if (ret == SPI_OK_SELECT || ret == SPI_OK_SELINTO ||
			ret == SPI_OK_INSERT_RETURNING || ret == SPI_OK_UPDATE_RETURNING ||
			ret == SPI_OK_DELETE_RETURNING)
		{
			NDB_CHECK_SPI_TUPTABLE();
		}

		/* Check for SPI error codes */
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

/*-------------------------------------------------------------------------
 * StringInfoData Management
 *-------------------------------------------------------------------------
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

	/* Initialize StringInfo in SPI context */
	oldcontext = MemoryContextSwitchTo(session->spi_context);
	initStringInfo(str);
	MemoryContextSwitchTo(oldcontext);
}

void
ndb_spi_stringinfo_free(NdbSpiSession *session, StringInfoData * str)
{
	if (session == NULL || str == NULL || str->data == NULL)
		return;

	/*
	 * pfree is context-aware through chunk headers, so we don't need to
	 * switch contexts. Always free explicitly for consistency.
	 */
	NDB_FREE(str->data);
}

void
ndb_spi_stringinfo_reset(NdbSpiSession *session, StringInfoData * str)
{
	if (session == NULL || str == NULL)
		return;

	/* Free old buffer */
	ndb_spi_stringinfo_free(session, str);

	/* Reinitialize in SPI context */
	ndb_spi_stringinfo_init(session, str);
}

/*-------------------------------------------------------------------------
 * Result Extraction
 *-------------------------------------------------------------------------
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

	/* Extract datum from SPI result */
	datum = SPI_getbinval(SPI_tuptable->vals[row_idx],
						  SPI_tuptable->tupdesc,
						  col_idx,
						  &isnull);

	if (isnull)
		return false;

	/* Get the type OID to ensure proper conversion */
	type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, col_idx);

	/* Convert to int32 based on the actual type */
	if (type_oid == INT4OID || type_oid == INT2OID || type_oid == INT8OID)
	{
		/* Direct conversion for integer types */
		if (type_oid == INT4OID)
			*out_value = DatumGetInt32(datum);
		else if (type_oid == INT2OID)
			*out_value = (int32) DatumGetInt16(datum);
		else if (type_oid == INT8OID)
			*out_value = (int32) DatumGetInt64(datum);
	}
	else
	{
		/*
		 * Type is not an integer type - this shouldn't happen for training
		 * functions
		 */
		/* Log warning and return false */
		elog(WARNING,
			 "neurondb: ndb_spi_get_int32: unexpected type OID %u (expected integer type)",
			 type_oid);
		return false;
	}

	return true;
}

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

	/* Extract datum from SPI result */
	datum = SPI_getbinval(SPI_tuptable->vals[row_idx],
						  SPI_tuptable->tupdesc,
						  col_idx,
						  &isnull);

	if (isnull)
		return NULL;

	/* Copy directly from datum to caller's context */
	oldcontext = MemoryContextSwitchTo(dest_context);
	result = (text *) PG_DETOAST_DATUM_COPY(datum);
	MemoryContextSwitchTo(oldcontext);

	return result;
}

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

	/* Extract datum from SPI result */
	datum = SPI_getbinval(SPI_tuptable->vals[row_idx],
						  SPI_tuptable->tupdesc,
						  col_idx,
						  &isnull);

	if (isnull)
		return NULL;

	/* Copy directly from datum to caller's context */
	oldcontext = MemoryContextSwitchTo(dest_context);
	result = (Jsonb *) PG_DETOAST_DATUM_COPY(datum);
	MemoryContextSwitchTo(oldcontext);

	return result;
}

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

	/* Extract datum from SPI result */
	datum = SPI_getbinval(SPI_tuptable->vals[row_idx],
						  SPI_tuptable->tupdesc,
						  col_idx,
						  &isnull);

	if (isnull)
		return NULL;

	/* Copy directly from datum to caller's context */
	oldcontext = MemoryContextSwitchTo(dest_context);
	result = (bytea *) PG_DETOAST_DATUM_COPY(datum);
	MemoryContextSwitchTo(oldcontext);

	return result;
}
