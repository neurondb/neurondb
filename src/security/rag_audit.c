/*-------------------------------------------------------------------------
 *
 * rag_audit.c
 *		Audit logging for RAG operations
 *
 * Provides comprehensive audit logging for RAG retrieve and generate
 * operations, tracking user_id, timestamp, operation_type, query hash,
 * result count, and metadata for compliance.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  src/security/rag_audit.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "executor/spi.h"
#include "access/xact.h"
#include "miscadmin.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "utils/guc.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"

/* GUC variable declarations - defined in neurondb_guc.c */
extern PGDLLIMPORT bool neurondb_audit_rag_enabled;

/*
 * log_rag_operation - Log RAG operation to audit table
 *
 * Parameters:
 *   pipeline_name - RAG pipeline name
 *   operation_type - Type of operation (retrieve, generate, chat)
 *   query_hash - SHA-256 hash of query text
 *   result_count - Number of results returned
 *   metadata - Additional metadata as JSONB
 */
PG_FUNCTION_INFO_V1(log_rag_operation);

Datum
log_rag_operation(PG_FUNCTION_ARGS)
{
	text	   *pipeline_name = NULL;
	text	   *operation_type = NULL;
	text	   *query_hash = NULL;
	int32		result_count = 0;
	Jsonb	   *metadata = NULL;
	char	   *pipeline_str = NULL;
	char	   *operation_str = NULL;
	char	   *query_hash_str = NULL;
	char	   *user_id_str = NULL;
	StringInfoData query;
	NdbSpiSession *session = NULL;
	int			ret;

	/* Check if audit logging is enabled */
	if (!neurondb_audit_rag_enabled)
		PG_RETURN_BOOL(true);

	/* Validate arguments */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: log_rag_operation: pipeline_name cannot be NULL")));

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: log_rag_operation: operation_type cannot be NULL")));

	pipeline_name = PG_GETARG_TEXT_PP(0);
	operation_type = PG_GETARG_TEXT_PP(1);
	query_hash = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
	result_count = PG_ARGISNULL(3) ? 0 : PG_GETARG_INT32(3);
	metadata = PG_ARGISNULL(4) ? NULL : PG_GETARG_JSONB_P(4);

	pipeline_str = text_to_cstring(pipeline_name);
	operation_str = text_to_cstring(operation_type);
	query_hash_str = query_hash ? text_to_cstring(query_hash) : NULL;
	user_id_str = pstrdup(GetUserNameFromId(GetUserId(), false));

	/* Build insert query */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "INSERT INTO neurondb.rag_operation_audit_log "
					 "(pipeline_name, operation_type, user_id, query_hash, result_count, metadata) "
					 "VALUES ($1, $2, $3, $4, $5, $6)");

	/* Execute via SPI */
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
	{
		pfree(pipeline_str);
		pfree(operation_str);
		if (query_hash_str)
			pfree(query_hash_str);
		pfree(user_id_str);
		pfree(query.data);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to begin SPI session for audit logging")));
	}

	/* Prepare and execute with parameters */
	{
		Oid			argtypes[6] = {TEXTOID, TEXTOID, TEXTOID, TEXTOID, INT4OID, JSONBOID};
		Datum		values[6];
		char		nulls[6] = {' ', ' ', ' ', ' ', ' ', ' '};

		values[0] = CStringGetTextDatum(pipeline_str);
		values[1] = CStringGetTextDatum(operation_str);
		values[2] = CStringGetTextDatum(user_id_str);

		if (query_hash_str != NULL)
		{
			values[3] = CStringGetTextDatum(query_hash_str);
			nulls[3] = ' ';
		}
		else
		{
			values[3] = (Datum) 0;
			nulls[3] = 'n';
		}

		values[4] = Int32GetDatum(result_count);

		if (metadata != NULL)
		{
			values[5] = PointerGetDatum(metadata);
			nulls[5] = ' ';
		}
		else
		{
			values[5] = (Datum) 0;
			nulls[5] = 'n';
		}

		ret = ndb_spi_execute_with_args(session, query.data, 6, argtypes, values, nulls, false, 0);
	}

	if (ret < 0)
	{
		ndb_spi_session_end(&session);
		pfree(pipeline_str);
		pfree(operation_str);
		if (query_hash_str)
			pfree(query_hash_str);
		pfree(user_id_str);
		pfree(query.data);
		ereport(WARNING,
				(errcode(ERRCODE_WARNING),
				 errmsg("neurondb: failed to log RAG operation audit entry")));
	}

	ndb_spi_session_end(&session);
	pfree(pipeline_str);
	pfree(operation_str);
	if (query_hash_str)
		pfree(query_hash_str);
	pfree(user_id_str);
	pfree(query.data);

	PG_RETURN_BOOL(true);
}
