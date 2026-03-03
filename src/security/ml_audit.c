/*-------------------------------------------------------------------------
 *
 * ml_audit.c
 *		Audit logging for ML inference operations
 *
 * Provides comprehensive audit logging for ML model inference calls,
 * tracking user_id, timestamp, operation_type, input/output hashes,
 * and metadata for compliance and security.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *	  src/security/ml_audit.c
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

/* GUC variable declarations - defined in neurondb_guc.c */
extern PGDLLIMPORT bool neurondb_audit_ml_enabled;

/*
 * log_ml_inference - Log ML inference operation to audit table
 *
 * Parameters:
 *   model_id - ML model ID
 *   operation_type - Type of operation (predict, batch_predict, etc.)
 *   input_hash - SHA-256 hash of input data
 *   output_hash - SHA-256 hash of output data
 *   metadata - Additional metadata as JSONB
 */
PG_FUNCTION_INFO_V1(log_ml_inference);

Datum
log_ml_inference(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *operation_type = NULL;
	text	   *input_hash = NULL;
	text	   *output_hash = NULL;
	Jsonb	   *metadata = NULL;
	char	   *operation_str = NULL;
	char	   *input_hash_str = NULL;
	char	   *output_hash_str = NULL;
	char	   *user_id_str = NULL;
	StringInfoData query;
	NdbSpiSession *session = NULL;
	int			ret;

	/* Check if audit logging is enabled */
	if (!neurondb_audit_ml_enabled)
		PG_RETURN_BOOL(true);

	/* Validate arguments */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: log_ml_inference: model_id cannot be NULL")));

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: log_ml_inference: operation_type cannot be NULL")));

	model_id = PG_GETARG_INT32(0);
	operation_type = PG_GETARG_TEXT_PP(1);
	input_hash = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
	output_hash = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);
	metadata = PG_ARGISNULL(4) ? NULL : PG_GETARG_JSONB_P(4);

	operation_str = text_to_cstring(operation_type);
	input_hash_str = input_hash ? text_to_cstring(input_hash) : NULL;
	output_hash_str = output_hash ? text_to_cstring(output_hash) : NULL;
	user_id_str = pstrdup(GetUserNameFromId(GetUserId(), false));

	/* Build insert query */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "INSERT INTO neurondb.ml_inference_audit_log "
					 "(model_id, operation_type, user_id, input_hash, output_hash, metadata) "
					 "VALUES ($1, $2, $3, $4, $5, $6)");

	/* Execute via SPI */
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
	{
		pfree(operation_str);
		if (input_hash_str)
			pfree(input_hash_str);
		if (output_hash_str)
			pfree(output_hash_str);
		pfree(user_id_str);
		pfree(query.data);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to begin SPI session for audit logging")));
	}

	/* Prepare and execute with parameters */
	{
		Oid			argtypes[6] = {INT4OID, TEXTOID, TEXTOID, TEXTOID, TEXTOID, JSONBOID};
		Datum		values[6];
		char		nulls[6] = {' ', ' ', ' ', ' ', ' ', ' '};
		text	   *operation_text = NULL;
		text	   *user_id_text = NULL;
		text	   *input_hash_text = NULL;
		text	   *output_hash_text = NULL;

		values[0] = Int32GetDatum(model_id);
		operation_text = cstring_to_text(operation_str);
		values[1] = PointerGetDatum(operation_text);
		user_id_text = cstring_to_text(user_id_str);
		values[2] = PointerGetDatum(user_id_text);

		if (input_hash_str != NULL)
		{
			input_hash_text = cstring_to_text(input_hash_str);
			values[3] = PointerGetDatum(input_hash_text);
			nulls[3] = ' ';
		}
		else
		{
			values[3] = (Datum) 0;
			nulls[3] = 'n';
		}

		if (output_hash_str != NULL)
		{
			output_hash_text = cstring_to_text(output_hash_str);
			values[4] = PointerGetDatum(output_hash_text);
			nulls[4] = ' ';
		}
		else
		{
			values[4] = (Datum) 0;
			nulls[4] = 'n';
		}

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
		pfree(operation_str);
		if (input_hash_str)
			pfree(input_hash_str);
		if (output_hash_str)
			pfree(output_hash_str);
		pfree(user_id_str);
		pfree(query.data);
		ereport(WARNING,
				(errcode(ERRCODE_WARNING),
				 errmsg("neurondb: failed to log ML inference audit entry")));
	}

	ndb_spi_session_end(&session);
	pfree(operation_str);
	if (input_hash_str)
		pfree(input_hash_str);
	if (output_hash_str)
		pfree(output_hash_str);
	pfree(user_id_str);
	pfree(query.data);

	PG_RETURN_BOOL(true);
}
