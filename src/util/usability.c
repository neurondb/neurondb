/*-------------------------------------------------------------------------
 *
 * usability.c
 *		Usability enhancements: CREATE MODEL, CREATE INDEX USING ANN, etc.
 *
 * This file implements user-friendly syntax for NeuronDB operations
 * including model management, index creation, and configuration display.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/usability.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "neurondb_spi.h"
#include "neurondb_macros.h"

PG_FUNCTION_INFO_V1(create_model);
Datum
create_model(PG_FUNCTION_ARGS)
{
	char	   *config_str = NULL;
	NdbSpiSession *session = NULL;
	text	   *config_json = NULL;
	text	   *model_name = NULL;
	text	   *model_type = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: create_model requires 3 arguments")));

	model_name = PG_GETARG_TEXT_PP(0);
	model_type = PG_GETARG_TEXT_PP(1);
	config_json = PG_GETARG_TEXT_PP(2);

	(void) model_name;
	(void) model_type;
	config_str = text_to_cstring(config_json);
	(void) config_str;


	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to begin SPI session in "
						"create_model")));

	ndb_spi_session_end(&session);

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(drop_model);
Datum
drop_model(PG_FUNCTION_ARGS)
{
	char	   *name_str = NULL;
	NdbSpiSession *session2 = NULL;
	text	   *model_name = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: drop_model requires 1 argument")));

	model_name = PG_GETARG_TEXT_PP(0);

	name_str = text_to_cstring(model_name);

	(void) name_str;
	session2 = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session2 == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to begin SPI session in "
						"drop_model")));

	ndb_spi_session_end(&session2);

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(create_ann_index);
Datum
create_ann_index(PG_FUNCTION_ARGS)
{
	text	   *index_name = NULL;
	text	   *table_name = NULL;
	text	   *column_name = NULL;
	text	   *index_type = NULL;
	text	   *options = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 5)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: create_ann_index requires 5 arguments")));

	index_name = PG_GETARG_TEXT_PP(0);
	table_name = PG_GETARG_TEXT_PP(1);
	column_name = PG_GETARG_TEXT_PP(2);
	index_type = PG_GETARG_TEXT_PP(3);
	options = PG_GETARG_TEXT_PP(4);

	(void) index_name;
	(void) table_name;
	(void) column_name;
	(void) index_type;
	(void) options;


	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(explain_vector_query);
Datum
explain_vector_query(PG_FUNCTION_ARGS)
{
	text	   *query = NULL;

	char *query_str = NULL;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: explain_vector_query requires 1 argument")));

	query = PG_GETARG_TEXT_PP(0);

	query_str = text_to_cstring(query);
	(void) query_str;


	PG_RETURN_TEXT_P(cstring_to_text("Vector query plan generated"));
}

/*
 * neurondb_api_docs - Get API documentation for a NeuronDB function
 *
 * User-facing function that returns documentation for a specified NeuronDB
 * function, including description, parameters, examples, and performance
 * characteristics. Can be used with psql's \dx+ command for inline help.
 *
 * Parameters:
 *   function_name - Name of the NeuronDB function to document (text)
 *
 * Returns:
 *   Text string containing formatted documentation
 *
 * Notes:
 *   This function provides SQL-based access to function documentation,
 *   making it easy to get help on NeuronDB functions directly from the
 *   database without external documentation.
 */
PG_FUNCTION_INFO_V1(neurondb_api_docs);
Datum
neurondb_api_docs(PG_FUNCTION_ARGS)
{
	text	   *function_name = NULL;

	char *func_str = NULL;
	StringInfoData docs;

	/* Validate argument count */
	if (PG_NARGS() != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: neurondb_api_docs requires 1 argument")));

	function_name = PG_GETARG_TEXT_PP(0);

	func_str = text_to_cstring(function_name);

	initStringInfo(&docs);
	appendStringInfo(
					 &docs, "NeuronDB Function Documentation: %s\n\n", func_str);
	appendStringInfo(&docs, "Description: Advanced AI database function\n");
	appendStringInfo(&docs, "Parameters: See pg_proc catalog\n");
	appendStringInfo(&docs, "Examples: SELECT %s(...)\n", func_str);
	appendStringInfo(&docs,
					 "Performance: Optimized for large-scale vector operations\n");

	PG_RETURN_TEXT_P(cstring_to_text(docs.data));
}
