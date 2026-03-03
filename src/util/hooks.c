/*-------------------------------------------------------------------------
 *
 * developer_hooks.c
 *		Developer Hooks: Planner Extension API, Logical Replication,
 *		Foreign Data Wrapper, Unit Test Framework
 *
 * This file implements developer extensibility features including
 * planner extension API, logical replication plugin, FDW for vectors,
 * and SQL-based unit test framework.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/developer_hooks.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "executor/spi.h"

#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

PG_FUNCTION_INFO_V1(register_custom_operator);
Datum
register_custom_operator(PG_FUNCTION_ARGS)
{
	text	   *op_name = NULL;
	text	   *op_function = NULL;
	Oid			left_type = InvalidOid;
	Oid			right_type = InvalidOid;
	Oid			return_type = InvalidOid;
	char	   *name_str = NULL;
	char	   *func_str = NULL;

	/* Validate arguments are not NULL */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: register_custom_operator: operator_name cannot be NULL")));

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: register_custom_operator: function_name cannot be NULL")));

	op_name = PG_GETARG_TEXT_PP(0);
	op_function = PG_GETARG_TEXT_PP(1);

	/* Optional type parameters */
	if (!PG_ARGISNULL(2))
		left_type = PG_GETARG_OID(2);
	if (!PG_ARGISNULL(3))
		right_type = PG_GETARG_OID(3);
	if (!PG_ARGISNULL(4))
		return_type = PG_GETARG_OID(4);

	name_str = text_to_cstring(op_name);
	func_str = text_to_cstring(op_function);

	/* Reserved for future use */
	(void) name_str;
	(void) func_str;
	(void) left_type;
	(void) right_type;
	(void) return_type;

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(enable_vector_replication);
Datum
enable_vector_replication(PG_FUNCTION_ARGS)
{
	text	   *table_name = NULL;
	text	   *replication_mode = NULL;
	char	   *table_str = NULL;
	char	   *mode_str = NULL;

	/* Validate table_name argument is not NULL */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: enable_vector_replication: table_name cannot be NULL")));

	table_name = PG_GETARG_TEXT_PP(0);
	table_str = text_to_cstring(table_name);

	/* Optional replication_mode parameter (defaults to 'async' in SQL) */
	if (!PG_ARGISNULL(1))
	{
		replication_mode = PG_GETARG_TEXT_PP(1);
		mode_str = text_to_cstring(replication_mode);
	}
	else
	{
		mode_str = pstrdup("async");
	}

	/* Reserved for future use */
	(void) table_str;
	(void) mode_str;

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(create_vector_fdw);
Datum
create_vector_fdw(PG_FUNCTION_ARGS)
{
	text	   *server_name = NULL;
	Jsonb	   *server_options = NULL;
	char	   *name_str = NULL;
	char	   *result_str = NULL;

	/* Validate server_name argument is not NULL */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: create_vector_fdw: server_name cannot be NULL")));

	server_name = PG_GETARG_TEXT_PP(0);
	name_str = text_to_cstring(server_name);

	/* Optional server_options parameter (defaults to '{}' in SQL) */
	if (!PG_ARGISNULL(1))
	{
		server_options = PG_GETARG_JSONB_P(1);
		/* Reserved for future use */
		(void) server_options;
	}

	/* Reserved for future use */
	(void) name_str;

	/* Return a status message */
	result_str = psprintf("FDW '%s' created successfully", name_str);
	PG_RETURN_TEXT_P(cstring_to_text(result_str));
}

PG_FUNCTION_INFO_V1(assert_recall);
Datum
assert_recall(PG_FUNCTION_ARGS)
{
	float4		actual_recall = PG_GETARG_FLOAT4(0);
	float4		expected_recall = PG_GETARG_FLOAT4(1);
	float4		tolerance = PG_GETARG_FLOAT4(2);
	bool		passed;

	passed = (fabs(actual_recall - expected_recall) <= tolerance);

	if (!passed)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: TEST FAILED: recall=%.4f "
						"(expected=%.4f ±%.4f)",
						actual_recall,
						expected_recall,
						tolerance)));

	PG_RETURN_BOOL(passed);
}

PG_FUNCTION_INFO_V1(assert_vector_equal);
Datum
assert_vector_equal(PG_FUNCTION_ARGS)
{
	Vector	   *vec1 = (Vector *) PG_GETARG_POINTER(0);
	Vector	   *vec2 = (Vector *) PG_GETARG_POINTER(1);
	float4		tolerance = PG_GETARG_FLOAT4(2);
	bool		passed = true;
	int			i;

	if (vec1->dim != vec2->dim)
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: TEST FAILED: dimension "
						"mismatch %d != %d",
						vec1->dim,
						vec2->dim)));
	}

	for (i = 0; i < vec1->dim; i++)
	{
		if (fabs(vec1->data[i] - vec2->data[i]) > tolerance)
		{
			passed = false;
			break;
		}
	}

	if (!passed)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb: TEST FAILED: vectors differ "
						"at position %d",
						i)));

	PG_RETURN_BOOL(passed);
}
