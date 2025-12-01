/*-------------------------------------------------------------------------
 *
 * neurondb_sql.c
 *    Centralized SQL query repository for Linear Regression
 *
 * Provides centralized SQL query templates for linear regression operations.
 * All SQL queries used in linear regression training, evaluation, and
 * prediction are defined here for maintainability.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/util/neurondb_sql.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "lib/stringinfo.h"
#include "utils/memutils.h"
#include "neurondb_macros.h"
#include "neurondb_sql.h"

/*-------------------------------------------------------------------------
 * SQL Query Templates for Linear Regression
 *-------------------------------------------------------------------------
 */

/* Query: Load full dataset for training */
#define LINREG_SQL_LOAD_DATASET \
	"SELECT %s, %s FROM %s"

/* Query: Load limited dataset with NULL filtering */
#define LINREG_SQL_LOAD_DATASET_LIMITED \
	"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d"

/* Query: Load dataset chunk for streaming processing */
#define LINREG_SQL_LOAD_DATASET_CHUNK \
	"SELECT %s, %s FROM %s LIMIT %d OFFSET %d"

/* Query: Check if dataset has valid rows */
#define LINREG_SQL_CHECK_DATASET \
	"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT 1"

/* Query: Count valid rows in dataset */
#define LINREG_SQL_COUNT_DATASET \
	"SELECT COUNT(*) FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL"

/* Query: Load dataset for evaluation */
#define LINREG_SQL_EVAL_DATASET \
	"SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL"

/*-------------------------------------------------------------------------
 * Get formatted query: Load full dataset
 *-------------------------------------------------------------------------
 */
const char *
ndb_sql_get_load_dataset(const char *quoted_feat_col,
						 const char *quoted_target_col,
						 const char *quoted_table)
{
	StringInfoData buf;
	NDB_DECLARE (char *, result);

	initStringInfo(&buf);
	appendStringInfo(&buf, LINREG_SQL_LOAD_DATASET,
					 quoted_feat_col,
					 quoted_target_col,
					 quoted_table);

	result = pstrdup(buf.data);
	NDB_FREE	(buf.data);

	return result;
}

/*-------------------------------------------------------------------------
 * Get formatted query: Load limited dataset
 *-------------------------------------------------------------------------
 */
const char *
ndb_sql_get_load_dataset_limited(const char *quoted_feat_col,
								 const char *quoted_target_col,
								 const char *quoted_table,
								 int max_rows)
{
	StringInfoData buf;
	NDB_DECLARE (char *, result);

	initStringInfo(&buf);
	appendStringInfo(&buf, LINREG_SQL_LOAD_DATASET_LIMITED,
					 quoted_feat_col,
					 quoted_target_col,
					 quoted_table,
					 quoted_feat_col,
					 quoted_target_col,
					 max_rows);

	result = pstrdup(buf.data);
	NDB_FREE	(buf.data);

	return result;
}

/*-------------------------------------------------------------------------
 * Get formatted query: Load dataset chunk
 *-------------------------------------------------------------------------
 */
const char *
ndb_sql_get_load_dataset_chunk(const char *quoted_feat_col,
							   const char *quoted_target_col,
							   const char *quoted_table,
							   int chunk_size,
							   int offset)
{
	StringInfoData buf;
	NDB_DECLARE (char *, result);

	initStringInfo(&buf);
	appendStringInfo(&buf, LINREG_SQL_LOAD_DATASET_CHUNK,
					 quoted_feat_col,
					 quoted_target_col,
					 quoted_table,
					 chunk_size,
					 offset);

	result = pstrdup(buf.data);
	NDB_FREE	(buf.data);

	return result;
}

/*-------------------------------------------------------------------------
 * Get formatted query: Check dataset validity
 *-------------------------------------------------------------------------
 */
const char *
ndb_sql_get_check_dataset(const char *quoted_feat_col,
						  const char *quoted_target_col,
						  const char *quoted_table)
{
	StringInfoData buf;
	NDB_DECLARE (char *, result);

	initStringInfo(&buf);
	appendStringInfo(&buf, LINREG_SQL_CHECK_DATASET,
					 quoted_feat_col,
					 quoted_target_col,
					 quoted_table,
					 quoted_feat_col,
					 quoted_target_col);

	result = pstrdup(buf.data);
	NDB_FREE	(buf.data);

	return result;
}

/*-------------------------------------------------------------------------
 * Get formatted query: Count dataset rows
 *-------------------------------------------------------------------------
 */
const char *
ndb_sql_get_count_dataset(const char *quoted_feat_col,
						  const char *quoted_target_col,
						  const char *quoted_table)
{
	StringInfoData buf;
	NDB_DECLARE (char *, result);

	initStringInfo(&buf);
	appendStringInfo(&buf, LINREG_SQL_COUNT_DATASET,
					 quoted_table,
					 quoted_feat_col,
					 quoted_target_col);

	result = pstrdup(buf.data);
	NDB_FREE	(buf.data);

	return result;
}

/*-------------------------------------------------------------------------
 * Get formatted query: Load dataset for evaluation
 *-------------------------------------------------------------------------
 */
const char *
ndb_sql_get_eval_dataset(const char *quoted_feat_col,
						 const char *quoted_target_col,
						 const char *quoted_table)
{
	StringInfoData buf;
	NDB_DECLARE (char *, result);

	initStringInfo(&buf);
	appendStringInfo(&buf, LINREG_SQL_EVAL_DATASET,
					 quoted_feat_col,
					 quoted_target_col,
					 quoted_table,
					 quoted_feat_col,
					 quoted_target_col);

	result = pstrdup(buf.data);
	NDB_FREE	(buf.data);

	return result;
}
