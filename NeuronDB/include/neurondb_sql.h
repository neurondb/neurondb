/*-------------------------------------------------------------------------
 *
 * neurondb_sql.h
 *    Centralized SQL query repository for Linear Regression
 *
 * Provides function declarations for accessing SQL query templates
 * used in linear regression operations.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    include/neurondb_sql.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_SQL_H
#define NEURONDB_SQL_H

#include "postgres.h"

/*-------------------------------------------------------------------------
 * SQL Query Functions for Linear Regression
 *-------------------------------------------------------------------------
 */

/* Get formatted query: Load full dataset */
extern const char *ndb_sql_get_load_dataset(const char *quoted_feat_col,
											const char *quoted_target_col,
											const char *quoted_table);

/* Get formatted query: Load limited dataset */
extern const char *ndb_sql_get_load_dataset_limited(const char *quoted_feat_col,
													const char *quoted_target_col,
													const char *quoted_table,
													int max_rows);

/* Get formatted query: Load dataset chunk */
extern const char *ndb_sql_get_load_dataset_chunk(const char *quoted_feat_col,
												  const char *quoted_target_col,
												  const char *quoted_table,
												  int chunk_size,
												  int offset);

/* Get formatted query: Check dataset validity */
extern const char *ndb_sql_get_check_dataset(const char *quoted_feat_col,
											 const char *quoted_target_col,
											 const char *quoted_table);

/* Get formatted query: Count dataset rows */
extern const char *ndb_sql_get_count_dataset(const char *quoted_feat_col,
											 const char *quoted_target_col,
											 const char *quoted_table);

/* Get formatted query: Load dataset for evaluation */
extern const char *ndb_sql_get_eval_dataset(const char *quoted_feat_col,
											const char *quoted_target_col,
											const char *quoted_table);

#endif							/* NEURONDB_SQL_H */
