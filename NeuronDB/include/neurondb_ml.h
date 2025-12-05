/*-------------------------------------------------------------------------
 *
 * neurondb_ml.h
 *    ML utilities and function declarations
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    include/neurondb_ml.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_H
#define NEURONDB_ML_H

#include "postgres.h"

/* Utility functions */
float	  **neurondb_fetch_vectors_from_table(const char *table,
											  const char *col,
											  int *out_count,
											  int *out_dim);

#endif							/* NEURONDB_ML_H */
