/*-------------------------------------------------------------------------
 *
 * ml_catalog.h
 *    Utilities for registering and fetching models in neurondb.ml_models.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_CATALOG_H
#define NEURONDB_ML_CATALOG_H

#include "postgres.h"
#include "utils/jsonb.h"

typedef struct MLCatalogModelSpec
{
	const char *algorithm;
	const char *model_type;
	const char *training_table;
	const char *training_column;
	const char *project_name;
	const char *model_name;
	Jsonb	   *parameters;
	Jsonb	   *metrics;
	bytea	   *model_data;
	int32		training_time_ms;
	int32		num_samples;
	int32		num_features;
} MLCatalogModelSpec;

extern int32 ml_catalog_register_model(const MLCatalogModelSpec *spec);
extern bool ml_catalog_fetch_model_payload(int32 model_id,
										   bytea * *model_data_out,
										   Jsonb * *parameters_out,
										   Jsonb * *metrics_out);

#endif							/* NEURONDB_ML_CATALOG_H */
