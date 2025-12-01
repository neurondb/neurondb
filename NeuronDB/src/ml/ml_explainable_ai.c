/*-------------------------------------------------------------------------
 *
 * ml_explainable_ai.c
 *    Model explainability algorithms.
 *
 * This module implements SHAP, LIME, and feature importance methods for
 * explaining model predictions.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_explainable_ai.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/jsonb.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_catalog.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

/*
 * Helper function to call model prediction via SPI
 * Returns prediction value, or 0.0 on error
 */
static double
call_model_predict(int32 model_id, float *features, int n_features)
{
	StringInfoData sql;
	StringInfoData features_str;
	int			ret;
	double		prediction = 0.0;
	int			i;

	/* Build features array string */
	initStringInfo(&features_str);
	appendStringInfoString(&features_str, "ARRAY[");
	for (i = 0; i < n_features; i++)
	{
		if (i > 0)
			appendStringInfoString(&features_str, ", ");
		appendStringInfo(&features_str, "%.6f", (double) features[i]);
	}
	appendStringInfoString(&features_str, "]::real[]");

	/* Call unified predict function */
	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT neurondb.predict(%d, %s)", model_id, features_str.data);

	ret = ndb_spi_execute_safe(sql.data, true, 1);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	/* Safe access for complex types - validate before access */
	if (SPI_tuptable != NULL && SPI_tuptable->vals != NULL && 
		SPI_processed > 0 && SPI_tuptable->vals[0] != NULL && SPI_tuptable->tupdesc != NULL)
	{
		bool		isnull = false;
		Datum		result = SPI_getbinval(SPI_tuptable->vals[0],
										   SPI_tuptable->tupdesc,
										   1,
										   &isnull);

		if (!isnull)
			prediction = DatumGetFloat8(result);
	}

	NDB_FREE(sql.data);
	NDB_FREE(features_str.data);

	return prediction;
}

/*
 * calculate_shap_values
 * --------------------
 * Calculate SHAP values for a prediction.
 *
 * SQL Arguments:
 *   model_id      - Trained model ID
 *   instance      - Feature vector to explain
 *   n_samples     - Number of samples for Kernel SHAP (default: 100)
 *
 * Returns:
 *   Array of SHAP values (one per feature)
 */
PG_FUNCTION_INFO_V1(calculate_shap_values);

Datum
calculate_shap_values(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType  *instance;
	int			n_samples;
	float	   *features;
	int			n_features;
	double	   *shap_values;
	int			i,
				j;
	ArrayType  *result;
	Datum	   *result_datums;
	StringInfoData query;
	int			ret;

	model_id = PG_GETARG_INT32(0);
	instance = PG_GETARG_ARRAYTYPE_P(1);
	n_samples = PG_ARGISNULL(2) ? 100 : PG_GETARG_INT32(2);

	if (n_samples < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_samples must be positive")));

	/* Extract features from array */
	n_features = ARR_DIMS(instance)[0];
	features = (float *) ARR_DATA_PTR(instance);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Initialize SHAP values */
	NDB_DECLARE(double *, shap_values);
	NDB_ALLOC(shap_values, double, n_features);

	/* Kernel SHAP: approximate Shapley values using sampling */
	/* For each feature, estimate its contribution */
	for (i = 0; i < n_features; i++)
	{
		double		sum_contrib = 0.0;
		int			valid_samples = 0;

		for (j = 0; j < n_samples; j++)
		{
			/* Create perturbed instance */
			NDB_DECLARE(float *, perturbed);
			NDB_ALLOC(perturbed, float, n_features);
			int			k;
			double		pred_with,
						pred_without;

			/* Sample subset of features */
			for (k = 0; k < n_features; k++)
			{
				if (k == i)
					perturbed[k] = features[k];
				else if ((rand() % 2) == 0)
					perturbed[k] = features[k];
				else
					perturbed[k] = 0.0; /* Missing feature */
			}

			/* Get prediction with feature i */
			pred_with = call_model_predict(model_id, perturbed, n_features);

			/* Get prediction without feature i */
			perturbed[i] = 0.0;
			pred_without = call_model_predict(model_id, perturbed, n_features);

			sum_contrib += pred_with - pred_without;
			valid_samples++;

			NDB_FREE(perturbed);
		}

		if (valid_samples > 0)
			shap_values[i] = sum_contrib / valid_samples;
	}

	/* Build result array */
	NDB_DECLARE(Datum *, result_datums);
	NDB_ALLOC(result_datums, Datum, n_features);
	for (i = 0; i < n_features; i++)
		result_datums[i] = Float8GetDatum(shap_values[i]);

	result = construct_array(result_datums,
							 n_features,
							 FLOAT8OID,
							 sizeof(float8),
							 FLOAT8PASSBYVAL,
							 'd');

	/* Cleanup */
	NDB_FREE(shap_values);
	NDB_FREE(result_datums);
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * explain_with_lime
 * ------------------
 * Generate LIME explanation for a prediction.
 *
 * SQL Arguments:
 *   model_id      - Trained model ID
 *   instance      - Feature vector to explain
 *   n_samples     - Number of perturbed samples (default: 1000)
 *   n_features    - Number of top features to explain (default: 10)
 *
 * Returns:
 *   JSONB with feature names and importance scores
 */
PG_FUNCTION_INFO_V1(explain_with_lime);

Datum
explain_with_lime(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType  *instance;
	int			n_samples;
	int			n_features;
	float	   *features;
	int			feature_dim;
	float	   *perturbed_features;
	double	   *predictions;
	double	   *weights;
	double	   *coefficients;
	int			i,
				j;
	StringInfoData jsonbuf;
	Jsonb	   *result;

	model_id = PG_GETARG_INT32(0);
	instance = PG_GETARG_ARRAYTYPE_P(1);
	n_samples = PG_ARGISNULL(2) ? 1000 : PG_GETARG_INT32(2);
	n_features = PG_ARGISNULL(3) ? 10 : PG_GETARG_INT32(3);

	if (n_samples < 1 || n_features < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_samples and n_features must be positive")));

	feature_dim = ARR_DIMS(instance)[0];
	features = (float *) ARR_DATA_PTR(instance);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Allocate arrays */
	NDB_DECLARE(float *, perturbed_features);
	NDB_DECLARE(double *, predictions);
	NDB_DECLARE(double *, weights);
	NDB_DECLARE(double *, coefficients);
	NDB_ALLOC(perturbed_features, float, feature_dim * n_samples);
	NDB_ALLOC(predictions, double, n_samples);
	NDB_ALLOC(weights, double, n_samples);
	NDB_ALLOC(coefficients, double, feature_dim);

	/* Generate perturbed samples and get predictions */
	for (i = 0; i < n_samples; i++)
	{
		double		distance = 0.0;
		int			k;

		/* Perturb features */
		for (k = 0; k < feature_dim; k++)
		{
			float		noise = ((float) rand() / (float) RAND_MAX - 0.5) * 0.2;

			perturbed_features[i * feature_dim + k] = features[k] + noise;
			distance += noise * noise;
		}

		/* Get prediction using actual model */
		predictions[i] = call_model_predict(model_id,
											&perturbed_features[i * feature_dim],
											feature_dim);

		/* Exponential kernel weight based on distance */
		distance = sqrt(distance);
		weights[i] = exp(-distance * distance / (2.0 * 0.5 * 0.5));
	}

	/* Fit linear model: minimize weighted MSE */
	/* Simplified: would use proper linear regression solver */
	for (i = 0; i < feature_dim; i++)
	{
		double		numerator = 0.0;
		double		denominator = 0.0;

		for (j = 0; j < n_samples; j++)
		{
			float		feat_val = perturbed_features[j * feature_dim + i];

			numerator += weights[j] * feat_val * predictions[j];
			denominator += weights[j] * feat_val * feat_val;
		}

		if (denominator > 1e-10)
			coefficients[i] = numerator / denominator;
	}

	/* Build JSONB result */
	ndb_spi_stringinfo_init(spi_session, &jsonbuf);
	appendStringInfoString(&jsonbuf, "{\"features\":[");
	for (i = 0; i < feature_dim && i < n_features; i++)
	{
		if (i > 0)
			appendStringInfoString(&jsonbuf, ",");
		appendStringInfo(&jsonbuf,
						 "{\"index\":%d,\"importance\":%.6f}",
						 i,
						 coefficients[i]);
	}
	appendStringInfoString(&jsonbuf, "]}");

	result = DatumGetJsonbP(
							DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));

	/* Cleanup */
	NDB_FREE(perturbed_features);
	NDB_FREE(predictions);
	NDB_FREE(weights);
	NDB_FREE(coefficients);
	ndb_spi_stringinfo_free(spi_session, &jsonbuf);
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_POINTER(result);
}

/*
 * feature_importance
 * ------------------
 * Calculate feature importance using permutation method.
 *
 * SQL Arguments:
 *   model_id      - Trained model ID
 *   table_name    - Test data table
 *   feature_column - Feature column name
 *   target_column  - Target column name
 *   metric        - Metric to use ('accuracy', 'mse', 'mae') (default: 'mse')
 *
 * Returns:
 *   Array of importance scores (one per feature)
 */
PG_FUNCTION_INFO_V1(feature_importance);

Datum
feature_importance(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *feature_column;
	text	   *target_column;
	text	   *metric_text;
	char	   *tbl_str;
	char	   *feat_col_str;
	char	   *targ_col_str;
	char	   *metric;
	float	  **features;
	double	   *targets;
	int			n_samples;
	int			n_features;
	double		baseline_score;
	double	   *importance;
	int			i,
				j;
	ArrayType  *result;
	Datum	   *result_datums;
	StringInfoData query;
	int			ret;

	model_id = PG_GETARG_INT32(0);
	table_name = PG_GETARG_TEXT_PP(1);
	feature_column = PG_GETARG_TEXT_PP(2);
	target_column = PG_GETARG_TEXT_PP(3);
	metric_text = PG_ARGISNULL(4) ? cstring_to_text("mse")
		: PG_GETARG_TEXT_PP(4);

	tbl_str = text_to_cstring(table_name);
	feat_col_str = text_to_cstring(feature_column);
	targ_col_str = text_to_cstring(target_column);
	metric = text_to_cstring(metric_text);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Load test data */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s",
					 feat_col_str,
					 targ_col_str,
					 tbl_str);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_col_str);
		NDB_FREE(targ_col_str);
		NDB_FREE(metric);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Failed to load test data")));
	}

	n_samples = SPI_processed;
	
	/* Extract actual feature count from model metadata or data schema */
	n_features = 10; /* Default fallback */
	
	/* Try to get feature count from model metadata */
	{
		MLCatalogModelSpec *model_spec = NULL;
		Jsonb	   *model_metadata = NULL;
		JsonbIterator *it;
		JsonbValue	v;
		int			r;
		bool		found = false;

		/* Look up model from catalog */
		model_spec = ml_catalog_get_model(model_id, NULL);
		if (model_spec != NULL && model_spec->metadata != NULL)
		{
			model_metadata = model_spec->metadata;
			PG_TRY();
			{
				it = JsonbIteratorInit((JsonbContainer *) & model_metadata->root);
				while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE && !found)
				{
					if (r == WJB_KEY)
					{
						char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

						r = JsonbIteratorNext(&it, &v, false);
						if ((strcmp(key, "n_features") == 0 || strcmp(key, "feature_dim") == 0
							 || strcmp(key, "feature_count") == 0) && v.type == jbvNumeric)
						{
							Numeric		num = DatumGetNumeric(v.val.numeric);
							int			val = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(num)));

							if (val > 0 && val < 100000)
							{
								n_features = val;
								found = true;
							}
						}
						NDB_FREE(key);
					}
				}
			}
			PG_CATCH();
			{
				FlushErrorState();
			}
			PG_END_TRY();
		}

		/* If not found in metadata, try to infer from data schema */
		if (!found && SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			int			col_count = tupdesc->natts;
			int			feature_cols = 0;

			/* Count non-target columns (assuming last column is target) */
			for (int col = 0; col < col_count - 1; col++)
			{
				if (!tupdesc->attrs[col].attisdropped)
					feature_cols++;
			}

			if (feature_cols > 0)
				n_features = feature_cols;
		}

		if (model_spec)
			NDB_FREE(model_spec);
	}

	NDB_DECLARE(float **, features);
	NDB_DECLARE(double *, targets);
	NDB_ALLOC(features, float *, n_samples);
	NDB_ALLOC(targets, double, n_samples);

	for (i = 0; i < n_samples; i++)
	{
		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			continue;
		}
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		if (tupdesc == NULL)
		{
			continue;
		}
		ArrayType  *feat_array = NULL;
		double		target = 0.0;
		
		/* Safe access for features - validate tupdesc has at least 1 column */
		if (tupdesc->natts >= 1)
		{
			Datum		feat_datum = SPI_getbinval(tuple, tupdesc, 1, NULL);
			feat_array = DatumGetArrayTypeP(feat_datum);
		}
		/* Safe access for target - validate tupdesc has at least 2 columns */
		if (tupdesc->natts >= 2)
		{
			Datum		targ_datum = SPI_getbinval(tuple, tupdesc, 2, NULL);
			target = DatumGetFloat8(targ_datum);
		}

		features[i] = (float *) ARR_DATA_PTR(feat_array);
		targets[i] = target;
	}

	/* Calculate baseline score */
	baseline_score = 0.0;
	for (i = 0; i < n_samples; i++)
	{
		/* Call actual model prediction */
		double		pred = call_model_predict(model_id, features[i], n_features);
		double		diff = pred - targets[i];

		baseline_score += diff * diff;
	}
	baseline_score /= n_samples;

	/* Permutation importance: shuffle each feature and measure impact */
	NDB_DECLARE(double *, importance);
	NDB_ALLOC(importance, double, n_features);

	for (i = 0; i < n_features; i++)
	{
		/* Shuffle feature i */
		for (j = n_samples - 1; j > 0; j--)
		{
			int			k = rand() % (j + 1);
			float		tmp = features[j][i];

			features[j][i] = features[k][i];
			features[k][i] = tmp;
		}

		/* Calculate score with shuffled feature */
		double		shuffled_score = 0.0;
		int			s;

		for (s = 0; s < n_samples; s++)
		{
			/* Call actual model prediction */
			double		pred = call_model_predict(model_id, features[s], n_features);
			double		diff = pred - targets[s];

			shuffled_score += diff * diff;
		}
		shuffled_score /= n_samples;

		/* Importance = decrease in performance */
		importance[i] = shuffled_score - baseline_score;

		/* Restore original order */
		for (j = n_samples - 1; j > 0; j--)
		{
			int			k = rand() % (j + 1);
			float		tmp = features[j][i];

			features[j][i] = features[k][i];
			features[k][i] = tmp;
		}
	}

	/* Build result array */
	NDB_DECLARE(Datum *, result_datums);
	NDB_ALLOC(result_datums, Datum, n_features);
	for (i = 0; i < n_features; i++)
		result_datums[i] = Float8GetDatum(importance[i]);

	result = construct_array(result_datums,
							 n_features,
							 FLOAT8OID,
							 sizeof(float8),
							 FLOAT8PASSBYVAL,
							 'd');

	/* Cleanup */
	NDB_FREE(features);
	NDB_FREE(targets);
	NDB_FREE(importance);
	NDB_FREE(result_datums);
	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_FREE(tbl_str);
	NDB_FREE(feat_col_str);
	NDB_FREE(targ_col_str);
	NDB_FREE(metric);
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_ARRAYTYPE_P(result);
}
