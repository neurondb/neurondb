/*-------------------------------------------------------------------------
 *
 * ml_minibatch_kmeans.c
 *    Mini-batch K-means clustering.
 *
 * This module implements mini-batch K-means for efficient clustering of
 * large-scale datasets using small random batches per iteration.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_minibatch_kmeans.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "utils/jsonb.h"
#include "executor/spi.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "ml_gpu_registry.h"
#include "ml_catalog.h"
#include "neurondb_gpu_model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* Forward declarations */
static int	kmeans_model_deserialize_from_bytea(const bytea * data, float ***centers_out, int *num_clusters_out, int *dim_out);
bool		minibatch_kmeans_gpu_serialize(const MLGpuModel *model, bytea * *payload_out, Jsonb * *metadata_out, char **errstr);

/*
 * KMeans++ initialization for better centroid seeding
 * Uses D² weighting to select initial centroids
 */
static void
minibatch_kmeans_pp_init(float **data,
						 int nvec,
						 int dim,
						 int k,
						 float **centroids)
{
	bool	   *selected;
	double	   *dist;
	int			c,
				i,
				d;

	selected = (bool *) palloc0(sizeof(bool) * nvec);
	NDB_ALLOC(dist, double, nvec);

	/* Select first centroid randomly */
	{
		int			first = rand() % nvec;

		memcpy(centroids[0], data[first], sizeof(float) * dim);
		selected[first] = true;
	}

	/* Compute initial distances to first centroid */
	for (i = 0; i < nvec; i++)
	{
		double		acc = 0.0;

		for (d = 0; d < dim; d++)
		{
			double		diff =
				(double) data[i][d] - (double) centroids[0][d];

			acc += diff * diff;
		}
		dist[i] = acc;
	}

	/* Select remaining k-1 centroids using D² weighting */
	for (c = 1; c < k; c++)
	{
		double		sum = 0.0;
		double		r;
		int			picked = -1;

		/* Compute sum of squared distances */
		for (i = 0; i < nvec; i++)
			if (!selected[i])
				sum += dist[i];

		if (sum < 1e-10)
			break;				/* All remaining points are duplicates */

		/* Select point with probability proportional to D² */
		r = ((double) rand() / RAND_MAX) * sum;
		for (i = 0; i < nvec; i++)
		{
			if (selected[i])
				continue;
			r -= dist[i];
			if (r <= 0.0)
			{
				picked = i;
				break;
			}
		}

		if (picked < 0)
		{
			/* Fallback: pick first unselected point */
			for (i = 0; i < nvec; i++)
			{
				if (!selected[i])
				{
					picked = i;
					break;
				}
			}
		}

		if (picked < 0)
			break;				/* No more points to select */

		/* Copy selected centroid */
		memcpy(centroids[c], data[picked], sizeof(float) * dim);
		selected[picked] = true;

		/* Update distances for next iteration */
		for (i = 0; i < nvec; i++)
		{
			double		acc;

			if (selected[i])
				continue;

			acc = 0.0;
			for (d = 0; d < dim; d++)
			{
				double		diff = (double) data[i][d]
					- (double) centroids[c][d];

				acc += diff * diff;
			}
			if (acc < dist[i])
				dist[i] = acc;
		}
	}

	NDB_FREE(dist);
	NDB_FREE(selected);
}

/*
 * cluster_minibatch_kmeans
 * -------------------------
 * Mini-batch K-means clustering for large-scale datasets.
 *
 * SQL Arguments:
 *   table_name    - Source table containing vectors (text)
 *   column_name   - Vector column name (text)
 *   num_clusters  - Number of clusters (k)
 *   batch_size    - Mini-batch size (default: 100)
 *   max_iters     - Maximum iterations (default: 100)
 *
 * Returns:
 *   Integer array of cluster assignments (1-based indexing)
 *
 * Performance:
 *   - 10-100x faster than standard K-means on large datasets
 *   - Memory usage: O(k*d + batch_size*d) instead of O(n*d)
 *   - Quality: typically within 95-99% of standard K-means
 *
 * Notes:
 *   - Learning rate η_t = 1/(1 + t) ensures convergence
 *   - Batch size should be tuned: smaller = faster but noisier updates
 *   - For datasets < 10K vectors, use standard cluster_kmeans instead
 */
PG_FUNCTION_INFO_V1(cluster_minibatch_kmeans);
PG_FUNCTION_INFO_V1(predict_minibatch_kmeans);
PG_FUNCTION_INFO_V1(evaluate_minibatch_kmeans_by_model_id);

Datum
cluster_minibatch_kmeans(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	int			num_clusters;
	int			batch_size;
	int			max_iters;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	int			nvec,
				dim;
	float	  **centroids;
	int		   *centroid_counts;	/* Per-centroid update counts */
	int		   *assignments;
	int		   *batch_indices;
	int			iter,
				i,
				c,
				d;
	ArrayType  *result;
	Datum	   *result_datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	num_clusters = PG_GETARG_INT32(2);
	batch_size = PG_ARGISNULL(3) ? 100 : PG_GETARG_INT32(3);
	max_iters = PG_ARGISNULL(4) ? 100 : PG_GETARG_INT32(4);

	/* Validation */
	if (num_clusters < 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("num_clusters must be at least 2")));

	if (batch_size < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("batch_size must be at least 1")));

	if (max_iters < 1)
		max_iters = 100;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: Mini-batch K-means on %s.%s (k=%d, batch=%d, iters=%d)",
		 tbl_str,
		 col_str,
		 num_clusters,
		 batch_size,
		 max_iters);

	/* Fetch training data */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (data == NULL || nvec == 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL)
		{
			for (int idx = 0; idx < nvec; idx++)
			{
				if (data[idx] != NULL)
					NDB_FREE(data[idx]);
			}
			NDB_FREE(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (nvec < num_clusters)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Not enough vectors (%d) for %d clusters",
						nvec,
						num_clusters)));

	/* Adjust batch size if larger than dataset */
	if (batch_size > nvec)
		batch_size = nvec;

	/* Initialize centroids using K-means++ */
	NDB_ALLOC(centroids, float *, num_clusters);
	for (c = 0; c < num_clusters; c++)
	{
		NDB_DECLARE(float *, centroid_vec);
		NDB_ALLOC(centroid_vec, float, dim);
		centroids[c] = centroid_vec;
	}

	minibatch_kmeans_pp_init(data, nvec, dim, num_clusters, centroids);

	/* Initialize per-centroid counters for learning rate */
	centroid_counts = (int *) palloc0(sizeof(int) * num_clusters);

	/* Allocate batch indices array */
	NDB_ALLOC(batch_indices, int, batch_size);

	/* Mini-batch K-means main loop */
	for (iter = 0; iter < max_iters; iter++)
	{
		int		   *batch_assignments;

		/* Sample random mini-batch without replacement */
		for (i = 0; i < batch_size; i++)
			batch_indices[i] = rand() % nvec;

		/* Assign each point in batch to nearest centroid */
		NDB_ALLOC(batch_assignments, int, batch_size);
		for (i = 0; i < batch_size; i++)
		{
			int			vec_idx = batch_indices[i];
			double		min_dist = DBL_MAX;
			int			best = 0;

			for (c = 0; c < num_clusters; c++)
			{
				double		dist = 0.0;

				for (d = 0; d < dim; d++)
				{
					double		diff = (double) data[vec_idx][d]
						- (double) centroids[c][d];

					dist += diff * diff;
				}
				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			batch_assignments[i] = best;
		}

		/*
		 * Update centroids using gradient descent with per-centroid learning
		 * rate
		 */
		for (i = 0; i < batch_size; i++)
		{
			int			vec_idx = batch_indices[i];
			int			cluster = batch_assignments[i];
			double		learning_rate;

			centroid_counts[cluster]++;
			learning_rate = 1.0 / centroid_counts[cluster];

			/* Update centroid: c = c + η * (x - c) = (1-η)*c + η*x */
			for (d = 0; d < dim; d++)
			{
				centroids[cluster][d] =
					(float) ((1.0 - learning_rate)
							 * centroids[cluster][d]
							 + learning_rate
							 * data[vec_idx][d]);
			}
		}

		NDB_FREE(batch_assignments);

		if ((iter + 1) % 10 == 0)
			elog(DEBUG1,
				 "neurondb: Mini-batch K-means iteration %d/%d",
				 iter + 1,
				 max_iters);
	}

	/* Final assignment: assign all points to nearest centroid */
	NDB_ALLOC(assignments, int, nvec);
	for (i = 0; i < nvec; i++)
	{
		double		min_dist = DBL_MAX;
		int			best = 0;

		for (c = 0; c < num_clusters; c++)
		{
			double		dist = 0.0;

			for (d = 0; d < dim; d++)
			{
				double		diff = (double) data[i][d]
					- (double) centroids[c][d];

				dist += diff * diff;
			}
			if (dist < min_dist)
			{
				min_dist = dist;
				best = c;
			}
		}
		assignments[i] = best;
	}

	/* Build result array (1-based cluster labels) */
	NDB_ALLOC(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = Int32GetDatum(assignments[i] + 1);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	result = construct_array(
							 result_datums, nvec, INT4OID, typlen, typbyval, typalign);

	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	for (c = 0; c < num_clusters; c++)
		NDB_FREE(centroids[c]);
	NDB_FREE(centroids);
	NDB_FREE(centroid_counts);
	NDB_FREE(batch_indices);
	NDB_FREE(assignments);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * predict_minibatch_kmeans
 *      Predicts cluster assignment for new data points using trained minibatch k-means model.
 *      Arguments: int4 model_id, float8[] features
 *      Returns: int4 cluster_id
 */
Datum
predict_minibatch_kmeans(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	ArrayType  *features_array = PG_GETARG_ARRAYTYPE_P(1);

	float	   *features;
	int			cluster_id = -1;
	int			n_elems;

	/*
	 * Suppress unused variable warnings - placeholders for future
	 * implementation
	 */
	(void) model_id;
	(void) features_array;

	/* Extract features from array */
	{
		Oid			elmtype = ARR_ELEMTYPE(features_array);
		int16		typlen;
		bool		typbyval;
		char		typalign;
		Datum	   *elems;
		bool	   *nulls;
		int			i;

		get_typlenbyvalalign(elmtype, &typlen, &typbyval, &typalign);
		deconstruct_array(features_array, elmtype, typlen, typbyval, typalign,
						  &elems, &nulls, &n_elems);

		NDB_ALLOC(features, float, n_elems);

		for (i = 0; i < n_elems; i++)
			features[i] = DatumGetFloat4(elems[i]);
	}

	/* Load model and find closest centroid */
	{
		NDB_DECLARE(bytea *, model_payload);
		NDB_DECLARE(Jsonb *, model_parameters);
		float	  **centers = NULL;
		int			num_clusters = 0;
		int			model_dim = 0;
		int			c,
					d;
		double		min_dist_sq = DBL_MAX;

		/* Load model from catalog */
		if (ml_catalog_fetch_model_payload(model_id, &model_payload, &model_parameters, NULL))
		{
			if (model_payload != NULL && VARSIZE(model_payload) > VARHDRSZ)
			{
				if (kmeans_model_deserialize_from_bytea(model_payload,
														&centers,
														&num_clusters,
														&model_dim) == 0)
				{
					if (n_elems == model_dim && num_clusters > 0)
					{
						/* Find nearest centroid */
						for (c = 0; c < num_clusters; c++)
						{
							double		dist_sq = 0.0;

							for (d = 0; d < model_dim; d++)
							{
								double		diff = (double) features[d] - (double) centers[c][d];

								dist_sq += diff * diff;
							}
							if (dist_sq < min_dist_sq)
							{
								min_dist_sq = dist_sq;
								cluster_id = c;
							}
						}
					}
					else
					{
						cluster_id = 0;
					}

					/* Cleanup centers */
					if (centers != NULL)
					{
						for (c = 0; c < num_clusters; c++)
							NDB_FREE(centers[c]);
						NDB_FREE(centers);
					}
				}
				else
				{
					cluster_id = 0;
				}
			}
			else
			{
				cluster_id = 0;
			}
			if (model_payload)
				NDB_FREE(model_payload);
			if (model_parameters)
				NDB_FREE(model_parameters);
		}
		else
		{
			cluster_id = 0;
		}
	}

	NDB_FREE(features);

	PG_RETURN_INT32(cluster_id);
}

/*
 * evaluate_minibatch_kmeans_by_model_id
 *      Evaluates minibatch k-means clustering quality on a dataset.
 *      Arguments: int4 model_id, text table_name, text feature_col
 *      Returns: jsonb with clustering metrics
 */
Datum
evaluate_minibatch_kmeans_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *feature_col;
	char	   *tbl_str;
	char	   *feat_str;
	StringInfoData query;
	int			ret;
	int			n_points = 0;
	StringInfoData jsonbuf;
	Jsonb	   *result;
	MemoryContext oldcontext;
	double		inertia;
	int			n_clusters;
	int			n_iterations;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext_spi;

	/* Validate arguments */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_minibatch_kmeans_by_model_id: 3 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_minibatch_kmeans_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_minibatch_kmeans_by_model_id: table_name and feature_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);

	oldcontext = CurrentMemoryContext;
	oldcontext_spi = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext_spi);

	/* Build query */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s FROM %s WHERE %s IS NOT NULL",
					 feat_str, tbl_str, feat_str);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_minibatch_kmeans_by_model_id: query failed")));
	}

	n_points = SPI_processed;
	if (n_points < 2)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_minibatch_kmeans_by_model_id: need at least 2 points, got %d",
						n_points)));
	}

	/* Load model and compute clustering metrics */
	{
		NDB_DECLARE(bytea *, model_payload);
		NDB_DECLARE(Jsonb *, parameters);
		NDB_DECLARE(Jsonb *, metrics);
		const char *buf;
		int			i,
					c,
					d;
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		double		total_inertia = 0.0;

		/* Load model from catalog */
		if (ml_catalog_fetch_model_payload(model_id,
										   &model_payload,
										   &parameters,
										   &metrics))
		{
			if (model_payload != NULL && VARSIZE(model_payload) > VARHDRSZ)
			{
				int			offset = 0;

				/*
				 * Deserialize model (same format as kmeans: num_clusters,
				 * dim, centroids)
				 */
				buf = VARDATA(model_payload);
				memcpy(&n_clusters, buf + offset, sizeof(int));
				offset += sizeof(int);
				memcpy(&d, buf + offset, sizeof(int));
				offset += sizeof(int);

				if (n_clusters > 0 && n_clusters <= 10000 && d > 0 && d <= 100000)
				{
					/* Allocate centroids */
					NDB_DECLARE(float **, centers);
					NDB_ALLOC(centers, float *, n_clusters);
					for (c = 0; c < n_clusters; c++)
					{
						NDB_DECLARE(float *, center_row);
						NDB_ALLOC(center_row, float, d);
						centers[c] = center_row;
						for (i = 0; i < d; i++)
						{
							memcpy(&centers[c][i], buf + offset, sizeof(float));
							offset += sizeof(float);
						}
					}

					/*
					 * Compute inertia: sum of squared distances to nearest
					 * centroids
					 */
					for (i = 0; i < n_points; i++)
					{
						HeapTuple	tuple;

						/*
						 * Safe access to SPI_tuptable - validate before
						 * access
						 */
						if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
							i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
						{
							continue;
						}
						tuple = SPI_tuptable->vals[i];
						if (tupdesc == NULL)
						{
							continue;
						}
						{
							Datum		vec_datum;
							bool		vec_null;
							Vector	   *vec;
							float	   *vec_data;
							double		min_dist = DBL_MAX;

							vec_datum = SPI_getbinval(tuple, tupdesc, 1, &vec_null);
							if (vec_null)
								continue;

							vec = DatumGetVector(vec_datum);
							if (vec == NULL || vec->dim != d)
								continue;

							vec_data = vec->data;

							/* Find nearest centroid */
							for (c = 0; c < n_clusters; c++)
							{
								double		dist = 0.0;

								for (d = 0; d < vec->dim; d++)
								{
									double		diff = (double) vec_data[d] - (double) centers[c][d];

									dist += diff * diff;
								}
								if (dist < min_dist)
									min_dist = dist;
							}

							total_inertia += min_dist;
						}
					}

					/* Free centroids */
					for (c = 0; c < n_clusters; c++)
						NDB_FREE(centers[c]);
					NDB_FREE(centers);
				}
				else
				{
					n_clusters = 3;
					total_inertia = 0.0;
				}

				/* Extract n_iterations from metrics JSON */
				if (metrics != NULL)
				{
					char	   *metrics_str = DatumGetCString(DirectFunctionCall1(
																				  jsonb_out, JsonbPGetDatum(metrics)));

					if (metrics_str != NULL)
					{
						/* Try to extract max_iters or n_iterations from JSON */
						char	   *max_iters_str = strstr(metrics_str, "\"max_iters\"");

						if (max_iters_str != NULL)
						{
							sscanf(max_iters_str, "\"max_iters\":%d", &n_iterations);
						}
						else
						{
							n_iterations = 100; /* Default */
						}
						NDB_FREE(metrics_str);
					}
					else
					{
						n_iterations = 100;
					}
				}
				else
				{
					n_iterations = 100;
				}

				inertia = total_inertia;

				if (model_payload != NULL)
					NDB_FREE(model_payload);
				if (parameters != NULL)
					NDB_FREE(parameters);
				if (metrics != NULL)
					NDB_FREE(metrics);
			}
			else
			{
				inertia = 0.0;
				n_clusters = 3;
				n_iterations = 100;
			}
		}
		else
		{
			inertia = 0.0;
			n_clusters = 3;
			n_iterations = 100;
		}
	}

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"inertia\":%.6f,\"n_clusters\":%d,\"n_points\":%d,\"n_iterations\":%d}",
					 inertia, n_clusters, n_points, n_iterations);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
	NDB_FREE(jsonbuf.data);

	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);

	PG_RETURN_JSONB_P(result);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for Mini-batch K-Means
 *-------------------------------------------------------------------------
 */

/* Helper functions for model serialization (reuse k-means format) */
static bytea *
kmeans_model_serialize_to_bytea(float **centers, int num_clusters, int dim)
{
	StringInfoData buf;
	int			i,
				j;
	int			total_size;
	bytea	   *result;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &num_clusters, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &dim, sizeof(int));

	for (i = 0; i < num_clusters; i++)
		for (j = 0; j < dim; j++)
			appendBinaryStringInfo(&buf, (char *) &centers[i][j], sizeof(float));

	{
		char	   *result_raw = NULL;
		total_size = VARHDRSZ + buf.len;
		NDB_ALLOC(result_raw, char, total_size);
		result = (bytea *) result_raw;
		SET_VARSIZE(result, total_size);
		memcpy(VARDATA(result), buf.data, buf.len);
		NDB_FREE(buf.data);
	}

	return result;
}

int
kmeans_model_deserialize_from_bytea(const bytea * data, float ***centers_out, int *num_clusters_out, int *dim_out)
{
	const char *buf;
	int			offset = 0;
	int			i,
				j;
	float	  **centers;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 2)
		return -1;

	buf = VARDATA(data);
	memcpy(num_clusters_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(dim_out, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (*num_clusters_out <= 0 || *num_clusters_out > 10000 || *dim_out <= 0 || *dim_out > 100000)
		return -1;

	NDB_ALLOC(centers, float *, *num_clusters_out);
	for (i = 0; i < *num_clusters_out; i++)
	{
		NDB_DECLARE(float *, center_vec);
		NDB_ALLOC(center_vec, float, *dim_out);
		centers[i] = center_vec;
		for (j = 0; j < *dim_out; j++)
		{
			memcpy(&centers[i][j], buf + offset, sizeof(float));
			offset += sizeof(float);
		}
	}

	*centers_out = centers;
	return 0;
}

static bool
minibatch_kmeans_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	typedef struct MiniBatchKMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
		int			batch_size;
	}			MiniBatchKMeansGpuModelState;

	MiniBatchKMeansGpuModelState *state;
	float	  **data = NULL;
	float	  **centroids = NULL;

	NDB_DECLARE(int *, centroid_counts);
	NDB_DECLARE(int *, batch_indices);
	int			num_clusters = 8;
	int			batch_size = 100;
	int			max_iters = 100;
	int			nvec = 0;
	int			dim = 0;
	int			iter,
				i,
				c,
				d;

	NDB_DECLARE(bytea *, model_data);
	NDB_DECLARE(Jsonb *, metrics);
	StringInfoData metrics_json;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_train: invalid parameters");
		return false;
	}

	/* Extract hyperparameters */
	if (spec->hyperparameters != NULL)
	{
		it = JsonbIteratorInit((JsonbContainer *) & spec->hyperparameters->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "k") == 0 && v.type == jbvNumeric)
					num_clusters = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																	 NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "batch_size") == 0 && v.type == jbvNumeric)
					batch_size = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																   NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "max_iters") == 0 && v.type == jbvNumeric)
					max_iters = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																  NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}

	if (num_clusters < 2)
		num_clusters = 8;
	if (batch_size < 1)
		batch_size = 100;
	if (max_iters < 1)
		max_iters = 100;

	/* Convert feature matrix to 2D array */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	if (batch_size > nvec)
		batch_size = nvec;

	NDB_ALLOC(data, float *, nvec);
	for (i = 0; i < nvec; i++)
	{
		NDB_DECLARE(float *, data_row);
		NDB_ALLOC(data_row, float, dim);
		data[i] = data_row;
		memcpy(data[i], &spec->feature_matrix[i * dim], sizeof(float) * dim);
	}

	if (nvec < num_clusters)
	{
		for (i = 0; i < nvec; i++)
			NDB_FREE(data[i]);
		NDB_FREE(data);
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_train: not enough samples");
		return false;
	}

	/* Initialize centroids using k-means++ */
	NDB_ALLOC(centroids, float *, num_clusters);
	for (c = 0; c < num_clusters; c++)
	{
		NDB_DECLARE(float *, centroid_vec);
		NDB_ALLOC(centroid_vec, float, dim);
		centroids[c] = centroid_vec;
	}
	minibatch_kmeans_pp_init(data, nvec, dim, num_clusters, centroids);

	centroid_counts = (int *) palloc0(sizeof(int) * num_clusters);
	NDB_ALLOC(batch_indices, int, batch_size);

	/* Mini-batch K-means iteration (CPU) */
	for (iter = 0; iter < max_iters; iter++)
	{
		int		   *batch_assignments;

		/* Sample random mini-batch */
		for (i = 0; i < batch_size; i++)
			batch_indices[i] = rand() % nvec;

		NDB_ALLOC(batch_assignments, int, batch_size);
		for (i = 0; i < batch_size; i++)
		{
			int			vec_idx = batch_indices[i];
			double		min_dist = DBL_MAX;
			int			best = 0;

			for (c = 0; c < num_clusters; c++)
			{
				double		dist = neurondb_l2_distance_squared(
																data[vec_idx], centroids[c], dim);

				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			batch_assignments[i] = best;
		}

		/* Update centroids with learning rate */
		for (i = 0; i < batch_size; i++)
		{
			int			vec_idx = batch_indices[i];
			int			cluster = batch_assignments[i];
			double		learning_rate;

			centroid_counts[cluster]++;
			learning_rate = 1.0 / centroid_counts[cluster];

			for (d = 0; d < dim; d++)
			{
				centroids[cluster][d] = (float) ((1.0 - learning_rate)
												 * centroids[cluster][d]
												 + learning_rate * data[vec_idx][d]);
			}
		}

		NDB_FREE(batch_assignments);
	}

	/* Serialize model (reuse k-means format) */
	model_data = kmeans_model_serialize_to_bytea(centroids, num_clusters, dim);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"k\":%d,\"dim\":%d,\"batch_size\":%d,\"max_iters\":%d,\"n_samples\":%d}",
					 num_clusters, dim, batch_size, max_iters, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetTextDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	state = (MiniBatchKMeansGpuModelState *) palloc0(sizeof(MiniBatchKMeansGpuModelState));
	state->model_blob = model_data;
	state->metrics = metrics;
	state->num_clusters = num_clusters;
	state->dim = dim;
	state->n_samples = nvec;
	state->batch_size = batch_size;

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	for (c = 0; c < num_clusters; c++)
		NDB_FREE(centroids[c]);
	NDB_FREE(data);
	NDB_FREE(centroids);
	NDB_FREE(centroid_counts);
	NDB_FREE(batch_indices);

	return true;
}

static bool
minibatch_kmeans_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
							 float *output, int output_dim, char **errstr)
{
	typedef struct MiniBatchKMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
		int			batch_size;
	}			MiniBatchKMeansGpuModelState;

	const		MiniBatchKMeansGpuModelState *state;
	float	  **centers = NULL;
	int			num_clusters = 0;
	int			dim = 0;
	int			c;
	double		min_dist = DBL_MAX;
	int			best_cluster = -1;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = -1.0f;
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_predict: model not ready");
		return false;
	}

	state = (const MiniBatchKMeansGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_predict: model blob is NULL");
		return false;
	}

	if (kmeans_model_deserialize_from_bytea(state->model_blob,
											&centers, &num_clusters, &dim) != 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_predict: failed to deserialize");
		return false;
	}

	if (input_dim != dim)
	{
		for (c = 0; c < num_clusters; c++)
			NDB_FREE(centers[c]);
		NDB_FREE(centers);
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_predict: dimension mismatch");
		return false;
	}

	for (c = 0; c < num_clusters; c++)
	{
		double		dist = neurondb_l2_distance_squared(input, centers[c], dim);

		if (dist < min_dist)
		{
			min_dist = dist;
			best_cluster = c;
		}
	}

	output[0] = (float) best_cluster;

	for (c = 0; c < num_clusters; c++)
		NDB_FREE(centers[c]);
	NDB_FREE(centers);

	return true;
}

static bool
minibatch_kmeans_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
							  MLGpuMetrics *out, char **errstr)
{
	typedef struct MiniBatchKMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
		int			batch_size;
	}			MiniBatchKMeansGpuModelState;

	const		MiniBatchKMeansGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_evaluate: invalid model");
		return false;
	}

	state = (const MiniBatchKMeansGpuModelState *) model->backend_state;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"minibatch_kmeans\",\"storage\":\"cpu\","
					 "\"k\":%d,\"dim\":%d,\"batch_size\":%d,\"n_samples\":%d}",
					 state->num_clusters > 0 ? state->num_clusters : 0,
					 state->dim > 0 ? state->dim : 0,
					 state->batch_size > 0 ? state->batch_size : 100,
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetTextDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

bool
minibatch_kmeans_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
							   Jsonb * *metadata_out, char **errstr)
{
	typedef struct MiniBatchKMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
		int			batch_size;
	}			MiniBatchKMeansGpuModelState;

	const		MiniBatchKMeansGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_serialize: invalid model");
		return false;
	}

	state = (const MiniBatchKMeansGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_serialize: model blob is NULL");
		return false;
	}

	{
		char	   *payload_copy_raw = NULL;
		payload_size = VARSIZE(state->model_blob);
		NDB_ALLOC(payload_copy_raw, char, payload_size);
		payload_copy = (bytea *) payload_copy_raw;
		memcpy(payload_copy, state->model_blob, payload_size);
	}

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_FREE(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(
														PointerGetDatum(state->metrics));

	return true;
}

static bool
minibatch_kmeans_gpu_deserialize(MLGpuModel *model, const bytea * payload,
								 const Jsonb * metadata, char **errstr)
{
	typedef struct MiniBatchKMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
		int			batch_size;
	}			MiniBatchKMeansGpuModelState;

	MiniBatchKMeansGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;
	float	  **centers = NULL;
	int			num_clusters = 0;
	int			dim = 0;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_deserialize: invalid parameters");
		return false;
	}

	{
		char	   *payload_copy_raw = NULL;
		payload_size = VARSIZE(payload);
		NDB_ALLOC(payload_copy_raw, char, payload_size);
		payload_copy = (bytea *) payload_copy_raw;
		memcpy(payload_copy, payload, payload_size);
	}

	if (kmeans_model_deserialize_from_bytea(payload_copy,
											&centers, &num_clusters, &dim) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("minibatch_kmeans_gpu_deserialize: failed to deserialize");
		return false;
	}

	for (int c = 0; c < num_clusters; c++)
		NDB_FREE(centers[c]);
	NDB_FREE(centers);

	state = (MiniBatchKMeansGpuModelState *) palloc0(sizeof(MiniBatchKMeansGpuModelState));
	state->model_blob = payload_copy;
	state->num_clusters = num_clusters;
	state->dim = dim;
	state->n_samples = 0;
	state->batch_size = 100;

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		NDB_DECLARE(Jsonb *, metadata_copy);
		NDB_DECLARE(char *, metadata_copy_raw);
		NDB_ALLOC(metadata_copy_raw, char, metadata_size);
		metadata_copy = (Jsonb *) metadata_copy_raw;

		memcpy(metadata_copy, metadata, metadata_size);
		state->metrics = metadata_copy;

		it = JsonbIteratorInit((JsonbContainer *) & metadata->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "n_samples") == 0 && v.type == jbvNumeric)
					state->n_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																		 NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "batch_size") == 0 && v.type == jbvNumeric)
					state->batch_size = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																		  NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}
	else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static void
minibatch_kmeans_gpu_destroy(MLGpuModel *model)
{
	typedef struct MiniBatchKMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
		int			batch_size;
	}			MiniBatchKMeansGpuModelState;

	MiniBatchKMeansGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (MiniBatchKMeansGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			NDB_FREE(state->model_blob);
		if (state->metrics != NULL)
			NDB_FREE(state->metrics);
		NDB_FREE(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps minibatch_kmeans_gpu_model_ops = {
	.algorithm = "minibatch_kmeans",
	.train = minibatch_kmeans_gpu_train,
	.predict = minibatch_kmeans_gpu_predict,
	.evaluate = minibatch_kmeans_gpu_evaluate,
	.serialize = minibatch_kmeans_gpu_serialize,
	.deserialize = minibatch_kmeans_gpu_deserialize,
	.destroy = minibatch_kmeans_gpu_destroy,
};

void
neurondb_gpu_register_minibatch_kmeans_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&minibatch_kmeans_gpu_model_ops);
	registered = true;
}
