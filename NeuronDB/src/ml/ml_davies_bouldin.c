/*-------------------------------------------------------------------------
 *
 * ml_davies_bouldin.c
 *    Davies-Bouldin index for cluster quality evaluation.
 *
 * This module implements the Davies-Bouldin index to measure cluster quality,
 * with lower values indicating better separated clusters.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_davies_bouldin.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "executor/spi.h"
#include "neurondb_simd.h"
#include "catalog/pg_type.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#include <math.h>
#include <float.h>

/*
 * euclidean_distance
 *	 Compute Euclidean distance between two vectors.
 */
static inline double
euclidean_distance(const float *a, const float *b, int dim)
{
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double		diff = (double) a[i] - (double) b[i];

		sum += diff * diff;
	}
	return sqrt(sum);
}

PG_FUNCTION_INFO_V1(davies_bouldin_index);

Datum
davies_bouldin_index(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	text	   *cluster_column;
	char	   *tbl_str;
	char	   *vec_col_str;
	char	   *cluster_col_str;
	float	  **vectors;
	NDB_DECLARE(int *, cluster_labels);
	int			nvec;
	int			dim;
	int			num_clusters;
	NDB_DECLARE(int *, cluster_sizes);
	NDB_DECLARE(float **, cluster_centroids);
	NDB_DECLARE(double *, cluster_scatter);
	double		db_index;
	int			i;
	int			c;
	int			d;
	StringInfoData sql;
	int			ret;
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;
#include "ml_gpu_registry.h"

	/* Parse input arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	cluster_column = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	vec_col_str = text_to_cstring(vector_column);
	cluster_col_str = text_to_cstring(cluster_column);

	elog(DEBUG1,
		 "neurondb: Computing Davies-Bouldin index for %s.%s (clusters=%s)",
		 tbl_str, vec_col_str, cluster_col_str);

	vectors = neurondb_fetch_vectors_from_table(tbl_str, vec_col_str, &nvec, &dim);

	if (vectors == NULL || nvec == 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(vec_col_str);
		NDB_FREE(cluster_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(vec_col_str);
		NDB_FREE(cluster_col_str);
		/* Free vectors array and rows if vectors is not NULL */
		if (vectors != NULL)
		{
			for (int i = 0; i < nvec; i++)
			{
				if (vectors[i] != NULL)
					NDB_FREE(vectors[i]);
			}
			NDB_FREE(vectors);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (nvec < 2)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 2 vectors for DBI calculation")));

	oldcontext = CurrentMemoryContext;

	NDB_ALLOC(cluster_labels, int, nvec);

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql, "SELECT %s FROM %s ORDER BY ctid", cluster_col_str, tbl_str);
	ret = ndb_spi_execute(spi_session, sql.data, true, 0);

	if (ret != SPI_OK_SELECT || (int) SPI_processed != nvec)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(cluster_labels);
		NDB_FREE(tbl_str);
		NDB_FREE(cluster_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Failed to fetch cluster labels (expected %d, got %d)",
						nvec, (int) SPI_processed)));
	}

	num_clusters = 0;
	for (i = 0; i < nvec; i++)
	{
		int32		val;

		if (ndb_spi_get_int32(spi_session, i, 1, &val))
		{
			cluster_labels[i] = val;
			if (cluster_labels[i] > num_clusters)
				num_clusters = cluster_labels[i];
		}
		else
		{
			cluster_labels[i] = -1;
		}
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	if (num_clusters < 2)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 2 clusters for DBI (found %d)", num_clusters)));


	NDB_ALLOC(cluster_sizes, int, num_clusters);
	NDB_ALLOC(cluster_centroids, float *, num_clusters);

	for (c = 0; c < num_clusters; c++)
	{
		NDB_DECLARE(float *, centroid_row);
		NDB_ALLOC(centroid_row, float, dim);
		cluster_centroids[c] = centroid_row;
	}

	for (i = 0; i < nvec; i++)
	{
		int			cluster = cluster_labels[i];

		if (cluster < 1 || cluster > num_clusters)
			continue;

		cluster = cluster - 1;

		cluster_sizes[cluster]++;

		for (d = 0; d < dim; d++)
			cluster_centroids[cluster][d] += vectors[i][d];
	}

	for (c = 0; c < num_clusters; c++)
	{
		if (cluster_sizes[c] > 0)
		{
			for (d = 0; d < dim; d++)
				cluster_centroids[c][d] /= cluster_sizes[c];
		}
	}

	NDB_ALLOC(cluster_scatter, double, num_clusters);

	for (i = 0; i < nvec; i++)
	{
		int			cluster = cluster_labels[i];

		if (cluster < 1 || cluster > num_clusters)
			continue;

		cluster = cluster - 1;
		cluster_scatter[cluster] += euclidean_distance(
													   vectors[i], cluster_centroids[cluster], dim);
	}

	for (c = 0; c < num_clusters; c++)
	{
		if (cluster_sizes[c] > 0)
			cluster_scatter[c] /= cluster_sizes[c];
	}

	db_index = 0.0;
	{
		int			valid_clusters = 0;

		for (i = 0; i < num_clusters; i++)
		{
			double		max_ratio = 0.0;
			int			j;

			if (cluster_sizes[i] < 2)
				continue;

			for (j = 0; j < num_clusters; j++)
			{
				double		centroid_dist;
				double		ratio;

				if (i == j || cluster_sizes[j] < 2)
					continue;

				centroid_dist = euclidean_distance(cluster_centroids[i],
												   cluster_centroids[j],
												   dim);

				if (centroid_dist < 1e-10)
					continue;

				ratio = (cluster_scatter[i] + cluster_scatter[j]) / centroid_dist;

				if (ratio > max_ratio)
					max_ratio = ratio;
			}

			db_index += max_ratio;
			valid_clusters++;
		}

		if (valid_clusters > 0)
			db_index /= valid_clusters;
		else
			db_index = -1.0;
	}


	for (i = 0; i < nvec; i++)
		NDB_FREE(vectors[i]);
	NDB_FREE(vectors);
	NDB_FREE(cluster_labels);
	NDB_FREE(cluster_sizes);

	for (c = 0; c < num_clusters; c++)
		NDB_FREE(cluster_centroids[c]);
	NDB_FREE(cluster_centroids);

	NDB_FREE(cluster_scatter);
	NDB_FREE(tbl_str);
	NDB_FREE(vec_col_str);
	NDB_FREE(cluster_col_str);

	PG_RETURN_FLOAT8(db_index);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for Davies-Bouldin
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

typedef struct DaviesBouldinGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	float	  **centroids;
	int		   *cluster_sizes;
	int			n_clusters;
	int			dim;
	int			n_samples;
}			DaviesBouldinGpuModelState;

static bytea *
davies_bouldin_model_serialize_to_bytea(float **centroids, int *cluster_sizes, int n_clusters, int dim)
{
	StringInfoData buf;
	int			i,
				j;
	int			total_size;
	bytea	   *result;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &n_clusters, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &dim, sizeof(int));
	for (i = 0; i < n_clusters; i++)
		appendBinaryStringInfo(&buf, (char *) &cluster_sizes[i], sizeof(int));
	for (i = 0; i < n_clusters; i++)
		for (j = 0; j < dim; j++)
			appendBinaryStringInfo(&buf, (char *) &centroids[i][j], sizeof(float));

	total_size = VARHDRSZ + buf.len;
	result = (bytea *) palloc(total_size);
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

static int
davies_bouldin_model_deserialize_from_bytea(const bytea * data, float ***centroids_out, int **cluster_sizes_out, int *n_clusters_out, int *dim_out)
{
	const char *buf;
	int			offset = 0;
	int			i,
				j;
	float	  **centroids;
	int		   *cluster_sizes;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 2)
		return -1;

	buf = VARDATA(data);
	memcpy(n_clusters_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(dim_out, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (*n_clusters_out < 0 || *n_clusters_out > 10000 || *dim_out <= 0 || *dim_out > 100000)
		return -1;

	cluster_sizes = (int *) palloc(sizeof(int) * *n_clusters_out);
	for (i = 0; i < *n_clusters_out; i++)
	{
		memcpy(&cluster_sizes[i], buf + offset, sizeof(int));
		offset += sizeof(int);
	}

	centroids = (float **) palloc(sizeof(float *) * *n_clusters_out);
	for (i = 0; i < *n_clusters_out; i++)
	{
		centroids[i] = (float *) palloc(sizeof(float) * *dim_out);
		for (j = 0; j < *dim_out; j++)
		{
			memcpy(&centroids[i][j], buf + offset, sizeof(float));
			offset += sizeof(float);
		}
	}

	*centroids_out = centroids;
	*cluster_sizes_out = cluster_sizes;
	return 0;
}

static bool
davies_bouldin_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	DaviesBouldinGpuModelState *state;
	float	  **data = NULL;
	NDB_DECLARE(int *, labels);
	float	  **centroids = NULL;
	NDB_DECLARE(int *, cluster_sizes);
	int			num_clusters = 8;
	int			nvec = 0;
	int			dim = 0;
	int			i,
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
			*errstr = pstrdup("davies_bouldin_gpu_train: invalid parameters");
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
				if (strcmp(key, "n_clusters") == 0 && v.type == jbvNumeric)
					num_clusters = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																	 NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}

	if (num_clusters < 2)
		num_clusters = 8;

	/* Convert feature matrix to 2D array */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("davies_bouldin_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	if (nvec < num_clusters)
	{
		if (errstr != NULL)
			*errstr = pstrdup("davies_bouldin_gpu_train: not enough samples");
		return false;
	}

	data = (float **) palloc(sizeof(float *) * nvec);
	for (i = 0; i < nvec; i++)
	{
		data[i] = (float *) palloc(sizeof(float) * dim);
		memcpy(data[i], &spec->feature_matrix[i * dim], sizeof(float) * dim);
	}

	/* Simple K-means clustering to get cluster assignments */
	labels = (int *) palloc0(sizeof(int) * nvec);
	centroids = (float **) palloc(sizeof(float *) * num_clusters);
	cluster_sizes = (int *) palloc0(sizeof(int) * num_clusters);

	/* Initialize centroids randomly */
	{
		int			rand_idx;

		for (c = 0; c < num_clusters; c++)
		{
			centroids[c] = (float *) palloc(sizeof(float) * dim);
			rand_idx = (c * nvec) / num_clusters;
			memcpy(centroids[c], data[rand_idx], sizeof(float) * dim);
		}
	}

	/* K-means iteration */
	for (int iter = 0; iter < 100; iter++)
	{
		bool		changed = false;

		/* Assignment */
		for (i = 0; i < nvec; i++)
		{
			double		min_dist = DBL_MAX;
			int			best = 0;

			for (c = 0; c < num_clusters; c++)
			{
				double		dist = sqrt(neurondb_l2_distance_squared(data[i], centroids[c], dim));

				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			if (labels[i] != best)
			{
				labels[i] = best;
				changed = true;
			}
		}

		if (!changed)
			break;

		/* Update centroids */
		for (c = 0; c < num_clusters; c++)
		{
			memset(centroids[c], 0, sizeof(float) * dim);
			cluster_sizes[c] = 0;
		}

		for (i = 0; i < nvec; i++)
		{
			c = labels[i];
			for (d = 0; d < dim; d++)
				centroids[c][d] += data[i][d];
			cluster_sizes[c]++;
		}

		for (c = 0; c < num_clusters; c++)
		{
			if (cluster_sizes[c] > 0)
			{
				for (d = 0; d < dim; d++)
					centroids[c][d] /= cluster_sizes[c];
			}
		}
	}

	/* Serialize model */
	model_data = davies_bouldin_model_serialize_to_bytea(centroids, cluster_sizes, num_clusters, dim);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"n_clusters\":%d,\"dim\":%d,\"n_samples\":%d}",
					 num_clusters, dim, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	state = (DaviesBouldinGpuModelState *) palloc0(sizeof(DaviesBouldinGpuModelState));
	state->model_blob = model_data;
	state->metrics = metrics;
	state->centroids = centroids;
	state->cluster_sizes = cluster_sizes;
	state->n_clusters = num_clusters;
	state->dim = dim;
	state->n_samples = nvec;

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	/* Cleanup temp data */
	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	NDB_FREE(labels);

	return true;
}

static bool
davies_bouldin_gpu_predict(const MLGpuModel * model, const float *input, int input_dim,
						   float *output, int output_dim, char **errstr)
{
	const		DaviesBouldinGpuModelState *state;
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
			*errstr = pstrdup("davies_bouldin_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("davies_bouldin_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("davies_bouldin_gpu_predict: model not ready");
		return false;
	}

	state = (const DaviesBouldinGpuModelState *) model->backend_state;
	if (state->centroids == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("davies_bouldin_gpu_predict: centroids are NULL");
		return false;
	}

	if (input_dim != state->dim)
	{
		if (errstr != NULL)
			*errstr = pstrdup("davies_bouldin_gpu_predict: dimension mismatch");
		return false;
	}

	for (c = 0; c < state->n_clusters; c++)
	{
		double		dist = sqrt(neurondb_l2_distance_squared(input, state->centroids[c], state->dim));

		if (dist < min_dist)
		{
			min_dist = dist;
			best_cluster = c;
		}
	}

	output[0] = (float) best_cluster;

	return true;
}

static bool
davies_bouldin_gpu_evaluate(const MLGpuModel * model, const MLGpuEvalSpec * spec,
							MLGpuMetrics * out, char **errstr)
{
	const		DaviesBouldinGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;
	double		db_index = 0.0;
	int			i,
				j,
				c;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("davies_bouldin_gpu_evaluate: invalid model");
		return false;
	}

	state = (const DaviesBouldinGpuModelState *) model->backend_state;

	/* Compute Davies-Bouldin index */
	if (state->centroids != NULL && state->n_clusters > 1)
	{
		double	   *scatter = (double *) palloc0(sizeof(double) * state->n_clusters);

		/* Compute scatter for each cluster */
		/* Scatter is the average distance from points to cluster centroid */

		/*
		 * If evaluation table is provided, fetch data and compute actual
		 * scatter
		 */
		if (spec != NULL && spec->evaluation_table != NULL)
		{
			/* Fetch data from evaluation table and compute scatter */
			float	  **data = NULL;
			int			nvec = 0;
			int			data_dim = 0;
			NDB_DECLARE(int *, cluster_assignments);
			NDB_DECLARE(int *, cluster_counts);
			int			s;
			char	   *vec_col = "vector"; /* Default vector column name */

			/* Try to fetch vectors from evaluation table */
			data = neurondb_fetch_vectors_from_table(spec->evaluation_table, vec_col, &nvec, &data_dim);

			if (data != NULL && nvec > 0 && data_dim == state->dim)
			{
				cluster_assignments = (int *) palloc0(sizeof(int) * nvec);
				cluster_counts = (int *) palloc0(sizeof(int) * state->n_clusters);

				/* Assign points to nearest centroids */
				for (s = 0; s < nvec; s++)
				{
					double		min_dist = DBL_MAX;
					int			best = 0;

					for (c = 0; c < state->n_clusters; c++)
					{
						double		dist = euclidean_distance(data[s], state->centroids[c], state->dim);

						if (dist < min_dist)
						{
							min_dist = dist;
							best = c;
						}
					}
					cluster_assignments[s] = best;
					cluster_counts[best]++;
				}

				/*
				 * Compute scatter: average distance to centroid for each
				 * cluster
				 */
				for (c = 0; c < state->n_clusters; c++)
				{
					if (cluster_counts[c] > 0)
					{
						double		total_dist = 0.0;

						for (s = 0; s < nvec; s++)
						{
							if (cluster_assignments[s] == c)
							{
								total_dist += euclidean_distance(data[s], state->centroids[c], state->dim);
							}
						}
						scatter[c] = total_dist / (double) cluster_counts[c];
					}
					else
					{
						scatter[c] = 0.0;
					}
				}

				for (s = 0; s < nvec; s++)
					NDB_FREE(data[s]);
				NDB_FREE(data);
				NDB_FREE(cluster_assignments);
				NDB_FREE(cluster_counts);
			}
			else
			{
				/* Failed to load data - use default */
				if (data != NULL)
				{
					for (s = 0; s < nvec; s++)
						NDB_FREE(data[s]);
					NDB_FREE(data);
				}
				for (c = 0; c < state->n_clusters; c++)
					scatter[c] = 0.0;
			}
		}
		else
		{
			/* No evaluation table - cannot compute actual scatter */
			/* Use default value */
			for (c = 0; c < state->n_clusters; c++)
				scatter[c] = 0.0;
		}

		for (i = 0; i < state->n_clusters; i++)
		{
			double		max_ratio = 0.0;

			for (j = 0; j < state->n_clusters; j++)
			{
				if (i != j)
				{
					double		centroid_dist = sqrt(neurondb_l2_distance_squared(
																				  state->centroids[i], state->centroids[j], state->dim));

					if (centroid_dist > 0.0)
					{
						double		ratio = (scatter[i] + scatter[j]) / centroid_dist;

						if (ratio > max_ratio)
							max_ratio = ratio;
					}
				}
			}
			db_index += max_ratio;
		}
		db_index /= state->n_clusters;

		NDB_FREE(scatter);
	}

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"davies_bouldin\",\"storage\":\"cpu\","
					 "\"db_index\":%.6f,\"n_clusters\":%d,\"dim\":%d,\"n_samples\":%d}",
					 db_index,
					 state->n_clusters > 0 ? state->n_clusters : 0,
					 state->dim > 0 ? state->dim : 0,
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
davies_bouldin_gpu_serialize(const MLGpuModel * model, bytea * *payload_out,
							 Jsonb * *metadata_out, char **errstr)
{
	const		DaviesBouldinGpuModelState *state;
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
			*errstr = pstrdup("davies_bouldin_gpu_serialize: invalid model");
		return false;
	}

	state = (const DaviesBouldinGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("davies_bouldin_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	payload_copy = (bytea *) palloc(payload_size);
	memcpy(payload_copy, state->model_blob, payload_size);

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
davies_bouldin_gpu_deserialize(MLGpuModel * model, const bytea * payload,
							   const Jsonb * metadata, char **errstr)
{
	DaviesBouldinGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;
	float	  **centroids = NULL;
	NDB_DECLARE(int *, cluster_sizes);
	int			n_clusters = 0;
	int			dim = 0;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("davies_bouldin_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	payload_copy = (bytea *) palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);

	if (davies_bouldin_model_deserialize_from_bytea(payload_copy,
													&centroids, &cluster_sizes, &n_clusters, &dim) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("davies_bouldin_gpu_deserialize: failed to deserialize");
		return false;
	}

	state = (DaviesBouldinGpuModelState *) palloc0(sizeof(DaviesBouldinGpuModelState));
	state->model_blob = payload_copy;
	state->centroids = centroids;
	state->cluster_sizes = cluster_sizes;
	state->n_clusters = n_clusters;
	state->dim = dim;
	state->n_samples = 0;

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		Jsonb	   *metadata_copy = (Jsonb *) palloc(metadata_size);

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
davies_bouldin_gpu_destroy(MLGpuModel * model)
{
	DaviesBouldinGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (DaviesBouldinGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			NDB_FREE(state->model_blob);
		if (state->metrics != NULL)
			NDB_FREE(state->metrics);
		if (state->centroids != NULL)
		{
			for (int c = 0; c < state->n_clusters; c++)
				NDB_FREE(state->centroids[c]);
			NDB_FREE(state->centroids);
		}
		if (state->cluster_sizes != NULL)
			NDB_FREE(state->cluster_sizes);
		NDB_FREE(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps davies_bouldin_gpu_model_ops = {
	.algorithm = "davies_bouldin",
	.train = davies_bouldin_gpu_train,
	.predict = davies_bouldin_gpu_predict,
	.evaluate = davies_bouldin_gpu_evaluate,
	.serialize = davies_bouldin_gpu_serialize,
	.deserialize = davies_bouldin_gpu_deserialize,
	.destroy = davies_bouldin_gpu_destroy,
};

/* Forward declaration to avoid missing prototype warning */
extern void neurondb_gpu_register_davies_bouldin_model(void);

void
neurondb_gpu_register_davies_bouldin_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&davies_bouldin_gpu_model_ops);
	registered = true;
}
