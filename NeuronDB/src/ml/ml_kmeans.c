/*-------------------------------------------------------------------------
 *
 * ml_kmeans.c
 *    K-means clustering algorithm.
 *
 * This module implements Lloyd's algorithm with K-means++ initialization
 * for centroid-based clustering.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_kmeans.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/jsonb.h"
#include "common/jsonapi.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"
#include "ml_gpu_registry.h"
#include "ml_catalog.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_json.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * kmeanspp_init
 *    K-means++ seeding (D^2 weighting).
 */
static void
kmeanspp_init(float **data, int nvec, int dim, int k, int *centroids)
{
	bool	   *selected;
	double	   *dist;
	int			c,
				i,
				d;

	selected = (bool *) palloc0(sizeof(bool) * nvec);
	dist = (double *) palloc(sizeof(double) * nvec);

	/* Pick first centroid randomly */
	centroids[0] = rand() % nvec;
	selected[centroids[0]] = true;

	/* Compute initial distances from first centroid */
	for (i = 0; i < nvec; i++)
	{
		double		acc = 0.0;

		for (d = 0; d < dim; d++)
		{
			float		diff = data[i][d] - data[centroids[0]][d];

			acc += diff * diff;
		}
		dist[i] = acc;
	}

	/* Pick remaining centroids */
	for (c = 1; c < k; c++)
	{
		double		sum = 0.0;
		double		r;
		int			picked = -1;

		for (i = 0; i < nvec; i++)
			if (!selected[i])
				sum += dist[i];

		r = ((double) rand() / (double) RAND_MAX) * sum;

		for (i = 0; i < nvec; i++)
		{
			if (selected[i])
				continue;
			r -= dist[i];
			if (r <= 0)
			{
				picked = i;
				break;
			}
		}
		if (picked < 0)
		{
			for (i = 0; i < nvec; i++)
			{
				if (!selected[i])
				{
					picked = i;
					break;
				}
			}
		}

		/* Safety check */
		if (picked < 0 || picked >= nvec)
		{
			NDB_FREE(dist);
			NDB_FREE(selected);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: k-means++ failed to select centroid %d",
							c)));
		}

		centroids[c] = picked;
		selected[picked] = true;

		/* Update distances if new centroid is closer */
		for (i = 0; i < nvec; i++)
		{
			double		acc = 0.0;

			for (d = 0; d < dim; d++)
				acc += (data[i][d] - data[picked][d])
					* (data[i][d] - data[picked][d]);
			if (acc < dist[i])
				dist[i] = acc;
		}
	}

	NDB_FREE(dist);
	NDB_FREE(selected);
}

/*
 * cluster_kmeans
 *    K-means actual implementation: Lloyd's algorithm with k-means++ seeding
 */
PG_FUNCTION_INFO_V1(cluster_kmeans);

Datum
cluster_kmeans(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *vector_col = PG_GETARG_TEXT_PP(1);
	int32		num_clusters = PG_GETARG_INT32(2);
	int32		max_iters = PG_GETARG_INT32(3);

	char	   *tbl_str;
	char	   *col_str;
	int			nvec = 0;
	int			dim = 0;
	int		   *assignments = NULL;
	int		   *centroids_idx = NULL;
	float	  **data = NULL;
	float	  **centers = NULL;
	bool		changed = false;
	int			iter,
				i,
				c,
				d;
	Datum	   *out_datums;
	ArrayType  *out_array;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_col);

	if (num_clusters <= 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("number of clusters must be at least "
						"2")));
	if (max_iters < 1)
		max_iters = 100;

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (!data || nvec < num_clusters)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("not enough vectors for cluster count (need >= %d, have %d)",
						num_clusters,
						nvec)));

	assignments = (int *) palloc0(sizeof(int) * nvec);
	centroids_idx = (int *) palloc(sizeof(int) * num_clusters);

	/* Initialize centroids using k-means++ */
	kmeanspp_init(data, nvec, dim, num_clusters, centroids_idx);

	centers = (float **) palloc(sizeof(float *) * num_clusters);
	for (c = 0; c < num_clusters; c++)
	{
		centers[c] = (float *) palloc(sizeof(float) * dim);
		memcpy(centers[c], data[centroids_idx[c]], sizeof(float) * dim);
	}

	changed = true;
	for (i = 0; i < nvec; i++)
		assignments[i] = -1;

	for (iter = 0; iter < max_iters && changed; iter++)
	{
		int		   *counts;

		changed = false;
		/* Assignment phase - SIMD optimized */
		for (i = 0; i < nvec; i++)
		{
			double		min_dist = DBL_MAX;
			int			best = -1;

			for (c = 0; c < num_clusters; c++)
			{
				/* Use SIMD-optimized L2 distance */
				double		dist = neurondb_l2_distance_squared(
																data[i], centers[c], dim);

				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			if (assignments[i] != best)
			{
				assignments[i] = best;
				changed = true;
			}
		}

		/* Update phase */
		for (c = 0; c < num_clusters; c++)
			memset(centers[c], 0, sizeof(float) * dim);

		counts = (int *) palloc0(sizeof(int) * num_clusters);

		for (i = 0; i < nvec; i++)
		{
			c = assignments[i];
			for (d = 0; d < dim; d++)
				centers[c][d] += data[i][d];
			counts[c]++;
		}
		for (c = 0; c < num_clusters; c++)
		{
			if (counts[c] > 0)
			{
				for (d = 0; d < dim; d++)
					centers[c][d] /= counts[c];
			}
		}
		NDB_FREE(counts);
	}

	/* Build output int array (nvec x 1) for 1-based cluster labels */
	out_datums = (Datum *) palloc(sizeof(Datum) * nvec);
	for (i = 0; i < nvec; i++)
		out_datums[i] = Int32GetDatum(assignments[i] + 1);

	out_array = construct_array(out_datums, nvec, INT4OID, 4, true, 'i');

	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	for (c = 0; c < num_clusters; c++)
		NDB_FREE(centers[c]);
	NDB_FREE(data);
	NDB_FREE(centers);
	NDB_FREE(assignments);
	NDB_FREE(centroids_idx);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);
	NDB_FREE(out_datums);

	PG_RETURN_ARRAYTYPE_P(out_array);
}

/*
 * Serialize K-Means model (centroids) to bytea
 */
static bytea *
kmeans_model_serialize_to_bytea(float **centers, int num_clusters, int dim, uint8 training_backend)
{
	StringInfoData buf;
	int			i,
				j;
	int			total_size;
	bytea	   *result;

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: kmeans_model_serialize_to_bytea: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	initStringInfo(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) */
	appendBinaryStringInfo(&buf, (char *) &training_backend, sizeof(uint8));

	/* Write header: num_clusters, dim */
	appendBinaryStringInfo(&buf, (char *) &num_clusters, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &dim, sizeof(int));

	/* Write centroids */
	for (i = 0; i < num_clusters; i++)
		for (j = 0; j < dim; j++)
			appendBinaryStringInfo(&buf, (char *) &centers[i][j], sizeof(float));

	/* Convert to bytea */
	total_size = VARHDRSZ + buf.len;
	result = (bytea *) palloc(total_size);
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

/*
 * Deserialize K-Means model from bytea
 */
static int
kmeans_model_deserialize_from_bytea(const bytea * data, float ***centers_out, int *num_clusters_out, int *dim_out, uint8 *training_backend_out)
{
	const char *buf;
	int			offset = 0;
	int			i,
				j;
	float	  **centers;
	uint8		training_backend = 0;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 2)
		return -1;

	buf = VARDATA(data);
	offset = 0;

	/* Read training_backend first */
	training_backend = (uint8) buf[offset];
	offset += sizeof(uint8);
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;

	/* Read header: num_clusters, dim */
	memcpy(num_clusters_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(dim_out, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (*num_clusters_out <= 0 || *num_clusters_out > 10000 || *dim_out <= 0 || *dim_out > 100000)
		return -1;

	/* Allocate centroids */
	centers = (float **) palloc(sizeof(float *) * *num_clusters_out);
	for (i = 0; i < *num_clusters_out; i++)
	{
		centers[i] = (float *) palloc(sizeof(float) * *dim_out);
		for (j = 0; j < *dim_out; j++)
		{
			memcpy(&centers[i][j], buf + offset, sizeof(float));
			offset += sizeof(float);
		}
	}

	*centers_out = centers;
	return 0;
}

/*
 * train_kmeans_model_id
 *
 * Trains K-Means and stores centroids in catalog, returns model_id
 */
PG_FUNCTION_INFO_V1(train_kmeans_model_id);

Datum
train_kmeans_model_id(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *vector_col = PG_GETARG_TEXT_PP(1);
	int32		num_clusters = PG_GETARG_INT32(2);
	int32		max_iters = PG_GETARG_INT32(3);

	char	   *tbl_str;
	char	   *col_str;
	int			nvec = 0;
	int			dim = 0;
	int		   *assignments = NULL;
	int		   *centroids_idx = NULL;
	float	  **data = NULL;
	float	  **centers = NULL;
	bool		changed = false;
	int			iter,
				i,
				c,
				d;
	bytea	   *model_data;
	MLCatalogModelSpec spec;
	Jsonb	   *metrics;
	StringInfoData metrics_json;
	int32		model_id;

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_col);

	if (num_clusters <= 1)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("number of clusters must be at least 2")));
	}
	if (max_iters < 1)
		max_iters = 100;

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (!data || nvec < num_clusters)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("not enough vectors for cluster count (need >= %d, have %d)",
						num_clusters,
						nvec)));
	}

	assignments = (int *) palloc0(sizeof(int) * nvec);
	centroids_idx = (int *) palloc(sizeof(int) * num_clusters);

	/* Initialize centroids using k-means++ */
	kmeanspp_init(data, nvec, dim, num_clusters, centroids_idx);

	centers = (float **) palloc(sizeof(float *) * num_clusters);
	for (c = 0; c < num_clusters; c++)
	{
		centers[c] = (float *) palloc(sizeof(float) * dim);
		memcpy(centers[c], data[centroids_idx[c]], sizeof(float) * dim);
	}

	changed = true;
	for (i = 0; i < nvec; i++)
		assignments[i] = -1;

	/* K-Means iteration */
	for (iter = 0; iter < max_iters && changed; iter++)
	{
		int		   *counts;

		changed = false;
		/* Assignment phase */
		for (i = 0; i < nvec; i++)
		{
			double		min_dist = DBL_MAX;
			int			best = -1;

			for (c = 0; c < num_clusters; c++)
			{
				double		dist = neurondb_l2_distance_squared(
																data[i], centers[c], dim);

				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			if (assignments[i] != best)
			{
				assignments[i] = best;
				changed = true;
			}
		}

		/* Update phase */
		for (c = 0; c < num_clusters; c++)
			memset(centers[c], 0, sizeof(float) * dim);

		counts = (int *) palloc0(sizeof(int) * num_clusters);

		for (i = 0; i < nvec; i++)
		{
			c = assignments[i];
			for (d = 0; d < dim; d++)
				centers[c][d] += data[i][d];
			counts[c]++;
		}
		for (c = 0; c < num_clusters; c++)
		{
			if (counts[c] > 0)
			{
				for (d = 0; d < dim; d++)
					centers[c][d] /= counts[c];
			}
		}
		NDB_FREE(counts);
	}

	/* Serialize model to bytea with training_backend=0 (CPU) */
	model_data = kmeans_model_serialize_to_bytea(centers, num_clusters, dim, 0);

	/* Build metrics JSONB */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json, "{\"storage\": \"cpu\", \"k\": %d, \"dim\": %d, \"max_iters\": %d, \"n_samples\": %d}",
					 num_clusters, dim, max_iters, nvec);
	/* Use ndb_jsonb_in_cstring like other ML algorithms fix */
	metrics = ndb_jsonb_in_cstring(metrics_json.data);
	if (metrics == NULL)
	{
		NDB_FREE(metrics_json.data);
		NDB_FREE(centers);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		NDB_FREE(model_data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("neurondb: train_kmeans_model_id: failed to parse metrics JSON")));
	}
	NDB_FREE(metrics_json.data);

	/* Store model in catalog */
	memset(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.project_name = NULL;
	spec.algorithm = "kmeans";
	spec.training_table = tbl_str;
	spec.training_column = NULL;
	spec.model_data = model_data;
	spec.metrics = metrics;
	spec.num_samples = nvec;
	spec.num_features = dim;

	model_id = ml_catalog_register_model(&spec);

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	for (c = 0; c < num_clusters; c++)
		NDB_FREE(centers[c]);
	NDB_FREE(data);
	NDB_FREE(centers);
	NDB_FREE(assignments);
	NDB_FREE(centroids_idx);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_INT32(model_id);
}

/*
 * Compute Euclidean distance squared
 */
static inline double
euclidean_distance_squared(const float *a, const float *b, int dim)
{
	double		sum = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double		diff = (double) a[i] - (double) b[i];

		sum += diff * diff;
	}
	return sum;
}

/*
 * Compute Euclidean distance
 */
static inline double
euclidean_distance(const float *a, const float *b, int dim)
{
	return sqrt(euclidean_distance_squared(a, b, dim));
}

/*
 * evaluate_kmeans_by_model_id
 *
 * Evaluates K-Means clustering model by computing:
 * - Inertia (within-cluster sum of squares)
 * - Silhouette score
 * - Davies-Bouldin index
 */
PG_FUNCTION_INFO_V1(evaluate_kmeans_by_model_id);

Datum
evaluate_kmeans_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *vector_col;
	char	   *tbl_str;
	char	   *col_str;
	int			nvec = 0;
	int			i,
				j,
				c;
	float	  **data = NULL;
	int			dim = 0;
	float	  **centers = NULL;
	int			num_clusters = 0;
	int		   *assignments = NULL;
	int		   *cluster_sizes = NULL;
	double		inertia = 0.0;
	double		silhouette = 0.0;
	double		davies_bouldin = 0.0;
	double	   *cluster_scatter = NULL;
	double	   *a_scores = NULL;
	double	   *b_scores = NULL;
	MemoryContext oldcontext;
	Jsonb	   *result_jsonb = NULL;
	bytea	   *model_payload = NULL;
	Jsonb	   *model_metrics = NULL;
	NDB_DECLARE(NdbSpiSession *, spi_session);

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_kmeans_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_kmeans_by_model_id: table_name and vector_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	vector_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_col);

	oldcontext = CurrentMemoryContext;

	/* Load model from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_payload, NULL, &model_metrics))
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_kmeans_by_model_id: model %d not found",
						model_id)));
	}

	if (model_payload == NULL)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		if (model_metrics)
			NDB_FREE(model_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_kmeans_by_model_id: model %d has no model_data",
						model_id)));
	}

	/* Deserialize model */
		if (kmeans_model_deserialize_from_bytea(model_payload, &centers, &num_clusters, &dim, NULL) != 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		if (model_payload)
			NDB_FREE(model_payload);
		if (model_metrics)
			NDB_FREE(model_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_kmeans_by_model_id: failed to deserialize model")));
	}

	/* Connect to SPI */
	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Fetch test data */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (!data || nvec < 1)
	{
		NDB_SPI_SESSION_END(spi_session);
		for (c = 0; c < num_clusters; c++)
			NDB_FREE(centers[c]);
		NDB_FREE(centers);
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		if (model_payload)
			NDB_FREE(model_payload);
		if (model_metrics)
			NDB_FREE(model_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_kmeans_by_model_id: no valid data found")));
	}

	/* Assign points to clusters */
	assignments = (int *) palloc(sizeof(int) * nvec);
	cluster_sizes = (int *) palloc0(sizeof(int) * num_clusters);

	for (i = 0; i < nvec; i++)
	{
		double		min_dist = DBL_MAX;
		int			best = 0;

		for (c = 0; c < num_clusters; c++)
		{
			double		dist = euclidean_distance_squared(data[i], centers[c], dim);

			if (dist < min_dist)
			{
				min_dist = dist;
				best = c;
			}
		}
		assignments[i] = best;
		cluster_sizes[best]++;
		inertia += min_dist;
	}

	/* Compute silhouette score */
	a_scores = (double *) palloc0(sizeof(double) * nvec);
	b_scores = (double *) palloc0(sizeof(double) * nvec);

	for (i = 0; i < nvec; i++)
	{
		int			my_cluster = assignments[i];
		int			same_count = 0;
		double		same_dist = 0.0;
		double		min_other_dist = DBL_MAX;

		if (cluster_sizes[my_cluster] <= 1)
		{
			a_scores[i] = 0.0;
			b_scores[i] = 0.0;
			continue;
		}

		/* Average distance to same cluster */
		for (j = 0; j < nvec; j++)
		{
			if (i == j)
				continue;
			if (assignments[j] == my_cluster)
			{
				same_dist += euclidean_distance(data[i], data[j], dim);
				same_count++;
			}
		}
		if (same_count > 0)
			a_scores[i] = same_dist / (double) same_count;
		else
			a_scores[i] = 0.0;

		/* Minimum average distance to other clusters */
		{
			double		other_dist = 0.0;
			int			other_count = 0;
			int			other_cluster_loop;

			for (other_cluster_loop = 0; other_cluster_loop < num_clusters; other_cluster_loop++)
			{
				if (other_cluster_loop == my_cluster)
					continue;
				if (cluster_sizes[other_cluster_loop] == 0)
					continue;

				for (j = 0; j < nvec; j++)
				{
					if (assignments[j] == other_cluster_loop)
					{
						other_dist += euclidean_distance(data[i], data[j], dim);
						other_count++;
					}
				}
				if (other_count > 0)
				{
					other_dist /= (double) other_count;
					if (other_dist < min_other_dist)
						min_other_dist = other_dist;
				}
			}
		}
		b_scores[i] = min_other_dist;
	}

	/* Compute average silhouette */
	{
		int			valid_count = 0;
		double		sum_silhouette = 0.0;

		for (i = 0; i < nvec; i++)
		{
			double		max_ab = (a_scores[i] > b_scores[i]) ? a_scores[i] : b_scores[i];

			if (max_ab > 0.0)
			{
				double		s = (b_scores[i] - a_scores[i]) / max_ab;

				sum_silhouette += s;
				valid_count++;
			}
		}
		if (valid_count > 0)
			silhouette = sum_silhouette / (double) valid_count;
	}

	/* Compute Davies-Bouldin index */
	cluster_scatter = (double *) palloc0(sizeof(double) * num_clusters);

	for (i = 0; i < nvec; i++)
	{
		c = assignments[i];
		cluster_scatter[c] += euclidean_distance(data[i], centers[c], dim);
	}

	for (c = 0; c < num_clusters; c++)
	{
		if (cluster_sizes[c] > 0)
			cluster_scatter[c] /= (double) cluster_sizes[c];
	}

	{
		int			valid_clusters = 0;
		double		sum_dbi = 0.0;

		for (i = 0; i < num_clusters; i++)
		{
			double		max_ratio = 0.0;

			if (cluster_sizes[i] < 2)
				continue;

			for (j = 0; j < num_clusters; j++)
			{
				double		centroid_dist;
				double		ratio;

				if (i == j || cluster_sizes[j] < 2)
					continue;

				centroid_dist = euclidean_distance(centers[i], centers[j], dim);
				if (centroid_dist < 1e-10)
					continue;

				ratio = (cluster_scatter[i] + cluster_scatter[j]) / centroid_dist;
				if (ratio > max_ratio)
					max_ratio = ratio;
			}
			sum_dbi += max_ratio;
			valid_clusters++;
		}
		if (valid_clusters > 0)
			davies_bouldin = sum_dbi / (double) valid_clusters;
	}

	/* End SPI session BEFORE creating JSONB to avoid context conflicts */
	NDB_SPI_SESSION_END(spi_session);

	/* Switch to old context and build JSONB directly using JSONB API */
	MemoryContextSwitchTo(oldcontext);
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;
		JsonbValue *final_value = NULL;
		Numeric		inertia_num, silhouette_num, davies_bouldin_num, n_samples_num, n_clusters_num;

		/* Start object */
		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			/* Add inertia */
			jkey.type = jbvString;
			jkey.val.string.len = 8;
			jkey.val.string.val = "inertia";
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			inertia_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(inertia)));
			jval.type = jbvNumeric;
			jval.val.numeric = inertia_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add silhouette_score */
			jkey.val.string.val = "silhouette_score";
			jkey.val.string.len = 16;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			silhouette_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(silhouette)));
			jval.type = jbvNumeric;
			jval.val.numeric = silhouette_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add davies_bouldin_index */
			jkey.val.string.val = "davies_bouldin_index";
			jkey.val.string.len = 19;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			davies_bouldin_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(davies_bouldin)));
			jval.type = jbvNumeric;
			jval.val.numeric = davies_bouldin_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add n_samples */
			jkey.val.string.val = "n_samples";
			jkey.val.string.len = 9;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(nvec)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_samples_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add n_clusters */
			jkey.val.string.val = "n_clusters";
			jkey.val.string.len = 10;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_clusters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(num_clusters)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_clusters_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add n_iterations - extract from model_metrics if available */
			{
				int			n_iterations = 100;	/* Default */
				JsonbIterator *it;
				JsonbValue	v;
				int			r;

				if (model_metrics != NULL)
				{
					it = JsonbIteratorInit((JsonbContainer *) & model_metrics->root);
					while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
					{
						if (r == WJB_KEY)
						{
							char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

							r = JsonbIteratorNext(&it, &v, false);
							if (strcmp(key, "max_iters") == 0 && v.type == jbvNumeric)
								n_iterations = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																				 NumericGetDatum(v.val.numeric)));
							else if (strcmp(key, "n_iterations") == 0 && v.type == jbvNumeric)
								n_iterations = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																				 NumericGetDatum(v.val.numeric)));
							NDB_FREE(key);
						}
					}
				}

				jkey.val.string.val = "n_iterations";
				jkey.val.string.len = 12;
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				{
					Numeric		n_iterations_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(n_iterations)));

					jval.type = jbvNumeric;
					jval.val.numeric = n_iterations_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);
				}
			}

			/* End object */
			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
			
			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: evaluate_kmeans: pushJsonbValue(WJB_END_OBJECT) returned NULL");
			}
			
			result_jsonb = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData *edata = CopyErrorData();
			elog(ERROR, "neurondb: evaluate_kmeans: JSONB construction failed: %s", edata->message);
			FlushErrorState();
			result_jsonb = NULL;
		}
		PG_END_TRY();
	}

	if (result_jsonb == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_kmeans_by_model_id: JSONB result is NULL")));
	}

	/*
	 * Cleanup - Note: data[], centers[] allocated by
	 * neurondb_fetch_vectors_from_table in SPI context, will be freed by
	 * NDB_SPI_SESSION_END(). Do not double-free.
	 */
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);
	if (model_payload)
		NDB_FREE(model_payload);
	if (model_metrics)
		NDB_FREE(model_metrics);

	PG_RETURN_JSONB_P(result_jsonb);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for K-Means
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

static bool
kmeans_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	typedef struct KMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
	}			KMeansGpuModelState;

	KMeansGpuModelState *state;
	float	  **data = NULL;
	float	  **centers = NULL;
	int		   *assignments = NULL;
	int		   *centroids_idx = NULL;
	int			num_clusters = 8;	/* Default */
	int			max_iters = 100;
	int			nvec = 0;
	int			dim = 0;
	bool		changed = true;
	int			iter,
				i,
				c,
				d;
	bytea	   *model_data = NULL;
	Jsonb	   *metrics = NULL;
	StringInfoData metrics_json;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_train: invalid parameters");
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
				else if (strcmp(key, "max_iters") == 0 && v.type == jbvNumeric)
					max_iters = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																  NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}

	if (num_clusters < 2)
		num_clusters = 8;
	if (max_iters < 1)
		max_iters = 100;

	/* Convert feature matrix to 2D array format */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	/* Allocate data array */
	data = (float **) palloc(sizeof(float *) * nvec);
	for (i = 0; i < nvec; i++)
	{
		data[i] = (float *) palloc(sizeof(float) * dim);
		memcpy(data[i], &spec->feature_matrix[i * dim], sizeof(float) * dim);
	}

	if (nvec < num_clusters)
	{
		for (i = 0; i < nvec; i++)
			NDB_FREE(data[i]);
		NDB_FREE(data);
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_train: not enough samples for clusters");
		return false;
	}

	assignments = (int *) palloc0(sizeof(int) * nvec);
	centroids_idx = (int *) palloc(sizeof(int) * num_clusters);

	/* Initialize centroids using k-means++ */
	kmeanspp_init(data, nvec, dim, num_clusters, centroids_idx);

	centers = (float **) palloc(sizeof(float *) * num_clusters);
	for (c = 0; c < num_clusters; c++)
	{
		centers[c] = (float *) palloc(sizeof(float) * dim);
		memcpy(centers[c], data[centroids_idx[c]], sizeof(float) * dim);
	}

	changed = true;
	for (i = 0; i < nvec; i++)
		assignments[i] = -1;

	/* K-Means iteration (CPU implementation) */
	for (iter = 0; iter < max_iters && changed; iter++)
	{
		int		   *counts;

		changed = false;
		/* Assignment phase */
		for (i = 0; i < nvec; i++)
		{
			double		min_dist = DBL_MAX;
			int			best = -1;

			for (c = 0; c < num_clusters; c++)
			{
				double		dist = neurondb_l2_distance_squared(
																data[i], centers[c], dim);

				if (dist < min_dist)
				{
					min_dist = dist;
					best = c;
				}
			}
			if (assignments[i] != best)
			{
				assignments[i] = best;
				changed = true;
			}
		}

		/* Update phase */
		for (c = 0; c < num_clusters; c++)
			memset(centers[c], 0, sizeof(float) * dim);

		counts = (int *) palloc0(sizeof(int) * num_clusters);
		for (i = 0; i < nvec; i++)
		{
			c = assignments[i];
			for (d = 0; d < dim; d++)
				centers[c][d] += data[i][d];
			counts[c]++;
		}
		for (c = 0; c < num_clusters; c++)
		{
			if (counts[c] > 0)
			{
				for (d = 0; d < dim; d++)
					centers[c][d] /= counts[c];
			}
		}
		NDB_FREE(counts);
	}

	/* Serialize model with training_backend=1 (GPU) */
	model_data = kmeans_model_serialize_to_bytea(centers, num_clusters, dim, 1);

	/* Build metrics JSONB */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"training_backend\":1,\"k\":%d,\"dim\":%d,\"max_iters\":%d,\"n_samples\":%d}",
					 num_clusters, dim, max_iters, nvec);
	/* Use ndb_jsonb_in_cstring like other ML algorithms fix */
	metrics = ndb_jsonb_in_cstring(metrics_json.data);
	if (metrics == NULL)
	{
		NDB_FREE(metrics_json.data);
		/* Cleanup allocated arrays */
		for (i = 0; i < nvec; i++)
			NDB_FREE(data[i]);
		for (c = 0; c < num_clusters; c++)
			NDB_FREE(centers[c]);
		NDB_FREE(data);
		NDB_FREE(centers);
		NDB_FREE(assignments);
		NDB_FREE(centroids_idx);
		NDB_FREE(model_data);
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_train: failed to parse metrics JSON");
		return false;
	}
	NDB_FREE(metrics_json.data);

	/* Store in model state */
	state = (KMeansGpuModelState *) palloc0(sizeof(KMeansGpuModelState));
	state->model_blob = model_data;
	state->metrics = metrics;
	state->num_clusters = num_clusters;
	state->dim = dim;
	state->n_samples = nvec;

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false; /* CPU fallback */

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	for (c = 0; c < num_clusters; c++)
		NDB_FREE(centers[c]);
	NDB_FREE(data);
	NDB_FREE(centers);
	NDB_FREE(assignments);
	NDB_FREE(centroids_idx);

	return true;
}

static bool
kmeans_gpu_predict(const MLGpuModel * model, const float *input, int input_dim,
				   float *output, int output_dim, char **errstr)
{
	typedef struct KMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
	}			KMeansGpuModelState;

	const		KMeansGpuModelState *state;
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
			*errstr = pstrdup("kmeans_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_predict: model not ready");
		return false;
	}

	state = (const KMeansGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_predict: model blob is NULL");
		return false;
	}

	/* Deserialize model */
	if (kmeans_model_deserialize_from_bytea(state->model_blob,
											&centers, &num_clusters, &dim, NULL) != 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_predict: failed to deserialize model");
		return false;
	}

	if (input_dim != dim)
	{
		for (c = 0; c < num_clusters; c++)
			NDB_FREE(centers[c]);
		NDB_FREE(centers);
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_predict: dimension mismatch");
		return false;
	}

	/* Find nearest centroid */
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

	/* Cleanup */
	for (c = 0; c < num_clusters; c++)
		NDB_FREE(centers[c]);
	NDB_FREE(centers);

	return true;
}

static bool
kmeans_gpu_evaluate(const MLGpuModel * model, const MLGpuEvalSpec * spec,
					MLGpuMetrics * out, char **errstr)
{
	typedef struct KMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
	}			KMeansGpuModelState;

	const		KMeansGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_evaluate: invalid model");
		return false;
	}

	state = (const KMeansGpuModelState *) model->backend_state;

	/* Create metrics JSON */
	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"kmeans\",\"storage\":\"cpu\","
					 "\"k\":%d,\"dim\":%d,\"n_samples\":%d}",
					 state->num_clusters > 0 ? state->num_clusters : 0,
					 state->dim > 0 ? state->dim : 0,
					 state->n_samples > 0 ? state->n_samples : 0);

	/* Use ndb_jsonb_in_cstring like other ML algorithms fix */
	metrics_json = ndb_jsonb_in_cstring(buf.data);
	if (metrics_json == NULL)
	{
		NDB_FREE(buf.data);
		if (errstr != NULL)
			*errstr = pstrdup("failed to parse metrics JSON");
		return false;
	}
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
kmeans_gpu_serialize(const MLGpuModel * model, bytea * *payload_out,
					 Jsonb * *metadata_out, char **errstr)
{
	typedef struct KMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
	}			KMeansGpuModelState;

	const		KMeansGpuModelState *state;
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
			*errstr = pstrdup("kmeans_gpu_serialize: invalid model");
		return false;
	}

	state = (const KMeansGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_serialize: model blob is NULL");
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
kmeans_gpu_deserialize(MLGpuModel * model, const bytea * payload,
					   const Jsonb * metadata, char **errstr)
{
	typedef struct KMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
	}			KMeansGpuModelState;

	KMeansGpuModelState *state;
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
			*errstr = pstrdup("kmeans_gpu_deserialize: invalid parameters");
		return false;
	}

	/* Copy payload */
	payload_size = VARSIZE(payload);
	payload_copy = (bytea *) palloc(payload_size);
	memcpy(payload_copy, payload, payload_size);

	/* Deserialize to get dimensions */
	if (kmeans_model_deserialize_from_bytea(payload_copy,
											&centers, &num_clusters, &dim, NULL) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("kmeans_gpu_deserialize: failed to deserialize");
		return false;
	}

	/* Free temporary deserialized data */
	for (int c = 0; c < num_clusters; c++)
		NDB_FREE(centers[c]);
	NDB_FREE(centers);

	/* Extract n_samples from metadata if available */
	state = (KMeansGpuModelState *) palloc0(sizeof(KMeansGpuModelState));
	state->model_blob = payload_copy;
	state->num_clusters = num_clusters;
	state->dim = dim;
	state->n_samples = 0;

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		Jsonb	   *metadata_copy = (Jsonb *) palloc(metadata_size);

		memcpy(metadata_copy, metadata, metadata_size);
		state->metrics = metadata_copy;

		/* Extract n_samples from metadata */
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
kmeans_gpu_destroy(MLGpuModel * model)
{
	typedef struct KMeansGpuModelState
	{
		bytea	   *model_blob;
		Jsonb	   *metrics;
		int			num_clusters;
		int			dim;
		int			n_samples;
	}			KMeansGpuModelState;

	KMeansGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (KMeansGpuModelState *) model->backend_state;
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

static const MLGpuModelOps kmeans_gpu_model_ops = {
	.algorithm = "kmeans",
	.train = kmeans_gpu_train,
	.predict = kmeans_gpu_predict,
	.evaluate = kmeans_gpu_evaluate,
	.serialize = kmeans_gpu_serialize,
	.deserialize = kmeans_gpu_deserialize,
	.destroy = kmeans_gpu_destroy,
};

void
neurondb_gpu_register_kmeans_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&kmeans_gpu_model_ops);
	registered = true;
}
