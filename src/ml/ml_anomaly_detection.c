/*-------------------------------------------------------------------------
 *
 * ml_anomaly_detection.c
 *    Anomaly detection algorithms.
 *
 * This module implements Isolation Forest, LOF, and One-Class SVM for
 * detecting anomalies in data.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_anomaly_detection.c
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

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Isolation Tree node structure */
typedef struct IsolationNode
{
	int			split_dim;
	float		split_value;
	struct IsolationNode *left;
	struct IsolationNode *right;
	int			size;
}			IsolationNode;

/* Isolation Forest structure */
typedef struct IsolationForest
{
	IsolationNode **trees;
	int			n_trees;
	int			max_depth;
	int			n_samples;
	int			dim;
}			IsolationForest;

/*
 * Random float in range [min, max]
 */
static float
random_float(float min, float max)
{
	return min + ((float) rand() / (float) RAND_MAX) * (max - min);
}

/*
 * Build isolation tree recursively
 */
static IsolationNode *
build_isolation_tree(float **data, int *indices, int n_points, int dim,
					 int current_depth, int max_depth)
{
	IsolationNode *node = NULL;
	int			i,
				d;
	float		min_val,
				max_val;
	int			split_dim;
	float		split_value;
	int			left_count,
				right_count;
	int		   *left_indices,
			   *right_indices;

	if (n_points <= 1 || current_depth >= max_depth)
	{
		IsolationNode *node = NULL;
		nalloc(node, IsolationNode, 1);
		node->split_dim = -1;
		node->split_value = 0.0;
		node->left = NULL;
		node->right = NULL;
		node->size = n_points;
		return node;
	}

	IsolationNode *node = NULL;
	nalloc(node, IsolationNode, 1);
	node->size = n_points;

	/* Randomly select dimension */
	split_dim = rand() % dim;

	/* Find min/max in selected dimension */
	min_val = max_val = data[indices[0]][split_dim];
	for (i = 1; i < n_points; i++)
	{
		float		val = data[indices[i]][split_dim];

		if (val < min_val)
			min_val = val;
		if (val > max_val)
			max_val = val;
	}

	/* Random split value */
	if (max_val - min_val < 1e-10)
		split_value = min_val;
	else
		split_value = random_float(min_val, max_val);

	node->split_dim = split_dim;
	node->split_value = split_value;

	/* Partition indices */
	int *left_indices = NULL;
	int *right_indices = NULL;
	nalloc(left_indices, int, n_points);
	nalloc(right_indices, int, n_points);
	left_count = right_count = 0;

	for (i = 0; i < n_points; i++)
	{
		if (data[indices[i]][split_dim] < split_value)
			left_indices[left_count++] = indices[i];
		else
			right_indices[right_count++] = indices[i];
	}

	/* Recursively build children */
	node->left = build_isolation_tree(data, left_indices, left_count, dim,
									  current_depth + 1, max_depth);
	node->right = build_isolation_tree(data, right_indices, right_count, dim,
									   current_depth + 1, max_depth);

	nfree(left_indices);
	nfree(right_indices);

	return node;
}

/*
 * Calculate path length in isolation tree
 */
static double
path_length(IsolationNode * node, const float *point, int dim, int depth)
{
	if (node->split_dim < 0)
		return depth + average_path_length(node->size);

	if (point[node->split_dim] < node->split_value)
		return path_length(node->left, point, dim, depth + 1);
	else
		return path_length(node->right, point, dim, depth + 1);
}

/*
 * Average path length for given sample size
 */
static double
average_path_length(int n)
{
	if (n <= 1)
		return 0.0;
	if (n == 2)
		return 1.0;
	return 2.0 * (log((double) (n - 1)) + 0.5772156649) -
		2.0 * ((double) (n - 1) / (double) n);
}

/*
 * Free isolation tree
 */
static void
free_isolation_tree(IsolationNode * node)
{
	if (node == NULL)
		return;
	free_isolation_tree(node->left);
	free_isolation_tree(node->right);
	nfree(node);
}

/*
 * detect_anomalies_isolation_forest
 * ---------------------------------
 * Isolation Forest anomaly detection.
 *
 * SQL Arguments:
 *   table_name    - Source table with vectors
 *   vector_column - Vector column name
 *   n_trees       - Number of trees (default: 100)
 *   contamination - Expected fraction of outliers (default: 0.1)
 *
 * Returns:
 *   Boolean array indicating anomalies (true = anomaly)
 */
PG_FUNCTION_INFO_V1(detect_anomalies_isolation_forest);

Datum
detect_anomalies_isolation_forest(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *vector_column = NULL;
	int			n_trees;
	double		contamination;
	char *tbl_str = NULL;
	char *vec_col_str = NULL;
	float	  **vectors;
	int			nvec,
				dim;
	IsolationForest *forest = NULL;
	int *indices = NULL;
	int			i,
				t;
	double *anomaly_scores = NULL;
	bool *anomalies = NULL;
	double		threshold;
	ArrayType *result = NULL;
	Datum *result_datums = NULL;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	n_trees = PG_ARGISNULL(2) ? 100 : PG_GETARG_INT32(2);
	contamination = PG_ARGISNULL(3) ? 0.1 : PG_GETARG_FLOAT8(3);

	if (n_trees < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_trees must be at least 1")));
	if (contamination < 0.0 || contamination > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("contamination must be between 0 and 1")));

	tbl_str = text_to_cstring(table_name);
	vec_col_str = text_to_cstring(vector_column);

	vectors = neurondb_fetch_vectors_from_table(
												tbl_str, vec_col_str, &nvec, &dim);
	if (vectors == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(vec_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(vec_col_str);
		/* Free vectors array and rows if vectors is not NULL */
		if (vectors != NULL)
		{
			for (int i = 0; i < nvec; i++)
			{
				if (vectors[i] != NULL)
					nfree(vectors[i]);
			}
			nfree(vectors);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (nvec < 2)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 2 vectors")));

	/* Build isolation forest */
	IsolationForest *forest = NULL;
	nalloc(forest, IsolationForest, 1);
	forest->n_trees = n_trees;
	forest->max_depth = (int) ceil(log2((double) nvec));
	forest->n_samples = nvec;
	forest->dim = dim;
	IsolationNode **trees = NULL;
	nalloc(trees, IsolationNode *, n_trees);
	forest->trees = trees;
											   n_trees);

	int *indices = NULL;
	nalloc(indices, int, nvec);
	for (i = 0; i < nvec; i++)
		indices[i] = i;

	for (t = 0; t < n_trees; t++)
	{
		/* Randomly shuffle indices for each tree */
		for (i = nvec - 1; i > 0; i--)
		{
			int			j = rand() % (i + 1);
			int			tmp = indices[i];

			indices[i] = indices[j];
			indices[j] = tmp;
		}
		forest->trees[t] = build_isolation_tree(vectors, indices, nvec, dim,
												0, forest->max_depth);
	}

	/* Calculate anomaly scores */
	double *anomaly_scores = NULL;
	nalloc(anomaly_scores, double, nvec);
	for (i = 0; i < nvec; i++)
	{
		double		avg_path = 0.0;

		for (t = 0; t < n_trees; t++)
			avg_path += path_length(forest->trees[t], vectors[i], dim, 0);
		avg_path /= n_trees;
		anomaly_scores[i] = pow(2.0, -avg_path / average_path_length(nvec));
	}

	/* Determine threshold based on contamination */
	{
		int			threshold_idx;

		double *sorted_scores = NULL;
		nalloc(sorted_scores, double, nvec);
		memcpy(sorted_scores, anomaly_scores, sizeof(double) * nvec);
		qsort(sorted_scores, nvec, sizeof(double), double_compare);

		threshold_idx = (int) ((1.0 - contamination) * nvec);
		if (threshold_idx >= nvec)
			threshold_idx = nvec - 1;
		threshold = sorted_scores[threshold_idx];

		nfree(sorted_scores);
	}

	/* Mark anomalies */
	anomalies = (bool *) palloc0(sizeof(bool) * nvec);
	for (i = 0; i < nvec; i++)
		anomalies[i] = (anomaly_scores[i] > threshold);

	/* Build result array */
	Datum *result_datums = NULL;
	nalloc(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = BoolGetDatum(anomalies[i]);

	get_typlenbyvalalign(BOOLOID, &typlen, &typbyval, &typalign);
	result = construct_array(
							 result_datums, nvec, BOOLOID, typlen, typbyval, typalign);

	for (t = 0; t < n_trees; t++)
		free_isolation_tree(forest->trees[t]);
	nfree(forest->trees);
	nfree(forest);
	for (i = 0; i < nvec; i++)
		nfree(vectors[i]);
	nfree(vectors);
	nfree(indices);
	nfree(anomaly_scores);
	nfree(anomalies);
	nfree(result_datums);
	nfree(tbl_str);
	nfree(vec_col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * Comparison function for qsort (ascending)
 */
static int
double_compare(const void *a, const void *b)
{
	double		da = *(const double *) a;
	double		db = *(const double *) b;

	if (da < db)
		return -1;
	if (da > db)
		return 1;
	return 0;
}

/*
 * Calculate k-distance (distance to k-th nearest neighbor)
 */
static double
k_distance(const float *point, float **data, int nvec, int dim, int k)
{
	int			i,
				d;
	double		dist;

	double *distances = NULL;
	nalloc(distances, double, nvec);

	for (i = 0; i < nvec; i++)
	{
		double		sum = 0.0;

		for (d = 0; d < dim; d++)
		{
			double		diff = (double) point[d] - (double) data[i][d];

			sum += diff * diff;
		}
		distances[i] = sqrt(sum);
	}

	qsort(distances, nvec, sizeof(double), double_compare);
	dist = distances[k];
	nfree(distances);
	return dist;
}

/*
 * Calculate reachability distance
 */
static double
reachability_distance(const float *point_a, const float *point_b,
					  float **data, int nvec, int dim, int k)
{
	double		k_dist_b = k_distance(point_b, data, nvec, dim, k);
	double		dist_ab = 0.0;
	int			d;

	for (d = 0; d < dim; d++)
	{
		double		diff = (double) point_a[d] - (double) point_b[d];

		dist_ab += diff * diff;
	}
	dist_ab = sqrt(dist_ab);

	return (dist_ab > k_dist_b) ? dist_ab : k_dist_b;
}

/*
 * Calculate local reachability density
 */
static double
local_reachability_density(const float *point, float **data, int nvec,
						   int dim, int k)
{
	double *distances = NULL;
	int *neighbors = NULL;
	int			i,
				j,
				d;
	double		sum_reach_dist = 0.0;
	double		lrd;

	nalloc(distances, double, nvec);
	nalloc(neighbors, int, k);

	/* Calculate distances to all points */
	for (i = 0; i < nvec; i++)
	{
		double		sum = 0.0;

		for (d = 0; d < dim; d++)
		{
			double		diff = (double) point[d] - (double) data[i][d];

			sum += diff * diff;
		}
		distances[i] = sqrt(sum);
	}

	/* Find k nearest neighbors */
	for (i = 0; i < k; i++)
	{
		int			min_idx = -1;
		double		min_dist = DBL_MAX;

		for (j = 0; j < nvec; j++)
		{
			bool		already_selected = false;
			int			l;

			for (l = 0; l < i; l++)
				if (neighbors[l] == j)
				{
					already_selected = true;
					break;
				}
			if (already_selected)
				continue;

			if (distances[j] < min_dist)
			{
				min_dist = distances[j];
				min_idx = j;
			}
		}
		neighbors[i] = min_idx;
	}

	/* Calculate sum of reachability distances */
	for (i = 0; i < k; i++)
		sum_reach_dist += reachability_distance(
												point, data[neighbors[i]], data, nvec, dim, k);

	lrd = (sum_reach_dist > 0.0) ? (double) k / sum_reach_dist : 0.0;

	nfree(distances);
	nfree(neighbors);

	return lrd;
}

/*
 * detect_anomalies_lof
 * --------------------
 * Local Outlier Factor (LOF) anomaly detection.
 *
 * SQL Arguments:
 *   table_name    - Source table with vectors
 *   vector_column - Vector column name
 *   k             - Number of neighbors (default: 20)
 *   threshold     - LOF threshold (default: 1.5)
 *
 * Returns:
 *   Boolean array indicating anomalies (true = anomaly)
 */
PG_FUNCTION_INFO_V1(detect_anomalies_lof);

Datum
detect_anomalies_lof(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *vector_column = NULL;
	int			k;
	double		threshold;
	char *tbl_str = NULL;
	char *vec_col_str = NULL;
	float	  **vectors;
	int			nvec,
				dim;
	double *lof_scores = NULL;
	bool *anomalies = NULL;
	int			i;
	ArrayType *result = NULL;
	Datum *result_datums = NULL;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	k = PG_ARGISNULL(2) ? 20 : PG_GETARG_INT32(2);
	threshold = PG_ARGISNULL(3) ? 1.5 : PG_GETARG_FLOAT8(3);

	if (k < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be at least 1")));
	if (threshold < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("threshold must be non-negative")));

	tbl_str = text_to_cstring(table_name);
	vec_col_str = text_to_cstring(vector_column);

	vectors = neurondb_fetch_vectors_from_table(
												tbl_str, vec_col_str, &nvec, &dim);
	if (vectors == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(vec_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(vec_col_str);
		/* Free vectors array and rows if vectors is not NULL */
		if (vectors != NULL)
		{
			for (int i = 0; i < nvec; i++)
			{
				if (vectors[i] != NULL)
					nfree(vectors[i]);
			}
			nfree(vectors);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (k >= nvec)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be less than number of vectors")));
	if (nvec < k + 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least k+1 vectors for LOF")));

	/* Calculate LOF scores */
	double *lof_scores = NULL;
	nalloc(lof_scores, double, nvec);
	for (i = 0; i < nvec; i++)
	{
		double		lrd_point = local_reachability_density(
														   vectors[i], vectors, nvec, dim, k);
		double		sum_lrd_ratio = 0.0;
		int			j;

		/* Find k nearest neighbors and sum their LRD ratios */
		{
			int			n;

			double *distances = NULL;
			nalloc(distances, double, nvec);
			int *neighbors = NULL;
			nalloc(neighbors, int, k);

			/* Calculate distances */
			for (j = 0; j < nvec; j++)
			{
				double		sum = 0.0;
				int			d;

				if (i == j)
				{
					distances[j] = DBL_MAX;
					continue;
				}

				for (d = 0; d < dim; d++)
				{
					double		diff = (double) vectors[i][d] -
						(double) vectors[j][d];

					sum += diff * diff;
				}
				distances[j] = sqrt(sum);
			}

			/* Find k nearest */
			for (n = 0; n < k; n++)
			{
				int			min_idx = -1;
				double		min_dist = DBL_MAX;

				for (j = 0; j < nvec; j++)
				{
					bool		already_selected = false;
					int			m;

					for (m = 0; m < n; m++)
						if (neighbors[m] == j)
						{
							already_selected = true;
							break;
						}
					if (already_selected)
						continue;

					if (distances[j] < min_dist)
					{
						min_dist = distances[j];
						min_idx = j;
					}
				}
				neighbors[n] = min_idx;
			}

			/* Sum LRD ratios */
			for (n = 0; n < k; n++)
			{
				double		lrd_neighbor = local_reachability_density(
																	  vectors[neighbors[n]], vectors, nvec, dim, k);

				if (lrd_neighbor > 0.0)
					sum_lrd_ratio += lrd_point / lrd_neighbor;
			}

			nfree(distances);
			nfree(neighbors);
		}

		lof_scores[i] = sum_lrd_ratio / (double) k;
	}

	/* Mark anomalies */
	anomalies = (bool *) palloc0(sizeof(bool) * nvec);
	for (i = 0; i < nvec; i++)
		anomalies[i] = (lof_scores[i] > threshold);

	/* Build result array */
	Datum *result_datums = NULL;
	nalloc(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = BoolGetDatum(anomalies[i]);

	get_typlenbyvalalign(BOOLOID, &typlen, &typbyval, &typalign);
	result = construct_array(
							 result_datums, nvec, BOOLOID, typlen, typbyval, typalign);

	for (i = 0; i < nvec; i++)
		nfree(vectors[i]);
	nfree(vectors);
	nfree(lof_scores);
	nfree(anomalies);
	nfree(result_datums);
	nfree(tbl_str);
	nfree(vec_col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * One-Class SVM implementation (simplified RBF kernel)
 * Note: Full SVM solver would require quadratic programming
 * This is a simplified version using approximate methods
 */
PG_FUNCTION_INFO_V1(detect_anomalies_ocsvm);

Datum
detect_anomalies_ocsvm(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *vector_column = NULL;
	double		nu;
	double		gamma;
	char *tbl_str = NULL;
	char *vec_col_str = NULL;
	float	  **vectors;
	int			nvec,
				dim;
	double *scores = NULL;
	bool *anomalies = NULL;
	int			i,
				j,
				d;
	ArrayType *result = NULL;
	Datum *result_datums = NULL;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	nu = PG_ARGISNULL(2) ? 0.1 : PG_GETARG_FLOAT8(2);
	gamma = PG_ARGISNULL(3) ? 1.0 : PG_GETARG_FLOAT8(3);

	if (nu <= 0.0 || nu > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("nu must be between 0 and 1")));
	if (gamma <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("gamma must be positive")));

	tbl_str = text_to_cstring(table_name);
	vec_col_str = text_to_cstring(vector_column);

	vectors = neurondb_fetch_vectors_from_table(
												tbl_str, vec_col_str, &nvec, &dim);
	if (vectors == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(vec_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(vec_col_str);
		/* Free vectors array and rows if vectors is not NULL */
		if (vectors != NULL)
		{
			for (int i = 0; i < nvec; i++)
			{
				if (vectors[i] != NULL)
					nfree(vectors[i]);
			}
			nfree(vectors);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (nvec < 2)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least 2 vectors")));

	/* Simplified One-Class SVM using RBF kernel */
	/* Calculate decision function for each point */
	double *scores = NULL;
	nalloc(scores, double, nvec);

	/* Use support vectors (subset of training data) */
	int			n_sv = (int) (nu * nvec);

	if (n_sv < 1)
		n_sv = 1;
	if (n_sv > nvec)
		n_sv = nvec;

	for (i = 0; i < nvec; i++)
	{
		double		decision = 0.0;
		int			sv_idx;

		/* Sum over support vectors */
		for (sv_idx = 0; sv_idx < n_sv; sv_idx++)
		{
			double		kernel_val = 0.0;
			int			sv = (sv_idx * nvec) / n_sv;

			/* RBF kernel: exp(-gamma * ||x - sv||^2) */
			for (d = 0; d < dim; d++)
			{
				double		diff = (double) vectors[i][d] -
					(double) vectors[sv][d];

				kernel_val += diff * diff;
			}
			kernel_val = exp(-gamma * kernel_val);
			decision += kernel_val;
		}
		decision /= n_sv;
		scores[i] = decision;
	}

	/* Determine threshold (use quantile based on nu) */
	{
		int			threshold_idx;

		double *sorted_scores = NULL;
		nalloc(sorted_scores, double, nvec);
		memcpy(sorted_scores, scores, sizeof(double) * nvec);
		qsort(sorted_scores, nvec, sizeof(double), double_compare);

		threshold_idx = (int) (nu * nvec);
		if (threshold_idx >= nvec)
			threshold_idx = nvec - 1;

		/* Mark anomalies (low decision scores) */
		anomalies = (bool *) palloc0(sizeof(bool) * nvec);
		for (i = 0; i < nvec; i++)
			anomalies[i] = (scores[i] < sorted_scores[threshold_idx]);

		nfree(sorted_scores);
	}

	/* Build result array */
	Datum *result_datums = NULL;
	nalloc(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = BoolGetDatum(anomalies[i]);

	get_typlenbyvalalign(BOOLOID, &typlen, &typbyval, &typalign);
	result = construct_array(
							 result_datums, nvec, BOOLOID, typlen, typbyval, typalign);

	for (i = 0; i < nvec; i++)
		nfree(vectors[i]);
	nfree(vectors);
	nfree(scores);
	nfree(anomalies);
	nfree(result_datums);
	nfree(tbl_str);
	nfree(vec_col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}
