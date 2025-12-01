/*-------------------------------------------------------------------------
 *
 * analytics.c
 *    Vector analytics and machine learning analysis.
 *
 * This module implements comprehensive vector analytics including clustering,
 * dimensionality reduction, outlier detection, and quality metrics.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/analytics.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

/*
 * feedback_loop_integrate
 *    Feedback loop integration: records feedback in a dedicated table,
 *    updating aggregations. Table: neurondb_feedback (query TEXT, result TEXT,
 *    rating REAL, ts TIMESTAMPTZ DEFAULT now()). If the table does not exist, creates it.
 */
PG_FUNCTION_INFO_V1(feedback_loop_integrate);

Datum
feedback_loop_integrate(PG_FUNCTION_ARGS)
{
	text	   *query = PG_GETARG_TEXT_PP(0);
	text	   *result = PG_GETARG_TEXT_PP(1);
	float4		user_rating = PG_GETARG_FLOAT4(2);
	char	   *query_str;
	char	   *result_str;
	StringInfoData sql;
	const char *tbl_def;
	int			ret;
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	query_str = text_to_cstring(query);
	result_str = text_to_cstring(result);

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	tbl_def = "CREATE TABLE IF NOT EXISTS neurondb_feedback ("
		"id SERIAL PRIMARY KEY, "
		"query TEXT NOT NULL, "
		"result TEXT NOT NULL, "
		"rating REAL NOT NULL, "
		"ts TIMESTAMPTZ NOT NULL DEFAULT now()"
		")";
	ret = ndb_spi_execute(spi_session, tbl_def, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(query_str);
		NDB_FREE(result_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create neurondb_feedback table")));
	}

	/* Insert feedback row. */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "INSERT INTO neurondb_feedback (query, result, rating) VALUES "
					 "($$%s$$, $$%s$$, %g)",
					 query_str,
					 result_str,
					 user_rating);
	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	if (ret != SPI_OK_INSERT)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(query_str);
		NDB_FREE(result_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to insert feedback row")));
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);
	NDB_FREE(query_str);
	NDB_FREE(result_str);

	PG_RETURN_BOOL(true);
}

/* DBSCAN moved to ml_dbscan.c */

/*
 * =============================================================================
 * PCA - Principal Component Analysis
 * =============================================================================
 * Dimensionality reduction via singular value decomposition (SVD)
 * - n_components: Target dimension (must be <= original dimension)
 * - Returns projected vectors in lower dimensional space
 */

/* Power iteration method for computing dominant eigenvector */
static void
pca_power_iteration(float **data,
					int nvec,
					int dim,
					float *eigvec,
					int max_iter)
{
	float	   *y;
	int			iter,
				i,
				j;
	double		norm;

	NDB_ALLOC(y, float, dim);

	/* Initialize with random vector */
	for (i = 0; i < dim; i++)
		eigvec[i] = (float) (rand() % 1000) / 1000.0f;

	/* Normalize */
	norm = 0.0;
	for (i = 0; i < dim; i++)
		norm += eigvec[i] * eigvec[i];
	norm = sqrt(norm);
	for (i = 0; i < dim; i++)
		eigvec[i] /= norm;

	/* Power iteration - SIMD optimized */
	for (iter = 0; iter < max_iter; iter++)
	{
		/* y = X^T * X * eigvec */
		memset(y, 0, sizeof(float) * dim);

		for (j = 0; j < nvec; j++)
		{
			/* Use SIMD-optimized dot product */
			double		dot = neurondb_dot_product(data[j], eigvec, dim);

			for (i = 0; i < dim; i++)
				y[i] += data[j][i] * dot;
		}

		/* Normalize y */
		norm = 0.0;
		for (i = 0; i < dim; i++)
			norm += y[i] * y[i];
		norm = sqrt(norm);

		if (norm < 1e-10)
			break;

		for (i = 0; i < dim; i++)
			eigvec[i] = y[i] / norm;
	}

	NDB_FREE(y);
}

/* Deflate matrix by removing component of eigenvector */
static void
pca_deflate(float **data, int nvec, int dim, const float *eigvec)
{
	int			i,
				j;

	for (j = 0; j < nvec; j++)
	{
		double		dot = 0.0;

		for (i = 0; i < dim; i++)
			dot += data[j][i] * eigvec[i];

		for (i = 0; i < dim; i++)
			data[j][i] -= dot * eigvec[i];
	}
}

PG_FUNCTION_INFO_V1(reduce_pca);

Datum
reduce_pca(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	int			n_components;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	float	  **components;
	float	  **projected;
	int			nvec,
				dim;
	int			i,
				j,
				c;
	ArrayType  *result_array;
	Datum	   *result_datums;
	float	   *mean;

	/* Parse arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	n_components = PG_GETARG_INT32(2);

	if (n_components < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_components must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: PCA dimensionality reduction on %s.%s "
		 "(n_components=%d)",
		 tbl_str,
		 col_str,
		 n_components);

	/* Fetch vectors */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (nvec == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));

	if (n_components > dim)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_components (%d) cannot exceed "
						"dimension (%d)",
						n_components,
						dim)));

	/* Center the data (subtract mean) */
	NDB_ALLOC(mean, float, dim);
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			mean[i] += data[j][i];
	for (i = 0; i < dim; i++)
		mean[i] /= nvec;
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			data[j][i] -= mean[i];

	/* Compute principal components using power iteration */
	NDB_ALLOC(components, float *, n_components);
	for (c = 0; c < n_components; c++)
	{
		NDB_DECLARE(float *, component_row);
		NDB_ALLOC(component_row, float, dim);
		components[c] = component_row;
		pca_power_iteration(data, nvec, dim, components[c], 100);
		pca_deflate(data, nvec, dim, components[c]);
	}

	/* Restore centered data for projection */
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			data[j][i] += mean[i];
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			data[j][i] -= mean[i];

	/* Project data onto principal components */
	NDB_ALLOC(projected, float *, nvec);
	for (j = 0; j < nvec; j++)
	{
		NDB_DECLARE(float *, projected_row);
		NDB_ALLOC(projected_row, float, n_components);
		projected[j] = projected_row;
		for (c = 0; c < n_components; c++)
		{
			double		dot = 0.0;

			for (i = 0; i < dim; i++)
				dot += data[j][i] * components[c][i];
			projected[j][c] = dot;
		}
	}

	/* Build result array of arrays */
	NDB_ALLOC(result_datums, Datum, nvec);
	for (j = 0; j < nvec; j++)
	{
		ArrayType  *vec_array;
		NDB_DECLARE(Datum *, vec_datums);
		int16		typlen;
		bool		typbyval;
		char		typalign;

		NDB_ALLOC(vec_datums, Datum, n_components);
		for (c = 0; c < n_components; c++)
			vec_datums[c] = Float4GetDatum(projected[j][c]);

		get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);
		vec_array = construct_array(vec_datums,
									n_components,
									FLOAT4OID,
									typlen,
									typbyval,
									typalign);
		result_datums[j] = PointerGetDatum(vec_array);
		NDB_FREE(vec_datums);
	}

	{
		int16		typlen;
		bool		typbyval;
		char		typalign;

		get_typlenbyvalalign(
							 FLOAT4ARRAYOID, &typlen, &typbyval, &typalign);
		result_array = construct_array(result_datums,
									   nvec,
									   FLOAT4ARRAYOID,
									   typlen,
									   typbyval,
									   typalign);
	}

	/* Cleanup */
	for (j = 0; j < nvec; j++)
	{
		NDB_FREE(data[j]);
		NDB_FREE(projected[j]);
	}
	for (c = 0; c < n_components; c++)
		NDB_FREE(components[c]);
	NDB_FREE(data);
	NDB_FREE(projected);
	NDB_FREE(components);
	NDB_FREE(mean);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * =============================================================================
 * Isolation Forest - Outlier Detection
 * =============================================================================
 * Anomaly detection using ensemble of isolation trees
 * - n_trees: Number of trees in the forest (default 100)
 * - contamination: Expected proportion of outliers (0.0-0.5)
 * - Returns anomaly scores (higher = more anomalous)
 */

typedef struct IsoTreeNode
{
	int			split_dim;		/* Dimension to split on (-1 = leaf) */
	float		split_val;		/* Value to split at */
	struct IsoTreeNode *left;
	struct IsoTreeNode *right;
	int			size;			/* Number of points in this node */
}			IsoTreeNode;

/* Build isolation tree recursively */
static IsoTreeNode *
build_iso_tree(float **data,
			   int *indices,
			   int n,
			   int dim,
			   int depth,
			   int max_depth)
{
	IsoTreeNode *node;
	int			i,
				split_dim;
	float		split_val,
				min_val,
				max_val;
	int			left_count,
				right_count;
	int		   *left_indices,
			   *right_indices;

	node = (IsoTreeNode *) palloc0(sizeof(IsoTreeNode));
	node->size = n;

	/* Stopping criteria */
	if (n <= 1 || depth >= max_depth)
	{
		node->split_dim = -1;	/* Leaf node */
		return node;
	}

	/* Random split dimension */
	split_dim = rand() % dim;
	node->split_dim = split_dim;

	/* Find min/max in this dimension */
	min_val = max_val = data[indices[0]][split_dim];
	for (i = 1; i < n; i++)
	{
		float		val = data[indices[i]][split_dim];

		if (val < min_val)
			min_val = val;
		if (val > max_val)
			max_val = val;
	}

	/* Random split value */
	if (max_val - min_val < 1e-6)
	{
		node->split_dim = -1;	/* Can't split */
		return node;
	}
	split_val = min_val + (float) (((double) rand() / (double) RAND_MAX)) * (max_val - min_val);
	node->split_val = split_val;

	/* Partition indices */
	NDB_ALLOC(left_indices, int, n);
	NDB_ALLOC(right_indices, int, n);
	left_count = right_count = 0;

	for (i = 0; i < n; i++)
	{
		if (data[indices[i]][split_dim] < split_val)
			left_indices[left_count++] = indices[i];
		else
			right_indices[right_count++] = indices[i];
	}

	/* Recursively build subtrees */
	if (left_count > 0)
		node->left = build_iso_tree(data,
									left_indices,
									left_count,
									dim,
									depth + 1,
									max_depth);
	if (right_count > 0)
		node->right = build_iso_tree(data,
									 right_indices,
									 right_count,
									 dim,
									 depth + 1,
									 max_depth);

	NDB_FREE(left_indices);
	NDB_FREE(right_indices);

	return node;
}

/* Compute path length for a point in the tree */
static double
iso_tree_path_length(IsoTreeNode * node, const float *point, int depth)
{
	double		h;

	if (node->split_dim == -1)
	{
		/* Leaf node - estimate average path length */
		if (node->size <= 1)
			return depth;
		/* Average path length for BST of size n */
		h = log(node->size) + 0.5772156649; /* Euler's constant */
		return depth + h;
	}

	/* Traverse tree */
	if (point[node->split_dim] < node->split_val && node->left)
		return iso_tree_path_length(node->left, point, depth + 1);
	else if (node->right)
		return iso_tree_path_length(node->right, point, depth + 1);
	else
		return depth;
}

/* Free isolation tree */
static void
free_iso_tree(IsoTreeNode * node)
{
	if (node == NULL)
		return;
	free_iso_tree(node->left);
	free_iso_tree(node->right);
	NDB_FREE(node);
}

PG_FUNCTION_INFO_V1(detect_outliers);

Datum
detect_outliers(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	int			n_trees;
	float		contamination;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	int			nvec,
				dim;
	IsoTreeNode **forest;
	double	   *scores;
	int			i,
				t;
	int		   *indices;
	int			max_depth;
	double		avg_path_length_full;
	ArrayType  *result_array;
	Datum	   *result_datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	/* Parse arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	n_trees = PG_GETARG_INT32(2);
	contamination = PG_GETARG_FLOAT4(3);

	if (n_trees < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_trees must be at least 1")));

	if (contamination < 0.0 || contamination > 0.5)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("contamination must be between 0.0 and "
						"0.5")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: Isolation Forest on %s.%s (n_trees=%d, contamination=%.3f)",
		 tbl_str,
		 col_str,
		 n_trees,
		 contamination);

	/* Fetch vectors */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (nvec == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));

	/* Build forest of isolation trees */
	max_depth = (int) ceil(log2(nvec));
	forest = (IsoTreeNode * *) palloc(sizeof(IsoTreeNode *) * n_trees);
	indices = (int *) palloc(sizeof(int) * nvec);

	for (t = 0; t < n_trees; t++)
	{
		/* Sample subset of data */
		int			sample_size = (nvec < 256) ? nvec : 256;

		for (i = 0; i < sample_size; i++)
			indices[i] = rand() % nvec;

		forest[t] = build_iso_tree(
								   data, indices, sample_size, dim, 0, max_depth);
	}

	/* Compute anomaly scores */
	avg_path_length_full = (nvec > 1) ? 2.0 * (log(nvec - 1) + 0.5772156649)
		- 2.0 * (nvec - 1.0) / nvec
		: 0.0;
	scores = (double *) palloc0(sizeof(double) * nvec);

	for (i = 0; i < nvec; i++)
	{
		double		avg_path = 0.0;

		for (t = 0; t < n_trees; t++)
			avg_path += iso_tree_path_length(forest[t], data[i], 0);
		avg_path /= n_trees;

		/* Anomaly score: 2^(-avg_path / c) where c is avg path length */
		if (avg_path_length_full > 0)
			scores[i] = pow(2.0, -avg_path / avg_path_length_full);
		else
			scores[i] = 0.0;
	}

	/* Build result array */
	NDB_ALLOC(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = Float4GetDatum((float) scores[i]);

	get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);
	result_array = construct_array(
								   result_datums, nvec, FLOAT4OID, typlen, typbyval, typalign);

	/* Cleanup */
	for (t = 0; t < n_trees; t++)
		free_iso_tree(forest[t]);
	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	NDB_FREE(forest);
	NDB_FREE(scores);
	NDB_FREE(indices);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * =============================================================================
 * KNN Graph Construction
 * =============================================================================
 * Build k-nearest neighbor graph for vectors
 * - k: Number of neighbors per point
 * - Returns edge list as array of (source, target, distance) tuples
 */

typedef struct KNNEdge
{
	int			target;
	float		distance;
}			KNNEdge;

/* Comparison function for sorting edges by distance */
static int
knn_edge_compare(const void *a, const void *b)
{
	const		KNNEdge *ea = (const KNNEdge *) a;
	const		KNNEdge *eb = (const KNNEdge *) b;

	if (ea->distance < eb->distance)
		return -1;
	if (ea->distance > eb->distance)
		return 1;
	return 0;
}

PG_FUNCTION_INFO_V1(build_knn_graph);

Datum
build_knn_graph(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	int			k;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	int			nvec,
				dim;
	int			i,
				j,
				n;
	KNNEdge    *edges;
	ArrayType  *result_array;
	Datum	   *result_datums;
	int			result_count;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	/* Parse arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	k = PG_GETARG_INT32(2);

	if (k < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	elog(DEBUG1,
		 "neurondb: Building KNN graph on %s.%s (k=%d)",
		 tbl_str,
		 col_str,
		 k);

	/* Fetch vectors */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (nvec == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));

	if (k >= nvec)
		k = nvec - 1;

	/* Build KNN graph */
	edges = (KNNEdge *) palloc(sizeof(KNNEdge) * nvec);
	result_count = 0;
	result_datums = (Datum *) palloc(sizeof(Datum) * nvec * k * 3);

	for (i = 0; i < nvec; i++)
	{
		double		dist_sq;
		double		diff;

		/* Compute distances to all other points */
		for (j = 0; j < nvec; j++)
		{
			if (i == j)
				continue;

			dist_sq = 0.0;
			for (n = 0; n < dim; n++)
			{
				diff = (double) data[i][n] - (double) data[j][n];
				dist_sq += diff * diff;
			}
			edges[j].target = j;
			edges[j].distance = sqrt(dist_sq);
		}

		/* Sort by distance and take k nearest */
		qsort(edges, nvec, sizeof(KNNEdge), knn_edge_compare);

		/* Add k nearest edges to result */
		for (j = 0; j < k && j < nvec - 1; j++)
		{
			result_datums[result_count++] = Int32GetDatum(i);
			result_datums[result_count++] =
				Int32GetDatum(edges[j].target);
			result_datums[result_count++] =
				Float4GetDatum(edges[j].distance);
		}
	}

	get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);
	result_array = construct_array(result_datums,
								   result_count,
								   FLOAT4OID,
								   typlen,
								   typbyval,
								   typalign);

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	NDB_FREE(edges);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * =============================================================================
 * Embedding Quality Metrics
 * =============================================================================
 * Compute quality metrics for embeddings (silhouette score, etc.)
 * - Returns quality score between -1 and 1 (higher = better)
 */

PG_FUNCTION_INFO_V1(compute_embedding_quality);

Datum
compute_embedding_quality(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *column_name;
	text	   *cluster_column;
	char	   *tbl_str;
	char	   *col_str;
	char	   *cluster_col_str;
	float	  **data;
	int		   *clusters;
	int			nvec,
				dim;
	int			i,
				j;
	double	   *a_scores;		/* Average distance to same cluster */
	double	   *b_scores;		/* Average distance to nearest other cluster */
	double		silhouette;
	StringInfoData sql;
	int			ret;
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	/* Parse arguments */
	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	cluster_column = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);
	cluster_col_str = text_to_cstring(cluster_column);

	elog(DEBUG1,
		 "neurondb: Computing embedding quality for %s.%s (clusters=%s)",
		 tbl_str,
		 col_str,
		 cluster_col_str);

	/* Fetch vectors */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (nvec == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));

	/* Fetch cluster assignments */
	oldcontext = CurrentMemoryContext;
	NDB_ALLOC(clusters, int, nvec);

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql, "SELECT %s FROM %s", cluster_col_str, tbl_str);
	ret = ndb_spi_execute(spi_session, sql.data, true, 0);

	if (ret != SPI_OK_SELECT || (int) SPI_processed != nvec)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(clusters);
		NDB_FREE(tbl_str);
		NDB_FREE(cluster_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Failed to fetch cluster assignments")));
	}

	for (i = 0; i < nvec; i++)
	{
		int32		val;

		if (ndb_spi_get_int32(spi_session, i, 1, &val))
		{
			clusters[i] = val;
		}
		else
		{
			clusters[i] = -1;
		}
	}

	ndb_spi_stringinfo_free(spi_session, &sql);
	NDB_SPI_SESSION_END(spi_session);

	/* Compute silhouette score */
	NDB_ALLOC(a_scores, double, nvec);
	NDB_ALLOC(b_scores, double, nvec);

	for (i = 0; i < nvec; i++)
	{
		int			my_cluster = clusters[i];
		int			same_count = 0;
		double		same_dist = 0.0;
		double		min_other_dist = DBL_MAX;
		double		dist;
		int			d;
		double		diff;

		if (my_cluster == -1)	/* Noise point */
			continue;

		for (j = 0; j < nvec; j++)
		{
			if (i == j)
				continue;

			dist = 0.0;
			for (d = 0; d < dim; d++)
			{
				diff = (double) data[i][d] - (double) data[j][d];
				dist += diff * diff;
			}
			dist = sqrt(dist);

			if (clusters[j] == my_cluster)
			{
				same_dist += dist;
				same_count++;
			}
			else if (clusters[j] != -1)
			{
				if (dist < min_other_dist)
					min_other_dist = dist;
			}
		}

		if (same_count > 0)
			a_scores[i] = same_dist / same_count;
		b_scores[i] = min_other_dist;
	}

	/* Average silhouette */
	{
		int			valid_count = 0;
		double		s;

		silhouette = 0.0;
		for (i = 0; i < nvec; i++)
		{
			if (clusters[i] == -1)
				continue;

			if (a_scores[i] < b_scores[i])
				s = 1.0 - a_scores[i] / b_scores[i];
			else if (a_scores[i] > b_scores[i])
				s = b_scores[i] / a_scores[i] - 1.0;
			else
				s = 0.0;

			silhouette += s;
			valid_count++;
		}

		if (valid_count > 0)
			silhouette /= valid_count;
	}

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	NDB_FREE(clusters);
	NDB_FREE(a_scores);
	NDB_FREE(b_scores);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);
	NDB_FREE(cluster_col_str);

	PG_RETURN_FLOAT8(silhouette);
}
