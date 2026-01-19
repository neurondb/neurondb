/*-------------------------------------------------------------------------
 *
 * analytics.c
 *    Vector analytics and machine learning analysis.
 *
 * This module implements comprehensive vector analytics including clustering,
 * dimensionality reduction, outlier detection, and quality metrics.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
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
#include "parser/parse_type.h"
#include "nodes/makefuncs.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_json.h"

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
	text	   *query;
	text	   *result;
	float4		user_rating;
	char *query_str = NULL;
	char *result_str = NULL;
	const char *tbl_def;
	int			ret;
	NdbSpiSession *spi_session = NULL;
	MemoryContext oldcontext;

	/* Validate argument count */
	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: feedback_loop_integrate requires at least 3 arguments")));

	query = PG_GETARG_TEXT_PP(0);
	result = PG_GETARG_TEXT_PP(1);
	user_rating = PG_GETARG_FLOAT4(2);

	query_str = text_to_cstring(query);
	result_str = text_to_cstring(result);

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* PostgreSQL 18 B-tree deduplication bug workaround: create sequence separately */
	tbl_def = "CREATE SEQUENCE IF NOT EXISTS neurondb_feedback_id_seq";
	ret = ndb_spi_execute(spi_session, tbl_def, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		NDB_SPI_SESSION_END(spi_session);
		nfree(query_str);
		nfree(result_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("failed to create feedback sequence")));
	}

	tbl_def = "CREATE TABLE IF NOT EXISTS neurondb_feedback ("
		"id INTEGER DEFAULT nextval('neurondb_feedback_id_seq') PRIMARY KEY, "
		"query TEXT NOT NULL, "
		"result TEXT NOT NULL, "
		"rating REAL NOT NULL, "
		"ts TIMESTAMPTZ NOT NULL DEFAULT now()"
		")";
	ret = ndb_spi_execute(spi_session, tbl_def, false, 0);
	if (ret != SPI_OK_UTILITY)
	{
		NDB_SPI_SESSION_END(spi_session);
		nfree(query_str);
		nfree(result_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: failed to create neurondb_feedback table")));
	}

	/* Use parameterized query to prevent SQL injection */
	{
		const char *insert_query = "INSERT INTO neurondb_feedback (query, result, rating) VALUES ($1, $2, $3)";
		Oid			argtypes[3] = {TEXTOID, TEXTOID, FLOAT4OID};
		Datum		values[3];
		const char nulls[3] = {' ', ' ', ' '};

		/* Validate rating is in reasonable range (0-5) */
		if (user_rating < 0.0f || user_rating > 5.0f)
		{
			NDB_SPI_SESSION_END(spi_session);
			nfree(query_str);
			nfree(result_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: rating must be between 0 and 5, got %g", user_rating)));
		}

		values[0] = PointerGetDatum(query);
		values[1] = PointerGetDatum(result);
		values[2] = Float4GetDatum(user_rating);

		ret = ndb_spi_execute_with_args(spi_session, insert_query, 3, argtypes, values, nulls, false, 0);
		if (ret != SPI_OK_INSERT)
		{
			NDB_SPI_SESSION_END(spi_session);
			nfree(query_str);
			nfree(result_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: failed to insert feedback row")));
		}
	}

	NDB_SPI_SESSION_END(spi_session);
	nfree(query_str);
	nfree(result_str);

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

static void
pca_power_iteration(float **data,
					int nvec,
					int dim,
					float *eigvec,
					int max_iter)
{
	float *y = NULL;
	int			iter,
				i,
				j;
	double		norm;

	nalloc(y, float, dim);

	for (i = 0; i < dim; i++)
		eigvec[i] = (float) (rand() % 1000) / 1000.0f;

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

	nfree(y);
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
	ArrayType  *result_array = NULL;
	char	   *col_str = NULL;
	char	   *tbl_str = NULL;
	float	   *mean = NULL;
	float	   **components = NULL;
	float	   **data = NULL;
	float	   **centered_data = NULL;	/* Copy of centered data for projection */
	float	   **projected = NULL;
	int			c = 0;
	int			dim = 0;
	int			i = 0;
	int			j = 0;
	int			n_components = 0;
	int			nvec = 0;
	text	   *column_name = NULL;
	text	   *table_name = NULL;
	char		typalign = 0;
	bool		typbyval = false;
	int16		typlen = 0;

	/* Validate argument count */
	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: reduce_pca requires at least 3 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	n_components = PG_GETARG_INT32(2);

	if (n_components < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_components must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);


	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (data == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		if (data != NULL)
		{
			for (j = 0; j < nvec; j++)
			{
				if (data[j] != NULL)
					nfree(data[j]);
			}
			nfree(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (n_components > dim)
	{
		nfree(tbl_str);
		nfree(col_str);
		for (j = 0; j < nvec; j++)
		{
			if (data[j] != NULL)
				nfree(data[j]);
		}
		nfree(data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_components (%d) cannot exceed "
						"dimension (%d)",
						n_components,
						dim)));
	}

	/* Compute mean */
	nalloc(mean, float, dim);
	memset(mean, 0, sizeof(float) * dim);
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			mean[i] += data[j][i];
	for (i = 0; i < dim; i++)
		mean[i] /= nvec;

	/* Center the data */
	for (j = 0; j < nvec; j++)
		for (i = 0; i < dim; i++)
			data[j][i] -= mean[i];

	/* Keep a copy of centered data for projection */
	nalloc(centered_data, float *, nvec);
	for (j = 0; j < nvec; j++)
	{
		nalloc(centered_data[j], float, dim);
		for (i = 0; i < dim; i++)
			centered_data[j][i] = data[j][i];
	}

	/* Compute principal components using power iteration and deflation */
	nalloc(components, float *, n_components);
	for (c = 0; c < n_components; c++)
	{
		float	   *component_row = NULL;

		nalloc(component_row, float, dim);
		components[c] = component_row;
		pca_power_iteration(data, nvec, dim, components[c], 100);
		pca_deflate(data, nvec, dim, components[c]);
	}

	/* Project centered data onto principal components */
	nalloc(projected, float *, nvec);
	for (j = 0; j < nvec; j++)
	{
		float *projected_row = NULL;
		nalloc(projected_row, float, n_components);
		projected[j] = projected_row;
		for (c = 0; c < n_components; c++)
		{
			double		dot = 0.0;

			/* Use original centered data, not deflated residuals */
			for (i = 0; i < dim; i++)
				dot += centered_data[j][i] * components[c][i];
			projected[j][c] = dot;
		}
	}

	/* Build 2D array real[][]: dims = [nvec][n_components] */
	{
		int			dims[2];
		int			lbs[2];
		Datum	   *flat_datums = NULL;
		int			idx = 0;

		dims[0] = nvec;
		dims[1] = n_components;
		lbs[0] = 1;
		lbs[1] = 1;

		nalloc(flat_datums, Datum, nvec * n_components);

		idx = 0;
		for (j = 0; j < nvec; j++)
		{
			for (c = 0; c < n_components; c++)
			{
				/* Validate projected value is finite */
				if (!isfinite(projected[j][c]))
				{
					nfree(flat_datums);
					for (i = 0; i < nvec; i++)
					{
						if (data[i] != NULL)
							nfree(data[i]);
						if (centered_data[i] != NULL)
							nfree(centered_data[i]);
						if (projected[i] != NULL)
							nfree(projected[i]);
					}
					for (c = 0; c < n_components; c++)
					{
						if (components[c] != NULL)
							nfree(components[c]);
					}
					nfree(data);
					nfree(centered_data);
					nfree(projected);
					nfree(components);
					nfree(mean);
					nfree(tbl_str);
					nfree(col_str);
					ereport(ERROR,
							(errcode(ERRCODE_DATA_EXCEPTION),
							 errmsg("reduce_pca: non-finite value in "
									"projected[%d][%d]", j, c)));
				}

				flat_datums[idx++] = Float4GetDatum(projected[j][c]);
			}
		}

		get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);

		result_array = construct_md_array(flat_datums,
										  NULL,
										  2,
										  dims,
										  lbs,
										  FLOAT4OID,
										  typlen,
										  typbyval,
										  typalign);

		if (result_array == NULL)
		{
			nfree(flat_datums);
			for (j = 0; j < nvec; j++)
			{
				if (data[j] != NULL)
					nfree(data[j]);
				if (centered_data[j] != NULL)
					nfree(centered_data[j]);
				if (projected[j] != NULL)
					nfree(projected[j]);
			}
			for (c = 0; c < n_components; c++)
			{
				if (components[c] != NULL)
					nfree(components[c]);
			}
			nfree(data);
			nfree(centered_data);
			nfree(projected);
			nfree(components);
			nfree(mean);
			nfree(tbl_str);
			nfree(col_str);
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("reduce_pca: failed to construct result array")));
		}

		nfree(flat_datums);
	}
	for (j = 0; j < nvec; j++)
	{
		if (data[j] != NULL)
			nfree(data[j]);
		if (centered_data[j] != NULL)
			nfree(centered_data[j]);
		if (projected[j] != NULL)
			nfree(projected[j]);
	}
	for (c = 0; c < n_components; c++)
	{
		if (components[c] != NULL)
			nfree(components[c]);
	}
	nfree(data);
	nfree(centered_data);
	nfree(projected);
	nfree(components);
	nfree(mean);
	nfree(tbl_str);
	nfree(col_str);

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

static IsoTreeNode *
build_iso_tree(float **data,
			   int *indices,
			   int n,
			   int dim,
			   int depth,
			   int max_depth)
{
	IsoTreeNode *node = NULL;
	int			i,
				split_dim;
	float		split_val,
				min_val,
				max_val;
	int			left_count,
				right_count;
	int *left_indices = NULL;
	int *right_indices = NULL;

	nalloc(node, IsoTreeNode, 1);
	node->size = n;

	if (n <= 1 || depth >= max_depth)
	{
		node->split_dim = -1;	/* Leaf node */
		return node;
	}

	split_dim = rand() % dim;
	node->split_dim = split_dim;

	min_val = max_val = data[indices[0]][split_dim];
	for (i = 1; i < n; i++)
	{
		float		val = data[indices[i]][split_dim];

		if (val < min_val)
			min_val = val;
		if (val > max_val)
			max_val = val;
	}

	if (max_val - min_val < 1e-6)
	{
		node->split_dim = -1;	/* Can't split */
		return node;
	}
	split_val = min_val + (float) (((double) rand() / (double) RAND_MAX)) * (max_val - min_val);
	node->split_val = split_val;

	nalloc(left_indices, int, n);
	nalloc(right_indices, int, n);
	left_count = right_count = 0;

	for (i = 0; i < n; i++)
	{
		if (data[indices[i]][split_dim] < split_val)
			left_indices[left_count++] = indices[i];
		else
			right_indices[right_count++] = indices[i];
	}

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

	nfree(left_indices);
	nfree(right_indices);

	return node;
}

static double
iso_tree_path_length(IsoTreeNode * node, const float *point, int depth)
{
	double		h;

	if (node->split_dim == -1)
	{
		if (node->size <= 1)
			return depth;
		h = log(node->size) + 0.5772156649; /* Euler's constant */
		return depth + h;
	}

	if (point[node->split_dim] < node->split_val && node->left)
		return iso_tree_path_length(node->left, point, depth + 1);
	else if (node->right)
		return iso_tree_path_length(node->right, point, depth + 1);
	else
		return depth;
}

static void
free_iso_tree(IsoTreeNode * node)
{
	if (node == NULL)
		return;
	free_iso_tree(node->left);
	free_iso_tree(node->right);
	nfree(node);
}

PG_FUNCTION_INFO_V1(detect_outliers);

Datum
detect_outliers(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *column_name = NULL;
	int			n_trees;
	float		contamination;
	char *tbl_str = NULL;
	char *col_str = NULL;
	float	  **data;
	int			nvec,
				dim;

	IsoTreeNode * * forest = NULL;
	double *scores = NULL;
	int			i,
				t;

	int *indices = NULL;
	int			max_depth;
	double		avg_path_length_full;
	ArrayType *result_array = NULL;

	Datum *result_datums = NULL;
	char		typalign = 0;
	bool		typbyval = false;
	int16		typlen = 0;

	/* Validate argument count */
	if (PG_NARGS() < 2 || PG_NARGS() > 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: detect_outliers requires 2 to 4 arguments")));

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


	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (nvec == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));

	max_depth = (int) ceil(log2(nvec));
	nalloc(forest, IsoTreeNode *, n_trees);
	nalloc(indices, int, nvec);

	/* Seed random number generator for reproducible results */
	/* Note: In production, this should be seeded once in module init */
	srand((unsigned int) time(NULL));

	for (t = 0; t < n_trees; t++)
	{
		int			sample_size = (nvec < 256) ? nvec : 256;

		for (i = 0; i < sample_size; i++)
			indices[i] = rand() % nvec;

		forest[t] = build_iso_tree(
								   data, indices, sample_size, dim, 0, max_depth);
	}

	avg_path_length_full = (nvec > 1) ? 2.0 * (log(nvec - 1) + 0.5772156649)
		- 2.0 * (nvec - 1.0) / nvec
		: 0.0;
	nalloc(scores, double, nvec);

	for (i = 0; i < nvec; i++)
	{
		double		avg_path = 0.0;

		for (t = 0; t < n_trees; t++)
			avg_path += iso_tree_path_length(forest[t], data[i], 0);
		avg_path /= n_trees;

		if (avg_path_length_full > 0)
			scores[i] = pow(2.0, -avg_path / avg_path_length_full);
		else
			scores[i] = 0.0;
	}

	nalloc(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = Float4GetDatum((float) scores[i]);

	get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);
	result_array = construct_array(
								   result_datums, nvec, FLOAT4OID, typlen, typbyval, typalign);

	for (t = 0; t < n_trees; t++)
		free_iso_tree(forest[t]);
	for (i = 0; i < nvec; i++)
		nfree(data[i]);
	nfree(data);
	nfree(forest);
	nfree(scores);
	nfree(indices);
	nfree(result_datums);
	nfree(tbl_str);
	nfree(col_str);

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
	text *table_name = NULL;
	text *column_name = NULL;
	int			k;
	char *tbl_str = NULL;
	char *col_str = NULL;
	float	  **data;
	int			nvec,
				dim;
	int			i,
				j,
				n;

	KNNEdge *edges = NULL;
	ArrayType *result_array = NULL;

	Datum *result_datums = NULL;
	int			result_count = 0;
	char		typalign = 0;
	bool		typbyval = false;
	int16		typlen = 0;

	/* Validate argument count */
	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: build_knn_graph requires at least 3 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	k = PG_GETARG_INT32(2);

	if (k < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be at least 1")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);


	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (data == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL)
		{
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					nfree(data[i]);
			}
			nfree(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (k >= nvec)
		k = nvec - 1;

	/* Allocate edges array - we need nvec-1 edges per node (excluding self) */
	nalloc(edges, KNNEdge, nvec - 1);
	result_count = 0;
	/* Store as array of real[3] arrays: [src, dst, dist] */
	nalloc(result_datums, Datum, nvec * k);

	for (i = 0; i < nvec; i++)
	{
		int			edge_count = 0;
		double		dist_sq;
		double		diff;

		/* Build edges list excluding self */
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
			edges[edge_count].target = j;
			edges[edge_count].distance = sqrt(dist_sq);
			edge_count++;
		}

		/* Sort by distance */
		qsort(edges, edge_count, sizeof(KNNEdge), knn_edge_compare);

		/* Take top k neighbors */
		for (j = 0; j < k && j < edge_count; j++)
		{
			/* Build array [src, dst, dist] for this edge */
			Datum	   *edge_datums = NULL;
			ArrayType  *edge_array = NULL;
			int			edge_idx = result_count++;

			nalloc(edge_datums, Datum, 3);
			edge_datums[0] = Int32GetDatum(i);
			edge_datums[1] = Int32GetDatum(edges[j].target);
			edge_datums[2] = Float4GetDatum(edges[j].distance);

			get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);
			edge_array = construct_array(edge_datums, 3, FLOAT4OID, typlen, typbyval, typalign);
			nfree(edge_datums);

			if (edge_array == NULL)
			{
				/* Cleanup */
				for (i = 0; i < nvec; i++)
					nfree(data[i]);
				nfree(data);
				nfree(edges);
				for (i = 0; i < edge_idx; i++)
				{
					if (result_datums[i] != 0)
					{
						ArrayType  *arr = DatumGetArrayTypeP(result_datums[i]);
						nfree(arr);
					}
				}
				nfree(result_datums);
				nfree(tbl_str);
				nfree(col_str);
				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("build_knn_graph: failed to construct edge array")));
			}

			result_datums[edge_idx] = PointerGetDatum(edge_array);
		}
	}

	/* Build 2D array real[][3]: dims = [result_count][3] */
	/* Convert array-of-arrays to true 2D array */
	{
		int			dims[2];
		int			lbs[2];
		Datum	   *flat_datums = NULL;
		int			idx = 0;

		dims[0] = result_count;
		dims[1] = 3;
		lbs[0] = 1;
		lbs[1] = 1;

		nalloc(flat_datums, Datum, result_count * 3);

		idx = 0;
		for (i = 0; i < result_count; i++)
		{
			ArrayType  *edge_array = NULL;
			Datum	   *edge_elems = NULL;
			bool	   *nulls = NULL;
			int			nelems;
			int			e;

			if (result_datums[i] != 0)
			{
				edge_array = DatumGetArrayTypeP(result_datums[i]);
				deconstruct_array(edge_array,
								  FLOAT4OID,
								  sizeof(float4),
								  true,
								  'i',
								  &edge_elems,
								  &nulls,
								  &nelems);

				for (e = 0; e < nelems && e < 3; e++)
				{
					if (!nulls[e])
						flat_datums[idx++] = edge_elems[e];
					else
						flat_datums[idx++] = Float4GetDatum(0.0);
				}
				/* Free the inner array */
				nfree(edge_array);
			}
			else
			{
				/* Fill with zeros if array is NULL */
				flat_datums[idx++] = Float4GetDatum(0.0);
				flat_datums[idx++] = Float4GetDatum(0.0);
				flat_datums[idx++] = Float4GetDatum(0.0);
			}
		}

		get_typlenbyvalalign(FLOAT4OID, &typlen, &typbyval, &typalign);

		result_array = construct_md_array(flat_datums,
										  NULL,
										  2,
										  dims,
										  lbs,
										  FLOAT4OID,
										  typlen,
										  typbyval,
										  typalign);

		nfree(flat_datums);
	}

	/* Free result_datums (inner arrays already freed above) */

	for (i = 0; i < nvec; i++)
		nfree(data[i]);
	nfree(data);
	nfree(edges);
	nfree(result_datums);
	nfree(tbl_str);
	nfree(col_str);

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
	text *table_name = NULL;
	text *column_name = NULL;
	text *cluster_column = NULL;
	char *tbl_str = NULL;
	char *col_str = NULL;
	char *cluster_col_str = NULL;
	float	  **data;

	int *clusters = NULL;
	int			nvec,
				dim;
	int			i,
				j;

	double *a_scores = NULL;	/* Average distance to same cluster */
	double *b_scores = NULL;	/* Average distance to nearest other
										 * cluster */
	double		silhouette;
	StringInfoData sql;
	int			ret;

	NdbSpiSession *spi_session = NULL;
	MemoryContext oldcontext;

	/* Validate argument count */
	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: compute_embedding_quality requires at least 3 arguments")));

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);
	cluster_column = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);
	cluster_col_str = text_to_cstring(cluster_column);


	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (data == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		nfree(cluster_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		nfree(cluster_col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL)
		{
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					nfree(data[i]);
			}
			nfree(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	oldcontext = CurrentMemoryContext;
	nalloc(clusters, int, nvec);

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &sql);
	/* Note: No ORDER BY clause - views don't have ctid, and ordering isn't required */
	appendStringInfo(&sql, "SELECT %s FROM %s", cluster_col_str, tbl_str);
	ret = ndb_spi_execute(spi_session, sql.data, true, 0);

	if (ret != SPI_OK_SELECT || (int) SPI_processed != nvec)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		NDB_SPI_SESSION_END(spi_session);
		nfree(clusters);
		nfree(tbl_str);
		nfree(col_str);
		nfree(cluster_col_str);
		/* Free data array and rows */
		for (i = 0; i < nvec; i++)
		{
			if (data[i] != NULL)
				nfree(data[i]);
		}
		nfree(data);
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

	nalloc(a_scores, double, nvec);
	nalloc(b_scores, double, nvec);

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

	for (i = 0; i < nvec; i++)
		nfree(data[i]);
	nfree(data);
	nfree(clusters);
	nfree(a_scores);
	nfree(b_scores);
	nfree(tbl_str);
	nfree(col_str);
	nfree(cluster_col_str);

	PG_RETURN_FLOAT8(silhouette);
}

/*
 * =============================================================================
 * VECTOR STATISTICS FUNCTIONS
 * =============================================================================
 */

/*
 * vector_statistics - Compute comprehensive statistics for a vector column
 * Returns JSONB with mean, variance, stddev, min, max, correlation matrix
 */
PG_FUNCTION_INFO_V1(vector_statistics);

Datum
vector_statistics(PG_FUNCTION_ARGS)
{
	text	   *table_name = NULL;
	text	   *column_name = NULL;
	char	   *tbl_str = NULL;
	char	   *col_str = NULL;
	float	  **data = NULL;
	int			nvec = 0;
	int			dim = 0;
	int			i, j, d;
	float	   *mean = NULL;
	float	   *variance = NULL;
	float	   *stddev = NULL;
	float	   *min_vals = NULL;
	float	   *max_vals = NULL;
	StringInfoData json;
	MemoryContext oldctx;

	if (PG_NARGS() < 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("vector_statistics requires 2 arguments: table_name, column_name")));

	table_name = PG_GETARG_TEXT_PP(0);
	column_name = PG_GETARG_TEXT_PP(1);

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(column_name);

	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);
	if (data == NULL || nvec == 0)
	{
		nfree(tbl_str);
		nfree(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found in table")));
	}

	oldctx = MemoryContextSwitchTo(CurrentMemoryContext);

	/* Allocate arrays for statistics */
	nalloc(mean, float, dim);
	nalloc(variance, float, dim);
	nalloc(stddev, float, dim);
	nalloc(min_vals, float, dim);
	nalloc(max_vals, float, dim);

	/* Initialize */
	memset(mean, 0, sizeof(float) * dim);
	memset(variance, 0, sizeof(float) * dim);
	for (d = 0; d < dim; d++)
	{
		min_vals[d] = data[0][d];
		max_vals[d] = data[0][d];
	}

	/* Compute mean, min, max */
	for (i = 0; i < nvec; i++)
	{
		for (d = 0; d < dim; d++)
		{
			mean[d] += data[i][d];
			if (data[i][d] < min_vals[d])
				min_vals[d] = data[i][d];
			if (data[i][d] > max_vals[d])
				max_vals[d] = data[i][d];
		}
	}

	for (d = 0; d < dim; d++)
		mean[d] /= nvec;

	/* Compute variance and stddev */
	for (i = 0; i < nvec; i++)
	{
		for (d = 0; d < dim; d++)
		{
			float		diff = data[i][d] - mean[d];

			variance[d] += diff * diff;
		}
	}

	for (d = 0; d < dim; d++)
	{
		variance[d] /= nvec;
		stddev[d] = sqrt(variance[d]);
	}

	/* Build JSONB result */
	initStringInfo(&json);
	appendStringInfoString(&json, "{");
	appendStringInfo(&json, "\"count\": %d,", nvec);
	appendStringInfo(&json, "\"dimension\": %d,", dim);
	appendStringInfoString(&json, "\"mean\": [");
	for (d = 0; d < dim; d++)
	{
		if (d > 0)
			appendStringInfoChar(&json, ',');
		appendStringInfo(&json, "%.6f", mean[d]);
	}
	appendStringInfoString(&json, "],\"variance\": [");
	for (d = 0; d < dim; d++)
	{
		if (d > 0)
			appendStringInfoChar(&json, ',');
		appendStringInfo(&json, "%.6f", variance[d]);
	}
	appendStringInfoString(&json, "],\"stddev\": [");
	for (d = 0; d < dim; d++)
	{
		if (d > 0)
			appendStringInfoChar(&json, ',');
		appendStringInfo(&json, "%.6f", stddev[d]);
	}
	appendStringInfoString(&json, "],\"min\": [");
	for (d = 0; d < dim; d++)
	{
		if (d > 0)
			appendStringInfoChar(&json, ',');
		appendStringInfo(&json, "%.6f", min_vals[d]);
	}
	appendStringInfoString(&json, "],\"max\": [");
	for (d = 0; d < dim; d++)
	{
		if (d > 0)
			appendStringInfoChar(&json, ',');
		appendStringInfo(&json, "%.6f", max_vals[d]);
	}
	appendStringInfoString(&json, "]");

	/* Compute correlation matrix (sample first 10 dimensions for performance) */
	if (dim > 1 && nvec > 1)
	{
		int			max_corr_dim = (dim > 10) ? 10 : dim;

		appendStringInfoString(&json, ",\"correlation\": [");
		for (i = 0; i < max_corr_dim; i++)
		{
			if (i > 0)
				appendStringInfoChar(&json, ',');
			appendStringInfoChar(&json, '[');
			for (j = 0; j < max_corr_dim; j++)
			{
				if (j > 0)
					appendStringInfoChar(&json, ',');
				if (i == j)
				{
					appendStringInfoString(&json, "1.0");
				}
				else
				{
					double		cov = 0.0;
					int			k;

					for (k = 0; k < nvec; k++)
						cov += (data[k][i] - mean[i]) * (data[k][j] - mean[j]);
					cov /= nvec;
					if (stddev[i] > 0 && stddev[j] > 0)
					{
						double		corr = cov / (stddev[i] * stddev[j]);

						appendStringInfo(&json, "%.6f", corr);
					}
					else
					{
						appendStringInfoString(&json, "0.0");
					}
				}
			}
			appendStringInfoChar(&json, ']');
		}
		appendStringInfoChar(&json, ']');
	}
	appendStringInfoChar(&json, '}');

	/* Cleanup */
	for (i = 0; i < nvec; i++)
		nfree(data[i]);
	nfree(data);
	nfree(mean);
	nfree(variance);
	nfree(stddev);
	nfree(min_vals);
	nfree(max_vals);
	nfree(tbl_str);
	nfree(col_str);

	MemoryContextSwitchTo(oldctx);

	/* Convert JSON string to JSONB using safe wrapper */
	{
		Jsonb *jsonb_result = NULL;

		PG_TRY();
		{
			jsonb_result = ndb_jsonb_in_cstring(json.data);
			if (jsonb_result == NULL)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("vector_statistics: failed to parse JSON")));
			}
		}
		PG_CATCH();
		{
			pfree(json.data);
			PG_RE_THROW();
		}
		PG_END_TRY();

		pfree(json.data);
		PG_RETURN_JSONB_P(jsonb_result);
	}
}

/*
 * index_quality_metrics - Compute quality metrics for an index
 * Returns JSONB with recall, precision, F1, index health
 */
PG_FUNCTION_INFO_V1(index_quality_metrics);

Datum
index_quality_metrics(PG_FUNCTION_ARGS)
{
	text	   *index_name = NULL;
	char	   *idx_str = NULL;
	StringInfoData json;
	int			ret;
	NdbSpiSession *spi_session = NULL;
	MemoryContext oldctx;
	int64		index_size = 0;
	int64		vector_count = 0;
	float8		avg_recall = 0.0;
	float8		avg_precision = 0.0;
	float8		f1_score = 0.0;
	char		health_status[32] = "unknown";

	if (PG_NARGS() < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("index_quality_metrics requires 1 argument: index_name")));

	index_name = PG_GETARG_TEXT_PP(0);
	idx_str = text_to_cstring(index_name);

	oldctx = CurrentMemoryContext;
	NDB_SPI_SESSION_BEGIN(spi_session, oldctx);

	/* Query index statistics from pg_class and pg_stat_user_indexes */
	{
		StringInfoData sql;

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT pg_relation_size(i.oid) as size, "
						 "COALESCE(s.idx_scan, 0) as scans, "
						 "COALESCE(s.idx_tup_read, 0) as tuples_read "
						 "FROM pg_class c "
						 "JOIN pg_index idx ON idx.indexrelid = c.oid "
						 "JOIN pg_class i ON i.oid = idx.indexrelid "
						 "LEFT JOIN pg_stat_user_indexes s ON s.indexrelid = i.oid "
						 "WHERE c.relname = $1");
		Oid			argtypes[1] = {TEXTOID};
		Datum		values[1] = {PointerGetDatum(index_name)};
		const char nulls[1] = {' '};

		ret = ndb_spi_execute_with_args(spi_session, sql.data, 1, argtypes, values, nulls, true, 0);
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int64		scans = 0;
			int64		tuples_read = 0;
			bool		isnull;
			Datum		datum;

			/* Get index size (int8/bigint) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
			if (!isnull)
				index_size = DatumGetInt64(datum);

			/* Get scans (int8/bigint) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &isnull);
			if (!isnull)
				scans = DatumGetInt64(datum);

			/* Get tuples_read (int8/bigint) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3, &isnull);
			if (!isnull)
				tuples_read = DatumGetInt64(datum);

			if (scans > 0)
				vector_count = tuples_read / scans;
		}
		pfree(sql.data);
	}

	/* Query recall/precision from neurondb.query_metrics if available */
	{
		StringInfoData sql;

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT AVG(recall) as avg_recall, AVG(precision) as avg_precision "
						 "FROM neurondb.query_metrics "
						 "WHERE index_name = $1 AND created_at > NOW() - INTERVAL '24 hours'");
		Oid			argtypes[1] = {TEXTOID};
		Datum		values[1] = {PointerGetDatum(index_name)};
		const char nulls[1] = {' '};

		ret = ndb_spi_execute_with_args(spi_session, sql.data, 1, argtypes, values, nulls, true, 0);
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			float8		recall_val = 0.0;
			float8		precision_val = 0.0;
			bool		isnull;
			Datum		datum;

			/* Get avg_recall (float8) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
			if (!isnull)
				recall_val = DatumGetFloat8(datum);
			avg_recall = recall_val;

			/* Get avg_precision (float8) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &isnull);
			if (!isnull)
				precision_val = DatumGetFloat8(datum);
			avg_precision = precision_val;

			if (avg_recall > 0 && avg_precision > 0)
				f1_score = 2.0 * (avg_recall * avg_precision) / (avg_recall + avg_precision);
		}
		pfree(sql.data);
	}

	NDB_SPI_SESSION_END(spi_session);

	/* Determine health status */
	if (avg_recall >= 0.9 && avg_precision >= 0.9)
		strlcpy(health_status, "excellent", sizeof(health_status));
	else if (avg_recall >= 0.8 && avg_precision >= 0.8)
		strlcpy(health_status, "good", sizeof(health_status));
	else if (avg_recall >= 0.7 && avg_precision >= 0.7)
		strlcpy(health_status, "fair", sizeof(health_status));
	else if (avg_recall > 0 || avg_precision > 0)
		strlcpy(health_status, "poor", sizeof(health_status));
	else
		strlcpy(health_status, "unknown", sizeof(health_status));

	/* Build JSONB result */
	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"index_name\": \"%s\", "
					 "\"index_size_bytes\": %lld, "
					 "\"vector_count\": %lld, "
					 "\"avg_recall\": %.4f, "
					 "\"avg_precision\": %.4f, "
					 "\"f1_score\": %.4f, "
					 "\"health_status\": \"%s\"}",
					 idx_str,
					 (long long) index_size,
					 (long long) vector_count,
					 avg_recall,
					 avg_precision,
					 f1_score,
					 health_status);

	nfree(idx_str);
	MemoryContextSwitchTo(oldctx);

	/* Convert JSON string to JSONB using safe wrapper */
	{
		Jsonb *jsonb_result = NULL;

		PG_TRY();
		{
			jsonb_result = ndb_jsonb_in_cstring(json.data);
			if (jsonb_result == NULL)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("vector_statistics: failed to parse JSON")));
			}
		}
		PG_CATCH();
		{
			pfree(json.data);
			PG_RE_THROW();
		}
		PG_END_TRY();

		pfree(json.data);
		PG_RETURN_JSONB_P(jsonb_result);
	}
}

/*
 * query_performance_analytics - Analyze query performance metrics
 * Returns JSONB with latency statistics, throughput, GPU utilization
 */
PG_FUNCTION_INFO_V1(query_performance_analytics);

Datum
query_performance_analytics(PG_FUNCTION_ARGS)
{
	text	   *index_name = NULL;
	char	   *idx_str = NULL;
	StringInfoData json;
	int			ret;
	NdbSpiSession *spi_session = NULL;
	MemoryContext oldctx;
	float8		avg_latency_ms = 0.0;
	float8		p50_latency_ms = 0.0;
	float8		p95_latency_ms = 0.0;
	float8		p99_latency_ms = 0.0;
	int64		total_queries = 0;
	int64		gpu_queries = 0;
	float8		gpu_utilization = 0.0;

	/* query_performance_analytics takes no arguments - it analyzes all queries */

	index_name = PG_GETARG_TEXT_PP(0);
	idx_str = text_to_cstring(index_name);

	oldctx = CurrentMemoryContext;
	NDB_SPI_SESSION_BEGIN(spi_session, oldctx);

	/* Query performance metrics from neurondb.query_metrics */
	{
		StringInfoData sql;

		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT "
						 "COUNT(*) as total_queries, "
						 "AVG(latency_ms) as avg_latency, "
						 "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50, "
						 "PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95, "
						 "PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99, "
						 "SUM(CASE WHEN used_gpu THEN 1 ELSE 0 END) as gpu_queries "
						 "FROM neurondb.query_metrics "
						 "WHERE index_name = $1 AND created_at > NOW() - INTERVAL '24 hours'");
		Oid			argtypes[1] = {TEXTOID};
		Datum		values[1] = {PointerGetDatum(index_name)};
		const char nulls[1] = {' '};

		ret = ndb_spi_execute_with_args(spi_session, sql.data, 1, argtypes, values, nulls, true, 0);
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			bool		isnull;
			Datum		datum;

			/* Get total_queries (int8/bigint) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
			if (!isnull)
				total_queries = DatumGetInt64(datum);

			/* Get avg_latency_ms (float8) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &isnull);
			if (!isnull)
				avg_latency_ms = DatumGetFloat8(datum);

			/* Get p50_latency_ms (float8) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3, &isnull);
			if (!isnull)
				p50_latency_ms = DatumGetFloat8(datum);

			/* Get p95_latency_ms (float8) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4, &isnull);
			if (!isnull)
				p95_latency_ms = DatumGetFloat8(datum);

			/* Get p99_latency_ms (float8) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 5, &isnull);
			if (!isnull)
				p99_latency_ms = DatumGetFloat8(datum);

			/* Get gpu_queries (int8/bigint) */
			datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 6, &isnull);
			if (!isnull)
				gpu_queries = DatumGetInt64(datum);

			if (total_queries > 0)
				gpu_utilization = ((float8) gpu_queries / (float8) total_queries) * 100.0;
		}
		pfree(sql.data);
	}

	NDB_SPI_SESSION_END(spi_session);

	/* Build JSONB result */
	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"total_queries\": %lld, "
					 "\"avg_latency_ms\": %.2f, "
					 "\"p50_latency_ms\": %.2f, "
					 "\"p95_latency_ms\": %.2f, "
					 "\"p99_latency_ms\": %.2f, "
					 "\"gpu_queries\": %lld, "
					 "\"gpu_utilization_percent\": %.2f}",
					 (long long) total_queries,
					 avg_latency_ms,
					 p50_latency_ms,
					 p95_latency_ms,
					 p99_latency_ms,
					 (long long) gpu_queries,
					 gpu_utilization);
	MemoryContextSwitchTo(oldctx);

	/* Convert JSON string to JSONB using safe wrapper */
	{
		Jsonb *jsonb_result = NULL;

		PG_TRY();
		{
			jsonb_result = ndb_jsonb_in_cstring(json.data);
			if (jsonb_result == NULL)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("vector_statistics: failed to parse JSON")));
			}
		}
		PG_CATCH();
		{
			pfree(json.data);
			PG_RE_THROW();
		}
		PG_END_TRY();

		pfree(json.data);
		PG_RETURN_JSONB_P(jsonb_result);
	}
}
