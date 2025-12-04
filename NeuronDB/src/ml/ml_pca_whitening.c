/*-------------------------------------------------------------------------
 *
 * ml_pca_whitening.c
 *    PCA whitening for embedding normalization.
 *
 * This module implements PCA whitening to transform data to have identity
 * covariance matrix, removing correlations and normalizing variance.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_pca_whitening.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#include <math.h>
#include <string.h>
#include <float.h>

#define WHITENING_EPSILON 1e-5	/* Regularization to avoid division by zero */

/*
 * Power iteration for computing dominant eigenvalue/eigenvector
 * (same as PCA implementation in analytics.c)
 */
static double
power_iteration_whitening(double **matrix,
						  int dim,
						  double *eigenvector,
						  int max_iter)
{
	double	   *v_new;
	double		eigenvalue;
	int			iter,
				i,
				j;

	v_new = (double *) palloc(sizeof(double) * dim);

	/* Initialize with random vector */
	for (i = 0; i < dim; i++)
		eigenvector[i] = ((double) rand() / RAND_MAX) - 0.5;

	/* Normalize */
	{
		double		norm = 0.0;

		for (i = 0; i < dim; i++)
			norm += eigenvector[i] * eigenvector[i];
		norm = sqrt(norm);
		for (i = 0; i < dim; i++)
			eigenvector[i] /= norm;
	}
/* GPU registry not needed here */

	/* Power iteration */
	for (iter = 0; iter < max_iter; iter++)
	{
		double		norm;

		/* v_new = matrix * eigenvector */
		for (i = 0; i < dim; i++)
		{
			v_new[i] = 0.0;
			for (j = 0; j < dim; j++)
				v_new[i] += matrix[i][j] * eigenvector[j];
		}

		/* Normalize */
		norm = 0.0;
		for (i = 0; i < dim; i++)
			norm += v_new[i] * v_new[i];
		norm = sqrt(norm);

		if (norm < 1e-12)
			break;

		for (i = 0; i < dim; i++)
			eigenvector[i] = v_new[i] / norm;
	}

	/* Compute eigenvalue: λ = vᵀAv */
	eigenvalue = 0.0;
	for (i = 0; i < dim; i++)
	{
		double		sum = 0.0;

		for (j = 0; j < dim; j++)
			sum += matrix[i][j] * eigenvector[j];
		eigenvalue += eigenvector[i] * sum;
	}

	NDB_FREE(v_new);
	return eigenvalue;
}

/*
 * Matrix deflation: remove contribution of eigenvector
 */
static void
deflate_matrix(double **matrix,
			   int dim,
			   const double *eigenvector,
			   double eigenvalue)
{
	int			i,
				j;

	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			matrix[i][j] -=
				eigenvalue * eigenvector[i] * eigenvector[j];
}

/*
 * whiten_embeddings
 * -----------------
 * Apply PCA whitening to normalize embeddings.
 *
 * SQL Arguments:
 *   table_name    - Source table with embeddings
 *   vector_column - Vector column name
 *   epsilon       - Regularization parameter (default: 1e-5)
 *                   Higher values = more regularization = less aggressive whitening
 *
 * Returns:
 *   2D array of whitened vectors [nvec][dim]
 *
 * Properties of Output:
 *   - Zero mean: E[X_white] = 0
 *   - Unit variance: Var(X_white) = 1 per dimension
 *   - Uncorrelated: Cov(X_white) ≈ I
 *
 * Example Usage:
 *   -- Whiten embeddings for better clustering:
 *   CREATE TEMP TABLE whitened_docs AS
 *   SELECT row_number() OVER () AS id,
 *          unnest(whiten_embeddings('documents', 'embedding')) AS whitened_vec
 *   FROM generate_series(1, (SELECT COUNT(*) FROM documents));
 *
 *   -- Then cluster on whitened embeddings:
 *   SELECT cluster_kmeans('whitened_docs', 'whitened_vec', 10, 100);
 *
 * Notes:
 *   - Dimensionality preserved (same as input)
 *   - Computationally expensive: O(n*d² + d³) for eigen decomposition
 *   - Best for d < 500; for higher dimensions, use approximate methods
 *   - epsilon prevents numerical instability with near-zero eigenvalues
 */
PG_FUNCTION_INFO_V1(whiten_embeddings);
PGDLLEXPORT Datum whiten_embeddings(PG_FUNCTION_ARGS);

Datum
whiten_embeddings(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	double		epsilon;
	char	   *tbl_str;
	char	   *vec_col_str;
	float	  **vectors;
	int			nvec,
				dim;
	double	   *mean;
	double	  **covariance;
	double	  **eigenvectors;
	double	   *eigenvalues;
	double	  **whitening_matrix;
	float	  **whitened_vectors;
	int			i,
				j,
				k,
				c;
	ArrayType  *result;
	Datum	   *result_datums;
	int			dims[2];
	int			lbs[2];

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	epsilon = PG_ARGISNULL(2) ? WHITENING_EPSILON : PG_GETARG_FLOAT8(2);

	if (epsilon < 0.0 || epsilon > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("epsilon must be in [0, 1]")));

	tbl_str = text_to_cstring(table_name);
	vec_col_str = text_to_cstring(vector_column);

	elog(DEBUG1,
		 "neurondb: PCA whitening on %s.%s (epsilon=%.2e)",
		 tbl_str,
		 vec_col_str,
		 epsilon);

	/* Fetch vectors */
	vectors = neurondb_fetch_vectors_from_table(
												tbl_str, vec_col_str, &nvec, &dim);
	if (vectors == NULL || nvec == 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(vec_col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(vec_col_str);
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

	if (nvec < dim)
	{
		elog(DEBUG1,
			 "Need at least d=%d vectors for %d-dimensional whitening",
			 dim,
			 dim);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Need at least d=%d vectors for %d-dimensional whitening",
						dim,
						dim)));
	}

	/* Step 1: Center data (compute mean and subtract) */
	mean = (double *) palloc0(sizeof(double) * dim);
	for (i = 0; i < nvec; i++)
		for (j = 0; j < dim; j++)
			mean[j] += (double) vectors[i][j];

	for (j = 0; j < dim; j++)
		mean[j] /= nvec;

	/* Subtract mean from all vectors (center data) */
	for (i = 0; i < nvec; i++)
		for (j = 0; j < dim; j++)
			vectors[i][j] -= (float) mean[j];

	/* Step 2: Compute covariance matrix Σ = XᵀX / n */
	covariance = (double **) palloc(sizeof(double *) * dim);
	for (i = 0; i < dim; i++)
		covariance[i] = (double *) palloc0(sizeof(double) * dim);

	for (i = 0; i < dim; i++)
	{
		for (j = i; j < dim; j++)
		{
			double		sum = 0.0;

			for (k = 0; k < nvec; k++)
				sum += (double) vectors[k][i]
					* (double) vectors[k][j];

			covariance[i][j] = sum / nvec;
			covariance[j][i] = covariance[i][j];	/* Symmetric */
		}
	}

	/* Step 3: Eigen decomposition using power iteration */
	eigenvectors = (double **) palloc(sizeof(double *) * dim);
	eigenvalues = (double *) palloc(sizeof(double) * dim);

	for (c = 0; c < dim; c++)
		eigenvectors[c] = (double *) palloc(sizeof(double) * dim);

	/* Compute all eigenvectors/eigenvalues */
	for (c = 0; c < dim; c++)
	{
		eigenvalues[c] = power_iteration_whitening(
												   covariance, dim, eigenvectors[c], 100);

		if (eigenvalues[c] < 0.0)
			eigenvalues[c] = 0.0;	/* Numerical stability */

		/* Deflate matrix for next eigenvector */
		if (c < dim - 1)
			deflate_matrix(covariance,
						   dim,
						   eigenvectors[c],
						   eigenvalues[c]);

		elog(DEBUG1,
			 "neurondb: Eigenvalue %d = %.6f",
			 c + 1,
			 eigenvalues[c]);
	}

	/* Step 4: Compute whitening matrix W = UΛ^(-1/2)Uᵀ */
	whitening_matrix = (double **) palloc(sizeof(double *) * dim);
	for (i = 0; i < dim; i++)
		whitening_matrix[i] = (double *) palloc0(sizeof(double) * dim);

	/*
	 * W = Σ_c eigenvector_c * eigenvector_cᵀ / sqrt(eigenvalue_c +
	 * epsilon)
	 */
	for (c = 0; c < dim; c++)
	{
		double		scale = 1.0 / sqrt(eigenvalues[c] + epsilon);

		for (i = 0; i < dim; i++)
			for (j = 0; j < dim; j++)
				whitening_matrix[i][j] += eigenvectors[c][i]
					* eigenvectors[c][j] * scale;
	}

	/* Step 5: Apply whitening transform */
	whitened_vectors = (float **) palloc(sizeof(float *) * nvec);
	for (i = 0; i < nvec; i++)
	{
		whitened_vectors[i] = (float *) palloc(sizeof(float) * dim);

		for (j = 0; j < dim; j++)
		{
			double		sum = 0.0;

			for (k = 0; k < dim; k++)
				sum += (double) vectors[i][k]
					* whitening_matrix[k][j];

			whitened_vectors[i][j] = (float) sum;
		}
	}

	/* Build 2D result array */
	result_datums = (Datum *) palloc(sizeof(Datum) * nvec * dim);
	for (i = 0; i < nvec; i++)
		for (j = 0; j < dim; j++)
			result_datums[i * dim + j] =
				Float4GetDatum(whitened_vectors[i][j]);

	dims[0] = nvec;
	dims[1] = dim;
	lbs[0] = 1;
	lbs[1] = 1;

	result = construct_md_array(result_datums,
								NULL,
								2,
								dims,
								lbs,
								FLOAT4OID,
								sizeof(float4),
								true,
								'i');

	for (i = 0; i < nvec; i++)
	{
		NDB_FREE(vectors[i]);
		NDB_FREE(whitened_vectors[i]);
	}
	NDB_FREE(vectors);
	NDB_FREE(whitened_vectors);
	NDB_FREE(mean);

	for (i = 0; i < dim; i++)
	{
		NDB_FREE(covariance[i]);
		NDB_FREE(eigenvectors[i]);
		NDB_FREE(whitening_matrix[i]);
	}
	NDB_FREE(covariance);
	NDB_FREE(eigenvectors);
	NDB_FREE(eigenvalues);
	NDB_FREE(whitening_matrix);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(vec_col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}

/* Optional: GPU registration noop to satisfy linkage in builds */
#include "neurondb_gpu_model.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
/* Forward declaration to avoid missing-prototypes warning */
void		neurondb_gpu_register_pca_whitening_model(void);
void
neurondb_gpu_register_pca_whitening_model(void)
{
}
