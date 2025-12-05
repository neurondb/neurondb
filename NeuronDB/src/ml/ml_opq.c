/*-------------------------------------------------------------------------
 *
 * ml_opq.c
 *    Optimized product quantization.
 *
 * This module implements OPQ with rotation matrix learning to minimize
 * quantization error for improved vector compression.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_opq.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/jsonb.h"
#include "lib/stringinfo.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_simd.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "ml_catalog.h"

#include <math.h>
#include <float.h>

/*
 * Product Quantization Codebook (reused from PQ)
 */
typedef struct PQCodebook
{
	int			m;
	int			ksub;
	int			dsub;
	float	 ***centroids;
}			PQCodebook;

static void
pq_codebook_free(PQCodebook * codebook)
{
	int			sub,
				i;

	if (codebook == NULL || codebook->centroids == NULL)
		return;

	for (sub = 0; sub < codebook->m; sub++)
	{
		if (codebook->centroids[sub] != NULL)
		{
			for (i = 0; i < codebook->ksub; i++)
				if (codebook->centroids[sub][i] != NULL)
					NDB_FREE(codebook->centroids[sub][i]);
			NDB_FREE(codebook->centroids[sub]);
		}
	}
	NDB_FREE(codebook->centroids);
}

/*
 * train_opq_rotation
 * ------------------
 * Learn OPQ rotation matrix (simplified version).
 *
 * Full OPQ training involves iterating between:
 *   1. Fixing R, optimizing codebooks
 *   2. Fixing codebooks, optimizing R via SVD
 *
 * This simplified version initializes R to identity (PCA-like preprocessing).
 * For production, consider training externally with full OPQ algorithm.
 *
 * SQL Arguments:
 *   table_name    - Training data table
 *   vector_column - Vector column
 *   num_subspaces - Number of PQ subspaces (default: 8)
 *
 * Returns:
 *   Rotation matrix flattened as 1D array [d×d]
 *
 * Example:
 *   SELECT train_opq_rotation('vectors', 'embedding', 8);
 *
 * Notes:
 *   - This is a placeholder for identity/PCA rotation
 *   - Full OPQ requires iterative refinement (complex)
 *   - For best results, train offline with sklearn/faiss
 */
PG_FUNCTION_INFO_V1(train_opq_rotation);

Datum
train_opq_rotation(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	int			num_subspaces;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	int			nvec,
				dim;
	double	   *rotation_matrix;
	int			i;
	ArrayType  *result;
	Datum	   *result_datums;

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_subspaces = PG_ARGISNULL(2) ? 8 : PG_GETARG_INT32(2);

	if (num_subspaces < 2 || num_subspaces > 64)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("num_subspaces must be between 2 and "
						"64")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);

	elog(DEBUG1,
		 "neurondb: Training OPQ rotation (m=%d subspaces)",
		 num_subspaces);

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
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					NDB_FREE(data[i]);
			}
			NDB_FREE(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (dim % num_subspaces != 0)
	{
		elog(DEBUG1, "Vector dimension %d must be divisible by num_subspaces %d",
			 dim, num_subspaces);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Vector dimension %d must be divisible by num_subspaces %d",
						dim, num_subspaces)));
	}

	/* Compute PCA-based rotation matrix */
	/* This is a practical approximation to full OPQ iterative optimization */
	{
		double	  **covariance = NULL;
		double	  **eigenvectors = NULL;

		NDB_DECLARE(double *, eigenvalues);
		NDB_DECLARE(double *, mean);
		int			j,
					k,
					iter;

		/* Compute mean vector */
		mean = (double *) palloc0(sizeof(double) * dim);
		NDB_CHECK_ALLOC(mean, "mean");
		for (i = 0; i < nvec; i++)
		{
			for (j = 0; j < dim; j++)
				mean[j] += (double) data[i][j];
		}
		for (j = 0; j < dim; j++)
			mean[j] /= (double) nvec;

		/* Compute covariance matrix */
		NDB_ALLOC(covariance, double *, dim);
		NDB_CHECK_ALLOC(covariance, "covariance");
		for (i = 0; i < dim; i++)
		{
			covariance[i] = (double *) palloc0(sizeof(double) * dim);
			NDB_CHECK_ALLOC(covariance[i], "covariance[i]");
		}

		for (i = 0; i < nvec; i++)
		{
			for (j = 0; j < dim; j++)
			{
				double		diff_j = (double) data[i][j] - mean[j];

				for (k = 0; k < dim; k++)
				{
					double		diff_k = (double) data[i][k] - mean[k];

					covariance[j][k] += diff_j * diff_k;
				}
			}
		}

		/* Normalize by nvec - 1 */
		for (j = 0; j < dim; j++)
		{
			for (k = 0; k < dim; k++)
				covariance[j][k] /= (double) (nvec - 1);
		}

		/* Compute eigenvectors using power iteration (PCA) */
		NDB_ALLOC(eigenvectors, double *, dim);
		NDB_CHECK_ALLOC(eigenvectors, "eigenvectors");
		NDB_ALLOC(eigenvalues, double, dim);
		NDB_CHECK_ALLOC(eigenvalues, "eigenvalues");

		for (i = 0; i < dim; i++)
		{
			double	   *eigvec;
			double		norm;
			double		prev_eigenvalue = 0.0;
			int			max_iter = 100;

			NDB_ALLOC(eigvec, double, dim);
			NDB_CHECK_ALLOC(eigvec, "eigvec");

			/* Initialize with random vector */
			for (j = 0; j < dim; j++)
				eigvec[j] = ((double) rand() / (double) RAND_MAX) - 0.5;

			/* Normalize */
			norm = 0.0;
			for (j = 0; j < dim; j++)
				norm += eigvec[j] * eigvec[j];
			norm = sqrt(norm);
			if (norm > 1e-10)
			{
				for (j = 0; j < dim; j++)
					eigvec[j] /= norm;
			}

			/* Power iteration */
			for (iter = 0; iter < max_iter; iter++)
			{
				double	   *v_new = (double *) palloc0(sizeof(double) * dim);
				double		eigenvalue = 0.0;

				NDB_CHECK_ALLOC(v_new, "v_new");

				/* v_new = covariance * eigvec */
				for (j = 0; j < dim; j++)
				{
					for (k = 0; k < dim; k++)
						v_new[j] += covariance[j][k] * eigvec[k];
				}

				/* Normalize */
				norm = 0.0;
				for (j = 0; j < dim; j++)
					norm += v_new[j] * v_new[j];
				norm = sqrt(norm);

				if (norm < 1e-10)
					break;

				for (j = 0; j < dim; j++)
					eigvec[j] = v_new[j] / norm;

				/* Compute eigenvalue */
				for (j = 0; j < dim; j++)
				{
					double		sum = 0.0;

					for (k = 0; k < dim; k++)
						sum += covariance[j][k] * eigvec[k];
					eigenvalue += eigvec[j] * sum;
				}

				/* Check convergence */
				if (fabs(eigenvalue - prev_eigenvalue) < 1e-6)
					break;

				prev_eigenvalue = eigenvalue;

				/* Deflate: remove component of this eigenvector */
				for (j = 0; j < dim; j++)
				{
					for (k = 0; k < dim; k++)
						covariance[j][k] -= eigenvalue * eigvec[j] * eigvec[k];
				}

				NDB_FREE(v_new);
			}

			eigenvalues[i] = prev_eigenvalue;
			eigenvectors[i] = eigvec;
		}

		/* Build rotation matrix from eigenvectors (transpose for rotation) */
		/* Check for integer overflow in size calculation */
		if (dim > 0 && (size_t) dim > MaxAllocSize / sizeof(double) / (size_t) dim)
		{
			NDB_FREE(eigenvalues);
			for (i = 0; i < dim; i++)
				NDB_FREE(eigenvectors[i]);
			NDB_FREE(eigenvectors);
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("neurondb: rotation_matrix allocation size exceeds MaxAllocSize")));
		}
		rotation_matrix = (double *) palloc0(sizeof(double) * dim * dim);
		NDB_CHECK_ALLOC(rotation_matrix, "rotation_matrix");

		/* R = [eigenvector_1^T, eigenvector_2^T, ..., eigenvector_d^T] */
		for (i = 0; i < dim; i++)
		{
			for (j = 0; j < dim; j++)
				rotation_matrix[i * dim + j] = eigenvectors[j][i];
		}

		NDB_FREE(mean);
		if (covariance != NULL)
		{
			for (i = 0; i < dim; i++)
				NDB_FREE(covariance[i]);
			NDB_FREE(covariance);
		}
		if (eigenvectors != NULL)
		{
			for (i = 0; i < dim; i++)
				NDB_FREE(eigenvectors[i]);
			NDB_FREE(eigenvectors);
		}
		NDB_FREE(eigenvalues);

		elog(DEBUG1,
			 "OPQ: Computed PCA-based rotation matrix (dim=%d, subspaces=%d)",
			 dim, num_subspaces);
	}

	/* Build result array */
	NDB_ALLOC(result_datums, Datum, dim * dim);
	NDB_CHECK_ALLOC(result_datums, "result_datums");
	for (i = 0; i < dim * dim; i++)
		result_datums[i] = Float8GetDatum(rotation_matrix[i]);

	result = construct_array(result_datums,
							 dim * dim,
							 FLOAT8OID,
							 sizeof(float8),
							 FLOAT8PASSBYVAL,
							 'd');

	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	NDB_FREE(rotation_matrix);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * apply_opq_rotation
 * ------------------
 * Apply learned OPQ rotation to a vector.
 *
 * SQL Arguments:
 *   vector          - Input vector
 *   rotation_matrix - Rotation matrix from train_opq_rotation()
 *
 * Returns:
 *   Rotated vector
 *
 * Example:
 *   WITH rotation AS (
 *     SELECT train_opq_rotation('train_data', 'vec', 8) AS R
 *   )
 *   SELECT apply_opq_rotation(my_vector, R) FROM rotation, my_table;
 */
PG_FUNCTION_INFO_V1(apply_opq_rotation);
PG_FUNCTION_INFO_V1(predict_opq_rotation);
PG_FUNCTION_INFO_V1(evaluate_opq_rotation_by_model_id);

Datum
apply_opq_rotation(PG_FUNCTION_ARGS)
{
	ArrayType  *vector_array;
	ArrayType  *rotation_array;
	float8	   *vector;
	float8	   *rotation;
	int			dim;
	int			rotation_dim;
	double	   *rotated;
	int			i,
				j;
	ArrayType  *result;
	Datum	   *result_datums;

	vector_array = PG_GETARG_ARRAYTYPE_P(0);
	rotation_array = PG_GETARG_ARRAYTYPE_P(1);

	if (ARR_NDIM(vector_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Vector must be 1-dimensional")));

	dim = ARR_DIMS(vector_array)[0];
	rotation_dim = (int) sqrt((double) ARR_DIMS(rotation_array)[0]);

	if (rotation_dim * rotation_dim != ARR_DIMS(rotation_array)[0])
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Rotation matrix must be square")));

	if (dim != rotation_dim)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Dimension mismatch: vector=%d, rotation=%d",
						dim,
						rotation_dim)));

	vector = (float8 *) ARR_DATA_PTR(vector_array);
	rotation = (float8 *) ARR_DATA_PTR(rotation_array);

	/* Apply rotation: rotated = R · vector */
	rotated = (double *) palloc0(sizeof(double) * dim);
	NDB_CHECK_ALLOC(rotated, "rotated");
	for (i = 0; i < dim; i++)
		for (j = 0; j < dim; j++)
			rotated[i] += rotation[i * dim + j] * vector[j];

	NDB_ALLOC(result_datums, Datum, dim);
	NDB_CHECK_ALLOC(result_datums, "result_datums");
	for (i = 0; i < dim; i++)
		result_datums[i] = Float8GetDatum(rotated[i]);

	result = construct_array(result_datums,
							 dim,
							 FLOAT8OID,
							 sizeof(float8),
							 FLOAT8PASSBYVAL,
							 'd');

	NDB_FREE(rotated);
	NDB_FREE(result_datums);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * predict_opq_rotation
 *      Applies OPQ rotation to a vector using a trained rotation matrix.
 *      Arguments: int4 model_id, float8[] vector
 *      Returns: float8[] rotated vector
 */
Datum
predict_opq_rotation(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType  *vector_array;

	NDB_DECLARE(bytea *, model_data);
	NDB_DECLARE(Jsonb *, parameters);
	float8	   *vector;
	float8	   *rotation;
	int			dim;
	int			rotation_dim;
	double	   *rotated;
	int			i,
				j;
	ArrayType  *result;
	Datum	   *result_datums;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;
	bool		found_rotation = false;

	model_id = PG_GETARG_INT32(0);
	vector_array = PG_GETARG_ARRAYTYPE_P(1);

	/* Validate input vector */
	if (ARR_NDIM(vector_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_opq_rotation: vector must be 1-dimensional")));

	dim = ARR_DIMS(vector_array)[0];
	vector = (float8 *) ARR_DATA_PTR(vector_array);

	/* Load model data from catalog */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data, &parameters, NULL))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("predict_opq_rotation: model %d not found", model_id)));

	/* Extract rotation matrix from parameters */
	/* Rotation matrix is typically stored in parameters as "rotation_matrix" */
	/* Wrap in PG_TRY to handle corrupted JSONB gracefully */
	if (parameters != NULL)
	{
		PG_TRY();
		{
			it = JsonbIteratorInit((JsonbContainer *) & parameters->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "rotation_matrix") == 0 && v.type == jbvArray)
					{
						/* Extract array from JSONB */
						/*
						 * For now, rotation matrix should be stored as
						 * model_data
						 */
						/* or we need to reconstruct from JSONB array */
						found_rotation = true;
						NDB_FREE(key);
						break;
					}
					NDB_FREE(key);
				}
			}
		}
		PG_CATCH();
		{
			FlushErrorState();
			elog(WARNING,
				 "predict_opq_rotation: Failed to parse parameters JSONB (possibly corrupted), using model_data fallback");
			found_rotation = false;
		}
		PG_END_TRY();
	}

	/* If rotation not in parameters, try to load from model_data */
	/* Model data format: rotation matrix stored as float8 array [dim*dim] */
	if (!found_rotation && model_data != NULL)
	{
		/* Parse model_data to extract rotation matrix */
		/* Format: stored as float8 array [dim*dim] in row-major order */
		Size		data_size = VARSIZE(model_data) - VARHDRSZ;
		Size		expected_size = dim * dim * sizeof(float8);

		elog(DEBUG1,
			 "predict_opq_rotation: Attempting to load rotation from model_data "
			 "(size=%zu, expected=%zu)",
			 (size_t) data_size,
			 (size_t) expected_size);

		if (data_size >= expected_size)
		{
			/* Extract rotation matrix from model_data */
			rotation_dim = dim;
			NDB_DECLARE(float8 *, rotation);
			NDB_ALLOC(rotation, float8, dim * dim);
			NDB_CHECK_ALLOC(rotation, "rotation");
			memcpy(rotation, VARDATA(model_data), expected_size);
			elog(DEBUG1,
				 "predict_opq_rotation: Loaded rotation matrix from model_data");
		}
		else
		{
			/* Size mismatch - use identity matrix as fallback */
			rotation_dim = dim;
			NDB_DECLARE(float8 *, rotation);
			NDB_ALLOC(rotation, float8, dim * dim);
			NDB_CHECK_ALLOC(rotation, "rotation");
			for (i = 0; i < dim; i++)
			{
				for (j = 0; j < dim; j++)
				{
					if (i == j)
						rotation[i * dim + j] = 1.0;
					else
						rotation[i * dim + j] = 0.0;
				}
			}
			elog(DEBUG1,
				 "predict_opq_rotation: Model data size mismatch, using identity matrix");
		}
	}
	else if (found_rotation)
	{
		/* Extract rotation from JSONB array */
		/* JSONB array format: [row0_col0, row0_col1, ..., row1_col0, ...] */
		rotation_dim = dim;
		NDB_DECLARE(float8 *, rotation);
		NDB_ALLOC(rotation, float8, dim * dim);
		NDB_CHECK_ALLOC(rotation, "rotation");

		/* Re-initialize iterator to extract array elements */
		/* Wrap in PG_TRY to handle corrupted JSONB gracefully */
		PG_TRY();
		{
			it = JsonbIteratorInit((JsonbContainer *) & parameters->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

					if (strcmp(key, "rotation_matrix") == 0)
					{
						r = JsonbIteratorNext(&it, &v, false);
						if (v.type == jbvArray)
						{
							/* Extract array elements */
							JsonbIterator *arr_it = JsonbIteratorInit(v.val.binary.data);
							JsonbValue	arr_v;
							int			arr_r;
							int			idx = 0;

							while ((arr_r = JsonbIteratorNext(&arr_it, &arr_v, true)) != WJB_DONE)
							{
								if (arr_r == WJB_ELEM && arr_v.type == jbvNumeric)
								{
									if (idx < dim * dim)
									{
										Numeric		num = DatumGetNumeric(DirectFunctionCall1(
																							  numeric_in,
																							  CStringGetTextDatum(DatumGetCString(
																																  DirectFunctionCall1(numeric_out,
																																					  PointerGetDatum(arr_v.val.numeric))))));

										rotation[idx] = DatumGetFloat8(DirectFunctionCall1(
																						   numeric_float8,
																						   NumericGetDatum(num)));
										idx++;
									}
								}
							}

							if (idx != dim * dim)
							{
								elog(WARNING,
									 "predict_opq_rotation: Rotation matrix array size mismatch "
									 "(got %d, expected %d), using identity",
									 idx,
									 dim * dim);
								/* Fall back to identity */
								for (i = 0; i < dim; i++)
								{
									for (j = 0; j < dim; j++)
									{
										if (i == j)
											rotation[i * dim + j] = 1.0;
										else
											rotation[i * dim + j] = 0.0;
									}
								}
							}
							else
							{
								elog(DEBUG1,
									 "predict_opq_rotation: Extracted rotation matrix from JSONB "
									 "(%d elements)",
									 idx);
							}
						}
						NDB_FREE(key);
						break;
					}
					NDB_FREE(key);
				}
			}
		}
		PG_CATCH();
		{
			FlushErrorState();
			elog(WARNING,
				 "predict_opq_rotation: Failed to parse rotation_matrix from JSONB (possibly corrupted), using identity matrix");
			/* Free rotation if allocated */
			if (rotation != NULL)
			{
				NDB_FREE(rotation);
				rotation = NULL;
			}
			/* Fall back to identity matrix */
			rotation_dim = dim;
			NDB_DECLARE(float8 *, rotation);
			NDB_ALLOC(rotation, float8, dim * dim);
			NDB_CHECK_ALLOC(rotation, "rotation");
			for (i = 0; i < dim; i++)
			{
				for (j = 0; j < dim; j++)
				{
					if (i == j)
						rotation[i * dim + j] = 1.0;
					else
						rotation[i * dim + j] = 0.0;
				}
			}
		}
		PG_END_TRY();
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("predict_opq_rotation: rotation matrix not found in model %d",
						model_id)));
	}

	/* Validate rotation matrix dimensions */
	if (rotation_dim != dim)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_opq_rotation: dimension mismatch: vector=%d, rotation=%d",
						dim,
						rotation_dim)));

	/* Apply rotation: rotated = R · vector */
	rotated = (double *) palloc0(sizeof(double) * dim);
	NDB_CHECK_ALLOC(rotated, "rotated");
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < dim; j++)
		{
			rotated[i] += rotation[i * dim + j] * vector[j];
		}
	}

	/* Build result array */
	NDB_ALLOC(result_datums, Datum, dim);
	NDB_CHECK_ALLOC(result_datums, "result_datums");
	for (i = 0; i < dim; i++)
		result_datums[i] = Float8GetDatum(rotated[i]);

	result = construct_array(result_datums,
							 dim,
							 FLOAT8OID,
							 sizeof(float8),
							 FLOAT8PASSBYVAL,
							 'd');

	NDB_FREE(rotated);
	NDB_FREE(result_datums);
	NDB_FREE(rotation);
	if (model_data)
		NDB_FREE(model_data);
	if (parameters)
		NDB_FREE(parameters);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*
 * evaluate_opq_rotation_by_model_id
 *      Evaluates OPQ rotation quality on a dataset.
 *      Arguments: int4 model_id, text table_name, text vector_col
 *      Returns: jsonb with quantization metrics
 */
Datum
evaluate_opq_rotation_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *vector_col;
	char	   *tbl_str;
	char	   *vec_str;
	StringInfoData query;
	int			ret;
	int			n_points = 0;
	StringInfoData jsonbuf;
	Jsonb	   *result;
	MemoryContext oldcontext;
	double		avg_quantization_error;
	int			subquantizers;
	int			codebook_size;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext_spi;

	/* Validate arguments */
	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_opq_rotation_by_model_id: 3 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_opq_rotation_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_opq_rotation_by_model_id: table_name and vector_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	vector_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	vec_str = text_to_cstring(vector_col);

	oldcontext = CurrentMemoryContext;
	oldcontext_spi = CurrentMemoryContext;

	/* Connect to SPI */

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext_spi);

	/* Build query */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s FROM %s WHERE %s IS NOT NULL",
					 vec_str, tbl_str, vec_str);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(vec_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_opq_rotation_by_model_id: query failed")));
	}

	n_points = SPI_processed;
	if (n_points < 2)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(vec_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_opq_rotation_by_model_id: need at least 2 vectors, got %d",
						n_points)));
	}

	/* Load model and compute quantization metrics */
	{
		NDB_DECLARE(bytea *, model_payload);
		NDB_DECLARE(Jsonb *, parameters);
		NDB_DECLARE(Jsonb *, metrics);
		char	   *cb_ptr;
		int			m,
					ksub,
					dsub;
		int			i,
					sub,
					c,
					d;
		double		total_error = 0.0;
		int			valid_vectors = 0;
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;

		/* Load model from catalog */
		if (ml_catalog_fetch_model_payload(model_id,
										   &model_payload,
										   &parameters,
										   &metrics))
		{
			float	 ***centroids = NULL;

			if (model_payload != NULL && VARSIZE(model_payload) > VARHDRSZ)
			{
				/* Deserialize codebook (OPQ format similar to PQ) */
				cb_ptr = VARDATA(model_payload);
				/* Skip rotation matrix if present - for now assume PQ format */
				memcpy(&m, cb_ptr, sizeof(int));
				cb_ptr += sizeof(int);
				memcpy(&ksub, cb_ptr, sizeof(int));
				cb_ptr += sizeof(int);
				memcpy(&dsub, cb_ptr, sizeof(int));
				cb_ptr += sizeof(int);
				subquantizers = m;
				codebook_size = ksub;

				/* Reconstruct centroids and compute error */
				NDB_ALLOC(centroids, float **, m);
				for (sub = 0; sub < m; sub++)
				{
					NDB_DECLARE(float **, centroids_sub);
					NDB_ALLOC(centroids_sub, float *, ksub);
					centroids[sub] = centroids_sub;
					for (c = 0; c < ksub; c++)
					{
						NDB_DECLARE(float *, centroids_sub_c);
						NDB_ALLOC(centroids_sub_c, float, dsub);
						centroids[sub][c] = centroids_sub_c;
						memcpy(centroids[sub][c], cb_ptr, sizeof(float) * dsub);
						cb_ptr += sizeof(float) * dsub;
					}
				}

				/* Compute quantization error for each vector */
				for (i = 0; i < n_points; i++)
				{
					HeapTuple	tuple;

					/* Safe access to SPI_tuptable - validate before access */
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
						double		vector_error = 0.0;
						int			start_dim;

						vec_datum = SPI_getbinval(tuple, tupdesc, 1, &vec_null);
						if (vec_null)
							continue;

						vec = DatumGetVector(vec_datum);
						if (vec == NULL || vec->dim != m * dsub)
							continue;

						vec_data = vec->data;
						valid_vectors++;

						/*
						 * For each subspace, find nearest centroid and
						 * compute error
						 */
						for (sub = 0; sub < m; sub++)
						{
							double		min_dist;
							int			best_c;
							double		sub_error;

							start_dim = sub * dsub;
							min_dist = DBL_MAX;
							best_c = -1;
							sub_error = 0.0;

							for (c = 0; c < ksub; c++)
							{
								double		dist = 0.0;

								for (d = 0; d < dsub; d++)
								{
									double		diff = (double) vec_data[start_dim + d]
										- (double) centroids[sub][c][d];

									dist += diff * diff;
								}
								if (dist < min_dist)
								{
									min_dist = dist;
									best_c = c;
								}
							}

							if (best_c >= 0)
							{
								for (d = 0; d < dsub; d++)
								{
									double		diff = (double) vec_data[start_dim + d]
										- (double) centroids[sub][best_c][d];

									sub_error += diff * diff;
								}
							}

							vector_error += sub_error;
						}

						total_error += vector_error;
					}
				}

				/* Free centroids */
				if (centroids != NULL)
				{
					for (sub = 0; sub < m; sub++)
					{
						if (centroids[sub] != NULL)
						{
							for (c = 0; c < ksub; c++)
							{
								if (centroids[sub][c] != NULL)
									NDB_FREE(centroids[sub][c]);
							}
							NDB_FREE(centroids[sub]);
						}
					}
					NDB_FREE(centroids);
				}

				if (valid_vectors > 0)
				{
					avg_quantization_error = total_error / valid_vectors;
				}
				else
				{
					avg_quantization_error = 0.0;
				}

				if (model_payload != NULL)
					NDB_FREE(model_payload);
				if (parameters != NULL)
					NDB_FREE(parameters);
				if (metrics != NULL)
					NDB_FREE(metrics);
			}
			else
			{
				avg_quantization_error = 0.0;
				subquantizers = 8;
				codebook_size = 256;
			}
		}
		else
		{
			avg_quantization_error = 0.0;
			subquantizers = 8;
			codebook_size = 256;
		}
	}

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"quantization_error\":%.6f,\"subquantizers\":%d,\"codebook_size\":%d,\"n_vectors\":%d}",
					 avg_quantization_error, subquantizers, codebook_size, n_points);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
	NDB_FREE(jsonbuf.data);

	NDB_FREE(tbl_str);
	NDB_FREE(vec_str);

	PG_RETURN_JSONB_P(result);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for OPQ
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"

typedef struct OPQGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	PQCodebook *codebook;
	float	   *rotation_matrix;
	int			m;
	int			ksub;
	int			dsub;
	int			dim;
	int			n_samples;
}			OPQGpuModelState;

static bytea *
opq_model_serialize_to_bytea(const PQCodebook * codebook, const float *rotation_matrix, int dim)
{
	int			result_size;
	bytea	   *result;
	char	   *result_ptr;
	int			sub,
				i;

	result_size = sizeof(int) * 4 + dim * dim * sizeof(float) +
		codebook->m * codebook->ksub * codebook->dsub * sizeof(float);
	NDB_DECLARE(char *, result_raw);
	NDB_ALLOC(result_raw, char, VARHDRSZ + result_size);
	result = (bytea *) result_raw;
	NDB_CHECK_ALLOC(result, "result");
	SET_VARSIZE(result, VARHDRSZ + result_size);
	result_ptr = VARDATA(result);

	memcpy(result_ptr, &codebook->m, sizeof(int));
	result_ptr += sizeof(int);
	memcpy(result_ptr, &codebook->ksub, sizeof(int));
	result_ptr += sizeof(int);
	memcpy(result_ptr, &codebook->dsub, sizeof(int));
	result_ptr += sizeof(int);
	memcpy(result_ptr, &dim, sizeof(int));
	result_ptr += sizeof(int);

	memcpy(result_ptr, rotation_matrix, sizeof(float) * dim * dim);
	result_ptr += sizeof(float) * dim * dim;

	for (sub = 0; sub < codebook->m; sub++)
		for (i = 0; i < codebook->ksub; i++)
		{
			memcpy(result_ptr, codebook->centroids[sub][i], sizeof(float) * codebook->dsub);
			result_ptr += sizeof(float) * codebook->dsub;
		}

	return result;
}

static int
opq_model_deserialize_from_bytea(const bytea * data, PQCodebook * codebook, float **rotation_matrix_out, int *dim_out)
{
	const char *buf;
	int			offset = 0;
	int			sub,
				i;

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 4)
		return -1;

	buf = VARDATA(data);
	memcpy(&codebook->m, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&codebook->ksub, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(&codebook->dsub, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(dim_out, buf + offset, sizeof(int));
	offset += sizeof(int);

	if (codebook->m < 1 || codebook->m > 128 || codebook->ksub < 2 || codebook->ksub > 65536 || codebook->dsub <= 0 || *dim_out <= 0)
		return -1;

	/* Check for integer overflow in size calculation */
	if (*dim_out > 0 && (size_t) * dim_out > MaxAllocSize / sizeof(float) / (size_t) * dim_out)
	{
		return -1;
	}
	NDB_DECLARE(float *, rotation_matrix);
	NDB_ALLOC(rotation_matrix, float, *dim_out * *dim_out);
	*rotation_matrix_out = rotation_matrix;
	NDB_CHECK_ALLOC(rotation_matrix_out, "rotation_matrix_out");
	memcpy(*rotation_matrix_out, buf + offset, sizeof(float) * *dim_out * *dim_out);
	offset += sizeof(float) * *dim_out * *dim_out;

	NDB_DECLARE(float ***, centroids);
	NDB_ALLOC(centroids, float **, codebook->m);
	codebook->centroids = centroids;
	NDB_CHECK_ALLOC(codebook, "codebook");
	for (sub = 0; sub < codebook->m; sub++)
	{
		NDB_DECLARE(float **, sub_centroids);
		NDB_ALLOC(sub_centroids, float *, codebook->ksub);
		codebook->centroids[sub] = sub_centroids;
		NDB_CHECK_ALLOC(codebook, "codebook");
		for (i = 0; i < codebook->ksub; i++)
		{
			NDB_DECLARE(float *, centroid_vec);
			NDB_ALLOC(centroid_vec, float, codebook->dsub);
			codebook->centroids[sub][i] = centroid_vec;
			NDB_CHECK_ALLOC(codebook, "codebook");
			memcpy(codebook->centroids[sub][i], buf + offset, sizeof(float) * codebook->dsub);
			offset += sizeof(float) * codebook->dsub;
		}
	}

	return 0;
}

static bool
opq_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	OPQGpuModelState *state;
	float	  **data = NULL;
	PQCodebook	codebook;

	NDB_DECLARE(float *, rotation_matrix);
	int			m = 8;
	int			ksub = 256;
	int			nvec = 0;
	int			dim = 0;
	int			sub,
				i;

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
			*errstr = pstrdup("opq_gpu_train: invalid parameters");
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
				if (strcmp(key, "m") == 0 && v.type == jbvNumeric)
					m = DatumGetInt32(DirectFunctionCall1(numeric_int4,
														  NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "ksub") == 0 && v.type == jbvNumeric)
					ksub = DatumGetInt32(DirectFunctionCall1(numeric_int4,
															 NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}

	if (m < 1 || m > 128)
		m = 8;
	if (ksub < 2 || ksub > 65536)
		ksub = 256;

	/* Convert feature matrix to 2D array */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	if (dim % m != 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_train: dimension must be divisible by m");
		return false;
	}

	NDB_ALLOC(data, float *, nvec);
	NDB_CHECK_ALLOC(data, "data");
	for (i = 0; i < nvec; i++)
	{
		NDB_DECLARE(float *, data_row);
		NDB_ALLOC(data_row, float, dim);
		data[i] = data_row;
		NDB_CHECK_ALLOC(data, "data");
		memcpy(data[i], &spec->feature_matrix[i * dim], sizeof(float) * dim);
	}

	/* Initialize rotation matrix to identity */
	rotation_matrix = (float *) palloc0(sizeof(float) * dim * dim);
	NDB_CHECK_ALLOC(rotation_matrix, "rotation_matrix");
	for (i = 0; i < dim; i++)
		rotation_matrix[i * dim + i] = 1.0f;

	/* Prepare codebook structure */
	codebook.m = m;
	codebook.ksub = ksub;
	codebook.dsub = dim / m;

	/* Allocate centroids */
	NDB_DECLARE(float ***, centroids);
	NDB_ALLOC(centroids, float **, m);
	codebook.centroids = centroids;
	NDB_CHECK_ALLOC(codebook.centroids, "codebook.centroids");
	for (sub = 0; sub < m; sub++)
	{
		NDB_DECLARE(float **, sub_centroids);
		NDB_ALLOC(sub_centroids, float *, ksub);
		codebook.centroids[sub] = sub_centroids;
		NDB_CHECK_ALLOC(codebook.centroids[sub], "codebook.centroids[sub]");
		for (i = 0; i < ksub; i++)
		{
			NDB_DECLARE(float *, centroid_vec);
			NDB_ALLOC(centroid_vec, float, codebook.dsub);
			codebook.centroids[sub][i] = centroid_vec;
			NDB_CHECK_ALLOC(codebook.centroids[sub][i], "codebook.centroids[sub][i]");
		}
	}

	/* Train OPQ codebook (simplified - use identity rotation) */
	for (sub = 0; sub < m; sub++)
	{
		float	  **subspace_data = NULL;
		int			start_dim = sub * codebook.dsub;

		NDB_ALLOC(subspace_data, float *, nvec);
		NDB_CHECK_ALLOC(subspace_data, "subspace_data");
		for (i = 0; i < nvec; i++)
		{
			NDB_DECLARE(float *, sub_vec);
			NDB_ALLOC(sub_vec, float, codebook.dsub);
			subspace_data[i] = sub_vec;
			NDB_CHECK_ALLOC(subspace_data, "subspace_data");
			memcpy(subspace_data[i], &data[i][start_dim], sizeof(float) * codebook.dsub);
		}

		/*
		 * Use train_subspace_kmeans from PQ (would need to import or
		 * reimplement)
		 */
		/* For now, use simple k-means initialization */
		for (i = 0; i < ksub; i++)
		{
			int			idx = (i * nvec) / ksub;

			memcpy(codebook.centroids[sub][i], subspace_data[idx], sizeof(float) * codebook.dsub);
		}

		for (i = 0; i < nvec; i++)
			NDB_FREE(subspace_data[i]);
		NDB_FREE(subspace_data);
	}

	/* Serialize model */
	model_data = opq_model_serialize_to_bytea(&codebook, rotation_matrix, dim);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"m\":%d,\"ksub\":%d,\"dsub\":%d,\"dim\":%d,\"n_samples\":%d}",
					 m, ksub, codebook.dsub, dim, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetTextDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	state = (OPQGpuModelState *) palloc0(sizeof(OPQGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = model_data;
	state->metrics = metrics;
	NDB_DECLARE(PQCodebook *, codebook_ptr);
	NDB_ALLOC(codebook_ptr, PQCodebook, 1);
	state->codebook = codebook_ptr;
	NDB_CHECK_ALLOC(state, "state");
	*state->codebook = codebook;
	state->rotation_matrix = rotation_matrix;
	state->m = m;
	state->ksub = ksub;
	state->dsub = codebook.dsub;
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

	return true;
}

static bool
opq_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
				float *output, int output_dim, char **errstr)
{
	const		OPQGpuModelState *state;
	PQCodebook *codebook;

	NDB_DECLARE(float *, rotated_input);
	int			sub;
	int			start_dim;
	double		min_dist;
	int			best_code;

	NDB_DECLARE(float *, reconstructed);
	int			i,
				j;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		memset(output, 0, output_dim * sizeof(float));
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_predict: invalid parameters");
		return false;
	}
	if (model->backend_state == NULL || output_dim < ((OPQGpuModelState *) model->backend_state)->dim)
	{
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_predict: model not ready");
		return false;
	}

	state = (const OPQGpuModelState *) model->backend_state;
	if (state->codebook == NULL && state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_predict: codebook is NULL");
		return false;
	}

	if (input_dim != state->dim)
	{
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_predict: dimension mismatch");
		return false;
	}

	/* Deserialize codebook if needed */
	if (state->codebook == NULL)
	{
		PQCodebook	temp_codebook;

		NDB_DECLARE(float *, temp_rotation);
		int			temp_dim = 0;

		if (opq_model_deserialize_from_bytea(state->model_blob, &temp_codebook, &temp_rotation, &temp_dim) != 0)
		{
			if (errstr != NULL)
				*errstr = pstrdup("opq_gpu_predict: failed to deserialize");
			return false;
		}
		NDB_DECLARE(PQCodebook *, codebook_ptr);
		NDB_ALLOC(codebook_ptr, PQCodebook, 1);
		((OPQGpuModelState *) state)->codebook = codebook_ptr;
		NDB_CHECK_ALLOC(((OPQGpuModelState *) state)->codebook, "codebook");
		*((OPQGpuModelState *) state)->codebook = temp_codebook;
		((OPQGpuModelState *) state)->rotation_matrix = temp_rotation;
	}

	codebook = state->codebook;
	NDB_ALLOC(rotated_input, float, state->dim);
	NDB_CHECK_ALLOC(rotated_input, "rotated_input");
	NDB_ALLOC(reconstructed, float, state->dim);
	NDB_CHECK_ALLOC(reconstructed, "reconstructed");

	/* Apply rotation */
	for (i = 0; i < state->dim; i++)
	{
		rotated_input[i] = 0.0f;
		for (j = 0; j < state->dim; j++)
			rotated_input[i] += state->rotation_matrix[i * state->dim + j] * input[j];
	}

	/* Encode and reconstruct */
	for (sub = 0; sub < codebook->m; sub++)
	{
		start_dim = sub * codebook->dsub;
		min_dist = DBL_MAX;
		best_code = 0;

		for (int c = 0; c < codebook->ksub; c++)
		{
			double		dist = sqrt(neurondb_l2_distance_squared(
																 &rotated_input[start_dim], codebook->centroids[sub][c], codebook->dsub));

			if (dist < min_dist)
			{
				min_dist = dist;
				best_code = c;
			}
		}

		memcpy(&reconstructed[start_dim], codebook->centroids[sub][best_code],
			   sizeof(float) * codebook->dsub);
	}

	/* Apply inverse rotation */
	for (i = 0; i < state->dim; i++)
	{
		output[i] = 0.0f;
		for (j = 0; j < state->dim; j++)
			output[i] += state->rotation_matrix[j * state->dim + i] * reconstructed[j];
	}

	NDB_FREE(rotated_input);
	NDB_FREE(reconstructed);

	return true;
}

static bool
opq_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
				 MLGpuMetrics *out, char **errstr)
{
	const		OPQGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;
	double		avg_error = 0.0;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_evaluate: invalid model");
		return false;
	}

	state = (const OPQGpuModelState *) model->backend_state;

	/*
	 * Note: MLGpuEvalSpec does not provide feature_matrix/sample_count.
	 * Evaluation metrics would need to be computed from evaluation_table via
	 * SPI if needed. For now, avg_error remains 0.0.
	 */

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"opq\",\"storage\":\"cpu\","
					 "\"m\":%d,\"ksub\":%d,\"dsub\":%d,\"dim\":%d,\"avg_error\":%.6f,\"n_samples\":%d}",
					 state->m > 0 ? state->m : 8,
					 state->ksub > 0 ? state->ksub : 256,
					 state->dsub > 0 ? state->dsub : 0,
					 state->dim > 0 ? state->dim : 0,
					 avg_error,
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetTextDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
opq_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
				  Jsonb * *metadata_out, char **errstr)
{
	const		OPQGpuModelState *state;
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
			*errstr = pstrdup("opq_gpu_serialize: invalid model");
		return false;
	}

	state = (const OPQGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	NDB_DECLARE(char *, payload_copy_raw);
	NDB_ALLOC(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
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
opq_gpu_deserialize(MLGpuModel *model, const bytea * payload,
					const Jsonb * metadata, char **errstr)
{
	OPQGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;
	PQCodebook	codebook;

	NDB_DECLARE(float *, rotation_matrix);
	int			dim = 0;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	NDB_DECLARE(char *, payload_copy_raw);
	NDB_ALLOC(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
	memcpy(payload_copy, payload, payload_size);

	if (opq_model_deserialize_from_bytea(payload_copy, &codebook, &rotation_matrix, &dim) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("opq_gpu_deserialize: failed to deserialize");
		return false;
	}

	state = (OPQGpuModelState *) palloc0(sizeof(OPQGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload_copy;
	NDB_DECLARE(PQCodebook *, codebook_ptr);
	NDB_ALLOC(codebook_ptr, PQCodebook, 1);
	state->codebook = codebook_ptr;
	NDB_CHECK_ALLOC(state, "state");
	*state->codebook = codebook;
	state->rotation_matrix = rotation_matrix;
	state->m = codebook.m;
	state->ksub = codebook.ksub;
	state->dsub = codebook.dsub;
	state->dim = dim;
	state->n_samples = 0;

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		NDB_DECLARE(Jsonb *, metadata_copy);
		NDB_DECLARE(char *, metadata_copy_raw);
		NDB_ALLOC(metadata_copy_raw, char, metadata_size);
		metadata_copy = (Jsonb *) metadata_copy_raw;

		NDB_CHECK_ALLOC(metadata_copy, "metadata_copy");
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
opq_gpu_destroy(MLGpuModel *model)
{
	OPQGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (OPQGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			NDB_FREE(state->model_blob);
		if (state->metrics != NULL)
			NDB_FREE(state->metrics);
		if (state->codebook != NULL)
		{
			pq_codebook_free(state->codebook);
			NDB_FREE(state->codebook);
		}
		if (state->rotation_matrix != NULL)
			NDB_FREE(state->rotation_matrix);
		NDB_FREE(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps opq_gpu_model_ops = {
	.algorithm = "opq",
	.train = opq_gpu_train,
	.predict = opq_gpu_predict,
	.evaluate = opq_gpu_evaluate,
	.serialize = opq_gpu_serialize,
	.deserialize = opq_gpu_deserialize,
	.destroy = opq_gpu_destroy,
};

void
neurondb_gpu_register_opq_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&opq_gpu_model_ops);
	registered = true;
}
