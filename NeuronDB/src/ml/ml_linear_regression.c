/*-------------------------------------------------------------------------
 *
 * ml_linear_regression.c
 *    Ordinary least squares linear regression.
 *
 * This module implements linear regression using normal equations to compute
 * regression coefficients. Training uses streaming accumulation for large
 * datasets, with model serialization and catalog storage.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_linear_regression.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_linear_regression_internal.h"
#include "ml_catalog.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_linear_regression.h"
#include "neurondb_cuda_linreg.h"
#include "neurondb_safe_memory.h"
#include "neurondb_validation.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_spi_safe.h"
#include "neurondb_sql.h"
#include "neurondb_constants.h"
#include "neurondb_json.h"
#include "utils/lsyscache.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
extern int	ndb_cuda_linreg_evaluate(const bytea * model_data,
									 const float *features,
									 const double *targets,
									 int n_samples,
									 int feature_dim,
									 double *mse_out,
									 double *mae_out,
									 double *rmse_out,
									 double *r_squared_out,
									 char **errstr);
#endif

#include <math.h>
#include <float.h>

typedef struct LinRegDataset
{
	float	   *features;
	double	   *targets;
	int			n_samples;
	int			feature_dim;
}			LinRegDataset;

/*
 * Streaming accumulator for incremental X'X and X'y computation
 * This avoids loading all data into memory at once
 */
typedef struct LinRegStreamAccum
{
	double	  **XtX;
	double	   *Xty;
	int			feature_dim;
	int			n_samples;
	double		y_sum;
	double		y_sq_sum;
	bool		initialized;
}			LinRegStreamAccum;

static void linreg_dataset_init(LinRegDataset * dataset);
static void linreg_dataset_free(LinRegDataset * dataset);
static void linreg_dataset_load_limited(const char *quoted_tbl,
										const char *quoted_feat,
										const char *quoted_target,
										LinRegDataset * dataset,
										int max_rows);
static void linreg_stream_accum_init(LinRegStreamAccum * accum, int dim);
static void linreg_stream_accum_free(LinRegStreamAccum * accum);
static void linreg_stream_accum_add_row(LinRegStreamAccum * accum,
										const float *features,
										double target);
static void linreg_stream_process_chunk(const char *quoted_tbl,
										const char *quoted_feat,
										const char *quoted_target,
										LinRegStreamAccum * accum,
										int chunk_size,
										int offset,
										int *rows_processed);
static bytea * linreg_model_serialize(const LinRegModel * model, uint8 training_backend);
static LinRegModel * linreg_model_deserialize(const bytea * data, uint8 *training_backend_out);
static bool linreg_metadata_is_gpu(Jsonb * metadata) __attribute__((unused));
static bool linreg_try_gpu_predict_catalog(int32 model_id,
										   const Vector *feature_vec,
										   double *result_out);
static bool linreg_load_model_from_catalog(int32 model_id, LinRegModel * *out);
static Jsonb * evaluate_linear_regression_by_model_id_jsonb(int32 model_id,
															text * table_name,
															text * feature_col,
															text * label_col);

/*
 * Matrix inversion using Gauss-Jordan elimination
 * Returns false if matrix is singular
 */
static bool
matrix_invert(double **matrix, int n, double **result)
{
	NDB_DECLARE (double **, augmented);
	int			i,
				j,
				k;
	double		pivot,
				factor;

	NDB_ALLOC(augmented, double *, n);

	for (i = 0; i < n; i++)
	{
		NDB_ALLOC(augmented[i], double, 2 * n);

		for (j = 0; j < n; j++)
		{
			augmented[i][j] = matrix[i][j];
			augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
		}
	}

	for (i = 0; i < n; i++)
	{
		pivot = augmented[i][i];
		if (fabs(pivot) < 1e-10)
		{
			bool		found = false;

			for (k = i + 1; k < n; k++)
			{
				if (fabs(augmented[k][i]) > 1e-10)
				{
					double	   *temp = augmented[i];

					augmented[i] = augmented[k];
					augmented[k] = temp;
					pivot = augmented[i][i];
					found = true;
					break;
				}
			}
			if (!found)
			{
				for (j = 0; j < n; j++)
					NDB_FREE(augmented[j]);
				NDB_FREE(augmented);

				return false;
			}
		}

		for (j = 0; j < 2 * n; j++)
			augmented[i][j] /= pivot;

		for (k = 0; k < n; k++)
		{
			if (k != i)
			{
				factor = augmented[k][i];
				for (j = 0; j < 2 * n; j++)
					augmented[k][j] -=
						factor * augmented[i][j];
			}
		}
	}

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			result[i][j] = augmented[i][j + n];

	for (i = 0; i < n; i++)
		NDB_FREE(augmented[i]);
	NDB_FREE(augmented);

	return true;
}

/*
 * linreg_dataset_init
 */
static void
linreg_dataset_init(LinRegDataset * dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(LinRegDataset));
}

/*
 * linreg_dataset_free
 */
static void
linreg_dataset_free(LinRegDataset * dataset)
{
	Assert(dataset != NULL);
	if (dataset == NULL)
		return;
	NDB_FREE(dataset->features);
	NDB_FREE(dataset->targets);

	linreg_dataset_init(dataset);
}

/*
 * linreg_dataset_load_limited
 *
 * Load dataset with LIMIT clause to avoid loading too much data
 */
static void
linreg_dataset_load_limited(const char *quoted_tbl,
							const char *quoted_feat,
							const char *quoted_target,
							LinRegDataset * dataset,
							int max_rows)
{
	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;
	int			ret;
	int			n_samples = 0;
	int			feature_dim = 0;
	int			i;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;

	if (dataset == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_MSG("linreg_dataset_load_limited: dataset is NULL"))));

	if (max_rows <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("linreg_dataset_load_limited: max_rows must be positive"))));

	oldcontext = CurrentMemoryContext;

	/* Begin SPI session - handles connection state automatically */
	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Use centralized SQL query function */
	{
		NDB_DECLARE (char *, query_str);

		query_str = (char *) ndb_sql_get_load_dataset_limited(quoted_feat, quoted_target, quoted_tbl, max_rows);
		ret = ndb_spi_execute(spi_session, query_str, true, 0);
		NDB_FREE(query_str);
	}
	if (ret != SPI_OK_SELECT)
	{
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_MSG("linreg_dataset_load_limited: query failed"))));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("linreg_dataset_load_limited: need at least 10 samples, got %d"),
						n_samples)));
	}

	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* Safe access for complex types - validate before access */
	if (SPI_processed > 0 && SPI_tuptable != NULL && SPI_tuptable->vals != NULL && 
		SPI_tuptable->vals[0] != NULL && SPI_tuptable->tupdesc != NULL)
	{
		HeapTuple	first_tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		bool		feat_null;
		Vector	   *vec;

		feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
		if (!feat_null)
		{
			if (feat_is_array)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

				if (ARR_NDIM(arr) != 1)
				{
					NDB_SPI_SESSION_END(spi_session);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg(NDB_ERR_MSG("linreg_dataset_load_limited: features array must be 1-D"))));
				}
				feature_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				vec = DatumGetVector(feat_datum);
				if (vec == NULL)
				{
					NDB_SPI_SESSION_END(spi_session);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg(NDB_ERR_MSG("linreg_dataset_load_limited: failed to get vector from datum"))));
				}
				feature_dim = vec->dim;
			}
		}
	}

	if (feature_dim <= 0)
	{
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("linreg_dataset_load_limited: could not determine ")
						"feature dimension")));
	}

	MemoryContextSwitchTo(oldcontext);
	NDB_ALLOC(dataset->features, float, (size_t) n_samples * (size_t) feature_dim);
	NDB_ALLOC(dataset->targets, double, (size_t) n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple;
		TupleDesc	tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		float	   *row;

		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			continue;
		}
		tuple = SPI_tuptable->vals[i];
		tupdesc = SPI_tuptable->tupdesc;
		if (tupdesc == NULL)
		{
			continue;
		}

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		if (feat_null)
			continue;

		row = dataset->features + (i * feature_dim);
		if (feat_is_array)
		{
			ArrayType  *arr = DatumGetArrayTypeP(feat_datum);
			int			ndims = ARR_NDIM(arr);
			int			dimlen = (ndims == 1) ? ARR_DIMS(arr)[0] : 0;

			if (ndims != 1 || dimlen != feature_dim)
			{
				NDB_SPI_SESSION_END(spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg(NDB_ERR_MSG("linreg_dataset_load_limited: inconsistent array feature dimensions"))));
			}
			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				float8	   *data = (float8 *) ARR_DATA_PTR(arr);
				int			j;

				for (j = 0; j < feature_dim; j++)
					row[j] = (float) data[j];
			}
			else
			{
				float4	   *data = (float4 *) ARR_DATA_PTR(arr);

				memcpy(row, data, sizeof(float) * feature_dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec == NULL)
			{
				NDB_SPI_SESSION_END(spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg(NDB_ERR_MSG("linreg_dataset_load_limited: failed to get vector from datum"))));
			}
			if (vec->dim != feature_dim)
			{
				NDB_SPI_SESSION_END(spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg(NDB_ERR_MSG("linreg_dataset_load_limited: inconsistent ")
								"vector dimensions")));
			}
			memcpy(row, vec->data, sizeof(float) * feature_dim);
		}

		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (targ_null)
			continue;

		{
			Oid			targ_type = SPI_gettypeid(tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID)
				dataset->targets[i] =
					(double) DatumGetInt32(targ_datum);
			else if (targ_type == INT8OID)
				dataset->targets[i] =
					(double) DatumGetInt64(targ_datum);
			else
				dataset->targets[i] =
					DatumGetFloat8(targ_datum);
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	NDB_SPI_SESSION_END(spi_session);
}

/*
 * linreg_stream_accum_init
 *
 * Initialize streaming accumulator for incremental X'X and X'y computation
 */
static void
linreg_stream_accum_init(LinRegStreamAccum * accum, int dim)
{
	int			i;
	int			dim_with_intercept;

	if (accum == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_MSG("linreg_stream_accum_init: accum is NULL"))));

	if (dim <= 0 || dim > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("linreg_stream_accum_init: invalid feature dimension %d"),
						dim)));

	dim_with_intercept = dim + 1;

	memset(accum, 0, sizeof(LinRegStreamAccum));

	accum->feature_dim = dim;
	accum->n_samples = 0;
	accum->y_sum = 0.0;
	accum->y_sq_sum = 0.0;
	accum->initialized = false;

	NDB_ALLOC(accum->XtX, double *, dim_with_intercept);

	for (i = 0; i < dim_with_intercept; i++)
	{
		NDB_ALLOC(accum->XtX[i], double, dim_with_intercept);
	}

	NDB_ALLOC(accum->Xty, double, dim_with_intercept);

	accum->initialized = true;
}

/*
 * linreg_stream_accum_free
 *
 * Free memory allocated for streaming accumulator
 */
static void
linreg_stream_accum_free(LinRegStreamAccum * accum)
{
	int			i;

	Assert(accum != NULL);
	if (accum == NULL)
		return;

	if (accum->XtX != NULL)
	{
		int			dim_with_intercept = accum->feature_dim + 1;

		for (i = 0; i < dim_with_intercept; i++)
		{
			NDB_FREE(accum->XtX[i]);
		}
		NDB_FREE(accum->XtX);
	}

	NDB_FREE(accum->Xty);

	memset(accum, 0, sizeof(LinRegStreamAccum));
}

/*
 * linreg_stream_accum_add_row
 *
 * Add a single row to the streaming accumulator, updating X'X and X'y
 */
static void
linreg_stream_accum_add_row(LinRegStreamAccum * accum,
							const float *features,
							double target)
{
	int			i;
	int			j;
	int			dim_with_intercept;
	NDB_DECLARE (double *, xi);

	if (accum == NULL || !accum->initialized)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_MSG("linreg_stream_accum_add_row: accumulator not initialized"))));

	if (features == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("linreg_stream_accum_add_row: features is NULL"))));

	dim_with_intercept = accum->feature_dim + 1;

	NDB_ALLOC(xi, double, dim_with_intercept);

	xi[0] = 1.0;
	for (i = 0; i < accum->feature_dim; i++)
		xi[i + 1] = (double) features[i];

	for (j = 0; j < dim_with_intercept; j++)
	{
		for (i = 0; i < dim_with_intercept; i++)
			accum->XtX[j][i] += xi[j] * xi[i];

		accum->Xty[j] += xi[j] * target;
	}

	accum->n_samples++;
	accum->y_sum += target;
	accum->y_sq_sum += target * target;

	NDB_FREE(xi);
}

/*
 * linreg_stream_process_chunk
 *
 * Process a chunk of data from the table, accumulating statistics
 * Returns number of rows processed in this chunk
 */
static void
linreg_stream_process_chunk(const char *quoted_tbl,
							const char *quoted_feat,
							const char *quoted_target,
							LinRegStreamAccum * accum,
							int chunk_size,
							int offset,
							int *rows_processed)
{
	int			ret;
	int			i;
	int			n_rows;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	TupleDesc	tupdesc;
	NDB_DECLARE (float *, row_buffer);

	if (quoted_tbl == NULL || quoted_feat == NULL || quoted_target == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: linreg_stream_process_chunk: NULL parameter")));

	if (accum == NULL || !accum->initialized)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_MSG("linreg_stream_process_chunk: accumulator not initialized"))));

	if (chunk_size <= 0 || chunk_size > 100000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("linreg_stream_process_chunk: invalid chunk_size %d"),
						chunk_size)));

	if (rows_processed == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("linreg_stream_process_chunk: rows_processed is NULL"))));

	*rows_processed = 0;

	/* Use centralized SQL query function */

	/*
	 * Note: For views, we can't use ctid, so we use LIMIT/OFFSET without
	 * ORDER BY
	 */
	/* This is non-deterministic but efficient for large datasets */
	{
		NDB_DECLARE (char *, query_str);

		query_str = (char *) ndb_sql_get_load_dataset_chunk(quoted_feat, quoted_target, quoted_tbl, chunk_size, offset);
		ret = ndb_spi_execute_safe(query_str, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		NDB_FREE(query_str);
	}
	if (ret != SPI_OK_SELECT)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_MSG("linreg_stream_process_chunk: query failed"))));
	}

	n_rows = SPI_processed;
	if (n_rows == 0)
	{
		*rows_processed = 0;
		return;
	}

	/* Determine feature type from first row */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	{
		tupdesc = SPI_tuptable->tupdesc;
		feat_type_oid = SPI_gettypeid(tupdesc, 1);
		if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			feat_is_array = true;
	}

	NDB_ALLOC(row_buffer, float, accum->feature_dim);

		for (i = 0; i < n_rows; i++)
		{
			HeapTuple	tuple;
			Datum		feat_datum;
			Datum		targ_datum;
			bool		feat_null;
			bool		targ_null;
			double		target;
			Vector	   *vec;
			ArrayType  *arr;
			int			j;

			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
				i >= SPI_processed || SPI_tuptable->vals[i] == NULL || tupdesc == NULL)
			{
				continue;
			}
			tuple = SPI_tuptable->vals[i];

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			if (tupdesc->natts < 2)
			{
				continue;
			}
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		/* Extract features */
		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
			if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != accum->feature_dim)
			{
				NDB_FREE(row_buffer);

				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg(NDB_ERR_MSG("linreg_stream_process_chunk: inconsistent feature dimensions"))));
			}

			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				double	   *arr_data = (double *) ARR_DATA_PTR(arr);

				for (j = 0; j < accum->feature_dim; j++)
					row_buffer[j] = (float) arr_data[j];
			}
			else
			{
				float	   *arr_data = (float *) ARR_DATA_PTR(arr);

				memcpy(row_buffer, arr_data, sizeof(float) * accum->feature_dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec->dim != accum->feature_dim)
			{
				NDB_FREE(row_buffer);

				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg(NDB_ERR_MSG("linreg_stream_process_chunk: feature dimension mismatch"))));
			}
			memcpy(row_buffer, vec->data, sizeof(float) * accum->feature_dim);
		}

		/* Extract target */
		{
			Oid			targ_type = SPI_gettypeid(tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID)
				target = (double) DatumGetInt32(targ_datum);
			else if (targ_type == INT8OID)
				target = (double) DatumGetInt64(targ_datum);
			else
				target = DatumGetFloat8(targ_datum);
		}

		/* Add row to accumulator */
		linreg_stream_accum_add_row(accum, row_buffer, target);
		(*rows_processed)++;
	}

	NDB_FREE(row_buffer);
}

/*
 * linreg_model_serialize
 */
static bytea *
linreg_model_serialize(const LinRegModel * model, uint8 training_backend)
{
	StringInfoData buf = {0};
	int			i;

	Assert(model != NULL);
	if (model == NULL)
		return NULL;

	/* Validate model before serialization */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: linreg_model_serialize: invalid n_features %d",
						model->n_features),
				 errdetail("Feature count is %d, must be between 1 and 10000", model->n_features),
				 errhint("Model data may be corrupted. Verify the model structure.")));
	}

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: linreg_model_serialize: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	pq_begintypsend(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) */
	pq_sendbyte(&buf, training_backend);

	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendfloat8(&buf, model->intercept);
	pq_sendfloat8(&buf, model->r_squared);
	pq_sendfloat8(&buf, model->mse);
	pq_sendfloat8(&buf, model->mae);

	if (model->coefficients != NULL && model->n_features > 0)
	{
		for (i = 0; i < model->n_features; i++)
			pq_sendfloat8(&buf, model->coefficients[i]);
	}

	return pq_endtypsend(&buf);
}

/*
 * linreg_model_deserialize
 */
static LinRegModel *
linreg_model_deserialize(const bytea * data, uint8 *training_backend_out)
{
	NDB_DECLARE (LinRegModel *, model);
	StringInfoData buf = {0};
	int			i;
	uint8		training_backend = 0;

	Assert(data != NULL);
	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	/* Read training_backend first */
	training_backend = (uint8) pq_getmsgbyte(&buf);

	NDB_ALLOC(model, LinRegModel, 1);

	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->intercept = pq_getmsgfloat8(&buf);
	model->r_squared = pq_getmsgfloat8(&buf);
	model->mse = pq_getmsgfloat8(&buf);
	model->mae = pq_getmsgfloat8(&buf);

	/* Validate deserialized values */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		NDB_FREE(model);

		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid n_features %d in deserialized model",
						model->n_features),
				 errdetail("Feature count is %d, must be between 1 and 10000", model->n_features),
				 errhint("Model data may be corrupted. Verify the model was serialized correctly.")));
	}
	if (model->n_samples < 0 || model->n_samples > 100000000)
	{
		NDB_FREE(model);

		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: invalid n_samples %d in deserialized model",
						model->n_samples),
				 errdetail("Sample count is %d, must be between 0 and 100000000", model->n_samples),
				 errhint("Model data may be corrupted. Verify the model was serialized correctly.")));
	}

	if (model->n_features > 0)
	{
		NDB_ALLOC(model->coefficients, double, model->n_features);

		for (i = 0; i < model->n_features; i++)
			model->coefficients[i] = pq_getmsgfloat8(&buf);
	}

	/* Return training_backend if output parameter provided */
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;

	return model;
}

/*
 * linreg_metadata_is_gpu
 *
 * Checks if a model's metadata indicates it's a GPU-trained model.
 * Now checks for training_backend integer (1=GPU, 0=CPU) instead of "storage" string.
 */
static bool
linreg_metadata_is_gpu(Jsonb * metadata)
{
	bool		is_gpu = false;
	JsonbIterator *it;
	JsonbValue	v;
	JsonbIteratorToken r;

	if (metadata == NULL)
		return false;

	/* Check for training_backend integer in metrics */
	it = JsonbIteratorInit((JsonbContainer *) &metadata->root);
	while ((r = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
	{
		if (r == WJB_KEY && v.type == jbvString)
		{
			char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

			if (strcmp(key, "training_backend") == 0)
			{
				r = JsonbIteratorNext(&it, &v, true);
				if (r == WJB_VALUE && v.type == jbvNumeric)
				{
					int			backend = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					is_gpu = (backend == 1);
				}
			}
			NDB_FREE(key);
		}
	}

	return is_gpu;
}

/*
 * linreg_try_gpu_predict_catalog
 */
static bool
linreg_try_gpu_predict_catalog(int32 model_id,
							   const Vector *feature_vec,
							   double *result_out)
{
	NDB_DECLARE (bytea *, payload);
	NDB_DECLARE (Jsonb *, metrics);
	NDB_DECLARE (char *, gpu_err);
	double		prediction = 0.0;
	bool		success = false;

	if (!neurondb_gpu_is_available())
		return false;
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
		goto cleanup;

	/* Check if this is a GPU model using training_backend */
	{
		bool is_gpu_model = false;
		JsonbIterator *it;
		JsonbValue	v;
		JsonbIteratorToken r;
		
		if (metrics != NULL)
		{
			it = JsonbIteratorInit((JsonbContainer *) &metrics->root);
			while ((r = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
			{
				if (r == WJB_KEY && v.type == jbvString)
				{
					char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

					if (strcmp(key, "training_backend") == 0)
					{
						r = JsonbIteratorNext(&it, &v, true);
						if (r == WJB_VALUE && v.type == jbvNumeric)
						{
							int			backend = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
							is_gpu_model = (backend == 1);
						}
					}
					NDB_FREE(key);
				}
			}
		}
		
		if (!is_gpu_model)
			goto cleanup;
	}
	if (ndb_gpu_linreg_predict(payload,
							   feature_vec->data,
							   feature_vec->dim,
							   &prediction,
							   &gpu_err)
		== 0)
	{
		if (result_out != NULL)
			*result_out = prediction;
		success = true;
	}

cleanup:
	NDB_FREE(payload);
	NDB_FREE(metrics);
	NDB_FREE(gpu_err);

	return success;
}

/*
 * linreg_load_model_from_catalog
 */
static bool
linreg_load_model_from_catalog(int32 model_id, LinRegModel * *out)
{
	NDB_DECLARE (bytea *, payload);
	NDB_DECLARE (Jsonb *, metrics);

	if (out == NULL)
		return false;

	*out = NULL;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
	{
		NDB_FREE(metrics);

		return false;
	}

	/* Skip GPU models - they should be handled by GPU prediction */
	/* Inline the GPU check to avoid return statement crash */
	{
		bool is_gpu_model = false;

		if (metrics != NULL)
		{
			JsonbIterator *it;
			JsonbValue	v;
			JsonbIteratorToken r;

			it = JsonbIteratorInit((JsonbContainer *) &metrics->root);
			while ((r = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
			{
				if (r == WJB_KEY && v.type == jbvString)
				{
					char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

					if (strcmp(key, "training_backend") == 0)
					{
						r = JsonbIteratorNext(&it, &v, true);
						if (r == WJB_VALUE && v.type == jbvNumeric)
						{
							int			backend = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
							is_gpu_model = (backend == 1);
						}
					}
					NDB_FREE(key);
				}
			}
		}

		if (is_gpu_model)
		{
			NDB_FREE(payload);
			NDB_FREE(metrics);

			return false;
		}
	}

	/* Try to deserialize CPU model, but handle failures gracefully */
	PG_TRY();
	{
		*out = linreg_model_deserialize(payload, NULL);
	}
	PG_CATCH();
	{
		FlushErrorState();
		*out = NULL;
		ereport(WARNING, (errmsg("linreg_load_model_from_catalog: CPU model deserialization failed, assuming GPU model")));
	}
	PG_END_TRY();

	NDB_FREE(payload);
	NDB_FREE(metrics);

	return (*out != NULL);
}

/*
 * train_linear_regression
 *
 * Trains a linear regression model using OLS
 * Returns model_id (for GPU path) or coefficients array (for CPU path)
 */
PG_FUNCTION_INFO_V1(train_linear_regression);

Datum
train_linear_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	NDB_DECLARE (char *, tbl_str);
	NDB_DECLARE (char *, feat_str);
	NDB_DECLARE (char *, targ_str);
	int			nvec = 0;
	int			dim = 0;
	LinRegDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_target;
	MLGpuTrainResult gpu_result;
	NDB_DECLARE (char *, gpu_err);
	NDB_DECLARE (Jsonb *, gpu_hyperparams);
	int32		model_id = 0;
	MemoryContext oldcontext;

	/*
	 * Save the function's memory context - this is the per-call context that
	 * Postgres manages
	 */
	oldcontext = CurrentMemoryContext;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_target = quote_identifier(targ_str);

	/*
	 * First, determine feature dimension and row count without loading all
	 * data
	 */
	{
		int			ret;
		Oid			feat_type_oid = InvalidOid;
		bool		feat_is_array = false;
		NDB_DECLARE (NdbSpiSession *, check_spi_session);
		MemoryContext check_oldcontext;

		check_oldcontext = CurrentMemoryContext;
		Assert(check_oldcontext != NULL);
		NDB_SPI_SESSION_BEGIN(check_spi_session, check_oldcontext);

		/* Get feature dimension from first row - use centralized SQL query */
		{
			NDB_DECLARE (char *, check_query);

			check_query = (char *) ndb_sql_get_check_dataset(quoted_feat, quoted_target, quoted_tbl);
			ret = ndb_spi_execute(check_spi_session, check_query, true, 0);
			NDB_FREE(check_query);
		}
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			NDB_SPI_SESSION_END(check_spi_session);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_linear_regression: no valid rows found"),
					 errdetail("SPI execution returned code %d (expected %d), processed %lu rows", ret, SPI_OK_SELECT, (unsigned long) SPI_processed),
					 errhint("Verify the table exists and contains valid feature and target columns.")));
		}

		/* Determine feature dimension */
		if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL
			&& SPI_tuptable->vals != NULL && SPI_processed > 0)
		{
			HeapTuple	first_tuple = SPI_tuptable->vals[0];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Datum		feat_datum;
			bool		feat_null;

			if (first_tuple == NULL)
			{
				NDB_SPI_SESSION_END(check_spi_session);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: train_linear_regression: first tuple is NULL")));
			}

			feat_type_oid = SPI_gettypeid(tupdesc, 1);
			if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
				feat_is_array = true;

			feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
			if (!feat_null)
			{
				if (feat_is_array)
				{
					ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

					if (ARR_NDIM(arr) != 1)
					{
						NDB_SPI_SESSION_END(check_spi_session);
						NDB_FREE(tbl_str);
						NDB_FREE(feat_str);
						NDB_FREE(targ_str);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neurondb: train_linear_regression: features array must be 1-D"),
								 errdetail("Array has %d dimensions, expected 1", ARR_NDIM(arr)),
								 errhint("Ensure feature column contains 1-dimensional arrays only.")));
					}
					dim = ARR_DIMS(arr)[0];
				}
				else
				{
					Vector	   *vec = DatumGetVector(feat_datum);

					if (vec == NULL)
					{
						NDB_SPI_SESSION_END(check_spi_session);
						NDB_FREE(tbl_str);
						NDB_FREE(feat_str);
						NDB_FREE(targ_str);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neurondb: train_linear_regression: failed to get vector from datum")));
					}
					dim = vec->dim;
				}
			}
		}

		if (dim <= 0)
		{
			NDB_SPI_SESSION_END(check_spi_session);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_linear_regression: could not determine feature dimension"),
					 errdetail("Feature dimension is %d (must be > 0)", dim),
					 errhint("Ensure feature column contains valid vector or array data.")));
		}

		/* Get row count - use centralized SQL query */
		{
			NDB_DECLARE (char *, count_query);

			count_query = (char *) ndb_sql_get_count_dataset(quoted_feat, quoted_target, quoted_tbl);
			ret = ndb_spi_execute(check_spi_session, count_query, true, 0);
			NDB_FREE(count_query);
		}
		if (ret == SPI_OK_SELECT && SPI_processed > 0
			&& SPI_tuptable != NULL && SPI_tuptable->vals != NULL
			&& SPI_tuptable->tupdesc != NULL)
		{
			bool		count_null;
			HeapTuple	tuple = SPI_tuptable->vals[0];

			if (tuple != NULL)
			{
				Datum		count_datum = SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, &count_null);

				if (!count_null)
					nvec = DatumGetInt32(count_datum);
			}
		}

		NDB_SPI_SESSION_END(check_spi_session);

		if (nvec < 10)
		{
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_linear_regression: need at least 10 samples, got %d",
							nvec),
					 errdetail("Dataset contains %d rows, minimum required is 10", nvec),
					 errhint("Add more data rows to the training table.")));
		}
	}

	/* Define max_samples limit for large datasets */
	{
		int			max_samples = 500000;	/* Limit to 500k samples for very
											 * large datasets */

		/*
		 * Limit sample size for very large datasets to avoid excessive
		 * training time
		 */
		if (nvec > max_samples)
		{
			nvec = max_samples;
		}

		/*
		 * Try GPU training first - always use GPU when enabled and kernel
		 * available
		 */
		/* Initialize GPU if needed (lazy initialization) */
		if (NDB_SHOULD_TRY_GPU())
		{
			ndb_gpu_init_if_needed();
		}

		if (neurondb_gpu_is_available() && nvec > 0 && dim > 0
			&& ndb_gpu_kernel_enabled("linreg_train"))
		{
			int			gpu_sample_limit = nvec;
			StringInfoData hyperbuf = {0};
			bool		gpu_train_result = false;

			/* Load limited dataset for GPU training */
			linreg_dataset_init(&dataset);
			linreg_dataset_load_limited(quoted_tbl,
										quoted_feat,
										quoted_target,
										&dataset,
										gpu_sample_limit);
			
			initStringInfo(&hyperbuf);
			appendStringInfo(&hyperbuf, "{}");
			/* Use ndb_jsonb_in_cstring like test 006 fix */
			gpu_hyperparams = ndb_jsonb_in_cstring(hyperbuf.data);
			if (gpu_hyperparams == NULL)
			{
				NDB_FREE(hyperbuf.data);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("neurondb: failed to parse hyperparams JSON")));
			}
			NDB_FREE(hyperbuf.data);
			hyperbuf.data = NULL;

			{
				PG_TRY();
				{
					gpu_train_result = ndb_gpu_try_train_model("linear_regression",
																NULL,
																NULL,
																tbl_str,
																targ_str,
																NULL,
																0,
																gpu_hyperparams,
																dataset.features,
																dataset.targets,
																dataset.n_samples,
																dataset.feature_dim,
																0,
																&gpu_result,
																&gpu_err);
				}
				PG_CATCH();
				{
					FlushErrorState();
					gpu_train_result = false;
				}
				PG_END_TRY();
			}
			if (gpu_train_result && gpu_result.spec.model_data != NULL)
			{
#ifdef NDB_GPU_CUDA
				MLCatalogModelSpec spec;
				LinRegModel		linreg_model;
				NDB_DECLARE(bytea *, unified_model_data);
				NDB_DECLARE(Jsonb *, updated_metrics);
				char	   *base;
				NdbCudaLinRegModelHeader *hdr;
				int			i;
				size_t		expected_size;
				uint32		payload_size;

				elog(INFO,
					 "neurondb: linear_regression: GPU training succeeded");

				/* Validate GPU model data size before accessing */
				payload_size = VARSIZE(gpu_result.spec.model_data) - VARHDRSZ;
				if (payload_size < sizeof(NdbCudaLinRegModelHeader))
				{
					elog(ERROR,
						 "neurondb: linear_regression: GPU model data too small (%u bytes, minimum %zu bytes)",
						 payload_size,
						 sizeof(NdbCudaLinRegModelHeader));
				}

				/* Convert GPU format to unified format */
				base = VARDATA(gpu_result.spec.model_data);
				hdr = (NdbCudaLinRegModelHeader *) base;
				
				/* Validate header fields before use */
				if (hdr->feature_dim <= 0 || hdr->feature_dim > 100000)
				{
					elog(ERROR,
						 "neurondb: linear_regression: invalid feature_dim in GPU model header: %d",
						 hdr->feature_dim);
				}
				
				/* Validate expected payload size */
				expected_size = sizeof(NdbCudaLinRegModelHeader) + 
									sizeof(float) * (size_t) hdr->feature_dim;
				if (payload_size < expected_size)
				{
					elog(ERROR,
						 "neurondb: linear_regression: GPU model data too small for %d features (%u bytes, expected %zu bytes)",
						 hdr->feature_dim,
						 payload_size,
						 expected_size);
				}
				
				/* Coefficients are stored as floats after the header */
				{
					float	   *coef_src_float = (float *) (base + sizeof(NdbCudaLinRegModelHeader));
					
					/* Build LinRegModel structure */
					memset(&linreg_model, 0, sizeof(LinRegModel));
					linreg_model.n_features = hdr->feature_dim;
					linreg_model.n_samples = hdr->n_samples;
					linreg_model.intercept = (double) hdr->intercept; /* Convert float to double */
					linreg_model.r_squared = hdr->r_squared;
					linreg_model.mse = hdr->mse;
					linreg_model.mae = hdr->mae;

					/* Copy coefficients, converting from float to double */
					if (linreg_model.n_features > 0)
					{
						NDB_ALLOC(linreg_model.coefficients, double, linreg_model.n_features);
						for (i = 0; i < linreg_model.n_features; i++)
							linreg_model.coefficients[i] = (double) coef_src_float[i];
					}
				}


				/* Serialize using unified format with training_backend=1 (GPU) */
				unified_model_data = linreg_model_serialize(&linreg_model, 1);

				if (linreg_model.coefficients != NULL)
				{
					NDB_FREE(linreg_model.coefficients);
					linreg_model.coefficients = NULL;
				}

				/* Update metrics to use training_backend=1 */
				if (gpu_result.spec.metrics != NULL)
				{
					StringInfoData metrics_buf;
					initStringInfo(&metrics_buf);
					appendStringInfo(&metrics_buf,
									 "{\"algorithm\":\"linear_regression\","
									 "\"training_backend\":1,"
									 "\"n_features\":%d,"
									 "\"n_samples\":%d}",
									 linreg_model.n_features > 0 ? linreg_model.n_features : 0,
									 linreg_model.n_samples > 0 ? linreg_model.n_samples : 0);

					/* Use ndb_jsonb_in_cstring like test 006 fix */
					updated_metrics = ndb_jsonb_in_cstring(metrics_buf.data);
					if (updated_metrics == NULL)
					{
						NDB_FREE(metrics_buf.data);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
								 errmsg("neurondb: failed to parse metrics JSON")));
					}
					NDB_FREE(metrics_buf.data);
				}

				spec = gpu_result.spec;
				spec.model_data = unified_model_data;
				spec.metrics = updated_metrics;

				/* ALWAYS copy all string pointers to current memory context before switching contexts */
				/* This ensures the pointers remain valid after memory context switch */
				
				/* Copy algorithm */
				if (spec.algorithm != NULL)
				{
					PG_TRY();
					{
						spec.algorithm = pstrdup(spec.algorithm);
					}
					PG_CATCH();
					{
						FlushErrorState();
						spec.algorithm = pstrdup("linear_regression");
					}
					PG_END_TRY();
				}
				else
				{
					spec.algorithm = pstrdup("linear_regression");
				}

				/* Copy training_table */
				if (spec.training_table != NULL)
				{
					PG_TRY();
					{
						spec.training_table = pstrdup(spec.training_table);
					}
					PG_CATCH();
					{
						FlushErrorState();
						spec.training_table = (tbl_str != NULL) ? pstrdup(tbl_str) : NULL;
					}
					PG_END_TRY();
				}
				else if (tbl_str != NULL)
				{
					spec.training_table = pstrdup(tbl_str);
				}

				/* Copy training_column */
				if (spec.training_column != NULL)
				{
					PG_TRY();
					{
						spec.training_column = pstrdup(spec.training_column);
					}
					PG_CATCH();
					{
						FlushErrorState();
						spec.training_column = (targ_str != NULL) ? pstrdup(targ_str) : NULL;
					}
					PG_END_TRY();
				}
				else if (targ_str != NULL)
				{
					spec.training_column = pstrdup(targ_str);
				}

				/* Copy project_name - use fallback if invalid */
				if (spec.project_name != NULL)
				{
					PG_TRY();
					{
						/* Try to validate by checking first byte */
						if (spec.project_name[0] != '\0')
						{
							spec.project_name = pstrdup(spec.project_name);
						}
						else
						{
							spec.project_name = NULL;
						}
					}
					PG_CATCH();
					{
						FlushErrorState();
						spec.project_name = NULL;
					}
					PG_END_TRY();
				}

				if (spec.parameters == NULL)
				{
					spec.parameters = gpu_hyperparams;
					gpu_hyperparams = NULL;
				}

				spec.model_type = pstrdup("regression");

				/*
				 * Ensure we're in a valid memory context before calling
				 * ml_catalog_register_model
				 */
				MemoryContextSwitchTo(oldcontext);

				model_id = ml_catalog_register_model(&spec);
#else
				MLCatalogModelSpec spec;

				elog(INFO,
					 "neurondb: linear_regression: GPU training succeeded");
				spec = gpu_result.spec;

				if (spec.training_table == NULL)
					spec.training_table = tbl_str;
				if (spec.training_column == NULL)
					spec.training_column = targ_str;
				if (spec.parameters == NULL)
				{
					spec.parameters = gpu_hyperparams;
					gpu_hyperparams = NULL;
				}

				spec.algorithm = "linear_regression";
				spec.model_type = "regression";

				/*
				 * Ensure we're in a valid memory context before calling
				 * ml_catalog_register_model
				 */
				MemoryContextSwitchTo(oldcontext);

				model_id = ml_catalog_register_model(&spec);
#endif

				NDB_FREE(gpu_err);
				NDB_FREE(gpu_hyperparams);

				ndb_gpu_free_train_result(&gpu_result);
				linreg_dataset_free(&dataset);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);

				PG_RETURN_INT32(model_id);
			}
			else
			{
				if (gpu_err != NULL)
				{
					NDB_FREE(gpu_err);
				}
				NDB_FREE(gpu_hyperparams);

				ndb_gpu_free_train_result(&gpu_result);
				linreg_dataset_free(&dataset);
			}
		}
		else if (neurondb_gpu_is_available() && !ndb_gpu_kernel_enabled("linreg_train"))
		{
			/* GPU available but kernel not enabled, using CPU */
		}

		/* CPU training path using streaming accumulator */
		{
			LinRegStreamAccum stream_accum;
			double	  **XtX_inv = NULL;
			NDB_DECLARE(double *, beta);
			int			i,
						j;
			int			dim_with_intercept;
			NDB_DECLARE(LinRegModel *, model);
			bytea	   *model_blob;
			Jsonb	   *metrics_json;
			StringInfoData metricsbuf = {0};
			int			chunk_size;
			int			offset = 0;
			int			rows_in_chunk = 0;

			/* Use larger chunks for better performance */
			if (nvec > 1000000)
				chunk_size = 100000;	/* 100k chunks for very large datasets */
			else if (nvec > 100000)
				chunk_size = 50000; /* 50k chunks for large datasets */
			else
				chunk_size = 10000; /* 10k chunks for smaller datasets */

			/* Initialize streaming accumulator */
			linreg_stream_accum_init(&stream_accum, dim);
			dim_with_intercept = dim + 1;

			/* Process data in chunks */

			{
				NDB_DECLARE (NdbSpiSession *, stream_spi_session);
				MemoryContext stream_oldcontext;

				stream_oldcontext = CurrentMemoryContext;
				Assert(stream_oldcontext != NULL);
				NDB_SPI_SESSION_BEGIN(stream_spi_session, stream_oldcontext);

			offset = 0;
			while (offset < nvec)
			{
				linreg_stream_process_chunk(quoted_tbl,
											quoted_feat,
											quoted_target,
											&stream_accum,
											chunk_size,
											offset,
											&rows_in_chunk);

				if (rows_in_chunk == 0)
					break;

				offset += rows_in_chunk;

				/* Progress tracking for large datasets */
			}

				NDB_SPI_SESSION_END(stream_spi_session);
			}

			if (stream_accum.n_samples < 10)
			{
				linreg_stream_accum_free(&stream_accum);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: train_linear_regression: insufficient samples processed (%d)",
								stream_accum.n_samples),
						 errdetail("Processed %d samples, minimum required is 10", stream_accum.n_samples),
						 errhint("Ensure the training table contains at least 10 valid rows.")));
			}

		/* Allocate matrices for inversion */
		XtX_inv = NULL;
		NDB_ALLOC(XtX_inv, double *, dim_with_intercept);
		
		/* palloc0 already initializes to NULL, so XtX_inv[i] are all NULL */
		/* Allocate each row */
		for (i = 0; i < dim_with_intercept; i++)
		{
			/* XtX_inv[i] is already NULL from palloc0 above */
			NDB_ALLOC(XtX_inv[i], double, dim_with_intercept);
		}
			
		beta = NULL;
		NDB_ALLOC(beta, double, dim_with_intercept);

			/*
			 * Add small regularization (ridge) to diagonal to handle
			 * near-singular matrices
			 */
			{
				double		lambda =
					1e-6;		/* Small regularization parameter */

				for (i = 0; i < dim_with_intercept; i++)
				{
					stream_accum.XtX[i][i] += lambda;
				}
			}

			/* Invert X'X */
			if (!matrix_invert(stream_accum.XtX, dim_with_intercept, XtX_inv))
			{
				/* If still singular after regularization, try larger lambda */
				double		lambda = 1e-3;

				for (i = 0; i < dim_with_intercept; i++)
				{
					stream_accum.XtX[i][i] += lambda;
				}

				if (!matrix_invert(stream_accum.XtX, dim_with_intercept, XtX_inv))
				{
					for (i = 0; i < dim_with_intercept; i++)
						NDB_FREE(XtX_inv[i]);
					NDB_FREE(XtX_inv);
					NDB_FREE(beta);

					linreg_stream_accum_free(&stream_accum);
					NDB_FREE(tbl_str);
					NDB_FREE(feat_str);
					NDB_FREE(targ_str);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: linear_regression: matrix is singular even after regularization"),
							 errdetail("Feature matrix (X'X) cannot be inverted even with regularization"),
							 errhint("Try removing correlated features, reducing feature count, or using ridge regression")));
				}
				else
				{
					/* Used regularization to handle near-singular matrix */
				}
			}

			/* Compute β = (X'X)^(-1)X'y */
			for (i = 0; i < dim_with_intercept; i++)
			{
				beta[i] = 0.0;
				for (j = 0; j < dim_with_intercept; j++)
					beta[i] += XtX_inv[i][j] * stream_accum.Xty[j];
			}

			/* Build LinRegModel */
			NDB_ALLOC(model, LinRegModel, 1);

			model->n_features = dim;
			model->n_samples = stream_accum.n_samples;
			model->intercept = beta[0];
			NDB_ALLOC(model->coefficients, double, dim);

			for (i = 0; i < dim; i++)
				model->coefficients[i] = beta[i + 1];

			/*
			 * Compute metrics (R², MSE, MAE) using second pass through
			 * training data to compute exact residuals
			 */
			{
				double		ss_tot;
				double		ss_res = 0.0;
				double		mse = 0.0;
				double		mae = 0.0;
				int			metrics_chunk_size;
				int			metrics_offset = 0;

				/* Compute ss_tot from accumulator */
				if (stream_accum.n_samples > 0)
				{
					ss_tot = stream_accum.y_sq_sum - (stream_accum.y_sum * stream_accum.y_sum / stream_accum.n_samples);
				}
				else
				{
					ss_tot = 0.0;
				}

				/* Compute MSE and MAE by processing chunks for metrics */
				/* Limit metrics computation to avoid excessive time */
				metrics_chunk_size = (stream_accum.n_samples > 100000) ? 100000 : stream_accum.n_samples;

				elog(DEBUG1,
					 "neurondb: linear_regression: computing metrics on %d samples",
					 metrics_chunk_size);

				/* Metrics computation uses new SPI session */
				{
					NDB_DECLARE (NdbSpiSession *, metrics_spi_session);
					MemoryContext metrics_oldcontext;

					metrics_oldcontext = CurrentMemoryContext;
					Assert(metrics_oldcontext != NULL);
					NDB_SPI_SESSION_BEGIN(metrics_spi_session, metrics_oldcontext);

					while (metrics_offset < metrics_chunk_size)
					{
						StringInfoData metrics_query;
						int			metrics_ret;
						int			metrics_n_rows;
						TupleDesc	metrics_tupdesc;
						NDB_DECLARE(float *, metrics_row_buffer);
						int			metrics_i;

						initStringInfo(&metrics_query);
						elog(DEBUG1,
							 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d OFFSET %d",
							 quoted_feat,
							 quoted_target,
							 quoted_tbl,
							 quoted_feat,
							 quoted_target,
							 10000,
							 metrics_offset);
						appendStringInfo(&metrics_query,
										 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d OFFSET %d",
										 quoted_feat,
										 quoted_target,
										 quoted_tbl,
										 quoted_feat,
										 quoted_target,
										 10000,
										 metrics_offset);

						metrics_ret = ndb_spi_execute(metrics_spi_session, metrics_query.data, true, 0);
						NDB_CHECK_SPI_TUPTABLE();
						if (metrics_ret != SPI_OK_SELECT)
						{
							NDB_FREE(metrics_query.data);
							break;
						}
						NDB_CHECK_SPI_TUPTABLE();
						metrics_n_rows = SPI_processed;
						if (metrics_n_rows == 0)
						{
							NDB_FREE(metrics_query.data);
							break;
						}

						metrics_tupdesc = SPI_tuptable->tupdesc;
						NDB_ALLOC(metrics_row_buffer, float, dim);

						for (metrics_i = 0; metrics_i < metrics_n_rows && (metrics_offset + metrics_i) < metrics_chunk_size; metrics_i++)
						{
							HeapTuple	metrics_tuple = SPI_tuptable->vals[metrics_i];
							Datum		metrics_feat_datum;
							Datum		metrics_targ_datum;
							bool		metrics_feat_null;
							bool		metrics_targ_null;
							double		metrics_y_true;
							double		metrics_y_pred;
							double		metrics_error;
							Vector	   *metrics_vec;
							ArrayType  *metrics_arr;
							int			metrics_j;
							Oid			metrics_feat_type_oid = SPI_gettypeid(metrics_tupdesc, 1);
							bool		metrics_feat_is_array = (metrics_feat_type_oid == FLOAT8ARRAYOID || metrics_feat_type_oid == FLOAT4ARRAYOID);

							metrics_feat_datum = SPI_getbinval(metrics_tuple, metrics_tupdesc, 1, &metrics_feat_null);
							metrics_targ_datum = SPI_getbinval(metrics_tuple, metrics_tupdesc, 2, &metrics_targ_null);

							if (metrics_feat_null || metrics_targ_null)
								continue;

							/* Extract features */
							if (metrics_feat_is_array)
							{
								metrics_arr = DatumGetArrayTypeP(metrics_feat_datum);
								if (ARR_NDIM(metrics_arr) != 1 || ARR_DIMS(metrics_arr)[0] != dim)
									continue;
								if (metrics_feat_type_oid == FLOAT8ARRAYOID)
								{
									float8	   *data = (float8 *) ARR_DATA_PTR(metrics_arr);

									for (metrics_j = 0; metrics_j < dim; metrics_j++)
										metrics_row_buffer[metrics_j] = (float) data[metrics_j];
								}
								else
								{
									float4	   *data = (float4 *) ARR_DATA_PTR(metrics_arr);

									memcpy(metrics_row_buffer, data, sizeof(float) * dim);
								}
							}
							else
							{
								metrics_vec = DatumGetVector(metrics_feat_datum);
								if (metrics_vec->dim != dim)
									continue;
								memcpy(metrics_row_buffer, metrics_vec->data, sizeof(float) * dim);
							}

							/* Extract target */
							{
								Oid			metrics_targ_type = SPI_gettypeid(metrics_tupdesc, 2);

								if (metrics_targ_type == INT2OID || metrics_targ_type == INT4OID || metrics_targ_type == INT8OID)
									metrics_y_true = (double) DatumGetInt32(metrics_targ_datum);
								else
									metrics_y_true = DatumGetFloat8(metrics_targ_datum);
							}

							/* Compute prediction */
							metrics_y_pred = model->intercept;
							for (metrics_j = 0; metrics_j < dim; metrics_j++)
								metrics_y_pred += model->coefficients[metrics_j] * metrics_row_buffer[metrics_j];

							/* Accumulate errors */
							metrics_error = metrics_y_true - metrics_y_pred;
							mse += metrics_error * metrics_error;
							mae += fabs(metrics_error);
							ss_res += metrics_error * metrics_error;
						}

						NDB_FREE(metrics_row_buffer);
						NDB_FREE(metrics_query.data);

						metrics_offset += metrics_n_rows;
						if (metrics_offset >= metrics_chunk_size)
							break;
					}

					NDB_SPI_SESSION_END(metrics_spi_session);
				}

				/* Normalize metrics */
				if (metrics_chunk_size > 0)
				{
					mse /= metrics_chunk_size;
					mae /= metrics_chunk_size;
				}

				/* Compute R² */
				if (ss_tot > 1e-10)
					model->r_squared = 1.0 - (ss_res / ss_tot);
				else
					model->r_squared = 0.0;
				model->mse = mse;
				model->mae = mae;
			}

			/* Validate model before serialization */
			if (model->n_features <= 0 || model->n_features > 10000)
			{
				NDB_FREE(model->coefficients);
				NDB_FREE(model);

				for (i = 0; i < dim_with_intercept; i++)
					NDB_FREE(XtX_inv[i]);
				NDB_FREE(XtX_inv);
				NDB_FREE(beta);

				linreg_stream_accum_free(&stream_accum);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);

				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: train_linear_regression: model.n_features is invalid (%d) before serialization",
								model->n_features)));
			}


			/* Serialize model with training_backend=0 (CPU) */
			model_blob = linreg_model_serialize(model, 0);

			/*
			 * Note: GPU packing is disabled for CPU-trained models to avoid
			 * format conflicts. GPU packing should only be used when the
			 * model was actually trained on GPU. CPU models must use CPU
			 * serialization format for proper deserialization.
			 */

			/* Build metrics JSON */
			initStringInfo(&metricsbuf);
			appendStringInfo(&metricsbuf,
							 "{\"algorithm\":\"linear_regression\","
							 "\"training_backend\":0,"
							 "\"n_features\":%d,"
							 "\"n_samples\":%d,"
							 "\"r_squared\":%.6f,"
							 "\"mse\":%.6f,"
							 "\"mae\":%.6f}",
							 model->n_features,
							 model->n_samples,
							 model->r_squared,
							 model->mse,
							 model->mae);

			/* Use ndb_jsonb_in_cstring like test 006 fix */
			metrics_json = ndb_jsonb_in_cstring(metricsbuf.data);
			if (metrics_json == NULL)
			{
				NDB_FREE(metricsbuf.data);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("neurondb: failed to parse metrics JSON")));
			}
			NDB_FREE(metricsbuf.data);
			metricsbuf.data = NULL;

			/* Register in catalog */
			{
				MLCatalogModelSpec spec;

				/*
				 * Ensure we're in a valid memory context before calling
				 * ml_catalog_register_model
				 */

				/*
				 * This function may have been called after SPI_finish() or
				 * other context changes
				 */
				MemoryContextSwitchTo(oldcontext);

				memset(&spec, 0, sizeof(MLCatalogModelSpec));
				spec.algorithm = "linear_regression";
				spec.model_type = "regression";
				spec.training_table = tbl_str;
				spec.training_column = targ_str;
				spec.model_data = model_blob;
				spec.metrics = metrics_json;

				model_id = ml_catalog_register_model(&spec);
			}

			for (i = 0; i < dim_with_intercept; i++)
				NDB_FREE(XtX_inv[i]);
			NDB_FREE(XtX_inv);
			NDB_FREE(beta);
			NDB_FREE(model->coefficients);
			NDB_FREE(model);

			linreg_stream_accum_free(&stream_accum);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);

			PG_RETURN_INT32(model_id);
		}
	}
}

/*
 * predict_linear_regression_model_id
 *
 * Makes predictions using trained linear regression model from catalog
 */
PG_FUNCTION_INFO_V1(predict_linear_regression_model_id);

Datum
predict_linear_regression_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	Vector	   *features;
	NDB_DECLARE(LinRegModel *, model);
	double		prediction;
	int			i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: model_id is required"),
				 errdetail("First argument (model_id) is NULL"),
				 errhint("Provide a valid model_id from neurondb.ml_models table.")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: features vector is required"),
				 errdetail("Second argument (features) is NULL"),
				 errhint("Provide a valid vector or array of features matching the model's feature dimension.")));

	features = PG_GETARG_VECTOR_P(1);

	/* Try GPU prediction first */
	if (linreg_try_gpu_predict_catalog(model_id, features, &prediction))
	{
		PG_RETURN_FLOAT8(prediction);
	}

	/* Load model from catalog */
	if (!linreg_load_model_from_catalog(model_id, &model))
	{
		/* Check if model is GPU-only */
		NDB_DECLARE (bytea *, payload);
		NDB_DECLARE (Jsonb *, metrics);
		bool		is_gpu = false;

		if (ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		{
			/* Check if this is a GPU model using training_backend */
			is_gpu = false;
			
			if (metrics != NULL)
			{
				JsonbIterator *it;
				JsonbValue	v;
				JsonbIteratorToken r;

				it = JsonbIteratorInit((JsonbContainer *) &metrics->root);
				while ((r = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
				{
					if (r == WJB_KEY && v.type == jbvString)
					{
						char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

						if (strcmp(key, "training_backend") == 0)
						{
							r = JsonbIteratorNext(&it, &v, true);
							if (r == WJB_VALUE && v.type == jbvNumeric)
							{
								int			backend = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
								is_gpu = (backend == 1);
							}
						}
						NDB_FREE(key);
					}
				}
			}
			
			NDB_FREE(payload);
			NDB_FREE(metrics);
		}

		if (is_gpu)
		{
			elog(WARNING,
				 "neurondb: linear_regression: model %d is GPU-only and GPU prediction failed",
				 model_id);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: model %d is GPU-only and GPU prediction failed", model_id),
					 errdetail("Model was trained on GPU but GPU is not available or prediction failed"),
					 errhint("Ensure GPU is available and properly configured, or retrain the model on CPU.")));
		}
		else
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: model %d not found", model_id),
					 errdetail("Model does not exist in neurondb.ml_models catalog"),
					 errhint("Verify the model_id is correct and the model exists in the catalog.")));
	}

	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: feature dimension mismatch (expected %d, got %d)",
						model->n_features,
						features->dim),
				 errdetail("Model was trained with %d features, but input vector has %d dimensions", model->n_features, features->dim),
				 errhint("Ensure the feature vector has the same dimension as the training data.")));

	/* Compute prediction: y = intercept + coef1*x1 + coef2*x2 + ... */
	prediction = model->intercept;
	for (i = 0; i < model->n_features && i < features->dim; i++)
		prediction += model->coefficients[i] * features->data[i];

	if (model != NULL)
	{
		NDB_FREE(model->coefficients);
		NDB_FREE(model);
	}

	PG_RETURN_FLOAT8(prediction);
}

/*
 * predict_linear_regression
 *
 * Makes predictions using trained linear regression coefficients (legacy)
 */
PG_FUNCTION_INFO_V1(predict_linear_regression);

Datum
predict_linear_regression(PG_FUNCTION_ARGS)
{
	ArrayType  *coef_array;
	Vector	   *features;
	int			ncoef;
	float8	   *coef;
	float	   *x;
	int			dim;
	double		prediction;
	int			i;

	coef_array = PG_GETARG_ARRAYTYPE_P(0);
	features = PG_GETARG_VECTOR_P(1);

	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: coefficients must be 1-dimensional array"),
				 errdetail("Array has %d dimensions, expected 1", ARR_NDIM(coef_array)),
				 errhint("Provide a 1-dimensional array of coefficients.")));

	ncoef = ARR_DIMS(coef_array)[0];
	coef = (float8 *) ARR_DATA_PTR(coef_array);

	x = features->data;
	dim = features->dim;

	if (ncoef != dim + 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: coefficient dimension mismatch: expected %d, got %d",
						dim + 1,
						ncoef),
				 errdetail("Coefficient array has %d elements, expected %d (dimension + 1 for intercept)", ncoef, dim + 1),
				 errhint("Provide coefficient array with dimension + 1 elements (intercept + feature coefficients).")));

	/* Compute prediction: y = β0 + β1*x1 + β2*x2 + ... */
	prediction = coef[0];		/* intercept */
	for (i = 0; i < dim; i++)
		prediction += coef[i + 1] * x[i];

	PG_RETURN_FLOAT8(prediction);
}

/*
 * evaluate_linear_regression
 *
 * Evaluates model performance (R², MSE, MAE)
 * Returns: [r_squared, mse, mae, rmse]
 */
PG_FUNCTION_INFO_V1(evaluate_linear_regression);

Datum
evaluate_linear_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *target_col;
	ArrayType  *coef_array;
	NDB_DECLARE (char *, tbl_str);
	NDB_DECLARE (char *, feat_str);
	NDB_DECLARE (char *, targ_str);
	StringInfoData query = {0};
	int			ret;
	int			nvec = 0;
	int			ncoef;
	float8	   *coef;
	double		mse = 0.0;
	double		mae = 0.0;
	double		ss_tot = 0.0;
	double		ss_res = 0.0;
	double		y_mean = 0.0;
	double		r_squared;
	double		rmse;
	int			i;
	NDB_DECLARE (Datum *, result_datums);
	ArrayType  *result_array;
	MemoryContext oldcontext;
	NDB_DECLARE (NdbSpiSession *, eval_spi_session);

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	coef_array = PG_GETARG_ARRAYTYPE_P(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: coefficients must be 1-dimensional array"),
				 errdetail("Array has %d dimensions, expected 1", ARR_NDIM(coef_array)),
				 errhint("Provide a 1-dimensional array of coefficients.")));

	ncoef = ARR_DIMS(coef_array)[0];
	(void) ncoef;				/* Suppress unused variable warning */
	coef = (float8 *) ARR_DATA_PTR(coef_array);

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);

	NDB_SPI_SESSION_BEGIN(eval_spi_session, oldcontext);

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str,
					 targ_str,
					 tbl_str,
					 feat_str,
					 targ_str);

	ret = ndb_spi_execute(eval_spi_session, query.data, true, 0);
	NDB_FREE(query.data);
	query.data = NULL;
	if (ret != SPI_OK_SELECT)
	{
		NDB_SPI_SESSION_END(eval_spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_linear_regression: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table and columns exist and are accessible.")));
	}

	nvec = SPI_processed;

	/* First pass: compute mean of y */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		targ_datum;
		bool		targ_null;

		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (!targ_null)
			y_mean += DatumGetFloat8(targ_datum);
	}
	y_mean /= nvec;

	/* Second pass: compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		double		y_true;
		double		y_pred;
		double		error;
		int			j;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		vec = DatumGetVector(feat_datum);
		y_true = DatumGetFloat8(targ_datum);

		/* Compute prediction */
		y_pred = coef[0];		/* intercept */
		for (j = 0; j < vec->dim; j++)
			y_pred += coef[j + 1] * vec->data[j];

		/* Compute errors */
		error = y_true - y_pred;
		mse += error * error;
		mae += fabs(error);
		ss_res += error * error;
		ss_tot += (y_true - y_mean) * (y_true - y_mean);
	}

	mse /= nvec;
	mae /= nvec;
	rmse = sqrt(mse);
	r_squared = 1.0 - (ss_res / ss_tot);

	NDB_SPI_SESSION_END(eval_spi_session);

	/* Build result array: [r_squared, mse, mae, rmse] */
	MemoryContextSwitchTo(oldcontext);

	NDB_ALLOC(result_datums, Datum, 4);

	result_datums[0] = Float8GetDatum(r_squared);
	result_datums[1] = Float8GetDatum(mse);
	result_datums[2] = Float8GetDatum(mae);
	result_datums[3] = Float8GetDatum(rmse);

	result_array = construct_array(result_datums,
								   4,
								   FLOAT8OID,
								   sizeof(float8),
								   FLOAT8PASSBYVAL,
								   'd');

	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);
	NDB_FREE(targ_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

Jsonb *
evaluate_linear_regression_by_model_id_jsonb(int32 model_id, text * table_name, text * feature_col, text * label_col)
{
	NDB_DECLARE (LinRegModel *, model);
	NDB_DECLARE (char *, tbl_str);
	NDB_DECLARE (char *, feat_str);
	NDB_DECLARE (char *, targ_str);
	StringInfoData query = {0};
	StringInfoData jsonbuf = {0};
	int			ret;
	int			nvec = 0;
	
	double		mse = 0.0;
	double		mae = 0.0;
	double		ss_tot = 0.0;
	double		ss_res = 0.0;
	double		y_mean = 0.0;

	double		r_squared;
	double		rmse;
	int			i;
	NDB_DECLARE(Jsonb *, result);
	MemoryContext oldcontext;

	Oid			feat_type_oid;
	bool		feat_is_array;

	NDB_DECLARE (bytea *, gpu_payload);
	NDB_DECLARE (Jsonb *, gpu_metrics);
	bool		is_gpu_model = false;

	MemoryContext currctx;
	NDB_DECLARE (double *, model_coefficients_ptr);
	NDB_DECLARE (NdbSpiSession *, eval_jsonb_spi_session);

#ifdef NDB_GPU_CUDA
	NDB_DECLARE (float *, h_features);
	NDB_DECLARE (double *, h_targets);
	int			valid_rows = 0;
#endif


	/* Load model from catalog - try CPU first, then GPU */
	if (!linreg_load_model_from_catalog(model_id, &model))
	{
		/* CPU model load failed - this might indicate a GPU-trained model or corrupted data */
		/* Return NULL instead of trying to create JSONB to avoid corruption */
		return NULL;
	}

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);

	NDB_SPI_SESSION_BEGIN(eval_jsonb_spi_session, oldcontext);

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str,
					 targ_str,
					 tbl_str,
					 feat_str,
					 targ_str);

	ret = ndb_spi_execute(eval_jsonb_spi_session, query.data, true, 0);
	NDB_FREE(query.data);
	query.data = NULL;
	if (ret != SPI_OK_SELECT)
	{
		NDB_SPI_SESSION_END(eval_jsonb_spi_session);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		if (model != NULL)
		{
			NDB_FREE(model->coefficients);
			NDB_FREE(model);
		}
		NDB_FREE(gpu_payload);
		NDB_FREE(gpu_metrics);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_linear_regression_by_model_id_jsonb: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table and columns exist and are accessible.")));
	}

	nvec = SPI_processed;
	if (nvec < 2)
	{
		NDB_SPI_SESSION_END(eval_jsonb_spi_session);
		if (model != NULL)
		{
			NDB_FREE(model->coefficients);
			NDB_FREE(model);
		}
		NDB_FREE(gpu_payload);
		NDB_FREE(gpu_metrics);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_linear_regression_by_model_id_jsonb: need at least 2 samples, got %d",
						nvec),
				 errdetail("Dataset contains %d rows, minimum required is 2 for evaluation", nvec),
				 errhint("Add more data rows to the evaluation table.")));
	}

	/* Determine feature type from first row */
	feat_type_oid = InvalidOid;
	feat_is_array = false;
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	{
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
		if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			feat_is_array = true;
	}

	/*
	 * GPU batch evaluation path for GPU models - uses optimized evaluation
	 * kernel
	 */
	if (is_gpu_model && neurondb_gpu_is_available())
	{
#ifdef NDB_GPU_CUDA
		const		NdbCudaLinRegModelHeader *gpu_hdr;
		int			feat_dim = 0;

		/* Load GPU model header */
		if (VARSIZE(gpu_payload) - VARHDRSZ < sizeof(NdbCudaLinRegModelHeader))
		{
			NDB_SPI_SESSION_END(eval_jsonb_spi_session);
			NDB_FREE(gpu_payload);
			NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_linear_regression_by_model_id_jsonb: GPU payload too small"),
					 errdetail("Payload size is %u bytes, minimum required is %zu bytes", VARSIZE(gpu_payload) - VARHDRSZ, sizeof(NdbCudaLinRegModelHeader)),
					 errhint("Model data may be corrupted. Verify the model was stored correctly.")));
		}

		gpu_hdr = (const NdbCudaLinRegModelHeader *) VARDATA(gpu_payload);
		feat_dim = gpu_hdr->feature_dim;

		/* Allocate host buffers for features and targets */
		NDB_ALLOC(h_features, float, (size_t) nvec * (size_t) feat_dim);
		NDB_ALLOC(h_targets, double, nvec);

		/*
		 * Extract features and targets from SPI results - optimized batch
		 * extraction
		 */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple = SPI_tuptable->vals[i];
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				Vector	   *vec;
				ArrayType  *arr;
				float	   *feat_row;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				feat_row = h_features + (valid_rows * feat_dim);
				h_targets[valid_rows] = DatumGetFloat8(targ_datum);
				y_mean += h_targets[valid_rows];

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);
						int			j;
						int			j_remain = feat_dim % 4;
						int			j_end = feat_dim - j_remain;

						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float) data[j];
							feat_row[j + 1] = (float) data[j + 1];
							feat_row[j + 2] = (float) data[j + 2];
							feat_row[j + 3] = (float) data[j + 3];
						}
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float) data[j];
					}
					else
					{
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);

						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows < 2)
		{
			NDB_FREE(h_features);
			NDB_FREE(h_targets);
			NDB_FREE(gpu_payload);
			NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			NDB_SPI_SESSION_END(eval_jsonb_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_linear_regression_by_model_id_jsonb: need at least 2 valid samples, got %d",
							valid_rows),
					 errdetail("Processed %d valid rows, minimum required is 2", valid_rows),
					 errhint("Ensure the evaluation table contains at least 2 valid feature-target pairs.")));
		}

		y_mean /= valid_rows;

		/* Use optimized GPU evaluation kernel */
		{
			double		gpu_mse = 0.0;
			double		gpu_mae = 0.0;
			double		gpu_rmse = 0.0;
			double		gpu_r_squared = 0.0;
			NDB_DECLARE (char *, gpu_err);
			int			eval_rc;

			eval_rc = ndb_cuda_linreg_evaluate(gpu_payload,
											   h_features,
											   h_targets,
											   valid_rows,
											   feat_dim,
											   &gpu_mse,
											   &gpu_mae,
											   &gpu_rmse,
											   &gpu_r_squared,
											   &gpu_err);

			if (eval_rc == 0)
			{
				/* GPU evaluation succeeded */
				mse = gpu_mse;
				mae = gpu_mae;
				rmse = gpu_rmse;
				r_squared = gpu_r_squared;
				nvec = valid_rows;

				NDB_FREE(h_features);
				NDB_FREE(h_targets);
				NDB_FREE(gpu_payload);
				NDB_FREE(gpu_metrics);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);

				NDB_SPI_SESSION_END(eval_jsonb_spi_session);

				/* Build result JSON */
				MemoryContextSwitchTo(oldcontext);
				{
					initStringInfo(&jsonbuf);
					appendStringInfo(&jsonbuf,
									 "{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
									 mse, mae, rmse, r_squared, nvec);

					/* Use ndb_jsonb_in_cstring to parse JSON string to JSONB */
					result = ndb_jsonb_in_cstring(jsonbuf.data);
					NDB_FREE(jsonbuf.data);
					jsonbuf.data = NULL;
					
					if (result == NULL)
					{
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
								 errmsg("neurondb: failed to parse GPU evaluation metrics JSON")));
					}
					
					return result;
				}
			}
			else
			{
				/* GPU evaluation failed, fall back to CPU */
				elog(WARNING,
					 "neurondb: evaluate_linear_regression_by_model_id_jsonb: GPU evaluation kernel failed: %s, falling back to CPU",
					 gpu_err ? gpu_err : "unknown error");
				NDB_FREE(gpu_err);
				NDB_FREE(h_features);
				NDB_FREE(h_targets);

				/* Fall through to CPU path */
			}
		}
#endif							/* NDB_GPU_CUDA */
	}

	/* CPU evaluation path */
	/* Handle GPU model fallback - convert GPU model to CPU format */
	if (is_gpu_model && gpu_payload != NULL && model == NULL)
	{
		const		NdbCudaLinRegModelHeader *gpu_hdr;
		const float *gpu_coefficients;
		NDB_DECLARE (double *, cpu_coefficients);
		double		cpu_intercept = 0.0;
		int			feat_dim;

		if (VARSIZE(gpu_payload) - VARHDRSZ < sizeof(NdbCudaLinRegModelHeader))
		{
			NDB_SPI_SESSION_END(eval_jsonb_spi_session);
			NDB_FREE(gpu_payload);
			NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_linear_regression_by_model_id_jsonb: GPU payload too small for CPU fallback"),
					 errdetail("Payload size is %u bytes, minimum required is %zu bytes", VARSIZE(gpu_payload) - VARHDRSZ, sizeof(NdbCudaLinRegModelHeader)),
					 errhint("Model data may be corrupted. Verify the model was stored correctly.")));
		}

		gpu_hdr = (const NdbCudaLinRegModelHeader *) VARDATA(gpu_payload);
		gpu_coefficients = (const float *) ((const char *) gpu_hdr + sizeof(NdbCudaLinRegModelHeader));
		feat_dim = gpu_hdr->feature_dim;
		cpu_intercept = gpu_hdr->intercept;

		/* Convert coefficients from float to double */
		NDB_ALLOC(cpu_coefficients, double, feat_dim);

		for (i = 0; i < feat_dim; i++)
			cpu_coefficients[i] = (double) gpu_coefficients[i];

		/* Create temporary CPU model structure */
		NDB_ALLOC(model, LinRegModel, 1);

		model->n_features = feat_dim;
		model->intercept = cpu_intercept;
		model->coefficients = cpu_coefficients;
	}

	/* Ensure model is available for CPU evaluation */
	if (model == NULL)
	{
		NDB_SPI_SESSION_END(eval_jsonb_spi_session);
		NDB_FREE(gpu_payload);
		NDB_FREE(gpu_metrics);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_linear_regression_by_model_id_jsonb: CPU model evaluation requires model data"),
				 errdetail("Model structure is NULL after loading from catalog"),
				 errhint("Verify the model exists and can be deserialized correctly.")));
	}

	/* Store coefficients pointer while model is known to be valid */
	if (model != NULL)
		model_coefficients_ptr = model->coefficients;

	/* First pass: compute mean of y */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		targ_datum;
		bool		targ_null;

		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (!targ_null)
			y_mean += DatumGetFloat8(targ_datum);
	}
	y_mean /= nvec;

	/* Second pass: compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec;
		ArrayType  *arr;
		double		y_true;
		double		y_pred;
		double		error;
		int			j;
		int			actual_dim;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		y_true = DatumGetFloat8(targ_datum);

		/* Extract features and determine dimension */
		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
			if (ARR_NDIM(arr) != 1)
			{
				NDB_SPI_SESSION_END(eval_jsonb_spi_session);
				if (model != NULL)
				{
					NDB_FREE(model->coefficients);
					NDB_FREE(model);
				}
				NDB_FREE(gpu_payload);
				NDB_FREE(gpu_metrics);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: features array must be 1-D"),
						 errdetail("Array has %d dimensions, expected 1", ARR_NDIM(arr)),
						 errhint("Ensure feature column contains 1-dimensional arrays only.")));
			}
			actual_dim = ARR_DIMS(arr)[0];
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			actual_dim = vec->dim;
		}

		/* Validate feature dimension */
		if (actual_dim != model->n_features)
		{
			NDB_SPI_SESSION_END(eval_jsonb_spi_session);
			if (model != NULL)
			{
				NDB_FREE(model->coefficients);
				NDB_FREE(model);
			}
			NDB_FREE(gpu_payload);
			NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: feature dimension mismatch (expected %d, got %d)",
							model->n_features,
							actual_dim),
					 errdetail("Model was trained with %d features, but input has %d features", model->n_features, actual_dim),
					 errhint("Ensure the feature column has the same dimension as the training data.")));
		}

		/* Compute prediction using model */
		y_pred = model->intercept;

		if (feat_is_array)
		{
			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				double	   *feat_data = (double *) ARR_DATA_PTR(arr);

				for (j = 0; j < model->n_features; j++)
					y_pred += model->coefficients[j] * feat_data[j];
			}
			else
			{
				float	   *feat_data = (float *) ARR_DATA_PTR(arr);

				for (j = 0; j < model->n_features; j++)
					y_pred += model->coefficients[j] * (double) feat_data[j];
			}
		}
		else
		{
			for (j = 0; j < model->n_features && j < vec->dim; j++)
				y_pred += model->coefficients[j] * vec->data[j];
		}

		/* Compute errors */
		error = y_true - y_pred;
		mse += error * error;
		mae += fabs(error);
		ss_res += error * error;
		ss_tot += (y_true - y_mean) * (y_true - y_mean);
	}

	mse /= nvec;
	mae /= nvec;
	rmse = sqrt(mse);

	/*
	 * Handle R² calculation - if ss_tot is zero (no variance in y), R² is
	 * undefined
	 */
	if (ss_tot == 0.0)
		r_squared = 0.0;		/* Convention: set to 0 when there's no
								 * variance to explain */
	else
		r_squared = 1.0 - (ss_res / ss_tot);

	/* End SPI session BEFORE creating JSONB to avoid memory context issues */
	NDB_SPI_SESSION_END(eval_jsonb_spi_session);
	
	/* Switch to oldcontext to create result */
	MemoryContextSwitchTo(oldcontext);

	/* Build result JSON string */
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
					 mse, mae, rmse, r_squared, nvec);

	/* Use ndb_jsonb_in_cstring */
	result = ndb_jsonb_in_cstring(jsonbuf.data);
	NDB_FREE(jsonbuf.data);
	jsonbuf.data = NULL;
	
	if (result == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("neurondb: failed to parse metrics JSON")));
	}

	if (result == NULL)
	{
		NDB_FREE(jsonbuf.data);
		jsonbuf.data = NULL;
		NDB_FREE(model_coefficients_ptr);
		NDB_FREE(model);
		NDB_FREE(gpu_payload);
		NDB_FREE(gpu_metrics);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_linear_regression_by_model_id_jsonb: JSONB result is NULL")));
	}

	NDB_FREE(jsonbuf.data);
	jsonbuf.data = NULL;

	/*
	 * Bulletproof cleanup: ensure correct context and pointer integrity
	 * before freeing
	 */
	currctx = CurrentMemoryContext;
	if (oldcontext && currctx != oldcontext)
		MemoryContextSwitchTo(oldcontext);

	/* Cleanup - NDB_FREE handles NULL checks */
	NDB_FREE(model_coefficients_ptr);
	NDB_FREE(model);
	NDB_FREE(gpu_payload);
	NDB_FREE(gpu_metrics);
	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);
	NDB_FREE(targ_str);

	return result;
}

/*
 * evaluate_linear_regression_by_model_id
 *
 * One-shot evaluation function: loads model, fetches all data in one query,
 * loops through rows in C, computes predictions and metrics, returns jsonb.
 * This is much more efficient than calling predict() for each row in SQL.
 */
PG_FUNCTION_INFO_V1(evaluate_linear_regression_by_model_id);

Datum
evaluate_linear_regression_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	Jsonb	   *result;

	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_linear_regression_by_model_id: 4 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_linear_regression_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_linear_regression_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	result = evaluate_linear_regression_by_model_id_jsonb(model_id, table_name, feature_col, label_col);
	if (result == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_linear_regression_by_model_id: result is NULL")));

	PG_RETURN_JSONB_P(result);
}

/* GPU Model State */
typedef struct LinRegGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			feature_dim;
	int			n_samples;
}			LinRegGpuModelState;

static void
linreg_gpu_release_state(LinRegGpuModelState * state)
{
	Assert(state != NULL);
	if (state == NULL)
		return;
	NDB_FREE(state->model_blob);
	NDB_FREE(state->metrics);
	NDB_FREE(state);
}

static bool
linreg_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	NDB_DECLARE (LinRegGpuModelState *, state);
	NDB_DECLARE (bytea *, payload);
	NDB_DECLARE (Jsonb *, metrics);
	int			rc;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
		return false;
	if (!neurondb_gpu_is_available())
		return false;
	if (spec->feature_matrix == NULL || spec->label_vector == NULL)
		return false;
	if (spec->sample_count <= 0 || spec->feature_dim <= 0)
		return false;

	payload = NULL;
	metrics = NULL;

	rc = ndb_gpu_linreg_train(spec->feature_matrix,
							  spec->label_vector,
							  spec->sample_count,
							  spec->feature_dim,
							  spec->hyperparameters,
							  &payload,
							  &metrics,
							  errstr);
		if (rc != 0 || payload == NULL)
		{
			NDB_FREE(payload);
			NDB_FREE(metrics);
			return false;
		}

	if (model->backend_state != NULL)
	{
		linreg_gpu_release_state((LinRegGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	NDB_ALLOC(state, LinRegGpuModelState, 1);

	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	if (metrics != NULL)
	{
		state->metrics = (Jsonb *) PG_DETOAST_DATUM_COPY(
														 PointerGetDatum(metrics));
	}
	else
	{
		state->metrics = NULL;
	}

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
linreg_gpu_predict(const MLGpuModel * model,
				   const float *input,
				   int input_dim,
				   float *output,
				   int output_dim,
				   char **errstr)
{
	const		LinRegGpuModelState *state;
	double		prediction;
	int			rc;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const LinRegGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	rc = ndb_gpu_linreg_predict(state->model_blob,
								input,
								state->feature_dim > 0 ? state->feature_dim : input_dim,
								&prediction,
								errstr);
	if (rc != 0)
		return false;

	output[0] = (float) prediction;

	return true;
}

static bool
linreg_gpu_evaluate(const MLGpuModel * model,
					const MLGpuEvalSpec * spec,
					MLGpuMetrics * out,
					char **errstr)
{
	const		LinRegGpuModelState *state;
	Jsonb	   *metrics_json;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const LinRegGpuModelState *) model->backend_state;

	{
		StringInfoData buf = {0};
		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"linear_regression\","
						 "\"storage\":\"gpu\","
						 "\"n_features\":%d,"
						 "\"n_samples\":%d}",
						 state->feature_dim > 0 ? state->feature_dim : 0,
						 state->n_samples > 0 ? state->n_samples : 0);

		/* Use ndb_jsonb_in_cstring like test 006 fix */
		metrics_json = ndb_jsonb_in_cstring(buf.data);
		if (metrics_json == NULL)
		{
			NDB_FREE(buf.data);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("neurondb: failed to parse metrics JSON")));
		}
		NDB_FREE(buf.data);
		buf.data = NULL;
	}

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
linreg_gpu_serialize(const MLGpuModel * model,
					 bytea * *payload_out,
					 Jsonb * *metadata_out,
					 char **errstr)
{
	const		LinRegGpuModelState *state;
#ifdef NDB_GPU_CUDA
	LinRegModel	linreg_model;
	bytea	   *unified_payload;
	char	   *base;
	NdbCudaLinRegModelHeader *hdr;
	float	   *coef_src_float;
	int			i;
#endif

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const LinRegGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

#ifdef NDB_GPU_CUDA
	/* Convert GPU format to unified format */
	unified_payload = NULL;

	base = VARDATA(state->model_blob);
	hdr = (NdbCudaLinRegModelHeader *) base;
	coef_src_float = (float *) (base + sizeof(NdbCudaLinRegModelHeader));

	/* Build LinRegModel structure */
	memset(&linreg_model, 0, sizeof(LinRegModel));
	linreg_model.n_features = hdr->feature_dim;
	linreg_model.n_samples = hdr->n_samples;
	linreg_model.intercept = (double) hdr->intercept;
	linreg_model.r_squared = hdr->r_squared;
	linreg_model.mse = hdr->mse;
	linreg_model.mae = hdr->mae;

	/* Convert float coefficients to double */
	if (linreg_model.n_features > 0)
	{
		NDB_ALLOC(linreg_model.coefficients, double, linreg_model.n_features);
		for (i = 0; i < linreg_model.n_features; i++)
			linreg_model.coefficients[i] = (double) coef_src_float[i];
	}

	/* Serialize using unified format with training_backend=1 (GPU) */
	unified_payload = linreg_model_serialize(&linreg_model, 1);

	if (linreg_model.coefficients != NULL)
	{
		NDB_FREE(linreg_model.coefficients);
		linreg_model.coefficients = NULL;
	}

	if (payload_out != NULL)
		*payload_out = unified_payload;
	else if (unified_payload != NULL)
		NDB_FREE(unified_payload);

	/* Update metrics to use training_backend=1 */
	if (metadata_out != NULL)
	{
		StringInfoData metrics_buf;
		initStringInfo(&metrics_buf);
		appendStringInfo(&metrics_buf,
						 "{\"algorithm\":\"linear_regression\","
						 "\"training_backend\":1,"
						 "\"n_features\":%d,"
						 "\"n_samples\":%d}",
						 state->feature_dim > 0 ? state->feature_dim : 0,
						 state->n_samples > 0 ? state->n_samples : 0);

		/* Use ndb_jsonb_in_cstring like test 006 fix */
		*metadata_out = ndb_jsonb_in_cstring(metrics_buf.data);
		if (*metadata_out == NULL)
		{
			NDB_FREE(metrics_buf.data);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("neurondb: failed to parse metadata JSON")));
		}
		NDB_FREE(metrics_buf.data);
	}

	return true;
#else
	/* For non-CUDA builds, GPU serialization is not supported */
	if (errstr != NULL)
		*errstr = pstrdup("linreg_gpu_serialize: CUDA not available");
	return false;
#endif
}

static bool
linreg_gpu_deserialize(MLGpuModel * model,
					   const bytea * payload,
					   const Jsonb * metadata,
					   char **errstr)
{
	NDB_DECLARE (LinRegGpuModelState *, state);
	NDB_DECLARE (bytea *, payload_copy);
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	NDB_ALLOC(payload_copy, bytea, payload_size);

	memcpy(payload_copy, payload, payload_size);

	NDB_ALLOC(state, LinRegGpuModelState, 1);

	state->model_blob = payload_copy;
	state->feature_dim = -1;
	state->n_samples = -1;

	if (model->backend_state != NULL)
		linreg_gpu_release_state((LinRegGpuModelState *) model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static void
linreg_gpu_destroy(MLGpuModel * model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		linreg_gpu_release_state((LinRegGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps linreg_gpu_model_ops = {
	.algorithm = "linear_regression",
	.train = linreg_gpu_train,
	.predict = linreg_gpu_predict,
	.evaluate = linreg_gpu_evaluate,
	.serialize = linreg_gpu_serialize,
	.deserialize = linreg_gpu_deserialize,
	.destroy = linreg_gpu_destroy,
};

void
neurondb_gpu_register_linreg_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&linreg_gpu_model_ops);
	registered = true;

	return;
}
