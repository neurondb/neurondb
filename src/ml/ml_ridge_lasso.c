/*-------------------------------------------------------------------------
 *
 * ml_ridge_lasso.c
 *    Regularized linear regression routines.
 *
 * This module implements L2 (Ridge) and L1 (Lasso) regression, including
 * model training, prediction, and catalog-ready serialization.
 *
 * Training uses streaming accumulation to build X'X and X'y incrementally,
 * avoiding full in-memory materialization of large datasets. All model
 * structures follow the standard binary layout used for persistent storage.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_ridge_lasso.c
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
#include "utils/jsonb.h"
#include "common/jsonapi.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_json.h"
#include "ml_ridge_regression_internal.h"
#include "ml_lasso_regression_internal.h"
#include "ml_catalog.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_ridge_regression.h"
#include "ml_gpu_lasso_regression.h"
#include "neurondb_cuda_ridge.h"
#include "neurondb_cuda_lasso.h"
#include "neurondb_validation.h"
#include "neurondb_gpu_model.h"
#include "neurondb_constants.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
#include "neurondb_sql.h"
#include "utils/elog.h"
#include "utils/lsyscache.h"
#include "neurondb_guc.h"

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
extern int	ndb_cuda_ridge_evaluate(const bytea * model_data,
									const float *features,
									const double *targets,
									int n_samples,
									int feature_dim,
									double *mse_out,
									double *mae_out,
									double *rmse_out,
									double *r_squared_out,
									char **errstr);
extern int	ndb_cuda_lasso_evaluate(const bytea * model_data,
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
#endif

#include <math.h>
#include <float.h>

typedef struct RidgeDataset
{
	float	   *features;
	double	   *targets;
	int			n_samples;
	int			feature_dim;
}			RidgeDataset;

typedef RidgeDataset LassoDataset;

/*
 * Streaming accumulator structure for incremental computation of X'X and X'y
 * matrices required for Ridge regression. This structure enables training on
 * datasets that exceed available memory by processing data in chunks and
 * accumulating statistical summaries incrementally. The X'X matrix represents
 * the covariance matrix of features scaled by the number of samples, while
 * X'y represents the correlation between features and targets. By computing
 * these matrices incrementally, the training algorithm can handle arbitrarily
 * large datasets without materializing the entire feature matrix in memory.
 * The accumulator tracks the number of samples processed, sum and sum of
 * squares of target values for metrics computation, and maintains separate
 * storage for the symmetric X'X matrix and the X'y vector.
 */
typedef struct RidgeStreamAccum
{
	double	  **XtX;
	double	   *Xty;
	int			feature_dim;
	int			n_samples;
	double		y_sum;
	double		y_sq_sum;
	bool		initialized;
}			RidgeStreamAccum;

static void ridge_dataset_init(RidgeDataset * dataset);
static void ridge_dataset_free(RidgeDataset * dataset);
static void ridge_dataset_load(const char *quoted_tbl,
							   const char *quoted_feat,
							   const char *quoted_target,
							   RidgeDataset * dataset);
static void ridge_dataset_load_limited(const char *quoted_tbl,
									   const char *quoted_feat,
									   const char *quoted_target,
									   RidgeDataset * dataset,
									   int max_rows);
static void ridge_stream_accum_init(RidgeStreamAccum * accum, int dim);
static void ridge_stream_accum_free(RidgeStreamAccum * accum);
static void ridge_stream_accum_add_row(RidgeStreamAccum * accum,
									   const float *features,
									   double target);
static void ridge_stream_process_chunk(const char *quoted_tbl,
									   const char *quoted_feat,
									   const char *quoted_target,
									   RidgeStreamAccum * accum,
									   int chunk_size,
									   int offset,
									   int *rows_processed);
static bytea * ridge_model_serialize(const RidgeModel *model, uint8 training_backend);
static RidgeModel *ridge_model_deserialize(const bytea * data, uint8 * training_backend_out);
static bool ridge_metadata_is_gpu(Jsonb * metadata);
static bool ridge_try_gpu_predict_catalog(int32 model_id,
										  const Vector *feature_vec,
										  double *result_out);
static bool ridge_load_model_from_catalog(int32 model_id, RidgeModel **out);

/*
 * Streaming accumulator for Lasso coordinate descent
 * Stores residuals and feature data for incremental updates
 */
typedef struct LassoStreamAccum
{
	double	   *residuals;
	float	   *features;
	double	   *targets;
	int			feature_dim;
	int			n_samples;
	double		y_mean;
	bool		initialized;
}			LassoStreamAccum;

static void lasso_dataset_init(LassoDataset * dataset);
static void lasso_dataset_free(LassoDataset * dataset);
static void lasso_dataset_load_limited(const char *quoted_tbl,
									   const char *quoted_feat,
									   const char *quoted_target,
									   LassoDataset * dataset,
									   int max_rows);
static bytea * lasso_model_serialize(const LassoModel *model, uint8 training_backend);
static LassoModel *lasso_model_deserialize(const bytea * data, uint8 * training_backend_out);
static bool lasso_metadata_is_gpu(Jsonb * metadata);
static bool lasso_try_gpu_predict_catalog(int32 model_id,
										  const Vector *feature_vec,
										  double *result_out);
static bool lasso_load_model_from_catalog(int32 model_id, LassoModel **out);

/*
 * matrix_invert
 *    Invert a square matrix using Gauss-Jordan elimination with partial pivoting.
 *
 * This function computes the inverse of an n-by-n matrix using the Gauss-Jordan
 * elimination algorithm, which transforms the input matrix into reduced row
 * echelon form by performing elementary row operations. The algorithm creates
 * an augmented matrix by appending the identity matrix to the right of the
 * input matrix, then applies row swaps, scaling, and addition operations to
 * transform the left half into the identity matrix. When the left half becomes
 * the identity, the right half contains the inverse matrix. The function uses
 * partial pivoting, selecting the largest absolute value in each column as the
 * pivot element to improve numerical stability and reduce rounding errors. If
 * the matrix is singular or nearly singular, the algorithm cannot complete the
 * inversion and returns false. This situation occurs when the matrix has zero
 * or very small determinant, indicating linear dependence among rows or columns,
 * which commonly happens with collinear features in regression problems.
 */
static bool
matrix_invert(double **matrix, int n, double **result)
{
	double **augmented = NULL;
	int			i,
				j,
				k;
	double		pivot,
				factor;

	nalloc(augmented, double *, n);
	for (i = 0; i < n; i++)
	{
		nalloc(augmented[i], double, 2 * n);
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
				{
					nfree(augmented[j]);
				}
				nfree(augmented);
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
	{
		nfree(augmented[i]);
	}
	nfree(augmented);

	return true;
}

static void
ridge_dataset_init(RidgeDataset * dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(RidgeDataset));
}

static void
ridge_dataset_free(RidgeDataset * dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
	{
		nfree(dataset->features);
		dataset->features = NULL;
	}
	if (dataset->targets != NULL)
	{
		nfree(dataset->targets);
		dataset->targets = NULL;
	}
	ridge_dataset_init(dataset);
}

static void
ridge_dataset_load(const char *quoted_tbl,
				   const char *quoted_feat,
				   const char *quoted_target,
				   RidgeDataset * dataset)
{
	StringInfoData query = {0};
	MemoryContext oldcontext;

	NdbSpiSession *load_spi_session = NULL;
	int			ret;
	int			n_samples = 0;
	int			feature_dim = 0;
	int			i;

	if (dataset == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ridge_dataset_load: dataset is NULL"),
				 errdetail("Dataset pointer is NULL"),
				 errhint("Ensure a valid dataset structure is provided.")));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(load_spi_session, oldcontext);

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quoted_feat,
					 quoted_target,
					 quoted_tbl,
					 quoted_feat,
					 quoted_target);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		nfree(query.data);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ridge_dataset_load: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and target columns.")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		nfree(query.data);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_dataset_load: need at least 10 samples, got %d", n_samples),
				 errdetail("Dataset contains %d rows, minimum required is 10", n_samples),
				 errhint("Add more data rows to the training table.")));
	}

	if (SPI_processed > 0 && SPI_tuptable != NULL && SPI_tuptable->vals != NULL &&
		SPI_tuptable->vals[0] != NULL && SPI_tuptable->tupdesc != NULL)
	{
		HeapTuple	first_tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		bool		feat_null;
		Vector	   *vec = NULL;

		feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
		if (!feat_null)
		{
			vec = DatumGetVector(feat_datum);
			feature_dim = vec->dim;
		}
	}

	if (feature_dim <= 0)
	{
		nfree(query.data);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_dataset_load: could not determine feature dimension"),
				 errdetail("Feature dimension is %d (must be > 0)", feature_dim),
				 errhint("Ensure the feature column contains valid vector data.")));
	}

	MemoryContextSwitchTo(oldcontext);
	{
		float *features_tmp = NULL;
		double *targets_tmp = NULL;
		nalloc(features_tmp, float, (size_t) n_samples * (size_t) feature_dim);
		nalloc(targets_tmp, double, (size_t) n_samples);
		dataset->features = features_tmp;
		dataset->targets = targets_tmp;
	}

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector *vec = NULL;
		float *row = NULL;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		if (feat_null)
			continue;

		vec = DatumGetVector(feat_datum);
		if (vec->dim != feature_dim)
		{
			nfree(query.data);
			NDB_SPI_SESSION_END(load_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: ridge_dataset_load: inconsistent vector dimensions"),
					 errdetail("Expected dimension %d but got %d", feature_dim, vec->dim),
					 errhint("Ensure all feature vectors have the same dimension.")));
		}

		row = dataset->features + (i * feature_dim);
		memcpy(row, vec->data, sizeof(float) * feature_dim);

		if (tupdesc->natts < 2)
		{
			continue;
		}
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (targ_null)
			continue;

		{
			Oid			targ_type = SPI_gettypeid(tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID
				|| targ_type == INT8OID)
				dataset->targets[i] =
					(double) DatumGetInt32(targ_datum);
			else
				dataset->targets[i] =
					DatumGetFloat8(targ_datum);
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	nfree(query.data);
	NDB_SPI_SESSION_END(load_spi_session);
}

/*
 * ridge_dataset_load_limited
 *
 * Load dataset with LIMIT clause to avoid loading too much data
 */
static void
ridge_dataset_load_limited(const char *quoted_tbl,
						   const char *quoted_feat,
						   const char *quoted_target,
						   RidgeDataset * dataset,
						   int max_rows)
{
	StringInfoData query = {0};
	MemoryContext oldcontext;

	NdbSpiSession *load_limited_spi_session = NULL;
	int			ret;
	int			n_samples = 0;
	int			feature_dim = 0;
	int			i;

	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;

	if (dataset == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ridge_dataset_load_limited: dataset is NULL"),
				 errdetail("Dataset pointer is NULL"),
				 errhint("Ensure a valid dataset structure is provided.")));

	if (max_rows <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_dataset_load_limited: max_rows must be positive"),
				 errdetail("Received max_rows=%d, must be > 0", max_rows),
				 errhint("Specify a positive number of rows to load.")));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(load_limited_spi_session, oldcontext);

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d",
					 quoted_feat,
					 quoted_target,
					 quoted_tbl,
					 quoted_feat,
					 quoted_target,
					 max_rows);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		nfree(query.data);
		NDB_SPI_SESSION_END(load_limited_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ridge_dataset_load_limited: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and target columns.")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		nfree(query.data);
		NDB_SPI_SESSION_END(load_limited_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_dataset_load_limited: need at least 10 samples, got %d", n_samples),
				 errdetail("Dataset contains %d rows, minimum required is 10", n_samples),
				 errhint("Add more data rows to the training table.")));
	}

	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	if (SPI_processed > 0 && SPI_tuptable != NULL && SPI_tuptable->vals != NULL &&
		SPI_tuptable->vals[0] != NULL && SPI_tuptable->tupdesc != NULL)
	{
		HeapTuple	first_tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		bool		feat_null;
		Vector	   *vec = NULL;

		feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
		if (!feat_null)
		{
			if (feat_is_array)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

				if (ARR_NDIM(arr) != 1)
				{
					nfree(query.data);
					NDB_SPI_SESSION_END(load_limited_spi_session);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: ridge_dataset_load_limited: features array must be 1-D"),
							 errdetail("Array has %d dimensions, expected 1", ARR_NDIM(arr)),
							 errhint("Ensure the feature column contains 1-dimensional arrays.")));
				}
				feature_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				vec = DatumGetVector(feat_datum);
				feature_dim = vec->dim;
			}
		}
	}

	if (feature_dim <= 0)
	{
		nfree(query.data);
		NDB_SPI_SESSION_END(load_limited_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_dataset_load_limited: could not determine feature dimension"),
				 errdetail("Feature dimension is %d (must be > 0)", feature_dim),
				 errhint("Ensure the feature column contains valid vector or array data.")));
	}

	MemoryContextSwitchTo(oldcontext);
	{
		float *features_tmp = NULL;
		double *targets_tmp = NULL;
		nalloc(features_tmp, float, (size_t) n_samples * (size_t) feature_dim);
		nalloc(targets_tmp, double, (size_t) n_samples);
		dataset->features = features_tmp;
		dataset->targets = targets_tmp;
	}

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple;
		TupleDesc	tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector *vec = NULL;
		float *row = NULL;

		/* Safe access to SPI_tuptable - validate before access */
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
				nfree(query.data);
				NDB_SPI_SESSION_END(load_limited_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: ridge_dataset_load_limited: inconsistent array feature dimensions"),
						 errdetail("Expected dimension %d but got %d (ndims=%d)", feature_dim, dimlen, ndims),
						 errhint("Ensure all feature arrays have consistent dimensions.")));
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
			if (vec->dim != feature_dim)
			{
				nfree(query.data);
				NDB_SPI_SESSION_END(load_limited_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: ridge_dataset_load_limited: inconsistent vector dimensions"),
						 errdetail("Expected dimension %d but got %d", feature_dim, vec->dim),
						 errhint("Ensure all feature vectors have the same dimension.")));
			}
			memcpy(row, vec->data, sizeof(float) * feature_dim);
		}

		/* Safe access for target - validate tupdesc has at least 2 columns */
		if (tupdesc->natts < 2)
		{
			continue;
		}
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (targ_null)
			continue;

		{
			Oid			targ_type = SPI_gettypeid(tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID
				|| targ_type == INT8OID)
				dataset->targets[i] =
					(double) DatumGetInt32(targ_datum);
			else
				dataset->targets[i] =
					DatumGetFloat8(targ_datum);
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	nfree(query.data);
	NDB_SPI_SESSION_END(load_limited_spi_session);
}

/*
 * ridge_stream_accum_init
 *
 * Initialize streaming accumulator for incremental X'X and X'y computation
 */
static void
ridge_stream_accum_init(RidgeStreamAccum * accum, int dim)
{
	int			i;

	int			dim_with_intercept;

	if (accum == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ridge_stream_accum_init: accum is NULL")));

	if (dim <= 0 || dim > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_stream_accum_init: invalid feature dimension %d",
						dim)));

	dim_with_intercept = dim + 1;

	memset(accum, 0, sizeof(RidgeStreamAccum));

	accum->feature_dim = dim;
	accum->n_samples = 0;
	accum->y_sum = 0.0;
	accum->y_sq_sum = 0.0;
	accum->initialized = false;

	/* Allocate X'X matrix */
	{
		double **XtX_tmp = NULL;
		double *Xty_tmp = NULL;
		nalloc(XtX_tmp, double *, dim_with_intercept);

		for (i = 0; i < dim_with_intercept; i++)
		{
			double *XtX_row = NULL;
			nalloc(XtX_row, double, dim_with_intercept);
			XtX_tmp[i] = XtX_row;
		}

		/* Allocate X'y vector */
		nalloc(Xty_tmp, double, dim_with_intercept);
		accum->XtX = XtX_tmp;
		accum->Xty = Xty_tmp;
		
		if (XtX_tmp == NULL || Xty_tmp == NULL)
		{
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("neurondb: ridge_stream_accum_init: failed to allocate Xty vector")));
		}
	}

	accum->initialized = true;
}

/*
 * ridge_stream_accum_free
 *
 * Free memory allocated for streaming accumulator
 */
static void
ridge_stream_accum_free(RidgeStreamAccum * accum)
{
	int			i;

	if (accum == NULL)
		return;

	if (accum->XtX != NULL)
	{
		int			dim_with_intercept = accum->feature_dim + 1;

		for (i = 0; i < dim_with_intercept; i++)
		{
			if (accum->XtX[i] != NULL)
			{
				nfree(accum->XtX[i]);
				accum->XtX[i] = NULL;
			}
		}
		nfree(accum->XtX);
		accum->XtX = NULL;
	}

	if (accum->Xty != NULL)
	{
		nfree(accum->Xty);
		accum->Xty = NULL;
	}

	memset(accum, 0, sizeof(RidgeStreamAccum));
}

/*
 * ridge_stream_accum_add_row
 *
 * Add a single row to the streaming accumulator, updating X'X and X'y
 */
static void
ridge_stream_accum_add_row(RidgeStreamAccum * accum,
						   const float *features,
						   double target)
{
	int			i;

	int			j;
	int			dim_with_intercept;

	double *xi = NULL;

	if (accum == NULL || !accum->initialized)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ridge_stream_accum_add_row: accumulator not initialized")));

	if (features == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_stream_accum_add_row: features is NULL")));

	dim_with_intercept = accum->feature_dim + 1;

	/* Allocate temporary vector for this row (with intercept) */
	nalloc(xi, double, dim_with_intercept);

	/* Build row vector: [1, x1, x2, ..., xd] */
	xi[0] = 1.0;				/* intercept term */
	for (i = 0; i < accum->feature_dim; i++)
		xi[i + 1] = (double) features[i];

	/* Update X'X: XtX[j][k] += xi[j] * xi[k] */
	for (j = 0; j < dim_with_intercept; j++)
	{
		for (i = 0; i < dim_with_intercept; i++)
			accum->XtX[j][i] += xi[j] * xi[i];

		/* Update X'y: Xty[j] += xi[j] * y */
		accum->Xty[j] += xi[j] * target;
	}

	/* Update statistics for metrics computation */
	accum->n_samples++;
	accum->y_sum += target;
	accum->y_sq_sum += target * target;

	nfree(xi);
}

/*
 * ridge_stream_process_chunk
 *
 * Process a chunk of data from the table, accumulating statistics
 * Returns number of rows processed in this chunk
 */
static void
ridge_stream_process_chunk(const char *quoted_tbl,
						   const char *quoted_feat,
						   const char *quoted_target,
						   RidgeStreamAccum * accum,
						   int chunk_size,
						   int offset,
						   int *rows_processed)
{
	StringInfoData query;
	int			ret;
	int			i __attribute__((unused));
	int			n_rows __attribute__((unused));
	Oid			feat_type_oid __attribute__((unused)) = InvalidOid;
	bool		feat_is_array __attribute__((unused)) = false;
	TupleDesc	tupdesc __attribute__((unused));
	float	   *row_buffer __attribute__((unused)) = NULL;

	if (quoted_tbl == NULL || quoted_feat == NULL || quoted_target == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_stream_process_chunk: NULL parameter")));

	if (accum == NULL || !accum->initialized)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ridge_stream_process_chunk: accumulator not initialized")));

	if (chunk_size <= 0 || chunk_size > 100000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_stream_process_chunk: invalid chunk_size %d",
						chunk_size)));

	if (rows_processed == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_stream_process_chunk: rows_processed is NULL")));

	*rows_processed = 0;

	/* Build query with LIMIT and OFFSET for chunking */

	/*
	 * Note: For views, we can't use ctid, so we use LIMIT/OFFSET without
	 * ORDER BY
	 */
	/* This is non-deterministic but efficient for large datasets */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL "
					 "LIMIT %d OFFSET %d",
					 quoted_feat,
					 quoted_target,
					 quoted_tbl,
					 quoted_feat,
					 quoted_target,
					 chunk_size,
					 offset);


	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		char	   *query_str = pstrdup(query.data);
		const char *error_msg;


		/* Provide more specific error messages for common SPI errors */
		switch (ret)
		{
			case SPI_ERROR_UNCONNECTED:
				error_msg = "SPI not connected";
				break;
			case SPI_ERROR_COPY:
				error_msg = "COPY command in progress (possible nested SPI issue or SPI disconnected)";
				break;
			case SPI_ERROR_TRANSACTION:
				error_msg = "transaction state error";
				break;
			case SPI_ERROR_ARGUMENT:
				error_msg = "invalid argument";
				break;
			default:
				error_msg = "unknown SPI error";
				break;
		}

		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ridge_stream_process_chunk: query failed (ret=%d, %s)",
						ret, error_msg),
				 errhint("Query: %s. Ensure SPI is connected and no COPY command is in progress.", query_str)));
		nfree(query_str);
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

	/* Allocate temporary buffer for one row */
	nalloc(row_buffer, float, accum->feature_dim);

	/* Process each row in the chunk */
	for (i = 0; i < n_rows; i++)
	{
		HeapTuple	tuple;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		double		target;
		Vector *vec = NULL;
		ArrayType *arr = NULL;
		int			j;

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

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		/* Safe access for target - validate tupdesc has at least 2 columns */
		if (tupdesc->natts < 2)
		{
			continue;
		}
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
			if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != accum->feature_dim)
			{
				nfree(row_buffer);
				row_buffer = NULL;
				continue;		/* Skip inconsistent rows */
			}
			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				float8	   *data = (float8 *) ARR_DATA_PTR(arr);

				for (j = 0; j < accum->feature_dim; j++)
					row_buffer[j] = (float) data[j];
			}
			else
			{
				float4	   *data = (float4 *) ARR_DATA_PTR(arr);

				memcpy(row_buffer, data, sizeof(float) * accum->feature_dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec->dim != accum->feature_dim)
			{
				nfree(row_buffer);
				row_buffer = NULL;
				continue;		/* Skip inconsistent rows */
			}
			memcpy(row_buffer, vec->data, sizeof(float) * accum->feature_dim);
		}

		/* Extract target */
		{
			Oid			targ_type = SPI_gettypeid(tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID || targ_type == INT8OID)
				target = (double) DatumGetInt32(targ_datum);
			else
				target = DatumGetFloat8(targ_datum);
		}

		/* Add row to accumulator */
		ridge_stream_accum_add_row(accum, row_buffer, target);
		(*rows_processed)++;
	}

	nfree(row_buffer);
	row_buffer = NULL;
}

/*
 * ridge_model_serialize
 */
static bytea *
ridge_model_serialize(const RidgeModel *model, uint8 training_backend)
{
	StringInfoData buf;
	int			i;

	if (model == NULL)
		return NULL;

	/* Validate model before serialization */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ridge_model_serialize: invalid n_features %d (corrupted model?)",
						model->n_features)));
	}

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: ridge_model_serialize: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	pq_begintypsend(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) */
	pq_sendbyte(&buf, training_backend);

	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendfloat8(&buf, model->intercept);
	pq_sendfloat8(&buf, model->lambda);
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
 * ridge_model_deserialize
 */
static RidgeModel *
ridge_model_deserialize(const bytea * data, uint8 * training_backend_out)
{
	RidgeModel *model = NULL;
	StringInfoData buf;
	int			i;
	uint8		training_backend = 0;

	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	/* Read training_backend first */
	training_backend = (uint8) pq_getmsgbyte(&buf);

	nalloc(model, RidgeModel, 1);
	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->intercept = pq_getmsgfloat8(&buf);
	model->lambda = pq_getmsgfloat8(&buf);
	model->r_squared = pq_getmsgfloat8(&buf);
	model->mse = pq_getmsgfloat8(&buf);
	model->mae = pq_getmsgfloat8(&buf);

	/* Validate deserialized values */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		nfree(model);
		model = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_model_deserialize: invalid n_features in deserialized model (corrupted data?)")));
	}
	if (model != NULL && (model->n_samples < 0 || model->n_samples > 100000000))
	{
		nfree(model);
		model = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: ridge_model_deserialize: invalid n_samples in deserialized model")));
	}

	if (model->n_features > 0)
	{
		double *coefficients_tmp = NULL;
		nalloc(coefficients_tmp, double, model->n_features);
		for (i = 0; i < model->n_features; i++)
			coefficients_tmp[i] = pq_getmsgfloat8(&buf);
		model->coefficients = coefficients_tmp;
	}

	/* Return training_backend if output parameter provided */
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;

	return model;
}

/*
 * ridge_metadata_is_gpu
 */
static bool
ridge_metadata_is_gpu(Jsonb * metadata)
{
	bool		is_gpu = false;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	JsonbIteratorToken r;

	if (metadata == NULL)
		return false;

	/* Check for training_backend integer in metrics */
	it = JsonbIteratorInit((JsonbContainer *) & metadata->root);
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
			nfree(key);
		}
	}

	return is_gpu;
}

/*
 * ridge_try_gpu_predict_catalog
 */
static bool
ridge_try_gpu_predict_catalog(int32 model_id,
							  const Vector *feature_vec,
							  double *result_out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	char *gpu_err = NULL;
	double		prediction = 0.0;
	bool		success = false;

	/* Check compute mode - only try GPU if compute mode allows it */
	if (!NDB_SHOULD_TRY_GPU())
		return false;

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

	/* Check if this is a GPU model - must have both GPU metrics AND GPU payload format */
	{
		bool		is_gpu_model = false;
		bool		has_gpu_metrics = false;
		uint32		payload_size;

		/* First check metrics for training_backend */
		has_gpu_metrics = ridge_metadata_is_gpu(metrics);

		/* Check payload format - GPU models must have raw GPU format (NdbCudaRidgeModelHeader) */
		/* Unified format starts with uint8 training_backend, GPU format starts with int32 feature_dim */
		payload_size = VARSIZE(payload) - VARHDRSZ;
		
		/* Check if payload is in raw GPU format (starts with int32 feature_dim) */
		if (payload_size >= sizeof(int32))
		{
			const int32 *first_int = (const int32 *) VARDATA(payload);
			int32		first_value = *first_int;
			
			/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
			if (first_value > 0 && first_value <= 100000)
			{
				/* Check if payload size matches GPU format */
				if (payload_size >= sizeof(NdbCudaRidgeModelHeader))
				{
					const NdbCudaRidgeModelHeader *hdr = (const NdbCudaRidgeModelHeader *) VARDATA(payload);
					
					/* Validate header fields match the first int32 */
					if (hdr->feature_dim == first_value &&
						hdr->n_samples >= 0 && hdr->n_samples <= 1000000000)
					{
						size_t		expected_gpu_size = sizeof(NdbCudaRidgeModelHeader) +
							sizeof(float) * (size_t) hdr->feature_dim;
						
						/* Size matches GPU format - likely a GPU model */
						if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
						{
							/* Has GPU payload format - treat as GPU model */
							is_gpu_model = true;
						}
					}
				}
			}
		}

		/* If metrics says GPU but payload is unified format, fall back to CPU */
		/* Unified format starts with uint8 (training_backend), which is < 256, so first int32 would be small */
		if (has_gpu_metrics && !is_gpu_model)
		{
			/* Metrics says GPU but payload is unified format - use CPU path */
			goto cleanup;
		}

		if (!is_gpu_model)
			goto cleanup;
	}

	if (ndb_gpu_ridge_predict(payload,
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
	if (payload != NULL)
		nfree(payload);
	if (metrics != NULL)
		nfree(metrics);
	if (gpu_err != NULL)
		nfree(gpu_err);

	return success;
}

/*
 * ridge_load_model_from_catalog
 */
static bool
ridge_load_model_from_catalog(int32 model_id, RidgeModel **out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;

	if (out == NULL)
		return false;

	*out = NULL;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
	{
		if (metrics != NULL)
		{
			nfree(metrics);
			metrics = NULL;
		}
		return false;
	}

	/* Skip GPU models with raw GPU payload format - they should be handled by GPU prediction */
	/* Unified format models (training_backend=1 but payload is unified) should be loaded as CPU */
	{
		bool		has_gpu_metrics = false;
		bool		has_gpu_payload = false;
		uint32		payload_size;

		/* Check metrics for training_backend */
		has_gpu_metrics = ridge_metadata_is_gpu(metrics);

		/* Check payload format - GPU models must have raw GPU format (NdbCudaRidgeModelHeader) */
		/* Unified format starts with uint8 training_backend (0 or 1), GPU format starts with int32 feature_dim */
		payload_size = VARSIZE(payload) - VARHDRSZ;
		
		/* Check if payload is in raw GPU format (starts with int32 feature_dim, not uint8 training_backend) */
		if (payload_size >= sizeof(uint8))
		{
			const uint8 *first_byte = (const uint8 *) VARDATA(payload);
			
			/* Unified format: first byte is training_backend (0 or 1) */
			/* GPU format: first 4 bytes are int32 feature_dim (typically > 1) */
			if (*first_byte > 1)
			{
				/* First byte > 1, likely GPU format - check further */
				if (payload_size >= sizeof(int32))
				{
					const int32 *first_int = (const int32 *) VARDATA(payload);
					int32		first_value = *first_int;
					
					/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
					if (first_value > 0 && first_value <= 100000)
					{
						/* Check if payload size matches GPU format */
						if (payload_size >= sizeof(NdbCudaRidgeModelHeader))
						{
							const NdbCudaRidgeModelHeader *hdr = (const NdbCudaRidgeModelHeader *) VARDATA(payload);
							
							/* Validate header fields match the first int32 */
							if (hdr->feature_dim == first_value &&
								hdr->n_samples >= 0 && hdr->n_samples <= 1000000000)
							{
								size_t		expected_gpu_size = sizeof(NdbCudaRidgeModelHeader) +
									sizeof(float) * (size_t) hdr->feature_dim;
								
								/* Size matches GPU format - has raw GPU payload */
								if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
								{
									has_gpu_payload = true;
								}
							}
						}
					}
				}
			}
			/* If first_byte is 0 or 1, it's unified format - not raw GPU */
		}

		/* Only skip if BOTH metrics say GPU AND payload is raw GPU format */
		/* Unified format models (GPU metrics but unified payload) should be loaded as CPU */
		if (has_gpu_metrics && has_gpu_payload)
		{
			/* Has both GPU metrics and raw GPU payload - skip, let GPU prediction handle it */
			nfree(payload);
			payload = NULL;
			if (metrics != NULL)
			{
				nfree(metrics);
				metrics = NULL;
			}
			return false;
		}
		/* Otherwise (unified format or CPU format), load as CPU model */
	}

	*out = ridge_model_deserialize(payload, NULL);

	nfree(payload);
	payload = NULL;
	if (metrics != NULL)
	{
		nfree(metrics);
		metrics = NULL;
	}

	return (*out != NULL);
}

/*
 * lasso_dataset_init
 */
static void
lasso_dataset_init(LassoDataset * dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(LassoDataset));
}

/*
 * lasso_dataset_free
 */
static void
lasso_dataset_free(LassoDataset * dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
		nfree(dataset->features);
	if (dataset->targets != NULL)
		nfree(dataset->targets);
	lasso_dataset_init(dataset);
}

/*
 * lasso_dataset_load_limited
 *
 * Load dataset with LIMIT clause to avoid loading too much data
 * Reuses ridge_dataset_load_limited since they have the same structure
 */
static void
lasso_dataset_load_limited(const char *quoted_tbl,
						   const char *quoted_feat,
						   const char *quoted_target,
						   LassoDataset * dataset,
						   int max_rows)
{
	ridge_dataset_load_limited(quoted_tbl,
							   quoted_feat,
							   quoted_target,
							   (RidgeDataset *) dataset,
							   max_rows);
}

/*
 * lasso_model_serialize
 */
static bytea *
lasso_model_serialize(const LassoModel *model, uint8 training_backend)
{
	StringInfoData buf;
	int			i;

	if (model == NULL)
		return NULL;

	/* Validate model before serialization */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lasso_model_serialize: invalid n_features %d (corrupted model?)",
						model->n_features)));
	}

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lasso_model_serialize: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	pq_begintypsend(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) */
	pq_sendbyte(&buf, training_backend);

	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendfloat8(&buf, model->intercept);
	pq_sendfloat8(&buf, model->lambda);
	pq_sendint32(&buf, model->max_iters);
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
 * lasso_model_deserialize
 */
static LassoModel *
lasso_model_deserialize(const bytea * data, uint8 * training_backend_out)
{
	LassoModel *model = NULL;
	StringInfoData buf;
	int			i;
	uint8		training_backend = 0;

	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	/* Read training_backend first */
	training_backend = (uint8) pq_getmsgbyte(&buf);

	nalloc(model, LassoModel, 1);
	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->intercept = pq_getmsgfloat8(&buf);
	model->lambda = pq_getmsgfloat8(&buf);
	model->max_iters = pq_getmsgint(&buf, 4);
	model->r_squared = pq_getmsgfloat8(&buf);
	model->mse = pq_getmsgfloat8(&buf);
	model->mae = pq_getmsgfloat8(&buf);

	/* Validate deserialized values */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		nfree(model);
		model = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lasso_model_deserialize: invalid n_features in deserialized model (corrupted data?)")));
	}
	if (model != NULL && (model->n_samples < 0 || model->n_samples > 100000000))
	{
		nfree(model);
		model = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lasso_model_deserialize: invalid n_samples in deserialized model")));
	}

	if (model->n_features > 0)
	{
		double *coefficients_tmp = NULL;
		nalloc(coefficients_tmp, double, model->n_features);
		for (i = 0; i < model->n_features; i++)
			coefficients_tmp[i] = pq_getmsgfloat8(&buf);
		model->coefficients = coefficients_tmp;
	}

	/* Return training_backend if output parameter provided */
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;

	return model;
}

/*
 * lasso_metadata_is_gpu
 */
static bool
lasso_metadata_is_gpu(Jsonb * metadata)
{
	bool		is_gpu = false;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	JsonbIteratorToken r;

	if (metadata == NULL)
		return false;

	/* Check for training_backend integer in metrics */
	it = JsonbIteratorInit((JsonbContainer *) & metadata->root);
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
			nfree(key);
		}
	}

	return is_gpu;
}

/*
 * lasso_try_gpu_predict_catalog
 */
static bool
lasso_try_gpu_predict_catalog(int32 model_id,
							  const Vector *feature_vec,
							  double *result_out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	char *gpu_err = NULL;
	double		prediction = 0.0;
	bool		success = false;

	/* Check compute mode - only try GPU if compute mode allows it */
	if (!NDB_SHOULD_TRY_GPU())
		return false;

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

	/* Check if this is a GPU model - must have both GPU metrics AND GPU payload format */
	{
		bool		is_gpu_model = false;
		bool		has_gpu_metrics = false;
		uint32		payload_size;

		/* First check metrics for training_backend */
		has_gpu_metrics = lasso_metadata_is_gpu(metrics);

		/* Check payload format - GPU models must have raw GPU format (NdbCudaLassoModelHeader) */
		/* Unified format starts with uint8 training_backend (0 or 1), GPU format starts with int32 feature_dim */
		payload_size = VARSIZE(payload) - VARHDRSZ;
		
		/* Check if payload is in raw GPU format (starts with int32 feature_dim, not uint8 training_backend) */
		if (payload_size >= sizeof(uint8))
		{
			const uint8 *first_byte = (const uint8 *) VARDATA(payload);
			
			/* Unified format: first byte is training_backend (0 or 1) */
			/* GPU format: first 4 bytes are int32 feature_dim (typically > 1) */
			if (*first_byte > 1)
			{
				/* First byte > 1, likely GPU format - check further */
				if (payload_size >= sizeof(int32))
				{
					const int32 *first_int = (const int32 *) VARDATA(payload);
					int32		first_value = *first_int;
					
					/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
					if (first_value > 0 && first_value <= 100000)
					{
						/* Check if payload size matches GPU format */
						if (payload_size >= sizeof(NdbCudaLassoModelHeader))
						{
							const NdbCudaLassoModelHeader *hdr = (const NdbCudaLassoModelHeader *) VARDATA(payload);
							
							/* Validate header fields match the first int32 */
							if (hdr->feature_dim == first_value &&
								hdr->n_samples >= 0 && hdr->n_samples <= 1000000000)
							{
								size_t		expected_gpu_size = sizeof(NdbCudaLassoModelHeader) +
									sizeof(float) * (size_t) hdr->feature_dim;
								
								/* Size matches GPU format - has raw GPU payload */
								if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
								{
									is_gpu_model = true;
								}
							}
						}
					}
				}
			}
			/* If first_byte is 0 or 1, it's unified format - not raw GPU */
		}

		/* If metrics says GPU but payload is unified format, fall back to CPU */
		/* Unified format starts with uint8 (training_backend), which is < 256, so first int32 would be small */
		if (has_gpu_metrics && !is_gpu_model)
		{
			/* Metrics says GPU but payload is unified format - use CPU path */
			goto cleanup;
		}

		if (!is_gpu_model)
			goto cleanup;
	}

	if (ndb_gpu_lasso_predict(payload,
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
	if (payload != NULL)
		nfree(payload);
	if (metrics != NULL)
		nfree(metrics);
	if (gpu_err != NULL)
		nfree(gpu_err);

	return success;
}

/*
 * lasso_load_model_from_catalog
 */
static bool
lasso_load_model_from_catalog(int32 model_id, LassoModel **out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;

	if (out == NULL)
		return false;

	*out = NULL;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
	{
		if (metrics != NULL)
			nfree(metrics);
		return false;
	}

	/* Skip GPU models with raw GPU payload format - they should be handled by GPU prediction */
	/* Unified format models (training_backend=1 but payload is unified) should be loaded as CPU */
	{
		bool		has_gpu_metrics = false;
		bool		has_gpu_payload = false;
		uint32		payload_size;

		/* Check metrics for training_backend */
		has_gpu_metrics = lasso_metadata_is_gpu(metrics);

		/* Check payload format - GPU models must have raw GPU format (NdbCudaLassoModelHeader) */
		/* Unified format starts with uint8 training_backend (0 or 1), GPU format starts with int32 feature_dim */
		payload_size = VARSIZE(payload) - VARHDRSZ;
		
		/* Check if payload is in raw GPU format (starts with int32 feature_dim, not uint8 training_backend) */
		if (payload_size >= sizeof(uint8))
		{
			const uint8 *first_byte = (const uint8 *) VARDATA(payload);
			
			/* Unified format: first byte is training_backend (0 or 1) */
			/* GPU format: first 4 bytes are int32 feature_dim (typically > 1) */
			if (*first_byte > 1)
			{
				/* First byte > 1, likely GPU format - check further */
				if (payload_size >= sizeof(int32))
				{
					const int32 *first_int = (const int32 *) VARDATA(payload);
					int32		first_value = *first_int;
					
					/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
					if (first_value > 0 && first_value <= 100000)
					{
						/* Check if payload size matches GPU format */
						if (payload_size >= sizeof(NdbCudaLassoModelHeader))
						{
							const NdbCudaLassoModelHeader *hdr = (const NdbCudaLassoModelHeader *) VARDATA(payload);
							
							/* Validate header fields match the first int32 */
							if (hdr->feature_dim == first_value &&
								hdr->n_samples >= 0 && hdr->n_samples <= 1000000000)
							{
								size_t		expected_gpu_size = sizeof(NdbCudaLassoModelHeader) +
									sizeof(float) * (size_t) hdr->feature_dim;
								
								/* Size matches GPU format - has raw GPU payload */
								if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
								{
									has_gpu_payload = true;
								}
							}
						}
					}
				}
			}
			/* If first_byte is 0 or 1, it's unified format - not raw GPU */
		}

		/* Only skip if BOTH metrics say GPU AND payload is raw GPU format */
		/* Unified format models (GPU metrics but unified payload) should be loaded as CPU */
		if (has_gpu_metrics && has_gpu_payload)
		{
			/* Has both GPU metrics and raw GPU payload - skip, let GPU prediction handle it */
			nfree(payload);
			payload = NULL;
			if (metrics != NULL)
			{
				nfree(metrics);
				metrics = NULL;
			}
			return false;
		}
		/* Otherwise (unified format or CPU format), load as CPU model */
	}

	*out = lasso_model_deserialize(payload, NULL);

	nfree(payload);
	if (metrics != NULL)
		nfree(metrics);

	return (*out != NULL);
}

/*
 * soft_threshold
 *    Apply soft thresholding operator for L1 regularization.
 *
 * The soft thresholding operator shrinks values toward zero by subtracting
 * the threshold when the absolute value exceeds it, and sets values to zero
 * when they fall below the threshold. This operation is the solution to the
 * L1-regularized least squares problem and produces sparse solutions by
 * driving small coefficients to exactly zero. When x is positive and greater
 * than lambda, the result is x minus lambda. When x is negative and its
 * absolute value is greater than lambda, the result is x plus lambda, making
 * it less negative. When the absolute value of x is less than or equal to
 * lambda, the result is zero. This selective zeroing enables feature selection
 * in Lasso regression, where irrelevant features have their weights set to
 * zero, automatically performing model selection during training.
 */
static double
soft_threshold(double x, double lambda)
{
	if (x > lambda)
		return x - lambda;
	else if (x < -lambda)
		return x + lambda;
	else
		return 0.0;
}

/*
 * train_ridge_regression
 *
 * Trains Ridge Regression (L2 regularization)
 * Uses closed-form solution: β = (X'X + λI)^(-1)X'y
 * Returns model_id
 */
PG_FUNCTION_INFO_V1(train_ridge_regression);

Datum
train_ridge_regression(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *feature_col = NULL;
	text *target_col = NULL;
	double		lambda = PG_GETARG_FLOAT8(3);	/* Regularization parameter */
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	int			nvec = 0;
	int			dim = 0;
	RidgeDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_target;
	MLGpuTrainResult gpu_result;
	char *gpu_err = NULL;
	Jsonb *gpu_hyperparams = NULL;
	StringInfoData hyperbuf = {0};
	MemoryContext oldcontext;
	NdbSpiSession *train_spi_session = NULL;
	int32		model_id = 0;

	/* Initialize gpu_result to zero to avoid undefined behavior */
	memset(&gpu_result, 0, sizeof(MLGpuTrainResult));

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);

	if (lambda < 0.0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: train_ridge_regression: lambda must be non-negative, got %.6f", lambda),
				 errdetail("Regularization parameter lambda=%f is negative", lambda),
				 errhint("Specify a non-negative regularization parameter (typically 0.0 to 1.0).")));
	}

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_target = quote_identifier(targ_str);

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(train_spi_session, oldcontext);

	/*
	 * First, determine feature dimension and row count without loading all
	 * data
	 */
	{
		StringInfoData count_query = {0};
		int			ret;
		Oid			feat_type_oid = InvalidOid;
		bool		feat_is_array = false;

		/* Get feature dimension from first row */
		initStringInfo(&count_query);
		appendStringInfo(&count_query,
						 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT 1",
						 quoted_feat,
						 quoted_target,
						 quoted_tbl,
						 quoted_feat,
						 quoted_target);
		ret = ndb_spi_execute_safe(count_query.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			nfree(count_query.data);
			NDB_SPI_SESSION_END(train_spi_session);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_ridge_regression: no valid rows found"),
					 errdetail("Query returned %d rows, expected at least 1", (int) SPI_processed),
					 errhint("Ensure the table contains valid data with non-NULL feature and target columns.")));
		}

		/* Determine feature dimension */
		if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		{
			HeapTuple	first_tuple = SPI_tuptable->vals[0];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Datum		feat_datum;
			bool		feat_null;

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
						nfree(count_query.data);
						NDB_SPI_SESSION_END(train_spi_session);
						nfree(tbl_str);
						nfree(feat_str);
						nfree(targ_str);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neurondb: train_ridge_regression: features array must be 1-D"),
								 errdetail("Array has %d dimensions, expected 1", ARR_NDIM(arr)),
								 errhint("Ensure the feature column contains 1-dimensional arrays.")));
					}
					dim = ARR_DIMS(arr)[0];
				}
				else
				{
					Vector	   *vec = DatumGetVector(feat_datum);

					dim = vec->dim;
				}
			}
		}

		if (dim <= 0)
		{
			nfree(count_query.data);
			NDB_SPI_SESSION_END(train_spi_session);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_ridge_regression: could not determine feature dimension"),
					 errdetail("Feature dimension is %d (must be > 0)", dim),
					 errhint("Ensure the feature column contains valid vector or array data.")));
		}

		/* Get row count */
		nfree(count_query.data);
		initStringInfo(&count_query);
		appendStringInfo(&count_query,
						 "SELECT COUNT(*) FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
						 quoted_tbl,
						 quoted_feat,
						 quoted_target);
		ret = ndb_spi_execute_safe(count_query.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			/* Use safe function to get int32 count */
			int32		count_val;

			if (ndb_spi_get_int32(train_spi_session, 0, 1, &count_val))
			{
				nvec = count_val;
			}
		}

		nfree(count_query.data);
		NDB_SPI_SESSION_END(train_spi_session);

		if (nvec < 10)
		{
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_ridge_regression: need at least 10 samples, got %d", nvec),
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

		if (neurondb_gpu_is_available() && nvec > 0 && dim > 0)
		{
			int			gpu_sample_limit = nvec;
			bool		gpu_train_result = false;
			/* Load limited dataset for GPU training */
			ridge_dataset_init(&dataset);
			ridge_dataset_load_limited(quoted_tbl,
									   quoted_feat,
									   quoted_target,
									   &dataset,
									   gpu_sample_limit);

			/* Create hyperparameters JSONB with lambda, same pattern as linear regression */
			initStringInfo(&hyperbuf);
			appendStringInfo(&hyperbuf, "{\"lambda\":%.6f}", lambda);
			/* Use ndb_jsonb_in_cstring like linear regression */
			gpu_hyperparams = ndb_jsonb_in_cstring(hyperbuf.data);
			if (gpu_hyperparams == NULL)
			{
				nfree(hyperbuf.data);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("neurondb: train_ridge_regression: failed to parse GPU hyperparameters JSON")));
			}
			nfree(hyperbuf.data);
			hyperbuf.data = NULL;

			PG_TRY();
			{
				gpu_train_result = ndb_gpu_try_train_model("ridge",
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
				/* GPU mode: re-throw error - don't catch it, let it propagate */
				if (NDB_REQUIRE_GPU())
				{
					PG_RE_THROW();
				}
				/* AUTO/CPU mode: capture error and fall back to CPU */
				/* Capture error message before flushing */
				if (gpu_err == NULL)
				{
					/* Set a simple error message - don't try to capture from ErrorData as it may be corrupted */
					gpu_err = pstrdup("Exception during GPU training - check GPU availability and backend registration");
				}
				FlushErrorState();
				gpu_train_result = false;
			}
			PG_END_TRY();

			/* AUTO/CPU mode: error if GPU training failed - will fall back to CPU */
			if (!gpu_train_result && !NDB_REQUIRE_GPU())
			{
				/* In AUTO mode, this is expected - will fall back to CPU below */
			}
			/* AUTO mode: GPU training failed - will fall back to CPU (handled below) */

			/* Only proceed with GPU model if training succeeded AND model_data exists */
			if (gpu_train_result && gpu_result.spec.model_data != NULL)
			{
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
				MLCatalogModelSpec spec;
				RidgeModel	ridge_model;

				bytea *unified_model_data = NULL;
				Jsonb *updated_metrics = NULL;
				char *base = NULL;
				NdbCudaRidgeModelHeader *hdr = NULL;
				float *coef_src_float = NULL;
				int			i;
				/* Convert GPU format to unified format */
				base = VARDATA(gpu_result.spec.model_data);
				hdr = (NdbCudaRidgeModelHeader *) base;
				coef_src_float = (float *) (base + sizeof(NdbCudaRidgeModelHeader));

				/* Build RidgeModel structure */
				memset(&ridge_model, 0, sizeof(RidgeModel));
				ridge_model.n_features = hdr->feature_dim;
				ridge_model.n_samples = hdr->n_samples;
				ridge_model.intercept = (double) hdr->intercept;	/* Convert float to
																	 * double */
				ridge_model.lambda = hdr->lambda;
				ridge_model.r_squared = hdr->r_squared;
				ridge_model.mse = hdr->mse;
				ridge_model.mae = hdr->mae;

				/* Copy coefficients, converting from float to double */
				if (ridge_model.n_features > 0)
				{
					double *coefficients_tmp = NULL;
					nalloc(coefficients_tmp, double, ridge_model.n_features);
					for (i = 0; i < ridge_model.n_features; i++)
						coefficients_tmp[i] = (double) coef_src_float[i];
					ridge_model.coefficients = coefficients_tmp;
				}

				/*
				 * Serialize using unified format with training_backend=1
				 * (GPU)
				 */
				unified_model_data = ridge_model_serialize(&ridge_model, 1);

				if (ridge_model.coefficients != NULL)
				{
					nfree(ridge_model.coefficients);
					ridge_model.coefficients = NULL;
				}

				/* Update metrics to use training_backend=1 */
				if (gpu_result.spec.metrics != NULL)
				{
					StringInfoData metrics_buf;

					initStringInfo(&metrics_buf);
					appendStringInfo(&metrics_buf,
									 "{\"algorithm\":\"ridge\","
									 "\"storage\":\"gpu\","
									 "\"training_backend\":1,"
									 "\"n_features\":%d,"
									 "\"n_samples\":%d,"
									 "\"lambda\":%.6f,"
									 "\"r_squared\":%.6f,"
									 "\"mse\":%.6f,"
									 "\"mae\":%.6f}",
									 ridge_model.n_features > 0 ? ridge_model.n_features : 0,
									 ridge_model.n_samples > 0 ? ridge_model.n_samples : 0,
									 ridge_model.lambda,
									 ridge_model.r_squared,
									 ridge_model.mse,
									 ridge_model.mae);

					/* Use ndb_jsonb_in_cstring (consistent with other ML algorithms) */
					updated_metrics = ndb_jsonb_in_cstring(metrics_buf.data);
					if (updated_metrics == NULL)
					{
						nfree(metrics_buf.data);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
								 errmsg("neurondb: failed to parse metrics JSON")));
					}
					nfree(metrics_buf.data);
				}

				spec = gpu_result.spec;
				spec.model_data = unified_model_data;
				spec.metrics = updated_metrics;

				/*
				 * ALWAYS copy all string pointers to current memory context
				 * before switching contexts. This ensures the pointers remain
				 * valid after memory context switch.
				 */

				/* Copy algorithm */
				spec.algorithm = pstrdup("ridge");

				/* Copy training_table */
				if (spec.training_table != NULL)
				{
					spec.training_table = pstrdup(spec.training_table);
				}
				else if (tbl_str != NULL)
				{
					spec.training_table = pstrdup(tbl_str);
				}

				/* Copy training_column */
				if (spec.training_column != NULL)
				{
					spec.training_column = pstrdup(spec.training_column);
				}
				else if (targ_str != NULL)
				{
					spec.training_column = pstrdup(targ_str);
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

				spec.algorithm = "ridge";
				spec.model_type = "regression";

				model_id = ml_catalog_register_model(&spec);
#endif

				if (gpu_err != NULL)
					nfree(gpu_err);
				if (gpu_hyperparams != NULL)
					nfree(gpu_hyperparams);
				ndb_gpu_free_train_result(&gpu_result);
				ridge_dataset_free(&dataset);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);

				PG_RETURN_INT32(model_id);
			}
			else
			{
				/* GPU training failed - check if we're in strict GPU mode */
				/* EXCEPT: Allow CPU fallback if Metal backend reports unsupported algorithm */
				bool metal_unsupported = false;
				if (gpu_err != NULL && 
					(strstr(gpu_err, "Metal backend:") != NULL || 
					 (strstr(gpu_err, "Metal") != NULL && strstr(gpu_err, "unsupported") != NULL)))
				{
					metal_unsupported = true;
				}
				
				if (NDB_REQUIRE_GPU() && !metal_unsupported)
				{
					/* Strict GPU mode: error out, no CPU fallback (unless Metal unsupported) */
					char *error_msg = NULL;

					if (gpu_err != NULL)
					{
						error_msg = pstrdup(gpu_err);
						nfree(gpu_err);
						gpu_err = NULL;
					}
					else
					{
						error_msg = pstrdup("GPU training failed");
					}

					/* Safely free GPU result if it was partially initialized */
					if (gpu_result.spec.model_data != NULL || gpu_result.spec.metrics != NULL ||
						gpu_result.spec.algorithm != NULL || gpu_result.spec.training_table != NULL ||
						gpu_result.spec.training_column != NULL || gpu_result.payload != NULL)
					{
						ndb_gpu_free_train_result(&gpu_result);
					}
					else
					{
						/* Just zero it to be safe */
						memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
					}

					if (gpu_hyperparams != NULL)
					{
						nfree(gpu_hyperparams);
						gpu_hyperparams = NULL;
					}

					ridge_dataset_free(&dataset);
					nfree(tbl_str);
					nfree(feat_str);
					nfree(targ_str);

					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: train_ridge_regression: GPU training failed - GPU mode requires GPU to be available"),
							 errdetail("%s", error_msg),
							 errhint("Check GPU availability and model compatibility, or set compute_mode='auto' for automatic CPU fallback.")));
				}

				/* AUTO mode: cleanup and fall back to CPU */
				if (gpu_err != NULL)
				{
					nfree(gpu_err);
					gpu_err = NULL;
				}

				/* Safely free GPU result if it was partially initialized */
				if (gpu_result.spec.model_data != NULL || gpu_result.spec.metrics != NULL ||
					gpu_result.spec.algorithm != NULL || gpu_result.spec.training_table != NULL ||
					gpu_result.spec.training_column != NULL || gpu_result.payload != NULL)
				{
					ndb_gpu_free_train_result(&gpu_result);
				}
				else
				{
					/* Just zero it to be safe */
					memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
				}

				if (gpu_hyperparams != NULL)
				{
					nfree(gpu_hyperparams);
					gpu_hyperparams = NULL;
				}

				ridge_dataset_free(&dataset);
			}
		}
		else if (neurondb_gpu_is_available() && !ndb_gpu_kernel_enabled("ridge_train"))
		{
		}

		/* CPU training path using streaming accumulator */
		{
			RidgeStreamAccum stream_accum;
			double **XtX_inv = NULL;

			double *beta = NULL;
			int			i,
						j;
			int			dim_with_intercept;

			RidgeModel *model = NULL;
			bytea *model_blob = NULL;
			Jsonb *metrics_json = NULL;
			int			chunk_size;
			int			offset = 0;
			int			rows_in_chunk = 0;

			NdbSpiSession *spi_session = NULL;
			MemoryContext chunk_oldcontext = CurrentMemoryContext;

			/* Ensure SPI is connected for CPU streaming path */
			/* (ridge_dataset_load_limited may have disconnected SPI) */
			NDB_SPI_SESSION_BEGIN(spi_session, chunk_oldcontext);

			/* Use larger chunks for better performance */
			if (nvec > 1000000)
				chunk_size = 100000;	/* 100k chunks for very large datasets */
			else if (nvec > 100000)
				chunk_size = 50000; /* 50k chunks for large datasets */
			else
				chunk_size = 10000; /* 10k chunks for smaller datasets */

			/* Initialize streaming accumulator */
			ridge_stream_accum_init(&stream_accum, dim);
			dim_with_intercept = dim + 1;

			/* Process data in chunks */
			while (offset < nvec)
			{
				ridge_stream_process_chunk(quoted_tbl,
										   quoted_feat,
										   quoted_target,
										   &stream_accum,
										   chunk_size,
										   offset,
										   &rows_in_chunk);

				if (rows_in_chunk == 0)
				{
					break;		/* No more rows */
				}

				offset += rows_in_chunk;
			}

			if (stream_accum.n_samples < 10)
			{
				ridge_stream_accum_free(&stream_accum);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: train_ridge_regression: need at least 10 samples, got %d",
								stream_accum.n_samples)));
			}

			/*
			 * Allocate matrices for normal equations: β = (X'X +
			 * λI)^(-1)X'y
			 */
			nalloc(XtX_inv, double *, dim_with_intercept);
			for (i = 0; i < dim_with_intercept; i++)
			{
				nalloc(XtX_inv[i], double, dim_with_intercept);
			}
			nalloc(beta, double, dim_with_intercept);

			/* Add Ridge penalty (λI) to diagonal (excluding intercept) */
			for (i = 1; i < dim_with_intercept; i++)
				stream_accum.XtX[i][i] += lambda;

			/* Invert X'X + λI */
			if (!matrix_invert(stream_accum.XtX, dim_with_intercept, XtX_inv))
			{
				for (i = 0; i < dim_with_intercept; i++)
					nfree(XtX_inv[i]);
				nfree(XtX_inv);
				nfree(beta);
				ridge_stream_accum_free(&stream_accum);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: train_ridge_regression: matrix is singular, "
								"cannot compute Ridge regression"),
						 errhint("Try increasing lambda or "
								 "removing correlated "
								 "features")));
			}

			/* Compute β = (X'X + λI)^(-1)X'y */
			/* Compute beta coefficients */
			for (i = 0; i < dim_with_intercept; i++)
			{
				beta[i] = 0.0;
				for (j = 0; j < dim_with_intercept; j++)
					beta[i] += XtX_inv[i][j] * stream_accum.Xty[j];
			}

			/* Build RidgeModel */
			nalloc(model, RidgeModel, 1);
			model->model_id = 0;	/* Will be set by catalog */
			model->n_features = dim;
			model->n_samples = stream_accum.n_samples;
			model->intercept = beta[0];
			model->lambda = lambda;
			/* Metrics will be computed below - initialize to 0.0 */
			model->r_squared = 0.0;
			model->mse = 0.0;
			model->mae = 0.0;
			{
				double *coefficients_tmp = NULL;
				nalloc(coefficients_tmp, double, dim);
				for (i = 0; i < dim; i++)
					coefficients_tmp[i] = beta[i + 1];
				model->coefficients = coefficients_tmp;
			}

			/*
			 * Compute metrics (R², MSE, MAE) using streaming accumulator
			 * statistics
			 */
			{
				double		ss_tot;
				double		ss_res = 0.0;
				double		mse = 0.0;
				double		mae = 0.0;
				int			metrics_chunk_size;
				int			metrics_offset = 0;

				/* Compute ss_tot from accumulator */
				ss_tot = stream_accum.y_sq_sum - (stream_accum.y_sum * stream_accum.y_sum / stream_accum.n_samples);

				/* Compute MSE and MAE by processing chunks for metrics */
				/* Limit metrics computation to avoid excessive time */
				metrics_chunk_size = (stream_accum.n_samples > 100000) ? 100000 : stream_accum.n_samples;
				/* Metrics computation uses existing SPI session */
				{
					while (metrics_offset < metrics_chunk_size)
					{
						StringInfoData metrics_query;
						int			metrics_ret;
						int			metrics_n_rows;
						TupleDesc	metrics_tupdesc;

						float *metrics_row_buffer = NULL;
						int			metrics_i;

						initStringInfo(&metrics_query);
						appendStringInfo(&metrics_query,
										 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d OFFSET %d",
										 quoted_feat,
										 quoted_target,
										 quoted_tbl,
										 quoted_feat,
										 quoted_target,
										 10000,
										 metrics_offset);

						metrics_ret = ndb_spi_execute_safe(metrics_query.data, true, 0);
						NDB_CHECK_SPI_TUPTABLE();
						if (metrics_ret != SPI_OK_SELECT)
						{
							nfree(metrics_query.data);
							break;
						}
						NDB_CHECK_SPI_TUPTABLE();
						metrics_n_rows = SPI_processed;
						if (metrics_n_rows == 0)
						{
							nfree(metrics_query.data);
							break;
						}

						metrics_tupdesc = SPI_tuptable->tupdesc;
						nalloc(metrics_row_buffer, float, dim);

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
							Vector *metrics_vec = NULL;
							ArrayType *metrics_arr = NULL;
							int			metrics_j;
							Oid			metrics_feat_type_oid = SPI_gettypeid(metrics_tupdesc, 1);
							bool		metrics_feat_is_array = (metrics_feat_type_oid == FLOAT8ARRAYOID || metrics_feat_type_oid == FLOAT4ARRAYOID);

							metrics_feat_datum = SPI_getbinval(metrics_tuple, metrics_tupdesc, 1, &metrics_feat_null);
							metrics_targ_datum = SPI_getbinval(metrics_tuple, metrics_tupdesc, 2, &metrics_targ_null);

							if (metrics_feat_null || metrics_targ_null)
								continue;

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

						nfree(metrics_row_buffer);
						nfree(metrics_query.data);

						metrics_offset += metrics_n_rows;
						if (metrics_offset >= metrics_chunk_size)
							break;
					}

					/* Normalize metrics */
					if (metrics_chunk_size > 0)
					{
						mse /= metrics_chunk_size;
						mae /= metrics_chunk_size;
					}
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
				if (model->coefficients != NULL)
				{
					nfree(model->coefficients);
					model->coefficients = NULL;
				}
				nfree(model);
				model = NULL;
				for (i = 0; i < dim_with_intercept; i++)
				{
					nfree(XtX_inv[i]);
					XtX_inv[i] = NULL;
				}
				nfree(XtX_inv);
				XtX_inv = NULL;
				nfree(beta);
				beta = NULL;
				ridge_stream_accum_free(&stream_accum);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: train_ridge_regression: model.n_features is invalid (%d) before serialization",
								model->n_features)));
			}
			/* Serialize CPU model in CPU format (RidgeModel) */
			model_blob = ridge_model_serialize(model, 0);	/* 0 = CPU backend */

			/* Build metrics JSON using JSONB API */
			{
				JsonbParseState *state = NULL;
				JsonbValue	jkey;
				JsonbValue	jval;

				JsonbValue *final_value = NULL;
				Numeric		n_features_num,
							n_samples_num,
							lambda_num,
							r_squared_num,
							mse_num,
							mae_num;

				PG_TRY();
				{
					(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

					/* Add algorithm */
					jkey.type = jbvString;
					jkey.val.string.val = "algorithm";
					jkey.val.string.len = strlen("algorithm");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					jval.type = jbvString;
					jval.val.string.val = "ridge";
					jval.val.string.len = strlen("ridge");
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add storage */
					jkey.val.string.val = "storage";
					jkey.val.string.len = strlen("storage");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					jval.type = jbvString;
					jval.val.string.val = "cpu";
					jval.val.string.len = strlen("cpu");
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add n_features */
					jkey.val.string.val = "n_features";
					jkey.val.string.len = strlen("n_features");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					n_features_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(model->n_features)));
					jval.type = jbvNumeric;
					jval.val.numeric = n_features_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add n_samples */
					jkey.val.string.val = "n_samples";
					jkey.val.string.len = strlen("n_samples");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(model->n_samples)));
					jval.type = jbvNumeric;
					jval.val.numeric = n_samples_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add lambda */
					jkey.val.string.val = "lambda";
					jkey.val.string.len = strlen("lambda");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					lambda_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(lambda)));
					jval.type = jbvNumeric;
					jval.val.numeric = lambda_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add r_squared */
					jkey.val.string.val = "r_squared";
					jkey.val.string.len = strlen("r_squared");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					r_squared_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(model->r_squared)));
					jval.type = jbvNumeric;
					jval.val.numeric = r_squared_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add mse */
					jkey.val.string.val = "mse";
					jkey.val.string.len = strlen("mse");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					mse_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(model->mse)));
					jval.type = jbvNumeric;
					jval.val.numeric = mse_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add mae */
					jkey.val.string.val = "mae";
					jkey.val.string.len = strlen("mae");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					mae_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(model->mae)));
					jval.type = jbvNumeric;
					jval.val.numeric = mae_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

					if (final_value == NULL)
					{
						elog(ERROR, "neurondb: train_ridge_regression: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
					}

					metrics_json = JsonbValueToJsonb(final_value);
				}
				PG_CATCH();
				{
					ErrorData  *edata = CopyErrorData();

					elog(ERROR, "neurondb: train_ridge_regression: metrics JSONB construction failed: %s", edata->message);
					FlushErrorState();
					metrics_json = NULL;
				}
				PG_END_TRY();
			}

			if (metrics_json == NULL)
			{
			}

			/* Register in catalog */
			{
				MLCatalogModelSpec spec;

				memset(&spec, 0, sizeof(MLCatalogModelSpec));
				spec.algorithm = "ridge";
				spec.model_type = "regression";
				spec.training_table = tbl_str;
				spec.training_column = targ_str;
				spec.model_data = model_blob;
				spec.metrics = metrics_json;

				/* Build hyperparameters JSON using JSONB API */
				{
					JsonbParseState *state = NULL;
					JsonbValue	jkey;
					JsonbValue	jval;

					JsonbValue *final_value = NULL;
					Numeric		lambda_num;

					PG_TRY();
					{
						(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

						jkey.type = jbvString;
						jkey.val.string.val = "lambda";
						jkey.val.string.len = strlen("lambda");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						lambda_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(lambda)));
						jval.type = jbvNumeric;
						jval.val.numeric = lambda_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

						if (final_value == NULL)
						{
							elog(ERROR, "neurondb: train_ridge_regression: pushJsonbValue(WJB_END_OBJECT) returned NULL for hyperparameters");
						}

						spec.parameters = JsonbValueToJsonb(final_value);
					}
					PG_CATCH();
					{
						ErrorData  *edata = CopyErrorData();

						elog(ERROR, "neurondb: train_ridge_regression: hyperparameters JSONB construction failed: %s", edata->message);
						FlushErrorState();
						spec.parameters = NULL;
					}
					PG_END_TRY();
				}

				model_id = ml_catalog_register_model(&spec);
			}

			for (i = 0; i < dim_with_intercept; i++)
			{
				nfree(XtX_inv[i]);
				XtX_inv[i] = NULL;
			}
			nfree(XtX_inv);
			XtX_inv = NULL;
			nfree(beta);
			beta = NULL;
			if (model->coefficients != NULL)
			{
				nfree(model->coefficients);
				model->coefficients = NULL;
			}
			nfree(model);
			model = NULL;
			ridge_stream_accum_free(&stream_accum);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);

			/* Finish SPI connection before returning */
			NDB_SPI_SESSION_END(spi_session);

			PG_RETURN_INT32(model_id);
		}
	}
}

/*
 * predict_ridge_regression_model_id
 *
 * Makes predictions using trained Ridge Regression model from catalog
 */
PG_FUNCTION_INFO_V1(predict_ridge_regression_model_id);

Datum
predict_ridge_regression_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	Vector *features = NULL;

	RidgeModel *model = NULL;
	double		prediction;
	int			i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ridge: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ridge: features vector is required")));

	features = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(features);

	/* Try GPU prediction first */
	if (ridge_try_gpu_predict_catalog(model_id, features, &prediction))
	{
		PG_RETURN_FLOAT8(prediction);
	}
	else
	{
	}

	/* Load model from catalog */
	if (!ridge_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ridge: model %d not found", model_id)));

	if (model->n_features > 0 && features->dim != model->n_features)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ridge: feature dimension mismatch (expected %d, got %d)",
						model->n_features,
						features->dim)));
	}

	/* Compute prediction: y = intercept + coef1*x1 + coef2*x2 + ... */
	prediction = model->intercept;
	for (i = 0; i < model->n_features && i < features->dim; i++)
		prediction += model->coefficients[i] * features->data[i];

	if (model != NULL)
	{
		if (model->coefficients != NULL)
			nfree(model->coefficients);
		nfree(model);
	}

	PG_RETURN_FLOAT8(prediction);
}

/*
 * train_lasso_regression
 *
 * Trains Lasso Regression (L1 regularization)
 * Uses coordinate descent algorithm
 * Returns model_id
 */
PG_FUNCTION_INFO_V1(train_lasso_regression);

Datum
train_lasso_regression(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *feature_col = NULL;
	text *target_col = NULL;
	double		lambda = PG_GETARG_FLOAT8(3);	/* Regularization parameter */
	int			max_iters = PG_NARGS() > 4 ? PG_GETARG_INT32(4) : 1000;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	int			nvec = 0;
	int			dim = 0;
	LassoDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_target;
	MLGpuTrainResult gpu_result;
	char *gpu_err = NULL;
	Jsonb *gpu_hyperparams = NULL;
	StringInfoData hyperbuf;
	MemoryContext oldcontext;
	int32		model_id = 0;

	/* Initialize gpu_result to zero to avoid undefined behavior */
	memset(&gpu_result, 0, sizeof(MLGpuTrainResult));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);

	if (lambda < 0.0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: train_lasso_regression: lambda must be non-negative, got %.6f",
						lambda)));
	}
	if (max_iters <= 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: train_lasso_regression: max_iters must be positive, got %d",
						max_iters)));
	}

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
		NdbSpiSession *spi_session = NULL;
		StringInfoData count_query;
		int			ret;
		Oid			feat_type_oid = InvalidOid;
		bool		feat_is_array = false;

		NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

		/* Get feature dimension from first row */
		ndb_spi_stringinfo_init(spi_session, &count_query);
		appendStringInfo(&count_query,
						 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT 1",
						 quoted_feat,
						 quoted_target,
						 quoted_tbl,
						 quoted_feat,
						 quoted_target);
		ret = ndb_spi_execute(spi_session, count_query.data, true, 0);
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			ndb_spi_stringinfo_free(spi_session, &count_query);
			NDB_SPI_SESSION_END(spi_session);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_lasso_regression: no valid rows found")));
		}

		/* Determine feature dimension */
		if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		{
			Datum		feat_datum;
			bool		feat_null;
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			feat_type_oid = SPI_gettypeid(tupdesc, 1);
			if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
				feat_is_array = true;

			/* Use safe access pattern for SPI_tuptable */
			if (SPI_processed > 0 && SPI_tuptable->vals != NULL)
			{
				feat_datum = SPI_getbinval(SPI_tuptable->vals[0], tupdesc, 1, &feat_null);
				if (!feat_null)
				{
					if (feat_is_array)
					{
						ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

						if (ARR_NDIM(arr) != 1)
						{
							ndb_spi_stringinfo_free(spi_session, &count_query);
							NDB_SPI_SESSION_END(spi_session);
							nfree(tbl_str);
							nfree(feat_str);
							nfree(targ_str);
							ereport(ERROR,
									(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
									 errmsg("neurondb: train_lasso_regression: features array must be 1-D")));
						}
						dim = ARR_DIMS(arr)[0];
					}
					else
					{
						Vector	   *vec = DatumGetVector(feat_datum);

						dim = vec->dim;
					}
				}
			}
		}

		if (dim <= 0)
		{
			ndb_spi_stringinfo_free(spi_session, &count_query);
			NDB_SPI_SESSION_END(spi_session);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_lasso_regression: could not determine feature dimension")));
		}

		/* Get row count */
		ndb_spi_stringinfo_reset(spi_session, &count_query);
		appendStringInfo(&count_query,
						 "SELECT COUNT(*) FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
						 quoted_tbl,
						 quoted_feat,
						 quoted_target);
		ret = ndb_spi_execute(spi_session, count_query.data, true, 0);
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int32		count_value;

			if (ndb_spi_get_int32(spi_session, 0, 1, &count_value))
				nvec = count_value;
		}

		ndb_spi_stringinfo_free(spi_session, &count_query);
		NDB_SPI_SESSION_END(spi_session);

		if (nvec < 10)
		{
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_lasso_regression: need at least 10 samples, got %d",
							nvec)));
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

		if (neurondb_gpu_is_available() && nvec > 0 && dim > 0)
		{
			int			gpu_sample_limit = nvec;
			bool		gpu_train_result = false;
			/* Load limited dataset for GPU training */
			lasso_dataset_init(&dataset);
			lasso_dataset_load_limited(quoted_tbl,
									   quoted_feat,
									   quoted_target,
									   &dataset,
									   gpu_sample_limit);

			initStringInfo(&hyperbuf);
			appendStringInfo(&hyperbuf,
							 "{\"lambda\":%.6f,\"max_iters\":%d}",
							 lambda,
							 max_iters);
			gpu_hyperparams = ndb_jsonb_in_cstring(hyperbuf.data);
			if (gpu_hyperparams == NULL)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("neurondb: train_lasso_regression: failed to parse GPU hyperparameters JSON")));
			}

			PG_TRY();
			{
				gpu_train_result = ndb_gpu_try_train_model("lasso",
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
				/* GPU mode: re-throw error - don't catch it, let it propagate */
				if (NDB_REQUIRE_GPU())
				{
					PG_RE_THROW();
				}
				/* AUTO/CPU mode: capture error and fall back to CPU */
				FlushErrorState();
				gpu_train_result = false;
			}
			PG_END_TRY();

			/* AUTO/CPU mode: error if GPU training failed - will fall back to CPU */
			if (!gpu_train_result && !NDB_REQUIRE_GPU())
			{
				/* In AUTO mode, this is expected - will fall back to CPU below */
			}
			/* AUTO mode: GPU training failed - will fall back to CPU (handled below) */

			/* Only proceed with GPU model if training succeeded AND model_data exists */
			if (gpu_train_result && gpu_result.spec.model_data != NULL)
			{
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
				MLCatalogModelSpec spec;
				LassoModel	lasso_model;

				bytea *unified_model_data = NULL;
				Jsonb *updated_metrics = NULL;
				char *base = NULL;
				NdbCudaLassoModelHeader *hdr = NULL;
				float *coef_src_float = NULL;
				int			i;
				/* Convert GPU format to unified format */
				base = VARDATA(gpu_result.spec.model_data);
				hdr = (NdbCudaLassoModelHeader *) base;
				coef_src_float = (float *) (base + sizeof(NdbCudaLassoModelHeader));

				/* Build LassoModel structure */
				memset(&lasso_model, 0, sizeof(LassoModel));
				lasso_model.n_features = hdr->feature_dim;
				lasso_model.n_samples = hdr->n_samples;
				lasso_model.intercept = (double) hdr->intercept;	/* Convert float to
																	 * double */
				lasso_model.lambda = hdr->lambda;
				lasso_model.max_iters = hdr->max_iters;
				lasso_model.r_squared = hdr->r_squared;
				lasso_model.mse = hdr->mse;
				lasso_model.mae = hdr->mae;

				/* Copy coefficients, converting from float to double */
				if (lasso_model.n_features > 0)
				{
					double *coefficients_tmp = NULL;
					nalloc(coefficients_tmp, double, lasso_model.n_features);
					for (i = 0; i < lasso_model.n_features; i++)
						coefficients_tmp[i] = (double) coef_src_float[i];
					lasso_model.coefficients = coefficients_tmp;
				}

				/*
				 * Serialize using unified format with training_backend=1
				 * (GPU)
				 */
				unified_model_data = lasso_model_serialize(&lasso_model, 1);

				if (lasso_model.coefficients != NULL)
				{
					nfree(lasso_model.coefficients);
					lasso_model.coefficients = NULL;
				}

				if (gpu_result.spec.metrics != NULL)
				{
					JsonbParseState *state = NULL;
					JsonbValue	jkey;
					JsonbValue	jval;

					JsonbValue *final_value = NULL;
					Numeric		n_features_num,
								n_samples_num,
								lambda_num,
								max_iters_num,
								r_squared_num,
								mse_num,
								mae_num;

					PG_TRY();
					{
						(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

						jkey.type = jbvString;
						jkey.val.string.len = 5;
						jkey.val.string.val = "algorithm";
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						jval.type = jbvString;
						jval.val.string.len = 5;
						jval.val.string.val = "lasso";
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "storage";
						jkey.val.string.len = 7;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						jval.type = jbvString;
						jval.val.string.len = 3;
						jval.val.string.val = "gpu";
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "training_backend";
						jkey.val.string.len = 16;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						jval.type = jbvNumeric;
						jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(1)));
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "n_features";
						jkey.val.string.len = 10;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						n_features_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(lasso_model.n_features > 0 ? lasso_model.n_features : 0)));
						jval.type = jbvNumeric;
						jval.val.numeric = n_features_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "n_samples";
						jkey.val.string.len = 9;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(lasso_model.n_samples > 0 ? lasso_model.n_samples : 0)));
						jval.type = jbvNumeric;
						jval.val.numeric = n_samples_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "lambda";
						jkey.val.string.len = 6;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						lambda_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(lasso_model.lambda)));
						jval.type = jbvNumeric;
						jval.val.numeric = lambda_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "max_iters";
						jkey.val.string.len = 9;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(lasso_model.max_iters)));
						jval.type = jbvNumeric;
						jval.val.numeric = max_iters_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "r_squared";
						jkey.val.string.len = 9;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						r_squared_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(lasso_model.r_squared)));
						jval.type = jbvNumeric;
						jval.val.numeric = r_squared_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "mse";
						jkey.val.string.len = 3;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						mse_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(lasso_model.mse)));
						jval.type = jbvNumeric;
						jval.val.numeric = mse_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "mae";
						jkey.val.string.len = 3;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						mae_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(lasso_model.mae)));
						jval.type = jbvNumeric;
						jval.val.numeric = mae_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

						if (final_value == NULL)
						{
							elog(ERROR, "neurondb: train_lasso: pushJsonbValue(WJB_END_OBJECT) returned NULL");
						}

						updated_metrics = JsonbValueToJsonb(final_value);
					}
					PG_CATCH();
					{
						ErrorData  *edata = CopyErrorData();

						elog(ERROR, "neurondb: train_lasso: JSONB construction failed: %s", edata->message);
						FlushErrorState();
						updated_metrics = NULL;
					}
					PG_END_TRY();
				}

				spec = gpu_result.spec;
				spec.model_data = unified_model_data;
				spec.metrics = updated_metrics;

				/*
				 * ALWAYS copy all string pointers to current memory context
				 * before switching contexts. This ensures the pointers remain
				 * valid after memory context switch.
				 */

				/* Copy algorithm */
				spec.algorithm = pstrdup("lasso");

				/* Copy training_table */
				if (spec.training_table != NULL)
				{
					spec.training_table = pstrdup(spec.training_table);
				}
				else if (tbl_str != NULL)
				{
					spec.training_table = pstrdup(tbl_str);
				}

				/* Copy training_column */
				if (spec.training_column != NULL)
				{
					spec.training_column = pstrdup(spec.training_column);
				}
				else if (targ_str != NULL)
				{
					spec.training_column = pstrdup(targ_str);
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

				spec.algorithm = "lasso";
				spec.model_type = "regression";

				model_id = ml_catalog_register_model(&spec);
#endif

				if (gpu_err != NULL)
					nfree(gpu_err);
				if (gpu_hyperparams != NULL)
					nfree(gpu_hyperparams);
				ndb_gpu_free_train_result(&gpu_result);
				lasso_dataset_free(&dataset);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);

				PG_RETURN_INT32(model_id);
			}
			else
			{
				/* GPU training failed - check if we're in strict GPU mode */
				/* EXCEPT: Allow CPU fallback if Metal backend reports unsupported algorithm */
				bool metal_unsupported = false;
				if (gpu_err != NULL && 
					(strstr(gpu_err, "Metal backend:") != NULL || 
					 (strstr(gpu_err, "Metal") != NULL && strstr(gpu_err, "unsupported") != NULL)))
				{
					metal_unsupported = true;
				}
				
				if (NDB_REQUIRE_GPU() && !metal_unsupported)
				{
					/* Strict GPU mode: error out, no CPU fallback (unless Metal unsupported) */
					char *error_msg = NULL;

					if (gpu_err != NULL)
					{
						error_msg = pstrdup(gpu_err);
						nfree(gpu_err);
						gpu_err = NULL;
					}
					else
					{
						error_msg = pstrdup("GPU training failed");
					}

					/* Safely free GPU result if it was partially initialized */
					if (gpu_result.spec.model_data != NULL || gpu_result.spec.metrics != NULL ||
						gpu_result.spec.algorithm != NULL || gpu_result.spec.training_table != NULL ||
						gpu_result.spec.training_column != NULL || gpu_result.payload != NULL)
					{
						ndb_gpu_free_train_result(&gpu_result);
					}
					else
					{
						/* Just zero it to be safe */
						memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
					}

					if (gpu_hyperparams != NULL)
					{
						nfree(gpu_hyperparams);
						gpu_hyperparams = NULL;
					}

					lasso_dataset_free(&dataset);
					nfree(tbl_str);
					nfree(feat_str);
					nfree(targ_str);

					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: train_lasso_regression: GPU training failed - GPU mode requires GPU to be available"),
							 errdetail("%s", error_msg),
							 errhint("Check GPU availability and model compatibility, or set compute_mode='auto' for automatic CPU fallback.")));
				}

				/* AUTO mode: cleanup and fall back to CPU */
				if (gpu_err != NULL)
				{
					nfree(gpu_err);
					gpu_err = NULL;
				}

				/* Safely free GPU result if it was partially initialized */
				if (gpu_result.spec.model_data != NULL || gpu_result.spec.metrics != NULL ||
					gpu_result.spec.algorithm != NULL || gpu_result.spec.training_table != NULL ||
					gpu_result.spec.training_column != NULL || gpu_result.payload != NULL)
				{
					ndb_gpu_free_train_result(&gpu_result);
				}
				else
				{
					/* Just zero it to be safe */
					memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
				}

				if (gpu_hyperparams != NULL)
				{
					nfree(gpu_hyperparams);
					gpu_hyperparams = NULL;
				}

				lasso_dataset_free(&dataset);
			}
		}
		else if (neurondb_gpu_is_available() && !ndb_gpu_kernel_enabled("lasso_train"))
		{
		}

		/* CPU training path - use limited dataset loading */

		/*
		 * For CPU training, use a smaller limit to avoid excessive training
		 * time
		 */
		{
			int			cpu_sample_limit = nvec;

			if (cpu_sample_limit > 100000)
			{
				cpu_sample_limit = 100000;
			}

			lasso_dataset_init(&dataset);
			lasso_dataset_load_limited(quoted_tbl,
									   quoted_feat,
									   quoted_target,
									   &dataset,
									   cpu_sample_limit);

			/* CPU training path - Coordinate Descent */
			{
				double *weights = NULL;
				double *weights_old = NULL;
				double *residuals = NULL;
				double		y_mean = 0.0;
				int			iter,
							i,
							j;
				bool		converged = false;

				LassoModel *model = NULL;
				bytea *model_blob = NULL;
				Jsonb *metrics_json = NULL;

				nvec = dataset.n_samples;
				dim = dataset.feature_dim;

				if (nvec < 10)
				{
					lasso_dataset_free(&dataset);
					nfree(tbl_str);
					nfree(feat_str);
					nfree(targ_str);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: train_lasso_regression: need at least 10 samples, got %d",
									nvec)));
				}

				for (i = 0; i < nvec; i++)
					y_mean += dataset.targets[i];
				y_mean /= nvec;

				nalloc(weights, double, dim);
				nalloc(weights_old, double, dim);
				nalloc(residuals, double, nvec);

				for (i = 0; i < nvec; i++)
					residuals[i] = dataset.targets[i] - y_mean;

				/*
				 * Coordinate descent algorithm for L1-regularized regression.
				 * The algorithm iteratively updates one weight at a time
				 * while holding others fixed, cycling through all features
				 * until convergence. For each feature j, it computes the
				 * correlation rho between the feature and current residuals,
				 * and the sum of squares z of feature values. The new weight
				 * is computed by applying soft thresholding to the
				 * unregularized solution rho/z with threshold lambda/z, which
				 * shrinks small weights to zero, producing sparse solutions.
				 * When a weight changes, the residuals are updated by
				 * subtracting the contribution of the old weight and adding
				 * the contribution of the new weight. This incremental
				 * residual update avoids recomputing predictions from scratch
				 * and maintains numerical stability. The algorithm converges
				 * when weights change by less than a threshold between
				 * iterations, indicating that the objective function has
				 * reached a minimum.
				 */
				for (iter = 0; iter < max_iters && !converged; iter++)
				{
					double		diff;

					memcpy(weights_old, weights, sizeof(double) * dim);

					for (j = 0; j < dim; j++)
					{
						double		rho = 0.0;
						double		z = 0.0;
						double		old_weight;
						float *feature_col_j = NULL;

						for (i = 0; i < nvec; i++)
						{
							feature_col_j = dataset.features
								+ (i * dim + j);
							rho += (*feature_col_j) * residuals[i];
						}

						for (i = 0; i < nvec; i++)
						{
							feature_col_j = dataset.features
								+ (i * dim + j);
							z += (*feature_col_j)
								* (*feature_col_j);
						}

						if (z < 1e-10)
							continue;

						old_weight = weights[j];
						weights[j] =
							soft_threshold(rho / z, lambda / z);

						if (weights[j] != old_weight)
						{
							double		weight_diff;

							weight_diff = weights[j] - old_weight;
							for (i = 0; i < nvec; i++)
							{
								feature_col_j = dataset.features
									+ (i * dim + j);
								residuals[i] -= (*feature_col_j)
									* weight_diff;
							}
						}
					}

					diff = 0.0;
					for (j = 0; j < dim; j++)
					{
						double		d = weights[j] - weights_old[j];

						diff += d * d;
					}

					if (sqrt(diff) < 1e-6)
					{
						converged = true;
					}
				}

				if (!converged)
				{
				}

				nalloc(model, LassoModel, 1);
				model->n_features = dim;
				model->n_samples = nvec;
				model->intercept = y_mean;
				model->lambda = lambda;
				model->max_iters = max_iters;
				{
					double *coefficients_tmp = NULL;
					nalloc(coefficients_tmp, double, dim);
					for (i = 0; i < dim; i++)
						coefficients_tmp[i] = weights[i];
					model->coefficients = coefficients_tmp;
				}

				{
					double		ss_tot = 0.0;
					double		ss_res = 0.0;
					double		mse = 0.0;
					double		mae = 0.0;

					for (i = 0; i < nvec; i++)
					{
						float	   *row = dataset.features + (i * dim);
						double		y_pred = model->intercept;
						double		error;
						int			feat_idx;

						for (feat_idx = 0; feat_idx < dim; feat_idx++)
							y_pred += model->coefficients[feat_idx]
								* row[feat_idx];

						error = dataset.targets[i] - y_pred;
						mse += error * error;
						mae += fabs(error);
						ss_res += error * error;
						ss_tot += (dataset.targets[i] - y_mean)
							* (dataset.targets[i] - y_mean);
					}

					mse /= nvec;
					mae /= nvec;
					model->r_squared = (ss_tot > 0.0)
						? (1.0 - (ss_res / ss_tot))
						: 0.0;
					model->mse = mse;
					model->mae = mae;
				}

				if (model->n_features <= 0 || model->n_features > 10000)
				{
					if (model->coefficients != NULL)
					{
						nfree(model->coefficients);
						model->coefficients = NULL;
					}
					nfree(model);
					model = NULL;
					nfree(weights);
					weights = NULL;
					nfree(weights_old);
					weights_old = NULL;
					nfree(residuals);
					residuals = NULL;
					lasso_dataset_free(&dataset);
					nfree(tbl_str);
					nfree(feat_str);
					nfree(targ_str);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: train_lasso_regression: model.n_features is invalid (%d) before serialization",
									model->n_features)));
				}
				/* Serialize model */
				model_blob = lasso_model_serialize(model, 0);

				/* Build metrics JSON using JSONB API */
				{
					JsonbParseState *state = NULL;
					JsonbValue	jkey;
					JsonbValue	jval;

					JsonbValue *final_value = NULL;
					Numeric		n_features_num,
								n_samples_num,
								lambda_num,
								max_iters_num,
								r_squared_num,
								mse_num,
								mae_num;

					PG_TRY();
					{
						(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

						jkey.type = jbvString;
						jkey.val.string.len = 9;
						jkey.val.string.val = "algorithm";
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						jval.type = jbvString;
						jval.val.string.len = 5;
						jval.val.string.val = "lasso";
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "training_backend";
						jkey.val.string.len = 16;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						jval.type = jbvNumeric;
						jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(0)));
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "n_features";
						jkey.val.string.len = 10;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						n_features_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(model->n_features)));
						jval.type = jbvNumeric;
						jval.val.numeric = n_features_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "n_samples";
						jkey.val.string.len = 9;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(model->n_samples)));
						jval.type = jbvNumeric;
						jval.val.numeric = n_samples_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "lambda";
						jkey.val.string.len = 6;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						lambda_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(model->lambda)));
						jval.type = jbvNumeric;
						jval.val.numeric = lambda_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "max_iters";
						jkey.val.string.len = 9;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(model->max_iters)));
						jval.type = jbvNumeric;
						jval.val.numeric = max_iters_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "r_squared";
						jkey.val.string.len = 9;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						r_squared_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(model->r_squared)));
						jval.type = jbvNumeric;
						jval.val.numeric = r_squared_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "mse";
						jkey.val.string.len = 3;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						mse_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(model->mse)));
						jval.type = jbvNumeric;
						jval.val.numeric = mse_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						jkey.val.string.val = "mae";
						jkey.val.string.len = 3;
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						mae_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(model->mae)));
						jval.type = jbvNumeric;
						jval.val.numeric = mae_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

						if (final_value == NULL)
						{
							elog(ERROR, "neurondb: train_lasso: pushJsonbValue(WJB_END_OBJECT) returned NULL");
						}

						metrics_json = JsonbValueToJsonb(final_value);
					}
					PG_CATCH();
					{
						ErrorData  *edata = CopyErrorData();

						elog(ERROR, "neurondb: train_lasso: JSONB construction failed: %s", edata->message);
						FlushErrorState();
						metrics_json = NULL;
					}
					PG_END_TRY();
				}

				{
					MLCatalogModelSpec spec;

					memset(&spec, 0, sizeof(MLCatalogModelSpec));
					spec.algorithm = "lasso";
					spec.model_type = "regression";
					spec.training_table = tbl_str;
					spec.training_column = targ_str;
					spec.model_data = model_blob;
					spec.metrics = metrics_json;

					/* Build hyperparameters JSON using JSONB API */
					{
						JsonbParseState *state = NULL;
						JsonbValue	jkey;
						JsonbValue	jval;

						JsonbValue *final_value = NULL;
						Numeric		lambda_num,
									max_iters_num;

						PG_TRY();
						{
							(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

							jkey.type = jbvString;
							jkey.val.string.len = 6;
							jkey.val.string.val = "lambda";
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							lambda_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(lambda)));
							jval.type = jbvNumeric;
							jval.val.numeric = lambda_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							jkey.val.string.val = "max_iters";
							jkey.val.string.len = 9;
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_iters)));
							jval.type = jbvNumeric;
							jval.val.numeric = max_iters_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

							if (final_value == NULL)
							{
								elog(ERROR, "neurondb: train_lasso: pushJsonbValue(WJB_END_OBJECT) returned NULL for hyperparameters");
							}

							spec.parameters = JsonbValueToJsonb(final_value);
						}
						PG_CATCH();
						{
							ErrorData  *edata = CopyErrorData();

							elog(ERROR, "neurondb: train_lasso: hyperparameters JSONB construction failed: %s", edata->message);
							FlushErrorState();
							spec.parameters = NULL;
						}
						PG_END_TRY();
					}

					model_id = ml_catalog_register_model(&spec);
				}

				nfree(weights);
				weights = NULL;
				nfree(weights_old);
				weights_old = NULL;
				nfree(residuals);
				residuals = NULL;
				if (model->coefficients != NULL)
				{
					nfree(model->coefficients);
					model->coefficients = NULL;
				}
				nfree(model);
				model = NULL;
				lasso_dataset_free(&dataset);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);

				PG_RETURN_INT32(model_id);
			}
		}
	}
}

/*
 * predict_lasso_regression_model_id
 *
 * Makes predictions using trained Lasso Regression model from catalog
 */
PG_FUNCTION_INFO_V1(predict_lasso_regression_model_id);

Datum
predict_lasso_regression_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	Vector *features = NULL;

	LassoModel *model = NULL;
	double		prediction;
	int			i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_lasso_regression_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_lasso_regression_model_id: features vector is required")));

	features = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(features);

	if (lasso_try_gpu_predict_catalog(model_id, features, &prediction))
	{
		PG_RETURN_FLOAT8(prediction);
	}
	else
	{
	}

	/* Load model from catalog */
	if (!lasso_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_lasso_regression_model_id: model %d not found", model_id)));

	if (model->n_features > 0 && features->dim != model->n_features)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_lasso_regression_model_id: feature dimension mismatch (expected %d, got %d)",
						model->n_features,
						features->dim)));
	}

	/* Compute prediction: y = intercept + coef1*x1 + coef2*x2 + ... */
	prediction = model->intercept;
	for (i = 0; i < model->n_features && i < features->dim; i++)
		prediction += model->coefficients[i] * features->data[i];

	if (model != NULL)
	{
		if (model->coefficients != NULL)
			nfree(model->coefficients);
		nfree(model);
	}

	PG_RETURN_FLOAT8(prediction);
}

/*
 * evaluate_ridge_regression_by_model_id
 *
 * Evaluates Ridge Regression model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: mse, mae, rmse, r_squared, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_ridge_regression_by_model_id);

Datum
evaluate_ridge_regression_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *table_name = NULL;
	text *feature_col = NULL;
	text *label_col = NULL;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	int			ret;
	int			nvec = 0;
	int			i;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	double		mse = 0.0;
	double		mae = 0.0;
	double		rmse = 0.0;
	double		r_squared = 0.0;
	double		y_mean = 0.0;
	MemoryContext oldcontext;
	StringInfoData query;

	RidgeModel *model = NULL;

	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool		is_gpu_model = false;

	double		ss_res = 0.0;
	double		ss_tot = 0.0;

	NdbSpiSession *spi_session = NULL;
	int			feat_dim = 0;


	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_ridge_regression_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	/* Validate model_id before attempting to load */
	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_ridge_regression_by_model_id: model_id must be positive, got %d", model_id),
				 errdetail("Invalid model_id: %d", model_id),
				 errhint("Provide a valid model_id from neurondb.ml_models catalog.")));

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_ridge_regression_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	if (ridge_load_model_from_catalog(model_id, &model))
	{
		is_gpu_model = false;	/* Ensure we know this is a CPU model */
	}
	else
	{
		if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			if (gpu_payload == NULL)
			{
				if (gpu_metrics)
					nfree(gpu_metrics);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_ridge_regression_by_model_id: model %d has NULL payload",
								model_id)));
			}
			
			/* Check if this is a GPU model - either by metrics or by payload format */
			{
				uint32		payload_size;

				/* First check metrics for training_backend */
				if (ridge_metadata_is_gpu(gpu_metrics))
				{
					is_gpu_model = true;
				}
				else
				{
					/* If metrics check didn't find GPU indicator, check payload format */
					/* GPU models start with NdbCudaRidgeModelHeader, CPU models start with uint8 training_backend */
					payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
					
					/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
					/* GPU format: first field is feature_dim (int32) */
					/* Check if payload looks like GPU format (starts with int32, not uint8) */
					if (payload_size >= sizeof(int32))
					{
						const int32 *first_int = (const int32 *) VARDATA(gpu_payload);
						int32		first_value = *first_int;
						
						/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
						if (first_value > 0 && first_value <= 100000)
						{
							/* Check if payload size matches GPU format */
							if (payload_size >= sizeof(NdbCudaRidgeModelHeader))
							{
								const NdbCudaRidgeModelHeader *hdr = (const NdbCudaRidgeModelHeader *) VARDATA(gpu_payload);
								
								/* Validate header fields match the first int32 */
								if (hdr->feature_dim == first_value &&
									hdr->n_samples >= 0 && hdr->n_samples <= 1000000000)
								{
									size_t		expected_gpu_size = sizeof(NdbCudaRidgeModelHeader) +
										sizeof(float) * (size_t) hdr->feature_dim;
									
									/* Size matches GPU format - likely a GPU model */
									if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
									{
										is_gpu_model = true;
									}
								}
							}
						}
					}
				}
			}
			
			if (!is_gpu_model)
			{
				if (gpu_payload)
					nfree(gpu_payload);
				if (gpu_metrics)
					nfree(gpu_metrics);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_ridge_regression_by_model_id: model %d not found",
								model_id)));
			}
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_ridge_regression_by_model_id: model %d not found",
							model_id)));
		}
	}

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(targ_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(targ_str));
	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		if (model != NULL)
		{
			if (model->coefficients != NULL)
			{
				nfree(model->coefficients);
				model->coefficients = NULL;
			}
			nfree(model);
			model = NULL;
		}
		if (tbl_str)
		{
			nfree(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			nfree(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			nfree(targ_str);
			targ_str = NULL;
		}
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_ridge_regression_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		if (model != NULL)
		{
			if (model->coefficients != NULL)
			{
				nfree(model->coefficients);
				model->coefficients = NULL;
			}
			nfree(model);
			model = NULL;
		}
		if (tbl_str)
		{
			nfree(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			nfree(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			nfree(targ_str);
			targ_str = NULL;
		}
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_ridge_regression_by_model_id: no valid rows found")));
	}

	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* Unified evaluation: Determine predict function based on compute mode */
	/* All metrics calculation is the same - only difference is predict function */
	{
		bool		use_gpu_predict = false;
		int			processed_count = 0;
		const NdbCudaRidgeModelHeader *gpu_hdr = NULL;
		const float *gpu_coefficients = NULL;

		/* Determine if we should use GPU predict or CPU predict based on compute mode */
		if (is_gpu_model && neurondb_gpu_is_available() && !NDB_COMPUTE_MODE_IS_CPU())
		{
			/* GPU model and GPU mode: use GPU predict */
			if (gpu_payload != NULL && VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaRidgeModelHeader))
			{
				gpu_hdr = (const NdbCudaRidgeModelHeader *) VARDATA(gpu_payload);
				feat_dim = gpu_hdr->feature_dim;
				gpu_coefficients = (const float *) ((const char *) gpu_hdr + sizeof(NdbCudaRidgeModelHeader));
				use_gpu_predict = true;
			}
		}
		else if (model != NULL)
		{
			/* CPU model or CPU mode: use CPU predict */
			feat_dim = model->n_features;
			use_gpu_predict = false;
		}
		else if (is_gpu_model && gpu_payload != NULL)
		{
			/* GPU model but CPU mode: convert to CPU format for CPU predict */
			if (VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaRidgeModelHeader))
			{
				gpu_hdr = (const NdbCudaRidgeModelHeader *) VARDATA(gpu_payload);
				gpu_coefficients = (const float *) ((const char *) gpu_hdr + sizeof(NdbCudaRidgeModelHeader));
				feat_dim = gpu_hdr->feature_dim;

				/* Convert GPU model to CPU format */
				{
					double *cpu_coefs = NULL;
					double		cpu_intercept = 0.0;
					int			coef_idx;

					cpu_intercept = gpu_hdr->intercept;
					nalloc(cpu_coefs, double, feat_dim);

					for (coef_idx = 0; coef_idx < feat_dim; coef_idx++)
						cpu_coefs[coef_idx] = (double) gpu_coefficients[coef_idx];

					/* Create temporary CPU model structure */
					nalloc(model, RidgeModel, 1);
					model->n_features = feat_dim;
					model->intercept = cpu_intercept;
					model->coefficients = cpu_coefs;
				}
				use_gpu_predict = false;
			}
		}

		/* Ensure we have a valid model or GPU payload */
		if (model == NULL && !use_gpu_predict)
		{
			NDB_SPI_SESSION_END(spi_session);
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_ridge_regression_by_model_id: no valid model found"),
					 errdetail("Neither CPU model nor GPU payload is available"),
					 errhint("Verify the model exists in the catalog and is in the correct format.")));
		}

		if (feat_dim <= 0)
		{
			NDB_SPI_SESSION_END(spi_session);
			if (model != NULL)
			{
				if (model->coefficients != NULL)
					nfree(model->coefficients);
				nfree(model);
			}
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_ridge_regression_by_model_id: invalid feature dimension %d",
							feat_dim)));
		}

		/* First pass: compute mean of y using only valid rows */
		{
			int			valid_count = 0;

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple = SPI_tuptable->vals[i];
				TupleDesc	tupdesc = SPI_tuptable->tupdesc;
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				int			actual_dim;

				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
					i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
					continue;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				if (tupdesc->natts < 2)
					continue;
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				/* Check feature dimension matches */
				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1)
						continue;
					actual_dim = ARR_DIMS(arr)[0];
				}
				else
				{
					Vector *vec = DatumGetVector(feat_datum);
					actual_dim = vec->dim;
				}

				if (actual_dim != feat_dim)
					continue;

				y_mean += DatumGetFloat8(targ_datum);
				valid_count++;
			}

			if (valid_count > 0)
				y_mean /= valid_count;
			else
				y_mean = 0.0;
		}

		/* Second pass: unified evaluation loop - prediction based on compute mode */
		for (i = 0; i < nvec; i++)
		{
			HeapTuple	tuple = SPI_tuptable->vals[i];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Datum		feat_datum;
			Datum		targ_datum;
			bool		feat_null;
			bool		targ_null;
			double		y_true;
			double		y_pred = 0.0;
			double		error;
			int			j;
			int			actual_dim;
			float	   *feat_row = NULL;

			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
				i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
				continue;

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			if (tupdesc->natts < 2)
				continue;
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

			if (feat_null || targ_null)
				continue;

			y_true = DatumGetFloat8(targ_datum);

			/* Extract features and determine dimension */
			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(feat_datum);
				if (ARR_NDIM(arr) != 1)
					continue;
				actual_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				Vector *vec = DatumGetVector(feat_datum);
				actual_dim = vec->dim;
			}

			/* Validate feature dimension matches model */
			if (actual_dim != feat_dim)
				continue;

			/* Extract features to float array for prediction */
			nalloc(feat_row, float, feat_dim);
			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(feat_datum);
				if (feat_type_oid == FLOAT8ARRAYOID)
				{
					float8	   *data = (float8 *) ARR_DATA_PTR(arr);
					for (j = 0; j < feat_dim; j++)
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
				Vector *vec = DatumGetVector(feat_datum);
				memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
			}

			/* Call appropriate predict function based on compute mode */
			if (use_gpu_predict)
			{
				/* GPU predict path - prediction based on compute mode */
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
				int			predict_rc;
				char	   *gpu_err = NULL;

				predict_rc = ndb_gpu_ridge_predict(gpu_payload,
												   feat_row,
												   feat_dim,
												   &y_pred,
												   &gpu_err);
				if (predict_rc != 0)
				{
					/* GPU predict failed - check compute mode */
					if (NDB_REQUIRE_GPU())
					{
						/* Strict GPU mode: error out */
						if (gpu_err)
							nfree(gpu_err);
						nfree(feat_row);
						NDB_SPI_SESSION_END(spi_session);
						if (model != NULL)
						{
							if (model->coefficients != NULL)
								nfree(model->coefficients);
							nfree(model);
						}
						nfree(gpu_payload);
						nfree(gpu_metrics);
						nfree(tbl_str);
						nfree(feat_str);
						nfree(targ_str);
						ndb_spi_stringinfo_free(spi_session, &query);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("neurondb: evaluate_ridge_regression_by_model_id: GPU prediction failed in GPU mode"),
								 errdetail("GPU prediction failed for row %d: %s", i, gpu_err ? gpu_err : "unknown error"),
								 errhint("GPU mode requires GPU prediction to succeed. Check GPU availability and model compatibility.")));
					}
					else
					{
						/* AUTO mode: fall back to CPU if available */
						if (gpu_err)
							nfree(gpu_err);
						if (model != NULL)
						{
							/* Compute prediction using CPU model */
							y_pred = model->intercept;
							for (j = 0; j < feat_dim; j++)
								y_pred += model->coefficients[j] * (double) feat_row[j];
						}
						else
						{
							/* No CPU model available - use GPU coefficients */
							y_pred = gpu_hdr->intercept;
							for (j = 0; j < feat_dim; j++)
								y_pred += (double) gpu_coefficients[j] * (double) feat_row[j];
						}
					}
				}
				if (gpu_err)
					nfree(gpu_err);
#endif
			}
			else
			{
				/* CPU predict path - prediction based on compute mode */
				if (model == NULL)
				{
					nfree(feat_row);
					continue;
				}

				y_pred = model->intercept;
				for (j = 0; j < feat_dim; j++)
					y_pred += model->coefficients[j] * (double) feat_row[j];
			}

			/* Compute errors (same for both CPU and GPU) */
			error = y_true - y_pred;
			mse += error * error;
			mae += fabs(error);
			ss_res += error * error;
			ss_tot += (y_true - y_mean) * (y_true - y_mean);

			processed_count++;
			nfree(feat_row);
		}

		/* Normalize metrics using actual processed count */
		if (processed_count > 0)
		{
			mse /= processed_count;
			mae /= processed_count;
		}
		else
		{
			mse = 0.0;
			mae = 0.0;
		}
		rmse = sqrt(mse);

		/* Compute R-squared */
		if (ss_tot > 1e-10 && processed_count > 0)
			r_squared = 1.0 - (ss_res / ss_tot);
		else
			r_squared = 0.0;

		nvec = processed_count;

		/* Cleanup */
		if (model != NULL)
		{
			if (model->coefficients != NULL)
				nfree(model->coefficients);
			nfree(model);
		}
		if (gpu_payload != NULL)
			nfree(gpu_payload);
		if (gpu_metrics != NULL)
			nfree(gpu_metrics);
	}

	/* Build JSONB result */
	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	/* Switch to old context and build JSONB directly using JSONB API */
	MemoryContextSwitchTo(oldcontext);
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;
		JsonbValue *final_value = NULL;
		Numeric		mse_num,
					mae_num,
					rmse_num,
					r_squared_num,
					n_samples_num;


		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			mse_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(mse)));
			jkey.type = jbvString;
			jkey.val.string.len = 3;
			jkey.val.string.val = "mse";
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = mse_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			mae_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(mae)));
			jkey.val.string.val = "mae";
			jkey.val.string.len = 3;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = mae_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			rmse_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(rmse)));
			jkey.val.string.val = "rmse";
			jkey.val.string.len = 4;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = rmse_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			r_squared_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(r_squared)));
			jkey.val.string.val = "r_squared";
			jkey.val.string.len = 9;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = r_squared_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(nvec)));
			jkey.val.string.val = "n_samples";
			jkey.val.string.len = 9;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = n_samples_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: evaluate_ridge: pushJsonbValue(WJB_END_OBJECT) returned NULL - this should not happen");
			}

			result_jsonb = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData  *edata = CopyErrorData();

			elog(ERROR, "neurondb: evaluate_ridge: JSONB construction failed at step: %s, input values: mse=%.6f, mae=%.6f, rmse=%.6f, r_squared=%.6f, n_samples=%d",
				 edata->message, mse, mae, rmse, r_squared, nvec);
			FlushErrorState();
			result_jsonb = NULL;
		}
		PG_END_TRY();
	}

	if (result_jsonb == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_ridge_regression_by_model_id: failed to create JSONB result")));
	}

	PG_RETURN_JSONB_P(result_jsonb);
}

/*
 * evaluate_lasso_regression_by_model_id
 *
 * Evaluates Lasso Regression model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: mse, mae, rmse, r_squared, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_lasso_regression_by_model_id);

Datum
evaluate_lasso_regression_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *table_name = NULL;
	text *feature_col = NULL;
	text *label_col = NULL;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	int			ret;
	int			nvec = 0;
	int			i;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	double		mse = 0.0;
	double		mae = 0.0;
	double		rmse = 0.0;
	double		r_squared = 0.0;
	double		y_mean = 0.0;
	MemoryContext oldcontext;
	StringInfoData query;

	LassoModel *model = NULL;
	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool		is_gpu_model = false;

	double		ss_res = 0.0;
	double		ss_tot = 0.0;

	NdbSpiSession *spi_session = NULL;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_lasso_regression_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	/* Defensive: validate model_id range */
	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_lasso_regression_by_model_id: model_id must be positive, got %d",
						model_id)));

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_lasso_regression_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	/* Defensive: validate text pointers */
	if (table_name == NULL || feature_col == NULL || label_col == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_lasso_regression_by_model_id: table_name, feature_col, and label_col cannot be NULL")));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	if (tbl_str == NULL || feat_str == NULL || targ_str == NULL)
	{
		if (tbl_str)
		{
			nfree(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			nfree(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			nfree(targ_str);
			targ_str = NULL;
		}
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_lasso_regression_by_model_id: failed to convert text parameters")));
	}

	oldcontext = CurrentMemoryContext;

	/* Load model from catalog - try CPU first, then GPU */
	if (!lasso_load_model_from_catalog(model_id, &model))
	{
		/* Try GPU model */
		if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			if (gpu_payload == NULL)
			{
				if (gpu_metrics)
					nfree(gpu_metrics);
				if (tbl_str)
					nfree(tbl_str);
				if (feat_str)
					nfree(feat_str);
				if (targ_str)
					nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_lasso_regression_by_model_id: model %d has NULL payload",
								model_id)));
			}
			
			/* Check if this is a GPU model - either by metrics or by payload format */
			{
				uint32		payload_size;

				/* First check metrics for training_backend */
				if (lasso_metadata_is_gpu(gpu_metrics))
				{
					is_gpu_model = true;
				}
				else
				{
					/* If metrics check didn't find GPU indicator, check payload format */
					/* GPU models start with NdbCudaLassoModelHeader, CPU models start with uint8 training_backend */
					payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
					
					/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
					/* GPU format: first field is feature_dim (int32) */
					/* Check if payload looks like GPU format (starts with int32, not uint8) */
					if (payload_size >= sizeof(int32))
					{
						const int32 *first_int = (const int32 *) VARDATA(gpu_payload);
						int32		first_value = *first_int;
						
						/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
						if (first_value > 0 && first_value <= 100000)
						{
							/* Check if payload size matches GPU format */
							if (payload_size >= sizeof(NdbCudaLassoModelHeader))
							{
								const NdbCudaLassoModelHeader *hdr = (const NdbCudaLassoModelHeader *) VARDATA(gpu_payload);
								
								/* Validate header fields match the first int32 */
								if (hdr->feature_dim == first_value &&
									hdr->n_samples >= 0 && hdr->n_samples <= 1000000000)
								{
									size_t		expected_gpu_size = sizeof(NdbCudaLassoModelHeader) +
										sizeof(float) * (size_t) hdr->feature_dim;
									
									/* Size matches GPU format - likely a GPU model */
									if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
									{
										is_gpu_model = true;
									}
								}
							}
						}
					}
				}
			}
			
			if (!is_gpu_model)
			{
				if (gpu_payload)
					nfree(gpu_payload);
				if (gpu_metrics)
					nfree(gpu_metrics);
				if (tbl_str)
					nfree(tbl_str);
				if (feat_str)
					nfree(feat_str);
				if (targ_str)
					nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_lasso_regression_by_model_id: model %d not found",
								model_id)));
			}
		}
		else
		{
			if (tbl_str)
				nfree(tbl_str);
			if (feat_str)
				nfree(feat_str);
			if (targ_str)
				nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_lasso_regression_by_model_id: model %d not found",
							model_id)));
		}
	}

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Build query - single query to fetch all data */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(targ_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(targ_str));
	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		if (model != NULL)
		{
			if (model->coefficients != NULL)
			{
				nfree(model->coefficients);
				model->coefficients = NULL;
			}
			nfree(model);
			model = NULL;
		}
		if (gpu_payload)
		{
			nfree(gpu_payload);
			gpu_payload = NULL;
		}
		if (gpu_metrics)
		{
			nfree(gpu_metrics);
			gpu_metrics = NULL;
		}
		if (tbl_str)
		{
			nfree(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			nfree(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			nfree(targ_str);
			targ_str = NULL;
		}
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_lasso_regression_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		if (model != NULL)
		{
			if (model->coefficients != NULL)
			{
				nfree(model->coefficients);
				model->coefficients = NULL;
			}
			nfree(model);
			model = NULL;
		}
		if (gpu_payload)
		{
			nfree(gpu_payload);
			gpu_payload = NULL;
		}
		if (gpu_metrics)
		{
			nfree(gpu_metrics);
			gpu_metrics = NULL;
		}
		if (tbl_str)
		{
			nfree(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			nfree(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			nfree(targ_str);
			targ_str = NULL;
		}
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_lasso_regression_by_model_id: no valid rows found")));
	}

	/* Determine feature column type */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* Unified evaluation: Determine predict function based on compute mode */
	/* All metrics calculation is the same - only difference is predict function */
	{
		bool		use_gpu_predict = false;
		int			processed_count = 0;
		int			feat_dim = 0;
		const NdbCudaLassoModelHeader *gpu_hdr = NULL;
		const float *gpu_coefficients = NULL;

		/* Determine if we should use GPU predict or CPU predict based on compute mode */
		if (is_gpu_model && neurondb_gpu_is_available() && !NDB_COMPUTE_MODE_IS_CPU())
		{
			/* GPU model and GPU mode: use GPU predict */
			if (gpu_payload != NULL && VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaLassoModelHeader))
			{
				gpu_hdr = (const NdbCudaLassoModelHeader *) VARDATA(gpu_payload);
				feat_dim = gpu_hdr->feature_dim;
				gpu_coefficients = (const float *) ((const char *) gpu_hdr + sizeof(NdbCudaLassoModelHeader));
				use_gpu_predict = true;
			}
		}
		else if (model != NULL)
		{
			/* CPU model or CPU mode: use CPU predict */
			feat_dim = model->n_features;
			use_gpu_predict = false;
		}
		else if (is_gpu_model && gpu_payload != NULL)
		{
			/* GPU model but CPU mode: convert to CPU format for CPU predict */
			if (VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaLassoModelHeader))
			{
				gpu_hdr = (const NdbCudaLassoModelHeader *) VARDATA(gpu_payload);
				gpu_coefficients = (const float *) ((const char *) gpu_hdr + sizeof(NdbCudaLassoModelHeader));
				feat_dim = gpu_hdr->feature_dim;

				/* Convert GPU model to CPU format */
				{
					double *cpu_coefs = NULL;
					double		cpu_intercept = 0.0;
					int			coef_idx;

					cpu_intercept = gpu_hdr->intercept;
					nalloc(cpu_coefs, double, feat_dim);

					for (coef_idx = 0; coef_idx < feat_dim; coef_idx++)
						cpu_coefs[coef_idx] = (double) gpu_coefficients[coef_idx];

					/* Create temporary CPU model structure */
					nalloc(model, LassoModel, 1);
					model->n_features = feat_dim;
					model->intercept = cpu_intercept;
					model->coefficients = cpu_coefs;
				}
				use_gpu_predict = false;
			}
		}

		/* Ensure we have a valid model or GPU payload */
		if (model == NULL && !use_gpu_predict)
		{
			NDB_SPI_SESSION_END(spi_session);
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_lasso_regression_by_model_id: no valid model found"),
					 errdetail("Neither CPU model nor GPU payload is available"),
					 errhint("Verify the model exists in the catalog and is in the correct format.")));
		}

		if (feat_dim <= 0)
		{
			NDB_SPI_SESSION_END(spi_session);
			if (model != NULL)
			{
				if (model->coefficients != NULL)
					nfree(model->coefficients);
				nfree(model);
			}
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_lasso_regression_by_model_id: invalid feature dimension %d",
							feat_dim)));
		}

		/* First pass: compute mean of y using only valid rows */
		{
			int			valid_count = 0;

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple = SPI_tuptable->vals[i];
				TupleDesc	tupdesc = SPI_tuptable->tupdesc;
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				int			actual_dim;

				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
					i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
					continue;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				if (tupdesc->natts < 2)
					continue;
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				/* Check feature dimension matches */
				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(feat_datum);
					if (arr == NULL || ARR_NDIM(arr) != 1)
						continue;
					actual_dim = ARR_DIMS(arr)[0];
				}
				else
				{
					Vector *vec = DatumGetVector(feat_datum);
					if (vec == NULL)
						continue;
					actual_dim = vec->dim;
				}

				if (actual_dim != feat_dim)
					continue;

				y_mean += DatumGetFloat8(targ_datum);
				valid_count++;
			}

			if (valid_count > 0)
				y_mean /= valid_count;
			else
				y_mean = 0.0;
		}

		/* Second pass: unified evaluation loop - prediction based on compute mode */
		for (i = 0; i < nvec; i++)
		{
			HeapTuple	tuple = SPI_tuptable->vals[i];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Datum		feat_datum;
			Datum		targ_datum;
			bool		feat_null;
			bool		targ_null;
			double		y_true;
			double		y_pred = 0.0;
			double		error;
			int			j;
			int			actual_dim;
			float	   *feat_row = NULL;

			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
				i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
				continue;

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			if (tupdesc->natts < 2)
				continue;
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

			if (feat_null || targ_null)
				continue;

			y_true = DatumGetFloat8(targ_datum);

			/* Extract features and determine dimension */
			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(feat_datum);
				if (arr == NULL || ARR_NDIM(arr) != 1)
					continue;
				actual_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				Vector *vec = DatumGetVector(feat_datum);
				if (vec == NULL)
					continue;
				actual_dim = vec->dim;
			}

			/* Validate feature dimension matches model */
			if (actual_dim != feat_dim)
				continue;

			/* Extract features to float array for prediction */
			nalloc(feat_row, float, feat_dim);
			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(feat_datum);
				if (feat_type_oid == FLOAT8ARRAYOID)
				{
					float8	   *data = (float8 *) ARR_DATA_PTR(arr);
					for (j = 0; j < feat_dim; j++)
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
				Vector *vec = DatumGetVector(feat_datum);
				memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
			}

			/* Call appropriate predict function based on compute mode */
			if (use_gpu_predict)
			{
				/* GPU predict path - prediction based on compute mode */
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
				int			predict_rc;
				char	   *gpu_err = NULL;

				predict_rc = ndb_gpu_lasso_predict(gpu_payload,
												   feat_row,
												   feat_dim,
												   &y_pred,
												   &gpu_err);
				if (predict_rc != 0)
				{
					/* GPU predict failed - check compute mode */
					if (NDB_REQUIRE_GPU())
					{
						/* Strict GPU mode: error out */
						if (gpu_err)
							nfree(gpu_err);
						nfree(feat_row);
						NDB_SPI_SESSION_END(spi_session);
						if (model != NULL)
						{
							if (model->coefficients != NULL)
								nfree(model->coefficients);
							nfree(model);
						}
						nfree(gpu_payload);
						nfree(gpu_metrics);
						nfree(tbl_str);
						nfree(feat_str);
						nfree(targ_str);
						ndb_spi_stringinfo_free(spi_session, &query);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("neurondb: evaluate_lasso_regression_by_model_id: GPU prediction failed in GPU mode"),
								 errdetail("GPU prediction failed for row %d: %s", i, gpu_err ? gpu_err : "unknown error"),
								 errhint("GPU mode requires GPU prediction to succeed. Check GPU availability and model compatibility.")));
					}
					else
					{
						/* AUTO mode: fall back to CPU if available */
						if (gpu_err)
							nfree(gpu_err);
						if (model != NULL)
						{
							/* Compute prediction using CPU model */
							y_pred = model->intercept;
							for (j = 0; j < feat_dim; j++)
								y_pred += model->coefficients[j] * (double) feat_row[j];
						}
						else
						{
							/* No CPU model available - use GPU coefficients */
							y_pred = gpu_hdr->intercept;
							for (j = 0; j < feat_dim; j++)
								y_pred += (double) gpu_coefficients[j] * (double) feat_row[j];
						}
					}
				}
				if (gpu_err)
					nfree(gpu_err);
#endif
			}
			else
			{
				/* CPU predict path - prediction based on compute mode */
				if (model == NULL)
				{
					nfree(feat_row);
					continue;
				}

				y_pred = model->intercept;
				for (j = 0; j < feat_dim; j++)
					y_pred += model->coefficients[j] * (double) feat_row[j];
			}

			/* Compute errors (same for both CPU and GPU) */
			error = y_true - y_pred;
			mse += error * error;
			mae += fabs(error);
			ss_res += error * error;
			ss_tot += (y_true - y_mean) * (y_true - y_mean);

			processed_count++;
			nfree(feat_row);
		}

		/* Normalize metrics using actual processed count */
		if (processed_count > 0)
		{
			mse /= processed_count;
			mae /= processed_count;
		}
		else
		{
			mse = 0.0;
			mae = 0.0;
		}
		rmse = sqrt(mse);

		/* Compute R-squared */
		if (ss_tot > 1e-10 && processed_count > 0)
			r_squared = 1.0 - (ss_res / ss_tot);
		else
			r_squared = 0.0;

		nvec = processed_count;

		/* Cleanup */
		if (model != NULL)
		{
			if (model->coefficients != NULL)
				nfree(model->coefficients);
			nfree(model);
		}
		if (gpu_payload != NULL)
			nfree(gpu_payload);
		if (gpu_metrics != NULL)
			nfree(gpu_metrics);
	}

	/* Build JSONB result */
	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	/* Switch to old context and build JSONB directly using JSONB API */
	MemoryContextSwitchTo(oldcontext);
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;
		JsonbValue *final_value = NULL;
		Numeric		mse_num,
					mae_num,
					rmse_num,
					r_squared_num,
					n_samples_num;

		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			mse_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(mse)));
			jkey.type = jbvString;
			jkey.val.string.len = 3;
			jkey.val.string.val = "mse";
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = mse_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			mae_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(mae)));
			jkey.val.string.val = "mae";
			jkey.val.string.len = 3;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = mae_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			rmse_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(rmse)));
			jkey.val.string.val = "rmse";
			jkey.val.string.len = 4;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = rmse_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			r_squared_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(r_squared)));
			jkey.val.string.val = "r_squared";
			jkey.val.string.len = 9;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = r_squared_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(nvec)));
			jkey.val.string.val = "n_samples";
			jkey.val.string.len = 10;
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = n_samples_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: evaluate_lasso_regression_by_model_id: pushJsonbValue(WJB_END_OBJECT) returned NULL");
			}

			result_jsonb = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData  *edata = CopyErrorData();

			elog(ERROR, "neurondb: evaluate_lasso_regression_by_model_id: JSONB construction failed: %s", edata->message);
			FlushErrorState();
			result_jsonb = NULL;
		}
		PG_END_TRY();
	}

	if (result_jsonb == NULL)
	{
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_lasso_regression_by_model_id: failed to create JSONB result")));
	}

	nfree(tbl_str);
	nfree(feat_str);
	nfree(targ_str);
	PG_RETURN_JSONB_P(result_jsonb);
}


/*
 * train_elastic_net
 *
 * Trains Elastic Net (L1 + L2 regularization)
 * Combines Ridge and Lasso penalties
 */
PG_FUNCTION_INFO_V1(train_elastic_net);

Datum
train_elastic_net(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *feature_col = PG_GETARG_TEXT_PP(1);
	text	   *target_col = PG_GETARG_TEXT_PP(2);
	double		alpha =
		PG_GETARG_FLOAT8(3);	/* Overall regularization strength */
	double		l1_ratio =
		PG_GETARG_FLOAT8(4);	/* L1 vs L2 ratio (0=Ridge, 1=Lasso) */

	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	RidgeDataset dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_target;
	int			nvec;
	int			dim;
	int			i;
	int			j;

	double *coefficients = NULL;
	double		intercept = 0.0;
	double		l1_penalty;
	double		l2_penalty;

	double *Xty = NULL;
	double *beta = NULL;
	Datum *result_datums = NULL;
	ArrayType *result_array = NULL;
	MemoryContext oldcontext;
	MemoryContext elastic_context;

	if (alpha < 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("elastic_net: alpha must be non-negative")));

	if (l1_ratio < 0.0 || l1_ratio > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("elastic_net: l1_ratio must be between 0 and 1")));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);
	elastic_context = AllocSetContextCreate(CurrentMemoryContext,
											"elastic net training context",
											ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(elastic_context);

	ridge_dataset_init(&dataset);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_target = quote_identifier(targ_str);

	ridge_dataset_load(quoted_tbl, quoted_feat, quoted_target, &dataset);

	nvec = dataset.n_samples;
	dim = dataset.feature_dim;

	if (nvec < 10)
	{
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(elastic_context);
		ridge_dataset_free(&dataset);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("elastic_net: need at least 10 samples, got %d",
						nvec)));
	}

	l1_penalty = alpha * l1_ratio;
	l2_penalty = alpha * (1.0 - l1_ratio);

	nalloc(coefficients, double, dim);
	nalloc(beta, double, (dim + 1));

	{
		double **XtX_2d = NULL;
		double **XtX_inv_2d = NULL;
		double *Xty_local = NULL;
		int			dim_with_intercept = dim + 1;
		int			k;

		nalloc(XtX_2d, double *, dim_with_intercept);
		nalloc(XtX_inv_2d, double *, dim_with_intercept);

		for (i = 0; i < dim_with_intercept; i++)
		{
			nalloc(XtX_2d[i], double, dim_with_intercept);
			nalloc(XtX_inv_2d[i], double, dim_with_intercept);
		}

		nalloc(Xty_local, double, (dim + 1));

		for (i = 0; i < dim; i++)
		{
			for (j = 0; j < dim; j++)
			{
				double		sum = 0.0;

				for (k = 0; k < nvec; k++)
					sum += dataset.features[k * dim + i] *
						dataset.features[k * dim + j];

				XtX_2d[i + 1][j + 1] = sum;
				if (i == j)
					XtX_2d[i + 1][j + 1] += l2_penalty;
			}
		}

		for (i = 0; i < dim; i++)
		{
			double		sum = 0.0;

			for (k = 0; k < nvec; k++)
				sum += dataset.features[k * dim + i];
			XtX_2d[i + 1][0] = sum;
			XtX_2d[0][i + 1] = sum;
		}
		XtX_2d[0][0] = nvec;

		for (i = 0; i < dim; i++)
		{
			double		sum = 0.0;

			for (k = 0; k < nvec; k++)
				sum += dataset.features[k * dim + i] * dataset.targets[k];
			Xty_local[i + 1] = sum;
		}
		{
			double		sum = 0.0;

			for (k = 0; k < nvec; k++)
				sum += dataset.targets[k];
			Xty_local[0] = sum;
		}

		{
			bool		invert_ok;

			invert_ok = matrix_invert(XtX_2d, dim_with_intercept,
									  XtX_inv_2d);

			if (invert_ok)
			{
				for (i = 0; i < dim_with_intercept; i++)
				{
					beta[i] = 0.0;
					for (j = 0; j < dim_with_intercept; j++)
						beta[i] += XtX_inv_2d[i][j] * Xty_local[j];
				}

				for (i = 1; i < dim_with_intercept; i++)
				{
					if (beta[i] > l1_penalty)
						beta[i] -= l1_penalty;
					else if (beta[i] < -l1_penalty)
						beta[i] += l1_penalty;
					else
						beta[i] = 0.0;
				}

				intercept = beta[0];
				for (i = 0; i < dim; i++)
					coefficients[i] = beta[i + 1];
			}
			else
			{
				for (i = 0; i < dim_with_intercept; i++)
				{
					nfree(XtX_2d[i]);
					nfree(XtX_inv_2d[i]);
				}
				nfree(XtX_2d);
				nfree(XtX_inv_2d);
				nfree(Xty);
				nfree(beta);
				nfree(coefficients);
				MemoryContextSwitchTo(oldcontext);
				MemoryContextDelete(elastic_context);
				ridge_dataset_free(&dataset);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("elastic_net: failed to solve linear system")));
			}

			for (i = 0; i < dim_with_intercept; i++)
			{
				nfree(XtX_2d[i]);
				nfree(XtX_inv_2d[i]);
			}
			nfree(XtX_2d);
			nfree(XtX_inv_2d);
		}
	}

	nalloc(result_datums, Datum, (dim + 1));
	result_datums[0] = Float8GetDatum(intercept);
	for (i = 0; i < dim; i++)
		result_datums[i + 1] = Float8GetDatum(coefficients[i]);

	result_array = construct_array(result_datums,
								   dim + 1,
								   FLOAT8OID,
								   8,
								   FLOAT8PASSBYVAL,
								   'd');

	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(elastic_context);
	ridge_dataset_free(&dataset);
	nfree(tbl_str);
	nfree(feat_str);
	nfree(targ_str);
	nfree(Xty);
	nfree(beta);
	nfree(coefficients);
	nfree(result_datums);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * predict_elastic_net
 *      Predicts using a trained Elastic Net model.
 *      Arguments: int4 model_id, float8[] features
 *      Returns: float8 prediction
 */
PG_FUNCTION_INFO_V1(predict_elastic_net);

Datum
predict_elastic_net(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	ArrayType  *features_array = PG_GETARG_ARRAYTYPE_P(1);

	return DirectFunctionCall2(predict_ridge_regression_model_id,
							   Int32GetDatum(model_id),
							   PointerGetDatum(features_array));
}

/*
 * evaluate_elastic_net_by_model_id
 *      Evaluates Elastic Net model performance on a dataset.
 *      Arguments: int4 model_id, text table_name, text feature_col, text label_col
 *      Returns: jsonb with regression metrics
 */
PG_FUNCTION_INFO_V1(evaluate_elastic_net_by_model_id);

Datum
evaluate_elastic_net_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *table_name = NULL;
	text *feature_col = NULL;
	text *label_col = NULL;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			i;
	double		mse = 0.0;

	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	double		mae = 0.0;
	double		ss_tot = 0.0;
	double		ss_res = 0.0;
	double		y_mean = 0.0;
	double		r_squared;
	double		rmse;

	StringInfoData jsonbuf;
	Jsonb *result = NULL;
	MemoryContext oldcontext;

	NdbSpiSession *spi_session = NULL;

	/* Validate arguments */
	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_elastic_net_by_model_id: 4 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_elastic_net_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_elastic_net_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str, targ_str, tbl_str, feat_str, targ_str);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_elastic_net_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 2)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_elastic_net_by_model_id: need at least 2 samples, got %d",
						nvec)));
	}

	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		targ_datum;
		bool		targ_null;

		/* Safe access for target - validate tupdesc has at least 2 columns */
		if (tupdesc->natts < 2)
		{
			continue;
		}
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (!targ_null)
			y_mean += DatumGetFloat8(targ_datum);
	}
	y_mean /= nvec;
	feat_type_oid = InvalidOid;
	feat_is_array = false;
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	{
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
		if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			feat_is_array = true;
	}

	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple;
		TupleDesc	tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		ArrayType *arr = NULL;
		Vector *vec = NULL;
		double		y_true;
		double		y_pred;
		double		error;
		int			actual_dim;
		int			j;

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
		/* Safe access for target - validate tupdesc has at least 2 columns */
		if (tupdesc->natts < 2)
		{
			continue;
		}
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		y_true = DatumGetFloat8(targ_datum);

		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
			if (ARR_NDIM(arr) != 1)
			{
				ndb_spi_stringinfo_free(spi_session, &query);
				NDB_SPI_SESSION_END(spi_session);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("elastic_net: features array must be 1-D")));
			}
			actual_dim = ARR_DIMS(arr)[0];
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			actual_dim = vec->dim;
		}

		if (feat_is_array)
		{
			Datum		features_datum = feat_datum;

			y_pred = DatumGetFloat8(DirectFunctionCall2(predict_elastic_net,
														Int32GetDatum(model_id),
														features_datum));
		}
		else
		{
			int			ndims = 1;
			int			dims[1];
			int			lbs[1];
			Datum *elems = NULL;
			ArrayType *feature_array = NULL;
			Datum		features_datum;

			dims[0] = actual_dim;
			lbs[0] = 1;
			nalloc(elems, Datum, actual_dim);

			for (j = 0; j < actual_dim; j++)
				elems[j] = Float8GetDatum(vec->data[j]);

			feature_array = construct_md_array(elems, NULL, ndims, dims, lbs,
											   FLOAT8OID, sizeof(float8), true, 'd');
			features_datum = PointerGetDatum(feature_array);

			y_pred = DatumGetFloat8(DirectFunctionCall2(predict_elastic_net,
														Int32GetDatum(model_id),
														features_datum));

			nfree(elems);
			nfree(feature_array);
		}

		error = y_true - y_pred;
		mse += error * error;
		mae += fabs(error);
		ss_res += error * error;
		ss_tot += (y_true - y_mean) * (y_true - y_mean);
	}

	mse /= nvec;
	mae /= nvec;
	rmse = sqrt(mse);

	if (ss_tot == 0.0)
		r_squared = 0.0;
	else
		r_squared = 1.0 - (ss_res / ss_tot);

	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
					 mse, mae, rmse, r_squared, nvec);

	result = ndb_jsonb_in_cstring(jsonbuf.data);
	if (result == NULL)
	{
		nfree(jsonbuf.data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("neurondb: failed to parse metrics JSON")));
	}
	nfree(jsonbuf.data);
	jsonbuf.data = NULL;

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);
	if (tbl_str)
	{
		nfree(tbl_str);
		tbl_str = NULL;
	}
	if (feat_str)
	{
		nfree(feat_str);
		feat_str = NULL;
	}
	if (targ_str)
	{
		nfree(targ_str);
		targ_str = NULL;
	}

	PG_RETURN_JSONB_P(result);
}

typedef struct RidgeGpuModelState
{
	bytea	   *model_blob;
		Jsonb	   *metrics;
	int			feature_dim;
	int			n_samples;
}			RidgeGpuModelState;

static void
ridge_gpu_release_state(RidgeGpuModelState * state)
{
	if (state == NULL)
		return;
	if (state->model_blob != NULL)
		nfree(state->model_blob);
	if (state->metrics != NULL)
		nfree(state->metrics);
	nfree(state);
}

static bool
ridge_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	RidgeGpuModelState *state = NULL;
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
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

	rc = ndb_gpu_ridge_train(spec->feature_matrix,
							 spec->label_vector,
							 spec->sample_count,
							 spec->feature_dim,
							 spec->hyperparameters,
							 &payload,
							 &metrics,
							 errstr);
	if (rc != 0 || payload == NULL)
	{
		/* On error, GPU backend should have cleaned up - don't free here */
		return false;
	}

	if (model->backend_state != NULL)
	{
		ridge_gpu_release_state((RidgeGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	nalloc(state, RidgeGpuModelState, 1);
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	if (metrics != NULL)
	{
		state->metrics = metrics;
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
ridge_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
				  float *output, int output_dim, char **errstr)
{
	const		RidgeGpuModelState *state;
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

	state = (const RidgeGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	rc = ndb_gpu_ridge_predict(state->model_blob, input,
							   state->feature_dim > 0 ? state->feature_dim : input_dim,
							   &prediction, errstr);
	if (rc != 0)
		return false;

	output[0] = (float) prediction;
	return true;
}

static bool
ridge_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
				   MLGpuMetrics *out, char **errstr)
{
	const		RidgeGpuModelState *state;
	Jsonb	   *metrics_json = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const RidgeGpuModelState *) model->backend_state;
	{
		StringInfoData buf;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"ridge\",\"training_backend\":1,\"n_features\":%d,\"n_samples\":%d}",
						 state->feature_dim > 0 ? state->feature_dim : 0,
						 state->n_samples > 0 ? state->n_samples : 0);
		metrics_json = ndb_jsonb_in_cstring(buf.data);
		nfree(buf.data);
	}
	if (out != NULL)
		out->payload = metrics_json;
	return true;
}

static bool
ridge_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
					Jsonb * *metadata_out, char **errstr)
{
	const		RidgeGpuModelState *state;
	RidgeModel	ridge_model;

	bytea *unified_payload = NULL;
	char *base = NULL;
	NdbCudaRidgeModelHeader *hdr = NULL;
	float *coef_src_float = NULL;
	int			i;

	(void) state;
	(void) ridge_model;
	(void) unified_payload;
	(void) base;
	(void) hdr;
	(void) coef_src_float;
	(void) i;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const RidgeGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
	base = VARDATA(state->model_blob);
	hdr = (NdbCudaRidgeModelHeader *) base;
	coef_src_float = (float *) (base + sizeof(NdbCudaRidgeModelHeader));

	memset(&ridge_model, 0, sizeof(RidgeModel));
	ridge_model.n_features = hdr->feature_dim;
	ridge_model.n_samples = hdr->n_samples;
	ridge_model.intercept = (double) hdr->intercept;
	ridge_model.lambda = hdr->lambda;
	ridge_model.r_squared = hdr->r_squared;
	ridge_model.mse = hdr->mse;
	ridge_model.mae = hdr->mae;

	if (ridge_model.n_features > 0)
	{
		double *coefficients_tmp = NULL;
		nalloc(coefficients_tmp, double, ridge_model.n_features);
		for (i = 0; i < ridge_model.n_features; i++)
			coefficients_tmp[i] = (double) coef_src_float[i];
		ridge_model.coefficients = coefficients_tmp;
	}

	unified_payload = ridge_model_serialize(&ridge_model, 1);

	if (ridge_model.coefficients != NULL)
	{
		nfree(ridge_model.coefficients);
		ridge_model.coefficients = NULL;
	}

	if (payload_out != NULL)
		*payload_out = unified_payload;
	else if (unified_payload != NULL)
		nfree(unified_payload);

	/* Return stored metrics; copy into caller context */
	if (metadata_out != NULL)
	{
		Jsonb	   *metrics_copy = NULL;

		if (state->metrics != NULL)
		{
			text   *metrics_text = DatumGetTextP(DirectFunctionCall1(jsonb_out,
											 PointerGetDatum(state->metrics)));
			char   *metrics_cstr = text_to_cstring(metrics_text);

			metrics_copy = ndb_jsonb_in_cstring(metrics_cstr);
			pfree(metrics_text);
			nfree(metrics_cstr);
		}
		else
		{
			StringInfoData metrics_buf;

			initStringInfo(&metrics_buf);
			appendStringInfo(&metrics_buf,
						 "{\"algorithm\":\"ridge\","
						 "\"training_backend\":1,"
						 "\"n_features\":%d,"
						 "\"n_samples\":%d}",
					 state->feature_dim > 0 ? state->feature_dim : 0,
					 state->n_samples > 0 ? state->n_samples : 0);

			metrics_copy = ndb_jsonb_in_cstring(metrics_buf.data);
			nfree(metrics_buf.data);
		}

		*metadata_out = metrics_copy;
	}

	return true;
#else
	/* For non-CUDA builds, GPU serialization is not supported */
	if (errstr != NULL)
		*errstr = pstrdup("ridge_gpu_serialize: CUDA not available");
	return false;
#endif
}

static bool
ridge_gpu_deserialize(MLGpuModel *model, const bytea * payload,
					  const Jsonb * metadata, char **errstr)
{
	RidgeGpuModelState *state = NULL;
	RidgeModel *ridge_model = NULL;
	uint8		training_backend = 0;

	bytea *gpu_payload = NULL;
	char *base = NULL;
	char *tmp = NULL;
	NdbCudaRidgeModelHeader *hdr = NULL;
	float *coef_dest = NULL;
	size_t		payload_bytes;
	int			i;


	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
	ridge_model = ridge_model_deserialize(payload, &training_backend);
	if (ridge_model == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("ridge_gpu_deserialize: failed to deserialize unified format");
		return false;
	}

	payload_bytes = sizeof(NdbCudaRidgeModelHeader) +
		sizeof(float) * (size_t) ridge_model->n_features;
	tmp = NULL;
	nalloc(tmp, char, VARHDRSZ + payload_bytes);
	gpu_payload = (bytea *) tmp;
	NDB_CHECK_ALLOC(gpu_payload, "gpu_payload");
	SET_VARSIZE(gpu_payload, VARHDRSZ + payload_bytes);
	base = VARDATA(gpu_payload);

	hdr = (NdbCudaRidgeModelHeader *) base;
	hdr->feature_dim = ridge_model->n_features;
	hdr->n_samples = ridge_model->n_samples;
	hdr->intercept = (float) ridge_model->intercept;
	hdr->lambda = ridge_model->lambda;
	hdr->r_squared = ridge_model->r_squared;
	hdr->mse = ridge_model->mse;
	hdr->mae = ridge_model->mae;

	coef_dest = (float *) (base + sizeof(NdbCudaRidgeModelHeader));
	if (ridge_model->coefficients != NULL)
	{
		for (i = 0; i < ridge_model->n_features; i++)
			coef_dest[i] = (float) ridge_model->coefficients[i];
	}

	if (ridge_model->coefficients != NULL)
	{
		nfree(ridge_model->coefficients);
		ridge_model->coefficients = NULL;
	}
	nfree(ridge_model);
	ridge_model = NULL;

	nalloc(state, RidgeGpuModelState, 1);
	memset(state, 0, sizeof(RidgeGpuModelState));
	state->model_blob = gpu_payload;
	state->feature_dim = hdr->feature_dim;
	state->n_samples = hdr->n_samples;
	state->metrics = NULL;

	if (metadata != NULL)
	{
		text   *metadata_text = DatumGetTextP(DirectFunctionCall1(jsonb_out,
											 PointerGetDatum(metadata)));
		char   *metadata_cstr = text_to_cstring(metadata_text);

		state->metrics = ndb_jsonb_in_cstring(metadata_cstr);
		pfree(metadata_text);
		nfree(metadata_cstr);
	}

	if (model->backend_state != NULL)
		ridge_gpu_release_state((RidgeGpuModelState *) model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
#else
	/* For non-CUDA builds, GPU deserialization is not supported */
	if (errstr != NULL)
		*errstr = pstrdup("ridge_gpu_deserialize: CUDA not available");
	return false;
#endif
}

static void
ridge_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		ridge_gpu_release_state((RidgeGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps ridge_gpu_model_ops = {
	.algorithm = "ridge",
	.train = ridge_gpu_train,
	.predict = ridge_gpu_predict,
	.evaluate = ridge_gpu_evaluate,
	.serialize = ridge_gpu_serialize,
	.deserialize = ridge_gpu_deserialize,
	.destroy = ridge_gpu_destroy,
};

typedef struct LassoGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			feature_dim;
	int			n_samples;
}			LassoGpuModelState;

static void
lasso_gpu_release_state(LassoGpuModelState * state)
{
	if (state == NULL)
		return;
	if (state->model_blob != NULL)
		nfree(state->model_blob);
	if (state->metrics != NULL)
		nfree(state->metrics);
	nfree(state);
}

static bool
lasso_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	LassoGpuModelState *state = NULL;
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
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

	rc = ndb_gpu_lasso_train(spec->feature_matrix,
							 spec->label_vector,
							 spec->sample_count,
							 spec->feature_dim,
							 spec->hyperparameters,
							 &payload,
							 &metrics,
							 errstr);
	if (rc != 0 || payload == NULL)
	{
		/* On error, GPU backend should have cleaned up - don't free here */
		return false;
	}

	if (model->backend_state != NULL)
	{
		lasso_gpu_release_state((LassoGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	nalloc(state, LassoGpuModelState, 1);
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	if (metrics != NULL)
	{
		state->metrics = metrics;
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
lasso_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
				  float *output, int output_dim, char **errstr)
{
	const		LassoGpuModelState *state;
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

	state = (const LassoGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	rc = ndb_gpu_lasso_predict(state->model_blob, input,
							   state->feature_dim > 0 ? state->feature_dim : input_dim,
							   &prediction, errstr);
	if (rc != 0)
		return false;

	output[0] = (float) prediction;
	return true;
}

static bool
lasso_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
				   MLGpuMetrics *out, char **errstr)
{
	const		LassoGpuModelState *state;
	Jsonb	   *metrics_json = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const LassoGpuModelState *) model->backend_state;
	{
		StringInfoData buf;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"lasso\",\"training_backend\":1,\"n_features\":%d,\"n_samples\":%d}",
						 state->feature_dim > 0 ? state->feature_dim : 0,
						 state->n_samples > 0 ? state->n_samples : 0);
		metrics_json = ndb_jsonb_in_cstring(buf.data);
		nfree(buf.data);
	}
	if (out != NULL)
		out->payload = metrics_json;
	return true;
}

static bool
lasso_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
					Jsonb * *metadata_out, char **errstr)
{
	const		LassoGpuModelState *state;
	LassoModel	lasso_model;

	bytea *unified_payload = NULL;
	char *base = NULL;
	NdbCudaLassoModelHeader *hdr = NULL;
	float *coef_src_float = NULL;
	int			i;

	(void) state;
	(void) lasso_model;
	(void) unified_payload;
	(void) base;
	(void) hdr;
	(void) coef_src_float;
	(void) i;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const LassoGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
	base = VARDATA(state->model_blob);
	hdr = (NdbCudaLassoModelHeader *) base;
	coef_src_float = (float *) (base + sizeof(NdbCudaLassoModelHeader));

	memset(&lasso_model, 0, sizeof(LassoModel));
	lasso_model.n_features = hdr->feature_dim;
	lasso_model.n_samples = hdr->n_samples;
	lasso_model.intercept = (double) hdr->intercept;
	lasso_model.lambda = hdr->lambda;
	lasso_model.max_iters = hdr->max_iters;
	lasso_model.r_squared = hdr->r_squared;
	lasso_model.mse = hdr->mse;
	lasso_model.mae = hdr->mae;

	if (lasso_model.n_features > 0)
	{
		double *coefficients_tmp = NULL;
		nalloc(coefficients_tmp, double, lasso_model.n_features);
		for (i = 0; i < lasso_model.n_features; i++)
			coefficients_tmp[i] = (double) coef_src_float[i];
		lasso_model.coefficients = coefficients_tmp;
	}

	unified_payload = lasso_model_serialize(&lasso_model, 1);

	if (lasso_model.coefficients != NULL)
	{
		nfree(lasso_model.coefficients);
		lasso_model.coefficients = NULL;
	}

	if (payload_out != NULL)
		*payload_out = unified_payload;
	else if (unified_payload != NULL)
		nfree(unified_payload);

	/* Return stored metrics; copy into caller context */
	if (metadata_out != NULL)
	{
		Jsonb	   *metrics_copy = NULL;

		if (state->metrics != NULL)
		{
			text   *metrics_text = DatumGetTextP(DirectFunctionCall1(jsonb_out,
											 PointerGetDatum(state->metrics)));
			char   *metrics_cstr = text_to_cstring(metrics_text);

			metrics_copy = ndb_jsonb_in_cstring(metrics_cstr);
			pfree(metrics_text);
			nfree(metrics_cstr);
		}
		else
		{
			StringInfoData metrics_buf;

			initStringInfo(&metrics_buf);
			appendStringInfo(&metrics_buf,
						 "{\"algorithm\":\"lasso\","
						 "\"training_backend\":1,"
						 "\"n_features\":%d,"
						 "\"n_samples\":%d}",
					 state->feature_dim > 0 ? state->feature_dim : 0,
					 state->n_samples > 0 ? state->n_samples : 0);

			metrics_copy = ndb_jsonb_in_cstring(metrics_buf.data);
			nfree(metrics_buf.data);
		}

		*metadata_out = metrics_copy;
	}

	return true;
#else
	if (errstr != NULL)
		*errstr = pstrdup("lasso_gpu_serialize: CUDA not available");
	return false;
#endif
}

static bool
lasso_gpu_deserialize(MLGpuModel *model, const bytea * payload,
					  const Jsonb * metadata, char **errstr)
{
	LassoGpuModelState *state = NULL;
	LassoModel *lasso_model = NULL;
	uint8		training_backend = 0;

	bytea *gpu_payload = NULL;
	char *base = NULL;
	char *tmp = NULL;
	NdbCudaLassoModelHeader *hdr = NULL;
	float *coef_dest = NULL;
	size_t		payload_bytes;
	int			i;

	(void) gpu_payload;
	(void) base;
	(void) hdr;
	(void) coef_dest;
	(void) payload_bytes;
	(void) i;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
	lasso_model = lasso_model_deserialize(payload, &training_backend);
	if (lasso_model == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lasso_gpu_deserialize: failed to deserialize unified format");
		return false;
	}

	payload_bytes = sizeof(NdbCudaLassoModelHeader) +
		sizeof(float) * (size_t) lasso_model->n_features;
	nalloc(tmp, char, VARHDRSZ + payload_bytes);
	gpu_payload = (bytea *) tmp;
	NDB_CHECK_ALLOC(gpu_payload, "gpu_payload");
	SET_VARSIZE(gpu_payload, VARHDRSZ + payload_bytes);
	base = VARDATA(gpu_payload);

	hdr = (NdbCudaLassoModelHeader *) base;
	hdr->feature_dim = lasso_model->n_features;
	hdr->n_samples = lasso_model->n_samples;
	hdr->intercept = (float) lasso_model->intercept;
	hdr->lambda = lasso_model->lambda;
	hdr->max_iters = lasso_model->max_iters;
	hdr->r_squared = lasso_model->r_squared;
	hdr->mse = lasso_model->mse;
	hdr->mae = lasso_model->mae;

	coef_dest = (float *) (base + sizeof(NdbCudaLassoModelHeader));
	if (lasso_model->coefficients != NULL)
	{
		for (i = 0; i < lasso_model->n_features; i++)
			coef_dest[i] = (float) lasso_model->coefficients[i];
	}

	if (lasso_model->coefficients != NULL)
	{
		nfree(lasso_model->coefficients);
		lasso_model->coefficients = NULL;
	}
	nfree(lasso_model);
	lasso_model = NULL;

	nalloc(state, LassoGpuModelState, 1);
	memset(state, 0, sizeof(LassoGpuModelState));
	state->model_blob = gpu_payload;
	state->feature_dim = hdr->feature_dim;
	state->n_samples = hdr->n_samples;
	state->metrics = NULL;

	if (metadata != NULL)
	{
		text   *metadata_text = DatumGetTextP(DirectFunctionCall1(jsonb_out,
											 PointerGetDatum(metadata)));
		char   *metadata_cstr = text_to_cstring(metadata_text);

		state->metrics = ndb_jsonb_in_cstring(metadata_cstr);
		pfree(metadata_text);
		nfree(metadata_cstr);
	}

	if (model->backend_state != NULL)
		lasso_gpu_release_state((LassoGpuModelState *) model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
#else
	if (errstr != NULL)
		*errstr = pstrdup("lasso_gpu_deserialize: CUDA not available");
	return false;
#endif
}

static void
lasso_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		lasso_gpu_release_state((LassoGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps lasso_gpu_model_ops = {
	.algorithm = "lasso",
	.train = lasso_gpu_train,
	.predict = lasso_gpu_predict,
	.evaluate = lasso_gpu_evaluate,
	.serialize = lasso_gpu_serialize,
	.deserialize = lasso_gpu_deserialize,
	.destroy = lasso_gpu_destroy,
};

void
neurondb_gpu_register_ridge_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&ridge_gpu_model_ops);
	registered = true;
}

void
neurondb_gpu_register_lasso_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&lasso_gpu_model_ops);
	registered = true;
}
