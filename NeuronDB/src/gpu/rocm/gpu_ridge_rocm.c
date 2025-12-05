/*-------------------------------------------------------------------------
 *
 * gpu_ridge_rocm.c
 *    ROCm backend bridge for Ridge Regression training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_ridge_rocm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_HIP

#include <float.h>
#include <math.h>
#include <string.h>

#include "neurondb_rocm_runtime.h"
#include "neurondb_rocm_launchers.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"

#include "ml_ridge_regression_internal.h"
#include "neurondb_rocm_ridge.h"

#ifdef NDB_GPU_HIP
#include <cublas_v2.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#endif

/* Reuse linear regression kernels */
extern hipError_t launch_linreg_compute_xtx_kernel(const float *features,
												   const double *targets,
												   int n_samples,
												   int feature_dim,
												   int dim_with_intercept,
												   double *XtX,
												   double *Xty);
extern hipError_t launch_linreg_eval_kernel(const float *features,
											const double *targets,
											const double *coefficients,
											double intercept,
											int n_samples,
											int feature_dim,
											double *sse_out,
											double *sae_out,
											long long *count_out);
extern hipError_t launch_ridge_add_lambda_diagonal_kernel(double *matrix,
														  double lambda,
														  int dim);

int
ndb_rocm_ridge_pack_model(const RidgeModel *model,
						  bytea * *model_data,
						  Jsonb * *metrics,
						  char **errstr)
{
	size_t		payload_bytes;
	char	   *base;
	NdbCudaRidgeModelHeader *hdr;
	float	   *coef_dest;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid Ridge model for HIP pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaRidgeModelHeader)
		+ sizeof(float) * (size_t) model->n_features;

	NDB_DECLARE(bytea *, blob);
	NDB_DECLARE(char *, blob_raw);
	NDB_ALLOC(blob_raw, char, VARHDRSZ + payload_bytes);
	blob = (bytea *) blob_raw;
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaRidgeModelHeader *) base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->intercept = (float) model->intercept;
	hdr->lambda = model->lambda;
	hdr->r_squared = model->r_squared;
	hdr->mse = model->mse;
	hdr->mae = model->mae;

	coef_dest = (float *) (base + sizeof(NdbCudaRidgeModelHeader));
	if (model->coefficients != NULL)
	{
		int			i;

		for (i = 0; i < model->n_features; i++)
			coef_dest[i] = (float) model->coefficients[i];
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb	   *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"ridge\","
						 "\"training_backend\":1,"
						 "\"n_features\":%d,"
						 "\"n_samples\":%d,"
						 "\"lambda\":%.6f,"
						 "\"r_squared\":%.6f,"
						 "\"mse\":%.6f,"
						 "\"mae\":%.6f}",
						 model->n_features,
						 model->n_samples,
						 model->lambda,
						 model->r_squared,
						 model->mse,
						 model->mae);

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
														  jsonb_in, CStringGetTextDatum(buf.data)));
		NDB_FREE(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_rocm_ridge_train(const float *features,
					 const double *targets,
					 int n_samples,
					 int feature_dim,
					 const Jsonb * hyperparams,
					 bytea * *model_data,
					 Jsonb * *metrics,
					 char **errstr)
{
	const double default_lambda = 0.01;
	double		lambda = default_lambda;
	float	   *d_features = NULL;
	double	   *d_targets = NULL;
	double	   *d_XtX = NULL;
	double	   *d_Xty = NULL;
	double	   *d_XtX_inv = NULL;
	double	   *d_beta = NULL;
	NDB_DECLARE(double *, h_XtX);
	NDB_DECLARE(double *, h_Xty);
	NDB_DECLARE(double *, h_XtX_inv);
	NDB_DECLARE(double *, h_beta);
	bytea	   *payload = NULL;
	Jsonb	   *metrics_json = NULL;
	hipError_t	status = hipSuccess;
	size_t		feature_bytes;
	size_t		target_bytes;
	size_t		XtX_bytes;
	size_t		Xty_bytes;
	size_t		beta_bytes;
	int			dim_with_intercept;
	int			i;
	int			j;
	int			rc = -1;

	if (errstr)
		*errstr = NULL;

	/* Defensive: Comprehensive parameter validation */
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ndb_rocm_ridge_train: model_data is NULL");
		return -1;
	}

	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ndb_rocm_ridge_train: features is NULL");
		return -1;
	}

	if (targets == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ndb_rocm_ridge_train: targets is NULL");
		return -1;
	}

	if (n_samples <= 0 || n_samples > 10000000)
	{
		if (errstr)
			*errstr = psprintf("ndb_rocm_ridge_train: invalid n_samples %d (must be 1-10000000)",
							   n_samples);
		return -1;
	}

	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = psprintf("ndb_rocm_ridge_train: invalid feature_dim %d (must be 1-10000)",
							   feature_dim);
		return -1;
	}

	elog(DEBUG1,
		 "ndb_rocm_ridge_train: entry: model_data=%p, features=%p, targets=%p, n_samples=%d, feature_dim=%d",
		 model_data,
		 features,
		 targets,
		 n_samples,
		 feature_dim);

	/* Extract and validate lambda from hyperparameters */
	if (hyperparams != NULL)
	{
		Datum		lambda_datum;
		Datum		numeric_datum;
		Numeric		num;
		double		parsed_lambda;

		lambda_datum = DirectFunctionCall2(jsonb_object_field,
										   JsonbPGetDatum(hyperparams),
										   CStringGetTextDatum("lambda"));
		if (DatumGetPointer(lambda_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(jsonb_numeric, lambda_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				parsed_lambda = DatumGetFloat8(
											   DirectFunctionCall1(numeric_float8, NumericGetDatum(num)));
				/* Defensive: Validate lambda range and check for NaN/Inf */
				if (isfinite(parsed_lambda) && parsed_lambda >= 0.0 && parsed_lambda <= 1000.0)
				{
					lambda = parsed_lambda;
				}
				else
				{
					elog(WARNING,
						 "ndb_rocm_ridge_train: invalid lambda %f (must be finite, >= 0, <= 1000.0), using default %f",
						 parsed_lambda, default_lambda);
					lambda = default_lambda;
				}
			}
		}
	}

	/* Defensive: Final validation of lambda */
	if (!isfinite(lambda) || lambda < 0.0 || lambda > 1000.0)
	{
		elog(WARNING,
			 "ndb_rocm_ridge_train: lambda %f invalid, using default %f",
			 lambda, default_lambda);
		lambda = default_lambda;
	}

	dim_with_intercept = feature_dim + 1;

	/* Allocate host memory for matrices */
	XtX_bytes = sizeof(double) * (size_t) dim_with_intercept * (size_t) dim_with_intercept;
	Xty_bytes = sizeof(double) * (size_t) dim_with_intercept;
	beta_bytes = sizeof(double) * (size_t) dim_with_intercept;

	NDB_DECLARE(double *, h_XtX);
	NDB_DECLARE(double *, h_Xty);
	NDB_DECLARE(double *, h_XtX_inv);
	NDB_DECLARE(double *, h_beta);
	NDB_ALLOC(h_XtX, double, dim_with_intercept * dim_with_intercept);
	NDB_ALLOC(h_Xty, double, dim_with_intercept);
	NDB_ALLOC(h_XtX_inv, double, dim_with_intercept * dim_with_intercept);
	NDB_ALLOC(h_beta, double, dim_with_intercept);

	/* Compute X'X and X'y on GPU */
	{
		/* Allocate device memory */
		feature_bytes = sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
		target_bytes = sizeof(double) * (size_t) n_samples;

		elog(DEBUG1,
			 "ndb_rocm_ridge_train: allocating GPU memory: feature_bytes=%zu (%.2f MB), target_bytes=%zu",
			 feature_bytes,
			 feature_bytes / (1024.0 * 1024.0),
			 target_bytes);

		/* Defensive: Check HIP context before proceeding */
		status = hipGetLastError();
		if (status != hipSuccess && status != hipErrorNotReady)
		{
			if (errstr)
				*errstr = psprintf("ndb_rocm_ridge_train: HIP error before allocation: %s",
								   hipGetErrorString(status));
			elog(WARNING,
				 "ndb_rocm_ridge_train: HIP error detected before allocation: %s",
				 hipGetErrorString(status));
			goto cpu_fallback;
		}

		status = hipMalloc((void **) &d_features, feature_bytes);
		if (status != hipSuccess)
		{
			elog(WARNING,
				 "ndb_rocm_ridge_train: hipMalloc d_features failed: %s",
				 hipGetErrorString(status));
			goto cpu_fallback;
		}

		status = hipMalloc((void **) &d_targets, target_bytes);
		if (status != hipSuccess)
		{
			hipFree(d_features);
			elog(WARNING,
				 "ndb_rocm_ridge_train: hipMalloc d_targets failed: %s",
				 hipGetErrorString(status));
			goto cpu_fallback;
		}

		status = hipMalloc((void **) &d_XtX, XtX_bytes);
		if (status != hipSuccess)
		{
			hipFree(d_features);
			hipFree(d_targets);
			elog(WARNING,
				 "ndb_rocm_ridge_train: hipMalloc d_XtX failed: %s",
				 hipGetErrorString(status));
			goto cpu_fallback;
		}

		status = hipMalloc((void **) &d_Xty, Xty_bytes);
		if (status != hipSuccess)
		{
			hipFree(d_features);
			hipFree(d_targets);
			hipFree(d_XtX);
			elog(WARNING,
				 "ndb_rocm_ridge_train: hipMalloc d_Xty failed: %s",
				 hipGetErrorString(status));
			goto cpu_fallback;
		}

		/* Initialize XtX and Xty to zero */
		status = hipMemset(d_XtX, 0, XtX_bytes);
		if (status != hipSuccess)
			goto gpu_cleanup;
		status = hipMemset(d_Xty, 0, Xty_bytes);
		if (status != hipSuccess)
			goto gpu_cleanup;

		/* Copy data to device */
		status = hipMemcpy(d_features, features, feature_bytes, hipMemcpyHostToDevice);
		if (status != hipSuccess)
		{
			elog(WARNING,
				 "ndb_rocm_ridge_train: hipMemcpy d_features failed: %s",
				 hipGetErrorString(status));
			goto gpu_cleanup;
		}
		status = hipMemcpy(d_targets, targets, target_bytes, hipMemcpyHostToDevice);
		if (status != hipSuccess)
		{
			elog(WARNING,
				 "ndb_rocm_ridge_train: hipMemcpy d_targets failed: %s",
				 hipGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Use GPU kernel to compute X'X and X'y */
		status = launch_linreg_compute_xtx_kernel(d_features,
												  d_targets,
												  n_samples,
												  feature_dim,
												  dim_with_intercept,
												  d_XtX,
												  d_Xty);
		if (status != hipSuccess)
		{
			elog(WARNING,
				 "ndb_rocm_ridge_train: launch_linreg_compute_xtx_kernel failed: %s",
				 hipGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Add Ridge penalty (λI) to diagonal (excluding intercept) on GPU */
		status = launch_ridge_add_lambda_diagonal_kernel(d_XtX, lambda, dim_with_intercept);
		if (status != hipSuccess)
		{
			elog(WARNING,
				 "ndb_rocm_ridge_train: launch_ridge_add_lambda_diagonal_kernel failed: %s",
				 hipGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Copy Xty to host for CPU fallback */
		status = hipMemcpy(h_Xty, d_Xty, Xty_bytes, hipMemcpyDeviceToHost);
		if (status != hipSuccess)
		{
			elog(WARNING,
				 "ndb_rocm_ridge_train: hipMemcpy h_Xty failed: %s",
				 hipGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Copy back to GPU for matrix inversion and rocBLAS operations */
		status = hipMemcpy(d_XtX, h_XtX, XtX_bytes, hipMemcpyHostToDevice);
		if (status != hipSuccess)
		{
			elog(WARNING,
				 "ndb_rocm_ridge_train: hipMemcpy d_XtX failed: %s",
				 hipGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Allocate device memory for XtX_inv and beta */
		status = hipMalloc((void **) &d_XtX_inv, XtX_bytes);
		if (status != hipSuccess)
		{
			elog(WARNING,
				 "ndb_rocm_ridge_train: hipMalloc d_XtX_inv failed: %s",
				 hipGetErrorString(status));
			goto gpu_cleanup;
		}

		status = hipMalloc((void **) &d_beta, beta_bytes);
		if (status != hipSuccess)
		{
			hipFree(d_XtX_inv);
			elog(WARNING,
				 "ndb_rocm_ridge_train: hipMalloc d_beta failed: %s",
				 hipGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Free feature and target memory (no longer needed) */
		hipFree(d_features);
		hipFree(d_targets);
		d_features = NULL;
		d_targets = NULL;

		/* Copy XtX to host for matrix inversion (CPU) */
		/* Matrix inversion is small, CPU is fine */
		goto matrix_inversion;
	}

gpu_cleanup:
	if (d_features)
		hipFree(d_features);
	if (d_targets)
		hipFree(d_targets);
	if (d_XtX)
		hipFree(d_XtX);
	if (d_Xty)
		hipFree(d_Xty);
	if (d_XtX_inv)
		hipFree(d_XtX_inv);
	if (d_beta)
		hipFree(d_beta);

cpu_fallback:
	/* Fallback to CPU computation */
	elog(DEBUG1, "ndb_rocm_ridge_train: falling back to CPU computation");
	for (i = 0; i < n_samples; i++)
	{
		const float *row = features + (i * feature_dim);
		double	   *xi;

		NDB_DECLARE(double *, xi);
		NDB_ALLOC(xi, double, dim_with_intercept);

		xi[0] = 1.0;			/* intercept */
		for (j = 1; j < dim_with_intercept; j++)
			xi[j] = row[j - 1];

		/* X'X accumulation */
		for (j = 0; j < dim_with_intercept; j++)
		{
			for (int k = 0; k < dim_with_intercept; k++)
				h_XtX[j * dim_with_intercept + k] += xi[j] * xi[k];

			/* X'y accumulation */
			h_Xty[j] += xi[j] * targets[i];
		}

		NDB_FREE(xi);
	}

	/* Add Ridge penalty (λI) to diagonal (excluding intercept) */
	for (i = 1; i < dim_with_intercept; i++)
		h_XtX[i * dim_with_intercept + i] += lambda;

matrix_inversion:

	/* Invert X'X + λI using Gauss-Jordan elimination */
	{
		double	  **augmented;
		int			row,
					col,
					k_local;
		double		pivot,
					factor;
		bool		invert_success = true;

		/* Create augmented matrix [A | I] */
		NDB_DECLARE(double **, augmented);
		NDB_ALLOC(augmented, double *, dim_with_intercept);
		for (row = 0; row < dim_with_intercept; row++)
		{
			NDB_ALLOC(augmented[row], double, 2 * dim_with_intercept);
			for (col = 0; col < dim_with_intercept; col++)
			{
				augmented[row][col] = h_XtX[row * dim_with_intercept + col];
				augmented[row][col + dim_with_intercept] = (row == col) ? 1.0 : 0.0;
			}
		}

		/* Gauss-Jordan elimination */
		for (row = 0; row < dim_with_intercept; row++)
		{
			pivot = augmented[row][row];
			if (fabs(pivot) < 1e-10)
			{
				bool		found = false;

				for (k_local = row + 1; k_local < dim_with_intercept; k_local++)
				{
					if (fabs(augmented[k_local][row]) > 1e-10)
					{
						double	   *temp = augmented[row];

						augmented[row] = augmented[k_local];
						augmented[k_local] = temp;
						pivot = augmented[row][row];
						found = true;
						break;
					}
				}
				if (!found)
				{
					invert_success = false;
					break;
				}
			}

			for (col = 0; col < 2 * dim_with_intercept; col++)
				augmented[row][col] /= pivot;

			for (k_local = 0; k_local < dim_with_intercept; k_local++)
			{
				if (k_local != row)
				{
					factor = augmented[k_local][row];
					for (col = 0; col < 2 * dim_with_intercept; col++)
						augmented[k_local][col] -= factor * augmented[row][col];
				}
			}
		}

		if (invert_success)
		{
			for (row = 0; row < dim_with_intercept; row++)
				for (col = 0; col < dim_with_intercept; col++)
					h_XtX_inv[row * dim_with_intercept + col] =
						augmented[row][col + dim_with_intercept];
		}

		for (row = 0; row < dim_with_intercept; row++)
			NDB_FREE(augmented[row]);
		NDB_FREE(augmented);

		if (!invert_success)
		{
			NDB_FREE(h_XtX);
			NDB_FREE(h_Xty);
			NDB_FREE(h_XtX_inv);
			NDB_FREE(h_beta);
			if (d_XtX)
				hipFree(d_XtX);
			if (d_Xty)
				hipFree(d_Xty);
			if (d_XtX_inv)
				hipFree(d_XtX_inv);
			if (d_beta)
				hipFree(d_beta);
			if (errstr)
				*errstr = pstrdup("Matrix is singular, cannot compute Ridge regression");
			return -1;
		}

		/* Copy XtX_inv to GPU for rocBLAS operations */
		if (d_XtX_inv != NULL)
		{
			status = hipMemcpy(d_XtX_inv, h_XtX_inv, XtX_bytes, hipMemcpyHostToDevice);
			if (status != hipSuccess)
			{
				elog(WARNING,
					 "ndb_rocm_ridge_train: hipMemcpy d_XtX_inv failed: %s",
					 hipGetErrorString(status));
				/* Fallback to CPU computation */
			}
			else
			{
				/* Use rocBLAS DGEMM for β = (X'X + λI)^(-1)X'y */
				rocblas_handle handle = ndb_rocm_get_cublas_handle();

				if (handle != NULL && d_Xty != NULL && d_beta != NULL)
				{
					rocblas_status rocblas_err;
					double		alpha = 1.0;
					double		beta = 0.0;

					/* DGEMM: beta = alpha * XtX_inv * Xty + beta * beta */
					/* XtX_inv is (dim_with_intercept x dim_with_intercept) */
					/* Xty is (dim_with_intercept x 1) */
					/* Result beta is (dim_with_intercept x 1) */
					rocblas_err = cublasDgemv(handle,
											  rocblas_operation_none,	/* No transpose */
											  dim_with_intercept,	/* M: rows of XtX_inv */
											  dim_with_intercept,	/* N: cols of XtX_inv */
											  &alpha,	/* alpha = 1.0 */
											  d_XtX_inv,	/* XtX_inv: (dim x dim) */
											  dim_with_intercept,	/* lda: leading
																	 * dimension */
											  d_Xty,	/* Xty: (dim x 1) */
											  1,	/* incx: stride */
											  &beta,	/* beta = 0.0 */
											  d_beta,	/* beta: (dim x 1) */
											  1);	/* incy: stride */

					if (rocblas_err == rocblas_status_success)
					{
						/* Copy beta back to host */
						status = hipMemcpy(h_beta, d_beta, beta_bytes, hipMemcpyDeviceToHost);
						if (status == hipSuccess)
						{
							/* Success - use GPU-computed beta */
							goto build_model;
						}
					}
					else
					{
						elog(WARNING,
							 "ndb_rocm_ridge_train: cublasDgemv failed: %d, falling back to CPU",
							 (int) rocblas_err);
					}
				}
			}
		}
	}

	/* Fallback to CPU computation: β = (X'X + λI)^(-1)X'y */
	for (i = 0; i < dim_with_intercept; i++)
	{
		h_beta[i] = 0.0;
		for (j = 0; j < dim_with_intercept; j++)
			h_beta[i] += h_XtX_inv[i * dim_with_intercept + j] * h_Xty[j];
	}

build_model:

	/* Build model */
	{
		RidgeModel	model;
		double		y_mean = 0.0;
		double		ss_tot = 0.0;
		double		ss_res = 0.0;
		double		mse = 0.0;
		double		mae = 0.0;

		model.n_features = feature_dim;
		model.n_samples = n_samples;
		model.intercept = h_beta[0];
		model.lambda = lambda;
		NDB_DECLARE(double *, model_coefficients);
		NDB_ALLOC(model_coefficients, double, feature_dim);
		model.coefficients = model_coefficients;
		for (i = 0; i < feature_dim; i++)
			model.coefficients[i] = h_beta[i + 1];

		/* Compute metrics */
		for (i = 0; i < n_samples; i++)
			y_mean += targets[i];
		y_mean /= n_samples;

		for (i = 0; i < n_samples; i++)
		{
			const float *row = features + (i * feature_dim);
			double		y_pred = model.intercept;
			double		error;
			int			j_local;

			for (j_local = 0; j_local < feature_dim; j_local++)
				y_pred += model.coefficients[j_local] * row[j_local];

			error = targets[i] - y_pred;
			mse += error * error;
			mae += fabs(error);
			ss_res += error * error;
			ss_tot += (targets[i] - y_mean) * (targets[i] - y_mean);
		}

		mse /= n_samples;
		mae /= n_samples;
		model.r_squared = (ss_tot > 0.0) ? (1.0 - (ss_res / ss_tot)) : 0.0;
		model.mse = mse;
		model.mae = mae;

		/* Pack model */
		rc = ndb_rocm_ridge_pack_model(
									   &model, &payload, &metrics_json, errstr);

		NDB_FREE(model.coefficients);
	}

	NDB_FREE(h_XtX);
	NDB_FREE(h_Xty);
	NDB_FREE(h_XtX_inv);
	NDB_FREE(h_beta);
	if (d_XtX)
		hipFree(d_XtX);
	if (d_Xty)
		hipFree(d_Xty);
	if (d_XtX_inv)
		hipFree(d_XtX_inv);
	if (d_beta)
		hipFree(d_beta);

	if (rc == 0 && payload != NULL)
	{
		*model_data = payload;
		if (metrics != NULL)
			*metrics = metrics_json;
		return 0;
	}

	if (payload != NULL)
		NDB_FREE(payload);
	if (metrics_json != NULL)
		NDB_FREE(metrics_json);

	return -1;
}

int
ndb_rocm_ridge_predict(const bytea * model_data,
					   const float *input,
					   int feature_dim,
					   double *prediction_out,
					   char **errstr)
{
	const NdbCudaRidgeModelHeader *hdr;
	const float *coefficients;
	const		bytea *detoasted;
	double		prediction;
	int			i;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || prediction_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for HIP Ridge predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted =
		(const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t		expected_size = sizeof(NdbCudaRidgeModelHeader)
			+ sizeof(float) * (size_t) feature_dim;
		size_t		actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr = psprintf("model data too small: "
								   "expected %zu bytes, got %zu",
								   expected_size,
								   actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaRidgeModelHeader *) VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		return -1;
	}

	coefficients = (const float *) ((const char *) hdr + sizeof(NdbCudaRidgeModelHeader));

	prediction = hdr->intercept;
	for (i = 0; i < feature_dim; i++)
		prediction += coefficients[i] * input[i];

	*prediction_out = prediction;
	return 0;
}

/*
 * ndb_rocm_ridge_evaluate
 *    GPU-accelerated batch evaluation for Ridge Regression
 *
 * Reuses the linear regression evaluation kernel since evaluation
 * is identical (just computes predictions and metrics).
 */
int
ndb_rocm_ridge_evaluate(const bytea * model_data,
						const float *features,
						const double *targets,
						int n_samples,
						int feature_dim,
						double *mse_out,
						double *mae_out,
						double *rmse_out,
						double *r_squared_out,
						char **errstr)
{
	const NdbCudaRidgeModelHeader *hdr;
	const float *coefficients;
	const		bytea *detoasted;
	hipError_t	cuda_err;
	float	   *d_features = NULL;
	double	   *d_targets = NULL;
	double	   *d_coefficients = NULL;
	double	   *d_sse = NULL;
	double	   *d_sae = NULL;
	long long  *d_count = NULL;
	double		h_sse = 0.0;
	double		h_sae = 0.0;
	long long	h_count = 0;
	double		y_mean = 0.0;
	double		ss_tot = 0.0;
	double		mse = 0.0;
	double		mae = 0.0;
	double		rmse = 0.0;
	double		r_squared = 0.0;
	size_t		feature_bytes;
	size_t		target_bytes;
	size_t		coeff_bytes;
	int			i;

	if (errstr)
		*errstr = NULL;

	/* Defensive parameter validation */
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_rocm_ridge_evaluate: model_data is NULL");
		return -1;
	}

	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_rocm_ridge_evaluate: features is NULL");
		return -1;
	}

	if (targets == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_rocm_ridge_evaluate: targets is NULL");
		return -1;
	}

	if (n_samples <= 0)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: invalid n_samples %d",
							   n_samples);
		return -1;
	}

	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: invalid feature_dim %d",
							   feature_dim);
		return -1;
	}

	if (mse_out == NULL || mae_out == NULL || rmse_out == NULL || r_squared_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_rocm_ridge_evaluate: output pointers are NULL");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t		expected_size = sizeof(NdbCudaRidgeModelHeader)
			+ sizeof(float) * (size_t) feature_dim;
		size_t		actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: model data too small: "
								   "expected %zu bytes, got %zu",
								   expected_size,
								   actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaRidgeModelHeader *) VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: feature dimension mismatch: "
							   "model has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		return -1;
	}

	coefficients = (const float *) ((const char *) hdr + sizeof(NdbCudaRidgeModelHeader));

	/* Compute y_mean for R-squared calculation */
	for (i = 0; i < n_samples; i++)
	{
		if (targets != NULL)
			y_mean += targets[i];
	}
	if (n_samples > 0)
		y_mean /= (double) n_samples;

	/* Allocate GPU memory for features */
	feature_bytes = sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
	cuda_err = hipMalloc((void **) &d_features, feature_bytes);
	if (cuda_err != hipSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to allocate GPU memory for features: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	/* Allocate GPU memory for targets */
	target_bytes = sizeof(double) * (size_t) n_samples;
	cuda_err = hipMalloc((void **) &d_targets, target_bytes);
	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to allocate GPU memory for targets: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	/* Allocate GPU memory for coefficients */
	coeff_bytes = sizeof(double) * (size_t) feature_dim;
	cuda_err = hipMalloc((void **) &d_coefficients, coeff_bytes);
	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		hipFree(d_targets);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to allocate GPU memory for coefficients: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	/* Allocate GPU memory for output accumulators */
	cuda_err = hipMalloc((void **) &d_sse, sizeof(double));
	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		hipFree(d_targets);
		hipFree(d_coefficients);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to allocate GPU memory for SSE: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	cuda_err = hipMalloc((void **) &d_sae, sizeof(double));
	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		hipFree(d_targets);
		hipFree(d_coefficients);
		hipFree(d_sse);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to allocate GPU memory for SAE: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	cuda_err = hipMalloc((void **) &d_count, sizeof(long long));
	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		hipFree(d_targets);
		hipFree(d_coefficients);
		hipFree(d_sse);
		hipFree(d_sae);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to allocate GPU memory for count: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	/* Copy features to GPU */
	cuda_err = hipMemcpy(d_features, features, feature_bytes, hipMemcpyHostToDevice);
	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		hipFree(d_targets);
		hipFree(d_coefficients);
		hipFree(d_sse);
		hipFree(d_sae);
		hipFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to copy features to GPU: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	/* Copy targets to GPU */
	cuda_err = hipMemcpy(d_targets, targets, target_bytes, hipMemcpyHostToDevice);
	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		hipFree(d_targets);
		hipFree(d_coefficients);
		hipFree(d_sse);
		hipFree(d_sae);
		hipFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to copy targets to GPU: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	/* Convert coefficients from float to double and copy to GPU */
	{
		NDB_DECLARE(double *, h_coefficients_double);
		NDB_ALLOC(h_coefficients_double, double, feature_dim);

		if (h_coefficients_double == NULL)
		{
			hipFree(d_features);
			hipFree(d_targets);
			hipFree(d_coefficients);
			hipFree(d_sse);
			hipFree(d_sae);
			hipFree(d_count);
			if (errstr)
				*errstr = pstrdup("neurondb: ndb_rocm_ridge_evaluate: failed to allocate host memory for coefficients");
			return -1;
		}

		for (i = 0; i < feature_dim; i++)
			h_coefficients_double[i] = (double) coefficients[i];

		cuda_err = hipMemcpy(d_coefficients, h_coefficients_double, coeff_bytes, hipMemcpyHostToDevice);
		NDB_FREE(h_coefficients_double);

		if (cuda_err != hipSuccess)
		{
			hipFree(d_features);
			hipFree(d_targets);
			hipFree(d_coefficients);
			hipFree(d_sse);
			hipFree(d_sae);
			hipFree(d_count);
			if (errstr)
				*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to copy coefficients to GPU: %s",
								   hipGetErrorString(cuda_err));
			return -1;
		}
	}

	/* Launch evaluation kernel (reuse linear regression kernel) */
	cuda_err = launch_linreg_eval_kernel(d_features,
										 d_targets,
										 d_coefficients,
										 (double) hdr->intercept,
										 n_samples,
										 feature_dim,
										 d_sse,
										 d_sae,
										 d_count);

	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		hipFree(d_targets);
		hipFree(d_coefficients);
		hipFree(d_sse);
		hipFree(d_sae);
		hipFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: evaluation kernel failed: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	/* Copy results back to host */
	cuda_err = hipMemcpy(&h_sse, d_sse, sizeof(double), hipMemcpyDeviceToHost);
	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		hipFree(d_targets);
		hipFree(d_coefficients);
		hipFree(d_sse);
		hipFree(d_sae);
		hipFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to copy SSE from GPU: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	cuda_err = hipMemcpy(&h_sae, d_sae, sizeof(double), hipMemcpyDeviceToHost);
	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		hipFree(d_targets);
		hipFree(d_coefficients);
		hipFree(d_sse);
		hipFree(d_sae);
		hipFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to copy SAE from GPU: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	cuda_err = hipMemcpy(&h_count, d_count, sizeof(long long), hipMemcpyDeviceToHost);
	if (cuda_err != hipSuccess)
	{
		hipFree(d_features);
		hipFree(d_targets);
		hipFree(d_coefficients);
		hipFree(d_sse);
		hipFree(d_sae);
		hipFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: failed to copy count from GPU: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	/* Cleanup GPU memory */
	hipFree(d_features);
	hipFree(d_targets);
	hipFree(d_coefficients);
	hipFree(d_sse);
	hipFree(d_sae);
	hipFree(d_count);

	/* Defensive check: ensure count matches expected */
	if (h_count != (long long) n_samples)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_ridge_evaluate: count mismatch: expected %d, got %lld",
							   n_samples,
							   (long long) h_count);
		return -1;
	}

	/* Compute final metrics */
	if (h_count > 0)
	{
		mse = h_sse / (double) h_count;
		mae = h_sae / (double) h_count;
		rmse = sqrt(mse);

		/* Compute SS_tot for R-squared */
		for (i = 0; i < n_samples; i++)
		{
			if (targets != NULL)
			{
				double		diff = targets[i] - y_mean;

				ss_tot += diff * diff;
			}
		}

		if (ss_tot > 1e-10)
			r_squared = 1.0 - (h_sse / ss_tot);
		else
			r_squared = 0.0;
	}
	else
	{
		mse = 0.0;
		mae = 0.0;
		rmse = 0.0;
		r_squared = 0.0;
	}

	/* Write outputs */
	*mse_out = mse;
	*mae_out = mae;
	*rmse_out = rmse;
	*r_squared_out = r_squared;

	return 0;
}

#endif							/* NDB_GPU_HIP */
