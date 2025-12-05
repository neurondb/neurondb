/*-------------------------------------------------------------------------
 *
 * gpu_ridge_cuda.c
 *    Ridge regression bridge implementation.
 *
 * This module provides bridge functions for ridge regression training
 * and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_ridge_cuda.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_CUDA

#include <float.h>
#include <math.h>
#include <string.h>

#include "neurondb_cuda_runtime.h"
#include "neurondb_cuda_launchers.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"

#include "ml_ridge_regression_internal.h"
#include "neurondb_cuda_ridge.h"

#ifdef NDB_GPU_CUDA
#include <cublas_v2.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_guc.h"
#include "neurondb_constants.h"
#endif

/* Reuse linear regression kernels */
extern cudaError_t launch_linreg_compute_xtx_kernel(const float *features,
													const double *targets,
													int n_samples,
													int feature_dim,
													int dim_with_intercept,
													double *XtX,
													double *Xty);
extern cudaError_t launch_linreg_eval_kernel(const float *features,
											 const double *targets,
											 const double *coefficients,
											 double intercept,
											 int n_samples,
											 int feature_dim,
											 double *sse_out,
											 double *sae_out,
											 long long *count_out);
extern cudaError_t launch_ridge_add_lambda_diagonal_kernel(double *matrix,
														   double lambda,
														   int dim);

int
ndb_cuda_ridge_pack_model(const RidgeModel *model,
						  bytea * *model_data,
						  Jsonb * *metrics,
						  char **errstr)
{
	size_t		payload_bytes;
	char	   *base;
	NdbCudaRidgeModelHeader *hdr;
	float	   *coef_dest;
	NDB_DECLARE(bytea *, blob);
	NDB_DECLARE(char *, blob_raw);

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid Ridge model for CUDA pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaRidgeModelHeader)
		+ sizeof(float) * (size_t) model->n_features;
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
ndb_cuda_ridge_train(const float *features,
					 const double *targets,
					 int n_samples,
					 int feature_dim,
					 const Jsonb * hyperparams,
					 bytea * *model_data,
					 Jsonb * *metrics,
					 char **errstr)
{
	const double default_lambda = 0.01;
	double		lambda;
	float	   *d_features;
	double	   *d_targets;
	double	   *d_XtX;
	double	   *d_Xty;
	double	   *d_XtX_inv;
	double	   *d_beta;
	NDB_DECLARE(double *, h_XtX);
	NDB_DECLARE(double *, h_Xty);
	NDB_DECLARE(double *, h_XtX_inv);
	NDB_DECLARE(double *, h_beta);
	bytea	   *payload;
	Jsonb	   *metrics_json;
	cudaError_t status;
	size_t		feature_bytes;
	size_t		target_bytes;
	size_t		XtX_bytes;
	size_t		Xty_bytes;
	size_t		beta_bytes;
	int			dim_with_intercept;
	int			i;
	int			j;
	int			rc;

	/* CPU mode: never execute GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = pstrdup("CUDA ridge_train: CPU mode - GPU code should not be called");
		return -1;
	}

	lambda = default_lambda;
	d_features = NULL;
	d_targets = NULL;
	d_XtX = NULL;
	d_Xty = NULL;
	d_XtX_inv = NULL;
	d_beta = NULL;
	h_XtX = NULL;
	h_Xty = NULL;
	h_XtX_inv = NULL;
	h_beta = NULL;
	payload = NULL;
	metrics_json = NULL;
	status = cudaSuccess;
	rc = -1;

	if (errstr)
		*errstr = NULL;

	/* Defensive: Comprehensive parameter validation */
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ndb_cuda_ridge_train: model_data is NULL");
		return -1;
	}

	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ndb_cuda_ridge_train: features is NULL");
		return -1;
	}

	if (targets == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ndb_cuda_ridge_train: targets is NULL");
		return -1;
	}

	if (n_samples <= 0 || n_samples > 10000000)
	{
		if (errstr)
			*errstr = psprintf("ndb_cuda_ridge_train: invalid n_samples %d (must be 1-10000000)",
							   n_samples);
		return -1;
	}

	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = psprintf("ndb_cuda_ridge_train: invalid feature_dim %d (must be 1-10000)",
							   feature_dim);
		return -1;
	}

	elog(DEBUG1,
		 "ndb_cuda_ridge_train: entry: model_data=%p, features=%p, targets=%p, n_samples=%d, feature_dim=%d",
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
						 "ndb_cuda_ridge_train: invalid lambda %f (must be finite, >= 0, <= 1000.0), using default %f",
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
			 "ndb_cuda_ridge_train: lambda %f invalid, using default %f",
			 lambda, default_lambda);
		lambda = default_lambda;
	}

	dim_with_intercept = feature_dim + 1;

	/* Allocate host memory for matrices */
	XtX_bytes = sizeof(double) * (size_t) dim_with_intercept * (size_t) dim_with_intercept;
	Xty_bytes = sizeof(double) * (size_t) dim_with_intercept;
	beta_bytes = sizeof(double) * (size_t) dim_with_intercept;

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
			 "ndb_cuda_ridge_train: allocating GPU memory: feature_bytes=%zu (%.2f MB), target_bytes=%zu",
			 feature_bytes,
			 feature_bytes / (1024.0 * 1024.0),
			 target_bytes);

		/* Defensive: Check CUDA context before proceeding */
		status = cudaGetLastError();
		if (status != cudaSuccess && status != cudaErrorNotReady)
		{
			if (errstr)
				*errstr = psprintf("ndb_cuda_ridge_train: CUDA error before allocation: %s",
								   cudaGetErrorString(status));
			elog(WARNING,
				 "ndb_cuda_ridge_train: CUDA error detected before allocation: %s",
				 cudaGetErrorString(status));
			goto cpu_fallback;
		}

		status = cudaMalloc((void **) &d_features, feature_bytes);
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_ridge_train: cudaMalloc d_features failed: %s",
				 cudaGetErrorString(status));
			goto cpu_fallback;
		}

		status = cudaMalloc((void **) &d_targets, target_bytes);
		if (status != cudaSuccess)
		{
			cudaFree(d_features);
			elog(WARNING,
				 "ndb_cuda_ridge_train: cudaMalloc d_targets failed: %s",
				 cudaGetErrorString(status));
			goto cpu_fallback;
		}

		status = cudaMalloc((void **) &d_XtX, XtX_bytes);
		if (status != cudaSuccess)
		{
			cudaFree(d_features);
			cudaFree(d_targets);
			elog(WARNING,
				 "ndb_cuda_ridge_train: cudaMalloc d_XtX failed: %s",
				 cudaGetErrorString(status));
			goto cpu_fallback;
		}

		status = cudaMalloc((void **) &d_Xty, Xty_bytes);
		if (status != cudaSuccess)
		{
			cudaFree(d_features);
			cudaFree(d_targets);
			cudaFree(d_XtX);
			elog(WARNING,
				 "ndb_cuda_ridge_train: cudaMalloc d_Xty failed: %s",
				 cudaGetErrorString(status));
			goto cpu_fallback;
		}

		/* Initialize XtX and Xty to zero */
		status = cudaMemset(d_XtX, 0, XtX_bytes);
		if (status != cudaSuccess)
			goto gpu_cleanup;
		status = cudaMemset(d_Xty, 0, Xty_bytes);
		if (status != cudaSuccess)
			goto gpu_cleanup;

		/* Copy data to device */
		status = cudaMemcpy(d_features, features, feature_bytes, cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_ridge_train: cudaMemcpy d_features failed: %s",
				 cudaGetErrorString(status));
			goto gpu_cleanup;
		}
		status = cudaMemcpy(d_targets, targets, target_bytes, cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_ridge_train: cudaMemcpy d_targets failed: %s",
				 cudaGetErrorString(status));
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
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_ridge_train: launch_linreg_compute_xtx_kernel failed: %s",
				 cudaGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Add Ridge penalty (λI) to diagonal (excluding intercept) on GPU */
		status = launch_ridge_add_lambda_diagonal_kernel(d_XtX, lambda, dim_with_intercept);
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_ridge_train: launch_ridge_add_lambda_diagonal_kernel failed: %s",
				 cudaGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Copy Xty to host for CPU fallback */
		status = cudaMemcpy(h_Xty, d_Xty, Xty_bytes, cudaMemcpyDeviceToHost);
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_ridge_train: cudaMemcpy h_Xty failed: %s",
				 cudaGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Copy back to GPU for matrix inversion and cuBLAS operations */
		status = cudaMemcpy(d_XtX, h_XtX, XtX_bytes, cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_ridge_train: cudaMemcpy d_XtX failed: %s",
				 cudaGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Allocate device memory for XtX_inv and beta */
		status = cudaMalloc((void **) &d_XtX_inv, XtX_bytes);
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_ridge_train: cudaMalloc d_XtX_inv failed: %s",
				 cudaGetErrorString(status));
			goto gpu_cleanup;
		}

		status = cudaMalloc((void **) &d_beta, beta_bytes);
		if (status != cudaSuccess)
		{
			cudaFree(d_XtX_inv);
			elog(WARNING,
				 "ndb_cuda_ridge_train: cudaMalloc d_beta failed: %s",
				 cudaGetErrorString(status));
			goto gpu_cleanup;
		}

		/* Free feature and target memory (no longer needed) */
		cudaFree(d_features);
		cudaFree(d_targets);
		d_features = NULL;
		d_targets = NULL;

		/* Copy XtX to host for matrix inversion (CPU) */
		/* Matrix inversion is small, CPU is fine */
		goto matrix_inversion;
	}

gpu_cleanup:
	if (d_features)
		cudaFree(d_features);
	if (d_targets)
		cudaFree(d_targets);
	if (d_XtX)
		cudaFree(d_XtX);
	if (d_Xty)
		cudaFree(d_Xty);
	if (d_XtX_inv)
		cudaFree(d_XtX_inv);
	if (d_beta)
		cudaFree(d_beta);

cpu_fallback:
	/* Fallback to CPU computation */
	elog(DEBUG1, "ndb_cuda_ridge_train: falling back to CPU computation");
	for (i = 0; i < n_samples; i++)
	{
		const float *row = features + (i * feature_dim);
		double	   *xi;

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
		double	  **augmented = NULL;
		int			row,
				col,
				k_local;
		double		pivot,
				factor;
		bool		invert_success = true;

		/* Create augmented matrix [A | I] */
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
				cudaFree(d_XtX);
			if (d_Xty)
				cudaFree(d_Xty);
			if (d_XtX_inv)
				cudaFree(d_XtX_inv);
			if (d_beta)
				cudaFree(d_beta);
			if (errstr)
				*errstr = pstrdup("Matrix is singular, cannot compute Ridge regression");
			return -1;
		}

		/* Copy XtX_inv to GPU for cuBLAS operations */
		if (d_XtX_inv != NULL)
		{
			status = cudaMemcpy(d_XtX_inv, h_XtX_inv, XtX_bytes, cudaMemcpyHostToDevice);
			if (status != cudaSuccess)
			{
				elog(WARNING,
					 "ndb_cuda_ridge_train: cudaMemcpy d_XtX_inv failed: %s",
					 cudaGetErrorString(status));
				/* Fallback to CPU computation */
			}
			else
			{
				/* Use cuBLAS DGEMM for β = (X'X + λI)^(-1)X'y */
				cublasHandle_t handle = ndb_cuda_get_cublas_handle();

				if (handle != NULL && d_Xty != NULL && d_beta != NULL)
				{
					cublasStatus_t cublas_err;
					double		alpha = 1.0;
					double		beta = 0.0;

					/* DGEMM: beta = alpha * XtX_inv * Xty + beta * beta */
					/* XtX_inv is (dim_with_intercept x dim_with_intercept) */
					/* Xty is (dim_with_intercept x 1) */
					/* Result beta is (dim_with_intercept x 1) */
					cublas_err = cublasDgemv(handle,
											 CUBLAS_OP_N,	/* No transpose */
											 dim_with_intercept,	/* M: rows of XtX_inv */
											 dim_with_intercept,	/* N: cols of XtX_inv */
											 &alpha,	/* alpha = 1.0 */
											 d_XtX_inv, /* XtX_inv: (dim x dim) */
											 dim_with_intercept,	/* lda: leading
																	 * dimension */
											 d_Xty, /* Xty: (dim x 1) */
											 1, /* incx: stride */
											 &beta, /* beta = 0.0 */
											 d_beta,	/* beta: (dim x 1) */
											 1);	/* incy: stride */

					if (cublas_err == CUBLAS_STATUS_SUCCESS)
					{
						/* Copy beta back to host */
						status = cudaMemcpy(h_beta, d_beta, beta_bytes, cudaMemcpyDeviceToHost);
						if (status == cudaSuccess)
						{
							/* Success - use GPU-computed beta */
							goto build_model;
						}
					}
					else
					{
						elog(WARNING,
							 "ndb_cuda_ridge_train: cublasDgemv failed: %d, falling back to CPU",
							 (int) cublas_err);
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
		NDB_DECLARE(double *, model_coefficients);

		model.n_features = feature_dim;
		model.n_samples = n_samples;
		model.intercept = h_beta[0];
		model.lambda = lambda;
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
		rc = ndb_cuda_ridge_pack_model(
									   &model, &payload, &metrics_json, errstr);

		NDB_FREE(model.coefficients);
	}

	NDB_FREE(h_XtX);
	NDB_FREE(h_Xty);
	NDB_FREE(h_XtX_inv);
	NDB_FREE(h_beta);
	if (d_XtX)
		cudaFree(d_XtX);
	if (d_Xty)
		cudaFree(d_Xty);
	if (d_XtX_inv)
		cudaFree(d_XtX_inv);
	if (d_beta)
		cudaFree(d_beta);

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
ndb_cuda_ridge_predict(const bytea * model_data,
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
			*errstr = pstrdup("invalid parameters for CUDA Ridge predict");
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
 * ndb_cuda_ridge_evaluate
 *    GPU-accelerated batch evaluation for Ridge Regression
 *
 * Reuses the linear regression evaluation kernel since evaluation
 * is identical (just computes predictions and metrics).
 */
int
ndb_cuda_ridge_evaluate(const bytea * model_data,
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
	cudaError_t cuda_err;
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
			*errstr = pstrdup("neurondb: ndb_cuda_ridge_evaluate: model_data is NULL");
		return -1;
	}

	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_cuda_ridge_evaluate: features is NULL");
		return -1;
	}

	if (targets == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_cuda_ridge_evaluate: targets is NULL");
		return -1;
	}

	if (n_samples <= 0)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: invalid n_samples %d",
							   n_samples);
		return -1;
	}

	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: invalid feature_dim %d",
							   feature_dim);
		return -1;
	}

	if (mse_out == NULL || mae_out == NULL || rmse_out == NULL || r_squared_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_cuda_ridge_evaluate: output pointers are NULL");
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
				*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: model data too small: "
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
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: feature dimension mismatch: "
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
	cuda_err = cudaMalloc((void **) &d_features, feature_bytes);
	if (cuda_err != cudaSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to allocate GPU memory for features: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Allocate GPU memory for targets */
	target_bytes = sizeof(double) * (size_t) n_samples;
	cuda_err = cudaMalloc((void **) &d_targets, target_bytes);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to allocate GPU memory for targets: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Allocate GPU memory for coefficients */
	coeff_bytes = sizeof(double) * (size_t) feature_dim;
	cuda_err = cudaMalloc((void **) &d_coefficients, coeff_bytes);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_targets);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to allocate GPU memory for coefficients: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Allocate GPU memory for output accumulators */
	cuda_err = cudaMalloc((void **) &d_sse, sizeof(double));
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_coefficients);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to allocate GPU memory for SSE: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	cuda_err = cudaMalloc((void **) &d_sae, sizeof(double));
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_coefficients);
		cudaFree(d_sse);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to allocate GPU memory for SAE: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	cuda_err = cudaMalloc((void **) &d_count, sizeof(long long));
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_coefficients);
		cudaFree(d_sse);
		cudaFree(d_sae);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to allocate GPU memory for count: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Copy features to GPU */
	cuda_err = cudaMemcpy(d_features, features, feature_bytes, cudaMemcpyHostToDevice);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_coefficients);
		cudaFree(d_sse);
		cudaFree(d_sae);
		cudaFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to copy features to GPU: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Copy targets to GPU */
	cuda_err = cudaMemcpy(d_targets, targets, target_bytes, cudaMemcpyHostToDevice);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_coefficients);
		cudaFree(d_sse);
		cudaFree(d_sae);
		cudaFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to copy targets to GPU: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Convert coefficients from float to double and copy to GPU */
	{
		NDB_DECLARE(double *, h_coefficients_double);
		NDB_ALLOC(h_coefficients_double, double, feature_dim);

		if (h_coefficients_double == NULL)
		{
			cudaFree(d_features);
			cudaFree(d_targets);
			cudaFree(d_coefficients);
			cudaFree(d_sse);
			cudaFree(d_sae);
			cudaFree(d_count);
			if (errstr)
				*errstr = pstrdup("neurondb: ndb_cuda_ridge_evaluate: failed to allocate host memory for coefficients");
			return -1;
		}

		for (i = 0; i < feature_dim; i++)
			h_coefficients_double[i] = (double) coefficients[i];

		cuda_err = cudaMemcpy(d_coefficients, h_coefficients_double, coeff_bytes, cudaMemcpyHostToDevice);
		NDB_FREE(h_coefficients_double);

		if (cuda_err != cudaSuccess)
		{
			cudaFree(d_features);
			cudaFree(d_targets);
			cudaFree(d_coefficients);
			cudaFree(d_sse);
			cudaFree(d_sae);
			cudaFree(d_count);
			if (errstr)
				*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to copy coefficients to GPU: %s",
								   cudaGetErrorString(cuda_err));
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

	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_coefficients);
		cudaFree(d_sse);
		cudaFree(d_sae);
		cudaFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: evaluation kernel failed: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Copy results back to host */
	cuda_err = cudaMemcpy(&h_sse, d_sse, sizeof(double), cudaMemcpyDeviceToHost);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_coefficients);
		cudaFree(d_sse);
		cudaFree(d_sae);
		cudaFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to copy SSE from GPU: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	cuda_err = cudaMemcpy(&h_sae, d_sae, sizeof(double), cudaMemcpyDeviceToHost);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_coefficients);
		cudaFree(d_sse);
		cudaFree(d_sae);
		cudaFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to copy SAE from GPU: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	cuda_err = cudaMemcpy(&h_count, d_count, sizeof(long long), cudaMemcpyDeviceToHost);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_coefficients);
		cudaFree(d_sse);
		cudaFree(d_sae);
		cudaFree(d_count);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: failed to copy count from GPU: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Cleanup GPU memory */
	cudaFree(d_features);
	cudaFree(d_targets);
	cudaFree(d_coefficients);
	cudaFree(d_sse);
	cudaFree(d_sae);
	cudaFree(d_count);

	/* Defensive check: ensure count matches expected */
	if (h_count != (long long) n_samples)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_ridge_evaluate: count mismatch: expected %d, got %lld",
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

#endif							/* NDB_GPU_CUDA */
