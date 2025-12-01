/*-------------------------------------------------------------------------
 *
 * gpu_linreg_rocm.c
 *    ROCm backend bridge for Linear Regression training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_linreg_rocm.c
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

#include "ml_linear_regression_internal.h"
#include "neurondb_rocm_linreg.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Forward declaration for kernel launch wrappers */
extern hipError_t launch_build_X_matrix_kernel(const float *features,
											   float *X_with_intercept,
											   int n_samples,
											   int feature_dim,
											   int dim_with_intercept);
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

int
ndb_rocm_linreg_pack_model(const LinRegModel * model,
						   bytea * *model_data,
						   Jsonb * *metrics,
						   char **errstr)
{
	size_t		payload_bytes;
	bytea	   *blob;
	char	   *base;
	NdbCudaLinRegModelHeader *hdr;
	float	   *coef_dest;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid LinReg model for HIP pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaLinRegModelHeader)
		+ sizeof(float) * (size_t) model->n_features;

	blob = (bytea *) palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaLinRegModelHeader *) base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->intercept = (float) model->intercept;
	hdr->r_squared = model->r_squared;
	hdr->mse = model->mse;
	hdr->mae = model->mae;

	coef_dest = (float *) (base + sizeof(NdbCudaLinRegModelHeader));
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
						 "{\"algorithm\":\"linear_regression\","
						 "\"training_backend\":1,"
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

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
														  jsonb_in, CStringGetTextDatum(buf.data)));
		NDB_FREE(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_rocm_linreg_train(const float *features,
					  const double *targets,
					  int n_samples,
					  int feature_dim,
					  const Jsonb * hyperparams,
					  bytea * *model_data,
					  Jsonb * *metrics,
					  char **errstr)
{
	float	   *d_features = NULL;
	double	   *d_targets = NULL;
	double	   *d_XtX = NULL;
	double	   *d_Xty = NULL;
	double	   *d_XtX_inv __attribute__((unused)) = NULL;
	double	   *d_beta __attribute__((unused)) = NULL;
	double	   *h_XtX = NULL;
	double	   *h_Xty = NULL;
	double	   *h_XtX_inv = NULL;
	double	   *h_beta = NULL;
	bytea	   *payload = NULL;
	Jsonb	   *metrics_json = NULL;
	hipError_t	status __attribute__((unused)) = hipSuccess;
	size_t		feature_bytes __attribute__((unused));
	size_t		target_bytes __attribute__((unused));
	size_t		XtX_bytes;
	size_t		Xty_bytes;
	size_t		beta_bytes;
	int			dim_with_intercept;
	int			i,
				j,
				k;
	int			rc = -1;

	if (errstr)
		*errstr = NULL;

	if (features == NULL || targets == NULL || n_samples <= 0
		|| feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for HIP "
							  "LinReg train");
		return -1;
	}

	dim_with_intercept = feature_dim + 1;

	/* Allocate host memory for matrices */
	XtX_bytes = sizeof(double) * (size_t) dim_with_intercept
		* (size_t) dim_with_intercept;
	Xty_bytes = sizeof(double) * (size_t) dim_with_intercept;
	beta_bytes = sizeof(double) * (size_t) dim_with_intercept;

	h_XtX = (double *) palloc0(XtX_bytes);
	h_Xty = (double *) palloc0(Xty_bytes);
	h_XtX_inv = (double *) palloc(XtX_bytes);
	h_beta = (double *) palloc(beta_bytes);

	/* Compute X'X and X'y on GPU */
	{
		hipError_t	err;

		/* Allocate device memory */
		feature_bytes = sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
		target_bytes = sizeof(double) * (size_t) n_samples;
		XtX_bytes = sizeof(double) * (size_t) dim_with_intercept * (size_t) dim_with_intercept;
		Xty_bytes = sizeof(double) * (size_t) dim_with_intercept;

		err = hipMalloc((void **) &d_features, feature_bytes);
		if (err != hipSuccess)
		{
			if (errstr)
				*errstr = pstrdup("HIP LinReg train: failed to allocate device memory for features");
			goto cpu_fallback;
		}

		err = hipMalloc((void **) &d_targets, target_bytes);
		if (err != hipSuccess)
		{
			hipFree(d_features);
			if (errstr)
				*errstr = pstrdup("HIP LinReg train: failed to allocate device memory for targets");
			goto cpu_fallback;
		}

		err = hipMalloc((void **) &d_XtX, XtX_bytes);
		if (err != hipSuccess)
		{
			hipFree(d_features);
			hipFree(d_targets);
			if (errstr)
				*errstr = pstrdup("HIP LinReg train: failed to allocate device memory for XtX");
			goto cpu_fallback;
		}

		err = hipMalloc((void **) &d_Xty, Xty_bytes);
		if (err != hipSuccess)
		{
			hipFree(d_features);
			hipFree(d_targets);
			hipFree(d_XtX);
			if (errstr)
				*errstr = pstrdup("HIP LinReg train: failed to allocate device memory for Xty");
			goto cpu_fallback;
		}

		/* Initialize XtX and Xty to zero */
		err = hipMemset(d_XtX, 0, XtX_bytes);
		if (err != hipSuccess)
			goto gpu_cleanup;
		err = hipMemset(d_Xty, 0, Xty_bytes);
		if (err != hipSuccess)
			goto gpu_cleanup;

		/* Copy data to device */
		err = hipMemcpy(d_features, features, feature_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
			goto gpu_cleanup;
		err = hipMemcpy(d_targets, targets, target_bytes, hipMemcpyHostToDevice);
		if (err != hipSuccess)
			goto gpu_cleanup;

		/* Use rocBLAS for efficient matrix operations */
#ifdef NDB_GPU_HIP
		{
			rocblas_handle handle = ndb_rocm_get_cublas_handle();
			float	   *d_X_with_intercept = NULL;
			size_t		X_bytes = sizeof(float) * (size_t) n_samples * (size_t) dim_with_intercept;

			elog(DEBUG1,
				 "neurondb: linear_regression: attempting rocBLAS path (handle=%p, n_samples=%d, dim=%d)",
				 (void *) handle,
				 n_samples,
				 dim_with_intercept);

			/* Allocate and build X matrix with intercept column */
			err = hipMalloc((void **) &d_X_with_intercept, X_bytes);
			if (err != hipSuccess)
			{
				elog(WARNING,
					 "neurondb: linear_regression: failed to allocate X matrix, falling back to kernel");
				/* Fallback to kernel approach */
				goto kernel_fallback;
			}

			/* Build X matrix with intercept using GPU kernel (much faster) */
			err = launch_build_X_matrix_kernel(d_features,
											   d_X_with_intercept,
											   n_samples,
											   feature_dim,
											   dim_with_intercept);
			if (err != hipSuccess)
			{
				elog(WARNING,
					 "neurondb: linear_regression: build_X_matrix kernel failed, falling back");
				goto kernel_fallback;
			}

			if (handle == NULL)
				goto kernel_fallback;

			/*
			 * Use optimized custom kernel for X'X and X'y - it's correct and
			 * fast
			 */
			elog(DEBUG1,
				 "neurondb: linear_regression: using optimized custom kernel for X'X/X'y");
			goto kernel_fallback;

	kernel_fallback:
			if (d_X_with_intercept)
				hipFree(d_X_with_intercept);

			/* Fallback to kernel approach */
			{
				err = launch_linreg_compute_xtx_kernel(d_features,
													   d_targets,
													   n_samples,
													   feature_dim,
													   dim_with_intercept,
													   d_XtX,
													   d_Xty);
				if (err != hipSuccess)
					goto gpu_cleanup;
			}
		}
#else

		/*
		 * Kernel launch not available when compiling with gcc, fall back to
		 * CPU
		 */
		if (d_features)
			hipFree(d_features);
		if (d_targets)
			hipFree(d_targets);
		if (d_XtX)
			hipFree(d_XtX);
		if (d_Xty)
			hipFree(d_Xty);
		goto cpu_fallback;
#endif

#ifdef NDB_GPU_HIP
		/* Wait for operations to complete */
		err = cudaDeviceSynchronize();
		if (err != hipSuccess)
			goto gpu_cleanup;

		/* Copy results back to host (only if not using rocBLAS path) */
		if (d_XtX && d_Xty)
		{
			err = hipMemcpy(h_XtX, d_XtX, XtX_bytes, hipMemcpyDeviceToHost);
			if (err != hipSuccess)
				goto gpu_cleanup;
			err = hipMemcpy(h_Xty, d_Xty, Xty_bytes, hipMemcpyDeviceToHost);
			if (err != hipSuccess)
				goto gpu_cleanup;
		}

		hipFree(d_features);
		hipFree(d_targets);
		hipFree(d_XtX);
		hipFree(d_Xty);
#endif

gpu_cleanup:
		if (d_features)
			hipFree(d_features);
		if (d_targets)
			hipFree(d_targets);
		if (d_XtX)
			hipFree(d_XtX);
		if (d_Xty)
			hipFree(d_Xty);

cpu_fallback:
		/* Fallback to CPU computation */
		for (i = 0; i < n_samples; i++)
		{
			const float *row = features + (i * feature_dim);
			double	   *xi = (double *) palloc(sizeof(double) * dim_with_intercept);

			xi[0] = 1.0;		/* intercept */
			for (k = 1; k < dim_with_intercept; k++)
				xi[k] = row[k - 1];

			/* X'X accumulation */
			for (j = 0; j < dim_with_intercept; j++)
			{
				for (k = 0; k < dim_with_intercept; k++)
					h_XtX[j * dim_with_intercept + k] += xi[j] * xi[k];

				/* X'y accumulation */
				h_Xty[j] += xi[j] * targets[i];
			}

			NDB_FREE(xi);
		}
	}

	/* Solve normal equations using Cholesky decomposition with regularization */

	/*
	 * X'X should be positive definite, but we add regularization for
	 * numerical stability
	 */
	{
		double		lambda = 1e-3;	/* Base regularization parameter */
		double	   *L = NULL;	/* Lower triangular Cholesky factor */
		double	   *y_work = NULL;	/* Working vector for forward substitution */
		double	   *XtX_work = NULL;	/* Working copy of X'X for Cholesky
										 * (preserve original) */
		int			row,
					col,
					k_local;
		bool		cholesky_success = true;
		double		diag_sum;
		double		max_diag = 0.0;
		double		max_off_diag = 0.0;
		double		trace = 0.0;

		/* Ensure matrix is symmetric (X'X should be symmetric) */

		/*
		 * Also validate that diagonal elements are positive (required for
		 * positive definiteness)
		 */
		for (row = 0; row < dim_with_intercept; row++)
		{
			double		diag_val = h_XtX[row * dim_with_intercept + row];

			/* Defensive check: ensure diagonal is positive */
			if (diag_val <= 0.0)
			{
				elog(WARNING,
					 "neurondb: GPU LinReg: X'X[%d][%d] = %.6e is not positive, forcing to small positive value",
					 row, row, diag_val);
				h_XtX[row * dim_with_intercept + row] = 1e-6;
				diag_val = 1e-6;
			}

			for (col = row + 1; col < dim_with_intercept; col++)
			{
				double		val1 = h_XtX[row * dim_with_intercept + col];
				double		val2 = h_XtX[col * dim_with_intercept + row];
				double		avg = (val1 + val2) / 2.0;

				h_XtX[row * dim_with_intercept + col] = avg;
				h_XtX[col * dim_with_intercept + row] = avg;
				if (fabs(avg) > max_off_diag)
					max_off_diag = fabs(avg);
			}
		}

		/* Find maximum diagonal element for adaptive regularization */
		/* After symmetry enforcement, all diagonals should be positive */
		for (row = 0; row < dim_with_intercept; row++)
		{
			double		diag_val = h_XtX[row * dim_with_intercept + row];

			/* Defensive: ensure positive */
			if (diag_val <= 0.0)
				diag_val = 1e-6;	/* Force to small positive value */

			if (diag_val > max_diag)
				max_diag = diag_val;
			trace += diag_val;	/* Sum of diagonal values */
		}

		/* Adaptive regularization: ensure matrix is well-conditioned */
		/* Use conservative regularization based on matrix scale */
		if (max_diag > 0.0 && trace > 0.0)
		{
			double		avg_diag = trace / (double) dim_with_intercept;

			/*
			 * Use the smaller of max_diag or avg_diag to avoid
			 * over-regularization
			 */
			double		scale = (max_diag < avg_diag) ? max_diag : avg_diag;

			/* Use 0.01% of scale, but cap at reasonable values */
			lambda = scale * 1e-4;
			/* Ensure lambda is within reasonable bounds */
			if (lambda < 1e-6)
				lambda = 1e-6;	/* Minimum regularization */
			if (lambda > 1.0)
				lambda = 1.0;	/* Maximum regularization (safety cap) */
		}
		else
		{
			/* Fallback: use fixed regularization */
			lambda = 1e-3;
		}

		/* Create working copy of X'X for Cholesky (don't modify original) */
		XtX_work = (double *) palloc(sizeof(double) * (size_t) dim_with_intercept * (size_t) dim_with_intercept);
		if (XtX_work == NULL)
		{
			if (errstr)
				*errstr = pstrdup("neurondb: failed to allocate memory for Cholesky decomposition");
			NDB_FREE(h_XtX);
			NDB_FREE(h_Xty);
			return -1;
		}

		/* Copy X'X to working matrix and add regularization to diagonal */
		for (row = 0; row < dim_with_intercept; row++)
		{
			for (col = 0; col < dim_with_intercept; col++)
			{
				XtX_work[row * dim_with_intercept + col] = h_XtX[row * dim_with_intercept + col];
			}
			/* Add regularization to diagonal: X'X + lambda*I */
			XtX_work[row * dim_with_intercept + row] += lambda;
		}

		/* Allocate Cholesky factor L (lower triangular, stored row-major) */
		L = (double *) palloc0(sizeof(double) * dim_with_intercept * dim_with_intercept);

		/* Cholesky decomposition: X'X_work = L * L^T */
		/* Use working copy (XtX_work) so we don't modify original matrix */
		/* X'X is symmetric, so we only need lower triangular part */
		for (row = 0; row < dim_with_intercept; row++)
		{
			diag_sum = XtX_work[row * dim_with_intercept + row];
			for (k_local = 0; k_local < row; k_local++)
			{
				diag_sum -= L[row * dim_with_intercept + k_local]
					* L[row * dim_with_intercept + k_local];
			}

			/* Check for positive definiteness */
			/* With proper regularization, diag_sum should be positive */
			if (diag_sum <= 1e-12)
			{
				/*
				 * If still too small after regularization, add more
				 * dynamically
				 */
				double		min_diag;
				double		extra_lambda;

				min_diag = lambda * 0.1;	/* 10% of regularization */
				if (min_diag < 1e-8)
					min_diag = 1e-8;

				extra_lambda = min_diag - diag_sum;
				if (extra_lambda > 0.0)
				{
					XtX_work[row * dim_with_intercept + row] += extra_lambda;
					diag_sum += extra_lambda;
				}

				/* Final check - if still too small, matrix is truly singular */
				if (diag_sum <= 1e-12)
				{
					cholesky_success = false;
					if (errstr)
						*errstr = psprintf("Matrix is not positive definite at row %d (diag_sum=%.6e, lambda=%.6e)",
										   row, diag_sum, lambda);
					break;
				}
			}

			L[row * dim_with_intercept + row] = sqrt(diag_sum);

			for (col = row + 1; col < dim_with_intercept; col++)
			{
				/* Use symmetric property: X'X[col][row] = X'X[row][col] */
				double		off_diag_sum = XtX_work[row * dim_with_intercept + col];

				for (k_local = 0; k_local < row; k_local++)
				{
					off_diag_sum -= L[col * dim_with_intercept + k_local]
						* L[row * dim_with_intercept + k_local];
				}
				L[col * dim_with_intercept + row] = off_diag_sum
					/ L[row * dim_with_intercept + row];
			}
		}

		if (!cholesky_success)
		{
			if (L)
				NDB_FREE(L);
			if (XtX_work)
				NDB_FREE(XtX_work);
			NDB_FREE(h_XtX);
			NDB_FREE(h_Xty);
			NDB_FREE(h_XtX_inv);
			NDB_FREE(h_beta);
			return -1;
		}

		/* Forward substitution: L * y = X'y, solve for y */
		y_work = (double *) palloc(sizeof(double) * dim_with_intercept);
		for (row = 0; row < dim_with_intercept; row++)
		{
			double		sum = h_Xty[row];

			for (k_local = 0; k_local < row; k_local++)
			{
				sum -= L[row * dim_with_intercept + k_local] * y_work[k_local];
			}
			y_work[row] = sum / L[row * dim_with_intercept + row];
		}

		/* Backward substitution: L^T * beta = y, solve for beta */
		for (row = dim_with_intercept - 1; row >= 0; row--)
		{
			double		sum = y_work[row];

			for (k_local = row + 1; k_local < dim_with_intercept; k_local++)
			{
				sum -= L[k_local * dim_with_intercept + row] * h_beta[k_local];
			}
			h_beta[row] = sum / L[row * dim_with_intercept + row];
		}

		/* Compute (X'X)^(-1) for metrics calculation using Cholesky */
		/* Solve L * L^T * X = I column by column */
		for (col = 0; col < dim_with_intercept; col++)
		{
			double	   *col_vec = (double *) palloc0(sizeof(double) * dim_with_intercept);

			col_vec[col] = 1.0;

			/* Forward substitution: L * y = e_col */
			for (row = 0; row < dim_with_intercept; row++)
			{
				double		sum = col_vec[row];

				for (k_local = 0; k_local < row; k_local++)
				{
					sum -= L[row * dim_with_intercept + k_local] * y_work[k_local];
				}
				y_work[row] = sum / L[row * dim_with_intercept + row];
			}

			/* Backward substitution: L^T * x = y */
			for (row = dim_with_intercept - 1; row >= 0; row--)
			{
				double		sum = y_work[row];

				for (k_local = row + 1; k_local < dim_with_intercept; k_local++)
				{
					sum -= L[k_local * dim_with_intercept + row]
						* h_XtX_inv[k_local * dim_with_intercept + col];
				}
				h_XtX_inv[row * dim_with_intercept + col] =
					sum / L[row * dim_with_intercept + row];
			}

			NDB_FREE(col_vec);
		}

		NDB_FREE(L);
		NDB_FREE(y_work);
		if (XtX_work)
			NDB_FREE(XtX_work);
		/* Note: h_beta is already computed above via Cholesky solve */
	}

	/* Build model */
	{
		LinRegModel model;
		double		y_mean = 0.0;
		double		ss_tot = 0.0;
		double		ss_res = 0.0;
		double		mse = 0.0;
		double		mae = 0.0;

		model.n_features = feature_dim;
		model.n_samples = n_samples;
		model.intercept = h_beta[0];
		model.coefficients =
			(double *) palloc(sizeof(double) * feature_dim);
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
		if (ss_tot > 1e-10)
			model.r_squared = 1.0 - (ss_res / ss_tot);
		else
			model.r_squared = 0.0;	/* All targets are the same */
		model.mse = mse;
		model.mae = mae;

		/* Pack model */
		rc = ndb_rocm_linreg_pack_model(
										&model, &payload, &metrics_json, errstr);

		NDB_FREE(model.coefficients);
	}

	/* Cleanup */
	NDB_FREE(h_XtX);
	NDB_FREE(h_Xty);
	NDB_FREE(h_XtX_inv);
	NDB_FREE(h_beta);

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
ndb_rocm_linreg_predict(const bytea * model_data,
						const float *input,
						int feature_dim,
						double *prediction_out,
						char **errstr)
{
	const		NdbCudaLinRegModelHeader *hdr;
	const float *coefficients;
	const		bytea *detoasted;
	double		prediction;
	int			i;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || prediction_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup(
							  "invalid parameters for HIP LinReg predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted =
		(const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t		expected_size = sizeof(NdbCudaLinRegModelHeader)
			+ sizeof(float) * (size_t) feature_dim;
		size_t		actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr =
					psprintf("model data too small: "
							 "expected %zu bytes, got %zu",
							 expected_size,
							 actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaLinRegModelHeader *) VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model "
							   "has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		return -1;
	}

	coefficients = (const float *) ((const char *) hdr
									+ sizeof(NdbCudaLinRegModelHeader));

	prediction = hdr->intercept;
	for (i = 0; i < feature_dim; i++)
		prediction += coefficients[i] * input[i];

	*prediction_out = prediction;
	return 0;
}

/*
 * ndb_rocm_linreg_evaluate
 *    GPU-accelerated batch evaluation for Linear Regression
 *
 * Computes predictions for all samples and accumulates SSE, SAE, and count
 * using a HIP kernel, then computes final metrics (MSE, RMSE, MAE, R-squared)
 * on the host.
 *
 * Parameters:
 *   model_data: GPU model bytea (contains header + coefficients)
 *   features: Feature matrix [n_samples, feature_dim] (float, row-major)
 *   targets: Target values [n_samples] (double)
 *   n_samples: Number of samples to evaluate
 *   feature_dim: Number of features per sample
 *   mse_out: Output MSE (mean squared error)
 *   mae_out: Output MAE (mean absolute error)
 *   rmse_out: Output RMSE (root mean squared error)
 *   r_squared_out: Output R-squared
 *   errstr: Error message output (if non-NULL)
 *
 * Returns:
 *   0 on success, -1 on error
 */
int
ndb_rocm_linreg_evaluate(const bytea * model_data,
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
	const		NdbCudaLinRegModelHeader *hdr;
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
			*errstr = pstrdup("neurondb: ndb_rocm_linreg_evaluate: model_data is NULL");
		return -1;
	}

	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_rocm_linreg_evaluate: features is NULL");
		return -1;
	}

	if (targets == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_rocm_linreg_evaluate: targets is NULL");
		return -1;
	}

	if (n_samples <= 0)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: invalid n_samples %d",
							   n_samples);
		return -1;
	}

	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: invalid feature_dim %d",
							   feature_dim);
		return -1;
	}

	if (mse_out == NULL || mae_out == NULL || rmse_out == NULL || r_squared_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_rocm_linreg_evaluate: output pointers are NULL");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t		expected_size = sizeof(NdbCudaLinRegModelHeader)
			+ sizeof(float) * (size_t) feature_dim;
		size_t		actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: model data too small: "
								   "expected %zu bytes, got %zu",
								   expected_size,
								   actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaLinRegModelHeader *) VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: feature dimension mismatch: "
							   "model has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		return -1;
	}

	coefficients = (const float *) ((const char *) hdr + sizeof(NdbCudaLinRegModelHeader));

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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to allocate GPU memory for features: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to allocate GPU memory for targets: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to allocate GPU memory for coefficients: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to allocate GPU memory for SSE: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to allocate GPU memory for SAE: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to allocate GPU memory for count: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to copy features to GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to copy targets to GPU: %s",
							   hipGetErrorString(cuda_err));
		return -1;
	}

	/* Convert coefficients from float to double and copy to GPU */
	{
		double	   *h_coefficients_double = (double *) palloc(sizeof(double) * (size_t) feature_dim);

		if (h_coefficients_double == NULL)
		{
			hipFree(d_features);
			hipFree(d_targets);
			hipFree(d_coefficients);
			hipFree(d_sse);
			hipFree(d_sae);
			hipFree(d_count);
			if (errstr)
				*errstr = pstrdup("neurondb: ndb_rocm_linreg_evaluate: failed to allocate host memory for coefficients");
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
				*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to copy coefficients to GPU: %s",
								   hipGetErrorString(cuda_err));
			return -1;
		}
	}

	/* Launch evaluation kernel */
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: evaluation kernel failed: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to copy SSE from GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to copy SAE from GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: failed to copy count from GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_linreg_evaluate: count mismatch: expected %d, got %lld",
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
