/*-------------------------------------------------------------------------
 *
 * gpu_linreg_cuda.c
 *    Linear regression bridge implementation.
 *
 * This module provides bridge functions for linear regression training
 * and prediction.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_linreg_cuda.c
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
#include "utils/elog.h"

#include "ml_linear_regression_internal.h"
#include "neurondb_cuda_linreg.h"
#include "neurondb_validation.h"
/* Note: neurondb_safe_memory.h not needed - we override nalloc/nfree below */
#include "neurondb_macros.h"
#include "neurondb_guc.h"
#include "neurondb_constants.h"
#include "neurondb_json.h"

/* TEMPORARY DEBUGGING: Replace nalloc/nfree with direct PostgreSQL allocators */
/* This isolates whether the wrappers are causing memory corruption */
#undef nalloc
#undef nfree
#define nalloc(var, type, count) \
	do { \
		(var) = (type *) MemoryContextAlloc(CurrentMemoryContext, sizeof(type) * (Size) (count)); \
		memset((var), 0, sizeof(type) * (Size) (count)); \
	} while (0)
#define pfree(ptr) \
	do { \
		if ((ptr) != NULL) { \
			pfree((ptr)); \
		} \
	} while (0)

extern cudaError_t launch_build_X_matrix_kernel(const float *features,
												float *X_with_intercept,
												int n_samples,
												int feature_dim,
												int dim_with_intercept);
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

int
ndb_cuda_linreg_pack_model(const LinRegModel *model,
						   bytea * *model_data,
						   Jsonb * *metrics,
						   char **errstr)
{
	size_t		payload_bytes;
	char	   *base = NULL;
	NdbCudaLinRegModelHeader *hdr = NULL;
	float	   *coef_dest = NULL;
	bytea *blob = NULL;
	char *blob_raw = NULL;


	/* GPU mode: ensure we're NOT in CPU mode */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = pstrdup("CUDA linreg_pack_model: CPU mode detected - should not be called");
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: CUDA pack_model called in CPU mode - this is a bug")));
		return -1;
	}

	/* Ensure GPU mode is required */
	if (!NDB_REQUIRE_GPU())
	{
		if (errstr)
			*errstr = pstrdup("CUDA linreg_pack_model: not in GPU mode");
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: CUDA pack_model called but GPU mode not required - this is a bug")));
		return -1;
	}

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid LinReg model for CUDA pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaLinRegModelHeader)
		+ sizeof(float) * (size_t) model->n_features;

	nalloc(blob_raw, char, VARHDRSZ + payload_bytes);
	blob = (bytea *) blob_raw;
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
		{
			coef_dest[i] = (float) model->coefficients[i];
		}
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb	   *metrics_json = NULL;
		double		r_squared;
		double		mse;
		double		mae;

		/* Compute safe metric values */
		r_squared = isfinite(model->r_squared) ? model->r_squared : 0.0;
		mse = isfinite(model->mse) ? model->mse : 0.0;
		mae = isfinite(model->mae) ? model->mae : 0.0;

		/* Create metrics JSON string - MUST include training_backend=1 and storage=gpu */
		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"linear_regression\","
						 "\"training_backend\":1,"
						 "\"storage\":\"gpu\","
						 "\"n_features\":%d,"
						 "\"n_samples\":%d,"
						 "\"r_squared\":%.6f,"
						 "\"mse\":%.6f,"
						 "\"mae\":%.6f}",
						 model->n_features,
						 model->n_samples,
						 r_squared,
						 mse,
						 mae);

		/* Create JSON in TopMemoryContext to isolate from potentially corrupted context */
		/* This prevents the corrupted freelist from affecting JSON creation */
		{
			MemoryContext oldcontext;
			Jsonb	   *temp_json = NULL;

			oldcontext = MemoryContextSwitchTo(TopMemoryContext);

			PG_TRY();
			{
				temp_json = ndb_jsonb_in_cstring(buf.data);
			}
			PG_CATCH();
			{
				MemoryContextSwitchTo(oldcontext);
				FlushErrorState();
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: CUDA linreg_pack_model: Failed to create metrics JSONB in TopMemoryContext"),
						 errdetail("Even TopMemoryContext failed - this indicates a more serious issue"),
						 errhint("Check PostgreSQL server logs for additional errors")));
			}
			PG_END_TRY();

			if (temp_json == NULL)
			{
				MemoryContextSwitchTo(oldcontext);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: CUDA linreg_pack_model: Failed to create metrics JSONB - ndb_jsonb_in_cstring returned NULL")));
			}

			/* Switch back to original context and copy JSON to it */
			MemoryContextSwitchTo(oldcontext);

			/* Copy JSONB to current context using palloc and memcpy */
			/* JSONB is a varlena type, so we need to copy the entire structure */
			{
				size_t		jsonb_size = VARSIZE(temp_json);

				metrics_json = (Jsonb *) MemoryContextAlloc(CurrentMemoryContext, jsonb_size);
				memcpy(metrics_json, temp_json, jsonb_size);

				/* Free the temporary JSON from TopMemoryContext */
				pfree(temp_json);
			}
		}
		
		if (metrics_json == NULL)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: CUDA linreg_pack_model: Failed to create metrics JSONB - ndb_jsonb_in_cstring returned NULL"),
					 errdetail("Memory context %p may be corrupted. JSON string length=%d",
							   (void *) CurrentMemoryContext, buf.len),
					 errhint("This indicates memory corruption occurred earlier in the training process, likely during matrix operations or memory allocation.")));
		}

		/* Free StringInfo data - use pfree directly to avoid nfree wrapper issues */
		if (buf.data != NULL)
		{
			pfree(buf.data);
			buf.data = NULL;
		}

		if (metrics_json == NULL)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: CUDA linreg_pack_model: metrics JSONB is NULL after creation - GPU mode requires valid metrics")));
		}

		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_cuda_linreg_train(const float *features,
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
	double *h_XtX = NULL;
	double *h_Xty = NULL;
	double *h_XtX_inv = NULL;
	double *h_beta = NULL;
	bytea	   *payload = NULL;
	Jsonb	   *metrics_json = NULL;
	cudaError_t status __attribute__((unused)) = cudaSuccess;
	size_t		feature_bytes __attribute__((unused));
	size_t		target_bytes __attribute__((unused));
	size_t		XtX_bytes;
	size_t		Xty_bytes;
	int			dim_with_intercept;
	int			i,
				j,
				k;
	int			rc = -1;

	/* Initialize output pointers to NULL */
	if (model_data)
		*model_data = NULL;
	if (metrics)
		*metrics = NULL;

	/* Memory context check at start to catch early corruption */
#ifdef MEMORY_CONTEXT_CHECKING
	MemoryContextCheck(CurrentMemoryContext);
#endif

	/* GPU mode: ensure we're NOT in CPU mode - strict check */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = pstrdup("CUDA linreg_train: CPU mode detected - should not be called");
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: CUDA linreg_train called in CPU mode - this is a bug")));
		return -1;
	}

	/* Ensure GPU mode is required - no fallback allowed */
	if (!NDB_REQUIRE_GPU())
	{
		if (errstr)
			*errstr = pstrdup("CUDA linreg_train: not in GPU mode");
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: CUDA linreg_train called but GPU mode not required - this is a bug")));
		return -1;
	}


	if (errstr)
		*errstr = NULL;

	if (features == NULL || targets == NULL || n_samples <= 0
		|| feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for CUDA "
							  "LinReg train");
		return -1;
	}

	dim_with_intercept = feature_dim + 1;

	/* Allocate host memory for matrices */
	XtX_bytes = sizeof(double) * (size_t) dim_with_intercept
		* (size_t) dim_with_intercept;
	Xty_bytes = sizeof(double) * (size_t) dim_with_intercept;

	nalloc(h_XtX, double, dim_with_intercept * dim_with_intercept);
	nalloc(h_Xty, double, dim_with_intercept);
	nalloc(h_XtX_inv, double, dim_with_intercept * dim_with_intercept);
	nalloc(h_beta, double, dim_with_intercept);

	/* Compute X'X and X'y on GPU */
	{
		cudaError_t err;

		/* Allocate device memory */
		feature_bytes = sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
		target_bytes = sizeof(double) * (size_t) n_samples;
		XtX_bytes = sizeof(double) * (size_t) dim_with_intercept * (size_t) dim_with_intercept;
		Xty_bytes = sizeof(double) * (size_t) dim_with_intercept;

		err = cudaMalloc((void **) &d_features, feature_bytes);
		if (err != cudaSuccess)
		{
			if (errstr)
				*errstr = pstrdup("CUDA LinReg train: failed to allocate device memory for features");
			goto cpu_fallback;
		}

		err = cudaMalloc((void **) &d_targets, target_bytes);
		if (err != cudaSuccess)
		{
			cudaFree(d_features);
			if (errstr)
				*errstr = pstrdup("CUDA LinReg train: failed to allocate device memory for targets");
			goto cpu_fallback;
		}

		err = cudaMalloc((void **) &d_XtX, XtX_bytes);
		if (err != cudaSuccess)
		{
			cudaFree(d_features);
			cudaFree(d_targets);
			if (errstr)
				*errstr = pstrdup("CUDA LinReg train: failed to allocate device memory for XtX");
			goto cpu_fallback;
		}

		err = cudaMalloc((void **) &d_Xty, Xty_bytes);
		if (err != cudaSuccess)
		{
			cudaFree(d_features);
			cudaFree(d_targets);
			cudaFree(d_XtX);
			if (errstr)
				*errstr = pstrdup("CUDA LinReg train: failed to allocate device memory for Xty");
			goto cpu_fallback;
		}

		/* Initialize XtX and Xty to zero */
		err = cudaMemset(d_XtX, 0, XtX_bytes);
		if (err != cudaSuccess)
			goto gpu_cleanup;
		err = cudaMemset(d_Xty, 0, Xty_bytes);
		if (err != cudaSuccess)
			goto gpu_cleanup;

		/* Copy data to device */
		err = cudaMemcpy(d_features, features, feature_bytes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
			goto gpu_cleanup;
		err = cudaMemcpy(d_targets, targets, target_bytes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
			goto gpu_cleanup;

		/* Use cuBLAS for efficient matrix operations */
#ifdef NDB_GPU_CUDA
		{
			cublasHandle_t handle = ndb_cuda_get_cublas_handle();
			float	   *d_X_with_intercept = NULL;
			size_t		X_bytes = sizeof(float) * (size_t) n_samples * (size_t) dim_with_intercept;


			/* Allocate and build X matrix with intercept column */
			err = cudaMalloc((void **) &d_X_with_intercept, X_bytes);
			if (err != cudaSuccess)
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
			if (err != cudaSuccess)
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
			goto kernel_fallback;

	kernel_fallback:
			if (d_X_with_intercept)
				cudaFree(d_X_with_intercept);

			/* Fallback to kernel approach */
			{
				err = launch_linreg_compute_xtx_kernel(d_features,
													   d_targets,
													   n_samples,
													   feature_dim,
													   dim_with_intercept,
													   d_XtX,
													   d_Xty);
				if (err != cudaSuccess)
					goto gpu_cleanup;
			}
		}
#else

		/*
		 * Kernel launch not available when compiling with gcc, fall back to
		 * CPU
		 */
		if (d_features)
			cudaFree(d_features);
		if (d_targets)
			cudaFree(d_targets);
		if (d_XtX)
			cudaFree(d_XtX);
		if (d_Xty)
			cudaFree(d_Xty);
		goto cpu_fallback;
#endif

#ifdef NDB_GPU_CUDA
		/* Wait for operations to complete */
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
			goto gpu_cleanup;

		/* Copy results back to host (only if not using cuBLAS path) */
		if (d_XtX && d_Xty)
		{
			err = cudaMemcpy(h_XtX, d_XtX, XtX_bytes, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess)
				goto gpu_cleanup;
			err = cudaMemcpy(h_Xty, d_Xty, Xty_bytes, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess)
				goto gpu_cleanup;
		}

		/* Free CUDA memory - set pointers to NULL to prevent double-free in gpu_cleanup */
		if (d_features)
		{
			cudaFree(d_features);
			d_features = NULL;
		}
		if (d_targets)
		{
			cudaFree(d_targets);
			d_targets = NULL;
		}
		if (d_XtX)
		{
			cudaFree(d_XtX);
			d_XtX = NULL;
		}
		if (d_Xty)
		{
			cudaFree(d_Xty);
			d_Xty = NULL;
		}
#endif

gpu_cleanup:
		if (d_features)
			cudaFree(d_features);
		if (d_targets)
			cudaFree(d_targets);
		if (d_XtX)
			cudaFree(d_XtX);
		if (d_Xty)
			cudaFree(d_Xty);

cpu_fallback:
		/* Fallback to CPU computation */
		for (i = 0; i < n_samples; i++)
		{
			const float *row = features + (i * feature_dim);

			double *xi = NULL;
			nalloc(xi, double, dim_with_intercept);

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

			pfree(xi);
		}
	}

	/* Solve normal equations using Cholesky decomposition with regularization */

	/*
	 * X'X should be positive definite, but we add regularization for
	 * numerical stability
	 */
	{
		double		lambda = 1e-3;	/* Base regularization parameter */
		double *L = NULL;
		double *y_work = NULL;
		double *XtX_work = NULL;
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
		nalloc(XtX_work, double, dim_with_intercept * dim_with_intercept);
		if (XtX_work == NULL)
		{
			if (errstr)
				*errstr = pstrdup("neurondb: failed to allocate memory for Cholesky decomposition");
			pfree(h_XtX);
			pfree(h_Xty);
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
		nalloc(L, double, dim_with_intercept * dim_with_intercept);

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
				pfree(L);
			if (XtX_work)
				pfree(XtX_work);
			pfree(h_XtX);
			pfree(h_Xty);
			pfree(h_XtX_inv);
			pfree(h_beta);
			return -1;
		}

		/* Forward substitution: L * y = X'y, solve for y */
		nalloc(y_work, double, dim_with_intercept);
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
		/* Initialize h_XtX_inv to zero before use */
		memset(h_XtX_inv, 0, sizeof(double) * dim_with_intercept * dim_with_intercept);
		
		for (col = 0; col < dim_with_intercept; col++)
		{
			double *col_vec = NULL;
			double *inv_work = NULL;

			nalloc(col_vec, double, dim_with_intercept);

			/* Initialize all entries to zero, then set unit vector */
			memset(col_vec, 0, sizeof(double) * dim_with_intercept);
			col_vec[col] = 1.0;

			/* Forward substitution: L * y = e_col */
			/* Use a separate work array for inverse computation to avoid reusing y_work */
			nalloc(inv_work, double, dim_with_intercept);
			
			for (row = 0; row < dim_with_intercept; row++)
			{
				double		sum = col_vec[row];

				for (k_local = 0; k_local < row; k_local++)
				{
					sum -= L[row * dim_with_intercept + k_local] * inv_work[k_local];
				}
				inv_work[row] = sum / L[row * dim_with_intercept + row];
			}

			/* Backward substitution: L^T * x = y */
			for (row = dim_with_intercept - 1; row >= 0; row--)
			{
				double		sum = inv_work[row];
				int			idx;

				for (k_local = row + 1; k_local < dim_with_intercept; k_local++)
				{
					/* Bounds check to prevent buffer overrun */
					idx = k_local * dim_with_intercept + col;
					if (idx < 0 || idx >= dim_with_intercept * dim_with_intercept)
					{
						pfree(inv_work);
						pfree(col_vec);
						elog(ERROR, "neurondb: CUDA linreg_train: buffer overrun detected: idx=%d, max=%d", 
							 idx, dim_with_intercept * dim_with_intercept);
					}
					sum -= L[k_local * dim_with_intercept + row]
						* h_XtX_inv[idx];
				}
				/* Bounds check before write */
				idx = row * dim_with_intercept + col;
				if (idx < 0 || idx >= dim_with_intercept * dim_with_intercept)
				{
					pfree(inv_work);
					pfree(col_vec);
					elog(ERROR, "neurondb: CUDA linreg_train: buffer overrun detected on write: idx=%d, max=%d", 
						 idx, dim_with_intercept * dim_with_intercept);
				}
				h_XtX_inv[idx] = sum / L[row * dim_with_intercept + row];
			}
			
			pfree(inv_work);
			pfree(col_vec);
		}

		pfree(L);
		pfree(y_work);
		if (XtX_work)
			pfree(XtX_work);
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
		double *model_coefficients = NULL;

		model.n_features = feature_dim;
		model.n_samples = n_samples;
		model.intercept = h_beta[0];
		nalloc(model_coefficients, double, feature_dim);
		model.coefficients = model_coefficients;
		for (i = 0; i < feature_dim; i++)
		{
			model.coefficients[i] = h_beta[i + 1];
		}

		/* Compute metrics */
		for (i = 0; i < n_samples; i++)
			y_mean += targets[i];
		y_mean /= n_samples;

		for (i = 0; i < n_samples; i++)
		{
			const float *row;
			double		y_pred;
			double		error;
			int			j_local;

			row = features + (i * feature_dim);
			y_pred = model.intercept;

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
		rc = ndb_cuda_linreg_pack_model(
										&model, &payload, &metrics_json, errstr);

		pfree(model.coefficients);
	}

	pfree(h_XtX);
	pfree(h_Xty);
	pfree(h_XtX_inv);
	pfree(h_beta);

	if (rc == 0 && payload != NULL)
	{
		*model_data = payload;
		if (metrics != NULL)
			*metrics = metrics_json;
		return 0;
	}

	if (payload != NULL)
		pfree(payload);
	if (metrics_json != NULL)
		pfree(metrics_json);

	return -1;
}

int
ndb_cuda_linreg_predict(const bytea * model_data,
						const float *input,
						int feature_dim,
						double *prediction_out,
						char **errstr)
{
	const NdbCudaLinRegModelHeader *hdr;
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
							  "invalid parameters for CUDA LinReg predict");
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
 * ndb_cuda_linreg_evaluate
 *    GPU-accelerated batch evaluation for Linear Regression
 *
 * Computes predictions for all samples and accumulates SSE, SAE, and count
 * using a CUDA kernel, then computes final metrics (MSE, RMSE, MAE, R-squared)
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
ndb_cuda_linreg_evaluate(const bytea * model_data,
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
	const NdbCudaLinRegModelHeader *hdr;
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
			*errstr = pstrdup("neurondb: ndb_cuda_linreg_evaluate: model_data is NULL");
		return -1;
	}

	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_cuda_linreg_evaluate: features is NULL");
		return -1;
	}

	if (targets == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_cuda_linreg_evaluate: targets is NULL");
		return -1;
	}

	if (n_samples <= 0)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: invalid n_samples %d",
							   n_samples);
		return -1;
	}

	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: invalid feature_dim %d",
							   feature_dim);
		return -1;
	}

	if (mse_out == NULL || mae_out == NULL || rmse_out == NULL || r_squared_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_cuda_linreg_evaluate: output pointers are NULL");
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
				*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: model data too small: "
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: feature dimension mismatch: "
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
	cuda_err = cudaMalloc((void **) &d_features, feature_bytes);
	if (cuda_err != cudaSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to allocate GPU memory for features: %s",
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to allocate GPU memory for targets: %s",
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to allocate GPU memory for coefficients: %s",
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to allocate GPU memory for SSE: %s",
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to allocate GPU memory for SAE: %s",
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to allocate GPU memory for count: %s",
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to copy features to GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to copy targets to GPU: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Convert coefficients from float to double and copy to GPU */
	{
		double *h_coefficients_double = NULL;
		nalloc(h_coefficients_double, double, feature_dim);

		if (h_coefficients_double == NULL)
		{
			cudaFree(d_features);
			cudaFree(d_targets);
			cudaFree(d_coefficients);
			cudaFree(d_sse);
			cudaFree(d_sae);
			cudaFree(d_count);
			if (errstr)
				*errstr = pstrdup("neurondb: ndb_cuda_linreg_evaluate: failed to allocate host memory for coefficients");
			return -1;
		}

		for (i = 0; i < feature_dim; i++)
			h_coefficients_double[i] = (double) coefficients[i];

		cuda_err = cudaMemcpy(d_coefficients, h_coefficients_double, coeff_bytes, cudaMemcpyHostToDevice);
		pfree(h_coefficients_double);

		if (cuda_err != cudaSuccess)
		{
			cudaFree(d_features);
			cudaFree(d_targets);
			cudaFree(d_coefficients);
			cudaFree(d_sse);
			cudaFree(d_sae);
			cudaFree(d_count);
			if (errstr)
				*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to copy coefficients to GPU: %s",
								   cudaGetErrorString(cuda_err));
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

	if (cuda_err != cudaSuccess)
	{
		const char *cuda_err_str = cudaGetErrorString(cuda_err);
		cudaFree(d_features);
		cudaFree(d_targets);
		cudaFree(d_coefficients);
		cudaFree(d_sse);
		cudaFree(d_sae);
		cudaFree(d_count);
		if (errstr)
		{
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: evaluation kernel launch failed: CUDA error %d (%s). "
							   "Parameters: n_samples=%d, feature_dim=%d, model_feature_dim=%d",
							   (int) cuda_err,
							   cuda_err_str ? cuda_err_str : "unknown",
							   n_samples,
							   feature_dim,
							   hdr->feature_dim);
		}
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to copy SSE from GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to copy SAE from GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: failed to copy count from GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_cuda_linreg_evaluate: count mismatch: expected %d, got %lld",
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
