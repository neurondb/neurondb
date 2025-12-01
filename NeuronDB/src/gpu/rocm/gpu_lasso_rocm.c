/*-------------------------------------------------------------------------
 *
 * gpu_lasso_rocm.c
 *    ROCm backend bridge for Lasso Regression training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_lasso_rocm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_HIP

#include <float.h>
#include <math.h>
#include <string.h>

#include "neurondb_rocm_runtime.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"

#include "ml_lasso_regression_internal.h"
#include "neurondb_rocm_lasso.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Reuse linear regression evaluation kernel */
extern hipError_t launch_linreg_eval_kernel(const float *features,
											const double *targets,
											const double *coefficients,
											double intercept,
											int n_samples,
											int feature_dim,
											double *sse_out,
											double *sae_out,
											long long *count_out);

/* Lasso coordinate descent kernels */
extern hipError_t launch_lasso_compute_rho_kernel(const float *features,
												  const double *residuals,
												  int n_samples,
												  int feature_dim,
												  int feature_idx,
												  double *rho_out);
extern hipError_t launch_lasso_compute_z_kernel(const float *features,
												int n_samples,
												int feature_dim,
												int feature_idx,
												double *z_out);
extern hipError_t launch_lasso_update_residuals_kernel(const float *features,
													   double *residuals,
													   int n_samples,
													   int feature_dim,
													   int feature_idx,
													   double weight_diff);

/*
 * Soft thresholding operator for Lasso
 */
static double
soft_threshold(double x, double lambda)
{
	if (x > lambda)
		return x - lambda;
	else if (x < -lambda)
		return x + lambda;
	return 0.0;
}

int
ndb_rocm_lasso_pack_model(const LassoModel * model,
						  bytea * *model_data,
						  Jsonb * *metrics,
						  char **errstr)
{
	size_t		payload_bytes;
	bytea	   *blob;
	char	   *base;
	NdbCudaLassoModelHeader *hdr;
	float	   *coef_dest;

	if (errstr)
		*errstr = NULL;
	if (!model || !model_data)
	{
		if (errstr)
			*errstr = pstrdup("invalid Lasso model for HIP pack");
		return -1;
	}
	payload_bytes = sizeof(NdbCudaLassoModelHeader)
		+ sizeof(float) * (size_t) model->n_features;
	blob = (bytea *) palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaLassoModelHeader *) base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->intercept = (float) model->intercept;
	hdr->lambda = model->lambda;
	hdr->max_iters = model->max_iters;
	hdr->r_squared = model->r_squared;
	hdr->mse = model->mse;
	hdr->mae = model->mae;

	coef_dest = (float *) (base + sizeof(NdbCudaLassoModelHeader));
	if (model->coefficients)
	{
		int			i;

		for (i = 0; i < model->n_features; i++)
			coef_dest[i] = (float) model->coefficients[i];
	}

	if (metrics)
	{
		StringInfoData buf;
		Jsonb	   *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"lasso\","
						 "\"storage\":\"gpu\","
						 "\"n_features\":%d,"
						 "\"n_samples\":%d,"
						 "\"lambda\":%.6f,"
						 "\"max_iters\":%d,"
						 "\"r_squared\":%.6f,"
						 "\"mse\":%.6f,"
						 "\"mae\":%.6f}",
						 model->n_features,
						 model->n_samples,
						 model->lambda,
						 model->max_iters,
						 model->r_squared,
						 model->mse,
						 model->mae);

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
														  jsonb_in,
														  CStringGetDatum(buf.data)));
		NDB_FREE(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_rocm_lasso_train(const float *features,
					 const double *targets,
					 int n_samples,
					 int feature_dim,
					 const Jsonb * hyperparams,
					 bytea * *model_data,
					 Jsonb * *metrics,
					 char **errstr)
{
	double		lambda = 0.01;
	int			max_iters = 1000;
	double	   *weights = NULL;
	double	   *weights_old = NULL;
	double	   *residuals = NULL;
	double		y_mean = 0.0;
	bytea	   *payload = NULL;
	Jsonb	   *metrics_json = NULL;
	int			iter,
				i,
				j;
	bool		converged = false;
	int			rc = -1;
	hipError_t	cuda_err;
	float	   *d_features = NULL;
	double	   *d_residuals = NULL;
	double	   *d_rho = NULL;
	double	   *d_z = NULL;
	double	   *h_rho = NULL;
	double	   *h_z = NULL;
	size_t		feature_bytes;
	size_t		residual_bytes;
	int			cleanup_needed = 0;

	if (errstr)
		*errstr = NULL;

	/* Defensive parameter validation */
	if (!features || !targets || n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for HIP "
							  "Lasso train");
		return -1;
	}

	if (n_samples > 10000000 || feature_dim > 100000)
	{
		if (errstr)
			*errstr = psprintf("neurondb: lasso: dataset too large: n_samples=%d, feature_dim=%d",
							   n_samples, feature_dim);
		return -1;
	}

	/* Extract hyperparameters from JSON */
	if (hyperparams)
	{
		JsonbIterator *it;
		JsonbValue	v;
		int			r;
		bool		parsed_lambda = false;
		bool		parsed_max_iters = false;

		PG_TRY();
		{
			it = JsonbIteratorInit((JsonbContainer *) & ((Jsonb *) hyperparams)->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "lambda") == 0)
					{
						if (v.type == jbvNumeric)
						{
							Numeric		num = DatumGetNumeric(v.val.numeric);
							double		val = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(num)));

							if (val > 0.0 && val < 1e6)
							{
								lambda = val;
								parsed_lambda = true;
							}
						}
						else if (v.type == jbvString)
						{
							lambda = strtod(v.val.string.val, NULL);
							if (lambda > 0.0 && lambda < 1e6)
								parsed_lambda = true;
						}
					}
					else if (strcmp(key, "max_iters") == 0)
					{
						if (v.type == jbvNumeric)
						{
							Numeric		num = DatumGetNumeric(v.val.numeric);
							int			val = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(num)));

							if (val > 0 && val <= 100000)
							{
								max_iters = val;
								parsed_max_iters = true;
							}
						}
						else if (v.type == jbvString)
						{
							max_iters = strtol(v.val.string.val, NULL, 10);
							if (max_iters > 0 && max_iters <= 100000)
								parsed_max_iters = true;
						}
					}
					NDB_FREE(key);
				}
			}
		}
		PG_CATCH();
		{
			FlushErrorState();
			elog(WARNING, "ROCm Lasso: Failed to parse hyperparameters JSONB");
		}
		PG_END_TRY();

		if (!parsed_lambda)
			elog(DEBUG2, "ROCm Lasso: lambda not found in hyperparameters, using default %.6f", lambda);
		if (!parsed_max_iters)
			elog(DEBUG2, "ROCm Lasso: max_iters not found in hyperparameters, using default %d", max_iters);
	}

	/* Compute mean of targets */
	for (i = 0; i < n_samples; i++)
	{
		if (!isfinite(targets[i]))
		{
			if (errstr)
				*errstr = psprintf("neurondb: lasso: invalid target value at index %d", i);
			return -1;
		}
		y_mean += targets[i];
	}
	y_mean /= (double) n_samples;

	weights = (double *) palloc0(sizeof(double) * feature_dim);
	weights_old = (double *) palloc(sizeof(double) * feature_dim);
	residuals = (double *) palloc(sizeof(double) * n_samples);
	h_rho = (double *) palloc(sizeof(double));
	h_z = (double *) palloc(sizeof(double));

	for (i = 0; i < n_samples; i++)
		residuals[i] = targets[i] - y_mean;

	/* Allocate GPU memory */
	feature_bytes = sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
	residual_bytes = sizeof(double) * (size_t) n_samples;

	cuda_err = hipMalloc((void **) &d_features, feature_bytes);
	if (cuda_err != hipSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: lasso: failed to allocate GPU memory for features: %s",
							   hipGetErrorString(cuda_err));
		goto cleanup_host;
	}
	cleanup_needed = 1;

	cuda_err = hipMalloc((void **) &d_residuals, residual_bytes);
	if (cuda_err != hipSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: lasso: failed to allocate GPU memory for residuals: %s",
							   hipGetErrorString(cuda_err));
		goto cleanup;
	}

	cuda_err = hipMalloc((void **) &d_rho, sizeof(double));
	if (cuda_err != hipSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: lasso: failed to allocate GPU memory for rho: %s",
							   hipGetErrorString(cuda_err));
		goto cleanup;
	}

	cuda_err = hipMalloc((void **) &d_z, sizeof(double));
	if (cuda_err != hipSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: lasso: failed to allocate GPU memory for z: %s",
							   hipGetErrorString(cuda_err));
		goto cleanup;
	}

	/* Copy features and residuals to GPU */
	cuda_err = hipMemcpy(d_features, features, feature_bytes, hipMemcpyHostToDevice);
	if (cuda_err != hipSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: lasso: failed to copy features to GPU: %s",
							   hipGetErrorString(cuda_err));
		goto cleanup;
	}

	cuda_err = hipMemcpy(d_residuals, residuals, residual_bytes, hipMemcpyHostToDevice);
	if (cuda_err != hipSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: lasso: failed to copy residuals to GPU: %s",
							   hipGetErrorString(cuda_err));
		goto cleanup;
	}

	/* GPU-accelerated coordinate descent */
	for (iter = 0; iter < max_iters && !converged; iter++)
	{
		double		diff;

		memcpy(weights_old, weights, sizeof(double) * feature_dim);

		for (j = 0; j < feature_dim; j++)
		{
			double		rho = 0.0;
			double		z = 0.0;
			double		old_weight;
			double		weight_diff;

			/* Compute rho on GPU */
			cuda_err = launch_lasso_compute_rho_kernel(d_features,
													   d_residuals,
													   n_samples,
													   feature_dim,
													   j,
													   d_rho);
			if (cuda_err != hipSuccess)
			{
				if (errstr)
					*errstr = psprintf("neurondb: lasso: rho kernel failed: %s",
									   hipGetErrorString(cuda_err));
				goto cleanup;
			}

			cuda_err = cudaDeviceSynchronize();
			if (cuda_err != hipSuccess)
			{
				if (errstr)
					*errstr = psprintf("neurondb: lasso: sync after rho failed: %s",
									   hipGetErrorString(cuda_err));
				goto cleanup;
			}

			cuda_err = hipMemcpy(h_rho, d_rho, sizeof(double), hipMemcpyDeviceToHost);
			if (cuda_err != hipSuccess)
			{
				if (errstr)
					*errstr = psprintf("neurondb: lasso: failed to copy rho from GPU: %s",
									   hipGetErrorString(cuda_err));
				goto cleanup;
			}
			rho = *h_rho;

			/* Compute z on GPU */
			cuda_err = launch_lasso_compute_z_kernel(d_features,
													 n_samples,
													 feature_dim,
													 j,
													 d_z);
			if (cuda_err != hipSuccess)
			{
				if (errstr)
					*errstr = psprintf("neurondb: lasso: z kernel failed: %s",
									   hipGetErrorString(cuda_err));
				goto cleanup;
			}

			cuda_err = cudaDeviceSynchronize();
			if (cuda_err != hipSuccess)
			{
				if (errstr)
					*errstr = psprintf("neurondb: lasso: sync after z failed: %s",
									   hipGetErrorString(cuda_err));
				goto cleanup;
			}

			cuda_err = hipMemcpy(h_z, d_z, sizeof(double), hipMemcpyDeviceToHost);
			if (cuda_err != hipSuccess)
			{
				if (errstr)
					*errstr = psprintf("neurondb: lasso: failed to copy z from GPU: %s",
									   hipGetErrorString(cuda_err));
				goto cleanup;
			}
			z = *h_z;

			if (z < 1e-10)
				continue;

			old_weight = weights[j];
			weights[j] = soft_threshold(rho / z, lambda / z);

			if (!isfinite(weights[j]))
			{
				weights[j] = 0.0;
			}

			weight_diff = weights[j] - old_weight;

			if (fabs(weight_diff) > 1e-12)
			{
				/* Update residuals on GPU */
				cuda_err = launch_lasso_update_residuals_kernel(d_features,
																d_residuals,
																n_samples,
																feature_dim,
																j,
																weight_diff);
				if (cuda_err != hipSuccess)
				{
					if (errstr)
						*errstr = psprintf("neurondb: lasso: residual update kernel failed: %s",
										   hipGetErrorString(cuda_err));
					goto cleanup;
				}

				cuda_err = cudaDeviceSynchronize();
				if (cuda_err != hipSuccess)
				{
					if (errstr)
						*errstr = psprintf("neurondb: lasso: sync after residual update failed: %s",
										   hipGetErrorString(cuda_err));
					goto cleanup;
				}
			}
		}

		diff = 0.0;
		for (j = 0; j < feature_dim; j++)
		{
			double		d = weights[j] - weights_old[j];

			diff += d * d;
		}
		if (sqrt(diff) < 1e-6)
			converged = true;
	}

	/* Copy residuals back to host for final metrics computation */
	cuda_err = hipMemcpy(residuals, d_residuals, residual_bytes, hipMemcpyDeviceToHost);
	if (cuda_err != hipSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: lasso: failed to copy residuals from GPU: %s",
							   hipGetErrorString(cuda_err));
		goto cleanup;
	}

	/* Build model */
	{
		LassoModel	model;
		double		ss_tot = 0.0;
		double		ss_res = 0.0;
		double		mse = 0.0;
		double		mae = 0.0;

		model.n_features = feature_dim;
		model.n_samples = n_samples;
		model.intercept = y_mean;
		model.lambda = lambda;
		model.max_iters = max_iters;
		model.coefficients = (double *) palloc(sizeof(double) * feature_dim);
		for (i = 0; i < feature_dim; i++)
		{
			if (!isfinite(weights[i]))
				weights[i] = 0.0;
			model.coefficients[i] = weights[i];
		}

		/* Compute metrics using residuals already on host */
		for (i = 0; i < n_samples; i++)
		{
			double		error = residuals[i];
			double		target_diff = targets[i] - y_mean;

			mse += error * error;
			mae += fabs(error);
			ss_res += error * error;
			ss_tot += target_diff * target_diff;
		}

		mse /= (double) n_samples;
		mae /= (double) n_samples;
		model.r_squared = (ss_tot > 1e-10) ? (1.0 - (ss_res / ss_tot)) : 0.0;
		model.mse = mse;
		model.mae = mae;

		if (!isfinite(model.r_squared))
			model.r_squared = 0.0;
		if (!isfinite(model.mse))
			model.mse = 0.0;
		if (!isfinite(model.mae))
			model.mae = 0.0;

		rc = ndb_rocm_lasso_pack_model(
									   &model, &payload, &metrics_json, errstr);

		NDB_FREE(model.coefficients);
	}

	rc = 0;

cleanup:
	if (cleanup_needed)
	{
		if (d_features)
			hipFree(d_features);
		if (d_residuals)
			hipFree(d_residuals);
		if (d_rho)
			hipFree(d_rho);
		if (d_z)
			hipFree(d_z);
	}

cleanup_host:
	NDB_FREE(weights);
	NDB_FREE(weights_old);
	NDB_FREE(residuals);
	NDB_FREE(h_rho);
	NDB_FREE(h_z);

	if (rc == 0 && payload)
	{
		*model_data = payload;
		if (metrics)
			*metrics = metrics_json;
		return 0;
	}

	if (payload)
		NDB_FREE(payload);
	if (metrics_json)
		NDB_FREE(metrics_json);

	return -1;
}

int
ndb_rocm_lasso_predict(const bytea * model_data,
					   const float *input,
					   int feature_dim,
					   double *prediction_out,
					   char **errstr)
{
	const		NdbCudaLassoModelHeader *hdr;
	const float *coefficients;
	const		bytea *detoasted;
	double		prediction;
	int			i;

	if (errstr)
		*errstr = NULL;
	if (!model_data || !input || !prediction_out)
	{
		if (errstr)
			*errstr = pstrdup(
							  "invalid parameters for HIP Lasso predict");
		return -1;
	}

	detoasted = (const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t		expected_size = sizeof(NdbCudaLassoModelHeader)
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

	hdr = (const NdbCudaLassoModelHeader *) VARDATA(detoasted);
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
									+ sizeof(NdbCudaLassoModelHeader));

	prediction = hdr->intercept;
	for (i = 0; i < feature_dim; i++)
		prediction += coefficients[i] * input[i];

	*prediction_out = prediction;
	return 0;
}

/*
 * ndb_rocm_lasso_evaluate
 *    GPU-accelerated batch evaluation for Lasso Regression
 *
 * Reuses the linear regression evaluation kernel since evaluation
 * is identical (just computes predictions and metrics).
 */
int
ndb_rocm_lasso_evaluate(const bytea * model_data,
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
	const		NdbCudaLassoModelHeader *hdr;
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
			*errstr = pstrdup("neurondb: ndb_rocm_lasso_evaluate: model_data is NULL");
		return -1;
	}

	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_rocm_lasso_evaluate: features is NULL");
		return -1;
	}

	if (targets == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_rocm_lasso_evaluate: targets is NULL");
		return -1;
	}

	if (n_samples <= 0)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: invalid n_samples %d",
							   n_samples);
		return -1;
	}

	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: invalid feature_dim %d",
							   feature_dim);
		return -1;
	}

	if (mse_out == NULL || mae_out == NULL || rmse_out == NULL || r_squared_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_rocm_lasso_evaluate: output pointers are NULL");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t		expected_size = sizeof(NdbCudaLassoModelHeader)
			+ sizeof(float) * (size_t) feature_dim;
		size_t		actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: model data too small: "
								   "expected %zu bytes, got %zu",
								   expected_size,
								   actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaLassoModelHeader *) VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: feature dimension mismatch: "
							   "model has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		return -1;
	}

	coefficients = (const float *) ((const char *) hdr + sizeof(NdbCudaLassoModelHeader));

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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to allocate GPU memory for features: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to allocate GPU memory for targets: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to allocate GPU memory for coefficients: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to allocate GPU memory for SSE: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to allocate GPU memory for SAE: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to allocate GPU memory for count: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to copy features to GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to copy targets to GPU: %s",
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
				*errstr = pstrdup("neurondb: ndb_rocm_lasso_evaluate: failed to allocate host memory for coefficients");
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
				*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to copy coefficients to GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: evaluation kernel failed: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to copy SSE from GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to copy SAE from GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: failed to copy count from GPU: %s",
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
			*errstr = psprintf("neurondb: ndb_rocm_lasso_evaluate: count mismatch: expected %d, got %lld",
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
