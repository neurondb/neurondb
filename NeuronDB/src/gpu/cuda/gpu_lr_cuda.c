/*-------------------------------------------------------------------------
 *
 * gpu_lr_cuda.c
 *    Logistic regression bridge implementation.
 *
 * This module provides bridge functions for logistic regression training
 * and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_lr_cuda.c
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

#include "ml_logistic_regression_internal.h"
#include "neurondb_cuda_lr.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_guc.h"
#include "neurondb_constants.h"

int
ndb_cuda_lr_pack_model(const LRModel *model,
					   bytea * *model_data,
					   Jsonb * *metrics,
					   char **errstr)
{
	size_t		payload_bytes;
	char	   *base;
	NdbCudaLrModelHeader *hdr;
	float	   *weights_dest;
	bytea	   *blob = NULL;
	char	   *blob_raw = NULL;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid LR model for CUDA pack");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaLrModelHeader) +
		sizeof(float) * (size_t) model->n_features;

	NDB_ALLOC(blob_raw, char, VARHDRSZ + payload_bytes);
	blob = (bytea *) blob_raw;
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaLrModelHeader *) base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->max_iters = model->max_iters;
	hdr->learning_rate = model->learning_rate;
	hdr->lambda = model->lambda;
	hdr->bias = model->bias;

	weights_dest = (float *) (base + sizeof(NdbCudaLrModelHeader));
	if (model->weights != NULL)
	{
		int			i;

		for (i = 0; i < model->n_features; i++)
			weights_dest[i] = (float) model->weights[i];
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb	   *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"logistic_regression\","
						 "\"training_backend\":1,"
						 "\"n_features\":%d,"
						 "\"n_samples\":%d,"
						 "\"max_iters\":%d,"
						 "\"learning_rate\":%.6f,"
						 "\"lambda\":%.6f,"
						 "\"final_loss\":%.6f,"
						 "\"accuracy\":%.6f}",
						 model->n_features,
						 model->n_samples,
						 model->max_iters,
						 model->learning_rate,
						 model->lambda,
						 model->final_loss,
						 model->accuracy);

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
ndb_cuda_lr_train(const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  const Jsonb * hyperparams,
				  bytea * *model_data,
				  Jsonb * *metrics,
				  char **errstr)
{
	const int	default_max_iters = 1000;
	const double default_learning_rate = 0.01;
	const double default_lambda = 0.001;
	int			max_iters = default_max_iters;
	double		learning_rate = default_learning_rate;
	double		lambda = default_lambda;
	float	   *d_features = NULL;
	double	   *d_predictions = NULL;
	double	   *d_z = NULL;
	double	   *weights = NULL;
	double	   *grad_weights = NULL;
	double		bias = 0.0;
	double		grad_bias = 0.0;
	bytea	   *payload = NULL;
	Jsonb	   *metrics_json = NULL;
	cudaError_t status = cudaSuccess;
	size_t		feature_bytes;
	size_t		label_bytes;
	size_t		pred_bytes;
	size_t		z_bytes;
	size_t		weight_bytes_gpu;
	size_t		free_mem;
	size_t		total_mem;
	float	   *d_weights = NULL;
	float	   *d_z_float = NULL;	/* Device buffer for float z (cuBLAS) */
	float	   *d_errors = NULL;	/* Device buffer for errors */
	float	   *d_grad_weights_float = NULL;	/* Device buffer for gradient
												 * weights (float) */
	double	   *d_labels_double = NULL; /* Device buffer for labels (double) */
	float	   *h_errors = NULL;	/* Host buffer for errors (allocated once,
									 * used only in fallback) */
	float	   *h_grad_weights_float = NULL;	/* Host buffer for gradient
												 * weights (used only in
												 * fallback) */
	double	   *d_error_sum = NULL; /* Device buffer for error sum reduction */
	double	   *d_bias = NULL;	/* Device buffer for bias (updated in-place) */
	size_t		z_float_bytes;
	size_t		error_bytes;
	size_t		grad_weight_bytes_float;
	int			iter;
	int			i;
	int			rc = -1;

	/* CPU mode: never execute GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = pstrdup("CUDA lr_train: CPU mode - GPU code should not be called");
		return -1;
	}

	if (errstr)
		*errstr = NULL;

	/* Defensive: Comprehensive parameter validation before proceeding */
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ndb_cuda_lr_train: model_data is NULL");
		return -1;
	}

	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ndb_cuda_lr_train: features is NULL");
		return -1;
	}

	if (labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ndb_cuda_lr_train: labels is NULL");
		return -1;
	}

	if (n_samples <= 0 || n_samples > 10000000)
	{
		if (errstr)
			*errstr = psprintf("ndb_cuda_lr_train: invalid n_samples %d (must be 1-10000000)",
							   n_samples);
		return -1;
	}

	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = psprintf("ndb_cuda_lr_train: invalid feature_dim %d (must be 1-10000)",
							   feature_dim);
		return -1;
	}

	/* Defensive: Check for potential overflow in size calculations */
	{
		size_t		max_feature_bytes = (size_t) n_samples * (size_t) feature_dim * sizeof(float);

		if (max_feature_bytes / sizeof(float) / (size_t) feature_dim != (size_t) n_samples)
		{
			if (errstr)
				*errstr = psprintf("ndb_cuda_lr_train: size overflow: n_samples=%d * feature_dim=%d",
								   n_samples, feature_dim);
			return -1;
		}
	}

	/* Defensive: Validate feature values are finite (check first few samples) */
	{
		int			check_samples = (n_samples < 100) ? n_samples : 100;
		int			check_i,
					check_j;

		for (check_i = 0; check_i < check_samples; check_i++)
		{
			for (check_j = 0; check_j < feature_dim; check_j++)
			{
				float		val = features[check_i * feature_dim + check_j];

				if (!isfinite(val))
				{
					if (errstr)
						*errstr = psprintf("ndb_cuda_lr_train: non-finite feature value at sample %d, feature %d: %f",
										   check_i, check_j, val);
					return -1;
				}
			}
		}
	}

	/*
	 * Defensive: Validate label values are 0.0 or 1.0 (check first few
	 * samples)
	 */
	{
		int			check_samples = (n_samples < 100) ? n_samples : 100;
		int			check_i;

		for (check_i = 0; check_i < check_samples; check_i++)
		{
			double		label_val = labels[check_i];

			if (!isfinite(label_val) || (label_val != 0.0 && label_val != 1.0))
			{
				if (errstr)
					*errstr = psprintf("ndb_cuda_lr_train: invalid label value at sample %d: %f (must be 0.0 or 1.0)",
									   check_i, label_val);
				return -1;
			}
		}
	}

	elog(DEBUG1,
		 "ndb_cuda_lr_train: entry: model_data=%p, features=%p, "
		 "labels=%p, n_samples=%d, feature_dim=%d",
		 model_data,
		 features,
		 labels,
		 n_samples,
		 feature_dim);

	elog(DEBUG1,
		 "ndb_cuda_lr_train: starting training: n_samples=%d, "
		 "feature_dim=%d",
		 n_samples,
		 feature_dim);

	/* Extract and validate hyperparameters from JSONB */
	if (hyperparams != NULL)
	{
		Datum		max_iters_datum;
		Datum		lr_datum;
		Datum		lambda_datum;
		Datum		numeric_datum;
		Numeric		num;
		int			parsed_max_iters;
		double		parsed_lr;
		double		parsed_lambda;

		max_iters_datum = DirectFunctionCall2(jsonb_object_field,
											  JsonbPGetDatum(hyperparams),
											  CStringGetTextDatum("max_iters"));
		if (DatumGetPointer(max_iters_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(jsonb_numeric, max_iters_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				parsed_max_iters = DatumGetInt32(DirectFunctionCall1(
																	 numeric_int4, NumericGetDatum(num)));
				/* Defensive: Validate max_iters range */
				if (parsed_max_iters > 0 && parsed_max_iters <= 1000000)
				{
					max_iters = parsed_max_iters;
				}
				else if (parsed_max_iters > 1000000)
				{
					elog(WARNING,
						 "ndb_cuda_lr_train: max_iters %d exceeds maximum 1000000, using default %d",
						 parsed_max_iters, default_max_iters);
					max_iters = default_max_iters;
				}
				else
				{
					elog(WARNING,
						 "ndb_cuda_lr_train: invalid max_iters %d, using default %d",
						 parsed_max_iters, default_max_iters);
					max_iters = default_max_iters;
				}
			}
		}

		lr_datum = DirectFunctionCall2(jsonb_object_field,
									   JsonbPGetDatum(hyperparams),
									   CStringGetTextDatum("learning_rate"));
		if (DatumGetPointer(lr_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(jsonb_numeric, lr_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				parsed_lr = DatumGetFloat8(
										   DirectFunctionCall1(numeric_float8, NumericGetDatum(num)));

				/*
				 * Defensive: Validate learning_rate range and check for
				 * NaN/Inf
				 */
				if (isfinite(parsed_lr) && parsed_lr > 0.0 && parsed_lr <= 10.0)
				{
					learning_rate = parsed_lr;
				}
				else
				{
					elog(WARNING,
						 "ndb_cuda_lr_train: invalid learning_rate %f (must be finite, > 0, <= 10.0), using default %f",
						 parsed_lr, default_learning_rate);
					learning_rate = default_learning_rate;
				}
			}
		}

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
						 "ndb_cuda_lr_train: invalid lambda %f (must be finite, >= 0, <= 1000.0), using default %f",
						 parsed_lambda, default_lambda);
					lambda = default_lambda;
				}
			}
		}
	}

	/* Defensive: Final validation of hyperparameters */
	if (max_iters <= 0 || max_iters > 1000000)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: max_iters %d out of range, using default %d",
			 max_iters, default_max_iters);
		max_iters = default_max_iters;
	}
	if (!isfinite(learning_rate) || learning_rate <= 0.0 || learning_rate > 10.0)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: learning_rate %f invalid, using default %f",
			 learning_rate, default_learning_rate);
		learning_rate = default_learning_rate;
	}
	if (!isfinite(lambda) || lambda < 0.0 || lambda > 1000.0)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: lambda %f invalid, using default %f",
			 lambda, default_lambda);
		lambda = default_lambda;
	}

	/* Allocate host memory with defensive checks */
	weights = (double *) palloc0(sizeof(double) * (size_t) feature_dim);
	if (weights == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ndb_cuda_lr_train: palloc0 weights failed");
		return -1;
	}
	NDB_ALLOC(grad_weights, double, feature_dim);
	if (grad_weights == NULL)
	{
		NDB_FREE(weights);
		if (errstr)
			*errstr = pstrdup("ndb_cuda_lr_train: palloc grad_weights failed");
		return -1;
	}

	/* Allocate GPU memory */
	feature_bytes = sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
	label_bytes = sizeof(double) * (size_t) n_samples;
	pred_bytes = sizeof(double) * (size_t) n_samples;
	z_bytes = sizeof(double) * (size_t) n_samples;
	weight_bytes_gpu = sizeof(float) * (size_t) feature_dim;

	elog(DEBUG1,
		 "ndb_cuda_lr_train: allocating GPU memory: feature_bytes=%zu "
		 "(%.2f MB), label_bytes=%zu",
		 feature_bytes,
		 feature_bytes / (1024.0 * 1024.0),
		 label_bytes);

	/* Defensive: Check CUDA context before proceeding */
	status = cudaGetLastError();
	if (status != cudaSuccess && status != cudaErrorNotReady)
	{
		if (errstr)
			*errstr = psprintf("ndb_cuda_lr_train: CUDA error before allocation: %s",
							   cudaGetErrorString(status));
		elog(WARNING,
			 "ndb_cuda_lr_train: CUDA error detected before allocation: %s",
			 cudaGetErrorString(status));
		return -1;
	}

	/* Check available GPU memory before allocation */
	status = cudaMemGetInfo(&free_mem, &total_mem);
	if (status != cudaSuccess)
	{
		if (errstr)
			*errstr = psprintf("ndb_cuda_lr_train: cudaMemGetInfo failed: %s",
							   cudaGetErrorString(status));
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMemGetInfo failed: %s",
			 cudaGetErrorString(status));
		return -1;
	}

	elog(DEBUG1,
		 "ndb_cuda_lr_train: GPU memory: free=%.2f MB, total=%.2f MB",
		 free_mem / (1024.0 * 1024.0),
		 total_mem / (1024.0 * 1024.0));

	/* Defensive: Check if we have enough GPU memory (with safety margin) */
	{
		size_t		required_mem;
		size_t		safety_margin;
		size_t		total_required;
		size_t		temp_buffers;

		/* Initialize buffer sizes before using them */
		z_float_bytes = sizeof(float) * (size_t) n_samples;
		error_bytes = sizeof(float) * (size_t) n_samples;
		grad_weight_bytes_float = sizeof(float) * (size_t) feature_dim;

		required_mem = feature_bytes + label_bytes + pred_bytes + z_bytes + weight_bytes_gpu;
		safety_margin = required_mem / 10;	/* 10% safety margin */
		total_required = required_mem + safety_margin;

		/* Also account for temporary buffers used during training */
		temp_buffers = z_float_bytes + error_bytes + grad_weight_bytes_float;
		total_required += temp_buffers;

		if (free_mem < total_required)
		{
			if (errstr)
				*errstr = psprintf("ndb_cuda_lr_train: insufficient GPU memory: need %zu bytes (%.2f MB), have %zu bytes (%.2f MB)",
								   total_required,
								   total_required / (1024.0 * 1024.0),
								   free_mem,
								   free_mem / (1024.0 * 1024.0));
			elog(WARNING,
				 "ndb_cuda_lr_train: insufficient GPU memory: need %.2f MB, have %.2f MB",
				 total_required / (1024.0 * 1024.0),
				 free_mem / (1024.0 * 1024.0));
			NDB_FREE(weights);
			NDB_FREE(grad_weights);
			return -1;
		}
	}

	status = cudaMalloc((void **) &d_features, feature_bytes);
	if (status != cudaSuccess)
	{
		if (errstr)
			*errstr = psprintf("ndb_cuda_lr_train: cudaMalloc d_features failed: %s",
							   cudaGetErrorString(status));
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMalloc d_features failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	status = cudaMalloc((void **) &d_predictions, pred_bytes);
	if (status != cudaSuccess)
	{
		if (errstr)
			*errstr = psprintf("ndb_cuda_lr_train: cudaMalloc d_predictions failed: %s",
							   cudaGetErrorString(status));
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMalloc d_predictions failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}
	status = cudaMalloc((void **) &d_z, z_bytes);
	if (status != cudaSuccess)
	{
		if (errstr)
			*errstr = psprintf("ndb_cuda_lr_train: cudaMalloc d_z failed: %s",
							   cudaGetErrorString(status));
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMalloc d_z failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	elog(DEBUG1,
		 "ndb_cuda_lr_train: all GPU memory allocated successfully");

	/*
	 * Convert features from row-major to column-major for cuBLAS
	 * compatibility
	 */

	/*
	 * cuBLAS expects column-major layout: X[i + j * n_samples] = features[i *
	 * feature_dim + j]
	 */
	/* Optimized: Use block-based conversion for better cache locality */
	{
		char	   *h_features_col_raw = NULL;
		float	   *h_features_col;
		int			conv_i;
		const int	BLOCK_SIZE = 64;	/* Cache-friendly block size */
		NDB_ALLOC(h_features_col_raw, char, feature_bytes);
		h_features_col = (float *) h_features_col_raw;

		if (h_features_col == NULL)
		{
			if (errstr && *errstr == NULL)
				*errstr = pstrdup("ndb_cuda_lr_train: failed to allocate column-major feature buffer");
			goto gpu_fail;
		}

		/* Optimized: Block-based conversion for better cache performance */
		for (conv_i = 0; conv_i < n_samples; conv_i += BLOCK_SIZE)
		{
			int			i_end = (conv_i + BLOCK_SIZE < n_samples) ? conv_i + BLOCK_SIZE : n_samples;
			int			block_i;
			int			feat_j;

			for (block_i = conv_i; block_i < i_end; block_i++)
			{
				for (feat_j = 0; feat_j < feature_dim; feat_j++)
				{
					float		val = features[block_i * feature_dim + feat_j];

					/* Defensive: Check for NaN/Inf during conversion */
					if (!isfinite(val))
					{
						NDB_FREE(h_features_col);
						if (errstr && *errstr == NULL)
							*errstr = psprintf("ndb_cuda_lr_train: non-finite feature value at sample %d, feature %d: %f",
											   block_i, feat_j, val);
						goto gpu_fail;
					}
					/* Row-major: features[i * feature_dim + j] */
					/* Column-major: h_features_col[i + j * n_samples] */
					h_features_col[block_i + feat_j * n_samples] = val;
				}
			}
		}

		/* Copy column-major features to GPU */
		status = cudaMemcpy(d_features, h_features_col, feature_bytes, cudaMemcpyHostToDevice);
		NDB_FREE(h_features_col);
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_lr_train: cudaMemcpy d_features (column-major) failed: %s",
				 cudaGetErrorString(status));
			if (errstr && *errstr == NULL)
				*errstr = psprintf("ndb_cuda_lr_train: cudaMemcpy d_features failed: %s",
								   cudaGetErrorString(status));
			goto gpu_fail;
		}

		/* Defensive: Verify the copy succeeded */
		status = cudaGetLastError();
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_lr_train: CUDA error after cudaMemcpy d_features: %s",
				 cudaGetErrorString(status));
			if (errstr && *errstr == NULL)
				*errstr = psprintf("ndb_cuda_lr_train: CUDA error after feature copy: %s",
								   cudaGetErrorString(status));
			goto gpu_fail;
		}
	}

	/* Allocate GPU memory for weights (needed for forward pass) */
	status = cudaMalloc((void **) &d_weights, weight_bytes_gpu);
	if (status != cudaSuccess)
	{
		if (errstr)
			*errstr = psprintf("ndb_cuda_lr_train: cudaMalloc d_weights failed: %s",
							   cudaGetErrorString(status));
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMalloc d_weights failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	/* Allocate fixed device buffers once before the loop (optimization) */
	z_float_bytes = sizeof(float) * (size_t) n_samples;
	error_bytes = sizeof(float) * (size_t) n_samples;
	grad_weight_bytes_float = sizeof(float) * (size_t) feature_dim;

	status = cudaMalloc((void **) &d_z_float, z_float_bytes);
	if (status != cudaSuccess)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMalloc d_z_float failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	status = cudaMalloc((void **) &d_errors, error_bytes);
	if (status != cudaSuccess)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMalloc d_errors failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	status = cudaMalloc((void **) &d_grad_weights_float, grad_weight_bytes_float);
	if (status != cudaSuccess)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMalloc d_grad_weights_float failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	status = cudaMalloc((void **) &d_labels_double, label_bytes);
	if (status != cudaSuccess)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMalloc d_labels_double failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	/* Copy labels to device once (they don't change during training) */
	status = cudaMemcpy(d_labels_double, labels, label_bytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMemcpy d_labels_double failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	/* Allocate host buffer for gradient weights (used only for fallback) */
	NDB_ALLOC(h_grad_weights_float, float, feature_dim);
	if (h_grad_weights_float == NULL)
	{
		goto gpu_fail;
	}

	/* Initialize weights on device once (convert double to float) */
	{
		int			j;

		NDB_DECLARE(float *, h_weights_init);
		NDB_ALLOC(h_weights_init, float, feature_dim);

		if (h_weights_init == NULL)
		{
			goto gpu_fail;
		}

		for (j = 0; j < feature_dim; j++)
			h_weights_init[j] = (float) weights[j];

		status = cudaMemcpy(d_weights, h_weights_init, weight_bytes_gpu, cudaMemcpyHostToDevice);
		NDB_FREE(h_weights_init);

		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_lr_train: initial cudaMemcpy weights failed: %s",
				 cudaGetErrorString(status));
			goto gpu_fail;
		}
	}

	/* Allocate device buffer for error sum reduction (for grad_bias) */
	status = cudaMalloc((void **) &d_error_sum, sizeof(double));
	if (status != cudaSuccess)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMalloc d_error_sum failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	/* Allocate device buffer for bias (for in-place updates) */
	status = cudaMalloc((void **) &d_bias, sizeof(double));
	if (status != cudaSuccess)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMalloc d_bias failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	/* Initialize bias on device */
	status = cudaMemcpy(d_bias, &bias, sizeof(double), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: cudaMemcpy d_bias failed: %s",
			 cudaGetErrorString(status));
		goto gpu_fail;
	}

	elog(DEBUG1, "ndb_cuda_lr_train: all fixed device buffers allocated successfully");

	/* Gradient descent */
	for (iter = 0; iter < max_iters; iter++)
	{
		/* Weights are already on device - no copy needed per iteration */

		/* Forward pass: compute z = X * w + b using cuBLAS SGEMV (optimized) */
		{
			cublasHandle_t handle = ndb_cuda_get_cublas_handle();
			cublasStatus_t cublas_err;
			float		alpha = 1.0f;
			float		beta = 0.0f;	/* Correct beta parameter */

			/* cuBLAS is required - fail if handle is not available */
			if (handle == NULL)
			{
				if ((iter % 100) == 0)
				{
					elog(WARNING,
						 "ndb_cuda_lr_train: cuBLAS handle not available at iter %d",
						 iter);
				}
				if (errstr && *errstr == NULL)
					*errstr = pstrdup("cuBLAS handle not available");
				status = cudaErrorUnknown;
				goto gpu_fail;
			}

			/* Initialize z_float to zero */
			status = cudaMemset(d_z_float, 0, z_float_bytes);
			if (status != cudaSuccess)
			{
				goto gpu_fail;
			}

			/*
			 * Defensive: Validate alpha and beta are finite before cuBLAS
			 * call
			 */
			if (!isfinite(alpha))
			{
				elog(WARNING,
					 "ndb_cuda_lr_train: non-finite alpha=%f at iter %d, using 1.0",
					 alpha, iter);
				alpha = 1.0f;
			}
			if (!isfinite(beta))
			{
				elog(WARNING,
					 "ndb_cuda_lr_train: non-finite beta=%f at iter %d, using 0.0",
					 beta, iter);
				beta = 0.0f;
			}

			/* Use cuBLAS SGEMV: z = alpha * X * w + beta * z */
			/* X is (n_samples x feature_dim), w is (feature_dim x 1) */
			/* Result z is (n_samples x 1) */
			/* CUBLAS_OP_N: no transpose, so X * w */
			/* z = 1.0 * X * w + 0.0 * z (z starts as zeros) */
			cublas_err = cublasSgemv(handle,
									 CUBLAS_OP_N,	/* No transpose X */
									 n_samples, /* M: rows of X */
									 feature_dim,	/* N: cols of X */
									 &alpha,	/* alpha = 1.0 */
									 d_features,	/* X: (n_samples x
													 * feature_dim), leading
													 * dim = n_samples */
									 n_samples, /* lda: leading dimension of X */
									 d_weights, /* w: (feature_dim x 1) */
									 1, /* incx: stride of w */
									 &beta, /* beta = 0.0: don't add to
											 * existing z */
									 d_z_float, /* z: (n_samples x 1) */
									 1);	/* incy: stride of z */

			/* Defensive: Check for CUDA errors after cuBLAS call */
			status = cudaGetLastError();
			if (status != cudaSuccess && cublas_err == CUBLAS_STATUS_SUCCESS)
			{
				elog(WARNING,
					 "ndb_cuda_lr_train: CUDA error after cuBLAS SGEMV forward: %s",
					 cudaGetErrorString(status));
				cublas_err = CUBLAS_STATUS_EXECUTION_FAILED;
			}

			if (cublas_err != CUBLAS_STATUS_SUCCESS)
			{
				if ((iter % 100) == 0)
				{
					elog(WARNING,
						 "ndb_cuda_lr_train: cuBLAS SGEMV failed with status %d at iter %d",
						 cublas_err, iter);
				}
				if (errstr && *errstr == NULL)
					*errstr = psprintf("cublasSgemv failed with status %d",
									   (int) cublas_err);
				status = cudaErrorUnknown;
				goto gpu_fail;
			}

			/*
			 * Convert float z to double z and add bias directly on device
			 * (optimized)
			 */
			/* Read bias from device */
			{
				int			rc_convert;
				double		bias_device;

				status = cudaMemcpy(&bias_device, d_bias, sizeof(double), cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
				{
					if ((iter % 100) == 0)
					{
						elog(WARNING,
							 "ndb_cuda_lr_train: cudaMemcpy d_bias failed: %s",
							 cudaGetErrorString(status));
					}
					goto gpu_fail;
				}

				rc_convert = ndb_cuda_lr_convert_z_add_bias_gpu(d_z_float, bias_device, n_samples, d_z);
				if (rc_convert != 0)
				{
					if ((iter % 100) == 0)
					{
						elog(WARNING,
							 "ndb_cuda_lr_train: convert_z_add_bias GPU kernel failed at iter %d",
							 iter);
					}
					if (errstr && *errstr == NULL)
						*errstr = pstrdup("convert_z_add_bias GPU kernel failed");
					status = cudaGetLastError();
					goto gpu_fail;
				}

				if ((iter % 100) == 0)
				{
					elog(DEBUG1,
						 "neurondb: logistic_regression: cuBLAS SGEMV forward pass succeeded at iter %d",
						 iter);
				}
			}
		}

		/*
		 * Apply sigmoid directly on device (optimization: avoid host/device
		 * copies)
		 */
		{
			int			rc_sigmoid;

			rc_sigmoid = ndb_cuda_lr_sigmoid_gpu(d_z, n_samples, d_predictions);
			if (rc_sigmoid != 0)
			{
				if ((iter % 100) == 0)
				{
					elog(WARNING,
						 "ndb_cuda_lr_train: sigmoid GPU kernel failed at iter %d",
						 iter);
				}
				if (errstr && *errstr == NULL)
					*errstr = pstrdup("sigmoid GPU kernel failed");
				status = cudaGetLastError();
				goto gpu_fail;
			}
		}

		/* Compute gradients using cuBLAS SGEMV for maximum GPU utilization */
		/* grad_w = X' * (predictions - labels) / n_samples */
		/* grad_bias = sum(predictions - labels) / n_samples */
		{
			cublasHandle_t handle = ndb_cuda_get_cublas_handle();
			cublasStatus_t cublas_err = CUBLAS_STATUS_NOT_INITIALIZED;
			float		alpha = 1.0f / (float) n_samples;	/* Average gradient */
			float		beta = 0.0f;
			double		error_sum = 0.0;
			double		grad_bias_device = 0.0;
			int			rc_errors;
			int			rc_reduce;

			/* Compute errors = predictions - labels directly on device */
			rc_errors = ndb_cuda_lr_compute_errors_gpu(d_predictions, d_labels_double, d_errors, n_samples);
			if (rc_errors != 0)
			{
				if ((iter % 100) == 0)
				{
					elog(WARNING,
						 "ndb_cuda_lr_train: compute_errors GPU kernel failed at iter %d",
						 iter);
				}
				if (errstr && *errstr == NULL)
					*errstr = pstrdup("compute_errors GPU kernel failed");
				status = cudaGetLastError();
				goto gpu_fail;
			}

			/*
			 * Compute error_sum on device using reduction kernel
			 * (optimization)
			 */
			status = cudaMemset(d_error_sum, 0, sizeof(double));
			if (status != cudaSuccess)
			{
				if ((iter % 100) == 0)
				{
					elog(WARNING,
						 "ndb_cuda_lr_train: cudaMemset d_error_sum failed: %s",
						 cudaGetErrorString(status));
				}
				goto gpu_fail;
			}

			rc_reduce = ndb_cuda_lr_reduce_errors_bias_gpu(d_errors, n_samples, d_error_sum);
			if (rc_reduce != 0)
			{
				if ((iter % 100) == 0)
				{
					elog(WARNING,
						 "ndb_cuda_lr_train: reduce_errors_bias GPU kernel failed at iter %d",
						 iter);
				}
				if (errstr && *errstr == NULL)
					*errstr = pstrdup("reduce_errors_bias GPU kernel failed");
				status = cudaGetLastError();
				goto gpu_fail;
			}

			/* Copy error_sum back to host (single value) */
			status = cudaMemcpy(&error_sum, d_error_sum, sizeof(double), cudaMemcpyDeviceToHost);
			if (status != cudaSuccess)
			{
				if ((iter % 100) == 0)
				{
					elog(WARNING,
						 "ndb_cuda_lr_train: cudaMemcpy d_error_sum failed: %s",
						 cudaGetErrorString(status));
				}
				goto gpu_fail;
			}

			grad_bias_device = error_sum / (double) n_samples;

			/*
			 * Use cuBLAS SGEMV with column-major features (converted before
			 * training loop)
			 */
			if (handle != NULL)
			{
				/* Initialize gradient weights to zero */
				status = cudaMemset(d_grad_weights_float, 0, grad_weight_bytes_float);
				if (status != cudaSuccess)
				{
					if ((iter % 100) == 0)
					{
						elog(WARNING,
							 "ndb_cuda_lr_train: cudaMemset d_grad_weights_float failed: %s",
							 cudaGetErrorString(status));
					}
					goto gpu_fail;
				}

				/* Use cuBLAS SGEMV: grad_w = alpha * X' * errors */
				/* X' is (feature_dim x n_samples), errors is (n_samples x 1) */
				/* Result grad_w is (feature_dim x 1) */
				/* CUBLAS_OP_T: transpose X, so X' * errors */
				cublas_err = cublasSgemv(handle,
										 CUBLAS_OP_T,	/* Transpose X: X' */
										 n_samples, /* M: rows of X (before
													 * transpose) */
										 feature_dim,	/* N: cols of X (before
														 * transpose) */
										 &alpha,	/* alpha = 1/n_samples
													 * (average) */
										 d_features,	/* X: (n_samples x
														 * feature_dim), leading
														 * dim = n_samples */
										 n_samples, /* lda: leading dimension
													 * of X */
										 d_errors,	/* errors: (n_samples x 1) */
										 1, /* incx: stride of errors */
										 &beta, /* beta = 0.0: don't add to
												 * existing */
										 d_grad_weights_float,	/* grad_w: (feature_dim
																 * x 1) */
										 1);	/* incy: stride of grad_w */

				if (cublas_err == CUBLAS_STATUS_SUCCESS)
				{
					/*
					 * Update weights and bias directly on device
					 * (optimization)
					 */
					int			rc_update;
					float		lr_float = (float) learning_rate;
					float		lambda_float = (float) lambda;

					rc_update = ndb_cuda_lr_update_weights_gpu(d_weights,
															   d_grad_weights_float,
															   lr_float,
															   lambda_float,
															   feature_dim,
															   d_bias,
															   grad_bias_device);

					if (rc_update == 0)
					{
						/* Update host-side bias for metrics */
						status = cudaMemcpy(&bias, d_bias, sizeof(double), cudaMemcpyDeviceToHost);
						if (status == cudaSuccess)
						{
							grad_bias = grad_bias_device;
							if ((iter % 100) == 0)
							{
								elog(DEBUG1,
									 "neurondb: logistic_regression: cuBLAS SGEMV gradient computation succeeded at iter %d",
									 iter);
							}
							goto gradient_done;
						}
					}

					if ((iter % 100) == 0)
					{
						elog(WARNING,
							 "ndb_cuda_lr_train: weight update kernel failed at iter %d",
							 iter);
					}
				}
				else
				{
					if ((iter % 100) == 0)
					{
						elog(WARNING,
							 "neurondb: logistic_regression: cuBLAS SGEMV gradient failed with status %d, falling back to kernel",
							 cublas_err);
					}
				}
			}

			/*
			 * Fallback to kernel-based gradient computation if cuBLAS failed
			 * or handle is NULL
			 */

			/*
			 * Note: kernel path expects row-major, but d_features is
			 * column-major
			 */
			/* For now, we'll use host-side gradient computation as fallback */
			if (handle == NULL || cublas_err != CUBLAS_STATUS_SUCCESS)
			{
				double	   *host_predictions;
				float	   *h_weights_fallback = NULL;

				NDB_ALLOC(host_predictions, double, n_samples);
				NDB_ALLOC(h_weights_fallback, float, feature_dim);

				/* Copy weights from device for fallback computation */
				status = cudaMemcpy(h_weights_fallback, d_weights, weight_bytes_gpu, cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
				{
					NDB_FREE(host_predictions);
					NDB_FREE(h_weights_fallback);
					goto gpu_fail;
				}

				/* Convert to double for host computation */
				for (i = 0; i < feature_dim; i++)
					weights[i] = (double) h_weights_fallback[i];

				status = cudaMemcpy(host_predictions,
									d_predictions,
									pred_bytes,
									cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
				{
					NDB_FREE(host_predictions);
					NDB_FREE(h_weights_fallback);
					goto gpu_fail;
				}

				if (ndb_cuda_lr_compute_gradients(features,
												  labels,
												  host_predictions,
												  n_samples,
												  feature_dim,
												  grad_weights,
												  &grad_bias) != 0)
				{
					NDB_FREE(host_predictions);
					NDB_FREE(h_weights_fallback);
					if ((iter % 100) == 0)
					{
						elog(WARNING,
							 "neurondb: logistic_regression: compute_gradients failed at iter %d",
							 iter);
					}
					goto gpu_fail;
				}

				NDB_FREE(host_predictions);

				/*
				 * Average gradients and add L2 regularization with defensive
				 * checks
				 */

				/*
				 * Defensive: Check for division by zero (shouldn't happen,
				 * but be safe)
				 */
				if (n_samples > 0)
				{
					grad_bias /= (double) n_samples;
					/* Defensive: Check for NaN/Inf in grad_bias */
					if (!isfinite(grad_bias))
					{
						elog(WARNING,
							 "ndb_cuda_lr_train: non-finite grad_bias at iter %d: %f",
							 iter, grad_bias);
						grad_bias = 0.0;
					}

					for (i = 0; i < feature_dim; i++)
					{
						double		avg_grad = grad_weights[i] / (double) n_samples;
						double		reg_term = lambda * weights[i];
						double		total_grad = avg_grad + reg_term;

						/* Defensive: Check for NaN/Inf in gradient */
						if (isfinite(total_grad))
						{
							/* Clamp gradient to prevent extreme values */
							if (total_grad > 1000.0)
								grad_weights[i] = 1000.0;
							else if (total_grad < -1000.0)
								grad_weights[i] = -1000.0;
							else
								grad_weights[i] = total_grad;
						}
						else
						{
							elog(WARNING,
								 "ndb_cuda_lr_train: non-finite grad_weights[%d] at iter %d: avg=%f, reg=%f",
								 i, iter, avg_grad, reg_term);
							grad_weights[i] = 0.0;
						}
					}

					/* Update weights and bias on host for fallback path */
					{
						double		new_bias = bias - learning_rate * grad_bias;

						if (isfinite(new_bias))
							bias = new_bias;
						else
							bias = (new_bias > 0.0) ? 100.0 : -100.0;

						for (i = 0; i < feature_dim; i++)
						{
							double		new_weight = weights[i] - learning_rate * grad_weights[i];

							if (isfinite(new_weight))
							{
								if (new_weight > 1000.0)
									weights[i] = 1000.0;
								else if (new_weight < -1000.0)
									weights[i] = -1000.0;
								else
									weights[i] = new_weight;
							}
							else
							{
								weights[i] = (new_weight > 0.0) ? 1000.0 : -1000.0;
							}
						}

						/* Copy updated weights back to device */
						for (i = 0; i < feature_dim; i++)
							h_weights_fallback[i] = (float) weights[i];
						status = cudaMemcpy(d_weights, h_weights_fallback, weight_bytes_gpu, cudaMemcpyHostToDevice);

						/* Copy updated bias back to device */
						status = cudaMemcpy(d_bias, &bias, sizeof(double), cudaMemcpyHostToDevice);
					}

					NDB_FREE(h_weights_fallback);
				}
				else
				{
					NDB_FREE(h_weights_fallback);
					elog(ERROR,
						 "ndb_cuda_lr_train: n_samples is zero in gradient computation");
					if (errstr && *errstr == NULL)
						*errstr = pstrdup("ndb_cuda_lr_train: division by zero in gradient computation");
					goto gpu_fail;
				}
			}

	gradient_done:
			;					/* Label must be followed by statement */
		}						/* End of gradient computation block */

		/* Weights and bias are updated on device in the main path */

		/*
		 * Fallback path updates weights on host (handled in fallback section
		 * above)
		 */

		/*
		 * No need to update weights here for main path - already done on
		 * device
		 */
	}

	/* Copy final weights and bias back from device to host */
	{
		NDB_DECLARE(float *, h_weights_final);

		NDB_ALLOC(h_weights_final, float, feature_dim);

		if (h_weights_final != NULL)
		{
			status = cudaMemcpy(h_weights_final, d_weights, weight_bytes_gpu, cudaMemcpyDeviceToHost);
			if (status == cudaSuccess)
			{
				/* Convert back to double for host-side weights array */
				for (i = 0; i < feature_dim; i++)
					weights[i] = (double) h_weights_final[i];
			}
			NDB_FREE(h_weights_final);
		}

		/* Copy final bias from device */
		status = cudaMemcpy(&bias, d_bias, sizeof(double), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess)
		{
			elog(WARNING,
				 "ndb_cuda_lr_train: failed to copy final bias from device: %s",
				 cudaGetErrorString(status));
		}
	}

	/* Free host buffers allocated before the loop */
	NDB_FREE(h_grad_weights_float);

	/* Build model payload */
	{
		size_t		payload_bytes;
		char	   *base;
		NdbCudaLrModelHeader *hdr;
		float	   *weights_dest;
		char	   *payload_raw = NULL;

		payload_bytes = sizeof(NdbCudaLrModelHeader) +
			sizeof(float) * (size_t) feature_dim;
		NDB_ALLOC(payload_raw, char, VARHDRSZ + payload_bytes);
		payload = (bytea *) payload_raw;
		SET_VARSIZE(payload, VARHDRSZ + payload_bytes);
		base = VARDATA(payload);

		hdr = (NdbCudaLrModelHeader *) base;
		hdr->feature_dim = feature_dim;
		hdr->n_samples = n_samples;
		hdr->max_iters = max_iters;
		hdr->learning_rate = learning_rate;
		hdr->lambda = lambda;
		hdr->bias = bias;

		weights_dest = (float *) (base + sizeof(NdbCudaLrModelHeader));
		for (i = 0; i < feature_dim; i++)
			weights_dest[i] = (float) weights[i];
	}

	/* Compute final loss for metrics and build metrics JSON */
	/* Use GPU predictions that were already computed in the last iteration */
	{
		double		final_loss = 0.0;
		double		accuracy = 0.0;
		double	   *host_preds = NULL;
		int			correct = 0;

		/*
		 * Copy final predictions from GPU (they were computed in the last
		 * iteration)
		 */
		NDB_ALLOC(host_preds, double, n_samples);
		status = cudaMemcpy(host_preds,
							d_predictions,
							pred_bytes,
							cudaMemcpyDeviceToHost);
		if (status == cudaSuccess)
		{
			int			idx;

			/* Predictions are already sigmoided from the last iteration */
			for (idx = 0; idx < n_samples; idx++)
			{
				double		pred = fmax(1e-15,
										fmin(1.0 - 1e-15, host_preds[idx]));

				if (labels[idx] > 0.5)
					final_loss -= log(pred);
				else
					final_loss -= log(1.0 - pred);

				if ((pred >= 0.5 && labels[idx] > 0.5) ||
					(pred < 0.5 && labels[idx] <= 0.5))
					correct++;
			}
			final_loss /= (double) n_samples;
			accuracy = (correct > 0 && n_samples > 0) ?
				((double) correct / (double) n_samples) :
				0.0;
		}
		else
		{
			elog(WARNING,
				 "ndb_cuda_lr_train: failed to copy predictions for metrics, using defaults");
			final_loss = 0.0;
			accuracy = 0.0;
		}

		/* Build metrics JSON - always build, even if forward_pass failed */
		{
			StringInfoData buf;

			initStringInfo(&buf);
			appendStringInfo(&buf,
							 "{\"algorithm\":\"logistic_regression\","
							 "\"training_backend\":1,"
							 "\"n_features\":%d,"
							 "\"n_samples\":%d,"
							 "\"max_iters\":%d,"
							 "\"learning_rate\":%.6f,"
							 "\"lambda\":%.6f,"
							 "\"final_loss\":%.6f,"
							 "\"accuracy\":%.6f}",
							 feature_dim,
							 n_samples,
							 max_iters,
							 learning_rate,
							 lambda,
							 final_loss,
							 accuracy);

			metrics_json = DatumGetJsonbP(DirectFunctionCall1(
															  jsonb_in,
															  CStringGetDatum(buf.data)));
			NDB_FREE(buf.data);
			if (metrics_json == NULL)
			{
				elog(ERROR,
					 "ndb_cuda_lr_train: failed to create metrics_json from JSON string");
			}
			elog(DEBUG1,
				 "ndb_cuda_lr_train: created metrics_json: %p",
				 (void *) metrics_json);
		}

		if (host_preds != NULL)
			NDB_FREE(host_preds);
	}

	*model_data = payload;

	/* Always set metrics - build default if metrics_json is NULL */
	if (metrics != NULL)
	{
		if (metrics_json == NULL)
		{
			elog(DEBUG1,
				 "ndb_cuda_lr_train: metrics_json is NULL, building default metrics");
			/* Build default metrics JSON */
			{
				StringInfoData buf;

				initStringInfo(&buf);
				appendStringInfo(&buf,
								 "{\"algorithm\":\"logistic_regression\","
								 "\"storage\":\"gpu\","
								 "\"n_features\":%d,"
								 "\"n_samples\":%d,"
								 "\"max_iters\":%d,"
								 "\"learning_rate\":%.6f,"
								 "\"lambda\":%.6f,"
								 "\"final_loss\":0.0,"
								 "\"accuracy\":0.0}",
								 feature_dim,
								 n_samples,
								 max_iters,
								 learning_rate,
								 lambda);

				PG_TRY();
				{
					metrics_json = DatumGetJsonbP(
												  DirectFunctionCall1(jsonb_in,
																	  CStringGetDatum(buf.data)));
				}
				PG_CATCH();
				{
					/* If JSONB creation fails, set metrics to NULL */
					FlushErrorState();
					metrics_json = NULL;
				}
				PG_END_TRY();

				NDB_FREE(buf.data);
			}
		}
		*metrics = metrics_json;
		elog(DEBUG1,
			 "ndb_cuda_lr_train: setting *metrics = %p (metrics_json=%p)",
			 (void *) *metrics,
			 (void *) metrics_json);
	}
	else
	{
		elog(WARNING,
			 "ndb_cuda_lr_train: metrics output parameter is NULL!");
	}
	rc = 0;

cleanup:
	/* Free host buffers */
	if (h_grad_weights_float)
		NDB_FREE(h_grad_weights_float);
	if (h_errors)
		NDB_FREE(h_errors);
	if (weights)
		NDB_FREE(weights);
	if (grad_weights)
		NDB_FREE(grad_weights);

	/* Free device buffers */
	if (d_features)
		cudaFree(d_features);
	if (d_predictions)
		cudaFree(d_predictions);
	if (d_z)
		cudaFree(d_z);
	if (d_weights)
		cudaFree(d_weights);
	if (d_z_float)
		cudaFree(d_z_float);
	if (d_errors)
		cudaFree(d_errors);
	if (d_grad_weights_float)
		cudaFree(d_grad_weights_float);
	if (d_labels_double)
		cudaFree(d_labels_double);
	if (d_error_sum)
		cudaFree(d_error_sum);
	if (d_bias)
		cudaFree(d_bias);

	return rc;

gpu_fail:
	if (errstr && *errstr == NULL)
		*errstr = pstrdup(cudaGetErrorString(status));
	rc = -1;
	goto cleanup;
}

int
ndb_cuda_lr_predict(const bytea * model_data,
					const float *input,
					int feature_dim,
					double *probability_out,
					char **errstr)
{
	const NdbCudaLrModelHeader *hdr;
	const float *weights;
	const		bytea *detoasted;
	double		z;
	int			i;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || probability_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for CUDA LR predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t		expected_size = sizeof(NdbCudaLrModelHeader) +
			sizeof(float) * (size_t) feature_dim;
		size_t		actual_size = VARSIZE(detoasted) - VARHDRSZ;

		elog(DEBUG1,
			 "ndb_cuda_lr_predict: payload size check: expected=%zu, actual=%zu, feature_dim=%d",
			 expected_size,
			 actual_size,
			 feature_dim);

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr = psprintf("model data too small: expected %zu bytes, got %zu",
								   expected_size, actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaLrModelHeader *) VARDATA(detoasted);
	elog(DEBUG1,
		 "ndb_cuda_lr_predict: header feature_dim=%d, input feature_dim=%d",
		 hdr->feature_dim,
		 feature_dim);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		return -1;
	}

	weights = (const float *) ((const char *) hdr +
							   sizeof(NdbCudaLrModelHeader));

	z = hdr->bias;
	for (i = 0; i < feature_dim; i++)
		z += weights[i] * input[i];

	*probability_out = 1.0 / (1.0 + exp(-z));
	return 0;
}

/*
 * ndb_cuda_lr_evaluate
 *    GPU-accelerated batch evaluation for Logistic Regression
 *
 * Computes predictions for all samples with sigmoid and accumulates confusion matrix
 * and log loss using a CUDA kernel, then computes final metrics (accuracy, precision,
 * recall, F1, log_loss) on the host.
 *
 * Parameters:
 *   model_data: GPU model bytea (contains header + weights)
 *   features: Feature matrix [n_samples, feature_dim] (float, row-major)
 *   labels: Target labels [n_samples] (double, 0.0 or 1.0)
 *   n_samples: Number of samples to evaluate
 *   feature_dim: Number of features per sample
 *   threshold: Classification threshold (typically 0.5)
 *   accuracy_out: Output accuracy
 *   precision_out: Output precision
 *   recall_out: Output recall
 *   f1_out: Output F1 score
 *   log_loss_out: Output log loss
 *   errstr: Error message output (if non-NULL)
 *
 * Returns:
 *   0 on success, -1 on error
 */
int
ndb_cuda_lr_evaluate(const bytea * model_data,
					 const float *features,
					 const double *labels,
					 int n_samples,
					 int feature_dim,
					 double threshold,
					 double *accuracy_out,
					 double *precision_out,
					 double *recall_out,
					 double *f1_out,
					 double *log_loss_out,
					 char **errstr)
{
	const NdbCudaLrModelHeader *hdr;
	const float *weights;
	const		bytea *detoasted;
	cudaError_t cuda_err;
	float	   *d_features = NULL;
	double	   *d_labels = NULL;
	double	   *d_weights = NULL;
	long long  *d_tp = NULL;
	long long  *d_tn = NULL;
	long long  *d_fp = NULL;
	long long  *d_fn = NULL;
	double	   *d_log_loss = NULL;
	long long  *d_count = NULL;
	long long	h_tp = 0;
	long long	h_tn = 0;
	long long	h_fp = 0;
	long long	h_fn = 0;
	double		h_log_loss = 0.0;
	long long	h_count = 0;
	double		accuracy = 0.0;
	double		precision = 0.0;
	double		recall = 0.0;
	double		f1 = 0.0;
	double		log_loss = 0.0;
	size_t		feature_bytes;
	size_t		label_bytes;
	size_t		weight_bytes;
	int			i;

	if (errstr)
		*errstr = NULL;

	/* Defensive parameter validation */
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_cuda_lr_evaluate: model_data is NULL");
		return -1;
	}

	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_cuda_lr_evaluate: features is NULL");
		return -1;
	}

	if (labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_cuda_lr_evaluate: labels is NULL");
		return -1;
	}

	if (n_samples <= 0)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_lr_evaluate: invalid n_samples %d",
							   n_samples);
		return -1;
	}

	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_lr_evaluate: invalid feature_dim %d",
							   feature_dim);
		return -1;
	}

	if (accuracy_out == NULL || precision_out == NULL || recall_out == NULL ||
		f1_out == NULL || log_loss_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("neurondb: ndb_cuda_lr_evaluate: output pointers are NULL");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted = (const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t		expected_size = sizeof(NdbCudaLrModelHeader)
			+ sizeof(float) * (size_t) feature_dim;
		size_t		actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < expected_size)
		{
			if (errstr)
				*errstr = psprintf("neurondb: ndb_cuda_lr_evaluate: model data too small: "
								   "expected %zu bytes, got %zu",
								   expected_size,
								   actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaLrModelHeader *) VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_lr_evaluate: feature dimension mismatch: "
							   "model has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		return -1;
	}

	weights = (const float *) ((const char *) hdr + sizeof(NdbCudaLrModelHeader));

	/* Allocate GPU memory for features */
	feature_bytes = sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
	cuda_err = cudaMalloc((void **) &d_features, feature_bytes);
	if (cuda_err != cudaSuccess)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_lr_evaluate: failed to allocate GPU memory for features: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Allocate GPU memory for labels */
	label_bytes = sizeof(double) * (size_t) n_samples;
	cuda_err = cudaMalloc((void **) &d_labels, label_bytes);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_lr_evaluate: failed to allocate GPU memory for labels: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Allocate GPU memory for weights */
	weight_bytes = sizeof(double) * (size_t) feature_dim;
	cuda_err = cudaMalloc((void **) &d_weights, weight_bytes);
	if (cuda_err != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_labels);
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_lr_evaluate: failed to allocate GPU memory for weights: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	/* Allocate GPU memory for output accumulators */
	cuda_err = cudaMalloc((void **) &d_tp, sizeof(long long));
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMalloc((void **) &d_tn, sizeof(long long));
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMalloc((void **) &d_fp, sizeof(long long));
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMalloc((void **) &d_fn, sizeof(long long));
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMalloc((void **) &d_log_loss, sizeof(double));
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMalloc((void **) &d_count, sizeof(long long));
	if (cuda_err != cudaSuccess)
		goto cleanup;

	/* Initialize accumulators to zero */
	cuda_err = cudaMemset(d_tp, 0, sizeof(long long));
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMemset(d_tn, 0, sizeof(long long));
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMemset(d_fp, 0, sizeof(long long));
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMemset(d_fn, 0, sizeof(long long));
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMemset(d_log_loss, 0, sizeof(double));
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMemset(d_count, 0, sizeof(long long));
	if (cuda_err != cudaSuccess)
		goto cleanup;

	/* Copy features to GPU */
	cuda_err = cudaMemcpy(d_features, features, feature_bytes, cudaMemcpyHostToDevice);
	if (cuda_err != cudaSuccess)
		goto cleanup;

	/* Copy labels to GPU */
	cuda_err = cudaMemcpy(d_labels, labels, label_bytes, cudaMemcpyHostToDevice);
	if (cuda_err != cudaSuccess)
		goto cleanup;

	/* Convert weights from float to double and copy to GPU */
	{
		NDB_DECLARE(double *, h_weights_double);

		NDB_ALLOC(h_weights_double, double, feature_dim);

		if (h_weights_double == NULL)
		{
			if (errstr)
				*errstr = pstrdup("neurondb: ndb_cuda_lr_evaluate: failed to allocate host memory for weights");
			cuda_err = cudaErrorMemoryAllocation;
			goto cleanup;
		}

		for (i = 0; i < feature_dim; i++)
			h_weights_double[i] = (double) weights[i];

		cuda_err = cudaMemcpy(d_weights, h_weights_double, weight_bytes, cudaMemcpyHostToDevice);
		NDB_FREE(h_weights_double);

		if (cuda_err != cudaSuccess)
			goto cleanup;
	}

	/* Launch evaluation kernel */
	cuda_err = launch_lr_eval_kernel(d_features,
									 d_labels,
									 d_weights,
									 hdr->bias,
									 threshold,
									 n_samples,
									 feature_dim,
									 d_tp,
									 d_tn,
									 d_fp,
									 d_fn,
									 d_log_loss,
									 d_count);

	if (cuda_err != cudaSuccess)
		goto cleanup;

	/* Copy results back to host */
	cuda_err = cudaMemcpy(&h_tp, d_tp, sizeof(long long), cudaMemcpyDeviceToHost);
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMemcpy(&h_tn, d_tn, sizeof(long long), cudaMemcpyDeviceToHost);
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMemcpy(&h_fp, d_fp, sizeof(long long), cudaMemcpyDeviceToHost);
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMemcpy(&h_fn, d_fn, sizeof(long long), cudaMemcpyDeviceToHost);
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMemcpy(&h_log_loss, d_log_loss, sizeof(double), cudaMemcpyDeviceToHost);
	if (cuda_err != cudaSuccess)
		goto cleanup;
	cuda_err = cudaMemcpy(&h_count, d_count, sizeof(long long), cudaMemcpyDeviceToHost);
	if (cuda_err != cudaSuccess)
		goto cleanup;

	/* Defensive check: ensure count matches expected */
	if (h_count != (long long) n_samples)
	{
		if (errstr)
			*errstr = psprintf("neurondb: ndb_cuda_lr_evaluate: count mismatch: expected %d, got %lld",
							   n_samples,
							   (long long) h_count);
		cuda_err = cudaErrorInvalidValue;
		goto cleanup;
	}

	/* Compute final metrics */
	if (h_count > 0)
	{
		long long	total = h_tp + h_tn + h_fp + h_fn;

		if (total > 0)
		{
			accuracy = (double) (h_tp + h_tn) / (double) total;
		}
		if (h_tp + h_fp > 0)
		{
			precision = (double) h_tp / (double) (h_tp + h_fp);
		}
		if (h_tp + h_fn > 0)
		{
			recall = (double) h_tp / (double) (h_tp + h_fn);
		}
		if (precision + recall > 0.0)
		{
			f1 = 2.0 * precision * recall / (precision + recall);
		}
		log_loss = h_log_loss / (double) h_count;
	}

	/* Write outputs */
	*accuracy_out = accuracy;
	*precision_out = precision;
	*recall_out = recall;
	*f1_out = f1;
	*log_loss_out = log_loss;

	/* Success - set cuda_err to success before cleanup */
	cuda_err = cudaSuccess;

cleanup:
	/* Single cleanup block that frees everything that is non-NULL */
	if (d_features)
		cudaFree(d_features);
	if (d_labels)
		cudaFree(d_labels);
	if (d_weights)
		cudaFree(d_weights);
	if (d_tp)
		cudaFree(d_tp);
	if (d_tn)
		cudaFree(d_tn);
	if (d_fp)
		cudaFree(d_fp);
	if (d_fn)
		cudaFree(d_fn);
	if (d_log_loss)
		cudaFree(d_log_loss);
	if (d_count)
		cudaFree(d_count);

	if (cuda_err != cudaSuccess)
	{
		if (errstr && *errstr == NULL)
			*errstr = psprintf("neurondb: ndb_cuda_lr_evaluate: CUDA operation failed: %s",
							   cudaGetErrorString(cuda_err));
		return -1;
	}

	return 0;
}

#else							/* !NDB_GPU_CUDA */

int
ndb_cuda_lr_pack_model(const LRModel *model,
					   bytea * *model_data,
					   Jsonb * *metrics,
					   char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA support not compiled");
	(void) model;
	(void) model_data;
	(void) metrics;
	return -1;
}

int
ndb_cuda_lr_train(const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  const Jsonb * hyperparams,
				  bytea * *model_data,
				  Jsonb * *metrics,
				  char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA support not compiled");
	(void) features;
	(void) labels;
	(void) n_samples;
	(void) feature_dim;
	(void) hyperparams;
	(void) model_data;
	(void) metrics;
	return -1;
}

int
ndb_cuda_lr_predict(const bytea * model_data,
					const float *input,
					int feature_dim,
					double *probability_out,
					char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA support not compiled");
	(void) model_data;
	(void) input;
	(void) feature_dim;
	(void) probability_out;
	return -1;
}

int
ndb_cuda_lr_evaluate(const bytea * model_data,
					 const float *features,
					 const double *labels,
					 int n_samples,
					 int feature_dim,
					 double threshold,
					 double *accuracy_out,
					 double *precision_out,
					 double *recall_out,
					 double *f1_out,
					 double *log_loss_out,
					 char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA support not compiled");
	(void) model_data;
	(void) features;
	(void) labels;
	(void) n_samples;
	(void) feature_dim;
	(void) threshold;
	(void) accuracy_out;
	(void) precision_out;
	(void) recall_out;
	(void) f1_out;
	(void) log_loss_out;
	return -1;
}

#endif							/* NDB_GPU_CUDA */
