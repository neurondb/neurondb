/*-------------------------------------------------------------------------
 *
 * gpu_gmm_cuda.c
 *    CUDA backend bridge for Gaussian Mixture Model training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_gmm_cuda.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_CUDA

#include <float.h>
#include <math.h>
#include <string.h>

#include "neurondb_cuda_runtime.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/palloc.h"
#include "utils/memutils.h"
#include "utils/elog.h"

#include "neurondb_cuda_gmm.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GMM_EPSILON 1e-6
#define GMM_MIN_PROB 1e-10

/* GMM model structure (matches ml_gmm.c) */
typedef struct GMMModel
{
	int			k;				/* Number of components */
	int			dim;			/* Dimensionality */
	double	   *mixing_coeffs;	/* π_k: mixing coefficients [k] */
	double	  **means;			/* μ_k: component means [k][dim] */
	double	  **variances;		/* Σ_k: diagonal variances [k][dim] */
}			GMMModel;

int
ndb_cuda_gmm_pack_model(const struct GMMModel *model,
						bytea * *model_data,
						Jsonb * *metrics,
						char **errstr)
{
	size_t		payload_bytes;
	size_t		mixing_bytes;
	size_t		means_bytes;
	size_t		variances_bytes;
	bytea	   *blob;
	char	   *base;
	NdbCudaGmmModelHeader *hdr;
	double	   *mixing_dest;
	double	   *means_dest;
	double	   *variances_dest;
	int			i,
				j;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid GMM model for CUDA pack: model or model_data is NULL");
		return -1;
	}

	/* Validate model structure */
	if (model->k <= 0 || model->k > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("invalid GMM model: k (n_components) must be between 1 and 1000000");
		return -1;
	}
	if (model->dim <= 0 || model->dim > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("invalid GMM model: dim (n_features) must be between 1 and 1000000");
		return -1;
	}

	/* Check for integer overflow in size calculations */
	mixing_bytes = sizeof(double) * (size_t) model->k;
	means_bytes = sizeof(double) * (size_t) model->k * (size_t) model->dim;
	variances_bytes = sizeof(double) * (size_t) model->k * (size_t) model->dim;

	if (means_bytes / sizeof(double) / (size_t) model->k != (size_t) model->dim ||
		variances_bytes / sizeof(double) / (size_t) model->k != (size_t) model->dim)
	{
		if (errstr)
			*errstr = pstrdup("invalid GMM model: integer overflow in size calculation");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaGmmModelHeader) + mixing_bytes + means_bytes + variances_bytes;

	/* Check for overflow in total payload */
	if (payload_bytes < sizeof(NdbCudaGmmModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("invalid GMM model: payload size underflow");
		return -1;
	}
	if (payload_bytes > (MaxAllocSize - VARHDRSZ))
	{
		if (errstr)
			*errstr = pstrdup("invalid GMM model: payload size exceeds MaxAllocSize");
		return -1;
	}

	blob = (bytea *) palloc(VARHDRSZ + payload_bytes);
	if (blob == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM pack: memory allocation failed");
		return -1;
	}

	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaGmmModelHeader *) base;
	hdr->n_components = model->k;
	hdr->n_features = model->dim;
	hdr->n_samples = 0;			/* Not stored in model */
	hdr->max_iters = 100;		/* Default */
	hdr->tolerance = 1e-6;

	mixing_dest = (double *) (base + sizeof(NdbCudaGmmModelHeader));
	means_dest = (double *) (base + sizeof(NdbCudaGmmModelHeader) + sizeof(double) * (size_t) model->k);
	variances_dest = (double *) (base + sizeof(NdbCudaGmmModelHeader) + sizeof(double) * (size_t) model->k + sizeof(double) * (size_t) model->k * (size_t) model->dim);

	if (model->mixing_coeffs != NULL)
	{
		double		sum = 0.0;

		for (i = 0; i < model->k; i++)
		{
			double		coeff = model->mixing_coeffs[i];

			/* Validate mixing coefficient: must be finite and non-negative */
			if (!isfinite(coeff) || coeff < 0.0)
			{
				if (errstr)
					*errstr = pstrdup("invalid GMM model: mixing_coeffs contains invalid value");
				NDB_FREE(blob);
				return -1;
			}
			mixing_dest[i] = coeff;
			sum += coeff;
		}
		/* Validate that mixing coefficients sum to approximately 1.0 */
		if (fabs(sum - 1.0) > 0.1)
		{
			if (errstr)
				*errstr = pstrdup("invalid GMM model: mixing_coeffs do not sum to 1.0");
			NDB_FREE(blob);
			return -1;
		}
	}
	else
	{
		/* Initialize with equal mixing coefficients if NULL */
		double		default_coeff = 1.0 / model->k;

		for (i = 0; i < model->k; i++)
			mixing_dest[i] = default_coeff;
	}

	if (model->means != NULL)
	{
		for (i = 0; i < model->k; i++)
		{
			if (model->means[i] != NULL)
			{
				for (j = 0; j < model->dim; j++)
				{
					double		mean_val = model->means[i][j];

					/* Validate mean: must be finite */
					if (!isfinite(mean_val))
					{
						if (errstr)
							*errstr = pstrdup("invalid GMM model: means contains non-finite value");
						NDB_FREE(blob);
						return -1;
					}
					means_dest[i * model->dim + j] = mean_val;
				}
			}
			else
			{
				/* Initialize with zeros if NULL */
				for (j = 0; j < model->dim; j++)
					means_dest[i * model->dim + j] = 0.0;
			}
		}
	}
	else
	{
		/* Initialize all means to zero if NULL */
		memset(means_dest, 0, means_bytes);
	}

	if (model->variances != NULL)
	{
		for (i = 0; i < model->k; i++)
		{
			if (model->variances[i] != NULL)
			{
				for (j = 0; j < model->dim; j++)
				{
					double		var_val = model->variances[i][j];

					/* Validate variance: must be finite and positive */
					if (!isfinite(var_val) || var_val < 0.0)
					{
						if (errstr)
							*errstr = pstrdup("invalid GMM model: variances contains invalid value");
						NDB_FREE(blob);
						return -1;
					}
					/* Regularize to avoid division by zero */
					if (var_val < GMM_EPSILON)
						var_val = GMM_EPSILON;
					variances_dest[i * model->dim + j] = var_val;
				}
			}
			else
			{
				/* Initialize with default regularization if NULL */
				for (j = 0; j < model->dim; j++)
					variances_dest[i * model->dim + j] = GMM_EPSILON;
			}
		}
	}
	else
	{
		/* Initialize all variances to regularization value if NULL */
		for (i = 0; i < model->k * model->dim; i++)
			variances_dest[i] = GMM_EPSILON;
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb	   *metrics_json = NULL;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"gmm\","
						 "\"storage\":\"gpu\","
						 "\"n_components\":%d,"
						 "\"n_features\":%d}",
						 model->k,
						 model->dim);

		PG_TRY();
		{
			metrics_json = DatumGetJsonbP(
										  DirectFunctionCall1(jsonb_in, CStringGetTextDatum(buf.data)));
		}
		PG_CATCH();
		{
			/* If JSONB creation fails, set metrics to NULL */
			FlushErrorState();
			metrics_json = NULL;
		}
		PG_END_TRY();

		NDB_FREE(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_cuda_gmm_train(const float *features,
				   int n_samples,
				   int feature_dim,
				   int n_components,
				   const Jsonb * hyperparams,
				   bytea * *model_data,
				   Jsonb * *metrics,
				   char **errstr)
{
	int			max_iters = 100;
	double		tolerance = 1e-6;
	double	   *mixing_coeffs = NULL;
	double	   *means = NULL;
	double	   *variances = NULL;
	double	  **means_2d = NULL;
	double	  **variances_2d = NULL;
	double	   *responsibilities = NULL;
	double		log_likelihood = 0.0;
	double		prev_log_likelihood = -DBL_MAX;
	struct GMMModel model;
	bytea	   *blob = NULL;
	Jsonb	   *metrics_json = NULL;
	int			iter;
	int			i,
				k,
				d;
	int			rc = -1;

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: features array is NULL");
		return -1;
	}
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: model_data output pointer is NULL");
		return -1;
	}
	if (n_samples <= 0 || n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: n_samples must be between 1 and 100000000");
		return -1;
	}
	if (feature_dim <= 0 || feature_dim > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: feature_dim must be between 1 and 1000000");
		return -1;
	}
	if (n_components <= 0 || n_components > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: n_components must be between 1 and 1000000");
		return -1;
	}
	if (n_components > n_samples)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: n_components cannot exceed n_samples");
		return -1;
	}

	/* Check for integer overflow in memory allocation */
	if (feature_dim > 0 && (size_t) n_samples > MaxAllocSize / sizeof(float) / (size_t) feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: feature array size exceeds MaxAllocSize");
		return -1;
	}
	if (feature_dim > 0 && (size_t) n_components > MaxAllocSize / sizeof(double) / (size_t) feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: means array size exceeds MaxAllocSize");
		return -1;
	}
	if ((size_t) n_samples > MaxAllocSize / sizeof(double) / (size_t) n_components)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: responsibilities array size exceeds MaxAllocSize");
		return -1;
	}

	elog(DEBUG1, "ndb_cuda_gmm_train: entry: n_samples=%d, feature_dim=%d, n_components=%d",
		 n_samples, feature_dim, n_components);

	/* Validate input data for NaN/Inf before processing */
	for (i = 0; i < n_samples; i++)
	{
		for (d = 0; d < feature_dim; d++)
		{
			if (!isfinite(features[i * feature_dim + d]))
			{
				if (errstr)
					*errstr = pstrdup("CUDA GMM train: non-finite value in features array");
				return -1;
			}
		}
	}

	/* Extract hyperparameters from JSONB */
	if (hyperparams != NULL)
	{
		Datum		max_iters_datum;
		Datum		tolerance_datum;
		Datum		numeric_datum;
		Numeric		num;

		max_iters_datum = DirectFunctionCall2(
											  jsonb_object_field,
											  JsonbPGetDatum(hyperparams),
											  CStringGetTextDatum("max_iters"));
		if (DatumGetPointer(max_iters_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
												jsonb_numeric, max_iters_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				max_iters = DatumGetInt32(
										  DirectFunctionCall1(numeric_int4,
															  NumericGetDatum(num)));
				if (max_iters <= 0)
					max_iters = 100;
			}
		}

		tolerance_datum = DirectFunctionCall2(
											  jsonb_object_field,
											  JsonbPGetDatum(hyperparams),
											  CStringGetTextDatum("tolerance"));
		if (DatumGetPointer(tolerance_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
												jsonb_numeric, tolerance_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				tolerance = DatumGetFloat8(
										   DirectFunctionCall1(numeric_float8,
															   NumericGetDatum(num)));
				if (tolerance <= 0.0)
					tolerance = 1e-6;
			}
		}
	}

	/* Allocate host memory with overflow checks */
	mixing_coeffs = (double *) palloc(sizeof(double) * n_components);
	if (mixing_coeffs == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: failed to allocate mixing_coeffs array");
		return -1;
	}

	if (feature_dim > 0 && (size_t) n_components > MaxAllocSize / sizeof(double) / (size_t) feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: means array size exceeds MaxAllocSize");
		goto cleanup;
	}
	means = (double *) palloc0(sizeof(double) * (size_t) n_components * (size_t) feature_dim);
	if (means == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: failed to allocate means array");
		goto cleanup;
	}

	variances = (double *) palloc0(sizeof(double) * (size_t) n_components * (size_t) feature_dim);
	if (variances == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: failed to allocate variances array");
		goto cleanup;
	}

	means_2d = (double **) palloc(sizeof(double *) * n_components);
	if (means_2d == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: failed to allocate means_2d array");
		goto cleanup;
	}

	variances_2d = (double **) palloc(sizeof(double *) * n_components);
	if (variances_2d == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: failed to allocate variances_2d array");
		goto cleanup;
	}

	if ((size_t) n_samples > MaxAllocSize / sizeof(double) / (size_t) n_components)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: responsibilities array size exceeds MaxAllocSize");
		goto cleanup;
	}
	responsibilities = (double *) palloc(sizeof(double) * (size_t) n_samples * (size_t) n_components);
	if (responsibilities == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: failed to allocate responsibilities array");
		goto cleanup;
	}

	/* Initialize means with random data points (K-means++ style) */
	if (n_components <= 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM train: n_components is zero (division by zero)");
		goto cleanup;
	}
	for (k = 0; k < n_components; k++)
	{
		int			idx = (k * n_samples) / n_components;	/* Spread initial means */

		if (idx < 0 || idx >= n_samples)
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM train: invalid index for initial mean");
			goto cleanup;
		}
		means_2d[k] = means + k * feature_dim;
		variances_2d[k] = variances + k * feature_dim;

		for (d = 0; d < feature_dim; d++)
		{
			double		val = (double) features[idx * feature_dim + d];

			if (!isfinite(val))
			{
				if (errstr)
					*errstr = pstrdup("CUDA GMM train: non-finite value in initial mean");
				goto cleanup;
			}
			means[k * feature_dim + d] = val;
		}

		/* Initialize variances to 1.0 */
		for (d = 0; d < feature_dim; d++)
			variances[k * feature_dim + d] = 1.0;

		/* Equal mixing coefficients initially */
		mixing_coeffs[k] = 1.0 / (double) n_components;
		if (!isfinite(mixing_coeffs[k]))
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM train: computed non-finite mixing coefficient");
			goto cleanup;
		}
	}

	/* EM algorithm */
	for (iter = 0; iter < max_iters; iter++)
	{
		/* E-step: Compute responsibilities using CUDA */
		if (ndb_cuda_gmm_estep(features, mixing_coeffs, means, variances, n_samples, feature_dim, n_components, responsibilities) != 0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM E-step failed");
			goto cleanup;
		}

		/* Compute log-likelihood for convergence check */
		log_likelihood = 0.0;
		if (n_samples <= 0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM train: n_samples is zero (division by zero)");
			goto cleanup;
		}
		for (i = 0; i < n_samples; i++)
		{
			double		sum = 0.0;

			for (k = 0; k < n_components; k++)
			{
				double		resp = responsibilities[i * n_components + k];

				if (!isfinite(resp))
				{
					if (errstr)
						*errstr = pstrdup("CUDA GMM train: non-finite responsibility value");
					goto cleanup;
				}
				sum += resp;
			}
			if (sum > GMM_MIN_PROB)
			{
				double		log_sum = log(sum);

				if (!isfinite(log_sum))
				{
					if (errstr)
						*errstr = pstrdup("CUDA GMM train: non-finite log-likelihood");
					goto cleanup;
				}
				log_likelihood += log_sum;
			}
		}
		log_likelihood /= (double) n_samples;
		if (!isfinite(log_likelihood))
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM train: computed non-finite log-likelihood");
			goto cleanup;
		}

		/* Check convergence */
		if (fabs(log_likelihood - prev_log_likelihood) < tolerance)
		{
			break;
		}
		prev_log_likelihood = log_likelihood;

		/* M-step: Update parameters using CUDA */
		if (ndb_cuda_gmm_mstep(features, responsibilities, n_samples, feature_dim, n_components, mixing_coeffs, means, variances) != 0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM M-step failed");
			goto cleanup;
		}

		/* Regularize variances */
		for (k = 0; k < n_components; k++)
		{
			for (d = 0; d < feature_dim; d++)
			{
				if (variances[k * feature_dim + d] < GMM_EPSILON)
					variances[k * feature_dim + d] = GMM_EPSILON;
			}
		}
	}

	/* Build model structure for packing */
	model.k = n_components;
	model.dim = feature_dim;
	model.mixing_coeffs = mixing_coeffs;
	model.means = means_2d;
	model.variances = variances_2d;

	/*
	 * Pack model - pass NULL for metrics if caller doesn't want them to avoid
	 * DirectFunctionCall issues
	 */
	if (ndb_cuda_gmm_pack_model(&model, &blob, metrics ? &metrics_json : NULL, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA GMM model packing failed");
		goto cleanup;
	}

	*model_data = blob;
	if (metrics != NULL)
		*metrics = metrics_json;

	rc = 0;

cleanup:
	/* Free model structure arrays (not the data they point to) */
	if (means_2d != NULL)
		NDB_FREE(means_2d);
	if (variances_2d != NULL)
		NDB_FREE(variances_2d);

	/* Free host memory */
	if (mixing_coeffs != NULL)
		NDB_FREE(mixing_coeffs);
	if (means != NULL)
		NDB_FREE(means);
	if (variances != NULL)
		NDB_FREE(variances);
	if (responsibilities != NULL)
		NDB_FREE(responsibilities);

	return rc;
}

int
ndb_cuda_gmm_predict(const bytea * model_data,
					 const float *input,
					 int feature_dim,
					 int *cluster_out,
					 double *probability_out,
					 char **errstr)
{
	const char *base;
	NdbCudaGmmModelHeader *hdr;
	const double *mixing;
	const double *means;
	const double *variances;
	double	   *component_probs;
	double		max_prob;
	int			best_component;
	int			i,
				j;
	size_t		expected_size;
	double		log_likelihood;
	double		log_det;
	double		diff;
	double		var;
	double		log_var;
	double		log_lik_contrib;
	double		exp_val;
	double		final_prob;

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: model_data is NULL");
		return -1;
	}
	if (input == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: input array is NULL");
		return -1;
	}
	if (cluster_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: cluster_out pointer is NULL");
		return -1;
	}
	if (probability_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: probability_out pointer is NULL");
		return -1;
	}

	/* Validate model_data bytea */
	if (VARSIZE_ANY_EXHDR(model_data) < sizeof(NdbCudaGmmModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: model_data too small (corrupted or invalid)");
		return -1;
	}

	base = VARDATA_ANY(model_data);
	hdr = (NdbCudaGmmModelHeader *) base;

	/* Validate model header */
	if (hdr->n_components <= 0 || hdr->n_components > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: invalid n_components in model header");
		return -1;
	}
	if (hdr->n_features <= 0 || hdr->n_features > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: invalid n_features in model header");
		return -1;
	}
	if (feature_dim != hdr->n_features)
	{
		if (errstr)
		{
			char	   *msg = psprintf("CUDA GMM predict: feature dimension mismatch (expected %d, got %d)", hdr->n_features, feature_dim);

			*errstr = pstrdup(msg);
			NDB_FREE(msg);
		}
		return -1;
	}

	/* Validate bytea size matches expected payload */
	expected_size = sizeof(NdbCudaGmmModelHeader)
		+ sizeof(double) * (size_t) hdr->n_components
		+ sizeof(double) * (size_t) hdr->n_components * (size_t) hdr->n_features
		+ sizeof(double) * (size_t) hdr->n_components * (size_t) hdr->n_features;
	if (VARSIZE_ANY_EXHDR(model_data) < expected_size)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: model_data size mismatch (corrupted model)");
		return -1;
	}

	/* Validate input array */
	for (i = 0; i < feature_dim; i++)
	{
		if (!isfinite(input[i]))
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM predict: non-finite value in input array");
			return -1;
		}
	}

	mixing = (const double *) (base + sizeof(NdbCudaGmmModelHeader));
	means = (const double *) (base + sizeof(NdbCudaGmmModelHeader) + sizeof(double) * (size_t) hdr->n_components);
	variances = (const double *) (base + sizeof(NdbCudaGmmModelHeader) + sizeof(double) * (size_t) hdr->n_components + sizeof(double) * (size_t) hdr->n_components * (size_t) hdr->n_features);

	/* Validate model data pointers */
	if (mixing == NULL || means == NULL || variances == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: model data pointers are NULL (corrupted model)");
		return -1;
	}

	component_probs = (double *) palloc(sizeof(double) * hdr->n_components);
	if (component_probs == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: failed to allocate component_probs array");
		return -1;
	}

	/* Compute probability for each component */
	for (i = 0; i < hdr->n_components; i++)
	{
		double		mix_coeff = mixing[i];

		/* Validate mixing coefficient */
		if (!isfinite(mix_coeff) || mix_coeff < 0.0 || mix_coeff > 1.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM predict: invalid mixing coefficient in model");
			NDB_FREE(component_probs);
			return -1;
		}

		log_likelihood = 0.0;
		log_det = 0.0;

		for (j = 0; j < feature_dim; j++)
		{
			double		mean_val = means[i * feature_dim + j];
			double		var_val = variances[i * feature_dim + j];

			/* Validate model parameters */
			if (!isfinite(mean_val) || !isfinite(var_val) || var_val < 0.0)
			{
				if (errstr)
					*errstr = pstrdup("CUDA GMM predict: invalid mean or variance in model");
				NDB_FREE(component_probs);
				return -1;
			}

			diff = (double) input[j] - mean_val;
			var = var_val + GMM_EPSILON;

			/* Check for division by zero */
			if (var <= 0.0)
			{
				if (errstr)
					*errstr = pstrdup("CUDA GMM predict: variance is zero or negative");
				NDB_FREE(component_probs);
				return -1;
			}

			log_var = log(var);
			if (!isfinite(log_var))
			{
				if (errstr)
					*errstr = pstrdup("CUDA GMM predict: computed non-finite log variance");
				NDB_FREE(component_probs);
				return -1;
			}

			log_lik_contrib = -0.5 * (diff * diff) / var;
			if (!isfinite(log_lik_contrib))
			{
				if (errstr)
					*errstr = pstrdup("CUDA GMM predict: computed non-finite log-likelihood contribution");
				NDB_FREE(component_probs);
				return -1;
			}

			log_likelihood += log_lik_contrib;
			log_det += log_var;
		}

		log_likelihood -= 0.5 * (feature_dim * log(2.0 * M_PI) + log_det);
		if (!isfinite(log_likelihood))
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM predict: computed non-finite log-likelihood");
			NDB_FREE(component_probs);
			return -1;
		}

		exp_val = exp(log_likelihood);
		if (!isfinite(exp_val) || exp_val < 0.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM predict: computed invalid exp value");
			NDB_FREE(component_probs);
			return -1;
		}

		component_probs[i] = mix_coeff * exp_val;
		if (!isfinite(component_probs[i]) || component_probs[i] < 0.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM predict: computed invalid component probability");
			NDB_FREE(component_probs);
			return -1;
		}
	}

	/* Find component with maximum probability */
	max_prob = component_probs[0];
	best_component = 0;
	if (!isfinite(max_prob))
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: first component probability is non-finite");
		NDB_FREE(component_probs);
		return -1;
	}

	for (i = 1; i < hdr->n_components; i++)
	{
		if (!isfinite(component_probs[i]))
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM predict: component probability is non-finite");
			NDB_FREE(component_probs);
			return -1;
		}
		if (component_probs[i] > max_prob)
		{
			max_prob = component_probs[i];
			best_component = i;
		}
	}

	/* Normalize probabilities */
	{
		double		sum = 0.0;
		int			k;

		for (k = 0; k < hdr->n_components; k++)
		{
			if (!isfinite(component_probs[k]))
			{
				if (errstr)
					*errstr = pstrdup("CUDA GMM predict: component probability is non-finite during normalization");
				NDB_FREE(component_probs);
				return -1;
			}
			sum += component_probs[k];
		}

		if (!isfinite(sum))
		{
			if (errstr)
				*errstr = pstrdup("CUDA GMM predict: sum of probabilities is non-finite");
			NDB_FREE(component_probs);
			return -1;
		}

		if (sum > 0.0)
		{
			/* Check for division by zero */
			if (sum <= 0.0 || !isfinite(sum))
			{
				if (errstr)
					*errstr = pstrdup("CUDA GMM predict: sum of probabilities is zero or non-finite");
				NDB_FREE(component_probs);
				return -1;
			}

			for (k = 0; k < hdr->n_components; k++)
			{
				component_probs[k] /= sum;
				if (!isfinite(component_probs[k]) || component_probs[k] < 0.0 || component_probs[k] > 1.0)
				{
					if (errstr)
						*errstr = pstrdup("CUDA GMM predict: computed invalid normalized probability");
					NDB_FREE(component_probs);
					return -1;
				}
			}
		}
		else
		{
			/* All probabilities are zero - use uniform distribution */
			double		uniform_prob = 1.0 / (double) hdr->n_components;

			for (k = 0; k < hdr->n_components; k++)
				component_probs[k] = uniform_prob;
			best_component = 0;
		}
	}

	/* Validate final outputs */
	if (best_component < 0 || best_component >= hdr->n_components)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: invalid best_component index");
		NDB_FREE(component_probs);
		return -1;
	}

	final_prob = component_probs[best_component];
	if (!isfinite(final_prob) || final_prob < 0.0 || final_prob > 1.0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA GMM predict: computed invalid final probability");
		NDB_FREE(component_probs);
		return -1;
	}

	*cluster_out = best_component;
	*probability_out = final_prob;
	NDB_FREE(component_probs);

	return 0;
}

#else

void
ndb_cuda_gmm_train(void)
{
}
void
ndb_cuda_gmm_predict(void)
{
}
void
ndb_cuda_gmm_pack_model(void)
{
}

#endif							/* NDB_GPU_CUDA */
