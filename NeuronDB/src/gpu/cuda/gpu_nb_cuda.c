/*-------------------------------------------------------------------------
 *
 * gpu_nb_cuda.c
 *    CUDA backend bridge for Naive Bayes training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_nb_cuda.c
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

#include "neurondb_cuda_nb.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Gaussian Naive Bayes model structure (matches ml_naive_bayes.c) */
typedef struct GaussianNBModel
{
	double	   *class_priors;	/* P(class) */
	double	  **means;			/* Mean for each feature per class */
	double	  **variances;		/* Variance for each feature per class */
	int			n_classes;
	int			n_features;
}			GaussianNBModel;

int
ndb_cuda_nb_pack_model(const GaussianNBModel * model,
					   bytea * *model_data,
					   Jsonb * *metrics,
					   char **errstr)
{
	size_t		payload_bytes;
	size_t		priors_bytes;
	size_t		means_bytes;
	size_t		variances_bytes;
	bytea	   *blob;
	char	   *base;
	NdbCudaNbModelHeader *hdr;
	double	   *priors_dest;
	double	   *means_dest;
	double	   *variances_dest;
	int			i,
				j;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid NB model for CUDA pack: model or model_data is NULL");
		return -1;
	}

	/* Validate model structure */
	if (model->n_classes <= 0 || model->n_classes > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("invalid NB model: n_classes must be between 1 and 1000000");
		return -1;
	}
	if (model->n_features <= 0 || model->n_features > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("invalid NB model: n_features must be between 1 and 1000000");
		return -1;
	}

	/* Check for integer overflow in size calculations */
	priors_bytes = sizeof(double) * (size_t) model->n_classes;
	means_bytes = sizeof(double) * (size_t) model->n_classes * (size_t) model->n_features;
	variances_bytes = sizeof(double) * (size_t) model->n_classes * (size_t) model->n_features;

	if (means_bytes / sizeof(double) / (size_t) model->n_classes != (size_t) model->n_features ||
		variances_bytes / sizeof(double) / (size_t) model->n_classes != (size_t) model->n_features)
	{
		if (errstr)
			*errstr = pstrdup("invalid NB model: integer overflow in size calculation");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaNbModelHeader) + priors_bytes + means_bytes + variances_bytes;

	/* Check for overflow in total payload (use division check) */
	if (payload_bytes < sizeof(NdbCudaNbModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("invalid NB model: payload size underflow");
		return -1;
	}
	/* Check if payload_bytes would overflow when adding VARHDRSZ */
	if (payload_bytes > (MaxAllocSize - VARHDRSZ))
	{
		if (errstr)
			*errstr = pstrdup("invalid NB model: payload size exceeds MaxAllocSize");
		return -1;
	}

	/* Note: palloc never returns NULL in PostgreSQL - it errors on failure */
	blob = (bytea *) palloc(VARHDRSZ + payload_bytes);

	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaNbModelHeader *) base;
	hdr->n_classes = model->n_classes;
	hdr->n_features = model->n_features;
	hdr->n_samples = 0;			/* Not stored in model */

	priors_dest = (double *) (base + sizeof(NdbCudaNbModelHeader));
	means_dest = (double *) (base + sizeof(NdbCudaNbModelHeader) + sizeof(double) * (size_t) model->n_classes);
	variances_dest = (double *) (base + sizeof(NdbCudaNbModelHeader) + sizeof(double) * (size_t) model->n_classes + sizeof(double) * (size_t) model->n_classes * (size_t) model->n_features);

	if (model->class_priors != NULL)
	{
		for (i = 0; i < model->n_classes; i++)
		{
			double		prior = model->class_priors[i];

			/* Validate prior: must be finite and in [0, 1] */
			if (!isfinite(prior) || prior < 0.0 || prior > 1.0)
			{
				if (errstr)
					*errstr = pstrdup("invalid NB model: class_priors contains invalid value (must be in [0, 1])");
				NDB_FREE(blob);
				return -1;
			}
			priors_dest[i] = prior;
		}
	}
	else
	{
		/* Initialize with default values if NULL */
		for (i = 0; i < model->n_classes; i++)
			priors_dest[i] = 1.0 / model->n_classes;
	}

	if (model->means != NULL)
	{
		for (i = 0; i < model->n_classes; i++)
		{
			if (model->means[i] != NULL)
			{
				for (j = 0; j < model->n_features; j++)
				{
					double		mean_val = model->means[i][j];

					/* Validate mean: must be finite */
					if (!isfinite(mean_val))
					{
						if (errstr)
							*errstr = pstrdup("invalid NB model: means contains non-finite value");
						NDB_FREE(blob);
						return -1;
					}
					means_dest[i * model->n_features + j] = mean_val;
				}
			}
			else
			{
				/* Initialize with zeros if NULL */
				for (j = 0; j < model->n_features; j++)
					means_dest[i * model->n_features + j] = 0.0;
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
		for (i = 0; i < model->n_classes; i++)
		{
			if (model->variances[i] != NULL)
			{
				for (j = 0; j < model->n_features; j++)
				{
					double		var_val = model->variances[i][j];

					/* Validate variance: must be finite and positive */
					if (!isfinite(var_val) || var_val < 0.0)
					{
						if (errstr)
							*errstr = pstrdup("invalid NB model: variances contains invalid value");
						NDB_FREE(blob);
						return -1;
					}
					/* Regularize to avoid division by zero */
					if (var_val < 1e-9)
						var_val = 1e-9;
					variances_dest[i * model->n_features + j] = var_val;
				}
			}
			else
			{
				/* Initialize with default regularization if NULL */
				for (j = 0; j < model->n_features; j++)
					variances_dest[i * model->n_features + j] = 1e-9;
			}
		}
	}
	else
	{
		/* Initialize all variances to regularization value if NULL */
		for (i = 0; i < model->n_classes * model->n_features; i++)
			variances_dest[i] = 1e-9;
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb	   *metrics_json = NULL;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"naive_bayes\","
						 "\"storage\":\"gpu\","
						 "\"n_classes\":%d,"
						 "\"n_features\":%d}",
						 model->n_classes,
						 model->n_features);

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
ndb_cuda_nb_train(const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  int class_count,
				  const Jsonb * hyperparams,
				  bytea * *model_data,
				  Jsonb * *metrics,
				  char **errstr)
{
	int		   *class_counts = NULL;
	double	   *class_priors = NULL;
	double	   *means = NULL;
	double	   *variances = NULL;
	struct GaussianNBModel model;
	bytea	   *blob = NULL;
	Jsonb	   *metrics_json = NULL;
	int			i;
	int			j;
	int			rc = -1;

	/* Initialize model structure to avoid undefined behavior in cleanup */
	memset(&model, 0, sizeof(GaussianNBModel));

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB train: features array is NULL");
		return -1;
	}
	if (labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB train: labels array is NULL");
		return -1;
	}
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB train: model_data output pointer is NULL");
		return -1;
	}
	if (n_samples <= 0 || n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB train: n_samples must be between 1 and 100000000");
		return -1;
	}
	if (feature_dim <= 0 || feature_dim > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB train: feature_dim must be between 1 and 1000000");
		return -1;
	}
	if (class_count <= 0 || class_count > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB train: class_count must be between 1 and 1000000");
		return -1;
	}

	/* Check for integer overflow in memory allocation */
	if (feature_dim > 0 && (size_t) n_samples > MaxAllocSize / sizeof(float) / (size_t) feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB train: feature array size exceeds MaxAllocSize");
		return -1;
	}

	elog(DEBUG1, "ndb_cuda_nb_train: entry: n_samples=%d, feature_dim=%d, class_count=%d",
		 n_samples, feature_dim, class_count);

	/* Allocate host memory with overflow checks */
	/* Note: palloc never returns NULL in PostgreSQL - it errors on failure */
	class_counts = (int *) palloc0(sizeof(int) * class_count);
	class_priors = (double *) palloc(sizeof(double) * class_count);

	if (feature_dim > 0 && (size_t) class_count > MaxAllocSize / sizeof(double) / (size_t) feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB train: means array size exceeds MaxAllocSize");
		goto cleanup;
	}
	means = (double *) palloc0(sizeof(double) * (size_t) class_count * (size_t) feature_dim);
	variances = (double *) palloc0(sizeof(double) * (size_t) class_count * (size_t) feature_dim);

	/* Validate input data for NaN/Inf before processing */
	/* Use rint() to match CPU training behavior - labels must be integral */
	for (i = 0; i < n_samples; i++)
	{
		double		label_val = labels[i];
		int			class;

		if (!isfinite(label_val))
		{
			if (errstr)
				*errstr = pstrdup("CUDA NB train: non-finite value in labels array");
			goto cleanup;
		}

		/* Round to nearest integer, matching CPU training behavior */
		class = (int) rint(label_val);

		/* Check label is close to an integer and in valid range */
		if (class < 0 || class >= class_count || fabs(label_val - (double) class) > 1e-6)
		{
			if (errstr)
				*errstr = pstrdup("CUDA NB train: invalid class label in labels array (must be integral 0 or 1 for binary)");
			goto cleanup;
		}
	}

	for (i = 0; i < n_samples; i++)
	{
		for (j = 0; j < feature_dim; j++)
		{
			if (!isfinite(features[i * feature_dim + j]))
			{
				if (errstr)
					*errstr = pstrdup("CUDA NB train: non-finite value in features array");
				goto cleanup;
			}
		}
	}

	/*
	 * Step 1: Count samples per class using CUDA NOTE: This may fail due to
	 * CUDA context issues in forked PostgreSQL backends. If GPU training
	 * consistently fails, disable GPU or use CPU training.
	 */
	elog(DEBUG1, "ndb_cuda_nb_train: Calling ndb_cuda_nb_count_classes");
	if (ndb_cuda_nb_count_classes(labels, n_samples, class_count, class_counts) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA class counting failed");
		goto cleanup;
	}
	elog(DEBUG1, "ndb_cuda_nb_train: count_classes returned, class_counts[0]=%d, class_counts[1]=%d",
		 class_counts[0], class_counts[1]);

	/* Step 2: Compute class priors with division by zero protection */
	elog(DEBUG1, "ndb_cuda_nb_train: Computing class priors");
	if (n_samples <= 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB train: n_samples is zero (division by zero)");
		goto cleanup;
	}
	for (i = 0; i < class_count; i++)
	{
		if (class_counts[i] > 0)
		{
			class_priors[i] = (double) class_counts[i] / (double) n_samples;
			/* Validate computed prior */
			if (!isfinite(class_priors[i]) || class_priors[i] < 0.0 || class_priors[i] > 1.0)
			{
				if (errstr)
					*errstr = pstrdup("CUDA NB train: computed invalid class prior");
				goto cleanup;
			}
		}
		else
		{
			class_priors[i] = 1e-10;	/* Avoid log(0) */
		}
	}
	elog(DEBUG1, "ndb_cuda_nb_train: Class priors computed, calling ndb_cuda_nb_compute_means");

	/* Step 3: Compute means using CUDA */
	if (ndb_cuda_nb_compute_means(features, labels, n_samples, feature_dim, class_count, means, class_counts) != 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA mean computation failed");
		goto cleanup;
	}

	/* Step 4: Compute variances using CUDA */
	if (ndb_cuda_nb_compute_variances(features, labels, means, n_samples, feature_dim, class_count, variances, class_counts) != 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA variance computation failed");
		goto cleanup;
	}

	/* Regularize variances to avoid division by zero */
	for (i = 0; i < class_count * feature_dim; i++)
	{
		if (variances[i] < 1e-9)
			variances[i] = 1e-9;
	}

	/* Build model structure for packing */
	model.n_classes = class_count;
	model.n_features = feature_dim;
	model.class_priors = class_priors;
	model.means = (double **) palloc(sizeof(double *) * class_count);
	model.variances = (double **) palloc(sizeof(double *) * class_count);

	for (i = 0; i < class_count; i++)
	{
		model.means[i] = means + i * feature_dim;
		model.variances[i] = variances + i * feature_dim;
	}

	/*
	 * Pack model - pass NULL for metrics if caller doesn't want them to avoid
	 * DirectFunctionCall issues
	 */
	if (ndb_cuda_nb_pack_model(&model, &blob, metrics ? &metrics_json : NULL, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA NB model packing failed");
		goto cleanup;
	}

	*model_data = blob;
	if (metrics != NULL)
		*metrics = metrics_json;

	rc = 0;

cleanup:
	/* Free model structure arrays (not the data they point to) */
	if (model.means != NULL)
		NDB_FREE(model.means);
	if (model.variances != NULL)
		NDB_FREE(model.variances);

	/* Free host memory */
	if (class_counts != NULL)
		NDB_FREE(class_counts);
	if (class_priors != NULL)
		NDB_FREE(class_priors);
	if (means != NULL)
		NDB_FREE(means);
	if (variances != NULL)
		NDB_FREE(variances);

	return rc;
}

int
ndb_cuda_nb_predict(const bytea * model_data,
					const float *input,
					int feature_dim,
					int *class_out,
					double *probability_out,
					char **errstr)
{
	const char *base;
	NdbCudaNbModelHeader *hdr;
	const double *priors;
	const double *means;
	const double *variances;
	double	   *class_log_probs;
	double		max_log_prob;
	int			best_class;
	int			i,
				j;
	size_t		expected_size;
	double		log_prob;
	double		diff;
	double		var;
	double		log_pdf;
	double		prob;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || class_out == NULL || probability_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for CUDA NB prediction");
		return -1;
	}

	/* Validate model_data bytea */
	if (VARSIZE_ANY_EXHDR(model_data) < sizeof(NdbCudaNbModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB predict: model_data too small (corrupted or invalid)");
		return -1;
	}

	base = VARDATA_ANY(model_data);
	hdr = (NdbCudaNbModelHeader *) base;

	/* Validate model header */
	if (hdr->n_classes <= 0 || hdr->n_classes > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB predict: invalid n_classes in model header");
		return -1;
	}
	if (hdr->n_features <= 0 || hdr->n_features > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB predict: invalid n_features in model header");
		return -1;
	}
	if (feature_dim != hdr->n_features)
	{
		if (errstr)
			*errstr = psprintf("CUDA NB predict: feature dimension mismatch (expected %d, got %d)", hdr->n_features, feature_dim);
		return -1;
	}

	/* Validate bytea size matches expected payload */
	expected_size = sizeof(NdbCudaNbModelHeader)
		+ sizeof(double) * (size_t) hdr->n_classes
		+ sizeof(double) * (size_t) hdr->n_classes * (size_t) hdr->n_features
		+ sizeof(double) * (size_t) hdr->n_classes * (size_t) hdr->n_features;
	if (VARSIZE_ANY_EXHDR(model_data) < expected_size)
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB predict: model_data size mismatch (corrupted model)");
		return -1;
	}

	/* Validate input array (input already checked at function entry) */
	for (i = 0; i < feature_dim; i++)
	{
		if (!isfinite(input[i]))
		{
			if (errstr)
				*errstr = pstrdup("CUDA NB predict: non-finite value in input array");
			return -1;
		}
	}

	priors = (const double *) (base + sizeof(NdbCudaNbModelHeader));
	means = (const double *) (base + sizeof(NdbCudaNbModelHeader) + sizeof(double) * (size_t) hdr->n_classes);
	variances = (const double *) (base + sizeof(NdbCudaNbModelHeader) + sizeof(double) * (size_t) hdr->n_classes + sizeof(double) * (size_t) hdr->n_classes * (size_t) hdr->n_features);

	/* Note: palloc never returns NULL in PostgreSQL - it errors on failure */
	class_log_probs = (double *) palloc(sizeof(double) * hdr->n_classes);

	/*
	 * Note: priors, means, variances are computed from valid bytea offsets,
	 * so they cannot be NULL
	 */

	/* Compute log probability for each class */
	for (i = 0; i < hdr->n_classes; i++)
	{
		double		prior = priors[i];

		if (!isfinite(prior) || prior < 0.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA NB predict: invalid prior in model");
			NDB_FREE(class_log_probs);
			return -1;
		}
		log_prob = log(prior + 1e-10);	/* Add small epsilon to avoid log(0) */

		for (j = 0; j < feature_dim; j++)
		{
			double		mean_val = means[i * feature_dim + j];
			double		var_val = variances[i * feature_dim + j];

			/* Validate model parameters */
			if (!isfinite(mean_val) || !isfinite(var_val) || var_val <= 0.0)
			{
				if (errstr)
					*errstr = pstrdup("CUDA NB predict: invalid mean or variance in model");
				NDB_FREE(class_log_probs);
				return -1;
			}

			diff = (double) input[j] - mean_val;
			var = var_val + 1e-9;	/* Regularization */

			/* Check for division by zero */
			if (var <= 0.0)
			{
				if (errstr)
					*errstr = pstrdup("CUDA NB predict: variance is zero or negative");
				NDB_FREE(class_log_probs);
				return -1;
			}

			log_pdf = -0.5 * log(2.0 * M_PI * var) - 0.5 * (diff * diff) / var;

			/* Validate computed log_pdf */
			if (!isfinite(log_pdf))
			{
				if (errstr)
					*errstr = pstrdup("CUDA NB predict: computed non-finite log_pdf");
				NDB_FREE(class_log_probs);
				return -1;
			}

			log_prob += log_pdf;
		}

		/* Validate computed log_prob */
		if (!isfinite(log_prob))
		{
			if (errstr)
				*errstr = pstrdup("CUDA NB predict: computed non-finite log_prob");
			NDB_FREE(class_log_probs);
			return -1;
		}

		class_log_probs[i] = log_prob;
	}

	/* Find class with maximum log probability */
	max_log_prob = class_log_probs[0];
	best_class = 0;
	for (i = 1; i < hdr->n_classes; i++)
	{
		if (class_log_probs[i] > max_log_prob)
		{
			max_log_prob = class_log_probs[i];
			best_class = i;
		}
	}

	/* Convert log probabilities to probabilities (normalize) */
	{
		double		max_log = max_log_prob;
		double		sum = 0.0;
		int			k;

		/* Validate max_log */
		if (!isfinite(max_log))
		{
			if (errstr)
				*errstr = pstrdup("CUDA NB predict: max_log_prob is non-finite");
			NDB_FREE(class_log_probs);
			return -1;
		}

		for (k = 0; k < hdr->n_classes; k++)
		{
			double		exp_val = exp(class_log_probs[k] - max_log);

			if (!isfinite(exp_val))
			{
				if (errstr)
					*errstr = pstrdup("CUDA NB predict: computed non-finite exp value");
				NDB_FREE(class_log_probs);
				return -1;
			}
			sum += exp_val;
		}

		/* Check for division by zero */
		if (sum <= 0.0 || !isfinite(sum))
		{
			if (errstr)
				*errstr = pstrdup("CUDA NB predict: sum of probabilities is zero or non-finite");
			NDB_FREE(class_log_probs);
			return -1;
		}

		prob = exp(class_log_probs[best_class] - max_log) / sum;
		if (!isfinite(prob) || prob < 0.0 || prob > 1.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA NB predict: computed invalid probability");
			NDB_FREE(class_log_probs);
			return -1;
		}

		*probability_out = prob;
	}

	*class_out = best_class;
	NDB_FREE(class_log_probs);

	return 0;
}

/*
 * Batch prediction: predict for multiple samples
 */
int
ndb_cuda_nb_predict_batch(const bytea * model_data,
						  const float *features,
						  int n_samples,
						  int feature_dim,
						  int *predictions_out,
						  char **errstr)
{
	const char *base;
	const		NdbCudaNbModelHeader *hdr;
	int			i;
	int			rc;

	if (errstr)
		*errstr = NULL;

	if (model_data == NULL || features == NULL || predictions_out == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid inputs for CUDA NB batch predict");
		return -1;
	}

	/* Validate model_data bytea */
	if (VARSIZE_ANY_EXHDR(model_data) < sizeof(NdbCudaNbModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("CUDA NB batch predict: model_data too small");
		return -1;
	}

	base = VARDATA_ANY(model_data);
	hdr = (const NdbCudaNbModelHeader *) base;

	/* Validate model header */
	if (hdr->n_features != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("CUDA NB batch predict: feature dimension mismatch (expected %d, got %d)",
							   hdr->n_features, feature_dim);
		return -1;
	}

	/* Predict for each sample */
	for (i = 0; i < n_samples; i++)
	{
		const float *input = features + (i * feature_dim);
		int			class_out = 0;
		double		probability_out = 0.0;

		rc = ndb_cuda_nb_predict(model_data,
								 input,
								 feature_dim,
								 &class_out,
								 &probability_out,
								 errstr);

		if (rc != 0)
		{
			/* On error, set default prediction */
			predictions_out[i] = 0;
			continue;
		}

		predictions_out[i] = class_out;
	}

	return 0;
}

/*
 * Batch evaluation: compute metrics for multiple samples
 */
int
ndb_cuda_nb_evaluate_batch(const bytea * model_data,
						   const float *features,
						   const int *labels,
						   int n_samples,
						   int feature_dim,
						   double *accuracy_out,
						   double *precision_out,
						   double *recall_out,
						   double *f1_out,
						   char **errstr)
{
	int		   *predictions = NULL;
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;
	int			i;
	int			total_correct = 0;
	int			rc;

	if (errstr)
		*errstr = NULL;

	if (model_data == NULL || features == NULL || labels == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid inputs for CUDA NB batch evaluate");
		return -1;
	}

	if (accuracy_out == NULL || precision_out == NULL
		|| recall_out == NULL || f1_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("output pointers are NULL");
		return -1;
	}

	/* Allocate predictions array */
	/* Note: palloc never returns NULL in PostgreSQL - it errors on failure */
	predictions = (int *) palloc(sizeof(int) * (size_t) n_samples);

	/* Batch predict */
	rc = ndb_cuda_nb_predict_batch(model_data,
								   features,
								   n_samples,
								   feature_dim,
								   predictions,
								   errstr);

	if (rc != 0)
	{
		NDB_FREE(predictions);
		return -1;
	}

	/* Compute confusion matrix for binary classification */
	for (i = 0; i < n_samples; i++)
	{
		int			true_label = labels[i];
		int			pred_label = predictions[i];

		if (true_label < 0 || true_label > 1)
			continue;
		if (pred_label < 0 || pred_label > 1)
			continue;

		if (true_label == 1 && pred_label == 1)
		{
			tp++;
			total_correct++;
		}
		else if (true_label == 0 && pred_label == 0)
		{
			tn++;
			total_correct++;
		}
		else if (true_label == 0 && pred_label == 1)
			fp++;
		else if (true_label == 1 && pred_label == 0)
			fn++;
	}

	/* Compute metrics */
	*accuracy_out = (n_samples > 0)
		? ((double) total_correct / (double) n_samples)
		: 0.0;

	if ((tp + fp) > 0)
		*precision_out = (double) tp / (double) (tp + fp);
	else
		*precision_out = 0.0;

	if ((tp + fn) > 0)
		*recall_out = (double) tp / (double) (tp + fn);
	else
		*recall_out = 0.0;

	if ((*precision_out + *recall_out) > 0.0)
		*f1_out = 2.0 * ((*precision_out) * (*recall_out))
			/ ((*precision_out) + (*recall_out));
	else
		*f1_out = 0.0;

	NDB_FREE(predictions);

	return 0;
}

#else

/* Stubs for non-CUDA builds - match function signatures from header */

int
ndb_cuda_nb_train(const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  int class_count,
				  const Jsonb * hyperparams,
				  bytea * *model_data,
				  Jsonb * *metrics,
				  char **errstr)
{
	(void) features;
	(void) labels;
	(void) n_samples;
	(void) feature_dim;
	(void) class_count;
	(void) hyperparams;
	(void) model_data;
	(void) metrics;
	if (errstr)
		*errstr = pstrdup("CUDA Naive Bayes not built with GPU support");
	return -1;
}

int
ndb_cuda_nb_predict(const bytea * model_data,
					const float *input,
					int feature_dim,
					int *class_out,
					double *probability_out,
					char **errstr)
{
	(void) model_data;
	(void) input;
	(void) feature_dim;
	(void) class_out;
	(void) probability_out;
	if (errstr)
		*errstr = pstrdup("CUDA Naive Bayes not built with GPU support");
	return -1;
}

int
ndb_cuda_nb_pack_model(const struct GaussianNBModel *model,
					   bytea * *model_data,
					   Jsonb * *metrics,
					   char **errstr)
{
	(void) model;
	(void) model_data;
	(void) metrics;
	if (errstr)
		*errstr = pstrdup("CUDA Naive Bayes not built with GPU support");
	return -1;
}

int
ndb_cuda_nb_predict_batch(const bytea * model_data,
						  const float *features,
						  int n_samples,
						  int feature_dim,
						  int *predictions_out,
						  char **errstr)
{
	(void) model_data;
	(void) features;
	(void) n_samples;
	(void) feature_dim;
	(void) predictions_out;
	if (errstr)
		*errstr = pstrdup("CUDA Naive Bayes not built with GPU support");
	return -1;
}

int
ndb_cuda_nb_evaluate_batch(const bytea * model_data,
						   const float *features,
						   const int *labels,
						   int n_samples,
						   int feature_dim,
						   double *accuracy_out,
						   double *precision_out,
						   double *recall_out,
						   double *f1_out,
						   char **errstr)
{
	(void) model_data;
	(void) features;
	(void) labels;
	(void) n_samples;
	(void) feature_dim;
	(void) accuracy_out;
	(void) precision_out;
	(void) recall_out;
	(void) f1_out;
	if (errstr)
		*errstr = pstrdup("CUDA Naive Bayes not built with GPU support");
	return -1;
}

#endif							/* NDB_GPU_CUDA */
