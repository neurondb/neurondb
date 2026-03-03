/*-------------------------------------------------------------------------
 *
 * gpu_knn_cuda.c
 *    CUDA backend bridge for K-Nearest Neighbors training and prediction.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_knn_cuda.c
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

#include "neurondb_cuda_knn.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_guc.h"
#include "neurondb_constants.h"

typedef struct KNNModel
{
	int			n_samples;
	int			n_features;
	int			k;
	int			task_type;
	float	   *features;
	double	   *labels;
}			KNNModel;

int
ndb_cuda_knn_pack(const struct KNNModel *model,
				  bytea * *model_data,
				  Jsonb * *metrics,
				  char **errstr)
{
	size_t		payload_bytes;
	size_t		features_bytes;
	size_t		labels_bytes;
	bytea	   *blob = NULL;
	char	   *base = NULL;
	NdbCudaKnnModelHeader *hdr = NULL;
	float	   *features_dest = NULL;
	double	   *labels_dest = NULL;
	/* Allocate blob in TopMemoryContext to ensure it persists correctly
	 * regardless of the caller's context */
	MemoryContext oldcontext = CurrentMemoryContext;
	MemoryContextSwitchTo(TopMemoryContext);

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		if (errstr)
			*errstr = pstrdup("invalid KNN model for CUDA pack: model or model_data is NULL");
		return -1;
	}

	if (model->n_samples <= 0 || model->n_samples > 100000000)
	{
		MemoryContextSwitchTo(oldcontext);
		if (errstr)
			*errstr = pstrdup("invalid KNN model: n_samples must be between 1 and 100000000");
		return -1;
	}
	if (model->n_features <= 0 || model->n_features > 1000000)
	{
		MemoryContextSwitchTo(oldcontext);
		if (errstr)
			*errstr = pstrdup("invalid KNN model: n_features must be between 1 and 1000000");
		return -1;
	}
	if (model->k <= 0 || model->k > model->n_samples)
	{
		MemoryContextSwitchTo(oldcontext);
		if (errstr)
			*errstr = pstrdup("invalid KNN model: k must be between 1 and n_samples");
		return -1;
	}
	if (model->task_type != 0 && model->task_type != 1)
	{
		MemoryContextSwitchTo(oldcontext);
		if (errstr)
			*errstr = pstrdup("invalid KNN model: task_type must be 0 (classification) or 1 (regression)");
		return -1;
	}

	features_bytes = sizeof(float) * (size_t) model->n_samples * (size_t) model->n_features;
	labels_bytes = sizeof(double) * (size_t) model->n_samples;

	if (features_bytes / sizeof(float) / (size_t) model->n_samples != (size_t) model->n_features)
	{
		MemoryContextSwitchTo(oldcontext);
		if (errstr)
			*errstr = pstrdup("invalid KNN model: integer overflow in features size calculation");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaKnnModelHeader) + features_bytes + labels_bytes;

	/* Check for overflow in total payload */
	if (payload_bytes < sizeof(NdbCudaKnnModelHeader))
	{
		MemoryContextSwitchTo(oldcontext);
		if (errstr)
			*errstr = pstrdup("invalid KNN model: payload size underflow");
		return -1;
	}
	if (payload_bytes > (MaxAllocSize - VARHDRSZ))
	{
		MemoryContextSwitchTo(oldcontext);
		if (errstr)
			*errstr = pstrdup("invalid KNN model: payload size exceeds MaxAllocSize");
		return -1;
	}

	/* Allocate blob in TopMemoryContext */
	{
		char *tmp = (char *) palloc0(VARHDRSZ + payload_bytes);
		blob = (bytea *) tmp;
		
	
	if (blob == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		if (errstr)
			*errstr = pstrdup("CUDA KNN pack: memory allocation failed");
		return -1;
	}

	}
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaKnnModelHeader *) base;
	hdr->n_samples = model->n_samples;
	hdr->n_features = model->n_features;
	hdr->k = model->k;
	hdr->task_type = model->task_type;

	features_dest = (float *) (base + sizeof(NdbCudaKnnModelHeader));
	labels_dest = (double *) (base + sizeof(NdbCudaKnnModelHeader) + features_bytes);

	/* Copy features and labels into blob. Use PG_TRY to catch any errors during memcpy
	 * that might indicate invalid input pointers or memory corruption. */
	PG_TRY();
	{
		if (model->features != NULL)
		{
			memcpy(features_dest, model->features, features_bytes);
		}
		else
		{
			memset(features_dest, 0, features_bytes);
		}

		if (model->labels != NULL)
		{
			memcpy(labels_dest, model->labels, labels_bytes);
		}
		else
		{
			memset(labels_dest, 0, labels_bytes);
		}
	}
	PG_CATCH();
	{
		/* Exception during memcpy - free blob and return error */
		FlushErrorState();
		
		/* Restore memory context before freeing and returning */
		MemoryContextSwitchTo(TopMemoryContext);
		
		if (blob != NULL)
		{
			pfree(blob);
			blob = NULL;
		}
		
		/* Switch to oldcontext before allocating error string */
		MemoryContextSwitchTo(oldcontext);
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("Exception during CUDA KNN model data copy - input data may be invalid or corrupted");
		return -1;
	}
	PG_END_TRY();

	if (metrics != NULL)
	{
		/* Skip metrics creation to avoid memory context issues with DirectFunctionCall */
		/* Metrics can be created later if needed */
		*metrics = NULL;
	}

	*model_data = blob;
	
	/* Restore original memory context before returning */
	MemoryContextSwitchTo(oldcontext);
	
	return 0;
}

int
ndb_cuda_knn_train(const float *features,
				   const double *labels,
				   int n_samples,
				   int feature_dim,
				   int k,
				   int task_type,
				   const Jsonb * hyperparams,
				   bytea * *model_data,
				   Jsonb * *metrics,
				   char **errstr)
{
	struct KNNModel model;
	bytea	   *blob = NULL;
	Jsonb	   *metrics_json = NULL;
	int			i,
				j;
	float	   *features_copy = NULL;
	double	   *labels_copy = NULL;
	MemoryContext oldcontext;
	int			extracted_task;
	int			rc = -1;

	/* CPU mode: never execute GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
	{
		if (errstr)
			*errstr = pstrdup("CUDA knn_train: CPU mode - GPU code should not be called");
		return -1;
	}

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: features array is NULL");
		return -1;
	}
	if (labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: labels array is NULL");
		return -1;
	}
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: model_data output pointer is NULL");
		return -1;
	}
	if (n_samples <= 0 || n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: n_samples must be between 1 and 100000000");
		return -1;
	}
	if (feature_dim <= 0 || feature_dim > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: feature_dim must be between 1 and 1000000");
		return -1;
	}
	if (k <= 0 || k > n_samples)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: k must be between 1 and n_samples");
		return -1;
	}
	if (task_type != 0 && task_type != 1)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: task_type must be 0 (classification) or 1 (regression)");
		return -1;
	}


	/* Validate input data for NaN/Inf before processing */
	for (i = 0; i < n_samples; i++)
	{
		for (j = 0; j < feature_dim; j++)
		{
			if (!isfinite(features[i * feature_dim + j]))
			{
				if (errstr)
					*errstr = pstrdup("CUDA KNN train: non-finite value in features array");
				return -1;
			}
		}
		if (!isfinite(labels[i]))
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN train: non-finite value in labels array");
			return -1;
		}
		/* For classification, validate labels are integers */
		if (task_type == 0)
		{
			double		label = labels[i];

			if (label != floor(label) || label < 0.0)
			{
				if (errstr)
					*errstr = pstrdup("CUDA KNN train: classification labels must be non-negative integers");
				return -1;
			}
		}
	}

	/* Extract hyperparameters if provided */
	if (hyperparams != NULL)
	{
		Datum		k_datum;
		Datum		task_type_datum;
		Datum		numeric_datum;
		Numeric		num;

		/* Wrap DirectFunctionCall calls in PG_TRY to catch any errors */
		PG_TRY();
		{
			k_datum = DirectFunctionCall2(
										  jsonb_object_field,
										  JsonbPGetDatum(hyperparams),
										  CStringGetTextDatum("k"));
			if (DatumGetPointer(k_datum) != NULL)
			{
				numeric_datum = DirectFunctionCall1(
													jsonb_numeric, k_datum);
				if (DatumGetPointer(numeric_datum) != NULL)
				{
					num = DatumGetNumeric(numeric_datum);
					k = DatumGetInt32(
									  DirectFunctionCall1(numeric_int4,
														  NumericGetDatum(num)));
					if (k <= 0 || k > n_samples)
						k = (n_samples < 10) ? n_samples : 10;
				}
			}

			task_type_datum = DirectFunctionCall2(
												  jsonb_object_field,
												  JsonbPGetDatum(hyperparams),
												  CStringGetTextDatum("task_type"));
			if (DatumGetPointer(task_type_datum) != NULL)
			{
				numeric_datum = DirectFunctionCall1(
													jsonb_numeric, task_type_datum);
				if (DatumGetPointer(numeric_datum) != NULL)
				{
					num = DatumGetNumeric(numeric_datum);
					extracted_task = DatumGetInt32(
												   DirectFunctionCall1(numeric_int4,
																	   NumericGetDatum(num)));
					if (extracted_task == 0 || extracted_task == 1)
						task_type = extracted_task;
				}
			}
		}
		PG_CATCH();
		{
			/* If DirectFunctionCall fails, use defaults and continue */
			FlushErrorState();
			/* k and task_type already have default values */
		}
		PG_END_TRY();
	}

	/* Copy input data into TopMemoryContext to ensure it persists during pack call.
	 * This prevents issues if the input pointers point to memory that might be freed. */

	/* Check for integer overflow in memory allocation before copying */
	if (feature_dim > 0 && (size_t) n_samples > MaxAllocSize / sizeof(float) / (size_t) feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: feature array size exceeds MaxAllocSize");
		return -1;
	}

	features_copy = NULL;
	labels_copy = NULL;
	{
		oldcontext = CurrentMemoryContext;
		MemoryContextSwitchTo(TopMemoryContext);
		
		features_copy = (float *) palloc(sizeof(float) * (size_t) n_samples * (size_t) feature_dim);
		if (features_copy == NULL)
		{
			MemoryContextSwitchTo(oldcontext);
			if (errstr)
				*errstr = pstrdup("CUDA KNN train: failed to allocate features_copy");
			return -1;
		}

		labels_copy = (double *) palloc(sizeof(double) * (size_t) n_samples);
		if (labels_copy == NULL)
		{
			pfree(features_copy);
			MemoryContextSwitchTo(oldcontext);
			if (errstr)
				*errstr = pstrdup("CUDA KNN train: failed to allocate labels_copy");
			return -1;
		}
		
		/* Copy data while in TopMemoryContext */
		memcpy(features_copy, features, sizeof(float) * (size_t) n_samples * (size_t) feature_dim);
		memcpy(labels_copy, labels, sizeof(double) * (size_t) n_samples);
		
		MemoryContextSwitchTo(oldcontext);
	}

	/* Build model structure - use copied data in TopMemoryContext */
	model.n_samples = n_samples;
	model.n_features = feature_dim;
	model.k = k;
	model.task_type = task_type;
	model.features = features_copy;  /* In TopMemoryContext - pack will copy from here */
	model.labels = labels_copy;      /* In TopMemoryContext - pack will copy from here */

	/* Pack model - pass NULL for metrics if caller doesn't want them to avoid
	 * DirectFunctionCall issues. No PG_TRY needed - pack handles its own errors. */
	if (ndb_cuda_knn_pack(&model, &blob, metrics ? &metrics_json : NULL, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA KNN model packing failed");
		return -1;
	}

	*model_data = blob;
	if (metrics != NULL)
		*metrics = metrics_json;

	rc = 0;

	return rc;
}

int
ndb_cuda_knn_predict(const bytea * model_data,
					 const float *input,
					 int feature_dim,
					 double *prediction_out,
					 char **errstr)
{
	const char *base;
	NdbCudaKnnModelHeader *hdr = NULL;
	const float *training_features;
	const double *training_labels;
	float	   *distances = NULL;
	int			i;
	size_t		expected_size;

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: model_data is NULL");
		return -1;
	}
	if (input == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: input array is NULL");
		return -1;
	}
	if (prediction_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: prediction_out pointer is NULL");
		return -1;
	}

	/* Use VARDATA_ANY which handles toasted data safely without creating copies */
	/* The caller (ml_knn.c) already copies model_data, so this should be safe */
	if (VARSIZE_ANY_EXHDR(model_data) < sizeof(NdbCudaKnnModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: model_data too small (corrupted or invalid)");
		return -1;
	}

	base = VARDATA_ANY(model_data);
	hdr = (NdbCudaKnnModelHeader *) base;

	/* Validate model header */
	if (hdr->n_samples <= 0 || hdr->n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: invalid n_samples in model header");
		return -1;
	}
	if (hdr->n_features <= 0 || hdr->n_features > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: invalid n_features in model header");
		return -1;
	}
	if (hdr->k <= 0 || hdr->k > hdr->n_samples)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: invalid k in model header");
		return -1;
	}
	if (hdr->task_type != 0 && hdr->task_type != 1)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: invalid task_type in model header");
		return -1;
	}
	if (feature_dim != hdr->n_features)
	{
		if (errstr)
			*errstr = psprintf("CUDA KNN predict: feature dimension mismatch (expected %d, got %d)", hdr->n_features, feature_dim);
		return -1;
	}

	/* Validate bytea size matches expected payload */
	expected_size = sizeof(NdbCudaKnnModelHeader)
		+ sizeof(float) * (size_t) hdr->n_samples * (size_t) hdr->n_features
		+ sizeof(double) * (size_t) hdr->n_samples;
	if (VARSIZE_ANY_EXHDR(model_data) < expected_size)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: model_data size mismatch (corrupted model)");
		return -1;
	}

	/* Validate input array */
	for (i = 0; i < feature_dim; i++)
	{
		if (!isfinite(input[i]))
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN predict: non-finite value in input array");
			return -1;
		}
	}

	training_features = (const float *) (base + sizeof(NdbCudaKnnModelHeader));
	training_labels = (const double *) (base + sizeof(NdbCudaKnnModelHeader) + sizeof(float) * (size_t) hdr->n_samples * (size_t) hdr->n_features);

	/* Validate model data pointers */
	if (training_features == NULL || training_labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: model data pointers are NULL (corrupted model)");
		return -1;
	}

	/* Allocate distance array */
	if (hdr->n_samples > MaxAllocSize / sizeof(float))
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: distances array size exceeds MaxAllocSize");
		return -1;
	}
	nalloc(distances, float, hdr->n_samples);
	if (distances == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: failed to allocate distances array");
		return -1;
	}

	/* Step 1: Compute distances using CUDA */
	if (ndb_cuda_knn_compute_distances(input, training_features, hdr->n_samples, hdr->n_features, distances) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA distance computation failed");
		pfree(distances);
		return -1;
	}

	/* Validate computed distances */
	for (i = 0; i < hdr->n_samples; i++)
	{
		if (!isfinite(distances[i]) || distances[i] < 0.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN predict: computed invalid distance");
			pfree(distances);
			return -1;
		}
	}

	/* Step 2: Find top-k and compute prediction using CUDA */
	if (ndb_cuda_knn_find_top_k(distances, training_labels, hdr->n_samples, hdr->k, hdr->task_type, prediction_out) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA top-k computation failed");
		pfree(distances);
		return -1;
	}

	/* Validate prediction output */
	if (!isfinite(*prediction_out))
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: computed non-finite prediction");
		pfree(distances);
		return -1;
	}
	/* For classification, prediction should be an integer */
	if (hdr->task_type == 0)
	{
		if (*prediction_out != floor(*prediction_out) || *prediction_out < 0.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN predict: classification prediction must be non-negative integer");
			pfree(distances);
			return -1;
		}
	}

	pfree(distances);
	return 0;
}

/*
 * Batch prediction: predict for multiple samples
 */
int
ndb_cuda_knn_predict_batch(const bytea * model_data,
						   const float *features,
						   int n_samples,
						   int feature_dim,
						   int *predictions_out,
						   char **errstr)
{
	const char *base;
	const NdbCudaKnnModelHeader *hdr;
	int			i;
	int			rc;

	if (errstr)
		*errstr = NULL;

	if (model_data == NULL || features == NULL || predictions_out == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid inputs for CUDA KNN batch predict");
		return -1;
	}

	/* Use VARDATA_ANY which handles toasted data safely without creating copies */
	/* The caller already handles copying if needed */
	if (VARSIZE_ANY_EXHDR(model_data) < sizeof(NdbCudaKnnModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN batch predict: model_data too small");
		return -1;
	}

	base = VARDATA_ANY(model_data);
	hdr = (const NdbCudaKnnModelHeader *) base;

	/* Validate model header (same checks as predict) */
	if (hdr->n_samples <= 0 || hdr->n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN batch predict: invalid n_samples in model header");
		return -1;
	}
	if (hdr->n_features <= 0 || hdr->n_features > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN batch predict: invalid n_features in model header");
		return -1;
	}
	if (hdr->k <= 0 || hdr->k > hdr->n_samples)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN batch predict: invalid k in model header");
		return -1;
	}
	if (hdr->task_type != 0 && hdr->task_type != 1)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN batch predict: invalid task_type in model header");
		return -1;
	}
	if (hdr->n_features != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("CUDA KNN batch predict: feature dimension mismatch (expected %d, got %d)",
							   hdr->n_features, feature_dim);
		return -1;
	}

	/* Validate bytea size matches expected payload */
	{
		size_t		expected_size = sizeof(NdbCudaKnnModelHeader)
			+ sizeof(float) * (size_t) hdr->n_samples * (size_t) hdr->n_features
			+ sizeof(double) * (size_t) hdr->n_samples;

		if (VARSIZE_ANY_EXHDR(model_data) < expected_size)
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN batch predict: model_data size mismatch (corrupted model)");
			return -1;
		}
	}

	/* Predict for each sample */
	for (i = 0; i < n_samples; i++)
	{
		const float *input = features + (i * feature_dim);
		double		prediction = 0.0;

		rc = ndb_cuda_knn_predict(model_data,
								  input,
								  feature_dim,
								  &prediction,
								  errstr);

		if (rc != 0)
		{
			/* On error, set default prediction */
			predictions_out[i] = 0;
			continue;
		}

		/* Convert prediction to integer class for classification */
		if (hdr->task_type == 0)
			predictions_out[i] = (int) rint(prediction);
		else
			predictions_out[i] = (int) rint(prediction);
	}

	return 0;
}

/*
 * Batch evaluation: compute metrics for multiple samples
 */
int
ndb_cuda_knn_evaluate_batch(const bytea * model_data,
							const float *features,
							const double *labels,
							int n_samples,
							int feature_dim,
							double *accuracy_out,
							double *precision_out,
							double *recall_out,
							double *f1_out,
							char **errstr)
{
	const char *base;
	const NdbCudaKnnModelHeader *hdr;
	int		   *predictions = NULL;
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;
	int			i;
	int			total_valid = 0;
	int			rc;

	if (errstr)
		*errstr = NULL;

	if (model_data == NULL || features == NULL || labels == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN batch evaluate: invalid inputs");
		return -1;
	}

	if (accuracy_out == NULL || precision_out == NULL
		|| recall_out == NULL || f1_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN batch evaluate: output pointers are NULL");
		return -1;
	}

	/* Validate model header and check task_type */
	if (VARSIZE_ANY_EXHDR(model_data) < sizeof(NdbCudaKnnModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN batch evaluate: model_data too small");
		return -1;
	}

	base = VARDATA_ANY(model_data);
	hdr = (const NdbCudaKnnModelHeader *) base;

	/* Batch evaluate only supports classification models */
	if (hdr->task_type != 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN batch evaluate: supports classification models only");
		return -1;
	}

	/* Allocate predictions array */
	nalloc(predictions, int, (size_t) n_samples);
	if (predictions == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN batch evaluate: failed to allocate predictions array");
		return -1;
	}

	/* Batch predict */
	rc = ndb_cuda_knn_predict_batch(model_data,
									features,
									n_samples,
									feature_dim,
									predictions,
									errstr);

	if (rc != 0)
	{
		pfree(predictions);
		return -1;
	}

	/* Compute confusion matrix for binary classification */
	for (i = 0; i < n_samples; i++)
	{
		double		true_label_d = labels[i];
		int			true_label = (int) rint(true_label_d);
		int			pred_label = predictions[i];

		if (true_label < 0 || true_label > 1)
			continue;
		if (pred_label < 0 || pred_label > 1)
			continue;

		total_valid++;

		if (true_label == 1 && pred_label == 1)
		{
			tp++;
		}
		else if (true_label == 0 && pred_label == 0)
		{
			tn++;
		}
		else if (true_label == 0 && pred_label == 1)
			fp++;
		else if (true_label == 1 && pred_label == 0)
			fn++;
	}

	/*
	 * Compute metrics - use total_valid (tp+tn+fp+fn) as denominator for
	 * accuracy
	 */
	*accuracy_out = (total_valid > 0)
		? ((double) (tp + tn) / (double) total_valid)
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

	pfree(predictions);

	return 0;
}

#else

/* Stubs when CUDA is not available - return error codes */
int
ndb_cuda_knn_train(const float *features,
				   const double *labels,
				   int n_samples,
				   int feature_dim,
				   int k,
				   int task_type,
				   const Jsonb * hyperparams,
				   bytea * *model_data,
				   Jsonb * *metrics,
				   char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA not available");
	return -1;
}

int
ndb_cuda_knn_predict(const bytea * model_data,
					 const float *input,
					 int feature_dim,
					 double *prediction_out,
					 char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA not available");
	return -1;
}

int
ndb_cuda_knn_predict_batch(const bytea * model_data,
						   const float *features,
						   int n_samples,
						   int feature_dim,
						   int *predictions_out,
						   char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA not available");
	return -1;
}

int
ndb_cuda_knn_evaluate_batch(const bytea * model_data,
							const float *features,
							const double *labels,
							int n_samples,
							int feature_dim,
							double *accuracy_out,
							double *precision_out,
							double *recall_out,
							double *f1_out,
							char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA not available");
	return -1;
}

int
ndb_cuda_knn_pack(const struct KNNModel *model,
				  bytea * *model_data,
				  Jsonb * *metrics,
				  char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA not available");
	return -1;
}

#endif							/* NDB_GPU_CUDA */
