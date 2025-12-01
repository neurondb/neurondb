/*-------------------------------------------------------------------------
 *
 * gpu_knn_cuda.c
 *    CUDA backend bridge for K-Nearest Neighbors training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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

/* KNN model structure */
typedef struct KNNModel
{
	int			n_samples;
	int			n_features;
	int			k;
	int			task_type;		/* 0=classification, 1=regression */
	float	   *features;		/* Training features [n_samples * n_features] */
	double	   *labels;			/* Training labels [n_samples] */
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
	bytea	   *blob;
	char	   *base;
	NdbCudaKnnModelHeader *hdr;
	float	   *features_dest;
	double	   *labels_dest;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model for CUDA pack: model or model_data is NULL");
		return -1;
	}

	/* Validate model structure */
	if (model->n_samples <= 0 || model->n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: n_samples must be between 1 and 100000000");
		return -1;
	}
	if (model->n_features <= 0 || model->n_features > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: n_features must be between 1 and 1000000");
		return -1;
	}
	if (model->k <= 0 || model->k > model->n_samples)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: k must be between 1 and n_samples");
		return -1;
	}
	if (model->task_type != 0 && model->task_type != 1)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: task_type must be 0 (classification) or 1 (regression)");
		return -1;
	}

	/* Check for integer overflow in size calculations */
	features_bytes = sizeof(float) * (size_t) model->n_samples * (size_t) model->n_features;
	labels_bytes = sizeof(double) * (size_t) model->n_samples;

	if (features_bytes / sizeof(float) / (size_t) model->n_samples != (size_t) model->n_features)
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: integer overflow in features size calculation");
		return -1;
	}

	payload_bytes = sizeof(NdbCudaKnnModelHeader) + features_bytes + labels_bytes;

	/* Check for overflow in total payload */
	if (payload_bytes < sizeof(NdbCudaKnnModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: payload size underflow");
		return -1;
	}
	if (payload_bytes > (MaxAllocSize - VARHDRSZ))
	{
		if (errstr)
			*errstr = pstrdup("invalid KNN model: payload size exceeds MaxAllocSize");
		return -1;
	}

	blob = (bytea *) palloc(VARHDRSZ + payload_bytes);
	if (blob == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN pack: memory allocation failed");
		return -1;
	}

	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaKnnModelHeader *) base;
	hdr->n_samples = model->n_samples;
	hdr->n_features = model->n_features;
	hdr->k = model->k;
	hdr->task_type = model->task_type;

	features_dest = (float *) (base + sizeof(NdbCudaKnnModelHeader));
	labels_dest = (double *) (base + sizeof(NdbCudaKnnModelHeader) + sizeof(float) * (size_t) model->n_samples * (size_t) model->n_features);

	if (model->features != NULL)
	{
		/* Validate features array for NaN/Inf */
		for (int i = 0; i < model->n_samples * model->n_features; i++)
		{
			if (!isfinite(model->features[i]))
			{
				if (errstr)
					*errstr = pstrdup("invalid KNN model: features array contains non-finite value");
				NDB_FREE(blob);
				return -1;
			}
		}
		memcpy(features_dest, model->features, features_bytes);
	}
	else
	{
		/* Initialize with zeros if NULL */
		memset(features_dest, 0, features_bytes);
	}

	if (model->labels != NULL)
	{
		/* Validate labels array for NaN/Inf and task-specific constraints */
		for (int i = 0; i < model->n_samples; i++)
		{
			if (!isfinite(model->labels[i]))
			{
				if (errstr)
					*errstr = pstrdup("invalid KNN model: labels array contains non-finite value");
				NDB_FREE(blob);
				return -1;
			}
			/* For classification, labels should be integers */
			if (model->task_type == 0)
			{
				double		label = model->labels[i];

				if (label != floor(label) || label < 0.0)
				{
					if (errstr)
						*errstr = pstrdup("invalid KNN model: classification labels must be non-negative integers");
					NDB_FREE(blob);
					return -1;
				}
			}
		}
		memcpy(labels_dest, model->labels, labels_bytes);
	}
	else
	{
		/* Initialize with zeros if NULL */
		memset(labels_dest, 0, labels_bytes);
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb	   *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"knn\","
						 "\"storage\":\"gpu\","
						 "\"n_samples\":%d,"
						 "\"n_features\":%d,"
						 "\"k\":%d,"
						 "\"task_type\":%d}",
						 model->n_samples,
						 model->n_features,
						 model->k,
						 model->task_type);

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
	float	   *features_copy = NULL;
	double	   *labels_copy = NULL;
	int			i,
				j;
	int			extracted_task;
	int			rc = -1;

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

	elog(DEBUG1, "ndb_cuda_knn_train: entry: n_samples=%d, feature_dim=%d, k=%d, task_type=%d",
		 n_samples, feature_dim, k, task_type);

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

	/* Check for integer overflow in memory allocation before copying */
	if (feature_dim > 0 && (size_t) n_samples > MaxAllocSize / sizeof(float) / (size_t) feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: feature array size exceeds MaxAllocSize");
		return -1;
	}
	features_copy = (float *) palloc(sizeof(float) * (size_t) n_samples * (size_t) feature_dim);
	if (features_copy == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: failed to allocate features_copy array");
		return -1;
	}

	labels_copy = (double *) palloc(sizeof(double) * (size_t) n_samples);
	if (labels_copy == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN train: failed to allocate labels_copy array");
		NDB_FREE(features_copy);
		return -1;
	}

	memcpy(features_copy, features, sizeof(float) * (size_t) n_samples * (size_t) feature_dim);
	memcpy(labels_copy, labels, sizeof(double) * (size_t) n_samples);

	/* Build model structure */
	model.n_samples = n_samples;
	model.n_features = feature_dim;
	model.k = k;
	model.task_type = task_type;
	model.features = features_copy;
	model.labels = labels_copy;

	/*
	 * Pack model - pass NULL for metrics if caller doesn't want them to avoid
	 * DirectFunctionCall issues
	 */
	if (ndb_cuda_knn_pack(&model, &blob, metrics ? &metrics_json : NULL, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA KNN model packing failed");
		goto cleanup;
	}

	*model_data = blob;
	if (metrics != NULL)
		*metrics = metrics_json;

	rc = 0;

cleanup:
	if (features_copy)
		NDB_FREE(features_copy);
	if (labels_copy)
		NDB_FREE(labels_copy);

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
	NdbCudaKnnModelHeader *hdr;
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

	/* Validate model_data bytea */
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
	distances = (float *) palloc(sizeof(float) * hdr->n_samples);
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
		NDB_FREE(distances);
		return -1;
	}

	/* Validate computed distances */
	for (i = 0; i < hdr->n_samples; i++)
	{
		if (!isfinite(distances[i]) || distances[i] < 0.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN predict: computed invalid distance");
			NDB_FREE(distances);
			return -1;
		}
	}

	/* Step 2: Find top-k and compute prediction using CUDA */
	if (ndb_cuda_knn_find_top_k(distances, training_labels, hdr->n_samples, hdr->k, hdr->task_type, prediction_out) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("CUDA top-k computation failed");
		NDB_FREE(distances);
		return -1;
	}

	/* Validate prediction output */
	if (!isfinite(*prediction_out))
	{
		if (errstr)
			*errstr = pstrdup("CUDA KNN predict: computed non-finite prediction");
		NDB_FREE(distances);
		return -1;
	}
	/* For classification, prediction should be an integer */
	if (hdr->task_type == 0)
	{
		if (*prediction_out != floor(*prediction_out) || *prediction_out < 0.0)
		{
			if (errstr)
				*errstr = pstrdup("CUDA KNN predict: classification prediction must be non-negative integer");
			NDB_FREE(distances);
			return -1;
		}
	}

	NDB_FREE(distances);
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
	const		NdbCudaKnnModelHeader *hdr;
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

	/* Validate model_data bytea */
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
	const		NdbCudaKnnModelHeader *hdr;
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
	predictions = (int *) palloc(sizeof(int) * (size_t) n_samples);
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
		NDB_FREE(predictions);
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

	NDB_FREE(predictions);

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
