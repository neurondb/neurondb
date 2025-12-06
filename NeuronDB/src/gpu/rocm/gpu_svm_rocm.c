/*-------------------------------------------------------------------------
 *
 * gpu_svm_rocm.c
 *    ROCm backend bridge for Support Vector Machine training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_svm_rocm.c
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
#include "utils/memutils.h"

#include "ml_svm_internal.h"
#include "neurondb_rocm_svm.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

extern int	ndb_rocm_svm_launch_compute_kernel_row(const float *features,
												   int n_samples,
												   int feature_dim,
												   int row_idx,
												   float *kernel_row);

extern int	ndb_rocm_svm_launch_update_errors(const float *kernel_row,
											  float delta_alpha,
											  float label_i,
											  int n_samples,
											  float *errors);

int
ndb_rocm_svm_pack_model(const SVMModel * model,
						bytea * *model_data,
						Jsonb * *metrics,
						char **errstr)
{
	size_t		payload_bytes;
	bytea	   *blob;
	char	   *base;
	NdbCudaSvmModelHeader *hdr;
	float	   *alphas_dest;
	float	   *sv_dest;
	int32	   *indices_dest;
	int			i,
				j;
	size_t		alphas_size;
	size_t		sv_size;
	size_t		indices_size;
	size_t		total_payload;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid SVM model for HIP pack");
		return -1;
	}

	if (model->n_support_vectors <= 0 || model->n_support_vectors > 100000)
	{
		if (errstr)
			*errstr = pstrdup("invalid n_support_vectors for HIP pack");
		return -1;
	}
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		if (errstr)
			*errstr = pstrdup("invalid n_features for HIP pack");
		return -1;
	}
	if (model->n_samples <= 0 || model->n_samples > 10000000)
	{
		if (errstr)
			*errstr = pstrdup("invalid n_samples for HIP pack");
		return -1;
	}

	alphas_size = sizeof(float) * (size_t) model->n_support_vectors;
	sv_size = sizeof(float) * (size_t) model->n_support_vectors * (size_t) model->n_features;
	indices_size = sizeof(int32) * (size_t) model->n_support_vectors;
	total_payload = sizeof(NdbCudaSvmModelHeader) + alphas_size + sv_size + indices_size;

	/* Check against MaxAllocSize */
	if (total_payload > MaxAllocSize || total_payload + VARHDRSZ > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("SVM model too large for HIP pack");
		return -1;
	}

	NDB_DECLARE(bytea *, blob);
	NDB_DECLARE(char *, blob_raw);

	payload_bytes = total_payload;
	NDB_ALLOC(blob_raw, char, VARHDRSZ + payload_bytes);
	blob = (bytea *) blob_raw;
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaSvmModelHeader *) base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->n_support_vectors = model->n_support_vectors;
	hdr->bias = (float) model->bias;
	hdr->C = model->C;
	hdr->max_iters = model->max_iters;

	alphas_dest = (float *) (base + sizeof(NdbCudaSvmModelHeader));
	sv_dest = alphas_dest + model->n_support_vectors;
	indices_dest = (int32 *) (sv_dest
							  + model->n_support_vectors * model->n_features);

	if (model->alphas != NULL && model->support_vectors != NULL
		&& model->support_vector_indices != NULL)
	{
		for (i = 0; i < model->n_support_vectors; i++)
		{
			alphas_dest[i] = (float) model->alphas[i];
			indices_dest[i] = model->support_vector_indices[i];
			for (j = 0; j < model->n_features; j++)
				sv_dest[i * model->n_features + j] =
					model->support_vectors[i
										   * model->n_features
										   + j];
		}
	}

	if (metrics != NULL)
	{
		StringInfoData buf;
		Jsonb	   *metrics_json;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"svm\","
						 "\"storage\":\"gpu\","
						 "\"n_features\":%d,"
						 "\"n_samples\":%d,"
						 "\"n_support_vectors\":%d,"
						 "\"C\":%.6f,"
						 "\"max_iters\":%d}",
						 model->n_features,
						 model->n_samples,
						 model->n_support_vectors,
						 model->C,
						 model->max_iters);

		metrics_json = DatumGetJsonbP(DirectFunctionCall1(
														  jsonb_in, CStringGetTextDatum(buf.data)));
		NDB_FREE(buf.data);
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

int
ndb_rocm_svm_train(const float *features,
				   const double *labels,
				   int n_samples,
				   int feature_dim,
				   const Jsonb * hyperparams,
				   bytea * *model_data,
				   Jsonb * *metrics,
				   char **errstr)
{
	double		C = 1.0;
	int			max_iters = 1000;
	NDB_DECLARE(float *, alphas);
	NDB_DECLARE(float *, errors);
	NDB_DECLARE(float *, kernel_matrix);
	NDB_DECLARE(float *, kernel_row);
	float		bias = 0.0f;
	int			actual_max_iters;
	int			sample_limit;
	int			iter;
	int			num_changed = 0;
	int			examine_all = 1;
	double		eps = 1e-3;
	int			sv_count = 0;
	SVMModel	model;
	int			i,
				j;
	int			rc = -1;
	size_t		alphas_size;
	size_t		errors_size;
	size_t		kernel_matrix_size;
	size_t		kernel_row_size;

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM train: features array is NULL");
		return -1;
	}
	if (labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM train: labels array is NULL");
		return -1;
	}
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM train: model_data output pointer is NULL");
		return -1;
	}
	if (n_samples <= 0 || n_samples > 100000)
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM train: n_samples must be between 1 and 100000");
		return -1;
	}
	if (feature_dim <= 0 || feature_dim > 10000)
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM train: feature_dim must be between 1 and 10000");
		return -1;
	}

	/* Extract hyperparameters */
	if (hyperparams != NULL)
	{
		Datum		C_datum;
		Datum		max_iters_datum;
		Datum		numeric_datum;
		Numeric		num;

		C_datum = DirectFunctionCall2(
									  jsonb_object_field,
									  JsonbPGetDatum(hyperparams),
									  CStringGetTextDatum("C"));
		if (DatumGetPointer(C_datum) != NULL)
		{
			numeric_datum = DirectFunctionCall1(
												jsonb_numeric, C_datum);
			if (DatumGetPointer(numeric_datum) != NULL)
			{
				num = DatumGetNumeric(numeric_datum);
				C = DatumGetFloat8(
								   DirectFunctionCall1(numeric_float8,
													   NumericGetDatum(num)));
				if (C <= 0.0)
					C = 1.0;
				if (C > 1000.0)
					C = 1000.0;
			}
		}

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
					max_iters = 1000;
				if (max_iters > 100000)
					max_iters = 100000;
			}
		}
	}

	/* Limit iterations and samples for large datasets */
	actual_max_iters = (max_iters > 1000 && n_samples > 1000) ? 1000 : max_iters;
	sample_limit = (n_samples > 5000) ? 5000 : n_samples;

	/* Validate input data for NaN/Inf and label domain */
	for (i = 0; i < n_samples && i < sample_limit; i++)
	{
		if (!isfinite(labels[i]))
		{
			if (errstr)
				*errstr = pstrdup("HIP SVM train: non-finite value in labels array");
			return -1;
		}
		/* SVM requires labels to be exactly -1 or 1 */
		if (labels[i] != -1.0 && labels[i] != 1.0)
		{
			if (errstr)
				*errstr = pstrdup("HIP SVM train: labels must be exactly -1.0 or 1.0");
			return -1;
		}
		for (j = 0; j < feature_dim; j++)
		{
			if (!isfinite(features[i * feature_dim + j]))
			{
				if (errstr)
					*errstr = pstrdup("HIP SVM train: non-finite value in features array");
				return -1;
			}
		}
	}

	/* Allocate memory with size validation */
	alphas_size = sizeof(float) * (size_t) sample_limit;
	errors_size = sizeof(float) * (size_t) sample_limit;
	kernel_matrix_size = sizeof(float) * (size_t) sample_limit * (size_t) sample_limit;
	kernel_row_size = sizeof(float) * (size_t) sample_limit;

	if (alphas_size > MaxAllocSize || errors_size > MaxAllocSize ||
		kernel_matrix_size > MaxAllocSize || kernel_row_size > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM train: training size too large");
		return -1;
	}

	NDB_ALLOC(alphas, float, sample_limit);
	NDB_ALLOC(errors, float, sample_limit);
	NDB_ALLOC(kernel_matrix, float, sample_limit * sample_limit);
	NDB_ALLOC(kernel_row, float, sample_limit);

	if (alphas == NULL || errors == NULL || kernel_matrix == NULL || kernel_row == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM train: failed to allocate memory");
		if (alphas)
			NDB_FREE(alphas);
		if (errors)
			NDB_FREE(errors);
		if (kernel_matrix)
			NDB_FREE(kernel_matrix);
		if (kernel_row)
			NDB_FREE(kernel_row);
		return -1;
	}

	/* Pre-compute kernel matrix using GPU */
	for (i = 0; i < sample_limit; i++)
	{
		if (ndb_rocm_svm_launch_compute_kernel_row(features, sample_limit, feature_dim, i, kernel_row) != 0)
		{
			if (errstr)
				*errstr = pstrdup("HIP SVM train: failed to compute kernel matrix");
			NDB_FREE(alphas);
			NDB_FREE(errors);
			NDB_FREE(kernel_matrix);
			NDB_FREE(kernel_row);
			return -1;
		}
		memcpy(kernel_matrix + i * sample_limit, kernel_row, sizeof(float) * (size_t) sample_limit);
	}

	/* Initialize errors: E_i = f(x_i) - y_i, where f(x_i) = 0 initially */
	/* Also initialize alphas to small values to help convergence */
	for (i = 0; i < sample_limit; i++)
	{
		errors[i] = -(float) labels[i];
		/* Initialize alphas to small random values to break symmetry */
		alphas[i] = ((float) (i % 10) + 1.0f) * 0.01f * (float) C;
		if (alphas[i] > (float) C)
			alphas[i] = (float) C * 0.1f;
	}

	/* Simplified SMO: iterate until convergence or max iterations */
	for (iter = 0; iter < actual_max_iters; iter++)
	{
		num_changed = 0;

		/* Simplified update: adjust alphas based on errors */
		for (i = 0; i < sample_limit; i++)
		{
			float		error_i = errors[i];
			float		label_i = (float) labels[i];
			float		alpha_i = alphas[i];
			float		eta;
			float		L = 0.0f;
			float		H = (float) C;
			float		new_alpha_i;
			float		delta_alpha;

			/* Compute eta: second derivative of objective function */
			/* For linear kernel: eta = 2 * K(x_i, x_i) = 2 * ||x_i||^2 */
			eta = 2.0f * kernel_matrix[i * sample_limit + i];

			/* Defensive: ensure eta is positive and reasonable */
			if (eta <= 1e-6f)
				eta = 1.0f;

			/* Update alpha using gradient descent-like approach */
			/* new_alpha = alpha - (error * label) / eta */
			/* But we need to respect KKT conditions */
			if (label_i * error_i < -(float) eps)
			{
				/* Violates KKT: alpha should increase */
				new_alpha_i = alpha_i + (-label_i * error_i) / eta;
			}
			else if (label_i * error_i > (float) eps)
			{
				/* Violates KKT: alpha should decrease */
				new_alpha_i = alpha_i - (label_i * error_i) / eta;
			}
			else
			{
				/* KKT satisfied, no update needed */
				continue;
			}

			/* Clip to bounds [0, C] */
			if (new_alpha_i < L)
				new_alpha_i = L;
			if (new_alpha_i > H)
				new_alpha_i = H;

			/* Only update if change is significant */
			delta_alpha = new_alpha_i - alpha_i;
			if (fabsf(delta_alpha) < (float) eps)
				continue;

			alphas[i] = new_alpha_i;

			/* Update errors using GPU */
			if (ndb_rocm_svm_launch_update_errors(
												  kernel_matrix + i * sample_limit,
												  delta_alpha,
												  label_i,
												  sample_limit,
												  errors) != 0)
			{
				/* Fallback to CPU update */
				for (j = 0; j < sample_limit; j++)
				{
					float		k_val = kernel_matrix[i * sample_limit + j];

					errors[j] -= delta_alpha * label_i * k_val;
				}
			}

			num_changed++;
		}

		/* Update bias (simplified) */
		if (num_changed > 0)
		{
			float		bias_sum = 0.0f;
			int			bias_count = 0;

			for (i = 0; i < sample_limit; i++)
			{
				if (alphas[i] > (float) eps && alphas[i] < ((float) C - (float) eps))
				{
					float		pred = 0.0f;

					for (j = 0; j < sample_limit; j++)
					{
						if (alphas[j] > (float) eps)
						{
							float		k_val = kernel_matrix[j * sample_limit + i];

							pred += alphas[j] * (float) labels[j] * k_val;
						}
					}
					bias_sum += (float) labels[i] - pred;
					bias_count++;
				}
			}
			if (bias_count > 0)
				bias = bias_sum / (float) bias_count;
		}

		if (examine_all)
			examine_all = 0;
		else if (num_changed == 0)
			examine_all = 1;

		if (num_changed == 0 && !examine_all)
			break;
	}

	/* Count support vectors */
	sv_count = 0;
	for (i = 0; i < sample_limit; i++)
	{
		if (alphas[i] > (float) eps)
			sv_count++;
	}

	/* Handle case when no support vectors found */
	if (sv_count == 0)
		sv_count = 1;

	/* Build SVMModel */
	memset(&model, 0, sizeof(model));
	model.n_features = feature_dim;
	model.n_samples = n_samples;
	model.n_support_vectors = sv_count;
	model.bias = (double) bias;
	model.C = C;
	model.max_iters = actual_max_iters;

	/* Allocate support vectors and alphas */
	NDB_DECLARE(double *, model_alphas);
	NDB_DECLARE(float *, model_support_vectors);
	NDB_DECLARE(int *, model_support_vector_indices);
	NDB_ALLOC(model_alphas, double, sv_count);
	NDB_ALLOC(model_support_vectors, float, sv_count * feature_dim);
	NDB_ALLOC(model_support_vector_indices, int, sv_count);
	model.alphas = model_alphas;
	model.support_vectors = model_support_vectors;
	model.support_vector_indices = model_support_vector_indices;

	if (model.alphas == NULL || model.support_vectors == NULL || model.support_vector_indices == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM train: failed to allocate support vectors");
		if (model.alphas)
			NDB_FREE(model.alphas);
		if (model.support_vectors)
			NDB_FREE(model.support_vectors);
		if (model.support_vector_indices)
			NDB_FREE(model.support_vector_indices);
		NDB_FREE(alphas);
		NDB_FREE(errors);
		NDB_FREE(kernel_matrix);
		NDB_FREE(kernel_row);
		return -1;
	}

	/* Copy support vectors */
	{
		int			sv_idx = 0;

		for (i = 0; i < sample_limit && sv_idx < sv_count; i++)
		{
			if (alphas[i] > (float) eps || (sv_count == 1 && sv_idx == 0))
			{
				model.alphas[sv_idx] = (double) alphas[i] * (double) labels[i];
				model.support_vector_indices[sv_idx] = i;
				memcpy(model.support_vectors + sv_idx * feature_dim,
					   features + i * feature_dim,
					   sizeof(float) * feature_dim);
				sv_idx++;
			}
		}
		if (sv_idx == 0)
		{
			/* Fallback: use first sample */
			model.alphas[0] = 1.0 * (double) labels[0];
			model.support_vector_indices[0] = 0;
			memcpy(model.support_vectors, features, sizeof(float) * feature_dim);
		}
	}

	/* Pack model */
	if (ndb_rocm_svm_pack_model(&model, model_data, metrics, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("HIP SVM train: model packing failed");
		if (model.alphas)
			NDB_FREE(model.alphas);
		if (model.support_vectors)
			NDB_FREE(model.support_vectors);
		if (model.support_vector_indices)
			NDB_FREE(model.support_vector_indices);
		NDB_FREE(alphas);
		NDB_FREE(errors);
		NDB_FREE(kernel_matrix);
		NDB_FREE(kernel_row);
		return -1;
	}

	if (model.alphas)
		NDB_FREE(model.alphas);
	if (model.support_vectors)
		NDB_FREE(model.support_vectors);
	if (model.support_vector_indices)
		NDB_FREE(model.support_vector_indices);
	NDB_FREE(alphas);
	NDB_FREE(errors);
	NDB_FREE(kernel_matrix);
	NDB_FREE(kernel_row);

	rc = 0;
	return rc;
}

int
ndb_rocm_svm_predict(const bytea * model_data,
					 const float *input,
					 int feature_dim,
					 int *class_out,
					 double *confidence_out,
					 char **errstr)
{
	const NdbCudaSvmModelHeader *hdr;
	const float *alphas;
	const float *support_vectors;
	const		int32 *indices __attribute__((unused));
	const		bytea *detoasted;
	double		prediction;
	int			i,
				j;
	size_t		expected_size;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || class_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup(
							  "invalid parameters for HIP SVM predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted =
		(const bytea *) PG_DETOAST_DATUM_COPY(PointerGetDatum(model_data));

	hdr = (const NdbCudaSvmModelHeader *) VARDATA(detoasted);

	/* Validate payload size */
	expected_size = sizeof(NdbCudaSvmModelHeader)
		+ sizeof(float) * (size_t) hdr->n_support_vectors
		+ sizeof(float) * (size_t) hdr->n_support_vectors * (size_t) hdr->feature_dim
		+ sizeof(int32) * (size_t) hdr->n_support_vectors;

	if (VARSIZE_ANY_EXHDR(detoasted) < expected_size)
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM predict: model_data too small for expected layout");
		NDB_FREE((void *) detoasted);
		return -1;
	}

	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model "
							   "has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		NDB_FREE((void *) detoasted);
		return -1;
	}

	alphas = (const float *) ((const char *) hdr
							  + sizeof(NdbCudaSvmModelHeader));
	support_vectors = alphas + hdr->n_support_vectors;
	indices = (const int32 *) (support_vectors
							   + hdr->n_support_vectors * hdr->feature_dim);

	/* Compute prediction: f(x) = Σ(alpha_i * y_i * K(x_i, x)) + bias */
	prediction = hdr->bias;
	for (i = 0; i < hdr->n_support_vectors; i++)
	{
		double		kernel_val = 0.0;
		const float *sv = support_vectors + (i * feature_dim);

		/* Linear kernel: K(x_i, x) = x_i · x */
		for (j = 0; j < feature_dim; j++)
			kernel_val += sv[j] * input[j];

		/* Note: y_i is stored explicitly via alpha sign in model building */
		prediction += alphas[i] * kernel_val;
	}

	*class_out = (prediction >= 0.0) ? 1 : 0;
	if (confidence_out != NULL)
		*confidence_out = fabs(prediction);

	/* Free detoasted copy */
	NDB_FREE((void *) detoasted);

	return 0;
}

/*
 * Batch prediction: predict for multiple samples
 */
int
ndb_rocm_svm_predict_batch(const bytea * model_data,
						   const float *features,
						   int n_samples,
						   int feature_dim,
						   int *predictions_out,
						   char **errstr)
{
	const char *base;
	const NdbCudaSvmModelHeader *hdr;
	const		bytea *detoasted;
	int			i;
	int			rc;
	size_t		expected_size;

	if (errstr)
		*errstr = NULL;

	if (model_data == NULL || features == NULL || predictions_out == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid inputs for HIP SVM batch predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted =
		(const bytea *) PG_DETOAST_DATUM_COPY(PointerGetDatum(model_data));

	/* Validate model_data bytea */
	if (VARSIZE_ANY_EXHDR(detoasted) < sizeof(NdbCudaSvmModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM batch predict: model_data too small");
		NDB_FREE((void *) detoasted);
		return -1;
	}

	base = VARDATA_ANY(detoasted);
	hdr = (const NdbCudaSvmModelHeader *) base;

	/* Validate full payload size */
	expected_size = sizeof(NdbCudaSvmModelHeader)
		+ sizeof(float) * (size_t) hdr->n_support_vectors
		+ sizeof(float) * (size_t) hdr->n_support_vectors * (size_t) hdr->feature_dim
		+ sizeof(int32) * (size_t) hdr->n_support_vectors;

	if (VARSIZE_ANY_EXHDR(detoasted) < expected_size)
	{
		if (errstr)
			*errstr = pstrdup("HIP SVM batch predict: model_data too small for expected layout");
		NDB_FREE((void *) detoasted);
		return -1;
	}

	/* Validate model header */
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("HIP SVM batch predict: feature dimension mismatch (expected %d, got %d)",
							   hdr->feature_dim, feature_dim);
		NDB_FREE((void *) detoasted);
		return -1;
	}

	/* Predict for each sample */
	for (i = 0; i < n_samples; i++)
	{
		const float *input = features + (i * feature_dim);
		int			class_out = 0;
		double		confidence_out = 0.0;

		rc = ndb_rocm_svm_predict(detoasted,
								  input,
								  feature_dim,
								  &class_out,
								  &confidence_out,
								  errstr);

		if (rc != 0)
		{
			/* On error, set default prediction */
			predictions_out[i] = 0;
			continue;
		}

		predictions_out[i] = class_out;
	}

	/* Free detoasted copy */
	NDB_FREE((void *) detoasted);

	return 0;
}

/*
 * Batch evaluation: compute metrics for multiple samples
 */
int
ndb_rocm_svm_evaluate_batch(const bytea * model_data,
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
	NDB_DECLARE(int *, predictions);
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
			*errstr = pstrdup("invalid inputs for HIP SVM batch evaluate");
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
	NDB_ALLOC(predictions, int, n_samples);
	if (predictions == NULL)
	{
		if (errstr)
			*errstr = pstrdup("failed to allocate predictions array");
		return -1;
	}

	/* Batch predict */
	rc = ndb_rocm_svm_predict_batch(model_data,
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

#endif							/* NDB_GPU_HIP */
