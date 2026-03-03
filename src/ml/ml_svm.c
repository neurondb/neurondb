/*-------------------------------------------------------------------------
 *
 * ml_svm.c
 *    Support vector machine implementation.
 *
 * This module implements linear SVM for binary classification using a
 * large-margin training algorithm. Models are serialized and stored in
 * the catalog for prediction.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_svm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_svm_internal.h"
#include "ml_catalog.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_svm.h"
#include "neurondb_cuda_svm.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"
#include "neurondb_guc.h"

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif
#endif

#include <math.h>
#include <float.h>

typedef struct SVMDataset
{
	float	   *features;
	double	   *labels;
	int			n_samples;
	int			feature_dim;
}			SVMDataset;

static void svm_dataset_init(SVMDataset * dataset);
static void svm_dataset_free(SVMDataset * dataset);
static void svm_dataset_load(const char *quoted_tbl,
							 const char *quoted_feat,
							 const char *quoted_label,
							 SVMDataset * dataset,
							 MemoryContext oldcontext);
static bytea * svm_model_serialize(const SVMModel * model, uint8 training_backend);
static SVMModel * svm_model_deserialize(const bytea * data, MemoryContext target_context, uint8 * training_backend_out);
static bool svm_metadata_is_gpu(Jsonb * metadata);
static double svm_decode_label_datum(Datum label_datum, Oid label_type_oid);
static bool svm_try_gpu_predict_catalog(int32 model_id,
										const Vector *feature_vec,
										double *result_out);
static bool svm_load_model_from_catalog(int32 model_id, SVMModel * *out);

/*
 * linear_kernel - Compute linear kernel between two vectors
 *
 * Computes the linear kernel (dot product) between two feature vectors.
 * Used in SVM for computing similarity between support vectors and query vectors.
 *
 * Parameters:
 *   x - First feature vector
 *   y - Second feature vector
 *   dim - Dimension of both vectors
 *
 * Returns:
 *   Dot product (x^T * y) as double
 *
 * Notes:
 *   The function validates inputs and reports errors if vectors are NULL or
 *   dimension is invalid. This is the simplest kernel function, equivalent
 *   to the dot product.
 */
static double
linear_kernel(float *x, float *y, int dim)
{
	double		result = 0.0;
	int			i;

	if (x == NULL || y == NULL || dim <= 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: linear_kernel: invalid inputs (x=%p, y=%p, dim=%d)",
						(void *) x,
						(void *) y,
						dim)));
		return 0.0;
	}

	for (i = 0; i < dim; i++)
		result += x[i] * y[i];

	return result;
}

/*
 * RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
 */
static double
__attribute__((unused))
rbf_kernel(float *x, float *y, int dim, double gamma)
{
	double		diff;
	double		dist_sq = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		diff = x[i] - y[i];

		dist_sq += diff * diff;
	}

	return exp(-gamma * dist_sq);
}

/*
 * svm_dataset_init
 */
static void
svm_dataset_init(SVMDataset * dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(SVMDataset));
}

/*
 * svm_dataset_free
 */
static void
svm_dataset_free(SVMDataset * dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
		nfree(dataset->features);
	if (dataset->labels != NULL)
		nfree(dataset->labels);
	memset(dataset, 0, sizeof(SVMDataset));
}

/*
 * svm_dataset_load
 */
static void
svm_dataset_load(const char *quoted_tbl,
				 const char *quoted_feat,
				 const char *quoted_label,
				 SVMDataset * dataset,
				 MemoryContext oldcontext)
{
	ArrayType  *arr = NULL;
	char	   *query_str = NULL;
	HeapTuple	tuple = NULL;
	int			arr_dim;
	int			dim = 0;
	int			i;
	int			nvec = 0;
	int			ret;
	int			valid_idx = 0;
	NdbSpiSession *load_spi_session = NULL;
	Oid			feat_type;
	Oid			label_type;
	size_t		query_len;
	TupleDesc	tupdesc = NULL;
	Vector	   *vec = NULL;
	bool		feat_null;
	bool		label_null;
	Datum		feat_datum;
	Datum		label_datum;

	if (dataset == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm_dataset_load: dataset is NULL"),
				 errdetail("The SVMDataset pointer provided is unexpectedly NULL."),
				 errhint("This indicates an internal programming error. Please report this issue.")));

	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(load_spi_session, oldcontext);

	query_len = strlen("SELECT , FROM  WHERE  IS NOT NULL AND  IS NOT NULL") +
		strlen(quoted_feat) * 2 + strlen(quoted_label) * 2 + strlen(quoted_tbl) + 100;
	nalloc(query_str, char, query_len);
	snprintf(query_str, query_len,
			 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
			 quoted_feat, quoted_label, quoted_tbl, quoted_feat, quoted_label);


	ret = ndb_spi_execute_safe(query_str, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		nfree(query_str);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: svm_dataset_load: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and label columns.")));
	}

	nvec = SPI_processed;
	if (nvec < 10)
	{
		nfree(query_str);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				 errmsg("neurondb: svm_dataset_load: need at least 10 samples for SVM"),
				 errdetail("Dataset contains %d rows, but SVM requires at least 10 rows", nvec),
				 errhint("Add more training data or use a different algorithm for small datasets.")));
	}

	/* Safe access for complex types - validate before access */
	if (nvec > 0 && SPI_tuptable != NULL && SPI_tuptable->vals != NULL &&
		SPI_tuptable->vals[0] != NULL && SPI_tuptable->tupdesc != NULL)
	{
		tuple = SPI_tuptable->vals[0];
		tupdesc = SPI_tuptable->tupdesc;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		feat_type = SPI_gettypeid(tupdesc, 1);

		if (!feat_null)
		{
			if (feat_type == FLOAT4ARRAYOID || feat_type == FLOAT8ARRAYOID)
			{
				arr = DatumGetArrayTypeP(feat_datum);

				if (arr != NULL)
				{
					arr_dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
					dim = arr_dim;
				}
			}
			else
			{
				vec = DatumGetVector(feat_datum);

				if (vec != NULL)
					dim = vec->dim;
			}
		}
	}

	if (dim <= 0)
	{
		nfree(query_str);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm_dataset_load: invalid feature dimension: %d", dim),
				 errdetail("Feature dimension is %d, must be positive", dim),
				 errhint("Ensure the feature column contains valid vector or array data.")));
	}

	MemoryContextSwitchTo(oldcontext);
	{
		double	   *labels_tmp = NULL;
		float	   *features_tmp = NULL;

		nalloc(features_tmp, float, (size_t) nvec * (size_t) dim);
		nalloc(labels_tmp, double, nvec);
		memset(features_tmp, 0, sizeof(float) * (size_t) nvec * (size_t) dim);
		memset(labels_tmp, 0, sizeof(double) * (size_t) nvec);
		dataset->features = features_tmp;
		dataset->labels = labels_tmp;
	}
	dataset->n_samples = nvec;
	dataset->feature_dim = dim;

	if (nvec > 0)
	{
		tupdesc = SPI_tuptable->tupdesc;

		label_type = SPI_gettypeid(tupdesc, 2);
		feat_type = SPI_gettypeid(tupdesc, 1);

		for (i = 0; i < nvec; i++)
		{
			ArrayType  *arr_local = NULL;
			int			arr_dim_local;
			Vector	   *vec_local = NULL;

			/* Safe access to SPI_tuptable - validate before access */
			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
				i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
			{
				continue;
			}
			tuple = SPI_tuptable->vals[i];
			tupdesc = SPI_tuptable->tupdesc;
			if (tupdesc == NULL)
			{
				continue;
			}

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			/* Safe access for label - validate tupdesc has at least 2 columns */
			if (tupdesc->natts < 2)
			{
				continue;
			}
			label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);

			if (feat_null || label_null)
				continue;

			if (feat_type == FLOAT4ARRAYOID || feat_type == FLOAT8ARRAYOID)
			{
				arr_local = DatumGetArrayTypeP(feat_datum);
				arr_dim_local = ArrayGetNItems(ARR_NDIM(arr_local), ARR_DIMS(arr_local));

				if (arr_dim_local != dim)
				{
					nfree(query_str);
					NDB_SPI_SESSION_END(load_spi_session);
					svm_dataset_free(dataset);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: svm_dataset_load: feature dimension mismatch"),
							 errdetail("Expected dimension %d but got %d at row %d", dim, arr_dim_local, i),
							 errhint("Ensure all feature arrays have consistent dimensions.")));
				}

				if (feat_type == FLOAT4ARRAYOID)
				{
					float	   *arr_data = (float *) ARR_DATA_PTR(arr_local);

					memcpy(dataset->features + valid_idx * dim,
						   arr_data,
						   sizeof(float) * dim);
				}
				else
				{
					double	   *arr_data = (double *) ARR_DATA_PTR(arr_local);

					for (int j = 0; j < dim; j++)
						dataset->features[valid_idx * dim + j] = (float) arr_data[j];
				}
			}
			else
			{
				vec_local = DatumGetVector(feat_datum);

				if (vec_local == NULL)
				{
					continue;
				}

				if (vec_local->dim != dim)
				{
					nfree(query_str);
					NDB_SPI_SESSION_END(load_spi_session);
					svm_dataset_free(dataset);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: svm_dataset_load: feature dimension mismatch"),
							 errdetail("Expected dimension %d but got %d at row %d", dim, vec_local->dim, i),
							 errhint("Ensure all feature vectors have the same dimension.")));
				}

				memcpy(dataset->features + valid_idx * dim,
					   vec_local->data,
					   sizeof(float) * dim);
			}

			if (label_type == INT2OID)
				dataset->labels[valid_idx] = (double) DatumGetInt16(label_datum);
			else if (label_type == INT4OID)
				dataset->labels[valid_idx] = (double) DatumGetInt32(label_datum);
			else if (label_type == INT8OID)
				dataset->labels[valid_idx] = (double) DatumGetInt64(label_datum);
			else
				dataset->labels[valid_idx] = DatumGetFloat8(label_datum);

			valid_idx++;
		}

		if (valid_idx == 0)
		{
			nfree(query_str);
			NDB_SPI_SESSION_END(load_spi_session);
			svm_dataset_free(dataset);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: svm_dataset_load: no valid rows found"),
					 errdetail("All %d rows in table '%s' had NULL features or labels", nvec, quoted_tbl),
					 errhint("Ensure the feature and label columns contain non-NULL values.")));
		}

		dataset->n_samples = valid_idx;
	}

	nfree(query_str);
	NDB_SPI_SESSION_END(load_spi_session);
}

/*
 * svm_model_serialize
 */
static bytea *
svm_model_serialize(const SVMModel * model, uint8 training_backend)
{
	StringInfoData buf;
	int			i;

	if (model == NULL)
		return NULL;

	/* Validate model before serialization */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("svm_model_serialize: invalid n_features %d (corrupted model?)",
						model->n_features)));
	}

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: svm_model_serialize: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	pq_begintypsend(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) */
	pq_sendbyte(&buf, training_backend);

	pq_sendint32(&buf, model->model_id);
	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendint32(&buf, model->n_support_vectors);
	pq_sendfloat8(&buf, model->bias);
	pq_sendfloat8(&buf, model->C);
	pq_sendint32(&buf, model->max_iters);

	if (model->alphas != NULL && model->n_support_vectors > 0)
	{
		for (i = 0; i < model->n_support_vectors; i++)
			pq_sendfloat8(&buf, model->alphas[i]);
	}

	if (model->support_vectors != NULL && model->n_support_vectors > 0
		&& model->n_features > 0)
	{
		for (i = 0; i < model->n_support_vectors * model->n_features;
			 i++)
			pq_sendfloat4(&buf, model->support_vectors[i]);
	}

	if (model->support_vector_indices != NULL
		&& model->n_support_vectors > 0)
	{
		for (i = 0; i < model->n_support_vectors; i++)
			pq_sendint32(&buf, model->support_vector_indices[i]);
	}

	if (model->support_labels != NULL && model->n_support_vectors > 0)
	{
		for (i = 0; i < model->n_support_vectors; i++)
			pq_sendfloat8(&buf, model->support_labels[i]);
	}

	return pq_endtypsend(&buf);
}

/*
 * svm_model_deserialize
 */
static SVMModel *
svm_model_deserialize(const bytea * data, MemoryContext target_context, uint8 * training_backend_out)
{
	StringInfoData buf;
	SVMModel *model = NULL;
	int			i;
	MemoryContext oldcontext;
	uint8		training_backend = 0;

	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	/* Read training_backend first */
	training_backend = (uint8) pq_getmsgbyte(&buf);

	oldcontext = MemoryContextSwitchTo(target_context);

	nalloc(model, SVMModel, 1);
	memset(model, 0, sizeof(SVMModel));

	model->model_id = pq_getmsgint(&buf, 4);
	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->n_support_vectors = pq_getmsgint(&buf, 4);
	model->bias = pq_getmsgfloat8(&buf);
	model->C = pq_getmsgfloat8(&buf);
	model->max_iters = pq_getmsgint(&buf, 4);

	if (model->n_features <= 0 || model->n_features > 10000)
	{
		nfree(model);
		MemoryContextSwitchTo(oldcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: invalid n_features %d in deserialized model (corrupted data?)",
						model->n_features)));
	}
	if (model->n_support_vectors < 0 || model->n_support_vectors > 100000)
	{
		nfree(model);
		MemoryContextSwitchTo(oldcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: invalid n_support_vectors %d in deserialized model (corrupted data?)",
						model->n_support_vectors)));
	}

	{
		double	   *alphas = NULL;
		double	   *support_labels = NULL;
		float	   *support_vectors = NULL;
		int		   *support_vector_indices = NULL;

		if (model->n_support_vectors > 0)
		{
			nalloc(alphas, double, model->n_support_vectors);
			model->alphas = alphas;
			for (i = 0; i < model->n_support_vectors; i++)
				model->alphas[i] = pq_getmsgfloat8(&buf);
		}
		else
		{
			model->alphas = NULL;
		}

		if (model->n_support_vectors > 0 && model->n_features > 0)
		{
			nalloc(support_vectors, float,
				   (size_t) model->n_support_vectors * (size_t) model->n_features);
			model->support_vectors = support_vectors;
			for (i = 0; i < model->n_support_vectors * model->n_features;
				 i++)
				model->support_vectors[i] = pq_getmsgfloat4(&buf);
		}
		else
		{
			model->support_vectors = NULL;
		}

		if (model->n_support_vectors > 0)
		{
			nalloc(support_vector_indices, int, model->n_support_vectors);
			model->support_vector_indices = support_vector_indices;
			for (i = 0; i < model->n_support_vectors; i++)
				model->support_vector_indices[i] =
					pq_getmsgint(&buf, 4);
		}
		else
		{
			model->support_vector_indices = NULL;
		}

		if (model->n_support_vectors > 0)
		{
			nalloc(support_labels, double, model->n_support_vectors);
			model->support_labels = support_labels;
			for (i = 0; i < model->n_support_vectors; i++)
				model->support_labels[i] = pq_getmsgfloat8(&buf);
		}
		else
		{
			model->support_labels = NULL;
		}
	}

	/* Return training_backend if output parameter provided */
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;

	MemoryContextSwitchTo(oldcontext);

	return model;
}

/*
 * svm_metadata_is_gpu
 *
 * Checks if a model's metadata indicates it's a GPU-trained model.
 * Now checks for training_backend integer (1=GPU, 0=CPU) instead of "storage" string.
 */
static bool
svm_metadata_is_gpu(Jsonb * metadata)
{
	bool		is_gpu = false;
	int			backend;
	JsonbIterator *it = NULL;
	JsonbIteratorToken r;
	JsonbValue	v;

	if (metadata == NULL)
		return false;

	/* Check for training_backend integer in metrics */
	it = JsonbIteratorInit((JsonbContainer *) & metadata->root);
	while ((r = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
	{
		if (r == WJB_KEY && v.type == jbvString)
		{
			char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

			if (strcmp(key, "training_backend") == 0)
			{
				r = JsonbIteratorNext(&it, &v, true);
				if (r == WJB_VALUE && v.type == jbvNumeric)
				{
					backend = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));

					is_gpu = (backend == 1);
				}
			}
			nfree(key);
		}
	}

	return is_gpu;
}

/*
 * svm_decode_label_datum
 *
 * Decode a label datum based on its PostgreSQL type OID, similar to
 * how svm_dataset_load handles different label types.
 */
static double
svm_decode_label_datum(Datum label_datum, Oid label_type_oid)
{
	if (label_type_oid == INT2OID)
		return (double) DatumGetInt16(label_datum);
	else if (label_type_oid == INT4OID)
		return (double) DatumGetInt32(label_datum);
	else if (label_type_oid == INT8OID)
		return (double) DatumGetInt64(label_datum);
	else
		return DatumGetFloat8(label_datum);
}

/*
 * svm_try_gpu_predict_catalog
 */
static bool
svm_try_gpu_predict_catalog(int32 model_id,
							const Vector *feature_vec,
							double *result_out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	char *gpu_err = NULL;
	double		prediction = 0.0;
	bool		success = false;

	/* Check compute mode - only try GPU if compute mode allows it */
	if (!NDB_SHOULD_TRY_GPU())
		return false;

	if (!neurondb_gpu_is_available())
		return false;
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
		goto cleanup;

	/* Check if this is a GPU model - either by metrics or by payload format */
	{
		bool		is_gpu_model = false;
		uint32		payload_size;

		/* First check metrics for training_backend */
		if (svm_metadata_is_gpu(metrics))
		{
			is_gpu_model = true;
		}
		else
		{
			/* If metrics check didn't find GPU indicator, check payload format */
			/* GPU models start with NdbCudaSvmModelHeader, CPU models start with uint8 training_backend */
			payload_size = VARSIZE(payload) - VARHDRSZ;
			
			/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
			/* GPU format: first field is feature_dim (int32) */
			/* Check if payload looks like GPU format (starts with int32, not uint8) */
			if (payload_size >= sizeof(int32))
			{
				const int32 *first_int = (const int32 *) VARDATA(payload);
				int32		first_value = *first_int;
				
				/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
				if (first_value > 0 && first_value <= 100000)
				{
					/* Check if payload size matches GPU format */
					if (payload_size >= sizeof(NdbCudaSvmModelHeader))
					{
						const NdbCudaSvmModelHeader *hdr = (const NdbCudaSvmModelHeader *) VARDATA(payload);
						
						/* Validate header fields match the first int32 */
						if (hdr->feature_dim == first_value &&
							hdr->n_samples >= 0 && hdr->n_samples <= 1000000000 &&
							hdr->n_support_vectors >= 0 && hdr->n_support_vectors <= 1000000)
						{
							size_t		expected_gpu_size = sizeof(NdbCudaSvmModelHeader) +
								sizeof(float) * (size_t) hdr->n_support_vectors +
								sizeof(float) * (size_t) hdr->n_support_vectors * (size_t) hdr->feature_dim +
								sizeof(int32) * (size_t) hdr->n_support_vectors;
							
							/* Size matches GPU format - likely a GPU model */
							if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
							{
								is_gpu_model = true;
							}
						}
					}
				}
			}
		}

		if (!is_gpu_model)
			goto cleanup;
	}

	if (ndb_gpu_svm_predict_double(payload,
								   feature_vec->data,
								   feature_vec->dim,
								   &prediction,
								   &gpu_err)
		== 0)
	{
		if (result_out != NULL)
			*result_out = prediction;
		success = true;
	}

cleanup:
	if (payload != NULL)
	{
		nfree(payload);
	}
	if (metrics != NULL)
		nfree(metrics);
	if (gpu_err != NULL)
		nfree(gpu_err);

	return success;
}

/*
 * svm_load_model_from_catalog
 */
static bool
svm_load_model_from_catalog(int32 model_id, SVMModel * *out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	SVMModel *decoded = NULL;
	MemoryContext oldcontext;

	if (out == NULL)
		return false;

	*out = NULL;

	oldcontext = CurrentMemoryContext;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
	{
		if (metrics != NULL)
			nfree(metrics);
		return false;
	}

	/* Skip GPU models - they should be handled by GPU prediction */
	/* Check both metrics and payload format to determine if this is a GPU model */
	{
		bool		is_gpu_model = false;
		uint32		payload_size;

		/* First check metrics for training_backend */
		if (svm_metadata_is_gpu(metrics))
		{
			is_gpu_model = true;
		}
		else
		{
			/* If metrics check didn't find GPU indicator, check payload format */
			/* GPU models start with NdbCudaSvmModelHeader, CPU models start with uint8 training_backend */
			payload_size = VARSIZE(payload) - VARHDRSZ;
			
			/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
			/* GPU format: first field is feature_dim (int32) */
			/* Check if payload looks like GPU format (starts with int32, not uint8) */
			if (payload_size >= sizeof(int32))
			{
				const int32 *first_int = (const int32 *) VARDATA(payload);
				int32		first_value = *first_int;
				
				/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
				if (first_value > 0 && first_value <= 100000)
				{
					/* Check if payload size matches GPU format */
					if (payload_size >= sizeof(NdbCudaSvmModelHeader))
					{
						const NdbCudaSvmModelHeader *hdr = (const NdbCudaSvmModelHeader *) VARDATA(payload);
						
						/* Validate header fields match the first int32 */
						if (hdr->feature_dim == first_value &&
							hdr->n_samples >= 0 && hdr->n_samples <= 1000000000 &&
							hdr->n_support_vectors >= 0 && hdr->n_support_vectors <= 1000000)
						{
							size_t		expected_gpu_size = sizeof(NdbCudaSvmModelHeader) +
								sizeof(float) * (size_t) hdr->n_support_vectors +
								sizeof(float) * (size_t) hdr->n_support_vectors * (size_t) hdr->feature_dim +
								sizeof(int32) * (size_t) hdr->n_support_vectors;
							
							/* Size matches GPU format - likely a GPU model */
							if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
							{
								is_gpu_model = true;
							}
						}
					}
				}
			}
		}

		if (is_gpu_model)
		{
			if (payload != NULL)
			{
				nfree(payload);
			}
			if (metrics != NULL)
				nfree(metrics);
			return false;
		}
	}

	decoded = svm_model_deserialize(payload, oldcontext, NULL);

	if (payload != NULL)
	{
		nfree(payload);
	}
	if (metrics != NULL)
		nfree(metrics);

	if (decoded == NULL)
		return false;

	*out = decoded;
	return true;
}

/*
 * train_svm_classifier
 *
 * Trains a linear SVM using heuristic large-margin algorithm
 * Returns model_id
 */
PG_FUNCTION_INFO_V1(train_svm_classifier);

Datum
train_svm_classifier(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *feature_col = NULL;
	text *label_col = NULL;
	double		c_param;
	int			max_iters;

	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *label_str = NULL;
	MemoryContext oldcontext;
	int			nvec = 0;
	int			dim = 0;
	int			i;
	int			j;
	SVMDataset	dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_label;
	MLGpuTrainResult gpu_result;
	char *gpu_err = NULL;
	Jsonb *gpu_hyperparams = NULL;
	int32		model_id = 0;
	SVMModel	model;

	/* Initialize gpu_result to zero to avoid undefined behavior */
	memset(&gpu_result, 0, sizeof(MLGpuTrainResult));

	if (PG_NARGS() < 3 || PG_NARGS() > 5)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: train_svm_classifier requires 3-5 arguments, got %d",
						PG_NARGS()),
				 errhint("Usage: "
						 "train_svm_classifier(table_name, "
						 "feature_col, label_col, [C], "
						 "[max_iters])")));
	}

	/* Get required arguments */
	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: table_name, feature_col, and "
						"label_col are required")));

	/* Extract input parameters */
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);

	/* Get optional parameters */
	c_param = PG_NARGS() >= 4 && !PG_ARGISNULL(3) ? PG_GETARG_FLOAT8(3) : 1.0;
	max_iters = PG_NARGS() >= 5 && !PG_ARGISNULL(4) ? PG_GETARG_INT32(4) : 1000;

	/* Convert text arguments to C strings */
	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col); /* Validate strings are not empty */
	if (tbl_str == NULL || strlen(tbl_str) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: table_name cannot be empty")));

	if (feat_str == NULL || strlen(feat_str) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: feature_col cannot be empty")));

	if (label_str == NULL || strlen(label_str) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: label_col cannot be empty")));

	/* Initialize dataset */
	svm_dataset_init(&dataset);

	/* Save current memory context before SPI operations */
	oldcontext = CurrentMemoryContext;

	/*
	 * Load training data via SPI - svm_dataset_load handles SPI
	 * connect/disconnect
	 */
	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_label = quote_identifier(label_str);

	svm_dataset_load(quoted_tbl, quoted_feat, quoted_label, &dataset, oldcontext);

	nvec = dataset.n_samples;
	dim = dataset.feature_dim;

	/* Validate dataset */
	if (nvec < 10)
	{
		svm_dataset_free(&dataset);

		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: need at least 10 samples for training, got %d",
						nvec)));
	}

	if (dim <= 0 || dim > 10000)
	{
		svm_dataset_free(&dataset);




		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: invalid feature dimension %d (must be in range [1, 10000])",
						dim)));
	}

	/* Validate labels are binary - check for at least two distinct values */
	{
		double		first_label;
		int			n_class0 = 0;
		int			n_class1 = 0;
		bool		found_two_classes = false;
		
		first_label = dataset.labels[0];

		/* Find first distinct label value */
		for (i = 0; i < nvec; i++)
		{
			if (fabs(dataset.labels[i] - first_label) > 1e-6)
			{
				found_two_classes = true;
				break;
			}
		}

		if (!found_two_classes)
		{
			svm_dataset_free(&dataset);

			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: svm: all labels have the same value (%.6f), need at least two classes",
							first_label)));
		}

		/* Normalize labels to {-1, 1} for SVM math */
		/* CRITICAL: This normalization MUST happen before GPU training */
		/* Always normalize - convert <=0.5 to -1.0, >0.5 to 1.0 */
		for (i = 0; i < nvec; i++)
		{
			double old_val = dataset.labels[i];
			if (dataset.labels[i] <= 0.5)
				dataset.labels[i] = -1.0;
			else
				dataset.labels[i] = 1.0;
			
			/* Fail immediately if normalization didn't work */
			if (fabs(dataset.labels[i] + 1.0) > 1e-6 && fabs(dataset.labels[i] - 1.0) > 1e-6)
			{
				svm_dataset_free(&dataset);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: svm: label normalization failed - label[%d] was %.15f, became %.15f (not -1.0 or 1.0)",
								i, old_val, dataset.labels[i]),
						 errhint("This indicates a bug in label normalization code.")));
			}
		}

		/* Count samples in each class after normalization */
		for (i = 0; i < nvec; i++)
		{
			if (dataset.labels[i] < 0.0)
				n_class0++;
			else
				n_class1++;
		}

		if (n_class0 == 0 || n_class1 == 0)
		{
			svm_dataset_free(&dataset);




			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: svm: labels must contain both classes (negative=%d, positive=%d)",
							n_class0,
							n_class1)));
		}

	}

	/* Try GPU training first */
	if (neurondb_gpu_is_available() && nvec > 0 && dim > 0)
	{
		/* Create hyperparameters JSONB using JSONB API */
		{
			JsonbParseState *state = NULL;
			JsonbValue	jkey;
			JsonbValue	jval;

			JsonbValue *final_value = NULL;
			Numeric		C_num,
						max_iters_num;

			PG_TRY();
			{
				(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

				/* Add C */
				jkey.type = jbvString;
				jkey.val.string.val = "C";
				jkey.val.string.len = strlen("C");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				C_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(c_param)));
				jval.type = jbvNumeric;
				jval.val.numeric = C_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add max_iters */
				jkey.val.string.val = "max_iters";
				jkey.val.string.len = strlen("max_iters");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_iters)));
				jval.type = jbvNumeric;
				jval.val.numeric = max_iters_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

				if (final_value == NULL)
				{
					elog(ERROR, "neurondb: train_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL for hyperparameters");
				}

				gpu_hyperparams = JsonbValueToJsonb(final_value);
			}
			PG_CATCH();
			{
				ErrorData  *edata = CopyErrorData();

				elog(ERROR, "neurondb: train_svm: hyperparameters JSONB construction failed: %s", edata->message);
				FlushErrorState();
				gpu_hyperparams = NULL;
			}
			PG_END_TRY();
		}

		if (gpu_hyperparams == NULL)
		{
		}
		else
		{
			/* Verify labels are normalized before GPU call - this is critical */
			/* Check a few labels to ensure normalization worked */
			for (i = 0; i < nvec && i < 10; i++)
			{
				if (fabs(dataset.labels[i] + 1.0) > 1e-6 && fabs(dataset.labels[i] - 1.0) > 1e-6)
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: svm: label normalization failed - label[%d]=%.15f is not -1.0 or 1.0",
									i, dataset.labels[i]),
							 errhint("This indicates a bug in label normalization. All labels must be normalized to -1.0 or 1.0 before GPU training.")));
				}
			}
			if (ndb_gpu_try_train_model("svm",
										 NULL,
										 NULL,
										 tbl_str,
										 label_str,
										 NULL,
										 0,
										 gpu_hyperparams,
										 dataset.features,
										 dataset.labels,
										 nvec,
										 dim,
										 2,
										 &gpu_result,
										 &gpu_err)
				 && gpu_result.spec.model_data != NULL)
			{
			/* GPU training succeeded - serialize already converted to unified format */
			MLCatalogModelSpec spec;


			spec = gpu_result.spec;
			if (spec.training_table == NULL)
				spec.training_table = tbl_str;
			if (spec.training_column == NULL)
				spec.training_column = label_str;
			if (spec.parameters == NULL)
			{
				spec.parameters = gpu_hyperparams;
				gpu_hyperparams = NULL;
			}
			spec.algorithm = "svm";
			spec.model_type = "classification";
			spec.training_time_ms = -1;
			spec.num_samples = nvec;
			spec.num_features = dim;

			model_id = ml_catalog_register_model(&spec);

			if (model_id <= 0)
			{
				ndb_gpu_free_train_result(&gpu_result);
				svm_dataset_free(&dataset);

				if (gpu_hyperparams)
					nfree(gpu_hyperparams);

				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: svm: failed to register GPU model in catalog")));
			}

			/*
			 * Success! Catalog took ownership of model_data. Let memory
			 * context handle cleanup of all allocations automatically.
			 */

			PG_RETURN_INT32(model_id);
		}
		else
		{
			/* GPU training failed - check if GPU mode is forced */
			if (NDB_REQUIRE_GPU())
			{
				/* Strict GPU mode: error out, no CPU fallback */
				char *error_msg = NULL;

				if (gpu_err != NULL)
				{
					error_msg = pstrdup(gpu_err);
					nfree(gpu_err);
					gpu_err = NULL;
				}
				else
				{
					error_msg = pstrdup("GPU training failed");
				}

				/* Clean up GPU result if it was partially initialized */
				if (gpu_result.spec.model_data != NULL || gpu_result.spec.metrics != NULL ||
					gpu_result.spec.algorithm != NULL || gpu_result.spec.training_table != NULL ||
					gpu_result.spec.training_column != NULL || gpu_result.payload != NULL)
				{
					ndb_gpu_free_train_result(&gpu_result);
				}
				else
				{
					/* Just zero it to be safe */
					memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
				}

				if (gpu_hyperparams != NULL)
				{
					nfree(gpu_hyperparams);
					gpu_hyperparams = NULL;
				}

				svm_dataset_free(&dataset);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(label_str);

				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: svm: GPU training failed - GPU mode requires GPU to be available"),
						 errdetail("%s", error_msg),
						 errhint("Check GPU availability and model compatibility, or set compute_mode='auto' for automatic CPU fallback.")));
			}

			/* AUTO mode: cleanup and fall back to CPU */
			if (gpu_err != NULL)
			{
				nfree(gpu_err);
				gpu_err = NULL;
			}
			else
			{
			}

			/* Clean up GPU result if it was partially initialized */
			if (gpu_result.spec.model_data != NULL || gpu_result.spec.metrics != NULL ||
				gpu_result.spec.algorithm != NULL || gpu_result.spec.training_table != NULL ||
				gpu_result.spec.training_column != NULL || gpu_result.payload != NULL)
			{
				ndb_gpu_free_train_result(&gpu_result);
			}
			else
			{
				/* Just zero it to be safe */
				memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
			}

			if (gpu_hyperparams != NULL)
			{
				nfree(gpu_hyperparams);
				gpu_hyperparams = NULL;
			}
		}
	}
	}

	/* Fall back to CPU training (only in AUTO mode) */

	/* CPU training implementation */
	{
		double		bias = 0.0;
		int			actual_max_iters;
		int			sample_limit;
		int			kernel_limit;
		int			iter;
		int			num_changed = 0;
		int			examine_all = 1;
		double		eps = 1e-3;
		int			sv_count = 0;
		double *alphas = NULL;
		double *errors = NULL;
		bytea *serialized = NULL;
		MLCatalogModelSpec spec;
		Jsonb *params_jsonb = NULL;
		Jsonb *metrics_jsonb = NULL;
		int			correct = 0;
		double		accuracy = 0.0;

		/* Limit iterations and samples for large datasets */
		actual_max_iters =
			(max_iters > 1000 && nvec > 1000) ? 1000 : max_iters;
		sample_limit = (nvec > 5000) ? 5000 : nvec;
		kernel_limit = (sample_limit > 1000) ? 1000 : sample_limit;


		/* Allocate memory for heuristic training algorithm */
		nalloc(alphas, double, sample_limit);
		nalloc(errors, double, sample_limit);

		/* Initialize errors: E_i = f(x_i) - y_i, where f(x_i) = 0 initially */
		/* Also initialize alphas to small values to help convergence */
		for (i = 0; i < sample_limit && i < nvec; i++)
		{
			errors[i] =
				-dataset.labels
				[i];			/* f(x_i) = 0 initially, so E_i = 0 - y_i =
								 * -y_i */
			/* Initialize alphas to small random values to break symmetry */
			alphas[i] = ((double) (i % 10) + 1.0) * 0.01 * c_param;
			if (alphas[i] > c_param)
				alphas[i] = c_param * 0.1;
		}

		/* Heuristic training: iterate until convergence or max iterations */
		for (iter = 0; iter < actual_max_iters; iter++)
		{
			num_changed = 0;

			if (examine_all)
			{
				for (i = 0; i < sample_limit && i < nvec; i++)
				{
					/* Update sample i */
					{
						double		error_i = errors[i];
						double		label_i = dataset.labels[i];
						double		alpha_i = alphas[i];
						double		eta;
						double		L = 0.0;
						double		H = c_param;
						double		new_alpha_i = 0.0;
						double		delta_alpha;

						/*
						 * Compute eta: second derivative of objective
						 * function
						 */
						eta = 2.0 * linear_kernel(dataset.features + i * dim,
												  dataset.features + i * dim, dim);
						if (eta <= 1e-10)
							eta = 1.0;

						/* Update alpha using gradient descent-like approach */
						if ((label_i * error_i < -eps && alpha_i < c_param) ||
							(label_i * error_i > eps && alpha_i > 0.0))
						{
							new_alpha_i = alpha_i - (error_i / eta);
							new_alpha_i = (new_alpha_i < L) ? L : (new_alpha_i > H) ? H : new_alpha_i;
							delta_alpha = new_alpha_i - alpha_i;

							if (fabs(delta_alpha) > eps)
							{
								alphas[i] = new_alpha_i;
								/* Update errors for all other samples */
								for (j = 0; j < sample_limit && j < nvec; j++)
								{
									double		kernel_val;

									if (j == i)
										continue;
									kernel_val = linear_kernel(
															   dataset.features + i * dim,
															   dataset.features + j * dim,
															   dim);
									errors[j] -= delta_alpha * label_i * kernel_val;
								}
								num_changed++;
							}
						}
					}
				}
			}
			else
			{
				for (i = 0; i < sample_limit && i < nvec; i++)
				{
					if (alphas[i] > eps && alphas[i] < (c_param - eps))
					{
						/* Update sample i */
						{
							double		error_i = errors[i];
							double		label_i = dataset.labels[i];
							double		alpha_i = alphas[i];
							double		eta;
							double		L = 0.0;
							double		H = c_param;
							double		new_alpha_i = 0.0;
							double		delta_alpha;

							/*
							 * Compute eta: second derivative of objective
							 * function
							 */
							eta = 2.0 * linear_kernel(dataset.features + i * dim,
													  dataset.features + i * dim, dim);
							if (eta <= 1e-10)
								eta = 1.0;

							/*
							 * Update alpha using gradient descent-like
							 * approach
							 */
							if ((label_i * error_i < -eps && alpha_i < c_param) ||
								(label_i * error_i > eps && alpha_i > 0.0))
							{
								new_alpha_i = alpha_i - (error_i / eta);
								new_alpha_i = (new_alpha_i < L) ? L : (new_alpha_i > H) ? H : new_alpha_i;
								delta_alpha = new_alpha_i - alpha_i;

								if (fabs(delta_alpha) > eps)
								{
									alphas[i] = new_alpha_i;
									/* Update errors for all other samples */
									for (j = 0; j < sample_limit && j < nvec; j++)
									{
										double		kernel_val;

										if (j == i)
											continue;
										kernel_val = linear_kernel(
																   dataset.features + i * dim,
																   dataset.features + j * dim,
																   dim);
										errors[j] -= delta_alpha * label_i * kernel_val;
									}
									num_changed++;
								}
							}
						}
					}
				}
			}

			if (examine_all)
				examine_all = 0;
			else if (num_changed == 0)
				examine_all = 1;

			if (num_changed == 0)
				break;

			/* Update bias after changes */
			{
				double		bias_sum = 0.0;
				int			bias_count = 0;

				for (i = 0; i < sample_limit && i < nvec; i++)
				{
					if (alphas[i] > eps && alphas[i] < (c_param - eps))
					{
						double		pred = 0.0;

						for (j = 0; j < kernel_limit && j < nvec; j++)
						{
							pred += alphas[j] * dataset.labels[j]
								* linear_kernel(dataset.features + j * dim,
												dataset.features + i * dim, dim);
						}
						bias_sum += dataset.labels[i] - pred;
						bias_count++;
					}
				}
				if (bias_count > 0)
					bias = bias_sum / bias_count;
			}
		}

		/* Count support vectors */
		sv_count = 0;
		for (i = 0; i < sample_limit && i < nvec; i++)
		{
			if (alphas[i] > eps)
				sv_count++;
		}


		/* Validate dim before building model */
		if (dim <= 0 || dim > 10000)
		{
			if (alphas)
				nfree(alphas);
			if (errors)
				nfree(errors);
			svm_dataset_free(&dataset);




			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: svm: invalid feature dimension %d before model serialization",
							dim)));
		}

		/* Build SVMModel */
		memset(&model, 0, sizeof(model));
		model.n_features = dim;
		model.n_samples = nvec;
		model.bias = bias;
		model.C = c_param;
		model.max_iters = actual_max_iters;


		/* Handle case when no support vectors found */
		if (sv_count == 0)
		{
			sv_count = 1;
			model.n_support_vectors = sv_count;

			/* Allocate support vectors and alphas */
			{
				double *alphas_local = NULL;
				int32 *support_vector_indices = NULL;
				double *support_labels = NULL;
				float *support_vectors = NULL;

				nalloc(alphas_local, double,
												 sizeof(double) * (size_t) sv_count);
				nalloc(support_vectors, float,
														 sizeof(float) * (size_t) sv_count * (size_t) dim);
				nalloc(support_vector_indices, int32, sv_count);
				nalloc(support_labels, double, sv_count);
				model.support_vector_indices = support_vector_indices;
				model.support_labels = support_labels;
				model.alphas = alphas_local;
				model.support_vectors = support_vectors;
			}

			if (model.alphas == NULL
				|| model.support_vectors == NULL
				|| model.support_vector_indices == NULL
				|| model.support_labels == NULL)
			{
				if (model.alphas)
					nfree(model.alphas);
				if (model.support_vectors)
					nfree(model.support_vectors);
				if (model.support_vector_indices)
					nfree(model.support_vector_indices);
				if (model.support_labels)
					nfree(model.support_labels);
				if (alphas)
					nfree(alphas);
				if (errors)
					nfree(errors);
				svm_dataset_free(&dataset);




				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("neurondb: svm: failed to "
								"allocate memory for "
								"support vectors")));
			}

			/* Create default support vector using first sample */
			model.alphas[0] = 1.0;
			model.support_vector_indices[0] = 0;
			model.support_labels[0] = 1.0;	/* pick positive side */
			/* Set bias to a small positive value to ensure predictions work */
			model.bias = 0.1;
			if (nvec > 0 && dim > 0)
			{
				memcpy(model.support_vectors,
					   dataset.features,
					   sizeof(float) * (size_t) dim);
			}
		}
		else
		{
			model.n_support_vectors = sv_count;

			/* Allocate support vectors and alphas */
			{
				double *alphas_local = NULL;
				float *support_vectors = NULL;
				int32 *support_vector_indices = NULL;
				double *support_labels = NULL;

				/* Ensure model fields are NULL before allocation */
				if (model.alphas != NULL)
					nfree(model.alphas);
				if (model.support_vectors != NULL)
					nfree(model.support_vectors);
				if (model.support_vector_indices != NULL)
					nfree(model.support_vector_indices);
				if (model.support_labels != NULL)
					nfree(model.support_labels);

				nalloc(alphas_local, double,
												 sizeof(double) * (size_t) sv_count);
				nalloc(support_vectors, float,
														 sizeof(float) * (size_t) sv_count * (size_t) dim);
				nalloc(support_vector_indices, int32, sv_count);
				nalloc(support_labels, double, sv_count);
				model.support_vector_indices = support_vector_indices;
				model.support_labels = support_labels;
				model.alphas = alphas_local;
				model.support_vectors = support_vectors;
			}

			if (model.alphas == NULL
				|| model.support_vectors == NULL
				|| model.support_vector_indices == NULL
				|| model.support_labels == NULL)
			{
				if (model.alphas)
					nfree(model.alphas);
				if (model.support_vectors)
					nfree(model.support_vectors);
				if (model.support_vector_indices)
					nfree(model.support_vector_indices);
				if (model.support_labels)
					nfree(model.support_labels);
				if (alphas)
					nfree(alphas);
				if (errors)
					nfree(errors);
				svm_dataset_free(&dataset);




				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("neurondb: svm: failed to "
								"allocate memory for "
								"support vectors")));
			}

			/* Copy support vectors */
			{
				int			sv_idx = 0;

				for (i = 0; i < sample_limit && i < nvec
					 && sv_idx < sv_count;
					 i++)
				{
					if (alphas[i] > eps)
					{
						model.alphas[sv_idx] =
							alphas[i];
						model.support_vector_indices
							[sv_idx] = i;
						model.support_labels[sv_idx] =
							dataset.labels[i];	/* {-1, 1} after normalization */
						memcpy(model.support_vectors
							   + sv_idx * dim,
							   dataset.features
							   + i * dim,
							   sizeof(float) * dim);
						sv_idx++;
					}
				}

				/* Validate we copied the expected number */
				if (sv_idx != sv_count)
				{
					model.n_support_vectors = sv_idx;
					if (sv_idx == 0)
					{
						/* Fallback: use first sample */
						model.n_support_vectors = 1;
						model.alphas[0] = 1.0;
						model.support_vector_indices
							[0] = 0;
						memcpy(model.support_vectors,
							   dataset.features,
							   sizeof(float)
							   * (size_t) dim);
					}
				}
			}
		}

		/* Compute accuracy on training set */
		for (i = 0; i < sample_limit && i < nvec; i++)
		{
			double		pred = bias;

			for (j = 0; j < sv_count; j++)
			{
				int			sv_idx = model.support_vector_indices[j];

				if (sv_idx >= 0 && sv_idx < nvec)
				{
					pred += model.alphas[j]
						* dataset.labels[sv_idx]
						* linear_kernel(
										model.support_vectors
										+ j * dim,
										dataset.features
										+ i * dim,
										dim);
				}
			}
			/* Labels are in {-1, 1}, predictions should be too */
			pred = (pred >= 0.0) ? 1.0 : -1.0;
			if (pred == dataset.labels[i])
				correct++;
		}
		accuracy = (sample_limit > 0)
			? ((double) correct / (double) sample_limit)
			: 0.0;

		/* Validate model before serialization */
		if (model.n_features <= 0 || model.n_features > 10000)
		{
			if (model.alphas)
			{
				nfree(model.alphas);
				model.alphas = NULL;
			}
			if (model.support_vectors)
			{
				nfree(model.support_vectors);
				model.support_vectors = NULL;
			}
			if (model.support_vector_indices)
			{
				nfree(model.support_vector_indices);
				model.support_vector_indices = NULL;
			}
			if (alphas)
				nfree(alphas);
			if (errors)
				nfree(errors);
			svm_dataset_free(&dataset);




			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: svm: model.n_features is invalid (%d) before serialization",
							model.n_features)));
		}


		/* Serialize model */
		serialized = svm_model_serialize(&model, 0);
		if (serialized == NULL)
		{
			if (model.alphas)
			{
				nfree(model.alphas);
				model.alphas = NULL;
			}
			if (model.support_vectors)
			{
				nfree(model.support_vectors);
				model.support_vectors = NULL;
			}
			if (model.support_vector_indices)
			{
				nfree(model.support_vector_indices);
				model.support_vector_indices = NULL;
			}
			if (alphas)
				nfree(alphas);
			if (errors)
				nfree(errors);
			svm_dataset_free(&dataset);




			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: svm: failed to serialize "
							"model")));
		}

		/*
		 * Note: GPU packing is disabled for CPU-trained models to avoid
		 * format conflicts. GPU packing should only be used when the model
		 * was actually trained on GPU. CPU models must use CPU serialization
		 * format for proper deserialization.
		 */

		/* Build hyperparameters JSON using JSONB API */
		{
			JsonbParseState *state = NULL;
			JsonbValue	jkey;
			JsonbValue	jval;

			JsonbValue *final_value = NULL;
			Numeric		C_num,
						max_iters_num;

			PG_TRY();
			{
				(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

				/* Add C */
				jkey.type = jbvString;
				jkey.val.string.val = "C";
				jkey.val.string.len = strlen("C");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				C_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(c_param)));
				jval.type = jbvNumeric;
				jval.val.numeric = C_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add max_iters */
				jkey.val.string.val = "max_iters";
				jkey.val.string.len = strlen("max_iters");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(actual_max_iters)));
				jval.type = jbvNumeric;
				jval.val.numeric = max_iters_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

				if (final_value == NULL)
				{
					elog(ERROR, "neurondb: train_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL for hyperparameters");
				}

				params_jsonb = JsonbValueToJsonb(final_value);
			}
			PG_CATCH();
			{
				ErrorData  *edata = CopyErrorData();

				elog(ERROR, "neurondb: train_svm: hyperparameters JSONB construction failed: %s", edata->message);
				FlushErrorState();
				params_jsonb = NULL;
			}
			PG_END_TRY();
		}

		if (params_jsonb == NULL)
		{
			if (model.alphas)
			{
				nfree(model.alphas);
				model.alphas = NULL;
			}
			if (model.support_vectors)
			{
				nfree(model.support_vectors);
				model.support_vectors = NULL;
			}
			if (model.support_vector_indices)
			{
				nfree(model.support_vector_indices);
				model.support_vector_indices = NULL;
			}
			if (alphas)
				nfree(alphas);
			if (errors)
				nfree(errors);
			if (serialized)
			{
				nfree(serialized);
				serialized = NULL;
			}
			svm_dataset_free(&dataset);




			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: svm: failed to create "
							"hyperparameters JSONB")));
		}

		/* Build metrics JSON using JSONB API */
		{
			JsonbParseState *state = NULL;
			JsonbValue	jkey;
			JsonbValue	jval;

			JsonbValue *final_value = NULL;
			Numeric		n_samples_num,
						n_features_num,
						n_support_vectors_num,
						C_num,
						max_iters_num,
						actual_iters_num,
						accuracy_num,
						bias_num;

			PG_TRY();
			{
				(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

				/* Add algorithm */
				jkey.type = jbvString;
				jkey.val.string.val = "algorithm";
				jkey.val.string.len = strlen("algorithm");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				jval.type = jbvString;
				jval.val.string.val = "svm";
				jval.val.string.len = strlen("svm");
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				jkey.val.string.val = "n_samples";
				jkey.val.string.len = strlen("n_samples");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(nvec)));
				jval.type = jbvNumeric;
				jval.val.numeric = n_samples_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add n_features */
				jkey.val.string.val = "n_features";
				jkey.val.string.len = strlen("n_features");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				n_features_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(dim)));
				jval.type = jbvNumeric;
				jval.val.numeric = n_features_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add n_support_vectors */
				jkey.val.string.val = "n_support_vectors";
				jkey.val.string.len = strlen("n_support_vectors");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				n_support_vectors_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(sv_count)));
				jval.type = jbvNumeric;
				jval.val.numeric = n_support_vectors_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add C */
				jkey.val.string.val = "C";
				jkey.val.string.len = strlen("C");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				C_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(c_param)));
				jval.type = jbvNumeric;
				jval.val.numeric = C_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add max_iters */
				jkey.val.string.val = "max_iters";
				jkey.val.string.len = strlen("max_iters");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_iters)));
				jval.type = jbvNumeric;
				jval.val.numeric = max_iters_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add actual_iters */
				jkey.val.string.val = "actual_iters";
				jkey.val.string.len = strlen("actual_iters");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				actual_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(iter)));
				jval.type = jbvNumeric;
				jval.val.numeric = actual_iters_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				jkey.val.string.val = "accuracy";
				jkey.val.string.len = strlen("accuracy");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy)));
				jval.type = jbvNumeric;
				jval.val.numeric = accuracy_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add bias */
				jkey.val.string.val = "bias";
				jkey.val.string.len = strlen("bias");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				bias_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(bias)));
				jval.type = jbvNumeric;
				jval.val.numeric = bias_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

				if (final_value == NULL)
				{
					elog(ERROR, "neurondb: train_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
				}

				metrics_jsonb = JsonbValueToJsonb(final_value);
			}
			PG_CATCH();
			{
				ErrorData  *edata = CopyErrorData();

				elog(ERROR, "neurondb: train_svm: metrics JSONB construction failed: %s", edata->message);
				FlushErrorState();
				metrics_jsonb = NULL;
			}
			PG_END_TRY();
		}

		if (metrics_jsonb == NULL)
		{
		}

		/* Register in catalog */
		memset(&spec, 0, sizeof(spec));
		spec.algorithm = "svm";
		spec.model_type = "classification";
		spec.training_table = tbl_str;
		spec.training_column = label_str;
		spec.parameters = params_jsonb;
		spec.metrics = metrics_jsonb;
		spec.model_data = serialized;
		spec.training_time_ms = -1;
		spec.num_samples = nvec;
		spec.num_features = dim;

		model_id = ml_catalog_register_model(&spec);


		if (model_id <= 0)
		{
			if (model.alphas)
			{
				nfree(model.alphas);
				model.alphas = NULL;
			}
			if (model.support_vectors)
			{
				nfree(model.support_vectors);
				model.support_vectors = NULL;
			}
			if (model.support_vector_indices)
			{
				nfree(model.support_vector_indices);
				model.support_vector_indices = NULL;
			}
			if (alphas)
				nfree(alphas);
			if (errors)
				nfree(errors);
			if (serialized)
			{
				nfree(serialized);
				serialized = NULL;
			}
			svm_dataset_free(&dataset);




			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: svm: failed to register model in catalog, model_id=%d",
							model_id)));
		}


		/*
		 * Note: serialized is owned by catalog. params_jsonb and
		 * metrics_jsonb are managed by memory context, don't free manually
		 */

		/*
		 * Cleanup - let memory context handle cleanup automatically. The
		 * catalog has taken ownership of the serialized model data. All other
		 * allocations (model arrays, alphas, errors, dataset, strings, etc.)
		 * will be automatically freed when the function's memory context is
		 * destroyed.
		 */
		PG_RETURN_INT32(model_id);
	}
	}

/*
 * predict_svm_model_id
 *
 * Makes predictions using trained SVM model from catalog
 */
PG_FUNCTION_INFO_V1(predict_svm_model_id);

Datum
predict_svm_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	Vector *features = NULL;

	SVMModel *model = NULL;
	double		prediction;
	int			i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: features vector is required")));

	features = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(features);

	/* Try GPU prediction first */
	if (svm_try_gpu_predict_catalog(model_id, features, &prediction))
	{

		/*
		 * Convert to binary class (-1 or 1) consistent with SVM label
		 * encoding
		 */
		prediction = (prediction >= 0.0) ? 1.0 : -1.0;
		PG_RETURN_FLOAT8(prediction);
	}

	/* Check if model is GPU-only before attempting CPU deserialization */
	{
		bytea *payload = NULL;
		Jsonb *metrics = NULL;
		bool		is_gpu_only = false;

		if (ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		{
			if (payload == NULL && svm_metadata_is_gpu(metrics))
			{
				/* GPU-only model, cannot deserialize on CPU */
				is_gpu_only = true;
			}
			if (payload != NULL)
				nfree(payload);
			if (metrics != NULL)
				nfree(metrics);
		}

		if (is_gpu_only)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: svm: model %d is GPU-only, GPU prediction failed",
							model_id),
					 errhint("Check GPU configuration and ensure GPU is available")));
		}
	}


	/* Load model from catalog */
	if (!svm_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: model %d not found", model_id)));

	if (model->n_features > 0 && features->dim != model->n_features)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: feature dimension mismatch (expected %d, got %d)",
						model->n_features,
						features->dim)));
	}

	/* Compute prediction using support vectors */
	prediction = model->bias;
	for (i = 0; i < model->n_support_vectors; i++)
	{
		float	   *sv = model->support_vectors + i * model->n_features;

		prediction += model->alphas[i]
			* model->support_labels[i]	/* y_i */
			* linear_kernel(sv, features->data, features->dim);
	}

	/* Convert to binary class (-1 or 1) consistent with SVM label encoding */
	prediction = (prediction >= 0.0) ? 1.0 : -1.0;

	if (model != NULL)
	{
		if (model->alphas != NULL)
			nfree(model->alphas);
		if (model->support_vectors != NULL)
			nfree(model->support_vectors);
		if (model->support_vector_indices != NULL)
			nfree(model->support_vector_indices);
		if (model->support_labels != NULL)
			nfree(model->support_labels);
		nfree(model);
	}

	PG_RETURN_FLOAT8(prediction);
}

/*
 * evaluate_svm_by_model_id
 *
 * Evaluates SVM model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: accuracy, precision, recall, f1_score, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_svm_by_model_id);

Datum
evaluate_svm_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *table_name = NULL;
	text *feature_col = NULL;
	text *label_col = NULL;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	int			ret;
	int			nvec = 0;
	int			i;
	int			feat_dim = 0;
	Oid			feat_type_oid = InvalidOid;
	Oid			label_type_oid = InvalidOid;
	bool		feat_is_array = false;
	double		accuracy = 0.0;
	double		precision = 0.0;
	double		recall = 0.0;
	double		f1_score = 0.0;
	MemoryContext oldcontext;
	StringInfoData query;

	SVMModel *model = NULL;

	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool		is_gpu_model = false;
	bool		gpu_payload_freed = false;
	bool		gpu_metrics_freed = false;
	int			valid_rows = 0;

	NdbSpiSession *eval_spi_session = NULL;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_svm_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	/* Validate model_id before attempting to load */
	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_svm_by_model_id: model_id must be positive, got %d", model_id),
				 errdetail("Invalid model_id: %d", model_id),
				 errhint("Provide a valid model_id from neurondb.ml_models catalog.")));

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_svm_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);


	oldcontext = CurrentMemoryContext;

	/* Load model from catalog - try CPU first, then GPU */
	if (!svm_load_model_from_catalog(model_id, &model))
	{
		/* Try GPU model */
		if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			if (gpu_payload == NULL)
			{
				if (gpu_metrics && !gpu_metrics_freed)
				{
					nfree(gpu_metrics);
					gpu_metrics_freed = true;
				}
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_svm_by_model_id: model %d has NULL payload",
								model_id)));
			}
			
			/* Check if this is a GPU model - either by metrics or by payload format */
			{
				uint32		payload_size;

				/* First check metrics for training_backend */
				if (svm_metadata_is_gpu(gpu_metrics))
				{
					is_gpu_model = true;
				}
				else
				{
					/* If metrics check didn't find GPU indicator, check payload format */
					/* GPU models start with NdbCudaSvmModelHeader, CPU models start with uint8 training_backend */
					payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
					
					/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
					/* GPU format: first field is feature_dim (int32) */
					/* Check if payload looks like GPU format (starts with int32, not uint8) */
					if (payload_size >= sizeof(int32))
					{
						const int32 *first_int = (const int32 *) VARDATA(gpu_payload);
						int32		first_value = *first_int;
						
						/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
						if (first_value > 0 && first_value <= 100000)
						{
							/* Check if payload size matches GPU format */
							if (payload_size >= sizeof(NdbCudaSvmModelHeader))
							{
								const NdbCudaSvmModelHeader *hdr = (const NdbCudaSvmModelHeader *) VARDATA(gpu_payload);
								
								/* Validate header fields match the first int32 */
								if (hdr->feature_dim == first_value &&
									hdr->n_samples >= 0 && hdr->n_samples <= 1000000000 &&
									hdr->n_support_vectors >= 0 && hdr->n_support_vectors <= 1000000)
								{
									size_t		expected_gpu_size = sizeof(NdbCudaSvmModelHeader) +
										sizeof(float) * (size_t) hdr->n_support_vectors +
										sizeof(float) * (size_t) hdr->n_support_vectors * (size_t) hdr->feature_dim +
										sizeof(int32) * (size_t) hdr->n_support_vectors;
									
									/* Size matches GPU format - likely a GPU model */
									if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
									{
										is_gpu_model = true;
									}
								}
							}
						}
					}
				}
			}
			
			if (!is_gpu_model)
			{
				if (gpu_payload && !gpu_payload_freed)
				{
					nfree(gpu_payload);
					gpu_payload_freed = true;
				}
				if (gpu_metrics && !gpu_metrics_freed)
				{
					nfree(gpu_metrics);
					gpu_metrics_freed = true;
				}
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_svm_by_model_id: model %d not found",
								model_id)));
			}
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_svm_by_model_id: model %d not found",
							model_id)));
		}
	}

	/* Connect to SPI */
	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(eval_spi_session, oldcontext);

	/* Build query - single query to fetch all data */
	ndb_spi_stringinfo_init(eval_spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(targ_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(targ_str));
	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		if (model != NULL)
		{
			if (model->alphas != NULL)
				nfree(model->alphas);
			if (model->support_vectors != NULL)
				nfree(model->support_vectors);
			if (model->support_vector_indices != NULL)
				nfree(model->support_vector_indices);
			nfree(model);
		}
		if (gpu_payload && !gpu_payload_freed)
		{
			nfree(gpu_payload);
			gpu_payload_freed = true;
		}
		if (gpu_metrics && !gpu_metrics_freed)
		{
			nfree(gpu_metrics);
			gpu_metrics_freed = true;
		}
		nfree(query.data);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		NDB_SPI_SESSION_END(eval_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_svm_by_model_id: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and label columns.")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		if (model != NULL)
		{
			if (model->alphas != NULL)
				nfree(model->alphas);
			if (model->support_vectors != NULL)
				nfree(model->support_vectors);
			if (model->support_vector_indices != NULL)
				nfree(model->support_vector_indices);
			nfree(model);
		}
		nfree(query.data);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		NDB_SPI_SESSION_END(eval_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_svm_by_model_id: no valid rows found"),
				 errdetail("Query returned 0 rows from table '%s'", tbl_str),
				 errhint("Ensure the table contains data and the WHERE clause is not too restrictive.")));
	}

	/* Determine feature and label column types */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	{
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
		label_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 2);
	}
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* Unified evaluation: Determine predict function based on compute mode */
	/* All metrics calculation is the same - only difference is predict function */

	/* Unified evaluation: Determine predict function based on compute mode */
	{
		bool		use_gpu_predict = false;
		int			tp = 0,
					tn = 0,
					fp = 0,
					fn = 0;
		int			total_predictions = 0;

		/* Determine if we should use GPU predict or CPU predict */
		if (is_gpu_model && neurondb_gpu_is_available() && !NDB_COMPUTE_MODE_IS_CPU())
		{
			/* GPU model and GPU mode: use GPU predict */
			if (gpu_payload != NULL && VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaSvmModelHeader))
			{
				const NdbCudaSvmModelHeader *gpu_hdr = (const NdbCudaSvmModelHeader *) VARDATA(gpu_payload);
				feat_dim = gpu_hdr->feature_dim;
				use_gpu_predict = true;
			}
		}
		else if (model != NULL)
		{
			/* CPU model or CPU mode: use CPU predict */
			feat_dim = model->n_features;
			use_gpu_predict = false;
		}
		else if (is_gpu_model && gpu_payload != NULL)
		{
			/* GPU model but CPU mode: convert to CPU format for CPU predict */
			if (VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaSvmModelHeader))
			{
				const NdbCudaSvmModelHeader *gpu_hdr = (const NdbCudaSvmModelHeader *) VARDATA(gpu_payload);
				feat_dim = gpu_hdr->feature_dim;
				/* Note: SVM GPU model conversion to CPU format would be complex,
				 * so we'll just use GPU predict even in CPU mode if model is available */
				use_gpu_predict = false;
			}
		}

		/* Ensure we have a valid model or GPU payload */
		if (model == NULL && !use_gpu_predict)
		{
			NDB_SPI_SESSION_END(eval_spi_session);
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(eval_spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_svm_by_model_id: no valid model found"),
					 errdetail("Neither CPU model nor GPU payload is available"),
					 errhint("Verify the model exists in the catalog and is in the correct format.")));
		}

		if (feat_dim <= 0)
		{
			NDB_SPI_SESSION_END(eval_spi_session);
			if (model != NULL)
			{
				if (model->alphas != NULL)
					nfree(model->alphas);
				if (model->support_vectors != NULL)
					nfree(model->support_vectors);
				if (model->support_vector_indices != NULL)
					nfree(model->support_vector_indices);
				if (model->support_labels != NULL)
					nfree(model->support_labels);
				nfree(model);
			}
			if (gpu_payload && !gpu_payload_freed)
			{
				nfree(gpu_payload);
				gpu_payload_freed = true;
			}
			if (gpu_metrics && !gpu_metrics_freed)
			{
				nfree(gpu_metrics);
				gpu_metrics_freed = true;
			}
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(eval_spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_svm_by_model_id: invalid feature dimension %d",
							feat_dim)));
		}

		/* Unified evaluation loop - only difference is predict function */
		/* Note: total_predictions counts only rows that pass all validation checks */
		/* This ensures metrics are calculated using the same set of rows */
		for (i = 0; i < nvec; i++)
		{
			HeapTuple	tuple = SPI_tuptable->vals[i];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Datum		feat_datum;
			Datum		targ_datum;
			bool		feat_null;
			bool		targ_null;
			double		y_true;
			double		y_pred = 0.0;
			int			pred_class;
			int			actual_class;
			int			actual_dim;
			int			j;
			float	   *feat_row = NULL;
			bool		prediction_made = false;

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

			/* Skip rows with null features or labels */
			if (feat_null || targ_null)
				continue;

			/* Extract target and normalize to {-1, 1} */
			{
				double		raw_label = svm_decode_label_datum(targ_datum, label_type_oid);
				y_true = (raw_label <= 0.5) ? -1.0 : 1.0;
			}

			/* Extract features and determine dimension */
			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(feat_datum);
				if (ARR_NDIM(arr) != 1)
					continue;
				actual_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				Vector *vec = DatumGetVector(feat_datum);
				actual_dim = vec->dim;
			}

			/* Validate feature dimension matches model */
			if (actual_dim != feat_dim)
				continue;

			/* Extract features to float array for prediction */
			nalloc(feat_row, float, feat_dim);
			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(feat_datum);
				if (feat_type_oid == FLOAT8ARRAYOID)
				{
					float8	   *data = (float8 *) ARR_DATA_PTR(arr);
					for (j = 0; j < feat_dim; j++)
						feat_row[j] = (float) data[j];
				}
				else
				{
					float4	   *data = (float4 *) ARR_DATA_PTR(arr);
					memcpy(feat_row, data, sizeof(float) * feat_dim);
				}
			}
			else
			{
				Vector *vec = DatumGetVector(feat_datum);
				memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
			}

			/* Call appropriate predict function based on compute mode */
			if (use_gpu_predict)
			{
				/* GPU predict path */
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
				int			predict_rc;
				char	   *gpu_err = NULL;

				predict_rc = ndb_gpu_svm_predict_double(gpu_payload,
													   feat_row,
													   feat_dim,
													   &y_pred,
													   &gpu_err);
				if (predict_rc == 0)
				{
					prediction_made = true;
				}
				else
				{
					/* GPU predict failed - check compute mode */
					if (NDB_REQUIRE_GPU())
					{
						/* Strict GPU mode: error out */
						if (gpu_err)
							nfree(gpu_err);
						nfree(feat_row);
						NDB_SPI_SESSION_END(eval_spi_session);
						if (gpu_payload && !gpu_payload_freed)
						{
							nfree(gpu_payload);
							gpu_payload_freed = true;
						}
						if (gpu_metrics && !gpu_metrics_freed)
						{
							nfree(gpu_metrics);
							gpu_metrics_freed = true;
						}
						nfree(tbl_str);
						nfree(feat_str);
						nfree(targ_str);
						ndb_spi_stringinfo_free(eval_spi_session, &query);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("neurondb: evaluate_svm_by_model_id: GPU prediction failed in GPU mode"),
								 errdetail("GPU prediction failed for row %d: %s", i, gpu_err ? gpu_err : "unknown error"),
								 errhint("GPU mode requires GPU prediction to succeed. Check GPU availability and model compatibility.")));
					}
					else
					{
						/* AUTO mode: fall back to CPU if available */
						if (gpu_err)
							nfree(gpu_err);
						if (model != NULL)
						{
							/* Compute prediction using CPU model */
							double		cpu_pred = model->bias;
							int			sv_idx;

							for (sv_idx = 0; sv_idx < model->n_support_vectors; sv_idx++)
							{
								float	   *sv = model->support_vectors + sv_idx * model->n_features;
								double		kernel_val = 0.0;
								int			k;

								/* Linear kernel: K(x, y) = x^T * y */
								for (k = 0; k < feat_dim; k++)
									kernel_val += (double) sv[k] * (double) feat_row[k];

								cpu_pred += model->alphas[sv_idx]
									* model->support_labels[sv_idx] /* y_i */
									* kernel_val;
							}
							y_pred = cpu_pred;
							prediction_made = true;
						}
						else
						{
							/* No CPU model available - skip this row */
							nfree(feat_row);
							continue;
						}
					}
				}
				if (gpu_err)
					nfree(gpu_err);
#endif
			}
			else
			{
				/* CPU predict path - compute prediction using model */
				if (model == NULL)
				{
					nfree(feat_row);
					continue;
				}

				y_pred = model->bias;
				for (j = 0; j < model->n_support_vectors; j++)
				{
					float	   *sv = model->support_vectors + j * model->n_features;
					double		kernel_val = 0.0;
					int			k;

					/* Linear kernel: K(x, y) = x^T * y */
					for (k = 0; k < feat_dim; k++)
						kernel_val += (double) sv[k] * (double) feat_row[k];

					y_pred += model->alphas[j]
						* model->support_labels[j] /* y_i */
						* kernel_val;
				}
				prediction_made = true;
			}

			/* Update confusion matrix (labels are normalized to {-1, 1}) */
			if (prediction_made)
			{
				pred_class = (y_pred >= 0.0) ? 1 : -1;
				actual_class = (int) y_true;	/* y_true is already -1 or 1 */

				if (pred_class == 1 && actual_class == 1)
					tp++;
				else if (pred_class == -1 && actual_class == -1)
					tn++;
				else if (pred_class == 1 && actual_class == -1)
					fp++;
				else if (pred_class == -1 && actual_class == 1)
					fn++;

				total_predictions++;
			}

			nfree(feat_row);
		}

		/* Calculate metrics from confusion matrix */
		if (total_predictions > 0)
		{
			accuracy = (double) (tp + tn) / (double) total_predictions;

			if ((tp + fp) > 0)
				precision = (double) tp / (double) (tp + fp);
			else
				precision = 0.0;

			if ((tp + fn) > 0)
				recall = (double) tp / (double) (tp + fn);
			else
				recall = 0.0;

			if ((precision + recall) > 0.0)
				f1_score = 2.0 * (precision * recall) / (precision + recall);
			else
				f1_score = 0.0;

			valid_rows = total_predictions;
		}
		else
		{
			/* No valid rows processed */
			NDB_SPI_SESSION_END(eval_spi_session);
			if (model != NULL)
			{
				if (model->alphas != NULL)
					nfree(model->alphas);
				if (model->support_vectors != NULL)
					nfree(model->support_vectors);
				if (model->support_vector_indices != NULL)
					nfree(model->support_vector_indices);
				if (model->support_labels != NULL)
					nfree(model->support_labels);
				nfree(model);
			}
			if (gpu_payload && !gpu_payload_freed)
			{
				nfree(gpu_payload);
				gpu_payload_freed = true;
			}
			if (gpu_metrics && !gpu_metrics_freed)
			{
				nfree(gpu_metrics);
				gpu_metrics_freed = true;
			}
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(eval_spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_svm_by_model_id: no valid rows found")));
		}

		/* Cleanup */
		if (model != NULL)
		{
			if (model->alphas != NULL)
				nfree(model->alphas);
			if (model->support_vectors != NULL)
				nfree(model->support_vectors);
			if (model->support_vector_indices != NULL)
				nfree(model->support_vector_indices);
			if (model->support_labels != NULL)
				nfree(model->support_labels);
			nfree(model);
		}
		if (gpu_payload != NULL && !gpu_payload_freed)
		{
			nfree(gpu_payload);
			gpu_payload_freed = true;
		}
		if (gpu_metrics != NULL && !gpu_metrics_freed)
		{
			nfree(gpu_metrics);
			gpu_metrics_freed = true;
		}
	}

	/* End SPI session BEFORE creating JSONB to avoid context conflicts */
	ndb_spi_stringinfo_free(eval_spi_session, &query);
	NDB_SPI_SESSION_END(eval_spi_session);

	/* Switch to old context and build JSONB directly using JSONB API */
	MemoryContextSwitchTo(oldcontext);
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;

		JsonbValue *final_value = NULL;
		Numeric		accuracy_num,
					precision_num,
					recall_num,
					f1_score_num,
					n_samples_num;

		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			jkey.type = jbvString;
			jkey.val.string.val = "accuracy";
			jkey.val.string.len = strlen("accuracy");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy)));
			jval.type = jbvNumeric;
			jval.val.numeric = accuracy_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "precision";
			jkey.val.string.len = strlen("precision");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			precision_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(precision)));
			jval.type = jbvNumeric;
			jval.val.numeric = precision_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "recall";
			jkey.val.string.len = strlen("recall");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			recall_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(recall)));
			jval.type = jbvNumeric;
			jval.val.numeric = recall_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "f1_score";
			jkey.val.string.len = strlen("f1_score");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			f1_score_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(f1_score)));
			jval.type = jbvNumeric;
			jval.val.numeric = f1_score_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "n_samples";
			jkey.val.string.len = strlen("n_samples");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(valid_rows)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_samples_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: evaluate_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL");
			}

			result_jsonb = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData  *edata = CopyErrorData();

			elog(ERROR, "neurondb: evaluate_svm: JSONB construction failed: %s", edata->message);
			FlushErrorState();
			result_jsonb = NULL;
		}
		PG_END_TRY();
	}

	if (result_jsonb == NULL)
	{
		if (tbl_str)
		{
			nfree(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			nfree(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			nfree(targ_str);
			targ_str = NULL;
		}
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_svm_by_model_id: JSONB result is NULL")));
	}

	if (tbl_str)
	{
		nfree(tbl_str);
		tbl_str = NULL;
	}
	if (feat_str)
	{
		nfree(feat_str);
		feat_str = NULL;
	}
	if (targ_str)
	{
		nfree(targ_str);
		targ_str = NULL;
	}

	PG_RETURN_JSONB_P(result_jsonb);
}

/*
 * GPU Model Ops for SVM
 */
typedef struct SVMGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			feature_dim;
	int			n_samples;
}			SVMGpuModelState;

static void
svm_gpu_release_state(SVMGpuModelState * state)
{
	if (state == NULL)
		return;
	if (state->model_blob)
		nfree(state->model_blob);
	if (state->metrics)
		nfree(state->metrics);
	nfree(state);
}

static bool
svm_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	SVMGpuModelState *state = NULL;
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	int			rc;
	int			i;


	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
		return false;
	if (!neurondb_gpu_is_available())
		return false;
	if (spec->feature_matrix == NULL || spec->label_vector == NULL)
		return false;
	if (spec->sample_count <= 0 || spec->feature_dim <= 0)
		return false;


	/* Normalize labels from {0, 1} to {-1, 1} for SVM training */
	/* CRITICAL: CUDA SVM requires labels to be exactly -1.0 or 1.0 */
	{
		double *normalized_labels = NULL;
		nalloc(normalized_labels, double, spec->sample_count);
		if (normalized_labels == NULL)
		{
			if (errstr)
				*errstr = pstrdup("svm_gpu_train: failed to allocate normalized labels");
			return false;
		}
		
		for (i = 0; i < spec->sample_count; i++)
		{
			/* Convert 0.0 -> -1.0, 1.0 -> 1.0 */
			if (spec->label_vector[i] <= 0.5)
				normalized_labels[i] = -1.0;
			else
				normalized_labels[i] = 1.0;
		}
		
		payload = NULL;
		metrics = NULL;

		PG_TRY();
		{
			rc = ndb_gpu_svm_train(spec->feature_matrix,
								   normalized_labels,
								   spec->sample_count,
								   spec->feature_dim,
								   spec->hyperparameters,
								   &payload,
								   &metrics,
								   errstr);
	}
	PG_CATCH();
	{
		/* Catch any PostgreSQL exceptions from GPU training */
		FlushErrorState();
		rc = -1;
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("SVM GPU training failed with exception");
	}
	PG_END_TRY();
	
	/* Free normalized labels array */
	if (normalized_labels)
		nfree(normalized_labels);
	
	if (rc != 0 || payload == NULL)
	{
		/* Caller must handle payload/metrics cleanup on failure */
		return false;
	}

	if (model->backend_state != NULL)
	{
		svm_gpu_release_state((SVMGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	state = NULL;
	nalloc(state, SVMGpuModelState, 1);
	memset(state, 0, sizeof(SVMGpuModelState));
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;
	state->metrics = metrics;

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	} /* End of normalized_labels block */

	return true;
}

static bool
svm_gpu_predict(const MLGpuModel *model,
				const float *input,
				int input_dim,
				float *output,
				int output_dim,
				char **errstr)
{
	const		SVMGpuModelState *state;
	double		prediction;
	int			rc;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = -1.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const SVMGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	/* Validate input dimension matches model */
	if (input_dim != state->feature_dim)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neurondb: svm: feature dimension mismatch");
		return false;
	}

	rc = ndb_gpu_svm_predict_double(state->model_blob,
									input,
									input_dim,
									&prediction,
									errstr);
	if (rc != 0)
		return false;

	output[0] = (float) prediction;

	return true;
}

static bool
svm_gpu_serialize(const MLGpuModel *model,
				  bytea * *payload_out,
				  Jsonb * *metadata_out,
				  char **errstr)
{
	const		SVMGpuModelState *state;
	SVMModel	svm_model;

	bytea *unified_payload = NULL;
	char *base = NULL;
	NdbCudaSvmModelHeader *hdr = NULL;
	float *sv_src_float = NULL;
	double *alpha_src = NULL;
	int			i;


	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const SVMGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
	/* Convert GPU format to unified format */
	base = VARDATA(state->model_blob);
	hdr = (NdbCudaSvmModelHeader *) base;
	alpha_src = (double *) (base + sizeof(NdbCudaSvmModelHeader));
	sv_src_float = (float *) (alpha_src + hdr->n_support_vectors);

	/* Build SVMModel structure */
	memset(&svm_model, 0, sizeof(SVMModel));
	svm_model.model_id = 0;		/* model_id not stored in GPU header */
	svm_model.n_features = hdr->feature_dim;
	svm_model.n_samples = hdr->n_samples;
	svm_model.n_support_vectors = hdr->n_support_vectors;
	svm_model.bias = (double) hdr->bias;
	svm_model.C = hdr->C;
	svm_model.max_iters = hdr->max_iters;

	/* Copy alphas */
	if (svm_model.n_support_vectors > 0)
	{
		double *alphas_tmp = NULL;
		nalloc(alphas_tmp, double, svm_model.n_support_vectors);
		memcpy(alphas_tmp, alpha_src, sizeof(double) * svm_model.n_support_vectors);
		svm_model.alphas = alphas_tmp;
	}

	/* Copy support vectors */
	if (svm_model.n_support_vectors > 0 && svm_model.n_features > 0)
	{
		float *support_vectors_tmp = NULL;
		nalloc(support_vectors_tmp, float, svm_model.n_support_vectors * svm_model.n_features);
		memcpy(support_vectors_tmp, sv_src_float, sizeof(float) * svm_model.n_support_vectors * svm_model.n_features);
		svm_model.support_vectors = support_vectors_tmp;
	}

	/* Create support_vector_indices (0, 1, 2, ..., n_support_vectors-1) */
	if (svm_model.n_support_vectors > 0)
	{
		int *indices_tmp = NULL;
		nalloc(indices_tmp, int, svm_model.n_support_vectors);
		for (i = 0; i < svm_model.n_support_vectors; i++)
			indices_tmp[i] = i;
		svm_model.support_vector_indices = indices_tmp;
	}

	/* Create support_labels (all +1.0 or -1.0, we don't have the original labels in GPU format) */
	if (svm_model.n_support_vectors > 0)
	{
		double *labels_tmp = NULL;
		nalloc(labels_tmp, double, svm_model.n_support_vectors);
		/* GPU format doesn't store original labels, use placeholder values */
		for (i = 0; i < svm_model.n_support_vectors; i++)
			labels_tmp[i] = (svm_model.alphas[i] > 0) ? 1.0 : -1.0;
		svm_model.support_labels = labels_tmp;
	}

	/* Serialize using unified format with training_backend=1 (GPU) */
	unified_payload = svm_model_serialize(&svm_model, 1);

	if (svm_model.alphas != NULL)
	{
		nfree(svm_model.alphas);
		svm_model.alphas = NULL;
	}
	if (svm_model.support_vectors != NULL)
	{
		nfree(svm_model.support_vectors);
		svm_model.support_vectors = NULL;
	}
	if (svm_model.support_vector_indices != NULL)
	{
		nfree(svm_model.support_vector_indices);
		svm_model.support_vector_indices = NULL;
	}
	if (svm_model.support_labels != NULL)
	{
		nfree(svm_model.support_labels);
		svm_model.support_labels = NULL;
	}

	if (payload_out != NULL)
		*payload_out = unified_payload;
	else if (unified_payload != NULL)
		nfree(unified_payload);

	/* Copy metrics from state (created during training) */
	if (metadata_out != NULL && state->metrics != NULL)
	{
		int			metadata_size;
		Jsonb *metadata_copy = NULL;

		metadata_size = VARSIZE(state->metrics);
		{
			char *metadata_buf = NULL;
			nalloc(metadata_buf, char, metadata_size);
			metadata_copy = (Jsonb *) metadata_buf;
		}
		NDB_CHECK_ALLOC(metadata_copy, "metadata_copy");
		memcpy(metadata_copy, state->metrics, metadata_size);
		*metadata_out = metadata_copy;
	}

	return true;
#else
	/* For non-CUDA builds, GPU serialization is not supported */
	if (errstr != NULL)
		*errstr = pstrdup("svm_gpu_serialize: CUDA not available");
	return false;
#endif
}

static void
svm_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		svm_gpu_release_state((SVMGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

/*
 * neurondb_gpu_register_svm_model
 */
void
neurondb_gpu_register_svm_model(void)
{
	static bool registered = false;
	static MLGpuModelOps svm_gpu_model_ops;

	if (registered)
		return;

	/* Initialize ops structure at runtime */
	memset(&svm_gpu_model_ops, 0, sizeof(MLGpuModelOps));
	svm_gpu_model_ops.algorithm = "svm";
	svm_gpu_model_ops.train = svm_gpu_train;
	svm_gpu_model_ops.predict = svm_gpu_predict;
	svm_gpu_model_ops.evaluate = NULL;
	svm_gpu_model_ops.serialize = svm_gpu_serialize;
	svm_gpu_model_ops.deserialize = NULL;
	svm_gpu_model_ops.destroy = svm_gpu_destroy;

	ndb_gpu_register_model_ops(&svm_gpu_model_ops);
	registered = true;
}
