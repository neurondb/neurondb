/*-------------------------------------------------------------------------
 *
 * ml_svm.c
 *    Support vector machine implementation.
 *
 * This module implements linear SVM for binary classification using a
 * large-margin training algorithm. Models are serialized and stored in
 * the catalog for prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
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
static SVMModel * svm_model_deserialize(const bytea * data, MemoryContext target_context, uint8 *training_backend_out);
static bool svm_metadata_is_gpu(Jsonb * metadata);
static double svm_decode_label_datum(Datum label_datum, Oid label_type_oid);
static bool svm_try_gpu_predict_catalog(int32 model_id,
										const Vector *feature_vec,
										double *result_out);
static bool svm_load_model_from_catalog(int32 model_id, SVMModel * *out);

static const MLGpuModelOps svm_gpu_model_ops;

/*
 * Linear kernel: K(x, y) = x^T * y
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
	double		dist_sq = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
	{
		double		diff = x[i] - y[i];

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
		NDB_FREE(dataset->features);
	if (dataset->labels != NULL)
		NDB_FREE(dataset->labels);
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
	NDB_DECLARE (NdbSpiSession *, load_spi_session);
	int			ret;
	int			nvec = 0;
	int			dim = 0;
	int			i;
	NDB_DECLARE (char *, query_str);
	size_t		query_len;

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
	NDB_ALLOC(query_str, char, query_len);
	snprintf(query_str, query_len,
			 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
			 quoted_feat, quoted_label, quoted_tbl, quoted_feat, quoted_label);

	elog(DEBUG1, "svm_dataset_load: executing query: %s", query_str);

	ret = ndb_spi_execute_safe(query_str, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(query_str);
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
		NDB_FREE(query_str);
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
		HeapTuple	tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		bool		feat_null;
		Oid			feat_type;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		feat_type = SPI_gettypeid(tupdesc, 1);

		if (!feat_null)
		{
			if (feat_type == FLOAT4ARRAYOID || feat_type == FLOAT8ARRAYOID)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

				if (arr != NULL)
					dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
			}
			else
			{
				Vector	   *vec = DatumGetVector(feat_datum);

				if (vec != NULL)
					dim = vec->dim;
			}
		}
	}

	if (dim <= 0)
	{
		NDB_FREE(query_str);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm_dataset_load: invalid feature dimension: %d", dim),
				 errdetail("Feature dimension is %d, must be positive", dim),
				 errhint("Ensure the feature column contains valid vector or array data.")));
	}

	MemoryContextSwitchTo(oldcontext);
	NDB_ALLOC(dataset->features, float, (size_t) nvec * (size_t) dim);
	NDB_ALLOC(dataset->labels, double, nvec);
	memset(dataset->features, 0, sizeof(float) * (size_t) nvec * (size_t) dim);
	memset(dataset->labels, 0, sizeof(double) * (size_t) nvec);
	dataset->n_samples = nvec;
	dataset->feature_dim = dim;

	{
		Oid			label_type;
		Oid			feat_type;
		int			valid_idx = 0;

		if (nvec > 0)
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			label_type = SPI_gettypeid(tupdesc, 2);
			feat_type = SPI_gettypeid(tupdesc, 1);
		}

		for (i = 0; i < nvec; i++)
		{
			HeapTuple	tuple;
			TupleDesc	tupdesc;
			Datum		feat_datum;
			Datum		label_datum;
			bool		feat_null;
			bool		label_null;

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
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);
				int			arr_dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));

				if (arr_dim != dim)
				{
					NDB_FREE(query_str);
					NDB_SPI_SESSION_END(load_spi_session);
					svm_dataset_free(dataset);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: svm_dataset_load: feature dimension mismatch"),
							 errdetail("Expected dimension %d but got %d at row %d", dim, arr_dim, i),
							 errhint("Ensure all feature arrays have consistent dimensions.")));
				}

				if (feat_type == FLOAT4ARRAYOID)
				{
					float	   *arr_data = (float *) ARR_DATA_PTR(arr);

					memcpy(dataset->features + valid_idx * dim,
						   arr_data,
						   sizeof(float) * dim);
				}
				else
				{
					double	   *arr_data = (double *) ARR_DATA_PTR(arr);

					for (int j = 0; j < dim; j++)
						dataset->features[valid_idx * dim + j] = (float) arr_data[j];
				}
			}
			else
			{
				Vector	   *vec = DatumGetVector(feat_datum);

				if (vec->dim != dim)
				{
					NDB_FREE(query_str);
					NDB_SPI_SESSION_END(load_spi_session);
					svm_dataset_free(dataset);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: svm_dataset_load: feature dimension mismatch"),
							 errdetail("Expected dimension %d but got %d at row %d", dim, vec->dim, i),
							 errhint("Ensure all feature vectors have the same dimension.")));
				}

				memcpy(dataset->features + valid_idx * dim,
					   vec->data,
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
			NDB_FREE(query_str);
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

	NDB_FREE(query_str);
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
svm_model_deserialize(const bytea * data, MemoryContext target_context, uint8 *training_backend_out)
{
	StringInfoData buf;
	SVMModel   *model;
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

	model = (SVMModel *) palloc(sizeof(SVMModel));
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
		NDB_FREE(model);
		MemoryContextSwitchTo(oldcontext);
		elog(DEBUG1,
			 "neurondb: svm: invalid n_features %d in deserialized model (corrupted data?)",
			 model->n_features);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: invalid n_features %d in deserialized model (corrupted data?)",
						model->n_features)));
	}
	if (model->n_support_vectors < 0 || model->n_support_vectors > 100000)
	{
		NDB_FREE(model);
		MemoryContextSwitchTo(oldcontext);
		elog(DEBUG1,
			 "neurondb: svm: invalid n_support_vectors %d in deserialized model (corrupted data?)",
			 model->n_support_vectors);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: invalid n_support_vectors %d in deserialized model (corrupted data?)",
						model->n_support_vectors)));
	}

	if (model->n_support_vectors > 0)
	{
		model->alphas = (double *) palloc(
										  sizeof(double) * (size_t) model->n_support_vectors);
		for (i = 0; i < model->n_support_vectors; i++)
			model->alphas[i] = pq_getmsgfloat8(&buf);
	}

	if (model->n_support_vectors > 0 && model->n_features > 0)
	{
		model->support_vectors = (float *) palloc(sizeof(float)
												  * (size_t) model->n_support_vectors
												  * (size_t) model->n_features);
		for (i = 0; i < model->n_support_vectors * model->n_features;
			 i++)
			model->support_vectors[i] = pq_getmsgfloat4(&buf);
	}

	if (model->n_support_vectors > 0)
	{
		model->support_vector_indices = (int *) palloc(
													   sizeof(int) * (size_t) model->n_support_vectors);
		for (i = 0; i < model->n_support_vectors; i++)
			model->support_vector_indices[i] =
				pq_getmsgint(&buf, 4);
	}

	if (model->n_support_vectors > 0)
	{
		model->support_labels = (double *) palloc(
												  sizeof(double) * (size_t) model->n_support_vectors);
		for (i = 0; i < model->n_support_vectors; i++)
			model->support_labels[i] = pq_getmsgfloat8(&buf);
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
	JsonbIterator *it;
	JsonbValue	v;
	JsonbIteratorToken r;

	if (metadata == NULL)
		return false;

	/* Check for training_backend integer in metrics */
	it = JsonbIteratorInit((JsonbContainer *) &metadata->root);
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
					int			backend = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					is_gpu = (backend == 1);
				}
			}
			NDB_FREE(key);
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
	bytea	   *payload = NULL;
	Jsonb	   *metrics = NULL;
	char	   *gpu_err = NULL;
	double		prediction = 0.0;
	bool		success = false;

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

	if (!svm_metadata_is_gpu(metrics))
		goto cleanup;

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
		elog(DEBUG1, "svm_try_gpu_predict_catalog: freeing payload=%p", (void *) payload);
		NDB_FREE(payload);
	}
	if (metrics != NULL)
		NDB_FREE(metrics);
	if (gpu_err != NULL)
		NDB_FREE(gpu_err);

	return success;
}

/*
 * svm_load_model_from_catalog
 */
static bool
svm_load_model_from_catalog(int32 model_id, SVMModel * *out)
{
	bytea	   *payload = NULL;
	Jsonb	   *metrics = NULL;
	SVMModel   *decoded;
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
			NDB_FREE(metrics);
		return false;
	}

	if (svm_metadata_is_gpu(metrics))
	{
		if (payload != NULL)
		{
			elog(DEBUG1, "svm_load_model_from_catalog: freeing payload=%p (GPU model)", (void *) payload);
			NDB_FREE(payload);
		}
		if (metrics != NULL)
			NDB_FREE(metrics);
		return false;
	}

	decoded = svm_model_deserialize(payload, oldcontext, NULL);

	if (payload != NULL)
	{
		elog(DEBUG1, "svm_load_model_from_catalog: freeing payload=%p (CPU model)", (void *) payload);
		NDB_FREE(payload);
	}
	if (metrics != NULL)
		NDB_FREE(metrics);

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
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	double		c_param;
	int			max_iters;
	char	   *tbl_str = NULL;
	char	   *feat_str = NULL;
	char	   *label_str = NULL;
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
	char	   *gpu_err = NULL;
	Jsonb	   *gpu_hyperparams = NULL;
	int32		model_id = 0;
	SVMModel	model;
	double	   *alphas = NULL;
	double	   *errors = NULL;

	if (PG_NARGS() < 3 || PG_NARGS() > 5)
	{
		elog(DEBUG1,
			 "neurondb: svm: train_svm_classifier requires 3-5 arguments, got %d",
			 PG_NARGS());
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




		elog(DEBUG1,
			 "neurondb: svm: need at least 10 samples for training, got %d",
			 nvec);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: need at least 10 samples for training, got %d",
						nvec)));
	}

	if (dim <= 0 || dim > 10000)
	{
		svm_dataset_free(&dataset);




		elog(DEBUG1,
			 "neurondb: svm: invalid feature dimension %d (must be in range [1, 10000])",
			 dim);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: invalid feature dimension %d (must be in range [1, 10000])",
						dim)));
	}

	/* Validate labels are binary - check for at least two distinct values */
	{
		double		first_label = dataset.labels[0];
		int			n_class0 = 0;
		int			n_class1 = 0;
		bool		found_two_classes = false;

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




			elog(DEBUG1,
				 "neurondb: svm: all labels have the same value (%.6f), need at least two classes",
				 first_label);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: svm: all labels have the same value (%.6f), need at least two classes",
							first_label)));
		}

		/* Normalize labels to {-1, 1} for SVM math */
		for (i = 0; i < nvec; i++)
		{
			if (dataset.labels[i] <= 0.5)
				dataset.labels[i] = -1.0;
			else
				dataset.labels[i] = 1.0;
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

		elog(DEBUG1,
			 "neurondb: svm: label normalization completed (negative=%d, positive=%d)",
			 n_class0,
			 n_class1);

		elog(DEBUG1, "neurondb: svm: labels normalized to {-1, 1} for SVM training");
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
			Numeric		C_num, max_iters_num;

			/* Start object */
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

				/* End object */
				final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
				
				if (final_value == NULL)
				{
					elog(ERROR, "neurondb: train_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL for hyperparameters");
				}
				
				gpu_hyperparams = JsonbValueToJsonb(final_value);
			}
			PG_CATCH();
			{
				ErrorData *edata = CopyErrorData();
				elog(ERROR, "neurondb: train_svm: hyperparameters JSONB construction failed: %s", edata->message);
				FlushErrorState();
				gpu_hyperparams = NULL;
			}
			PG_END_TRY();
		}

		if (gpu_hyperparams == NULL)
		{
			elog(DEBUG1,
				 "neurondb: svm: failed to create hyperparameters JSONB, falling back to CPU");
		}
		else if (ndb_gpu_try_train_model("svm",
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
#ifdef NDB_GPU_CUDA
			MLCatalogModelSpec spec;
			SVMModel		svm_model;
			bytea	   *unified_model_data = NULL;
			Jsonb	   *updated_metrics = NULL;
			char	   *base;
			NdbCudaSvmModelHeader *hdr;
			float	   *sv_src_float;
			double	   *alpha_src;
			int			i, j;

			elog(INFO,
				 "neurondb: svm: GPU training succeeded");
			elog(DEBUG1,
				 "neurondb: svm: converting GPU format to unified format");

			/* Convert GPU format to unified format */
			base = VARDATA(gpu_result.spec.model_data);
			hdr = (NdbCudaSvmModelHeader *) base;
			alpha_src = (double *) (base + sizeof(NdbCudaSvmModelHeader));
			sv_src_float = (float *) (alpha_src + hdr->n_support_vectors);

			/* Build SVMModel structure */
			memset(&svm_model, 0, sizeof(SVMModel));
			svm_model.model_id = hdr->model_id;
			svm_model.n_features = hdr->feature_dim;
			svm_model.n_samples = hdr->n_samples;
			svm_model.n_support_vectors = hdr->n_support_vectors;
			svm_model.bias = (double) hdr->bias;
			svm_model.C = hdr->C;
			svm_model.max_iters = hdr->max_iters;

			/* Copy alphas */
			if (svm_model.n_support_vectors > 0)
			{
				NDB_ALLOC(svm_model.alphas, double, svm_model.n_support_vectors);
				memcpy(svm_model.alphas, alpha_src, sizeof(double) * svm_model.n_support_vectors);
			}

			/* Copy support vectors, converting from float to float (same type) */
			if (svm_model.n_support_vectors > 0 && svm_model.n_features > 0)
			{
				NDB_ALLOC(svm_model.support_vectors, float, svm_model.n_support_vectors * svm_model.n_features);
				memcpy(svm_model.support_vectors, sv_src_float, sizeof(float) * svm_model.n_support_vectors * svm_model.n_features);
			}

			/* Serialize using unified format with training_backend=1 (GPU) */
			unified_model_data = svm_model_serialize(&svm_model, 1);

			/* Cleanup */
			if (svm_model.alphas != NULL)
			{
				NDB_FREE(svm_model.alphas);
				svm_model.alphas = NULL;
			}
			if (svm_model.support_vectors != NULL)
			{
				NDB_FREE(svm_model.support_vectors);
				svm_model.support_vectors = NULL;
			}

			/* Update metrics to use training_backend=1 */
			if (gpu_result.spec.metrics != NULL)
			{
				/* Build metrics JSON using JSONB API */
				{
					JsonbParseState *state = NULL;
					JsonbValue	jkey;
					JsonbValue	jval;
					JsonbValue *final_value = NULL;
					Numeric		n_features_num, n_samples_num, n_support_vectors_num, C_num, max_iters_num;

					/* Start object */
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

						/* Add training_backend */
						jkey.val.string.val = "training_backend";
						jkey.val.string.len = strlen("training_backend");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						jval.type = jbvNumeric;
						jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(1)));
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						/* Add n_features */
						jkey.val.string.val = "n_features";
						jkey.val.string.len = strlen("n_features");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						n_features_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(svm_model.n_features > 0 ? svm_model.n_features : 0)));
						jval.type = jbvNumeric;
						jval.val.numeric = n_features_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						/* Add n_samples */
						jkey.val.string.val = "n_samples";
						jkey.val.string.len = strlen("n_samples");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(svm_model.n_samples > 0 ? svm_model.n_samples : 0)));
						jval.type = jbvNumeric;
						jval.val.numeric = n_samples_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						/* Add n_support_vectors */
						jkey.val.string.val = "n_support_vectors";
						jkey.val.string.len = strlen("n_support_vectors");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						n_support_vectors_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(svm_model.n_support_vectors > 0 ? svm_model.n_support_vectors : 0)));
						jval.type = jbvNumeric;
						jval.val.numeric = n_support_vectors_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						/* Add C */
						jkey.val.string.val = "C";
						jkey.val.string.len = strlen("C");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						C_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(svm_model.C)));
						jval.type = jbvNumeric;
						jval.val.numeric = C_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						/* Add max_iters */
						jkey.val.string.val = "max_iters";
						jkey.val.string.len = strlen("max_iters");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(svm_model.max_iters)));
						jval.type = jbvNumeric;
						jval.val.numeric = max_iters_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						/* End object */
						final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
						
						if (final_value == NULL)
						{
							elog(ERROR, "neurondb: train_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
						}
						
						updated_metrics = JsonbValueToJsonb(final_value);
					}
					PG_CATCH();
					{
						ErrorData *edata = CopyErrorData();
						elog(ERROR, "neurondb: train_svm: metrics JSONB construction failed: %s", edata->message);
						FlushErrorState();
						updated_metrics = NULL;
					}
					PG_END_TRY();
				}
			}

			spec = gpu_result.spec;
			spec.model_data = unified_model_data;
			spec.metrics = updated_metrics;

			if (spec.training_table == NULL)
				spec.training_table = tbl_str;
			if (spec.training_column == NULL)
				spec.training_column = label_str;

			if (spec.parameters == NULL)
			{
				spec.parameters = gpu_hyperparams;
				gpu_hyperparams = NULL;
			}
			else
			{
				gpu_hyperparams = NULL;
			}

			spec.algorithm = "svm";
			spec.model_type = "classification";
			spec.training_time_ms = -1;
			spec.num_samples = nvec;
			spec.num_features = dim;

			model_id = ml_catalog_register_model(&spec);
#else
			MLCatalogModelSpec spec;

			spec = gpu_result.spec;

			if (spec.training_table == NULL)
				spec.training_table = tbl_str;
			if (spec.training_column == NULL)
				spec.training_column = label_str;

			/* Ensure parameters are set - use gpu_hyperparams if spec.parameters is NULL */
			if (spec.parameters == NULL)
			{
				spec.parameters = gpu_hyperparams;
				gpu_hyperparams = NULL;
			}
			else
			{
				/* spec.parameters already set by GPU, but ensure it's valid JSONB */
				if (gpu_hyperparams != NULL)
				{
					NDB_FREE(gpu_hyperparams);
					gpu_hyperparams = NULL;
				}
			}

			/* Ensure metrics are set - build if missing */
			if (spec.metrics == NULL)
			{
				/* Build metrics JSON using JSONB API */
				{
					JsonbParseState *state = NULL;
					JsonbValue	jkey;
					JsonbValue	jval;
					JsonbValue *final_value = NULL;
					Numeric		n_features_num, n_samples_num, n_support_vectors_num, C_num, max_iters_num;

					/* Start object */
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

						/* Add training_backend */
						jkey.val.string.val = "training_backend";
						jkey.val.string.len = strlen("training_backend");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						jval.type = jbvNumeric;
						jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(1)));
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						/* Add n_features */
						jkey.val.string.val = "n_features";
						jkey.val.string.len = strlen("n_features");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						n_features_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(dim)));
						jval.type = jbvNumeric;
						jval.val.numeric = n_features_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						/* Add n_samples */
						jkey.val.string.val = "n_samples";
						jkey.val.string.len = strlen("n_samples");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(nvec)));
						jval.type = jbvNumeric;
						jval.val.numeric = n_samples_num;
						(void) pushJsonbValue(&state, WJB_VALUE, &jval);

						/* Add n_support_vectors */
						jkey.val.string.val = "n_support_vectors";
						jkey.val.string.len = strlen("n_support_vectors");
						(void) pushJsonbValue(&state, WJB_KEY, &jkey);
						n_support_vectors_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(0)));
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

						/* End object */
						final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
						
						if (final_value == NULL)
						{
							elog(ERROR, "neurondb: train_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
						}
						
						spec.metrics = JsonbValueToJsonb(final_value);
					}
					PG_CATCH();
					{
						ErrorData *edata = CopyErrorData();
						elog(ERROR, "neurondb: train_svm: metrics JSONB construction failed: %s", edata->message);
						FlushErrorState();
						spec.metrics = NULL;
					}
					PG_END_TRY();
				}
			}

			spec.algorithm = "svm";
			spec.model_type = "classification";
			spec.training_time_ms = -1;
			spec.num_samples = nvec;
			spec.num_features = dim;

			model_id = ml_catalog_register_model(&spec);
#endif

			if (model_id <= 0)
			{
				ndb_gpu_free_train_result(&gpu_result);
				svm_dataset_free(&dataset);

				if (gpu_hyperparams)
					NDB_FREE(gpu_hyperparams);

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
			if (gpu_err != NULL)
			{
				elog(DEBUG1,
					 "neurondb: svm: GPU training failed: %s",
					 gpu_err);
				NDB_FREE(gpu_err);
			}
			else
			{
				elog(DEBUG1,
					 "neurondb: svm: GPU training unavailable, falling back to CPU");
			}
			if (gpu_hyperparams != NULL)
			{
				NDB_FREE(gpu_hyperparams);
				gpu_hyperparams = NULL;
			}
		}
	}

	/* Fall back to CPU training */
	elog(DEBUG1,
		 "neurondb: svm: starting CPU training (n_samples=%d, feature_dim=%d)",
		 nvec,
		 dim);

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
		bytea	   *serialized = NULL;
		MLCatalogModelSpec spec;
		Jsonb	   *params_jsonb = NULL;
		Jsonb	   *metrics_jsonb = NULL;
		int			correct = 0;
		double		accuracy = 0.0;

		/* Limit iterations and samples for large datasets */
		actual_max_iters =
			(max_iters > 1000 && nvec > 1000) ? 1000 : max_iters;
		sample_limit = (nvec > 5000) ? 5000 : nvec;
		kernel_limit = (sample_limit > 1000) ? 1000 : sample_limit;

		elog(DEBUG1,
			 "neurondb: svm: CPU training limits: actual_max_iters=%d, sample_limit=%d, kernel_limit=%d",
			 actual_max_iters,
			 sample_limit,
			 kernel_limit);

		/* Allocate memory for heuristic training algorithm */
		alphas = (double *) palloc0(
									sizeof(double) * (size_t) sample_limit);
		errors =
			(double *) palloc(sizeof(double) * (size_t) sample_limit);

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

		elog(DEBUG1,
			 "neurondb: svm: CPU heuristic training completed after %d iterations, %d support vectors",
			 iter,
			 sv_count);

		/* Validate dim before building model */
		if (dim <= 0 || dim > 10000)
		{
			if (alphas)
				NDB_FREE(alphas);
			if (errors)
				NDB_FREE(errors);
			svm_dataset_free(&dataset);




			elog(DEBUG1,
				 "neurondb: svm: invalid feature dimension %d before model serialization",
				 dim);
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

		elog(DEBUG1,
			 "neurondb: svm: building model with n_features=%d, n_samples=%d, sv_count=%d",
			 model.n_features,
			 model.n_samples,
			 sv_count);

		/* Handle case when no support vectors found */
		if (sv_count == 0)
		{
			elog(DEBUG1,
				 "neurondb: svm: no support vectors found, using default model with single support vector");
			sv_count = 1;
			model.n_support_vectors = sv_count;

			/* Allocate support vectors and alphas */
			model.alphas = (double *) palloc(
											 sizeof(double) * (size_t) sv_count);
			model.support_vectors = (float *) palloc0(
													  sizeof(float) * (size_t) sv_count * (size_t) dim);
			model.support_vector_indices = (int32 *) palloc(
															sizeof(int32) * (size_t) sv_count);
			model.support_labels = (double *) palloc(
													 sizeof(double) * (size_t) sv_count);

			if (model.alphas == NULL
				|| model.support_vectors == NULL
				|| model.support_vector_indices == NULL
				|| model.support_labels == NULL)
			{
				if (model.alphas)
					NDB_FREE(model.alphas);
				if (model.support_vectors)
					NDB_FREE(model.support_vectors);
				if (model.support_vector_indices)
					NDB_FREE(model.support_vector_indices);
				if (model.support_labels)
					NDB_FREE(model.support_labels);
				if (alphas)
					NDB_FREE(alphas);
				if (errors)
					NDB_FREE(errors);
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
			model.alphas = (double *) palloc(
											 sizeof(double) * (size_t) sv_count);
			model.support_vectors = (float *) palloc(
													 sizeof(float) * (size_t) sv_count * (size_t) dim);
			model.support_vector_indices = (int32 *) palloc(
															sizeof(int32) * (size_t) sv_count);
			model.support_labels = (double *) palloc(
													 sizeof(double) * (size_t) sv_count);

			if (model.alphas == NULL
				|| model.support_vectors == NULL
				|| model.support_vector_indices == NULL
				|| model.support_labels == NULL)
			{
				if (model.alphas)
					NDB_FREE(model.alphas);
				if (model.support_vectors)
					NDB_FREE(model.support_vectors);
				if (model.support_vector_indices)
					NDB_FREE(model.support_vector_indices);
				if (model.support_labels)
					NDB_FREE(model.support_labels);
				if (alphas)
					NDB_FREE(alphas);
				if (errors)
					NDB_FREE(errors);
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
					elog(DEBUG1,
						 "neurondb: svm: expected %d support vectors but copied %d",
						 sv_count,
						 sv_idx);
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
				NDB_FREE(model.alphas);
				model.alphas = NULL;
			}
			if (model.support_vectors)
			{
				NDB_FREE(model.support_vectors);
				model.support_vectors = NULL;
			}
			if (model.support_vector_indices)
			{
				NDB_FREE(model.support_vector_indices);
				model.support_vector_indices = NULL;
			}
			if (alphas)
				NDB_FREE(alphas);
			if (errors)
				NDB_FREE(errors);
			svm_dataset_free(&dataset);




			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: svm: model.n_features is invalid (%d) before serialization",
							model.n_features)));
		}

		elog(DEBUG1,
			 "neurondb: svm: serializing model with n_features=%d, n_support_vectors=%d",
			 model.n_features,
			 model.n_support_vectors);

		/* Serialize model */
		serialized = svm_model_serialize(&model, 0);
		if (serialized == NULL)
		{
			if (model.alphas)
			{
				NDB_FREE(model.alphas);
				model.alphas = NULL;
			}
			if (model.support_vectors)
			{
				NDB_FREE(model.support_vectors);
				model.support_vectors = NULL;
			}
			if (model.support_vector_indices)
			{
				NDB_FREE(model.support_vector_indices);
				model.support_vector_indices = NULL;
			}
			if (alphas)
				NDB_FREE(alphas);
			if (errors)
				NDB_FREE(errors);
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
			Numeric		C_num, max_iters_num;

			/* Start object */
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

				/* End object */
				final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
				
				if (final_value == NULL)
				{
					elog(ERROR, "neurondb: train_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL for hyperparameters");
				}
				
				params_jsonb = JsonbValueToJsonb(final_value);
			}
			PG_CATCH();
			{
				ErrorData *edata = CopyErrorData();
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
				NDB_FREE(model.alphas);
				model.alphas = NULL;
			}
			if (model.support_vectors)
			{
				NDB_FREE(model.support_vectors);
				model.support_vectors = NULL;
			}
			if (model.support_vector_indices)
			{
				NDB_FREE(model.support_vector_indices);
				model.support_vector_indices = NULL;
			}
			if (alphas)
				NDB_FREE(alphas);
			if (errors)
				NDB_FREE(errors);
			if (serialized)
			{
				NDB_FREE(serialized);
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
			Numeric		n_samples_num, n_features_num, n_support_vectors_num, C_num, max_iters_num, actual_iters_num, accuracy_num, bias_num;

			/* Start object */
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

				/* Add n_samples */
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

				/* Add accuracy */
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

				/* End object */
				final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
				
				if (final_value == NULL)
				{
					elog(ERROR, "neurondb: train_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
				}
				
				metrics_jsonb = JsonbValueToJsonb(final_value);
			}
			PG_CATCH();
			{
				ErrorData *edata = CopyErrorData();
				elog(ERROR, "neurondb: train_svm: metrics JSONB construction failed: %s", edata->message);
				FlushErrorState();
				metrics_jsonb = NULL;
			}
			PG_END_TRY();
		}

		if (metrics_jsonb == NULL)
		{
			elog(DEBUG1,
				 "neurondb: svm: failed to create metrics JSONB, continuing without metrics");
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
				NDB_FREE(model.alphas);
				model.alphas = NULL;
			}
			if (model.support_vectors)
			{
				NDB_FREE(model.support_vectors);
				model.support_vectors = NULL;
			}
			if (model.support_vector_indices)
			{
				NDB_FREE(model.support_vector_indices);
				model.support_vector_indices = NULL;
			}
			if (alphas)
				NDB_FREE(alphas);
			if (errors)
				NDB_FREE(errors);
			if (serialized)
			{
				NDB_FREE(serialized);
				serialized = NULL;
			}
			svm_dataset_free(&dataset);




			elog(DEBUG1,
				 "neurondb: svm: failed to register model in catalog, model_id=%d",
				 model_id);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: svm: failed to register model in catalog, model_id=%d",
							model_id)));
		}

		elog(DEBUG1,
			 "neurondb: svm: CPU training completed, model_id=%d",
			 model_id);

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

		elog(DEBUG1, "neurondb: svm: CPU training about to return model_id=%d", model_id);
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
	Vector	   *features;
	SVMModel   *model = NULL;
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
		elog(DEBUG1,
			 "neurondb: svm: GPU prediction succeeded, raw prediction=%.6f",
			 prediction);

		/*
		 * Convert to binary class (-1 or 1) consistent with SVM label
		 * encoding
		 */
		prediction = (prediction >= 0.0) ? 1.0 : -1.0;
		PG_RETURN_FLOAT8(prediction);
	}

	/* Check if model is GPU-only before attempting CPU deserialization */
	{
		bytea	   *payload = NULL;
		Jsonb	   *metrics = NULL;
		bool		is_gpu_only = false;

		if (ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		{
			if (payload == NULL && svm_metadata_is_gpu(metrics))
			{
				/* GPU-only model, cannot deserialize on CPU */
				is_gpu_only = true;
			}
			if (payload != NULL)
				NDB_FREE(payload);
			if (metrics != NULL)
				NDB_FREE(metrics);
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

	elog(DEBUG1,
		 "neurondb: svm: GPU prediction failed or not available, trying CPU");

	/* Load model from catalog */
	if (!svm_load_model_from_catalog(model_id, &model))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: svm: model %d not found", model_id)));

	/* Validate feature dimension */
	if (model->n_features > 0 && features->dim != model->n_features)
	{
		elog(DEBUG1,
			 "neurondb: svm: feature dimension mismatch (expected %d, got %d)",
			 model->n_features,
			 features->dim);
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

	/* Cleanup */
	if (model != NULL)
	{
		if (model->alphas != NULL)
			NDB_FREE(model->alphas);
		if (model->support_vectors != NULL)
			NDB_FREE(model->support_vectors);
		if (model->support_vector_indices != NULL)
			NDB_FREE(model->support_vector_indices);
		if (model->support_labels != NULL)
			NDB_FREE(model->support_labels);
		NDB_FREE(model);
	}

	PG_RETURN_FLOAT8(prediction);
}

/*
 * svm_predict_batch
 *
 * Helper function to predict a batch of samples using SVM model.
 * Updates confusion matrix.
 */
static void
svm_predict_batch(const SVMModel * model,
				  const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  int *tp_out,
				  int *tn_out,
				  int *fp_out,
				  int *fn_out)
{
	int			i;
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;

	if (model == NULL || features == NULL || labels == NULL || n_samples <= 0)
	{
		elog(DEBUG1,
			 "neurondb: svm_predict_batch: early return - model=%p, features=%p, labels=%p, n_samples=%d",
			 (void *) model, (void *) features, (void *) labels, n_samples);
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	elog(DEBUG1,
		 "neurondb: svm_predict_batch: starting - n_samples=%d, feature_dim=%d, model->n_support_vectors=%d, model->n_features=%d, model->bias=%.6f",
		 n_samples, feature_dim, model->n_support_vectors, model->n_features, model->bias);

	for (i = 0; i < n_samples; i++)
	{
		const float *row = features + (i * feature_dim);
		double		y_true = labels[i];
		int			true_class;
		double		prediction = 0.0;
		int			pred_class;
		int			j;

		if (!isfinite(y_true))
			continue;

		/* y_true is -1 or 1 after normalization */
		true_class = (y_true <= 0.0) ? -1 : 1;

		/* Compute prediction using support vectors */
		prediction = model->bias;
		if (model->n_support_vectors == 0)
		{
			elog(DEBUG1,
				 "neurondb: svm_predict_batch: WARNING - model has 0 support vectors, prediction will be bias only (%.6f)",
				 model->bias);
		}
		for (j = 0; j < model->n_support_vectors; j++)
		{
			float	   *sv = model->support_vectors + j * model->n_features;
			double		kernel_val = 0.0;
			int			k;

			/* Linear kernel: K(x, y) = x^T * y */
			for (k = 0; k < feature_dim; k++)
				kernel_val += (double) sv[k] * (double) row[k];

			prediction += model->alphas[j]
				* model->support_labels[j]	/* y_i */
				* kernel_val;
		}

		/*
		 * Convert to binary class (-1 or 1) consistent with SVM label
		 * encoding
		 */
		pred_class = (prediction >= 0.0) ? 1 : -1;

		if (i < 5)				/* Log first 5 predictions for debugging */
		{
			elog(DEBUG1,
				 "neurondb: svm_predict_batch: sample %d - y_true=%.6f (class=%d), prediction=%.6f (class=%d)",
				 i, y_true, true_class, prediction, pred_class);
		}

		/* Update confusion matrix (labels are -1 or 1) */
		if (true_class == 1 && pred_class == 1)
			tp++;
		else if (true_class == -1 && pred_class == -1)
			tn++;
		else if (true_class == -1 && pred_class == 1)
			fp++;
		else if (true_class == 1 && pred_class == -1)
			fn++;
	}

	if (tp_out)
		*tp_out = tp;
	if (tn_out)
		*tn_out = tn;
	if (fp_out)
		*fp_out = fp;
	if (fn_out)
		*fn_out = fn;
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
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
	int			ret;
	int			nvec = 0;
	int			i;
	int			j;
	int			feat_dim = 0;
	Oid			feat_type_oid = InvalidOid;
	Oid			label_type_oid = InvalidOid;
	bool		feat_is_array = false;
	double		accuracy = 0.0;
	double		precision = 0.0;
	double		recall = 0.0;
	double		f1_score = 0.0;
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;
	MemoryContext oldcontext;
	StringInfoData query;
	SVMModel   *model = NULL;
	StringInfoData jsonbuf;
	Jsonb	   *result_jsonb = NULL;
	bytea	   *gpu_payload = NULL;
	Jsonb	   *gpu_metrics = NULL;
	bool		is_gpu_model = false;
#ifdef NDB_GPU_CUDA
	int		   *h_labels = NULL;
	float	   *h_features = NULL;
#endif
	int			valid_rows = 0;
	NDB_DECLARE (NdbSpiSession *, eval_spi_session);

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_svm_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

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

	elog(DEBUG1, "evaluate_svm_by_model_id: tbl_str='%s', feat_str='%s', targ_str='%s'",
		 tbl_str, feat_str, targ_str);

	oldcontext = CurrentMemoryContext;

	/* Load model from catalog - try CPU first, then GPU */
	if (!svm_load_model_from_catalog(model_id, &model))
	{
		/* Try GPU model */
		if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			is_gpu_model = svm_metadata_is_gpu(gpu_metrics);
			if (!is_gpu_model)
			{
				if (gpu_payload)
				{
					elog(DEBUG1, "evaluate_svm_by_model_id: freeing gpu_payload=%p (CPU model)", (void *) gpu_payload);
					NDB_FREE(gpu_payload);
				}
				if (gpu_metrics)
					NDB_FREE(gpu_metrics);
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
	elog(DEBUG1, "evaluate_svm_by_model_id: executing query: %s", query.data);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	elog(DEBUG1, "evaluate_svm_by_model_id: SPI_execute returned %d, SPI_processed=%llu", ret, (unsigned long long) SPI_processed);
	if (ret != SPI_OK_SELECT)
	{
		if (model != NULL)
		{
			if (model->alphas != NULL)
				NDB_FREE(model->alphas);
			if (model->support_vectors != NULL)
				NDB_FREE(model->support_vectors);
			if (model->support_vector_indices != NULL)
				NDB_FREE(model->support_vector_indices);
			NDB_FREE(model);
		}
		NDB_FREE(query.data);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		NDB_SPI_SESSION_END(eval_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_svm_by_model_id: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and label columns.")));
	}

	nvec = SPI_processed;
	elog(DEBUG1, "evaluate_svm_by_model_id: nvec=%d", nvec);
	if (nvec < 1)
	{
		if (model != NULL)
		{
			if (model->alphas != NULL)
				NDB_FREE(model->alphas);
			if (model->support_vectors != NULL)
				NDB_FREE(model->support_vectors);
			if (model->support_vector_indices != NULL)
				NDB_FREE(model->support_vector_indices);
			NDB_FREE(model);
		}
		NDB_FREE(query.data);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
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

	/*
	 * GPU batch evaluation path for GPU models - uses optimized evaluation
	 * kernel
	 */
	if (is_gpu_model && neurondb_gpu_is_available())
	{
#ifdef NDB_GPU_CUDA
		const		NdbCudaSvmModelHeader *gpu_hdr;
		size_t		payload_size;

		/* Defensive check: validate payload size */
		payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
		if (payload_size < sizeof(NdbCudaSvmModelHeader))
		{
			elog(DEBUG1,
				 "neurondb: evaluate_svm_by_model_id: GPU payload too small (%zu bytes), falling back to CPU",
				 payload_size);
			goto cpu_evaluation_path;
		}

		/* Load GPU model header with defensive checks */
		if (gpu_payload == NULL || VARSIZE(gpu_payload) < VARHDRSZ)
		{
			elog(DEBUG1,
				 "neurondb: evaluate_svm_by_model_id: NULL or invalid GPU payload, falling back to CPU");
			goto cpu_evaluation_path;
		}
		gpu_hdr = (const NdbCudaSvmModelHeader *) VARDATA(gpu_payload);
		if (gpu_hdr == NULL)
		{
			elog(DEBUG1,
				 "neurondb: evaluate_svm_by_model_id: NULL GPU header, falling back to CPU");
			goto cpu_evaluation_path;
		}

		feat_dim = gpu_hdr->feature_dim;
		if (feat_dim <= 0 || feat_dim > 100000)
		{
			elog(DEBUG1,
				 "neurondb: evaluate_svm_by_model_id: invalid feature_dim (%d), falling back to CPU",
				 feat_dim);
			goto cpu_evaluation_path;
		}

		/* Allocate host buffers for features and labels with size checks */
		{
			size_t		features_size = sizeof(float) * (size_t) nvec * (size_t) feat_dim;
			size_t		labels_size = sizeof(int) * (size_t) nvec;

			if (features_size > MaxAllocSize || labels_size > MaxAllocSize)
			{
				elog(DEBUG1,
					 "evaluate_svm_by_model_id: allocation size too large (features=%zu, labels=%zu), falling back to CPU",
					 features_size, labels_size);
				goto cpu_evaluation_path;
			}

			/* Switch to the saved context before allocating */
			{
				MemoryContext saved_ctx = MemoryContextSwitchTo(oldcontext);

				h_features = (float *) palloc(features_size);
				h_labels = (int *) palloc(labels_size);
				MemoryContextSwitchTo(saved_ctx);	/* Switch back to SPI
													 * context */
			}

			if (h_features == NULL || h_labels == NULL)
			{
				elog(DEBUG1,
					 "evaluate_svm_by_model_id: memory allocation failed, falling back to CPU");
				if (h_features)
					NDB_FREE(h_features);
				if (h_labels)
					NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}
		}

		/*
		 * Extract features and labels from SPI results - optimized batch
		 * extraction
		 */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			if (tupdesc == NULL)
			{
				elog(DEBUG1,
					 "evaluate_svm_by_model_id: NULL TupleDesc, falling back to CPU");
				NDB_FREE(h_features);
				NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple;
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				Vector	   *vec;
				ArrayType  *arr;
				float	   *feat_row;

				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || i >= SPI_processed)
				{
					elog(DEBUG1, "evaluate_svm_by_model_id: breaking at i=%d (SPI_tuptable=%p, vals=%p, SPI_processed=%lu)",
						 i, (void *) SPI_tuptable, SPI_tuptable ? (void *) SPI_tuptable->vals : NULL, SPI_processed);
					break;
				}

				tuple = SPI_tuptable->vals[i];
				if (tuple == NULL)
				{
					elog(DEBUG1, "evaluate_svm_by_model_id: NULL tuple at i=%d", i);
					continue;
				}

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				/* Bounds check */
				if (valid_rows >= nvec)
				{
					elog(DEBUG1,
						 "evaluate_svm_by_model_id: valid_rows overflow, breaking");
					break;
				}

				feat_row = h_features + (valid_rows * feat_dim);
				if (feat_row == NULL || feat_row < h_features || feat_row >= h_features + (nvec * feat_dim))
				{
					elog(DEBUG1,
						 "evaluate_svm_by_model_id: feat_row out of bounds, skipping row");
					continue;
				}

				h_labels[valid_rows] = (int) rint(svm_decode_label_datum(targ_datum, label_type_oid));

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);
						int			j_remain = feat_dim % 4;
						int			j_end = feat_dim - j_remain;

						/*
						 * Process 4 elements at a time for better cache
						 * locality
						 */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float) data[j];
							feat_row[j + 1] = (float) data[j + 1];
							feat_row[j + 2] = (float) data[j + 2];
							feat_row[j + 3] = (float) data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float) data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);

						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			NDB_FREE(h_features);
			NDB_FREE(h_labels);
			if (gpu_payload)
			{
				elog(DEBUG1, "evaluate_svm_by_model_id: freeing gpu_payload=%p (no valid rows)", (void *) gpu_payload);
				NDB_FREE(gpu_payload);
			}
			if (gpu_metrics)
				NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ndb_spi_stringinfo_free(eval_spi_session, &query);
			NDB_SPI_SESSION_END(eval_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_svm_by_model_id: no valid rows found")));
		}

		/* Use optimized GPU batch evaluation */
		{
			int			rc;
			char	   *gpu_errstr = NULL;

			/* Defensive checks before GPU call */
			if (h_features == NULL || h_labels == NULL || valid_rows <= 0 || feat_dim <= 0)
			{
				elog(DEBUG1,
					 "evaluate_svm_by_model_id: invalid inputs for GPU evaluation (features=%p, labels=%p, rows=%d, dim=%d), falling back to CPU",
					 (void *) h_features, (void *) h_labels, valid_rows, feat_dim);
				NDB_FREE(h_features);
				NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}

			PG_TRY();
			{
				rc = ndb_cuda_svm_evaluate_batch(gpu_payload,
												 h_features,
												 h_labels,
												 valid_rows,
												 feat_dim,
												 &accuracy,
												 &precision,
												 &recall,
												 &f1_score,
												 &gpu_errstr);

				if (rc == 0)
				{
					Jsonb	   *result_copy;
					MemoryContext spi_context;

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
						Numeric		accuracy_num, precision_num, recall_num, f1_score_num, n_samples_num;

						/* Start object */
						PG_TRY();
						{
							(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

							/* Add accuracy */
							jkey.type = jbvString;
							jkey.val.string.val = "accuracy";
							jkey.val.string.len = strlen("accuracy");
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy)));
							jval.type = jbvNumeric;
							jval.val.numeric = accuracy_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							/* Add precision */
							jkey.val.string.val = "precision";
							jkey.val.string.len = strlen("precision");
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							precision_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(precision)));
							jval.type = jbvNumeric;
							jval.val.numeric = precision_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							/* Add recall */
							jkey.val.string.val = "recall";
							jkey.val.string.len = strlen("recall");
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							recall_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(recall)));
							jval.type = jbvNumeric;
							jval.val.numeric = recall_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							/* Add f1_score */
							jkey.val.string.val = "f1_score";
							jkey.val.string.len = strlen("f1_score");
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							f1_score_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(f1_score)));
							jval.type = jbvNumeric;
							jval.val.numeric = f1_score_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							/* Add n_samples */
							jkey.val.string.val = "n_samples";
							jkey.val.string.len = strlen("n_samples");
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(valid_rows)));
							jval.type = jbvNumeric;
							jval.val.numeric = n_samples_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							/* End object */
							final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
							
							if (final_value == NULL)
							{
								elog(ERROR, "neurondb: evaluate_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL");
							}
							
							result_jsonb = JsonbValueToJsonb(final_value);
						}
						PG_CATCH();
						{
							ErrorData *edata = CopyErrorData();
							elog(ERROR, "neurondb: evaluate_svm: JSONB construction failed: %s", edata->message);
							FlushErrorState();
							result_jsonb = NULL;
						}
						PG_END_TRY();
					}

					/*
					 * Switch to parent context and copy JSONB before
					 * SPI_finish()
					 */
					spi_context = CurrentMemoryContext;
					oldcontext = MemoryContextSwitchTo(fcinfo->flinfo->fn_mcxt);
					result_copy = (Jsonb *) PG_DETOAST_DATUM_COPY(PointerGetDatum(result_jsonb));
					MemoryContextSwitchTo(spi_context);

					elog(DEBUG1, "evaluate_svm_by_model_id: about to free h_features=%p", (void *) h_features);
					NDB_FREE(h_features);
					elog(DEBUG1, "evaluate_svm_by_model_id: about to free h_labels=%p", (void *) h_labels);
					NDB_FREE(h_labels);
					elog(DEBUG1, "evaluate_svm_by_model_id: freed arrays successfully");
					if (gpu_payload)
					{
						elog(DEBUG1, "evaluate_svm_by_model_id: freeing gpu_payload=%p (GPU success)", (void *) gpu_payload);
						NDB_FREE(gpu_payload);
					}
					if (gpu_metrics)
						NDB_FREE(gpu_metrics);
					if (gpu_errstr)
						NDB_FREE(gpu_errstr);
					NDB_FREE(query.data);
					NDB_FREE(tbl_str);
					NDB_FREE(feat_str);
					NDB_FREE(targ_str);
					NDB_SPI_SESSION_END(eval_spi_session);
					PG_RETURN_JSONB_P(result_copy);
				}
				else
				{
					/* GPU evaluation failed - fall back to CPU */
					elog(DEBUG1,
						 "evaluate_svm_by_model_id: GPU batch evaluation failed: %s, falling back to CPU",
						 gpu_errstr ? gpu_errstr : "unknown error");
					if (gpu_errstr)
						NDB_FREE(gpu_errstr);
					NDB_FREE(h_features);
					NDB_FREE(h_labels);
					goto cpu_evaluation_path;
				}
			}
			PG_CATCH();
			{
				elog(DEBUG1,
					 "evaluate_svm_by_model_id: exception during GPU evaluation, falling back to CPU");
				if (h_features)
					NDB_FREE(h_features);
				if (h_labels)
					NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}
			PG_END_TRY();
		}
#endif							/* NDB_GPU_CUDA */
	}
#ifndef NDB_GPU_CUDA
	/* When CUDA is not available, always use CPU path */
	if (false)
	{
	}
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-label"
cpu_evaluation_path:
#pragma GCC diagnostic pop

	/* CPU evaluation path */
	/* Use optimized batch prediction */
	{
		float	   *cpu_h_features = NULL;
		double	   *cpu_h_labels = NULL;
		int			cpu_valid_rows = 0;

		/* Determine feature dimension from model */
		if (model != NULL)
		{
			feat_dim = model->n_features;
			elog(DEBUG1, "evaluate_svm_by_model_id: CPU path - feat_dim=%d from model", feat_dim);
		}
		else if (is_gpu_model && gpu_payload != NULL)
		{
			size_t		payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
			const		NdbCudaSvmModelHeader *gpu_hdr;

			if (payload_size < sizeof(NdbCudaSvmModelHeader))
			{
				elog(DEBUG1,
					 "neurondb: evaluate_svm_by_model_id: GPU payload too small (%zu bytes) for CPU fallback",
					 payload_size);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_svm_by_model_id: GPU model payload corrupted, cannot determine feature dimension")));
			}

			gpu_hdr = (const NdbCudaSvmModelHeader *) VARDATA(gpu_payload);
			feat_dim = gpu_hdr->feature_dim;
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_svm_by_model_id: could not determine feature dimension")));
		}

		if (feat_dim <= 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_svm_by_model_id: invalid feature dimension %d",
							feat_dim)));
		}

		/* Allocate host buffers for features and labels in oldcontext */
		{
			MemoryContext saved_ctx = MemoryContextSwitchTo(oldcontext);

			cpu_h_features = (float *) palloc(sizeof(float) * (size_t) nvec * (size_t) feat_dim);
			cpu_h_labels = (double *) palloc(sizeof(double) * (size_t) nvec);
			MemoryContextSwitchTo(saved_ctx);
		}

		/*
		 * Extract features and labels from SPI results - optimized batch
		 * extraction
		 */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple = SPI_tuptable->vals[i];
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				Vector	   *vec;
				ArrayType  *arr;
				float	   *feat_row;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
				{
					if (i < 5)
						elog(DEBUG1, "evaluate_svm_by_model_id: CPU path row %d: NULL values (feat_null=%d, targ_null=%d)",
							 i, feat_null, targ_null);
					continue;
				}

				feat_row = cpu_h_features + (cpu_valid_rows * feat_dim);

				/*
				 * Read and normalize label to {-1, 1} for consistent
				 * evaluation
				 */
				{
					double		raw_label = svm_decode_label_datum(targ_datum, label_type_oid);

					cpu_h_labels[cpu_valid_rows] = (raw_label <= 0.5) ? -1.0 : 1.0;
				}

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
					{
						if (i < 5)
							elog(DEBUG1, "evaluate_svm_by_model_id: CPU path row %d: array dim mismatch (ndim=%d, dims[0]=%d, feat_dim=%d)",
								 i, ARR_NDIM(arr), ARR_DIMS(arr)[0], feat_dim);
						continue;
					}
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);
						int			j_remain = feat_dim % 4;
						int			j_end = feat_dim - j_remain;

						/*
						 * Process 4 elements at a time for better cache
						 * locality
						 */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float) data[j];
							feat_row[j + 1] = (float) data[j + 1];
							feat_row[j + 2] = (float) data[j + 2];
							feat_row[j + 3] = (float) data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float) data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);

						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				cpu_valid_rows++;
			}
		}

		if (cpu_valid_rows == 0)
		{
			NDB_FREE(cpu_h_features);
			NDB_FREE(cpu_h_labels);
			if (model != NULL)
			{
				if (model->alphas != NULL)
					NDB_FREE(model->alphas);
				if (model->support_vectors != NULL)
					NDB_FREE(model->support_vectors);
				if (model->support_vector_indices != NULL)
					NDB_FREE(model->support_vector_indices);
				NDB_FREE(model);
			}
			if (gpu_payload)
			{
				elog(DEBUG1, "evaluate_svm_by_model_id: freeing gpu_payload=%p (final cleanup)", (void *) gpu_payload);
				NDB_FREE(gpu_payload);
			}
			if (gpu_metrics)
				NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ndb_spi_stringinfo_free(eval_spi_session, &query);
			NDB_SPI_SESSION_END(eval_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_svm_by_model_id: no valid rows found")));
		}

		/* For GPU models, use GPU prediction */
		if (is_gpu_model && gpu_payload != NULL)
		{
			int			gpu_failures = 0;

			/* GPU batch evaluation using GPU prediction */
			for (i = 0; i < cpu_valid_rows; i++)
			{
				double		prediction = 0.0;
				double		actual = cpu_h_labels[i];
				int			rc;
				char	   *gpu_err = NULL;
				bool		prediction_made = false;

				rc = ndb_gpu_svm_predict_double(gpu_payload,
												cpu_h_features + (i * feat_dim),
												feat_dim,
												&prediction,
												&gpu_err);

				if (rc == 0)
				{
					prediction_made = true;
				}
				else
				{
					gpu_failures++;
					elog(DEBUG1,
						 "neurondb: evaluate_svm_by_model_id: GPU prediction failed for row %d: %s, falling back to CPU",
						 i, gpu_err ? gpu_err : "unknown error");
					if (gpu_err)
						NDB_FREE(gpu_err);
					/* Fall back to CPU if available */
					if (model != NULL)
					{
						/*
						 * Compute prediction using CPU model (same logic as
						 * svm_predict_batch)
						 */
						const float *row = cpu_h_features + (i * feat_dim);
						double		cpu_pred = model->bias;
						int			sv_idx;

						for (sv_idx = 0; sv_idx < model->n_support_vectors; sv_idx++)
						{
							float	   *sv = model->support_vectors + sv_idx * model->n_features;
							double		kernel_val = 0.0;
							int			k;

							/* Linear kernel: K(x, y) = x^T * y */
							for (k = 0; k < feat_dim; k++)
								kernel_val += (double) sv[k] * (double) row[k];

							cpu_pred += model->alphas[sv_idx]
								* model->support_labels[sv_idx] /* y_i */
								* kernel_val;
						}
						prediction = cpu_pred;
						prediction_made = true;
					}
					else
					{
						/*
						 * Cannot evaluate without model - use default
						 * prediction of 0
						 */
						elog(DEBUG1,
							 "neurondb: evaluate_svm_by_model_id: GPU prediction failed and no CPU model available for row %d, using default prediction",
							 i);
						prediction = 0.0;
						prediction_made = true;
					}
				}

				if (prediction_made)
				{
					/*
					 * Update confusion matrix (labels are normalized to {-1,
					 * 1})
					 */
					int			pred_class = (prediction >= 0.0) ? 1 : -1;
					int			actual_class = (int) actual;	/* actual is already -1
																 * or 1 */

					if (pred_class == 1 && actual_class == 1)
						tp++;
					else if (pred_class == -1 && actual_class == -1)
						tn++;
					else if (pred_class == 1 && actual_class == -1)
						fp++;
					else if (pred_class == -1 && actual_class == 1)
						fn++;
				}
			}

			if (gpu_failures > 0)
			{
				elog(DEBUG1,
					 "neurondb: evaluate_svm_by_model_id: GPU prediction failed for %d/%d rows",
					 gpu_failures, cpu_valid_rows);
			}
		}
		else
		{
			/* Ensure model is not NULL before prediction */
			if (model == NULL)
			{
				NDB_FREE(cpu_h_features);
				NDB_FREE(cpu_h_labels);
				if (gpu_payload)
					NDB_FREE(gpu_payload);
				if (gpu_metrics)
					NDB_FREE(gpu_metrics);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(query.data);
				NDB_FREE(targ_str);
				NDB_SPI_SESSION_END(eval_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: evaluate_svm_by_model_id: model is NULL"),
						 errdetail("Model pointer is unexpectedly NULL after loading from catalog"),
						 errhint("This indicates an internal programming error. Please report this issue.")));
			}

			/* Use batch prediction helper for CPU models */
			/* Add debug logging before prediction */
			if (model != NULL)
			{
				elog(DEBUG1,
					 "neurondb: evaluate_svm_by_model_id: CPU batch prediction: model->n_support_vectors=%d, model->n_features=%d, model->bias=%.6f, valid_rows=%d, feat_dim=%d",
					 model->n_support_vectors, model->n_features, model->bias, cpu_valid_rows, feat_dim);
			}
			else
			{
				elog(DEBUG1,
					 "neurondb: evaluate_svm_by_model_id: model is NULL before CPU batch prediction");
			}

			svm_predict_batch(model,
							  cpu_h_features,
							  cpu_h_labels,
							  cpu_valid_rows,
							  feat_dim,
							  &tp,
							  &tn,
							  &fp,
							  &fn);

			elog(DEBUG1,
				 "neurondb: evaluate_svm_by_model_id: after batch prediction: tp=%d, tn=%d, fp=%d, fn=%d, valid_rows=%d",
				 tp, tn, fp, fn, cpu_valid_rows);
		}

		/* Compute metrics */
		if (cpu_valid_rows > 0)
		{
			accuracy = (double) (tp + tn) / (double) cpu_valid_rows;

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

			valid_rows = cpu_valid_rows;
		}

		/* Cleanup */
		elog(DEBUG1, "evaluate_svm_by_model_id: about to free cpu_h_features=%p", (void *) cpu_h_features);
		NDB_FREE(cpu_h_features);
		elog(DEBUG1, "evaluate_svm_by_model_id: freed cpu_h_features, about to free cpu_h_labels=%p", (void *) cpu_h_labels);
		NDB_FREE(cpu_h_labels);
		elog(DEBUG1, "evaluate_svm_by_model_id: freed cpu_h_labels, about to free model=%p", (void *) model);
		if (model != NULL)
		{
			elog(DEBUG1, "evaluate_svm_by_model_id: about to free model->alphas=%p", (void *) model->alphas);
			NDB_FREE(model->alphas);
			elog(DEBUG1, "evaluate_svm_by_model_id: about to free model->support_vectors=%p", (void *) model->support_vectors);
			NDB_FREE(model->support_vectors);
			elog(DEBUG1, "evaluate_svm_by_model_id: about to free model->support_vector_indices=%p", (void *) model->support_vector_indices);
			NDB_FREE(model->support_vector_indices);
			elog(DEBUG1, "evaluate_svm_by_model_id: about to free model->support_labels=%p", (void *) model->support_labels);
			NDB_FREE(model->support_labels);
			elog(DEBUG1, "evaluate_svm_by_model_id: about to free model struct=%p", (void *) model);
			NDB_FREE(model);
			elog(DEBUG1, "evaluate_svm_by_model_id: freed model struct");
		}
		NDB_FREE(gpu_payload);
		NDB_FREE(gpu_metrics);
	}

	NDB_FREE(query.data);
	NDB_SPI_SESSION_END(eval_spi_session);
	elog(DEBUG1, "evaluate_svm_by_model_id: about to free tbl_str=%p", (void *) tbl_str);
	NDB_FREE(tbl_str);
	elog(DEBUG1, "evaluate_svm_by_model_id: freed tbl_str, about to free feat_str=%p", (void *) feat_str);
	NDB_FREE(feat_str);
	elog(DEBUG1, "evaluate_svm_by_model_id: freed feat_str, about to free targ_str=%p", (void *) targ_str);
	NDB_FREE(targ_str);
	elog(DEBUG1, "evaluate_svm_by_model_id: freed all string buffers");

	/* Build jsonb result */
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"n_samples\":%d}",
					 accuracy,
					 precision,
					 recall,
					 f1_score,
					 valid_rows);

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
			Numeric		accuracy_num, precision_num, recall_num, f1_score_num, n_samples_num;

			/* Start object */
			PG_TRY();
			{
				(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

				/* Add accuracy */
				jkey.type = jbvString;
				jkey.val.string.val = "accuracy";
				jkey.val.string.len = strlen("accuracy");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy)));
				jval.type = jbvNumeric;
				jval.val.numeric = accuracy_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add precision */
				jkey.val.string.val = "precision";
				jkey.val.string.len = strlen("precision");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				precision_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(precision)));
				jval.type = jbvNumeric;
				jval.val.numeric = precision_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add recall */
				jkey.val.string.val = "recall";
				jkey.val.string.len = strlen("recall");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				recall_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(recall)));
				jval.type = jbvNumeric;
				jval.val.numeric = recall_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add f1_score */
				jkey.val.string.val = "f1_score";
				jkey.val.string.len = strlen("f1_score");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				f1_score_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(f1_score)));
				jval.type = jbvNumeric;
				jval.val.numeric = f1_score_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add n_samples */
				jkey.val.string.val = "n_samples";
				jkey.val.string.len = strlen("n_samples");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(valid_rows)));
				jval.type = jbvNumeric;
				jval.val.numeric = n_samples_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* End object */
				final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
				
				if (final_value == NULL)
				{
					elog(ERROR, "neurondb: evaluate_svm: pushJsonbValue(WJB_END_OBJECT) returned NULL");
				}
				
				result_jsonb = JsonbValueToJsonb(final_value);
			}
			PG_CATCH();
			{
				ErrorData *edata = CopyErrorData();
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
				NDB_FREE(tbl_str);
				tbl_str = NULL;
			}
			if (feat_str)
			{
				NDB_FREE(feat_str);
				feat_str = NULL;
			}
			if (targ_str)
			{
				NDB_FREE(targ_str);
				targ_str = NULL;
			}
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: evaluate_svm_by_model_id: JSONB result is NULL")));
		}

		/* Cleanup */
		if (tbl_str)
		{
			NDB_FREE(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			NDB_FREE(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			NDB_FREE(targ_str);
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
		NDB_FREE(state->model_blob);
	if (state->metrics)
		NDB_FREE(state->metrics);
	NDB_FREE(state);
}

static bool
svm_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	SVMGpuModelState *state;
	bytea	   *payload;
	Jsonb	   *metrics;
	int			rc;

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

	payload = NULL;
	metrics = NULL;

	rc = ndb_gpu_svm_train(spec->feature_matrix,
						   spec->label_vector,
						   spec->sample_count,
						   spec->feature_dim,
						   spec->hyperparameters,
						   &payload,
						   &metrics,
						   errstr);
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

	state = (SVMGpuModelState *) palloc0(sizeof(SVMGpuModelState));
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	if (metrics != NULL)
		state->metrics = (Jsonb *) PG_DETOAST_DATUM_COPY(
														 PointerGetDatum(metrics));
	else
		state->metrics = NULL;

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
svm_gpu_predict(const MLGpuModel * model,
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
svm_gpu_serialize(const MLGpuModel * model,
				  bytea * *payload_out,
				  Jsonb * *metadata_out,
				  char **errstr)
{
	const		SVMGpuModelState *state;
	SVMModel	svm_model;
	bytea	   *unified_payload = NULL;
	char	   *base;
	NdbCudaSvmModelHeader *hdr;
	float	   *sv_src_float;
	double	   *alpha_src;
	int			i;

	(void) state;
	(void) svm_model;
	(void) unified_payload;
	(void) base;
	(void) hdr;
	(void) sv_src_float;
	(void) alpha_src;
	(void) i;

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

#ifdef NDB_GPU_CUDA
	/* Convert GPU format to unified format */
	base = VARDATA(state->model_blob);
	hdr = (NdbCudaSvmModelHeader *) base;
	alpha_src = (double *) (base + sizeof(NdbCudaSvmModelHeader));
	sv_src_float = (float *) (alpha_src + hdr->n_support_vectors);

	/* Build SVMModel structure */
	memset(&svm_model, 0, sizeof(SVMModel));
	svm_model.model_id = hdr->model_id;
	svm_model.n_features = hdr->feature_dim;
	svm_model.n_samples = hdr->n_samples;
	svm_model.n_support_vectors = hdr->n_support_vectors;
	svm_model.bias = (double) hdr->bias;
	svm_model.C = hdr->C;
	svm_model.max_iters = hdr->max_iters;

	/* Copy alphas */
	if (svm_model.n_support_vectors > 0)
	{
		NDB_ALLOC(svm_model.alphas, double, svm_model.n_support_vectors);
		memcpy(svm_model.alphas, alpha_src, sizeof(double) * svm_model.n_support_vectors);
	}

	/* Copy support vectors */
	if (svm_model.n_support_vectors > 0 && svm_model.n_features > 0)
	{
		NDB_ALLOC(svm_model.support_vectors, float, svm_model.n_support_vectors * svm_model.n_features);
		memcpy(svm_model.support_vectors, sv_src_float, sizeof(float) * svm_model.n_support_vectors * svm_model.n_features);
	}

	/* Serialize using unified format with training_backend=1 (GPU) */
	unified_payload = svm_model_serialize(&svm_model, 1);

	/* Cleanup */
	if (svm_model.alphas != NULL)
	{
		NDB_FREE(svm_model.alphas);
		svm_model.alphas = NULL;
	}
	if (svm_model.support_vectors != NULL)
	{
		NDB_FREE(svm_model.support_vectors);
		svm_model.support_vectors = NULL;
	}

	if (payload_out != NULL)
		*payload_out = unified_payload;
	else if (unified_payload != NULL)
		NDB_FREE(unified_payload);

	/* Update metrics to use training_backend=1 */
	if (metadata_out != NULL)
	{
		StringInfoData metrics_buf;
		initStringInfo(&metrics_buf);
		appendStringInfo(&metrics_buf,
						 "{\"algorithm\":\"svm\","
						 "\"training_backend\":1,"
						 "\"n_features\":%d,"
						 "\"n_samples\":%d,"
						 "\"n_support_vectors\":%d}",
						 state->feature_dim > 0 ? state->feature_dim : 0,
						 state->n_samples > 0 ? state->n_samples : 0,
						 state->n_support_vectors > 0 ? state->n_support_vectors : 0);

		*metadata_out = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
														   CStringGetTextDatum(metrics_buf.data)));
		NDB_FREE(metrics_buf.data);
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
svm_gpu_destroy(MLGpuModel * model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		svm_gpu_release_state((SVMGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps svm_gpu_model_ops = {
	.algorithm = "svm",
	.train = svm_gpu_train,
	.predict = svm_gpu_predict,
	.evaluate = NULL,
	.serialize = svm_gpu_serialize,
	.deserialize = NULL,
	.destroy = svm_gpu_destroy,
};

/*
 * neurondb_gpu_register_svm_model
 */
void
neurondb_gpu_register_svm_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&svm_gpu_model_ops);
	registered = true;
}
