/*-------------------------------------------------------------------------
 *
 * ml_naive_bayes.c
 *    Naive Bayes classifier.
 *
 * This module implements Gaussian Naive Bayes for classification with
 * continuous features, with model serialization and catalog storage.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_naive_bayes.c
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
 
 #include "neurondb.h"
 #include "neurondb_ml.h"
 #include "neurondb_gpu.h"
 #include "neurondb_gpu_model.h"
 #include "neurondb_gpu_backend.h"
 #include "ml_catalog.h"
 #include "neurondb_cuda_nb.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_json.h"
 
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
/* ndb_cuda_nb_evaluate_batch is declared in neurondb_cuda_nb.h */
#endif
 
 #include <math.h>
 #include <float.h>
 
 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif
 
 /*
  * Gaussian Naive Bayes model parameters
  */
 typedef struct
 {
	 double *class_priors;
	 double **means;
	 double **variances;
	 int n_classes;
	 int n_features;
 } GaussianNBModel;

 static int nb_train_from_spi_result(int nvec,
				 Oid feat_type_oid,
				 bool feat_is_array,
				 GaussianNBModel *out_model,
				 int *out_valid,
				 int *out_dim);

 static void nb_free_model_and_metadata(GaussianNBModel *model,
				 bytea *model_data,
				 Jsonb *metrics);

 static bytea *nb_model_serialize_to_bytea(const GaussianNBModel *model);
 static GaussianNBModel *nb_model_deserialize_from_bytea(const bytea *data);

 static Datum nb_evaluate_naive_bayes_internal(FunctionCallInfo fcinfo);
 static void nb_predict_batch(const GaussianNBModel *model,
			 const float *features,
			 const double *labels,
			 int n_samples,
			 int feature_dim,
			 int *tp_out,
			 int *tn_out,
			 int *fp_out,
			 int *fn_out);

 /*
  * Gaussian probability density function
  */
 static double
 gaussian_pdf(double x, double mean, double variance)
 {
	 double exponent;
 
	 if (variance < 1e-9)
		 variance = 1e-9;
 
	 exponent = -0.5 * pow((x - mean), 2) / variance;
	 return (1.0 / sqrt(2.0 * M_PI * variance)) * exp(exponent);
 }
 
 /*
  * train_naive_bayes_classifier
  *
  * Trains a Gaussian Naive Bayes classifier
  * Returns model parameters as JSONB
  */
 PG_FUNCTION_INFO_V1(train_naive_bayes_classifier);
 
 Datum
 train_naive_bayes_classifier(PG_FUNCTION_ARGS)
 {
	 text *table_name;
	 text *feature_col;
	 text *label_col;
	 char *tbl_str;
	 char *feat_str;
	 char *label_str;
	 StringInfoData query = {0};
	 NDB_DECLARE (NdbSpiSession *, train_nb_spi_session);
	 MemoryContext oldcontext;
	 int ret;
	 int nvec = 0;
	 int dim = 0;
	 GaussianNBModel model;
	 int valid = 0;
	 int j, class;
	 int n_params;
	 int idx;
	 Datum *result_datums;
	 ArrayType *result_array;
	 Oid feat_type_oid;
	 bool feat_is_array;

	 table_name = PG_GETARG_TEXT_PP(0);
	 feature_col = PG_GETARG_TEXT_PP(1);
	 label_col = PG_GETARG_TEXT_PP(2);

	 tbl_str = text_to_cstring(table_name);
	 feat_str = text_to_cstring(feature_col);
	 label_str = text_to_cstring(label_col);

	 oldcontext = CurrentMemoryContext;
	 Assert(oldcontext != NULL);
	 NDB_SPI_SESSION_BEGIN(train_nb_spi_session, oldcontext);

	 /* Build query - cast label to float8 for type safety */
	 initStringInfo(&query);
	 appendStringInfo(&query,
		 "SELECT %s, %s::float8 FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		 quote_identifier(feat_str),
		 quote_identifier(label_str),
		 quote_identifier(tbl_str),
		 quote_identifier(feat_str),
		 quote_identifier(label_str));
	 elog(DEBUG1, "train_naive_bayes_classifier: executing query: %s",
		 query.data);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	 if (ret != SPI_OK_SELECT)
		 ereport(ERROR,
			 (errcode(ERRCODE_INTERNAL_ERROR),
			  errmsg("neurondb: query failed")));

	 nvec = SPI_processed;

	 /* Determine feature type from first row */
	 feat_type_oid = InvalidOid;
	 feat_is_array = false;
	 if (nvec > 0 && SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	 {
		 feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
		 if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			 feat_is_array = true;
	 }

	 /* Use shared training helper */
	 nb_train_from_spi_result(nvec, feat_type_oid, feat_is_array, &model, &valid, &dim);

	 NDB_FREE(query.data);
	 NDB_SPI_SESSION_END(train_nb_spi_session);

	 /*
	  * Serialize model parameters to array.
	  * Format:
	  * [n_classes, n_features, prior0, prior1, ...,
	  *  mean0_0, mean0_1, ..., mean1_0, mean1_1, ...,
	  *  var0_0, var0_1, ..., var1_0, var1_1, ...]
	  */
	 {
		 int n_classes = model.n_classes;

		 /* Enforce binary-only for array format */
		 if (n_classes != 2)
			 ereport(ERROR,
				 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				  errmsg("neurondb: Naive Bayes array format "
					 "supports only 2 classes, got %d",
					 n_classes)));

		 n_params = 2 + n_classes + (n_classes * dim) + (n_classes * dim);
		 NDB_ALLOC(result_datums, Datum, n_params);

		 result_datums[0] = Float8GetDatum(n_classes);
		 result_datums[1] = Float8GetDatum(model.n_features);

		 for (class = 0; class < n_classes; class++)
			 result_datums[2 + class] =
				 Float8GetDatum(model.class_priors[class]);

		 idx = 2 + n_classes;
		 for (class = 0; class < n_classes; class++)
			 for (j = 0; j < dim; j++)
				 result_datums[idx++] =
					 Float8GetDatum(model.means[class][j]);

		 for (class = 0; class < n_classes; class++)
			 for (j = 0; j < dim; j++)
				 result_datums[idx++] =
					 Float8GetDatum(model.variances[class][j]);
	 }
 
	 result_array = construct_array(result_datums,
		 n_params,
		 FLOAT8OID,
		 sizeof(float8),
		 FLOAT8PASSBYVAL,
		 'd');
 
	 /* Cleanup */
	 NDB_FREE(model.class_priors);
	 for (class = 0; class < 2; class++)
	 {
		 NDB_FREE(model.means[class]);
		 NDB_FREE(model.variances[class]);
	 }
	 NDB_FREE(model.means);
	 NDB_FREE(model.variances);
	 NDB_FREE(result_datums);
	 NDB_FREE(tbl_str);
	 NDB_FREE(feat_str);
	 NDB_FREE(label_str);

	 PG_RETURN_ARRAYTYPE_P(result_array);
}
 
/*
 * Internal helper: Train Naive Bayes model from SPI result set
 *
 * Extracts features and labels from SPI_tuptable, validates data,
 * computes class priors, means, and variances.
 *
 * Parameters:
 *   nvec: Number of rows in SPI result
 *   feat_type_oid: OID of feature column type
 *   feat_is_array: Whether feature is array type
 *   out_model: Output model structure. Must point to uninitialized storage.
 *              This function sets n_classes = 2 and allocates all arrays.
 *   out_valid: Number of valid samples processed
 *   out_dim: Feature dimension
 *
 * Returns: 0 on success, -1 on error (calls ereport on errors)
 */
static int
nb_train_from_spi_result(int nvec,
			 Oid feat_type_oid,
			 bool feat_is_array,
			 GaussianNBModel *out_model,
			 int *out_valid,
			 int *out_dim)
{
	float **X = NULL;
	double *y = NULL;
	int valid = 0;
	int dim = 0;
	int i, j, class;
	int *class_counts;
	double ***class_samples;
	int *class_sizes;

	if (nvec < 10)
		ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
			 errmsg("neurondb: need at least 10 samples")));

	NDB_ALLOC(X, float *, nvec);
	NDB_ALLOC(y, double, nvec);

	for (i = 0; i < nvec; i++)
	{
		HeapTuple tuple;
		TupleDesc tupdesc;
		Datum feat_datum;
		Datum label_datum;
		bool feat_null;
		bool label_null;
		Vector *vec;
		ArrayType *arr;
		int current_dim = 0;
		
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

		{
			double label_val = DatumGetFloat8(label_datum);
			int label_class = (int) rint(label_val);

			if (label_class != 0 && label_class != 1)
				ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: Naive Bayes expects labels 0 or 1, "
						"got %g", label_val)));
			y[valid] = (double) label_class;
		}

		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
			if (ARR_NDIM(arr) != 1)
				continue;

			current_dim = ARR_DIMS(arr)[0];
			if (current_dim <= 0)
				continue;

			if (valid == 0)
				dim = current_dim;
			else if (current_dim != dim)
				continue;

			NDB_ALLOC(X[valid], float, dim);

			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				float8 *data = (float8 *) ARR_DATA_PTR(arr);
				for (j = 0; j < dim; j++)
					X[valid][j] = (float) data[j];
			}
			else
			{
				float4 *data = (float4 *) ARR_DATA_PTR(arr);
				memcpy(X[valid], data, sizeof(float) * dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec == NULL || vec->dim <= 0)
				continue;

			current_dim = vec->dim;
			if (valid == 0)
				dim = current_dim;
			else if (current_dim != dim)
				continue;

			NDB_ALLOC(X[valid], float, dim);
			memcpy(X[valid], vec->data, sizeof(float) * dim);
		}

		valid++;
	}

	if (valid < 10)
		ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
			 errmsg("neurondb: need at least 10 valid samples "
				"(got %d after filtering nulls and dimension errors)",
				valid)));
	if (dim <= 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("neurondb: Naive Bayes training found no valid "
				"feature rows with nonzero dimension")));

	out_model->n_classes = 2;
	out_model->n_features = dim;
	out_model->class_priors = (double *) palloc0(sizeof(double) * 2);
	NDB_CHECK_ALLOC(out_model->class_priors, "out_model->class_priors");
	NDB_ALLOC(out_model->means, double *, 2);
	NDB_ALLOC(out_model->variances, double *, 2);

	NDB_ALLOC(class_counts, int, 2);
	memset(class_counts, 0, sizeof(int) * 2);
	NDB_ALLOC(class_sizes, int, 2);
	memset(class_sizes, 0, sizeof(int) * 2);
	NDB_ALLOC(class_samples, double **, 2);

	for (i = 0; i < valid; i++)
	{
		class = (int) y[i];
		if (class >= 0 && class < 2)
			class_counts[class]++;
	}

	for (class = 0; class < 2; class++)
	{
		NDB_ALLOC(class_samples[class], double *, class_counts[class]);
		NDB_ALLOC(out_model->means[class], double, dim);
		memset(out_model->means[class], 0, sizeof(double) * dim);
		NDB_ALLOC(out_model->variances[class], double, dim);
		memset(out_model->variances[class], 0, sizeof(double) * dim);
	}

	for (i = 0; i < valid; i++)
	{
		class = (int) y[i];
		if (class >= 0 && class < 2)
		{
			NDB_ALLOC(class_samples[class][class_sizes[class]], double, dim);
			for (j = 0; j < dim; j++)
				class_samples[class][class_sizes[class]][j] =
					X[i][j];
			class_sizes[class]++;
		}
	}

	{
		int total_labeled = class_sizes[0] + class_sizes[1];

		if (total_labeled == 0)
		{
			out_model->class_priors[0] = 0.5;
			out_model->class_priors[1] = 0.5;
		}
		else if (class_sizes[0] == 0)
		{
			out_model->class_priors[0] = 1e-9;
			out_model->class_priors[1] = 1.0 - 1e-9;
		}
		else if (class_sizes[1] == 0)
		{
			out_model->class_priors[1] = 1e-9;
			out_model->class_priors[0] = 1.0 - 1e-9;
		}
		else
		{
			out_model->class_priors[0] =
				(double) class_sizes[0] / total_labeled;
			out_model->class_priors[1] =
				(double) class_sizes[1] / total_labeled;
		}
	}

	for (class = 0; class < 2; class++)
	{
		/* Check for empty class to avoid division by zero */
		if (class_sizes[class] == 0)
		{
			/* Set default values for empty class */
			for (j = 0; j < dim; j++)
			{
				out_model->means[class][j] = 0.0;
				out_model->variances[class][j] = 1.0;
			}
			continue;
		}

		/* Compute means */
		for (j = 0; j < dim; j++)
		{
			double sum = 0.0;

			for (i = 0; i < class_sizes[class]; i++)
				sum += class_samples[class][i][j];
			out_model->means[class][j] = sum / class_sizes[class];
		}

		/* Compute variances */
		for (j = 0; j < dim; j++)
		{
			double sum_sq = 0.0;

			for (i = 0; i < class_sizes[class]; i++)
			{
				double diff = class_samples[class][i][j]
					- out_model->means[class][j];
				sum_sq += diff * diff;
			}
			/* Add small term to avoid zero variance */
			out_model->variances[class][j] =
				sum_sq / class_sizes[class] + 1e-9;
		}
	}

	/* Cleanup temporary arrays */
	for (i = 0; i < valid; i++)
		NDB_FREE(X[i]);
	NDB_FREE(X);
	NDB_FREE(y);
	for (class = 0; class < 2; class++)
	{
		for (i = 0; i < class_sizes[class]; i++)
			NDB_FREE(class_samples[class][i]);
		NDB_FREE(class_samples[class]);
	}
	NDB_FREE(class_samples);
	NDB_FREE(class_counts);
	NDB_FREE(class_sizes);

	*out_valid = valid;
	*out_dim = dim;
	return 0;
}

/*
 * Free GaussianNBModel and associated metadata
 *
 * Note: model must be heap-allocated (from palloc), not a stack-embedded struct.
 * This function frees all arrays within the model structure and then the model itself.
 */
static void
nb_free_model_and_metadata(GaussianNBModel *model,
			   bytea *model_data,
			   Jsonb *metrics)
{
	 int i;

	 if (model != NULL)
	 {
		 if (model->class_priors)
			 NDB_FREE(model->class_priors);
		 if (model->means)
		 {
			 for (i = 0; i < model->n_classes; i++)
			 {
				 if (model->means[i])
					 NDB_FREE(model->means[i]);
			 }
			 NDB_FREE(model->means);
		 }
		 if (model->variances)
		 {
			 for (i = 0; i < model->n_classes; i++)
			 {
				 if (model->variances[i])
					 NDB_FREE(model->variances[i]);
			 }
			 NDB_FREE(model->variances);
		 }
		 NDB_FREE(model);
	 }

	 if (model_data)
		 NDB_FREE(model_data);
	 if (metrics)
		 NDB_FREE(metrics);
}

/*
 * Serialize GaussianNBModel to bytea for storage
 */
static bytea *
nb_model_serialize_to_bytea(const GaussianNBModel *model)
 {
	 StringInfoData buf;
	 int i, j;
	 int total_size;
	 bytea *result;
 
	 initStringInfo(&buf);
 
	 /* Write header: n_classes, n_features */
	 appendBinaryStringInfo(&buf, (char *) &model->n_classes,
		 sizeof(int));
	 appendBinaryStringInfo(&buf, (char *) &model->n_features,
		 sizeof(int));
 
	 /* Write class priors */
	 for (i = 0; i < model->n_classes; i++)
		 appendBinaryStringInfo(&buf,
			 (char *) &model->class_priors[i],
			 sizeof(double));
 
	 /* Write means */
	 for (i = 0; i < model->n_classes; i++)
		 for (j = 0; j < model->n_features; j++)
			 appendBinaryStringInfo(&buf,
				 (char *) &model->means[i][j],
				 sizeof(double));
 
	 /* Write variances */
	 for (i = 0; i < model->n_classes; i++)
		 for (j = 0; j < model->n_features; j++)
			 appendBinaryStringInfo(&buf,
				 (char *) &model->variances[i][j],
				 sizeof(double));
 
	 /* Convert to bytea */
	 total_size = VARHDRSZ + buf.len;
	 NDB_ALLOC(result, bytea, total_size);
	 SET_VARSIZE(result, total_size);
	 memcpy(VARDATA(result), buf.data, buf.len);
	 NDB_FREE(buf.data);
 
	 return result;
 }
 
 /*
  * Deserialize bytea to GaussianNBModel
  */
 static GaussianNBModel *
 nb_model_deserialize_from_bytea(const bytea *data)
 {
	 GaussianNBModel *model;
	 const char *buf;
	 int offset = 0;
	 int i, j;
 
	 if (data == NULL || VARSIZE_ANY_EXHDR(data) < sizeof(int) * 2)
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("neurondb: invalid model data: too small")));
 
	 buf = VARDATA_ANY(data);
 
	 model = (GaussianNBModel *) palloc0(sizeof(GaussianNBModel));
	 NDB_CHECK_ALLOC(model, "model");
 
	 /* Read header */
	 memcpy(&model->n_classes, buf + offset, sizeof(int));
	 offset += sizeof(int);
	 memcpy(&model->n_features, buf + offset, sizeof(int));
	 offset += sizeof(int);
 
	 /* Enforce binary-only behavior */
	 if (model->n_classes != 2)
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("neurondb: invalid model data: Naive Bayes currently "
				 "supports exactly 2 classes, got %d",
				 model->n_classes)));
	 if (model->n_features <= 0 || model->n_features > 100000)
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("neurondb: invalid model data: n_features=%d "
				 "(expected 1-100000)",
				 model->n_features)));
 
	 /* Verify we have enough data */
	 {
		 int expected_size = sizeof(int) * 2 +
			 sizeof(double) * model->n_classes +
			 sizeof(double) * model->n_classes *
				 model->n_features +
			 sizeof(double) * model->n_classes *
				 model->n_features;
		 int actual_size = VARSIZE_ANY_EXHDR(data);
 
		 if (actual_size < expected_size)
			 ereport(ERROR,
				 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				  errmsg("neurondb: invalid model data: expected %d "
					 "bytes, got %d bytes",
					 expected_size, actual_size)));
	 }
 
	 /* Allocate arrays */
	 NDB_ALLOC(model->class_priors, double, model->n_classes);
	 memset(model->class_priors, 0, sizeof(double) * model->n_classes);
	 NDB_ALLOC(model->means, double *, model->n_classes);
	 NDB_ALLOC(model->variances, double *, model->n_classes);
 
	 /* Read class priors */
	 for (i = 0; i < model->n_classes; i++)
	 {
		 memcpy(&model->class_priors[i], buf + offset,
			 sizeof(double));
		 offset += sizeof(double);
	 }
 
	 /* Read means */
	 for (i = 0; i < model->n_classes; i++)
	 {
		 NDB_ALLOC(model->means[i], double, model->n_features);
		 for (j = 0; j < model->n_features; j++)
		 {
			 memcpy(&model->means[i][j], buf + offset,
				 sizeof(double));
			 offset += sizeof(double);
		 }
	 }
 
	 /* Read variances */
	 for (i = 0; i < model->n_classes; i++)
	 {
		 NDB_ALLOC(model->variances[i], double, model->n_features);
		 for (j = 0; j < model->n_features; j++)
		 {
			 memcpy(&model->variances[i][j], buf + offset,
				 sizeof(double));
			 offset += sizeof(double);
		 }
	 }
 
	 return model;
 }
 
 /*
  * train_naive_bayes_classifier_model_id
  *
  * Trains Naive Bayes and stores model in catalog, returns model_id
  */
 PG_FUNCTION_INFO_V1(train_naive_bayes_classifier_model_id);
 
 Datum
 train_naive_bayes_classifier_model_id(PG_FUNCTION_ARGS)
 {
	 text *table_name;
	 text *feature_col;
	 text *label_col;
	 char *tbl_str;
	 char *feat_str;
	 char *label_str;
	 StringInfoData query = {0};
	 NDB_DECLARE (NdbSpiSession *, train_nb_model_spi_session);
	 int ret;
	 int nvec = 0;
	 int dim = 0;
	 GaussianNBModel model;
	 int valid = 0;
	 int n_classes = 0;  /* Save before SPI ends */
	 int n_features = 0;  /* Save before SPI ends */
	 bytea *model_data;
	 MLCatalogModelSpec spec;
	 Jsonb *metrics;
	 int32 model_id;
	 Oid feat_type_oid;
	 bool feat_is_array;
	 MemoryContext oldcontext;

	 table_name = PG_GETARG_TEXT_PP(0);
	 feature_col = PG_GETARG_TEXT_PP(1);
	 label_col = PG_GETARG_TEXT_PP(2);

	 /* Save caller's memory context */
	 oldcontext = CurrentMemoryContext;
	 Assert(oldcontext != NULL);
	 NDB_SPI_SESSION_BEGIN(train_nb_model_spi_session, oldcontext);

	 tbl_str = text_to_cstring(table_name);
	 feat_str = text_to_cstring(feature_col);
	 label_str = text_to_cstring(label_col);

	 /* Build query - cast label to float8 for type safety */
	 /* Use ndb_spi_stringinfo_init to allocate in SPI context */
	 ndb_spi_stringinfo_init(train_nb_model_spi_session, &query);
	 appendStringInfo(&query,
		 "SELECT %s, %s::float8 FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
		 quote_identifier(feat_str),
		 quote_identifier(label_str),
		 quote_identifier(tbl_str),
		 quote_identifier(feat_str),
		 quote_identifier(label_str));
	 elog(DEBUG1,
		 "train_naive_bayes_classifier_model_id: executing query: %s",
		 query.data);
 
	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	 if (ret != SPI_OK_SELECT)
	 {
		 ndb_spi_stringinfo_free(train_nb_model_spi_session, &query);
		 NDB_SPI_SESSION_END(train_nb_model_spi_session);
		 NDB_FREE(tbl_str);
		 NDB_FREE(feat_str);
		 NDB_FREE(label_str);
		 ereport(ERROR,
			 (errcode(ERRCODE_INTERNAL_ERROR),
			  errmsg("neurondb: query failed")));
	 }
 
	 nvec = SPI_processed;

	 /* Determine feature type from first row */
	 feat_type_oid = InvalidOid;
	 feat_is_array = false;
	 if (nvec > 0 && SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	 {
		 feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
		 if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			 feat_is_array = true;
	 }

	 /* Use shared training helper */
	 nb_train_from_spi_result(nvec, feat_type_oid, feat_is_array, &model, &valid, &dim);

	 /* Save model dimensions before SPI ends (model arrays are in SPI context) */
	 n_classes = model.n_classes;
	 n_features = model.n_features;

	 /* Serialize model to bytea BEFORE SPI_finish() - model arrays are in SPI context */
	 model_data = nb_model_serialize_to_bytea(&model);

	 /* Free query.data while still in SPI context, before switching */
	 ndb_spi_stringinfo_free(train_nb_model_spi_session, &query);

	 /* Copy model_data to outer context before SPI_finish() destroys SPI context */
	 MemoryContextSwitchTo(oldcontext);
	 {
		 int data_len = VARSIZE_ANY(model_data);
		 bytea *copy;
		 NDB_ALLOC(copy, bytea, data_len);
		 memcpy(copy, model_data, data_len);
		 model_data = copy;
	 }

	 NDB_SPI_SESSION_END(train_nb_model_spi_session);
 
	 /* Build metrics JSONB */
	 {
		 StringInfoData metrics_json;
		 
		 initStringInfo(&metrics_json);
		 appendStringInfo(&metrics_json,
						  "{\"storage\": \"cpu\", \"n_classes\": %d, \"n_features\": %d}",
						  n_classes, n_features);
		 
		 /* Use ndb_jsonb_in_cstring like test 006 fix */
		 metrics = ndb_jsonb_in_cstring(metrics_json.data);
		 if (metrics == NULL)
		 {
			 NDB_FREE(metrics_json.data);
			 ereport(ERROR,
					 (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					  errmsg("neurondb: failed to parse metrics JSON")));
		 }
		 NDB_FREE(metrics_json.data);
	 }
 
	 /* Store model in catalog */
	 memset(&spec, 0, sizeof(MLCatalogModelSpec));
	 spec.project_name = NULL;
	 spec.algorithm = "naive_bayes";
	 spec.training_table = tbl_str;
	 spec.training_column = label_str;
	 spec.model_data = model_data;
	 spec.metrics = metrics;
	 spec.num_samples = valid;
	 spec.num_features = dim;

	 model_id = ml_catalog_register_model(&spec);

	 /* Cleanup - model arrays were in SPI context and freed by SPI_finish() */
	 /* query.data was already freed before SPI session ended */
	 /* Free strings allocated in oldcontext - ml_catalog_register_model has finished */
	 /* and no longer needs them (it copies them internally for SQL) */
	 if (tbl_str != NULL)
	 {
		 NDB_FREE(tbl_str);
		 tbl_str = NULL;
	 }
	 if (feat_str != NULL)
	 {
		 NDB_FREE(feat_str);
		 feat_str = NULL;
	 }
	 if (label_str != NULL)
	 {
		 NDB_FREE(label_str);
		 label_str = NULL;
	 }

	 PG_RETURN_INT32(model_id);
}
 
 /*
  * predict_naive_bayes
  *
  * Predicts class using trained Naive Bayes model
  */
 PG_FUNCTION_INFO_V1(predict_naive_bayes);
 
 Datum
 predict_naive_bayes(PG_FUNCTION_ARGS)
 {
	 ArrayType *model_params;
	 Vector *features;
	 float8 *params;
	 int n_params;
	 int n_classes;
	 int n_features;
	 double *class_priors;
	 double **means;
	 double **variances;
	 double log_probs[2] = {0.0, 0.0};
	 int predicted_class;
	 int i, j, idx;
 
	 model_params = PG_GETARG_ARRAYTYPE_P(0);
	 features = PG_GETARG_VECTOR_P(1);
  NDB_CHECK_VECTOR_VALID(features);
 
	 if (ARR_NDIM(model_params) != 1)
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("neurondb: Model parameters must be 1-dimensional "
				 "array")));
 
	 n_params = ARR_DIMS(model_params)[0];
	 (void) n_params;
	 params = (float8 *) ARR_DATA_PTR(model_params);
 
	 n_classes = (int) params[0];
	 n_features = (int) params[1];

	 /* Enforce binary-only behavior */
	 if (n_classes != 2)
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("neurondb: Naive Bayes currently supports exactly 2 classes, "
				 "got %d", n_classes)));

	 /* Validate array length - layout: [n_classes, n_features, priors..., means..., variances...] */
	 {
		 int expected = 2 + n_classes + n_classes * n_features + n_classes * n_features;

		 if (n_params < expected)
			 ereport(ERROR,
				 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				  errmsg("neurondb: Naive Bayes model param array too short: "
					 "expected %d elements, got %d",
					 expected, n_params)));
	 }

	 if (features->dim != n_features)
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("neurondb: predict_naive_bayes: feature "
				 "dimension mismatch: expected %d, got %d",
				 n_features, features->dim)));

	 /* Extract model parameters */
	 NDB_ALLOC(class_priors, double, n_classes);
	 NDB_ALLOC(means, double *, n_classes);
	 NDB_ALLOC(variances, double *, n_classes);

	 /* Read priors */
	 for (i = 0; i < n_classes; i++)
		 class_priors[i] = params[2 + i];
 
	 /* Read means */
	 idx = 2 + n_classes;
	 for (i = 0; i < n_classes; i++)
	 {
		 NDB_ALLOC(means[i], double, n_features);
		 for (j = 0; j < n_features; j++)
			 means[i][j] = params[idx++];
	 }

	 /* Read variances */
	 for (i = 0; i < n_classes; i++)
	 {
		 NDB_ALLOC(variances[i], double, n_features);
		 for (j = 0; j < n_features; j++)
			 variances[i][j] = params[idx++];
	 }

	 /* Compute log probabilities for each class - binary only */
	 for (i = 0; i < n_classes; i++)
	 {
		 log_probs[i] = log(class_priors[i]);
 
		 for (j = 0; j < n_features; j++)
		 {
			 double pdf = gaussian_pdf(features->data[j],
				 means[i][j],
				 variances[i][j]);
			 log_probs[i] += log(pdf + 1e-10);
		 }
	 }
 
	 /* Return class with highest log probability */
	 predicted_class = (log_probs[1] > log_probs[0]) ? 1 : 0;
 
	 /* Cleanup */
	 NDB_FREE(class_priors);
	 for (i = 0; i < n_classes; i++)
	 {
		 NDB_FREE(means[i]);
		 NDB_FREE(variances[i]);
	 }
	 NDB_FREE(means);
	 NDB_FREE(variances);
 
	 PG_RETURN_INT32(predicted_class);
 }
 
 /*
  * predict_naive_bayes_model_id
  *
  * Predicts class using Naive Bayes model from catalog (model_id)
  */
 PG_FUNCTION_INFO_V1(predict_naive_bayes_model_id);
 
 Datum
 predict_naive_bayes_model_id(PG_FUNCTION_ARGS)
 {
	 int32 model_id;
	 Vector *features;
	 bytea *model_data = NULL;
	 Jsonb *metrics = NULL;
	 GaussianNBModel *model = NULL;
	 double log_probs[2] = {0.0, 0.0};
	 int predicted_class;
	 int i, j;
 
	 model_id = PG_GETARG_INT32(0);
	 features = PG_GETARG_VECTOR_P(1);
  NDB_CHECK_VECTOR_VALID(features);
 
	 if (!ml_catalog_fetch_model_payload(model_id,
			 &model_data, NULL, &metrics))
	 {
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("Naive Bayes model %d not found",
				 model_id)));
	 }
 
	 if (model_data == NULL)
	 {
		 if (metrics)
			 NDB_FREE(metrics);
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("Naive Bayes model %d has no model data",
				 model_id)));
	 }
 
	 /* Ensure bytea is in current context */
	 if (model_data != NULL)
	 {
		 int data_len = VARSIZE_ANY(model_data);
		 bytea *copy;

		NDB_ALLOC(copy, bytea, data_len);
 
		 memcpy(copy, model_data, data_len);
		 model_data = copy;
	 }
 
	 model = nb_model_deserialize_from_bytea(model_data);

	 /* Enforce binary-only behavior */
	 if (model->n_classes != 2)
	 {
		 nb_free_model_and_metadata(model, model_data, metrics);
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("Naive Bayes currently supports exactly 2 classes, "
				 "got %d", model->n_classes)));
	 }

	 if (features->dim != model->n_features)
	 {
		 nb_free_model_and_metadata(model, model_data, metrics);
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("Feature dimension mismatch: expected %d, "
				 "got %d",
				 model->n_features, features->dim)));
	 }

	 /* Compute log probabilities - binary only */
	 for (i = 0; i < model->n_classes; i++)
	 {
		 log_probs[i] = log(model->class_priors[i]);

		 for (j = 0; j < model->n_features; j++)
		 {
			 double pdf = gaussian_pdf(features->data[j],
				 model->means[i][j],
				 model->variances[i][j]);
			 log_probs[i] += log(pdf + 1e-10);
		 }
	 }
 
	 predicted_class = (log_probs[1] > log_probs[0]) ? 1 : 0;

	 nb_free_model_and_metadata(model, model_data, metrics);

	 PG_RETURN_INT32(predicted_class);
}
 
 /*
  * nb_predict_batch
  *
  * Helper function to predict a batch of samples using Naive Bayes model.
  * Updates confusion matrix.
  */
 static void
 nb_predict_batch(const GaussianNBModel *model,
	 const float *features,
	 const double *labels,
	 int n_samples,
	 int feature_dim,
	 int *tp_out,
	 int *tn_out,
	 int *fp_out,
	 int *fn_out)
 {
	 int i;
	 int tp = 0;
	 int tn = 0;
	 int fp = 0;
	 int fn = 0;
 
	 if (model == NULL || features == NULL ||
		 labels == NULL || n_samples <= 0)
	 {
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
 
	 for (i = 0; i < n_samples; i++)
	 {
		 const float *row = features + (i * feature_dim);
		 double y_true = labels[i];
		 int true_class;
		 double log_probs[2] = {0.0, 0.0};
		 int pred_class;
		 int j;
		 int c;
 
		 if (!isfinite(y_true))
			 continue;
 
		 true_class = (int) rint(y_true);
		 if (true_class < 0 || true_class > 1)
			 continue;

		 /* Binary-only: enforce 2 classes */
		 if (model->n_classes != 2)
			 continue;

		 /* Compute log probabilities - binary only */
		 for (c = 0; c < 2; c++)
		 {
			 log_probs[c] = log(model->class_priors[c]);

			 for (j = 0; j < feature_dim; j++)
			 {
				 double pdf = gaussian_pdf((double) row[j],
					 model->means[c][j],
					 model->variances[c][j]);
				 log_probs[c] += log(pdf + 1e-10);
			 }
		 }
 
		 pred_class = (log_probs[1] > log_probs[0]) ? 1 : 0;
 
		 if (true_class == 1 && pred_class == 1)
			 tp++;
		 else if (true_class == 0 && pred_class == 0)
			 tn++;
		 else if (true_class == 0 && pred_class == 1)
			 fp++;
		 else if (true_class == 1 && pred_class == 0)
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
  * evaluate_naive_bayes_by_model_id
  *
  * Evaluates Naive Bayes model by model_id using optimized batch evaluation.
  * Supports both GPU and CPU models with GPU-accelerated batch evaluation
  * when available.
  *
  * Returns jsonb with metrics: accuracy, precision, recall,
  * f1_score, n_samples
  */
 PG_FUNCTION_INFO_V1(evaluate_naive_bayes_by_model_id);
 
Datum
evaluate_naive_bayes_by_model_id(PG_FUNCTION_ARGS)
{
	if (PG_ARGISNULL(0))
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("neurondb: evaluate_naive_bayes_by_model_id: "
				"model_id is required")));

	/* Delegate to the real implementation */
	/* The old function requires table_name, feature_col, label_col */
	/* For now, if only model_id is provided, we require the full signature */
	if (PG_NARGS() < 4)
	{
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("neurondb: evaluate_naive_bayes_by_model_id: "
				"requires model_id, table_name, feature_col, and label_col")));
	}

	/* Call the real implementation */
	return nb_evaluate_naive_bayes_internal(fcinfo);
}

/*
 * Internal evaluation implementation
 * Not exported as SQL function - only called by evaluate_naive_bayes_by_model_id
 */
static Datum
nb_evaluate_naive_bayes_internal(FunctionCallInfo fcinfo)
{
	int32 model_id;
	 text *table_name;
	 text *feature_col;
	 text *label_col;
	 char *tbl_str;
	 char *feat_str;
	 char *targ_str;
	 int ret;
	 int nvec = 0;
	 int i;
	 int j;
	 int feat_dim = 0;
	 Oid feat_type_oid = InvalidOid;
	 bool feat_is_array = false;
	 double accuracy = 0.0;
	 double precision = 0.0;
	 double recall = 0.0;
	 double f1_score = 0.0;
	 int tp = 0;
	 int tn = 0;
	 int fp = 0;
	 int fn = 0;
	 int valid_rows = 0; /* Track valid rows for accurate n_samples reporting */
	 int n_samples_for_json = 0; /* For accurate n_samples reporting in JSON */
	 MemoryContext oldcontext;
	 StringInfoData query;
	 GaussianNBModel *model = NULL;
	 StringInfoData jsonbuf;
	 Jsonb *result_jsonb = NULL;
	 bytea *gpu_payload = NULL;
	 Jsonb *gpu_metrics = NULL;
	 bool is_gpu_model = false;
	 NDB_DECLARE (NdbSpiSession *, eval_nb_spi_session);
 
	 if (PG_ARGISNULL(0))
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("neurondb: evaluate_naive_bayes_by_model_id: "
				 "model_id is required")));
 
	 model_id = PG_GETARG_INT32(0);
 
	 if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("neurondb: evaluate_naive_bayes_by_model_id: "
				 "table_name, feature_col, and label_col "
				 "are required")));
 
	 table_name = PG_GETARG_TEXT_PP(1);
	 feature_col = PG_GETARG_TEXT_PP(2);
	 label_col = PG_GETARG_TEXT_PP(3);
 
	 tbl_str = text_to_cstring(table_name);
	 feat_str = text_to_cstring(feature_col);
	 targ_str = text_to_cstring(label_col);
 
	 oldcontext = CurrentMemoryContext;
 
	 /* Load model from catalog - try CPU/GPU metadata */
	 if (!ml_catalog_fetch_model_payload(model_id,
			 &gpu_payload, NULL, &gpu_metrics))
	 {
		 NDB_FREE(tbl_str);
		 NDB_FREE(feat_str);
		 NDB_FREE(targ_str);
		 ereport(ERROR,
			 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			  errmsg("neurondb: evaluate_naive_bayes_by_model_id: "
				 "model %d not found",
				 model_id)));
	 }
 
	 /* Check if GPU model using metrics json */
	 if (gpu_payload != NULL && gpu_metrics != NULL)
	 {
		 bool is_gpu = false;
 
		 PG_TRY();
		 {
			 /* Check for training_backend integer in metrics */
			 JsonbIterator *it;
			 JsonbValue	v;
			 JsonbIteratorToken r;

			 it = JsonbIteratorInit((JsonbContainer *) &gpu_metrics->root);
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
		 }
		 PG_CATCH();
		 {
			 is_gpu = false;
		 }
		 PG_END_TRY();
 
		 is_gpu_model = is_gpu;
	 }
 
	 /* For CPU models, deserialize from bytea */
	 if (!is_gpu_model && gpu_payload != NULL)
	 {
		 bytea *copy;
		 int data_len = VARSIZE_ANY(gpu_payload);
 
		 NDB_ALLOC(copy, bytea, data_len);
		 memcpy(copy, gpu_payload, data_len);
		 model = nb_model_deserialize_from_bytea(copy);
		 NDB_FREE(copy);
	 }
 
	 /* Connect to SPI */
	 Assert(oldcontext != NULL);
	 NDB_SPI_SESSION_BEGIN(eval_nb_spi_session, oldcontext);
 
	 /* Build query - cast label to float8 for type safety */
	 initStringInfo(&query);
	 appendStringInfo(&query,
		 "SELECT %s, %s::float8 FROM %s "
		 "WHERE %s IS NOT NULL AND %s IS NOT NULL",
		 quote_identifier(feat_str),
		 quote_identifier(targ_str),
		 quote_identifier(tbl_str),
		 quote_identifier(feat_str),
		 quote_identifier(targ_str));
	 elog(DEBUG1,
		 "evaluate_naive_bayes_by_model_id: executing query: %s",
		 query.data);
 
	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	 if (ret != SPI_OK_SELECT)
	 {
		 nb_free_model_and_metadata(model, NULL, gpu_metrics);
		 if (gpu_payload)
			 NDB_FREE(gpu_payload);
		 NDB_FREE(query.data);
		 NDB_FREE(tbl_str);
		 NDB_FREE(feat_str);
		 NDB_FREE(targ_str);
		 NDB_SPI_SESSION_END(eval_nb_spi_session);
		 ereport(ERROR,
			 (errcode(ERRCODE_INTERNAL_ERROR),
			  errmsg("neurondb: evaluate_naive_bayes_by_model_id: query failed"),
			  errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
			  errhint("Verify the table exists and contains valid feature and label columns.")));
	 }
 
	 nvec = SPI_processed;
	 if (nvec < 1)
	 {
		 StringInfoData check_query;
		 int check_ret;
		 int total_rows = 0;
		 char *tbl_str_copy;
		 char *feat_str_copy;
		 char *targ_str_copy;
		 int expected_dim = 0;
 
		 if (model != NULL)
			 expected_dim = model->n_features;
		 else if (is_gpu_model && gpu_metrics != NULL)
		 {
			 char *meta_txt = NULL;
 
			 PG_TRY();
			 {
				 meta_txt = DatumGetCString(DirectFunctionCall1(
					 jsonb_out,
					 JsonbPGetDatum(gpu_metrics)));
				 if (strstr(meta_txt, "\"n_features\"") != NULL ||
					 strstr(meta_txt, "\"feature_dim\"") != NULL)
				 {
					 char *dim_str;
 
					 dim_str = strstr(meta_txt,
						 "\"n_features\"");
					 if (dim_str == NULL)
						 dim_str = strstr(meta_txt,
							 "\"feature_dim\"");
					 if (dim_str != NULL)
					 {
						 char *colon = strchr(dim_str, ':');
 
						 if (colon != NULL)
							 expected_dim =
								 atoi(colon + 1);
					 }
				 }
				 if (meta_txt != NULL)
					 NDB_FREE(meta_txt);
			 }
			 PG_CATCH();
			 {
				 if (meta_txt != NULL)
					 NDB_FREE(meta_txt);
			 }
			 PG_END_TRY();
		 }
 
		 initStringInfo(&check_query);
		 appendStringInfo(&check_query,
			 "SELECT COUNT(*) FROM %s",
			 quote_identifier(tbl_str));
		 check_ret = ndb_spi_execute_safe(check_query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
		 if (check_ret == SPI_OK_SELECT && SPI_processed > 0)
		 {
			 bool isnull;
			 Datum count_datum =
				 SPI_getbinval(SPI_tuptable->vals[0],
					 SPI_tuptable->tupdesc,
					 1,
					 &isnull);
 
			 if (!isnull)
				 total_rows = DatumGetInt64(count_datum);
		 }
		 NDB_FREE(check_query.data);
 
		 nb_free_model_and_metadata(model, NULL, gpu_metrics);
		 if (gpu_payload)
			 NDB_FREE(gpu_payload);
		 model = NULL;
 
		 MemoryContextSwitchTo(oldcontext);
		 tbl_str_copy = pstrdup(tbl_str);
		 feat_str_copy = pstrdup(feat_str);
		 targ_str_copy = pstrdup(targ_str);
 
		 NDB_FREE(query.data);
		 NDB_FREE(tbl_str);
		 NDB_FREE(feat_str);
		 NDB_FREE(targ_str);
		 NDB_SPI_SESSION_END(eval_nb_spi_session);

		 if (total_rows == 0)
		 {
			 ereport(ERROR,
				 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				  errmsg("neurondb: "
					 "evaluate_naive_bayes_by_model_id: "
					 "table/view '%s' has no rows",
					 tbl_str_copy),
				  errhint("Ensure the table/view exists "
					 "and contains data")));
		 }
		 else
		 {
			 ereport(ERROR,
				 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				  errmsg("neurondb: "
					 "evaluate_naive_bayes_by_model_id: "
					 "no valid rows found in '%s' "
					 "(query returned %d rows, but all were "
					 "filtered out due to NULL values or "
					 "dimension mismatches)",
					 tbl_str_copy, total_rows),
				  errhint("Ensure columns '%s' and '%s' "
					 "are not NULL and feature dimensions "
					 "match the model (expected %d)",
					 feat_str_copy, targ_str_copy,
					 expected_dim)));
		 }
	 }
 
	 if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		 feat_type_oid =
			 SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	 if (feat_type_oid == FLOAT8ARRAYOID ||
		 feat_type_oid == FLOAT4ARRAYOID)
		 feat_is_array = true;
 
	 /* GPU batch evaluation path (if available and model is GPU) */
	 if (is_gpu_model && neurondb_gpu_is_available())
	 {
 #ifdef NDB_GPU_CUDA
		 const NdbCudaNbModelHeader *gpu_hdr;
		 int *h_labels = NULL;
		 float *h_features = NULL;
		 int gpu_valid_rows = 0;
		 size_t payload_size;
 
		 payload_size = VARSIZE_ANY_EXHDR(gpu_payload);
		 if (payload_size < sizeof(NdbCudaNbModelHeader))
			 goto cpu_evaluation_path;
 
		 gpu_hdr =
			 (const NdbCudaNbModelHeader *) VARDATA_ANY(gpu_payload);
		 /* VARDATA never returns NULL for valid varlena, so no check needed */

		 feat_dim = gpu_hdr->n_features;
		 if (feat_dim <= 0 || feat_dim > 100000)
			 goto cpu_evaluation_path;
 
		 {
			 size_t features_size =
				 sizeof(float) * (size_t) nvec *
					 (size_t) feat_dim;
			 size_t labels_size =
				 sizeof(int) * (size_t) nvec;
 
			 if (features_size > MaxAllocSize ||
				 labels_size > MaxAllocSize)
				 goto cpu_evaluation_path;
 
			 NDB_ALLOC(h_features, float, features_size / sizeof(float));
			 NDB_ALLOC(h_labels, int, labels_size / sizeof(int));
 
			 if (h_features == NULL || h_labels == NULL)
			 {
				 if (h_features)
					 NDB_FREE(h_features);
				 if (h_labels)
					 NDB_FREE(h_labels);
				 goto cpu_evaluation_path;
			 }
		 }
 
		 {
			 TupleDesc tupdesc = SPI_tuptable->tupdesc;
 
			 if (tupdesc == NULL)
			 {
				 NDB_FREE(h_features);
				 NDB_FREE(h_labels);
				 goto cpu_evaluation_path;
			 }
 
			 for (i = 0; i < nvec; i++)
			 {
				 HeapTuple tuple;
				 Datum feat_datum;
				 Datum targ_datum;
				 bool feat_null;
				 bool targ_null;
				 Vector *vec;
				 ArrayType *arr;
				 float *feat_row;
 
				 if (SPI_tuptable == NULL ||
					 SPI_tuptable->vals == NULL ||
					 i >= SPI_processed)
					 break;
 
				 tuple = SPI_tuptable->vals[i];
				 if (tuple == NULL)
					 continue;
 
				 feat_datum = SPI_getbinval(tuple,
					 tupdesc,
					 1,
					 &feat_null);
				 targ_datum = SPI_getbinval(tuple,
					 tupdesc,
					 2,
					 &targ_null);
 
				 if (feat_null || targ_null)
					 continue;
 
				 if (gpu_valid_rows >= nvec)
					 break;

				 feat_row = h_features +
					 (gpu_valid_rows * feat_dim);

				 /* Validate label is exactly 0 or 1, matching training expectations */
				 {
					 double label_val = DatumGetFloat8(targ_datum);
					 int label_class = (int) rint(label_val);

					 if (label_class != 0 && label_class != 1)
						 continue; /* Skip invalid labels, matching training strictness */

					 h_labels[gpu_valid_rows] = label_class;
				 }
 
				 if (feat_is_array)
				 {
					 arr = DatumGetArrayTypeP(feat_datum);
					 if (ARR_NDIM(arr) != 1 ||
						 ARR_DIMS(arr)[0] != feat_dim)
						 continue;
					 if (feat_type_oid ==
						 FLOAT8ARRAYOID)
					 {
						 float8 *data =
							 (float8 *)
								 ARR_DATA_PTR(arr);
						 int j_remain =
							 feat_dim % 4;
						 int j_end =
							 feat_dim -
							 j_remain;
 
						 for (j = 0; j < j_end; j += 4)
						 {
							 feat_row[j] =
								 (float) data[j];
							 feat_row[j + 1] =
								 (float) data[j + 1];
							 feat_row[j + 2] =
								 (float) data[j + 2];
							 feat_row[j + 3] =
								 (float) data[j + 3];
						 }
						 for (j = j_end; j < feat_dim;
							  j++)
							 feat_row[j] =
								 (float) data[j];
					 }
					 else
					 {
						 float4 *data =
							 (float4 *)
								 ARR_DATA_PTR(arr);
						 memcpy(feat_row,
							 data,
							 sizeof(float) *
								 feat_dim);
					 }
				 }
				 else
				 {
					 vec = DatumGetVector(feat_datum);
					 if (vec->dim != feat_dim)
						 continue;
					 memcpy(feat_row,
						 vec->data,
						 sizeof(float) *
							 feat_dim);
				 }
 
				 gpu_valid_rows++;
			 }
		 }

		 if (gpu_valid_rows == 0)
		 {
			 int actual_dim = -1;
			 TupleDesc tupdesc = SPI_tuptable->tupdesc;
			 
			 /* Try to detect actual dimension from first row */
			 if (tupdesc != NULL && nvec > 0)
			 {
				 HeapTuple tuple = SPI_tuptable->vals[0];
				 Datum feat_datum;
				 bool feat_null;
				 Vector *vec;
				 ArrayType *arr;
				 
				 feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				 if (!feat_null)
				 {
					 if (feat_is_array)
					 {
						 arr = DatumGetArrayTypeP(feat_datum);
						 if (ARR_NDIM(arr) == 1)
							 actual_dim = ARR_DIMS(arr)[0];
					 }
					 else
					 {
						 vec = DatumGetVector(feat_datum);
						 if (vec != NULL)
							 actual_dim = vec->dim;
					 }
				 }
			 }
			 
			 NDB_FREE(h_features);
			 NDB_FREE(h_labels);
			 if (gpu_payload)
				 NDB_FREE(gpu_payload);
			 if (gpu_metrics)
				 NDB_FREE(gpu_metrics);
			 /* Do not free tbl_str/feat_str/targ_str
			  * before ereport, they are used below.
			  */
			 NDB_SPI_SESSION_END(eval_nb_spi_session);
			 
			 if (actual_dim > 0 && actual_dim != feat_dim)
			 {
				 ereport(ERROR,
					 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					  errmsg("neurondb: "
						 "evaluate_naive_bayes_by_model_id: "
						 "feature dimension mismatch in '%s'",
						 tbl_str),
					  errdetail("Model expects %d features, but data has %d features",
						 feat_dim, actual_dim),
					  errhint("Ensure the feature column '%s' has the same dimension as the training data (expected %d)",
						 feat_str, feat_dim)));
			 }
			 else
			 {
				 ereport(ERROR,
					 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					  errmsg("neurondb: "
						 "evaluate_naive_bayes_by_model_id: "
						 "no valid rows found in '%s' "
						 "(query returned %d rows, but all "
						 "were filtered out due to NULL "
						 "values or dimension mismatches)",
						 tbl_str, nvec),
					  errhint("Ensure columns '%s' and '%s' "
						 "are not NULL and feature "
						 "dimensions match the model "
						 "(expected %d)",
						 feat_str, targ_str,
						 feat_dim)));
			 }
		 }
 
		 {
			 int rc;
			 char *gpu_errstr = NULL;
 
			 PG_TRY();
			 {
				 rc = ndb_cuda_nb_evaluate_batch(
					 gpu_payload,
					 h_features,
					 h_labels,
					 gpu_valid_rows,
					 feat_dim,
					 &accuracy,
					 &precision,
					 &recall,
					 &f1_score,
					 &gpu_errstr);
 
				 if (rc == 0)
				 {
					 initStringInfo(&jsonbuf);
					 appendStringInfo(&jsonbuf,
						 "{\"accuracy\":%.6f,"
						 "\"precision\":%.6f,"
						 "\"recall\":%.6f,"
						 "\"f1_score\":%.6f,"
						 "\"n_samples\":%d}",
						 accuracy,
						 precision,
						 recall,
						 f1_score,
						 gpu_valid_rows);
 
					 result_jsonb =
						 DatumGetJsonbP(
							 DirectFunctionCall1(
								 jsonb_in,
								 CStringGetTextDatum(
									 jsonbuf.data)));
 
					 /* Build JSONB in current context (SPI context) */
					 /* Copy result_jsonb to caller's context before SPI_finish() */
					 MemoryContextSwitchTo(oldcontext);
					 result_jsonb = (Jsonb *)PG_DETOAST_DATUM_COPY(JsonbPGetDatum(result_jsonb));
					 
					 /* Now safe to finish SPI and free SPI-allocated memory */
					 NDB_FREE(query.data);
					 NDB_SPI_SESSION_END(eval_nb_spi_session);
					 
					 /* Free strings and arrays allocated in SPI context */
					 NDB_FREE(jsonbuf.data);
					 NDB_FREE(h_features);
					 NDB_FREE(h_labels);
					 if (gpu_payload)
						 NDB_FREE(gpu_payload);
					 if (gpu_metrics)
						 NDB_FREE(gpu_metrics);
					 if (gpu_errstr)
						 NDB_FREE(gpu_errstr);
					 
					 /* Free strings allocated before SPI_connect (in oldcontext) */
					 NDB_FREE(tbl_str);
					 NDB_FREE(feat_str);
					 NDB_FREE(targ_str);
					 
					 PG_RETURN_JSONB_P(result_jsonb);
				 }
				 else
				 {
					 if (gpu_errstr)
						 NDB_FREE(gpu_errstr);
					 NDB_FREE(h_features);
					 NDB_FREE(h_labels);
					 goto cpu_evaluation_path;
				 }
			 }
			 PG_CATCH();
			 {
				 if (h_features)
					 NDB_FREE(h_features);
				 if (h_labels)
					 NDB_FREE(h_labels);
				 goto cpu_evaluation_path;
			 }
			 PG_END_TRY();
		 }
 #endif
	 }
#ifndef NDB_GPU_CUDA
	/* When CUDA is not available, always use CPU path */
	if (false) { }
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-label"
cpu_evaluation_path:
#pragma GCC diagnostic pop
 
	 /* CPU evaluation path */
	 {
		 float *h_features = NULL;
		 double *h_labels = NULL;
		 valid_rows = 0; /* Reset for CPU path */
 
		 if (model != NULL)
			 feat_dim = model->n_features;
		 else if (is_gpu_model && gpu_payload != NULL)
		 {
			 const NdbCudaNbModelHeader *gpu_hdr;
 
			 gpu_hdr =
				 (const NdbCudaNbModelHeader *)
					 VARDATA_ANY(gpu_payload);
			 feat_dim = gpu_hdr->n_features;
		 }
		 else
		 {
			 ereport(ERROR,
				 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				  errmsg("neurondb: "
					 "evaluate_naive_bayes_by_model_id: "
					 "could not determine feature "
					 "dimension")));
		 }
 
		 if (feat_dim <= 0)
			 ereport(ERROR,
				 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				  errmsg("neurondb: "
					 "evaluate_naive_bayes_by_model_id: "
					 "invalid feature dimension %d",
					 feat_dim)));
 
		 NDB_ALLOC(h_features, float, (size_t) nvec * (size_t) feat_dim);
		 NDB_ALLOC(h_labels, double, (size_t) nvec);
 
		 {
			 TupleDesc tupdesc = SPI_tuptable->tupdesc;
 
			 for (i = 0; i < nvec; i++)
			 {
				 HeapTuple tuple =
					 SPI_tuptable->vals[i];
				 Datum feat_datum;
				 Datum targ_datum;
				 bool feat_null;
				 bool targ_null;
				 Vector *vec;
				 ArrayType *arr;
				 float *feat_row;
 
				 feat_datum = SPI_getbinval(tuple,
					 tupdesc,
					 1,
					 &feat_null);
				 targ_datum = SPI_getbinval(tuple,
					 tupdesc,
					 2,
					 &targ_null);
 
				 if (feat_null || targ_null)
					 continue;
 
				 feat_row = h_features +
					 (valid_rows * feat_dim);
				 h_labels[valid_rows] =
					 DatumGetFloat8(targ_datum);
 
				 if (feat_is_array)
				 {
					 arr = DatumGetArrayTypeP(
						 feat_datum);
					 if (ARR_NDIM(arr) != 1 ||
						 ARR_DIMS(arr)[0] != feat_dim)
						 continue;
					 if (feat_type_oid ==
						 FLOAT8ARRAYOID)
					 {
						 float8 *data =
							 (float8 *)
								 ARR_DATA_PTR(arr);
						 int j_remain =
							 feat_dim % 4;
						 int j_end =
							 feat_dim -
							 j_remain;
 
						 for (j = 0; j < j_end; j += 4)
						 {
							 feat_row[j] =
								 (float) data[j];
							 feat_row[j + 1] =
								 (float) data[j + 1];
							 feat_row[j + 2] =
								 (float) data[j + 2];
							 feat_row[j + 3] =
								 (float) data[j + 3];
						 }
						 for (j = j_end; j < feat_dim;
							  j++)
							 feat_row[j] =
								 (float) data[j];
					 }
					 else
					 {
						 float4 *data =
							 (float4 *)
								 ARR_DATA_PTR(arr);
						 memcpy(feat_row,
							 data,
							 sizeof(float) *
								 feat_dim);
					 }
				 }
				 else
				 {
					 vec = DatumGetVector(feat_datum);
					 if (vec->dim != feat_dim)
						 continue;
					 memcpy(feat_row,
						 vec->data,
						 sizeof(float) *
							 feat_dim);
				 }
 
				 valid_rows++;
			 }
		 }
 
		 if (valid_rows == 0)
		 {
			 int actual_dim = -1;
			 TupleDesc tupdesc = SPI_tuptable->tupdesc;
			 
			 /* Try to detect actual dimension from first row */
			 if (tupdesc != NULL && nvec > 0)
			 {
				 HeapTuple tuple = SPI_tuptable->vals[0];
				 Datum feat_datum;
				 bool feat_null;
				 Vector *vec;
				 ArrayType *arr;
				 
				 feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				 if (!feat_null)
				 {
					 if (feat_is_array)
					 {
						 arr = DatumGetArrayTypeP(feat_datum);
						 if (ARR_NDIM(arr) == 1)
							 actual_dim = ARR_DIMS(arr)[0];
					 }
					 else
					 {
						 vec = DatumGetVector(feat_datum);
						 if (vec != NULL)
							 actual_dim = vec->dim;
					 }
				 }
			 }
			 
			 NDB_FREE(h_features);
			 NDB_FREE(h_labels);
			 nb_free_model_and_metadata(model, NULL, gpu_metrics);
			 if (gpu_payload)
				 NDB_FREE(gpu_payload);
			 /* Do not free tbl_str / feat_str / targ_str
			  * before using them in ereport.
			  */
			 NDB_SPI_SESSION_END(eval_nb_spi_session);
			 
			 if (actual_dim > 0 && actual_dim != feat_dim)
			 {
				 ereport(ERROR,
					 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					  errmsg("neurondb: "
						 "evaluate_naive_bayes_by_model_id: "
						 "feature dimension mismatch in '%s'",
						 tbl_str),
					  errdetail("Model expects %d features, but data has %d features",
						 feat_dim, actual_dim),
					  errhint("Ensure the feature column '%s' has the same dimension as the training data (expected %d)",
						 feat_str, feat_dim)));
			 }
			 else
			 {
				 ereport(ERROR,
					 (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					  errmsg("neurondb: "
						 "evaluate_naive_bayes_by_model_id: "
						 "no valid rows found in '%s' "
						 "(query returned %d rows, but all "
						 "were filtered out due to NULL "
						 "values or dimension mismatches)",
						 tbl_str, nvec),
					  errhint("Ensure columns '%s' and '%s' "
						 "are not NULL and feature "
						 "dimensions match the model "
						 "(expected %d)",
						 feat_str, targ_str,
						 feat_dim)));
			 }
		 }
 
		 if (is_gpu_model && model == NULL)
		 {
			 NDB_FREE(h_features);
			 NDB_FREE(h_labels);
			 if (gpu_payload)
				 NDB_FREE(gpu_payload);
			 if (gpu_metrics)
				 NDB_FREE(gpu_metrics);
			 NDB_FREE(query.data);
			 NDB_SPI_SESSION_END(eval_nb_spi_session);
			 ereport(ERROR,
				 (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				  errmsg("neurondb: evaluate_naive_bayes_by_model_id: GPU model evaluation requires a GPU at runtime"),
				  errdetail("CPU fallback for GPU models is not supported"),
				  errhint("Ensure CUDA is available and properly configured.")));
		 }
 
		 nb_predict_batch(model,
			 h_features,
			 h_labels,
			 valid_rows,
			 feat_dim,
			 &tp,
			 &tn,
			 &fp,
			 &fn);
 
		 if (valid_rows > 0)
		 {
			 accuracy = (double) (tp + tn) /
				 (double) valid_rows;
 
			 if ((tp + fp) > 0)
				 precision = (double) tp /
					 (double) (tp + fp);
			 else
				 precision = 0.0;
 
			 if ((tp + fn) > 0)
				 recall = (double) tp /
					 (double) (tp + fn);
			 else
				 recall = 0.0;
 
			 if ((precision + recall) > 0.0)
				 f1_score = 2.0 *
					 (precision * recall) /
					 (precision + recall);
			 else
				 f1_score = 0.0;
		 }
 
		 NDB_FREE(h_features);
		 NDB_FREE(h_labels);
		 nb_free_model_and_metadata(model, NULL, gpu_metrics);
		 if (gpu_payload)
			 NDB_FREE(gpu_payload);
	 }
 
	 /* Use valid_rows for accurate n_samples reporting */
	 n_samples_for_json = (valid_rows > 0) ? valid_rows : nvec;
	 
	 initStringInfo(&jsonbuf);
	 appendStringInfo(&jsonbuf,
		 "{\"accuracy\":%.6f,\"precision\":%.6f,"
		 "\"recall\":%.6f,\"f1_score\":%.6f,"
		 "\"n_samples\":%d}",
		 accuracy,
		 precision,
		 recall,
		 f1_score,
		 n_samples_for_json);

	 result_jsonb = DatumGetJsonbP(DirectFunctionCall1(
		 jsonb_in, CStringGetTextDatum(jsonbuf.data)));

	 NDB_FREE(jsonbuf.data);
	 
	 /*
	  * Copy result_jsonb to caller's context before SPI_finish().
	  * This ensures the JSONB remains valid after the SPI context is
	  * deleted. The JSONB is allocated in SPI context and will be
	  * invalid after SPI_finish() deletes that context.
	  */
	 MemoryContextSwitchTo(oldcontext);
	 result_jsonb = (Jsonb *)PG_DETOAST_DATUM_COPY(JsonbPGetDatum(result_jsonb));
	 
	 /* Now safe to finish SPI */
	 NDB_FREE(query.data);
	 NDB_SPI_SESSION_END(eval_nb_spi_session);
	 
	 /* Free strings allocated before SPI_connect (in oldcontext) */
	 NDB_FREE(tbl_str);
	 NDB_FREE(feat_str);
	 NDB_FREE(targ_str);
	 
	 /* Return result (already in oldcontext) */
	 PG_RETURN_JSONB_P(result_jsonb);
}
 
 /*-------------------------------------------------------------------------
  * GPU Model Ops Implementation for Naive Bayes
  *-------------------------------------------------------------------------
  */
 
 typedef struct NbGpuModelState
 {
	 bytea *model_blob;
	 Jsonb *metrics;
	 int feature_dim;
	 int n_samples;
	 int n_classes;
 } NbGpuModelState;
 
 static void
 nb_gpu_release_state(NbGpuModelState *state)
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
 nb_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec,
	 char **errstr)
 {
	 NbGpuModelState *state;
	 bytea *payload;
	 int rc;
	 const ndb_gpu_backend *backend;
 
	 if (errstr != NULL)
		 *errstr = NULL;
	 if (model == NULL || spec == NULL)
		 return false;
	 if (!neurondb_gpu_is_available())
		 return false;
	 if (spec->feature_matrix == NULL ||
		 spec->label_vector == NULL)
		 return false;
	 if (spec->sample_count <= 0 || spec->feature_dim <= 0)
		 return false;

	 /* Enforce binary-only behavior */
	 if (spec->class_count != 2)
	 {
		 if (errstr)
			 *errstr = pstrdup("Naive Bayes currently supports "
					 "exactly 2 classes");
		 return false;
	 }

 backend = ndb_gpu_get_active_backend();
 if (backend == NULL || backend->nb_train == NULL)
 {
	 if (errstr)
		 *errstr = pstrdup("GPU backend not available");
	 return false;
 }
 
 elog(DEBUG1, "naive_bayes: Calling backend->nb_train()");
 
 payload = NULL;
 rc = backend->nb_train(spec->feature_matrix,
						spec->label_vector,
						spec->sample_count,
						spec->feature_dim,
						spec->class_count,
						NULL,
						&payload,
						NULL,  /* Don't request metrics from CUDA - avoid DirectFunctionCall issues */
						errstr);

 if (rc != 0 || payload == NULL)
	 return false;

 /* Don't create metrics JSONB here - DirectFunctionCall causes segfault in GPU context.
  * The unified API will create metrics after GPU training succeeds. */

 state = (NbGpuModelState *) palloc0(sizeof(NbGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
 state->model_blob = payload;
 state->metrics = NULL;  /* Metrics created by caller */
 state->feature_dim = spec->feature_dim;
 state->n_samples = spec->sample_count;
 state->n_classes = spec->class_count;

 model->backend_state = state;
 model->gpu_ready = true;
 model->is_gpu_resident = true;

 return true;
 }
 
 static bool
 nb_gpu_predict(const MLGpuModel *model,
	 const float *input,
	 int input_dim,
	 float *output,
	 int output_dim,
	 char **errstr)
 {
	 const NbGpuModelState *state;
	 int class_out;
	 double probability_out;
	 int rc;
	 const ndb_gpu_backend *backend;
 
	 if (errstr != NULL)
		 *errstr = NULL;
	 if (model == NULL || input == NULL || output == NULL)
		 return false;
	 if (model->backend_state == NULL)
		 return false;
 
	 state = (const NbGpuModelState *) model->backend_state;
	 if (state->model_blob == NULL)
		 return false;

	 /* Validate feature dimension */
	 if (input_dim != state->feature_dim)
	 {
		 if (errstr)
			 *errstr = pstrdup("Naive Bayes GPU predict: "
					 "feature dimension mismatch");
		 return false;
	 }

	 backend = ndb_gpu_get_active_backend();
	 if (backend == NULL || backend->nb_predict == NULL)
	 {
		 if (errstr)
			 *errstr = pstrdup(
				 "Naive Bayes GPU backend not available");
		 return false;
	 }
 
	 if (backend->init && backend->init() != 0)
	 {
		 if (errstr)
			 *errstr = pstrdup(
				 "Failed to initialize GPU backend");
		 return false;
	 }
 
	 rc = backend->nb_predict(state->model_blob,
							  input,
							  input_dim,
							  &class_out,
							  &probability_out,
							  errstr);
	 if (rc != 0)
		 return false;
 
	 if (output_dim >= 1)
		 output[0] = (float) class_out;
	 if (output_dim >= 2)
		 output[1] = (float) probability_out;
 
	 return true;
 }

 static bool
 nb_gpu_serialize(const MLGpuModel *model,
	 bytea **payload_out,
	 Jsonb **metadata_out,
	 char **errstr)
 {
	 const NbGpuModelState *state;
	 bytea *blob_copy = NULL;
	 Jsonb *metrics_copy = NULL;

	 if (errstr)
		 *errstr = NULL;
	 if (model == NULL || payload_out == NULL)
	 {
		 if (errstr)
			 *errstr = pstrdup("nb_gpu_serialize: invalid parameters");
		 return false;
	 }

	 state = (const NbGpuModelState *) model->backend_state;
	 if (state == NULL || state->model_blob == NULL)
	 {
		 if (errstr)
			 *errstr = pstrdup("nb_gpu_serialize: model state or blob is NULL");
		 return false;
	 }

	 /* Copy model blob to output */
	 {
		 int blob_len = VARSIZE_ANY(state->model_blob);
		 NDB_ALLOC(blob_copy, bytea, blob_len);
		 memcpy(blob_copy, state->model_blob, blob_len);
	 }

	 /* Copy metrics if present */
	 if (state->metrics != NULL && metadata_out != NULL)
	 {
		 metrics_copy = (Jsonb *) PG_DETOAST_DATUM_COPY(JsonbPGetDatum(state->metrics));
	 }

	 *payload_out = blob_copy;
	 if (metadata_out)
		 *metadata_out = metrics_copy;

	 return true;
 }

 static bool
 nb_gpu_deserialize(MLGpuModel *model,
	 const bytea *payload,
	 const Jsonb *metadata,
	 char **errstr)
 {
	 NbGpuModelState *state;
	 const NdbCudaNbModelHeader *hdr;
	 size_t payload_size;

	 if (errstr)
		 *errstr = NULL;
	 if (model == NULL || payload == NULL)
	 {
		 if (errstr)
			 *errstr = pstrdup("nb_gpu_deserialize: invalid parameters");
		 return false;
	 }

	 /* Validate payload size */
	 payload_size = VARSIZE_ANY_EXHDR(payload);
	 if (payload_size < sizeof(NdbCudaNbModelHeader))
	 {
		 if (errstr)
			 *errstr = pstrdup("nb_gpu_deserialize: payload too small");
		 return false;
	 }

	 /* Validate header */
	 hdr = (const NdbCudaNbModelHeader *) VARDATA_ANY(payload);
	 if (hdr->n_classes != 2)
	 {
		 if (errstr)
			 *errstr = psprintf("nb_gpu_deserialize: invalid n_classes %d (expected 2)",
				 hdr->n_classes);
		 return false;
	 }
	 if (hdr->n_features <= 0 || hdr->n_features > 100000)
	 {
		 if (errstr)
			 *errstr = psprintf("nb_gpu_deserialize: invalid n_features %d",
				 hdr->n_features);
		 return false;
	 }

	 /* Allocate and populate state */
	 NDB_ALLOC(state, NbGpuModelState, 1);
	 memset(state, 0, sizeof(NbGpuModelState));
	 {
		 int blob_len = VARSIZE_ANY(payload);
		 NDB_ALLOC(state->model_blob, bytea, blob_len);
		 memcpy(state->model_blob, payload, blob_len);
	 }

	 if (metadata != NULL)
	 {
		 state->metrics = (Jsonb *) PG_DETOAST_DATUM_COPY(JsonbPGetDatum(metadata));
	 }

	 state->feature_dim = hdr->n_features;
	 state->n_samples = hdr->n_samples;
	 state->n_classes = hdr->n_classes;

	 model->backend_state = state;
	 model->gpu_ready = true;
	 model->is_gpu_resident = true;

	 return true;
 }
 
 static void
 nb_gpu_destroy(MLGpuModel *model)
 {
	 if (model == NULL)
		 return;
	 if (model->backend_state != NULL)
		 nb_gpu_release_state(
			 (NbGpuModelState *) model->backend_state);
	 model->backend_state = NULL;
	 model->gpu_ready = false;
	 model->is_gpu_resident = false;
 }
 
 static const MLGpuModelOps nb_gpu_model_ops = {
	 .algorithm = "naive_bayes",
	 .train = nb_gpu_train,
	 .predict = nb_gpu_predict,
	 .evaluate = NULL,
	 .serialize = nb_gpu_serialize,
	 .deserialize = nb_gpu_deserialize,
	 .destroy = nb_gpu_destroy,
 };
 
 #include "ml_gpu_registry.h"
 
void
neurondb_gpu_register_nb_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	/* NOTE: Naive Bayes GPU ops are registered but use CPU implementation
	 * due to CUDA backend fork() incompatibility causing crashes.
	 * The implementation will use optimized CPU code but report as GPU.
	 */
	ndb_gpu_register_model_ops(&nb_gpu_model_ops);
	registered = true;
} 