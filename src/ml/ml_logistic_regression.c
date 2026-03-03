/*-------------------------------------------------------------------------
 *
 * ml_logistic_regression.c
 *    Binary classification using logistic regression.
 *
 * This module implements logistic regression with gradient descent optimization
 * and L2 regularization. Training processes data in chunks for large datasets,
 * with model serialization and catalog storage.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_logistic_regression.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/memutils.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_logistic_regression_internal.h"
#include "ml_catalog.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu_backend.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_logistic_regression.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_sql.h"
#include "neurondb_constants.h"
#include "neurondb_json.h"
#include "utils/jsonb.h"
#include "utils/elog.h"
#include "neurondb_guc.h"
/* Include CUDA LR header for type definitions (needed even when GPU is disabled) */
/* We need the NdbCudaLrModelHeader type definition */
#ifndef __CUDACC__
#include "postgres.h"
#include "utils/jsonb.h"
#include "ml_logistic_regression_internal.h"
#endif

/* Include the header that defines NdbCudaLrModelHeader */
#include "neurondb_cuda_lr.h"

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif
#endif

#include "utils/builtins.h"

#include <math.h>
#include <float.h>

typedef struct LRDataset
{
	float	   *features;
	double	   *labels;
	int			n_samples;
	int			feature_dim;
}			LRDataset;

/*
 * Streaming accumulator for chunked gradient descent
 * Accumulates gradients across chunks without loading all data
 */
typedef struct LRStreamAccum
{
	double	   *grad_w;
	double		grad_b;
	int			feature_dim;
	int			n_samples;
	bool		initialized;
}			LRStreamAccum;

static void lr_dataset_init(LRDataset * dataset);
static void lr_dataset_free(LRDataset * dataset);
static void lr_dataset_load_limited(const char *quoted_tbl,
									const char *quoted_feat,
									const char *quoted_label,
									LRDataset * dataset,
									int max_rows);
static void lr_stream_accum_init(LRStreamAccum * accum, int dim);
static void lr_stream_accum_free(LRStreamAccum * accum);
static void lr_stream_accum_reset(LRStreamAccum * accum);
static void lr_stream_process_chunk(const char *quoted_tbl,
									const char *quoted_feat,
									const char *quoted_label,
									LRStreamAccum * accum,
									double *weights,
									double bias,
									int chunk_size,
									int offset,
									int *rows_processed);
static bytea * lr_model_serialize(const LRModel *model, uint8 training_backend);
static LRModel *lr_model_deserialize(const bytea * data, uint8 * training_backend_out);
static bool lr_try_gpu_predict_catalog(int32 model_id,
									   const Vector *feature_vec,
									   double *result_out);
static bool lr_load_model_from_catalog(int32 model_id, LRModel **out);

/*
 * Sigmoid function: 1 / (1 + exp(-z))
 */
static inline double
sigmoid(double z)
{
	if (z > 500)
		return 1.0;
	if (z < -500)
		return 0.0;
	return 1.0 / (1.0 + exp(-z));
}

/*
 * Compute log loss (binary cross-entropy)
 * Note: Currently unused as we compute loss directly in chunked metrics
 */
static double
			compute_log_loss(double *y_true, double *y_pred, int n)
			pg_attribute_unused();
static double
compute_log_loss(double *y_true, double *y_pred, int n)
{
	double		loss = 0.0;
	int			i;

	for (i = 0; i < n; i++)
	{
		double		pred = y_pred[i];

		pred = fmax(1e-15, fmin(1.0 - 1e-15, pred));

		if (y_true[i] > 0.5)
			loss -= log(pred);
		else
			loss -= log(1.0 - pred);
	}

	return loss / n;
}

/*
 * train_logistic_regression
 *
 * Trains a logistic regression model using gradient descent
 * Returns coefficients (intercept + feature weights)
 */
PG_FUNCTION_INFO_V1(train_logistic_regression);

Datum
train_logistic_regression(PG_FUNCTION_ARGS)
{
	char	   *feat_str = NULL;
	char	   *gpu_err = NULL;
	char	   *targ_str = NULL;
	char	   *tbl_str = NULL;
	const char *quoted_feat = NULL;
	const char *quoted_label = NULL;
	const char *quoted_tbl = NULL;
	double		accuracy_val = 0.0;
	double		bias = 0.0;
	double		final_loss = 0.0;
	double		lambda;
	double		learning_rate;
	double	   *weights = NULL;
	int			chunk_size;
	int			correct = 0;
	int			dim = 0;
	int			iter;
	int			j;
	int			max_iters;
	int			nvec = 0;
	int			offset;
	int			rows_in_chunk;
	int			total_rows_processed;
	int32		model_id = 0;
	Jsonb	   *gpu_hyperparams = NULL;
	LRStreamAccum stream_accum;
	MLGpuTrainResult gpu_result;
	StringInfoData hyperbuf = {0};
	text	   *feature_col = NULL;
	text	   *table_name = NULL;
	text	   *target_col = NULL;

	/* Initialize gpu_result to zero to avoid undefined behavior */
	memset(&gpu_result, 0, sizeof(MLGpuTrainResult));

	if (PG_NARGS() != 6)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("train_logistic_regression: 6 arguments are required")),
				 errdetail("Usage: train_logistic_regression(table_name, feature_col, target_col, max_iters, learning_rate, lambda)"),
				 errhint("Provide table name, feature column, target column, maximum iterations, learning rate, and regularization lambda.")));
	}

	max_iters = PG_GETARG_INT32(3);
	learning_rate = PG_GETARG_FLOAT8(4);
	lambda = PG_GETARG_FLOAT8(5);
	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_label = quote_identifier(targ_str);

	{
		StringInfoData count_query = {0};
		int			ret;
		Oid			feat_type_oid = InvalidOid;

		NdbSpiSession *check_spi_session = NULL;
		MemoryContext oldcontext;

		oldcontext = CurrentMemoryContext;
		Assert(oldcontext != NULL);
		NDB_SPI_SESSION_BEGIN(check_spi_session, oldcontext);

		initStringInfo(&count_query);
		appendStringInfo(&count_query,
						 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT 1",
						 quoted_feat,
						 quoted_label,
						 quoted_tbl,
						 quoted_feat,
						 quoted_label);

		ret = ndb_spi_execute(check_spi_session, count_query.data, true, 0);
		nfree(count_query.data);
		count_query.data = NULL;
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			NDB_SPI_SESSION_END(check_spi_session);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_MSG("train_logistic_regression: no valid rows found")),
					 errdetail("SPI execution returned code %d (expected %d), processed %lu rows", ret, SPI_OK_SELECT, (unsigned long) SPI_processed),
					 errhint("Verify the table exists and contains valid feature and target columns.")));
		}

		/*
		 * Safe access for complex types (arrays/vectors) - validate before
		 * access
		 */
		if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL &&
			SPI_tuptable->vals != NULL && SPI_processed > 0 && SPI_tuptable->vals[0] != NULL)
		{
			HeapTuple	first_tuple = SPI_tuptable->vals[0];
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Datum		feat_datum;
			bool		feat_null;

			feat_type_oid = SPI_gettypeid(tupdesc, 1);
			feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
			if (!feat_null)
			{
				if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
				{
					ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

					if (ARR_NDIM(arr) == 1)
						dim = ARR_DIMS(arr)[0];
				}
				else
				{
					Vector	   *vec = DatumGetVector(feat_datum);

					dim = vec->dim;
				}
			}
		}

		if (dim <= 0)
		{
			NDB_SPI_SESSION_END(check_spi_session);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_MSG("train_logistic_regression: could not determine feature dimension")),
					 errdetail("Feature dimension is %d (must be > 0)", dim),
					 errhint("Ensure feature column contains valid vector or array data.")));
		}

		initStringInfo(&count_query);
		appendStringInfo(&count_query,
						 "SELECT COUNT(*) FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
						 quoted_tbl,
						 quoted_feat,
						 quoted_label);

		ret = ndb_spi_execute(check_spi_session, count_query.data, true, 0);
		nfree(count_query.data);
		count_query.data = NULL;
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int32		count_val;

			/* Use safe function to get int32 count */
			if (ndb_spi_get_int32(check_spi_session, 0, 1, &count_val))
			{
				nvec = count_val;
			}
		}

		NDB_SPI_SESSION_END(check_spi_session);

		if (nvec < 10)
		{
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_MSG("train_logistic_regression: need at least 10 samples, got %d"),
							nvec),
					 errdetail("Dataset contains %d rows, minimum required is 10", nvec),
					 errhint("Add more data rows to the training table.")));
		}
	}

	{
		int			max_samples = 500000;

		if (nvec > max_samples)
		{
			nvec = max_samples;
		}

		if (NDB_SHOULD_TRY_GPU())
			ndb_gpu_init_if_needed();


		if (neurondb_gpu_is_available() && nvec > 0 && dim > 0 && nvec <= 1000000 && dim <= 10000
			&& ndb_gpu_kernel_enabled("lr_train"))
		{
			int			gpu_sample_limit = nvec;
			LRDataset	dataset;
			bool		gpu_training_succeeded = false;

			/* Ensure gpu_result is initialized before use */
			memset(&gpu_result, 0, sizeof(MLGpuTrainResult));

			if (gpu_sample_limit <= 0 || gpu_sample_limit > 10000000)
			{
			}
			else if (dim <= 0 || dim > 10000)
			{
			}
			else if (tbl_str == NULL || feat_str == NULL || targ_str == NULL)
			{
			}
			else
			{
				lr_dataset_init(&dataset);

				PG_TRY();
				{

					lr_dataset_load_limited(quoted_tbl,
											quoted_feat,
											quoted_label,
											&dataset,
											gpu_sample_limit);

					if (dataset.features == NULL)
					{
						lr_dataset_free(&dataset);
					}
					else
					{
						if (dataset.labels == NULL)
						{
							lr_dataset_free(&dataset);
						}
						else
						{
							if (dataset.n_samples <= 0 || dataset.n_samples > 10000000)
							{
								lr_dataset_free(&dataset);
							}
							else
							{
								if (dataset.feature_dim <= 0 || dataset.feature_dim > 10000)
								{
									lr_dataset_free(&dataset);
								}
								else
								{
									if (dataset.n_samples * dataset.feature_dim > 100000000)
									{
										lr_dataset_free(&dataset);
									}
									else
									{
										initStringInfo(&hyperbuf);
										if (hyperbuf.data == NULL)
										{
											lr_dataset_free(&dataset);
										}
										else
										{
											appendStringInfo(&hyperbuf,
															 "{\"max_iters\":%d,\"learning_rate\":%.6f,\"lambda\":%.6f}",
															 max_iters,
															 learning_rate,
															 lambda);

											if (hyperbuf.data == NULL || hyperbuf.len <= 0)
											{
												nfree(hyperbuf.data);
												lr_dataset_free(&dataset);
											}
											else
											{
												/*
												 * Parse hyperparams JSON
												 * string
												 */
												if (hyperbuf.data != NULL && hyperbuf.len > 0)
												{
													gpu_hyperparams = ndb_jsonb_in_cstring(hyperbuf.data);
													nfree(hyperbuf.data);
													hyperbuf.data = NULL;
												}
												else
												{
													/*
													 * Create empty JSONB as
													 * fallback
													 */
													char *tmp = NULL;
													nalloc(tmp, char, VARHDRSZ + sizeof(uint32));
													gpu_hyperparams = (Jsonb *) tmp;
													SET_VARSIZE(gpu_hyperparams, VARHDRSZ + sizeof(uint32));
													*((uint32 *) VARDATA(gpu_hyperparams)) = JB_CMASK;	/* Empty object header */
													if (hyperbuf.data != NULL)
													{
														nfree(hyperbuf.data);
														hyperbuf.data = NULL;
													}
												}

												if (gpu_hyperparams == NULL)
												{
													nfree(hyperbuf.data);
													lr_dataset_free(&dataset);
												}
												else
												{
													if (!neurondb_gpu_is_available())
													{
														if (gpu_hyperparams != NULL)
															nfree(gpu_hyperparams);
														nfree(hyperbuf.data);
														lr_dataset_free(&dataset);
													}
													else
													{
														if (tbl_str == NULL || targ_str == NULL)
														{
															nfree(gpu_hyperparams);
															nfree(hyperbuf.data);
															lr_dataset_free(&dataset);
														}
														else if (dataset.features == NULL || dataset.labels == NULL)
														{
															nfree(gpu_hyperparams);
															nfree(hyperbuf.data);
															lr_dataset_free(&dataset);
														}
														else if (dataset.n_samples <= 0 || dataset.feature_dim <= 0)
														{
															nfree(gpu_hyperparams);
															nfree(hyperbuf.data);
															lr_dataset_free(&dataset);
														}
														else if (!neurondb_gpu_is_available())
														{
															nfree(gpu_hyperparams);
															nfree(hyperbuf.data);
															lr_dataset_free(&dataset);
														}
														else
														{
															memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
															gpu_err = NULL;

															gpu_training_succeeded = ndb_gpu_try_train_model(NDB_ALGO_LOGISTIC_REGRESSION,
																											 NULL,
																											 NULL,
																											 tbl_str,
																											 targ_str,
																											 NULL,
																											 0,
																											 gpu_hyperparams,
																											 dataset.features,
																											 dataset.labels,
																											 dataset.n_samples,
																											 dataset.feature_dim,
																											 2,
																											 &gpu_result,
																											 &gpu_err);

															if (gpu_training_succeeded && gpu_result.spec.model_data != NULL)
															{
																/*
																 * Validate
																 * model_data
																 * size
																 */
																size_t		model_size = VARSIZE(gpu_result.spec.model_data) - VARHDRSZ;

																if (model_size == 0 || model_size > 100000000)
																{
																	gpu_training_succeeded = false;
																}
															}
															else
															{
																gpu_training_succeeded = false;
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
				PG_CATCH();
				{
					/* Defensive: cleanup on error */
					if (dataset.features != NULL || dataset.labels != NULL)
						lr_dataset_free(&dataset);
					if (gpu_hyperparams != NULL)
					{
						nfree(gpu_hyperparams);
						gpu_hyperparams = NULL;
					}
					if (hyperbuf.data != NULL)
						nfree(hyperbuf.data);
					memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
					gpu_training_succeeded = false;
					PG_RE_THROW();
				}
				PG_END_TRY();

				if (gpu_training_succeeded)
				{
					if (gpu_result.spec.model_data != NULL)
					{
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
						MLCatalogModelSpec spec;
						LRModel		lr_model;

						bytea *unified_model_data = NULL;
						Jsonb *updated_metrics = NULL;
						char	   *base = NULL;
						NdbCudaLrModelHeader *hdr = NULL;
						float	   *weights_src = NULL;
						int			i;
						double		accuracy = 0.0;


						/* Convert GPU format to unified format */
						base = VARDATA(gpu_result.spec.model_data);
						hdr = (NdbCudaLrModelHeader *) base;
						weights_src = (float *) (base + sizeof(NdbCudaLrModelHeader));

						/*
						 * Extract final_loss and accuracy from metrics if
						 * available
						 */
						if (gpu_result.spec.metrics != NULL)
						{
							text	   *metrics_text = DatumGetTextP(DirectFunctionCall1(jsonb_out, PointerGetDatum(gpu_result.spec.metrics)));
							char	   *metrics_str = text_to_cstring(metrics_text);
							char	   *loss_ptr = strstr(metrics_str, "\"final_loss\":");
							char	   *acc_ptr = strstr(metrics_str, "\"accuracy\":");

							if (loss_ptr != NULL)
								final_loss = strtod(loss_ptr + 13, NULL);
							if (acc_ptr != NULL)
								accuracy = strtod(acc_ptr + 12, NULL);

							nfree(metrics_str);
						}

						/* Build LRModel structure */
						memset(&lr_model, 0, sizeof(LRModel));
						lr_model.n_features = hdr->feature_dim;
						lr_model.n_samples = hdr->n_samples;
						lr_model.bias = hdr->bias;
						lr_model.learning_rate = hdr->learning_rate;
						lr_model.lambda = hdr->lambda;
						lr_model.max_iters = hdr->max_iters;
						lr_model.final_loss = final_loss;
						lr_model.accuracy = accuracy;

						/* Convert float weights to double */
						if (lr_model.n_features > 0)
						{
							double *weights_tmp = NULL;
							nalloc(weights_tmp, double, lr_model.n_features);
							for (i = 0; i < lr_model.n_features; i++)
								weights_tmp[i] = (double) weights_src[i];
							lr_model.weights = weights_tmp;
						}

						/*
						 * Serialize using unified format with
						 * training_backend=1 (GPU)
						 */
						unified_model_data = lr_model_serialize(&lr_model, 1);
						
						if (unified_model_data == NULL)
						{
							elog(ERROR,
								 "neurondb: train_logistic_regression: failed to serialize unified model data");
						}

						/* Cleanup LRModel weights */
						if (lr_model.weights != NULL)
						{
							nfree(lr_model.weights);
							lr_model.weights = NULL;
						}

						/* Build metrics JSON using JSONB API */
						{
							JsonbParseState *state = NULL;
							JsonbValue	jkey;
							JsonbValue	jval;

							JsonbValue *final_value = NULL;
							Numeric		n_features_num,
										n_samples_num,
										max_iters_num,
										learning_rate_num,
										lambda_num,
										final_loss_num,
										accuracy_num;

							PG_TRY();
							{
								(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

								/* Add algorithm */
								jkey.type = jbvString;
								jkey.val.string.val = "algorithm";
								jkey.val.string.len = strlen("algorithm");
								(void) pushJsonbValue(&state, WJB_KEY, &jkey);
								jval.type = jbvString;
								jval.val.string.val = "logistic_regression";
								jval.val.string.len = strlen("logistic_regression");
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
								n_features_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(lr_model.n_features > 0 ? lr_model.n_features : 0)));
								jval.type = jbvNumeric;
								jval.val.numeric = n_features_num;
								(void) pushJsonbValue(&state, WJB_VALUE, &jval);

								/* Add n_samples */
								jkey.val.string.val = "n_samples";
								jkey.val.string.len = strlen("n_samples");
								(void) pushJsonbValue(&state, WJB_KEY, &jkey);
								n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(lr_model.n_samples > 0 ? lr_model.n_samples : 0)));
								jval.type = jbvNumeric;
								jval.val.numeric = n_samples_num;
								(void) pushJsonbValue(&state, WJB_VALUE, &jval);

								/* Add max_iters */
								jkey.val.string.val = "max_iters";
								jkey.val.string.len = strlen("max_iters");
								(void) pushJsonbValue(&state, WJB_KEY, &jkey);
								max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(lr_model.max_iters)));
								jval.type = jbvNumeric;
								jval.val.numeric = max_iters_num;
								(void) pushJsonbValue(&state, WJB_VALUE, &jval);

								/* Add learning_rate */
								jkey.val.string.val = "learning_rate";
								jkey.val.string.len = strlen("learning_rate");
								(void) pushJsonbValue(&state, WJB_KEY, &jkey);
								learning_rate_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(lr_model.learning_rate)));
								jval.type = jbvNumeric;
								jval.val.numeric = learning_rate_num;
								(void) pushJsonbValue(&state, WJB_VALUE, &jval);

								/* Add lambda */
								jkey.val.string.val = "lambda";
								jkey.val.string.len = strlen("lambda");
								(void) pushJsonbValue(&state, WJB_KEY, &jkey);
								lambda_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(lr_model.lambda)));
								jval.type = jbvNumeric;
								jval.val.numeric = lambda_num;
								(void) pushJsonbValue(&state, WJB_VALUE, &jval);

								/* Add final_loss */
								jkey.val.string.val = "final_loss";
								jkey.val.string.len = strlen("final_loss");
								(void) pushJsonbValue(&state, WJB_KEY, &jkey);
								final_loss_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(final_loss)));
								jval.type = jbvNumeric;
								jval.val.numeric = final_loss_num;
								(void) pushJsonbValue(&state, WJB_VALUE, &jval);

								/* Add accuracy */
								jkey.val.string.val = "accuracy";
								jkey.val.string.len = strlen("accuracy");
								(void) pushJsonbValue(&state, WJB_KEY, &jkey);
								accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy)));
								jval.type = jbvNumeric;
								jval.val.numeric = accuracy_num;
								(void) pushJsonbValue(&state, WJB_VALUE, &jval);

								final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

								if (final_value == NULL)
								{
									elog(ERROR, "neurondb: train_logistic_regression: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
								}

								updated_metrics = JsonbValueToJsonb(final_value);
							}
							PG_CATCH();
							{
								ErrorData  *edata = CopyErrorData();

								elog(ERROR, "neurondb: train_logistic_regression: metrics JSONB construction failed: %s", edata->message);
								FlushErrorState();
								updated_metrics = NULL;
							}
							PG_END_TRY();
						}

						spec = gpu_result.spec;
						spec.model_data = unified_model_data;	/* Use unified format */
						spec.metrics = updated_metrics; /* Use updated metrics */

						if (spec.training_table == NULL)
							spec.training_table = tbl_str;
						if (spec.training_column == NULL)
							spec.training_column = targ_str;
						if (spec.parameters == NULL)
						{
							spec.parameters = gpu_hyperparams;
							gpu_hyperparams = NULL;
						}

						spec.algorithm = NDB_ALGO_LOGISTIC_REGRESSION;
						spec.model_type = "classification";

						model_id = ml_catalog_register_model(&spec);

						if (gpu_err != NULL)
							nfree(gpu_err);
						if (gpu_hyperparams != NULL)
							nfree(gpu_hyperparams);
						ndb_gpu_free_train_result(&gpu_result);
						lr_dataset_free(&dataset);
						if (hyperbuf.data != NULL)
							nfree(hyperbuf.data);
						nfree(tbl_str);
						tbl_str = NULL;
						nfree(feat_str);
						feat_str = NULL;
						nfree(targ_str);
						targ_str = NULL;

						PG_RETURN_INT32(model_id);
#endif
					}
					else
					{
					}
				}
				else
				{
				}

				if (!gpu_training_succeeded)
				{
					/* GPU training failed - determine fallback behavior */
					/* GPU mode + CUDA/ROCm: error out (GPU must work) */
					/* GPU mode + Metal: allow CPU fallback (Metal has limitations) */
					/* AUTO mode: always allow CPU fallback */
					/* CPU mode: should not reach here, but allow fallback */
					bool allow_fallback = true;
					GPUBackend backend = neurondb_gpu_get_backend();
					bool is_metal_backend = (backend == GPU_BACKEND_METAL);
					
					if (NDB_REQUIRE_GPU() && !is_metal_backend)
					{
						/* GPU mode with CUDA/ROCm: GPU must work, error out */
						allow_fallback = false;
					}
					
					if (!allow_fallback)
					{
						/* Strict GPU mode for CUDA/ROCm: error out, no CPU fallback */
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

						lr_dataset_free(&dataset);
						if (hyperbuf.data != NULL)
							nfree(hyperbuf.data);
						nfree(tbl_str);
						nfree(feat_str);
						nfree(targ_str);

						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("neurondb: logistic_regression: GPU training failed - GPU mode requires GPU to be available"),
								 errdetail("%s", error_msg),
								 errhint("Check GPU availability and model compatibility, or set compute_mode='auto' for automatic CPU fallback.")));
					}

					/* AUTO/CPU mode OR Metal backend: cleanup and fall back to CPU */
					if (gpu_err != NULL)
					{
						nfree(gpu_err);
						gpu_err = NULL;
					}

					/*
					 * Only call ndb_gpu_free_train_result if gpu_result was
					 * actually used (has non-NULL pointers)
					 */
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

					/* Defensive: cleanup dataset */
					if (dataset.features != NULL || dataset.labels != NULL)
					{
						lr_dataset_free(&dataset);
					}
					else
					{
					}

					/* Defensive: cleanup hyperparams */
					if (gpu_hyperparams != NULL)
					{
						nfree(gpu_hyperparams);
						gpu_hyperparams = NULL;
					}
					else
					{
					}

					/* Defensive: cleanup hyperbuf */
					if (hyperbuf.data != NULL)
					{
						nfree(hyperbuf.data);
					}
					else
					{
					}
				}
				else
				{
				}
			}
		}
	}

	/* CPU training path using chunked streaming */
	/* Use larger chunks for better performance */
	if (nvec > 1000000)
		chunk_size = 100000;	/* 100k chunks for very large datasets */
	else if (nvec > 100000)
		chunk_size = 50000;		/* 50k chunks for large datasets */
	else
		chunk_size = 10000;		/* 10k chunks for smaller datasets */

	/* Initialize streaming accumulator */
	lr_stream_accum_init(&stream_accum, dim);

	/* Initialize weights and bias */
	{
		int			k;

		nalloc(weights, double, dim);
		for (k = 0; k < dim; k++)
			weights[k] = 0.0;
	}


	{
		NdbSpiSession *stream_spi_session = NULL;
		MemoryContext stream_oldcontext;

		stream_oldcontext = CurrentMemoryContext;
		Assert(stream_oldcontext != NULL);
		NDB_SPI_SESSION_BEGIN(stream_spi_session, stream_oldcontext);

		/* Gradient descent with chunked processing */
		for (iter = 0; iter < max_iters; iter++)
		{
			/* Reset accumulator for this iteration */
			lr_stream_accum_reset(&stream_accum);

			/* Process all data in chunks */
			offset = 0;
			total_rows_processed = 0;
			while (offset < nvec)
			{
				lr_stream_process_chunk(quoted_tbl,
										quoted_feat,
										quoted_label,
										&stream_accum,
										weights,
										bias,
										chunk_size,
										offset,
										&rows_in_chunk);

				if (rows_in_chunk == 0)
					break;

				offset += rows_in_chunk;
				total_rows_processed += rows_in_chunk;

				/* Log progress for large datasets */
				if (offset % 100000 == 0 || offset >= nvec)
				{
				}
			}

			if (total_rows_processed == 0)
			{
				NDB_SPI_SESSION_END(stream_spi_session);
				lr_stream_accum_free(&stream_accum);
				nfree(weights);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg(NDB_ERR_MSG("train_logistic_regression: no rows processed in iteration %d"),
								iter),
						 errdetail("Iteration %d processed 0 rows, training cannot proceed", iter),
						 errhint("Verify the training table contains valid data and is accessible.")));
			}

			/* Average gradients and add L2 regularization */
			stream_accum.grad_b /= total_rows_processed;
			for (j = 0; j < dim; j++)
			{
				stream_accum.grad_w[j] = stream_accum.grad_w[j] / total_rows_processed + lambda * weights[j];
			}

			/* Update weights and bias */
			bias -= learning_rate * stream_accum.grad_b;
			for (j = 0; j < dim; j++)
				weights[j] -= learning_rate * stream_accum.grad_w[j];

			/* Log progress every 100 iterations */
			if (iter % 100 == 0)
			{
			}
		}

		NDB_SPI_SESSION_END(stream_spi_session);
	}

	/* Compute final metrics by processing data in chunks */
	/* Note: For very large datasets, we may want to sample for metrics */
	correct = 0;
	final_loss = 0.0;
	{
		NdbSpiSession *metrics_spi_session = NULL;
		MemoryContext metrics_oldcontext;

		metrics_oldcontext = CurrentMemoryContext;
		Assert(metrics_oldcontext != NULL);
		NDB_SPI_SESSION_BEGIN(metrics_spi_session, metrics_oldcontext);

		/* Process chunks for final metrics (limit to reasonable size) */
		{
			int			max_samples_for_metrics = (nvec > 100000) ? 100000 : nvec;
			int			metrics_chunk_size = (max_samples_for_metrics > 10000) ? 10000 : max_samples_for_metrics;
			int			metrics_offset = 0;
			int			metrics_rows = 0;
			double		loss_sum = 0.0;
			int			metrics_n = 0;

			while (metrics_offset < max_samples_for_metrics && metrics_n < max_samples_for_metrics)
			{
				StringInfoData metrics_query = {0};
				int			ret;
				int			metrics_i;
				int			metrics_j;
				Oid			feat_type_oid = InvalidOid;
				bool		feat_is_array = false;
				TupleDesc	metrics_tupdesc;

				float *row_buffer = NULL;

				initStringInfo(&metrics_query);
				appendStringInfo(&metrics_query,
								 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL "
								 "LIMIT %d OFFSET %d",
								 quoted_feat,
								 quoted_label,
								 quoted_tbl,
								 quoted_feat,
								 quoted_label,
								 metrics_chunk_size,
								 metrics_offset);

				ret = ndb_spi_execute(metrics_spi_session, metrics_query.data, true, 0);
				nfree(metrics_query.data);
				metrics_query.data = NULL;
				if (ret != SPI_OK_SELECT)
				{
					break;
				}

				metrics_rows = SPI_processed;
				if (metrics_rows == 0)
				{
					break;
				}

				/* Determine feature type */
				if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
				{
					metrics_tupdesc = SPI_tuptable->tupdesc;
					feat_type_oid = SPI_gettypeid(metrics_tupdesc, 1);
					if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
						feat_is_array = true;
				}

				nalloc(row_buffer, float, dim);

				for (metrics_i = 0; metrics_i < metrics_rows && metrics_n < max_samples_for_metrics; metrics_i++)
				{
					HeapTuple	tuple;
					TupleDesc	metrics_row_tupdesc;
					Datum		feat_datum;
					Datum		targ_datum;
					bool		feat_null;
					bool		targ_null;
					Vector	   *vec = NULL;
					ArrayType  *arr = NULL;
					double		y_true;
					double		z;
					double		prediction;
					double		clipped_pred;

					/* Safe access for complex types - validate before access */
					if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
						metrics_i >= SPI_processed || SPI_tuptable->vals[metrics_i] == NULL)
					{
						continue;
					}
					tuple = SPI_tuptable->vals[metrics_i];
					metrics_row_tupdesc = SPI_tuptable->tupdesc;

					if (metrics_row_tupdesc == NULL)
					{
						continue;
					}
					feat_datum = SPI_getbinval(tuple, metrics_row_tupdesc, 1, &feat_null);
					if (metrics_row_tupdesc->natts < 2)
					{
						continue;
					}
					targ_datum = SPI_getbinval(tuple, metrics_row_tupdesc, 2, &targ_null);

					if (feat_null || targ_null)
						continue;

					/* Extract feature vector */
					if (feat_is_array)
					{
						arr = DatumGetArrayTypeP(feat_datum);
						if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != dim)
							continue;
						if (feat_type_oid == FLOAT8ARRAYOID)
						{
							float8	   *data = (float8 *) ARR_DATA_PTR(arr);

							for (metrics_j = 0; metrics_j < dim; metrics_j++)
								row_buffer[metrics_j] = (float) data[metrics_j];
						}
						else
						{
							float4	   *data = (float4 *) ARR_DATA_PTR(arr);

							memcpy(row_buffer, data, sizeof(float) * dim);
						}
					}
					else
					{
						vec = DatumGetVector(feat_datum);
						if (vec->dim != dim)
							continue;
						memcpy(row_buffer, vec->data, sizeof(float) * dim);
					}

					/* Extract target */
					{
						Oid			targ_type = SPI_gettypeid(metrics_row_tupdesc, 2);

						if (targ_type == INT2OID || targ_type == INT4OID
							|| targ_type == INT8OID)
							y_true = (double) DatumGetInt32(targ_datum);
						else
							y_true = DatumGetFloat8(targ_datum);
					}

					if (y_true != 0.0 && y_true != 1.0)
						continue;

					/* Compute prediction */
					z = bias;
					for (metrics_j = 0; metrics_j < dim; metrics_j++)
						z += weights[metrics_j] * row_buffer[metrics_j];
					prediction = sigmoid(z);

					/* Compute loss */
					clipped_pred = fmax(1e-15, fmin(1.0 - 1e-15, prediction));
					if (y_true > 0.5)
						loss_sum -= log(clipped_pred);
					else
						loss_sum -= log(1.0 - clipped_pred);

					/* Count correct predictions */
					if ((prediction >= 0.5 && y_true > 0.5)
						|| (prediction < 0.5 && y_true <= 0.5))
						correct++;

					metrics_n++;
				}

				nfree(row_buffer);
				row_buffer = NULL;
				nfree(metrics_query.data);
				metrics_query.data = NULL;
				metrics_offset += metrics_rows;
			}

			if (metrics_n > 0)
			{
				final_loss = loss_sum / metrics_n;
				/* Scale correct count to full dataset */
				correct = (int) ((double) correct * (double) nvec / (double) metrics_n);
			}
			else
			{
				final_loss = 0.0;
				correct = 0;
			}
		}

		NDB_SPI_SESSION_END(metrics_spi_session);
	}

	lr_stream_accum_free(&stream_accum);

	accuracy_val = (nvec > 0) ? ((double) correct / (double) nvec) : 0.0;

	/* Build LRModel and register in catalog */
	{
		LRModel		model;
		bytea	   *serialized = NULL;
		MLCatalogModelSpec spec;
		Jsonb	   *params_jsonb = NULL;
		Jsonb	   *metrics_jsonb = NULL;

		/* Build model struct */
		/* Validate dim before using it - defensive check */
		if (dim <= 0 || dim > 10000)
		{
			nfree(weights);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_MSG("train_logistic_regression: invalid dim %d before model save"),
							dim),
					 errdetail("Feature dimension is %d, must be between 1 and 10000", dim),
					 errhint("This indicates corrupted state. Verify training data.")));
		}

		memset(&model, 0, sizeof(model));
		model.n_features = dim;
		model.n_samples = nvec;
		model.bias = bias;
		{
			int			k;
			double *weights_tmp = NULL;

			nalloc(weights_tmp, double, dim);
			for (k = 0; k < dim; k++)
				weights_tmp[k] = weights[k];
			model.weights = weights_tmp;
		}

		model.learning_rate = learning_rate;
		model.lambda = lambda;
		model.max_iters = max_iters;
		model.final_loss = final_loss;
		model.accuracy = accuracy_val;

		/* Serialize model with training_backend=0 (CPU) */
		serialized = lr_model_serialize(&model, 0);

		/* Build parameters JSON using JSONB API */
		{
			JsonbParseState *state = NULL;
			JsonbValue	jkey;
			JsonbValue	jval;

			JsonbValue *final_value = NULL;
			Numeric		max_iters_num,
						learning_rate_num,
						lambda_num;

			PG_TRY();
			{
				(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

				/* Add max_iters */
				jkey.type = jbvString;
				jkey.val.string.val = "max_iters";
				jkey.val.string.len = strlen("max_iters");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				max_iters_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_iters)));
				jval.type = jbvNumeric;
				jval.val.numeric = max_iters_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add learning_rate */
				jkey.val.string.val = "learning_rate";
				jkey.val.string.len = strlen("learning_rate");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				learning_rate_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(learning_rate)));
				jval.type = jbvNumeric;
				jval.val.numeric = learning_rate_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add lambda */
				jkey.val.string.val = "lambda";
				jkey.val.string.len = strlen("lambda");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				lambda_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(lambda)));
				jval.type = jbvNumeric;
				jval.val.numeric = lambda_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

				if (final_value == NULL)
				{
					elog(ERROR, "neurondb: train_logistic_regression: pushJsonbValue(WJB_END_OBJECT) returned NULL for parameters");
				}

				params_jsonb = JsonbValueToJsonb(final_value);
			}
			PG_CATCH();
			{
				ErrorData  *edata = CopyErrorData();

				elog(ERROR, "neurondb: train_logistic_regression: parameters JSONB construction failed: %s", edata->message);
				FlushErrorState();
				params_jsonb = NULL;
			}
			PG_END_TRY();
		}

		/* Build metrics JSON using JSONB API */
		{
			JsonbParseState *state = NULL;
			JsonbValue	jkey;
			JsonbValue	jval;

			JsonbValue *final_value = NULL;
			Numeric		n_features_num,
						n_samples_num,
						final_loss_num,
						accuracy_num;

			PG_TRY();
			{
				(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

				/* Add algorithm */
				jkey.type = jbvString;
				jkey.val.string.val = "algorithm";
				jkey.val.string.len = strlen("algorithm");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				jval.type = jbvString;
				jval.val.string.val = "logistic_regression";
				jval.val.string.len = strlen("logistic_regression");
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add training_backend */
				jkey.val.string.val = "training_backend";
				jkey.val.string.len = strlen("training_backend");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				{
					Numeric training_backend_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(0)));
					jval.type = jbvNumeric;
					jval.val.numeric = training_backend_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);
				}

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

				/* Add final_loss */
				jkey.val.string.val = "final_loss";
				jkey.val.string.len = strlen("final_loss");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				final_loss_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(final_loss)));
				jval.type = jbvNumeric;
				jval.val.numeric = final_loss_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add accuracy */
				jkey.val.string.val = "accuracy";
				jkey.val.string.len = strlen("accuracy");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy_val)));
				jval.type = jbvNumeric;
				jval.val.numeric = accuracy_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

				if (final_value == NULL)
				{
					elog(ERROR, "neurondb: train_logistic_regression: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
				}

				metrics_jsonb = JsonbValueToJsonb(final_value);
			}
			PG_CATCH();
			{
				ErrorData  *edata = CopyErrorData();

				elog(ERROR, "neurondb: train_logistic_regression: metrics JSONB construction failed: %s", edata->message);
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
		spec.algorithm = NDB_ALGO_LOGISTIC_REGRESSION;
		spec.model_type = "classification";
		spec.training_table = tbl_str;
		spec.training_column = targ_str;
		spec.parameters = params_jsonb;
		spec.metrics = metrics_jsonb;
		spec.model_data = serialized;
		spec.training_time_ms = -1;
		spec.num_samples = nvec;
		spec.num_features = dim;

		model_id = ml_catalog_register_model(&spec);


		nfree(model.weights);

		/*
		 * Note: serialized is owned by catalog now
		 */
	}

	nfree(weights);
	nfree(tbl_str);
	nfree(feat_str);
	nfree(targ_str);
	PG_RETURN_INT32(model_id);
}

/*
 * predict_logistic_regression
 *
 * Predicts probability for logistic regression
 */
PG_FUNCTION_INFO_V1(predict_logistic_regression);

Datum
predict_logistic_regression(PG_FUNCTION_ARGS)
{
	ArrayType  *coef_array = NULL;
	Vector	   *features = NULL;
	int			ncoef;
	float8	   *coef = NULL;
	float	   *x = NULL;
	int			dim;
	double		z;
	double		probability;
	int			i;

	coef_array = PG_GETARG_ARRAYTYPE_P(0);
	features = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(features);

	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
				(errmsg("Coefficients must be 1-dimensional array")));

	ncoef = ARR_DIMS(coef_array)[0];
	coef = (float8 *) ARR_DATA_PTR(coef_array);

	x = features->data;
	dim = features->dim;

	if (ncoef != dim + 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Coefficient dimension mismatch: expected %d, got %d",
						dim + 1,
						ncoef)));

	/* Compute z = bias + w1*x1 + w2*x2 + ... */
	z = coef[0];				/* bias */
	for (i = 0; i < dim; i++)
		z += coef[i + 1] * x[i];

	/* Apply sigmoid */
	probability = sigmoid(z);

	PG_RETURN_FLOAT8(probability);
}

/*
 * predict_logistic_regression(model_id, features)
 *
 * Predicts using a model loaded from the catalog by model_id.
 * Supports both CPU and GPU models.
 */
PG_FUNCTION_INFO_V1(predict_logistic_regression_model_id);

Datum
predict_logistic_regression_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	Vector	   *features = NULL;

	LRModel *model = NULL;
	double		probability;
	double		z;
	int			i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("logistic_regression: model_id is "
						"required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("logistic_regression: features vector "
						"is required")));

	features = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(features);

	/* Try GPU prediction first */
	if (lr_try_gpu_predict_catalog(model_id, features, &probability))
	{
		PG_RETURN_FLOAT8(probability);
	}
	else
	{
	}

	/*
	 * Load model from catalog - unified format supports both CPU and GPU
	 * models
	 */
	if (!lr_load_model_from_catalog(model_id, &model))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("logistic_regression: model %d not found"),
						model_id)));
	}

	/* Validate feature dimension - match linear regression */
	if (model->n_features > 0 && features->dim != model->n_features)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("logistic_regression: feature dimension mismatch (expected %d, got %d)"),
						model->n_features,
						features->dim)));

	/* Compute z = bias + w1*x1 + w2*x2 + ... */
	z = model->bias;
	for (i = 0; i < model->n_features && i < features->dim; i++)
		z += model->weights[i] * features->data[i];

	/* Apply sigmoid */
	probability = sigmoid(z);

	if (model != NULL)
	{
		if (model->weights != NULL)
		{
			nfree(model->weights);
			model->weights = NULL;
		}
		nfree(model);
		model = NULL;
	}

	PG_RETURN_FLOAT8(probability);
}

/*
 * evaluate_logistic_regression
 *
 * Evaluates model performance (accuracy, precision, recall, F1, AUC)
 * Returns: [accuracy, precision, recall, f1_score, log_loss]
 */
PG_FUNCTION_INFO_V1(evaluate_logistic_regression);

Datum
evaluate_logistic_regression(PG_FUNCTION_ARGS)
{
	text	   *table_name = NULL;
	text	   *feature_col = NULL;
	text	   *target_col = NULL;
	ArrayType  *coef_array = NULL;
	double		threshold = PG_GETARG_FLOAT8(4);
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	int			ncoef;
	float8	   *coef = NULL;
	int			tp = 0,
				tn = 0,
				fp = 0,
				fn = 0;
	double		log_loss = 0.0;
	double		accuracy,
				precision,
				recall,
				f1_score;
	int			i,
				j;

	Datum *result_datums = NULL;
	ArrayType  *result_array = NULL;
	MemoryContext oldcontext;

	NdbSpiSession *eval_spi_session = NULL;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	target_col = PG_GETARG_TEXT_PP(2);
	coef_array = PG_GETARG_ARRAYTYPE_P(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(target_col);

	/* Extract coefficients */
	if (ARR_NDIM(coef_array) != 1)
		ereport(ERROR,
				(errmsg("Coefficients must be 1-dimensional array")));

	ncoef = ARR_DIMS(coef_array)[0];
	(void) ncoef;				/* Suppress unused variable warning */
	coef = (float8 *) ARR_DATA_PTR(coef_array);

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(eval_spi_session, oldcontext);

	/* Build query */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str,
					 targ_str,
					 tbl_str,
					 feat_str,
					 targ_str);

	ret = ndb_spi_execute(eval_spi_session, query.data, true, 0);
	nfree(query.data);
	query.data = NULL;
	if (ret != SPI_OK_SELECT)
	{
		NDB_SPI_SESSION_END(eval_spi_session);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_MSG("evaluate_logistic_regression: query failed")),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table and columns exist and are accessible.")));
	}

	nvec = SPI_processed;

	/* Compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple;
		TupleDesc	tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec = NULL;
		double		y_true;
		double		z;
		double		probability;
		int			y_pred;

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
		/* Safe access for target - validate tupdesc has at least 2 columns */
		if (tupdesc->natts < 2)
		{
			continue;
		}
		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		vec = DatumGetVector(feat_datum);
		y_true = DatumGetFloat8(targ_datum);

		/* Compute probability */
		z = coef[0];			/* bias */
		for (j = 0; j < vec->dim; j++)
			z += coef[j + 1] * vec->data[j];
		probability = sigmoid(z);

		/* Apply threshold */
		y_pred = (probability >= threshold) ? 1 : 0;

		/* Update confusion matrix */
		if (y_true > 0.5)
		{
			if (y_pred == 1)
				tp++;
			else
				fn++;
		}
		else
		{
			if (y_pred == 1)
				fp++;
			else
				tn++;
		}

		/* Update log loss */
		probability = fmax(1e-15, fmin(1.0 - 1e-15, probability));
		if (y_true > 0.5)
			log_loss -= log(probability);
		else
			log_loss -= log(1.0 - probability);
	}

	log_loss /= nvec;

	NDB_SPI_SESSION_END(eval_spi_session);

	/* Compute metrics */
	accuracy = (double) (tp + tn) / (tp + tn + fp + fn);
	precision = (tp + fp > 0) ? (double) tp / (tp + fp) : 0.0;
	recall = (tp + fn > 0) ? (double) tp / (tp + fn) : 0.0;
	f1_score = (precision + recall > 0)
		? 2.0 * precision * recall / (precision + recall)
		: 0.0;

	/* Build result array: [accuracy, precision, recall, f1_score, log_loss] */
	MemoryContextSwitchTo(oldcontext);

	nalloc(result_datums, Datum, 5);
	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(precision);
	result_datums[2] = Float8GetDatum(recall);
	result_datums[3] = Float8GetDatum(f1_score);
	result_datums[4] = Float8GetDatum(log_loss);

	result_array = construct_array(result_datums,
								   5,
								   FLOAT8OID,
								   sizeof(float8),
								   FLOAT8PASSBYVAL,
								   'd');

	nfree(result_datums);
	nfree(tbl_str);
	nfree(feat_str);
	nfree(targ_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * evaluate_logistic_regression_by_model_id
 *
 * One-shot evaluation function: loads model, fetches all data in one query,
 * loops through rows in C, computes predictions and metrics, returns jsonb.
 * This is much more efficient than calling predict() for each row in SQL.
 * Supports both CPU and GPU models.
 */
PG_FUNCTION_INFO_V1(evaluate_logistic_regression_by_model_id);

Datum
evaluate_logistic_regression_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name = NULL;
	text	   *feature_col = NULL;
	text	   *label_col = NULL;
	double		threshold = PG_ARGISNULL(4) ? 0.5 : PG_GETARG_FLOAT8(4);
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	double		log_loss = 0.0;

	/* tp, tn, fp, fn not used in GPU path - metrics computed by GPU kernel */
	/* Initialize metrics to 0.0 to ensure no garbage values */
	double		accuracy = 0.0,
				precision = 0.0,
				recall = 0.0,
				f1_score = 0.0;
	int			i;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	Jsonb	   *result_jsonb = NULL;
	StringInfoData jsonbuf;

	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool		is_gpu_model = false;

	NdbSpiSession *spi_session = NULL;
	MemoryContext oldcontext;

	/* Load model from catalog - try CPU first, then GPU */
	LRModel *model = NULL;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("evaluate_logistic_regression_by_model_id: model_id is required"))));

	model_id = PG_GETARG_INT32(0);

	/* Validate model_id before attempting to load */
	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_logistic_regression_by_model_id: model_id must be positive, got %d", model_id),
				 errdetail("Invalid model_id: %d", model_id),
				 errhint("Provide a valid model_id from neurondb.ml_models catalog.")));

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("evaluate_logistic_regression_by_model_id: table_name, feature_col, and label_col are required"))));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);
	if (!lr_load_model_from_catalog(model_id, &model))
	{
		/*
		 * CPU model load failed - this might indicate a GPU-trained model.
		 * Try loading GPU payload instead.
		 */
		if (!ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			/* Neither CPU nor GPU model found */
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_MSG("evaluate_logistic_regression_by_model_id: model %d not found"),
							model_id)));
		}
		
		if (gpu_payload != NULL)
		{
			/* Mark this as a GPU model for later processing */
			is_gpu_model = true;
		}
		else
		{
			/* No model data found */
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_MSG("evaluate_logistic_regression_by_model_id: model %d not found"),
							model_id)));
		}
	}

	/* Connect to SPI */
	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Build query - single query to fetch all data */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(targ_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(targ_str));

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_MSG("evaluate_logistic_regression_by_model_id: query failed"))));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		nfree(gpu_payload);
		nfree(gpu_metrics);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("evaluate_logistic_regression_by_model_id: no valid rows found"))));
	}

	/* Determine feature column type */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* Unified evaluation: Determine predict function based on compute mode */
	/* All metrics calculation is the same - only difference is predict function */
	{
		bool		use_gpu_predict = false;
		int			feat_dim = 0;
		int			tp = 0,
					tn = 0,
					fp = 0,
					fn = 0;
		int			total_predictions = 0;

		/* Determine if we should use GPU predict or CPU predict */
		if (is_gpu_model && neurondb_gpu_is_available() && !NDB_COMPUTE_MODE_IS_CPU())
		{
			/* GPU model and GPU mode: use GPU predict */
			if (gpu_payload != NULL && VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaLrModelHeader))
			{
				const NdbCudaLrModelHeader *gpu_hdr = (const NdbCudaLrModelHeader *) VARDATA(gpu_payload);
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
			if (VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaLrModelHeader))
			{
				const NdbCudaLrModelHeader *gpu_hdr = (const NdbCudaLrModelHeader *) VARDATA(gpu_payload);
				const float *gpu_weights = (const float *) ((const char *) gpu_hdr + sizeof(NdbCudaLrModelHeader));
				feat_dim = gpu_hdr->feature_dim;

				/* Convert GPU model to CPU format */
				{
					double *cpu_weights = NULL;
					double		cpu_bias = 0.0;
					int			coef_idx;

					cpu_bias = gpu_hdr->bias;
					nalloc(cpu_weights, double, feat_dim);

					for (coef_idx = 0; coef_idx < feat_dim; coef_idx++)
						cpu_weights[coef_idx] = (double) gpu_weights[coef_idx];

					/* Create temporary CPU model structure */
					nalloc(model, LRModel, 1);
					model->n_features = feat_dim;
					model->bias = cpu_bias;
					model->weights = cpu_weights;
				}
				use_gpu_predict = false;
			}
		}

		/* Ensure we have a valid model or GPU payload */
		if (model == NULL && !use_gpu_predict)
		{
			NDB_SPI_SESSION_END(spi_session);
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_logistic_regression_by_model_id: no valid model found"),
					 errdetail("Neither CPU model nor GPU payload is available"),
					 errhint("Verify the model exists in the catalog and is in the correct format.")));
		}

		if (feat_dim <= 0)
		{
			NDB_SPI_SESSION_END(spi_session);
			if (model != NULL)
			{
				nfree(model->weights);
				nfree(model);
			}
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_logistic_regression_by_model_id: invalid feature dimension %d",
							feat_dim)));
		}
		
		/* Validate model has weights if using CPU prediction */
		if (model != NULL && !use_gpu_predict)
		{
			if (model->weights == NULL && model->n_features > 0)
			{
				NDB_SPI_SESSION_END(spi_session);
				nfree(model);
				nfree(gpu_payload);
				nfree(gpu_metrics);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_logistic_regression_by_model_id: model has no weights"),
						 errdetail("Model has %d features but weights array is NULL", model->n_features),
						 errhint("Model may be corrupted or not properly serialized.")));
			}
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
			double		y_pred_prob = 0.0;
			int			y_pred_class;
			int			y_true_class;
			int			actual_dim;
			int			j;

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

			/* Skip rows with null features or labels */
			if (feat_null || targ_null)
				continue;

			/* Extract target - handle both integer and float types */
			{
				Oid			targ_type = SPI_gettypeid(tupdesc, 2);

				if (targ_type == INT2OID || targ_type == INT4OID
					|| targ_type == INT8OID)
					y_true = (double) DatumGetInt32(targ_datum);
				else
					y_true = DatumGetFloat8(targ_datum);
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

			/* Call appropriate predict function based on compute mode */
			if (use_gpu_predict)
			{
				/* GPU predict path - use lr_try_gpu_predict_catalog */
				Vector *feat_vec = NULL;

				/* Create Vector structure for GPU predict */
				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(feat_datum);
					float *feat_data = NULL;
					nalloc(feat_data, float, feat_dim);
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						float8 *data = (float8 *) ARR_DATA_PTR(arr);
						for (j = 0; j < feat_dim; j++)
							feat_data[j] = (float) data[j];
					}
					else
					{
						float4 *data = (float4 *) ARR_DATA_PTR(arr);
						memcpy(feat_data, data, sizeof(float) * feat_dim);
					}
					/* Create temporary Vector structure */
					{
						size_t		vec_size = VARHDRSZ + sizeof(int16) * 2 + sizeof(float) * feat_dim;
						feat_vec = (Vector *) palloc0(vec_size);
						SET_VARSIZE(feat_vec, vec_size);
						feat_vec->dim = feat_dim;
						memcpy(feat_vec->data, feat_data, sizeof(float) * feat_dim);
					}
					nfree(feat_data);
				}
				else
				{
					feat_vec = DatumGetVector(feat_datum);
				}

				/* Use GPU predict */
				if (!lr_try_gpu_predict_catalog(model_id, feat_vec, &y_pred_prob))
				{
					/* GPU predict failed - in GPU mode, this is an error */
					if (NDB_REQUIRE_GPU())
					{
						if (feat_is_array)
							pfree(feat_vec);
						NDB_SPI_SESSION_END(spi_session);
						if (model != NULL)
						{
							nfree(model->weights);
							nfree(model);
						}
						nfree(gpu_payload);
						nfree(gpu_metrics);
						nfree(tbl_str);
						nfree(feat_str);
						nfree(targ_str);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("neurondb: evaluate_logistic_regression_by_model_id: GPU prediction failed"),
								 errdetail("GPU mode is required but GPU prediction failed"),
								 errhint("Check GPU availability and model compatibility.")));
					}
					/* AUTO mode: fall back to CPU prediction using GPU coefficients */
					{
						const NdbCudaLrModelHeader *gpu_hdr = (const NdbCudaLrModelHeader *) VARDATA(gpu_payload);
						const float *gpu_weights = (const float *) ((const char *) gpu_hdr + sizeof(NdbCudaLrModelHeader));
						double z = gpu_hdr->bias;
						if (feat_is_array)
						{
							ArrayType *arr = DatumGetArrayTypeP(feat_datum);
							if (feat_type_oid == FLOAT8ARRAYOID)
							{
								float8 *data = (float8 *) ARR_DATA_PTR(arr);
								for (j = 0; j < feat_dim; j++)
									z += (double) gpu_weights[j] * (double) data[j];
							}
							else
							{
								float4 *data = (float4 *) ARR_DATA_PTR(arr);
								for (j = 0; j < feat_dim; j++)
									z += (double) gpu_weights[j] * (double) data[j];
							}
						}
						else
						{
							Vector *vec = DatumGetVector(feat_datum);
							for (j = 0; j < feat_dim && j < vec->dim; j++)
								z += (double) gpu_weights[j] * (double) vec->data[j];
						}
						y_pred_prob = sigmoid(z);
					}
				}
				if (feat_is_array)
					pfree(feat_vec);
			}
			else
			{
				/* CPU predict path - compute prediction using model coefficients */
				double z = model->bias;

				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(feat_datum);
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						double *feat_data = (double *) ARR_DATA_PTR(arr);
						for (j = 0; j < model->n_features; j++)
							z += model->weights[j] * feat_data[j];
					}
					else
					{
						float *feat_data = (float *) ARR_DATA_PTR(arr);
						for (j = 0; j < model->n_features; j++)
							z += model->weights[j] * (double) feat_data[j];
					}
				}
				else
				{
					Vector *vec = DatumGetVector(feat_datum);
					for (j = 0; j < model->n_features && j < vec->dim; j++)
						z += model->weights[j] * vec->data[j];
				}

				/* Apply sigmoid to get probability */
				y_pred_prob = sigmoid(z);
			}

			/* Convert probability to class prediction */
			y_pred_class = (y_pred_prob >= threshold) ? 1 : 0;

			/* Convert true label to class */
			y_true_class = (y_true >= 0.5) ? 1 : 0;

			/* Update confusion matrix (same for both CPU and GPU) */
			if (y_true_class == 1 && y_pred_class == 1)
				tp++;
			else if (y_true_class == 0 && y_pred_class == 0)
				tn++;
			else if (y_true_class == 0 && y_pred_class == 1)
				fp++;
			else if (y_true_class == 1 && y_pred_class == 0)
				fn++;

			/* Compute log loss with numerical stability (same for both CPU and GPU) */
			if (y_pred_prob > 1e-15 && y_pred_prob < 1.0 - 1e-15)
			{
				log_loss += -(y_true * log(y_pred_prob) + (1.0 - y_true) * log(1.0 - y_pred_prob));
			}
			else if (y_pred_prob <= 1e-15)
			{
				log_loss += -(y_true * -30.0 + (1.0 - y_true) * 0.0);
			}
			else if (y_pred_prob >= 1.0 - 1e-15)
			{
				log_loss += -(y_true * 0.0 + (1.0 - y_true) * -30.0);
			}

			total_predictions++;
		}

		/* Compute final metrics (same for both CPU and GPU) */
		accuracy = (total_predictions > 0) ? (double) (tp + tn) / total_predictions : 0.0;
		
		/* Precision: tp / (tp + fp) - undefined when no positive predictions */
		if (tp + fp > 0)
			precision = (double) tp / (tp + fp);
		else
			precision = 0.0;
		
		/* Recall: tp / (tp + fn) - undefined when no positive true labels */
		if (tp + fn > 0)
			recall = (double) tp / (tp + fn);
		else
			recall = 0.0;
		
		/* F1 score: harmonic mean of precision and recall */
		if (precision > 0.0 && recall > 0.0)
			f1_score = 2.0 * precision * recall / (precision + recall);
		else
			f1_score = 0.0;
		
		log_loss = (total_predictions > 0) ? log_loss / total_predictions : 0.0;
		
		/* Debug logging: log confusion matrix and metrics */

		/* Cleanup */
		if (model != NULL)
		{
			nfree(model->weights);
			nfree(model);
		}
		nfree(gpu_payload);
		nfree(gpu_metrics);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);

		NDB_SPI_SESSION_END(spi_session);

		/* Build result JSON (same for both CPU and GPU) */
		MemoryContextSwitchTo(oldcontext);
		{
			Datum		d;

			initStringInfo(&jsonbuf);
			appendStringInfo(&jsonbuf,
							 "{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"log_loss\":%.6f,\"n_samples\":%d,\"tp\":%d,\"tn\":%d,\"fp\":%d,\"fn\":%d}",
							 accuracy, precision, recall, f1_score, log_loss, total_predictions, tp, tn, fp, fn);
			

			/* Use inline DirectFunctionCall1 to bypass ndb_jsonb_in_cstring helper */
			PG_TRY();
			{
				d = DirectFunctionCall1(jsonb_in, CStringGetDatum(jsonbuf.data));
				result_jsonb = DatumGetJsonbP(d);
			}
			PG_CATCH();
			{
				FlushErrorState();
				result_jsonb = NULL;
			}
			PG_END_TRY();
			nfree(jsonbuf.data);
			jsonbuf.data = NULL;

			if (result_jsonb == NULL)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg("neurondb: evaluate_logistic_regression_by_model_id: failed to parse evaluation metrics JSON")));
			}

			PG_RETURN_JSONB_P(result_jsonb);
		}
	}

	/* Should not reach here - unified evaluation handles both CPU and GPU */
	NDB_SPI_SESSION_END(spi_session);
	if (model != NULL)
	{
		nfree(model->weights);
		nfree(model);
	}
	nfree(gpu_payload);
	nfree(gpu_metrics);
	nfree(tbl_str);
	nfree(feat_str);
	nfree(targ_str);
	ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("neurondb: evaluate_logistic_regression_by_model_id: unexpected code path"),
			 errdetail("Unified evaluation should have handled all cases")));
	return (Datum) 0;
}

static void
lr_dataset_init(LRDataset * dataset)
{
	if (dataset == NULL)
		return;
	memset(dataset, 0, sizeof(LRDataset));
}

static void
lr_dataset_free(LRDataset * dataset)
{
	if (dataset == NULL)
	{
		return;
	}

	if (dataset->features != NULL)
	{
	}
	else
	{
	}

	if (dataset->labels != NULL)
	{
		nfree(dataset->labels);
	}
	else
	{
	}

	lr_dataset_init(dataset);
}

/*
 * lr_dataset_load_limited
 *
 * Load dataset with LIMIT clause to avoid loading too much data for GPU
 */
static void
lr_dataset_load_limited(const char *quoted_tbl,
						const char *quoted_feat,
						const char *quoted_label,
						LRDataset * dataset,
						int max_rows)
{
	StringInfoData query = {0};
	MemoryContext oldcontext;
	int			ret = 0;
	int			n_samples = 0;
	int			feature_dim = 0;
	int			i;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;

	NdbSpiSession *load_spi_session = NULL;

	if (dataset == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_dataset_load_limited: dataset is NULL")));

	if (max_rows <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_dataset_load_limited: max_rows must be positive")));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);

	NDB_SPI_SESSION_BEGIN(load_spi_session, oldcontext);

	/*
	 * Initialize query - use ndb_spi_stringinfo_init for proper context
	 * handling
	 */
	ndb_spi_stringinfo_init(load_spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL LIMIT %d",
					 quoted_feat,
					 quoted_label,
					 quoted_tbl,
					 quoted_feat,
					 quoted_label,
					 max_rows);

	ret = ndb_spi_execute(load_spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(load_spi_session, &query);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_dataset_load_limited: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and label columns.")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		ndb_spi_stringinfo_free(load_spi_session, &query);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_dataset_load_limited: need at least 10 samples, got %d",
						n_samples),
				 errdetail("Dataset contains %d rows, minimum required is 10", n_samples),
				 errhint("Add more data rows to the training table.")));
	}

	/* Determine feature column type and dimension before allocating */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* Get feature dimension from first row before allocating */
	if (SPI_processed > 0)
	{
		HeapTuple	first_tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		bool		feat_null;
		Vector	   *vec = NULL;
		ArrayType  *arr = NULL;

		feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
		if (!feat_null)
		{
			if (feat_is_array)
			{
				arr = DatumGetArrayTypeP(feat_datum);
				if (ARR_NDIM(arr) == 1)
					feature_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				vec = DatumGetVector(feat_datum);
				feature_dim = vec->dim;
			}
		}
	}

	if (feature_dim <= 0)
	{
		ndb_spi_stringinfo_free(load_spi_session, &query);
		NDB_SPI_SESSION_END(load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_dataset_load_limited: could not determine feature dimension"),
				 errdetail("Feature dimension is %d (must be > 0)", feature_dim),
				 errhint("Ensure feature column contains valid vector or array data.")));
	}

	MemoryContextSwitchTo(oldcontext);
	{
		float *features_tmp = NULL;
		double *labels_tmp = NULL;
		nalloc(features_tmp, float, (size_t) n_samples * (size_t) feature_dim);
		nalloc(labels_tmp, double, (size_t) n_samples);
		dataset->features = features_tmp;
		dataset->labels = labels_tmp;
	}

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec = NULL;
		ArrayType  *arr = NULL;
		float	   *row = NULL;

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		if (feat_null)
			continue;

		row = dataset->features + (i * feature_dim);
		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
			if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feature_dim)
			{
				ndb_spi_stringinfo_free(load_spi_session, &query);
				NDB_SPI_SESSION_END(load_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: lr_dataset_load_limited: inconsistent array feature dimensions"),
						 errdetail("Array has %d dimensions, first row had %d features", ARR_NDIM(arr), feature_dim),
						 errhint("Ensure all feature arrays have consistent dimensions.")));
			}
			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				float8	   *data = (float8 *) ARR_DATA_PTR(arr);
				int			j;

				for (j = 0; j < feature_dim; j++)
					row[j] = (float) data[j];
			}
			else
			{
				float4	   *data = (float4 *) ARR_DATA_PTR(arr);

				memcpy(row, data, sizeof(float) * feature_dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec->dim != feature_dim)
			{
				ndb_spi_stringinfo_free(load_spi_session, &query);
				NDB_SPI_SESSION_END(load_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: lr_dataset_load_limited: inconsistent vector dimensions"),
						 errdetail("Vector has %d dimensions, first row had %d features", vec->dim, feature_dim),
						 errhint("Ensure all feature vectors have consistent dimensions.")));
			}
			memcpy(row, vec->data, sizeof(float) * feature_dim);
		}

		targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
		if (targ_null)
			continue;

		{
			Oid			targ_type = SPI_gettypeid(tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID
				|| targ_type == INT8OID)
				dataset->labels[i] =
					(double) DatumGetInt32(targ_datum);
			else
				dataset->labels[i] = DatumGetFloat8(targ_datum);
		}

		if (dataset->labels[i] != 0.0 && dataset->labels[i] != 1.0)
		{
			ndb_spi_stringinfo_free(load_spi_session, &query);
			NDB_SPI_SESSION_END(load_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: lr_dataset_load_limited: binary target required (0 or 1)"),
					 errdetail("Row %d has target value %.6f, must be 0.0 or 1.0", i, dataset->labels[i]),
					 errhint("Ensure all target values are binary (0 or 1) for logistic regression.")));
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	ndb_spi_stringinfo_free(load_spi_session, &query);
	NDB_SPI_SESSION_END(load_spi_session);
}

/*
 * lr_stream_accum_init
 *
 * Initialize streaming accumulator for chunked gradient descent
 */
static void
lr_stream_accum_init(LRStreamAccum * accum, int dim)
{
	if (accum == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_stream_accum_init: accum is NULL")));

	if (dim <= 0 || dim > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_stream_accum_init: invalid feature dimension %d",
						dim)));

	memset(accum, 0, sizeof(LRStreamAccum));

	accum->feature_dim = dim;
	accum->n_samples = 0;
	accum->initialized = false;

	/* Allocate gradient vector */
		nalloc(accum->grad_w, double, dim);
		MemSet(accum->grad_w, 0, sizeof(double) * dim);
	if (accum->grad_w == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OUT_OF_MEMORY),
				 errmsg("neurondb: lr_stream_accum_init: failed to allocate gradient vector")));

	accum->grad_b = 0.0;
	accum->initialized = true;
}

/*
 * lr_stream_accum_free
 *
 * Free memory allocated for streaming accumulator
 */
static void
lr_stream_accum_free(LRStreamAccum * accum)
{
	if (accum == NULL)
		return;

	if (accum->grad_w != NULL)
	{
		nfree(accum->grad_w);
		accum->grad_w = NULL;
	}

	memset(accum, 0, sizeof(LRStreamAccum));
}

/*
 * lr_stream_accum_reset
 *
 * Reset accumulator for next iteration (keeps memory allocated)
 */
static void
lr_stream_accum_reset(LRStreamAccum * accum)
{
	int			i;

	if (accum == NULL || !accum->initialized)
		return;

	for (i = 0; i < accum->feature_dim; i++)
		accum->grad_w[i] = 0.0;
	accum->grad_b = 0.0;
	accum->n_samples = 0;
}

/*
 * lr_stream_process_chunk
 *
 * Process a chunk of data, computing gradients for gradient descent
 * Returns number of rows processed in this chunk
 */
static void
lr_stream_process_chunk(const char *quoted_tbl,
						const char *quoted_feat,
						const char *quoted_label,
						LRStreamAccum * accum,
						double *weights,
						double bias,
						int chunk_size,
						int offset,
						int *rows_processed)
{
	StringInfoData query = {0};
	int			ret;
	int			i;
	int			j;
	int			n_rows;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;

	float *row_buffer = NULL;

	if (quoted_tbl == NULL || quoted_feat == NULL || quoted_label == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_stream_process_chunk: NULL parameter")));

	if (accum == NULL || !accum->initialized)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_stream_process_chunk: accumulator not initialized")));

	if (weights == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_stream_process_chunk: weights is NULL")));

	if (chunk_size <= 0 || chunk_size > 100000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_stream_process_chunk: invalid chunk_size %d",
						chunk_size)));

	if (rows_processed == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_stream_process_chunk: rows_processed is NULL")));

	*rows_processed = 0;

	/* Build query with LIMIT and OFFSET for chunking */
	/* Note: This function must be called within an active SPI session */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL "
					 "LIMIT %d OFFSET %d",
					 quoted_feat,
					 quoted_label,
					 quoted_tbl,
					 quoted_feat,
					 quoted_label,
					 chunk_size,
					 offset);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	nfree(query.data);
	query.data = NULL;
	if (ret != SPI_OK_SELECT)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_stream_process_chunk: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table and columns exist and are accessible.")));
	}

	n_rows = SPI_processed;
	if (n_rows == 0)
	{
		*rows_processed = 0;
		return;
	}

	/* Determine feature type from first row */
	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
	{
		TupleDesc	chunk_tupdesc = SPI_tuptable->tupdesc;

		feat_type_oid = SPI_gettypeid(chunk_tupdesc, 1);
		if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
			feat_is_array = true;
	}

	/* Allocate temporary buffer for one row */
	nalloc(row_buffer, float, accum->feature_dim);

	/* Process each row in chunk */
	for (i = 0; i < n_rows; i++)
	{
		HeapTuple	tuple;
		TupleDesc	row_tupdesc;
		Datum		feat_datum;
		Datum		targ_datum;
		bool		feat_null;
		bool		targ_null;
		Vector	   *vec = NULL;
		ArrayType  *arr = NULL;
		double		y_true;
		double		z;
		double		prediction;
		double		error;

		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			continue;
		}
		tuple = SPI_tuptable->vals[i];
		row_tupdesc = SPI_tuptable->tupdesc;
		if (row_tupdesc == NULL)
		{
			continue;
		}

		feat_datum = SPI_getbinval(tuple, row_tupdesc, 1, &feat_null);
		/* Safe access for target - validate tupdesc has at least 2 columns */
		if (row_tupdesc->natts < 2)
		{
			continue;
		}
		targ_datum = SPI_getbinval(tuple, row_tupdesc, 2, &targ_null);

		if (feat_null || targ_null)
			continue;

		/* Extract feature vector */
		if (feat_is_array)
		{
			arr = DatumGetArrayTypeP(feat_datum);
			if (ARR_NDIM(arr) != 1)
				continue;
			if (ARR_DIMS(arr)[0] != accum->feature_dim)
				continue;
			if (feat_type_oid == FLOAT8ARRAYOID)
			{
				float8	   *data = (float8 *) ARR_DATA_PTR(arr);

				for (j = 0; j < accum->feature_dim; j++)
					row_buffer[j] = (float) data[j];
			}
			else
			{
				float4	   *data = (float4 *) ARR_DATA_PTR(arr);

				memcpy(row_buffer, data, sizeof(float) * accum->feature_dim);
			}
		}
		else
		{
			vec = DatumGetVector(feat_datum);
			if (vec->dim != accum->feature_dim)
				continue;
			memcpy(row_buffer, vec->data, sizeof(float) * accum->feature_dim);
		}

		/* Extract target */
		{
			Oid			targ_type = SPI_gettypeid(row_tupdesc, 2);

			if (targ_type == INT2OID || targ_type == INT4OID
				|| targ_type == INT8OID)
				y_true = (double) DatumGetInt32(targ_datum);
			else
				y_true = DatumGetFloat8(targ_datum);
		}

		if (y_true != 0.0 && y_true != 1.0)
			continue;

		/* Compute prediction: z = bias + weights * features */
		z = bias;
		for (j = 0; j < accum->feature_dim; j++)
			z += weights[j] * row_buffer[j];
		prediction = sigmoid(z);

		/* Compute error and accumulate gradients */
		error = prediction - y_true;
		accum->grad_b += error;
		for (j = 0; j < accum->feature_dim; j++)
			accum->grad_w[j] += error * row_buffer[j];

		(*rows_processed)++;
		accum->n_samples++;
	}

	nfree(row_buffer);
	row_buffer = NULL;
}

/* GPU model state for Logistic Regression */
typedef struct LRGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			feature_dim;
	int			n_samples;
}			LRGpuModelState;

static void
lr_gpu_release_state(LRGpuModelState * state)
{
	if (state == NULL)
		return;
	if (state->model_blob)
	{
		nfree(state->model_blob);
		state->model_blob = NULL;
	}
	if (state->metrics)
	{
		nfree(state->metrics);
		state->metrics = NULL;
	}
	nfree(state);
	state = NULL;
}

static bool
lr_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	LRGpuModelState *state = NULL;
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
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

	rc = ndb_gpu_lr_train(spec->feature_matrix,
						  spec->label_vector,
						  spec->sample_count,
						  spec->feature_dim,
						  spec->hyperparameters,
						  &payload,
						  &metrics,
						  errstr);
	if (rc != 0 || payload == NULL)
	{
		if (payload != NULL)
		{
			nfree(payload);
			payload = NULL;
		}
		if (metrics != NULL)
		{
			nfree(metrics);
			metrics = NULL;
		}
		return false;
	}

	if (model->backend_state != NULL)
	{
		lr_gpu_release_state((LRGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

		nalloc(state, LRGpuModelState, 1);
		MemSet(state, 0, sizeof(LRGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	/* Store metrics in model state for later retrieval */
	if (metrics != NULL)
	{
		state->metrics = metrics;
	}
	else
	{
		state->metrics = NULL;
		elog(WARNING,
			 "lr_gpu_train: metrics is NULL, cannot store in state!");
	}

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
lr_gpu_predict(const MLGpuModel *model,
			   const float *input,
			   int input_dim,
			   float *output,
			   int output_dim,
			   char **errstr)
{
	const		LRGpuModelState *state;
	double		probability;
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

	state = (const LRGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	rc = ndb_gpu_lr_predict(state->model_blob,
							input,
							state->feature_dim > 0 ? state->feature_dim : input_dim,
							&probability,
							errstr);
	if (rc != 0)
		return false;

	output[0] = (float) probability;

	return true;
}

static bool
lr_gpu_evaluate(const MLGpuModel *model,
				const MLGpuEvalSpec *spec,
				MLGpuMetrics *out,
				char **errstr)
{
	const		LRGpuModelState *state;
	Jsonb	   *metrics_json = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const LRGpuModelState *) model->backend_state;

	/* Build metrics JSON from stored model metadata */
	{
		StringInfoData buf;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"logistic_regression\","
						 "\"training_backend\":1,"
						 "\"n_features\":%d,"
						 "\"n_samples\":%d}",
						 state->feature_dim > 0 ? state->feature_dim : 0,
						 state->n_samples > 0 ? state->n_samples : 0);

		/* Parse JSON string into JSONB */
		metrics_json = ndb_jsonb_in_cstring(buf.data);
		if (metrics_json == NULL)
		{
			nfree(buf.data);
			return false;
		}
		nfree(buf.data);
		buf.data = NULL;
	}

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
lr_gpu_serialize(const MLGpuModel *model,
				 bytea * *payload_out,
				 Jsonb * *metadata_out,
				 char **errstr)
{
	const		LRGpuModelState *state;

	bytea *unified_payload = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const LRGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
	/* Convert GPU format to unified format */
	{
		LRModel		lr_model;
		char	   *base = NULL;
		NdbCudaLrModelHeader *hdr = NULL;
		float	   *weights_src = NULL;
		double		final_loss = 0.0;
		double		accuracy = 0.0;
		int			i;

		/* Extract GPU format data */
		base = VARDATA(state->model_blob);
		hdr = (NdbCudaLrModelHeader *) base;
		weights_src = (float *) (base + sizeof(NdbCudaLrModelHeader));

		/* Extract final_loss and accuracy from metrics if available */
		/* Use simple text extraction from JSONB for these values */
		if (state->metrics != NULL)
		{
			text	   *metrics_text = DatumGetTextP(DirectFunctionCall1(jsonb_out, PointerGetDatum(state->metrics)));
			char	   *metrics_str = text_to_cstring(metrics_text);
			char	   *loss_ptr = strstr(metrics_str, "\"final_loss\":");
			char	   *acc_ptr = strstr(metrics_str, "\"accuracy\":");

			if (loss_ptr != NULL)
			{
				final_loss = strtod(loss_ptr + 13, NULL);
			}
			if (acc_ptr != NULL)
			{
				accuracy = strtod(acc_ptr + 12, NULL);
			}

			nfree(metrics_str);
		}

		/* Build LRModel structure */
		memset(&lr_model, 0, sizeof(LRModel));
		lr_model.n_features = hdr->feature_dim;
		lr_model.n_samples = hdr->n_samples;
		lr_model.bias = hdr->bias;
		lr_model.learning_rate = hdr->learning_rate;
		lr_model.lambda = hdr->lambda;
		lr_model.max_iters = hdr->max_iters;
		lr_model.final_loss = final_loss;
		lr_model.accuracy = accuracy;

		/* Convert float weights to double */
		if (lr_model.n_features > 0)
		{
			double *weights_tmp = NULL;
			nalloc(weights_tmp, double, lr_model.n_features);
			for (i = 0; i < lr_model.n_features; i++)
				weights_tmp[i] = (double) weights_src[i];
			lr_model.weights = weights_tmp;
		}

		/* Serialize using unified format with training_backend=1 (GPU) */
		unified_payload = lr_model_serialize(&lr_model, 1);

		if (lr_model.weights != NULL)
		{
			nfree(lr_model.weights);
			lr_model.weights = NULL;
		}
	}
#else
	/* For non-CUDA builds, GPU serialization is not supported */
	if (errstr != NULL)
		*errstr = "GPU serialization requires CUDA support";
	return false;
#endif

	if (unified_payload == NULL)
		return false;

	if (payload_out != NULL)
		*payload_out = unified_payload;
	else
	{
		nfree(unified_payload);
		unified_payload = NULL;
	}

	/* Return stored metrics; copy into caller context */
	if (metadata_out != NULL)
	{
		Jsonb	   *metrics_copy = NULL;

		if (state->metrics != NULL)
		{
			text   *metrics_text = DatumGetTextP(DirectFunctionCall1(jsonb_out,
													 PointerGetDatum(state->metrics)));
			char   *metrics_cstr = text_to_cstring(metrics_text);

			metrics_copy = ndb_jsonb_in_cstring(metrics_cstr);
			pfree(metrics_text);
			nfree(metrics_cstr);
		}
		else
		{
			StringInfoData metrics_buf;

			initStringInfo(&metrics_buf);
			appendStringInfo(&metrics_buf,
						 "{\"algorithm\":\"logistic_regression\","
						 "\"training_backend\":1,"
						 "\"n_features\":%d,"
						 "\"n_samples\":%d}",
					 state->feature_dim > 0 ? state->feature_dim : 0,
					 state->n_samples > 0 ? state->n_samples : 0);

			metrics_copy = ndb_jsonb_in_cstring(metrics_buf.data);
			nfree(metrics_buf.data);
		}

		*metadata_out = metrics_copy;
	}

	return true;
}

static bool
lr_gpu_deserialize(MLGpuModel *model,
				   const bytea * payload,
				   const Jsonb * metadata,
				   char **errstr)
{
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
	LRGpuModelState *state = NULL;
	LRModel    *lr_model = NULL;
	uint8		training_backend;
	bytea	   *gpu_payload = NULL;
	char	   *base = NULL;
	char	   *tmp = NULL;
	float	   *weights_dest = NULL;
	size_t		payload_bytes;
	int			i;
	NdbCudaLrModelHeader *hdr = NULL;
#endif

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
	lr_model = NULL;
	training_backend = 0;
	gpu_payload = NULL;
	/* Deserialize unified format */
	lr_model = lr_model_deserialize(payload, &training_backend);
	if (lr_model == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("lr_gpu_deserialize: failed to deserialize unified format");
		return false;
	}

	/* Convert unified format to GPU format */
	payload_bytes = sizeof(NdbCudaLrModelHeader) +
		sizeof(float) * (size_t) lr_model->n_features;
	tmp = NULL;
	nalloc(tmp, char, VARHDRSZ + payload_bytes);
	gpu_payload = (bytea *) tmp;
	NDB_CHECK_ALLOC(gpu_payload, "gpu_payload");
	SET_VARSIZE(gpu_payload, VARHDRSZ + payload_bytes);
	base = VARDATA(gpu_payload);

	hdr = (NdbCudaLrModelHeader *) base;
	hdr->feature_dim = lr_model->n_features;
	hdr->n_samples = lr_model->n_samples;
	hdr->max_iters = lr_model->max_iters;
	hdr->learning_rate = lr_model->learning_rate;
	hdr->lambda = lr_model->lambda;
	hdr->bias = lr_model->bias;

	weights_dest = (float *) (base + sizeof(NdbCudaLrModelHeader));
	if (lr_model->weights != NULL)
	{
		for (i = 0; i < lr_model->n_features; i++)
			weights_dest[i] = (float) lr_model->weights[i];
	}

	/* Cleanup LRModel */
	if (lr_model->weights != NULL)
	{
		nfree(lr_model->weights);
		lr_model->weights = NULL;
	}
	nfree(lr_model);
	lr_model = NULL;

		nalloc(state, LRGpuModelState, 1);
		MemSet(state, 0, sizeof(LRGpuModelState));
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = gpu_payload;
	state->feature_dim = hdr->feature_dim;
	state->n_samples = hdr->n_samples;
	state->metrics = NULL;

	if (metadata != NULL)
	{
		text   *metadata_text = DatumGetTextP(DirectFunctionCall1(jsonb_out,
											 PointerGetDatum(metadata)));
		char   *metadata_cstr = text_to_cstring(metadata_text);

		state->metrics = ndb_jsonb_in_cstring(metadata_cstr);
		pfree(metadata_text);
		nfree(metadata_cstr);
	}

	if (model->backend_state != NULL)
		lr_gpu_release_state((LRGpuModelState *) model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
#else
	/* For non-CUDA builds, GPU deserialization is not supported */
	if (errstr != NULL)
		*errstr = pstrdup("lr_gpu_deserialize: CUDA not available");
	return false;
#endif
}

static void
lr_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		lr_gpu_release_state((LRGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps lr_gpu_model_ops = {
	.algorithm = "logistic_regression",
	.train = lr_gpu_train,
	.predict = lr_gpu_predict,
	.evaluate = lr_gpu_evaluate,
	.serialize = lr_gpu_serialize,
	.deserialize = lr_gpu_deserialize,
	.destroy = lr_gpu_destroy,
};

static bytea *
lr_model_serialize(const LRModel *model, uint8 training_backend)
{
	StringInfoData buf;
	int			i;

	if (model == NULL)
		return NULL;

	/* Validate model before serialization - match linear regression */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_model_serialize: invalid n_features %d (corrupted model?)",
						model->n_features)));
	}

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: lr_model_serialize: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	pq_begintypsend(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) */
	pq_sendbyte(&buf, training_backend);

	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendfloat8(&buf, model->bias);
	pq_sendfloat8(&buf, model->learning_rate);
	pq_sendfloat8(&buf, model->lambda);
	pq_sendint32(&buf, model->max_iters);
	pq_sendfloat8(&buf, model->final_loss);
	pq_sendfloat8(&buf, model->accuracy);

	if (model->weights != NULL && model->n_features > 0)
	{
		for (i = 0; i < model->n_features; i++)
			pq_sendfloat8(&buf, model->weights[i]);
	}

	return pq_endtypsend(&buf);
}

static LRModel *
lr_model_deserialize(const bytea * data, uint8 * training_backend_out)
{
	LRModel *model = NULL;
	StringInfoData buf;
	int			i;
	uint8		training_backend = 0;
	size_t		expected_size;
	size_t		header_size;

	if (data == NULL)
		return NULL;

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	/* Check if this might be GPU format (starts with int32 feature_dim instead of uint8 training_backend) */
	if (buf.len >= sizeof(int32))
	{
		const int32 *first_int = (const int32 *) buf.data;
		int32		first_value = *first_int;
		
		/* GPU format: first field is feature_dim (int32), typically 1-10000 */
		/* Unified format: first byte is training_backend (uint8), typically 0 or 1 */
		/* If first 4 bytes look like a reasonable feature_dim and size is small, might be GPU format */
		if (first_value > 0 && first_value <= 10000)
		{
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
			size_t		gpu_header_size = sizeof(NdbCudaLrModelHeader);
			size_t		expected_gpu_size = gpu_header_size + sizeof(float) * (size_t) first_value;
			
			/* Check if size matches GPU format (within reasonable tolerance) */
			if (buf.len >= gpu_header_size && 
				buf.len >= expected_gpu_size - 10 && 
				buf.len <= expected_gpu_size + 10)
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: lr_model_deserialize: model is in GPU format, not unified format"),
						 errdetail("Model data appears to be in GPU format (%d bytes, feature_dim=%d) but deserialization expects unified format. This indicates the model was not properly converted during training.",
								   buf.len, first_value),
						 errhint("GPU-trained models should be converted to unified format before storage. The training code should call lr_model_serialize() to convert GPU format to unified format.")));
				return NULL;
			}
#endif
		}
	}

	/* Minimum header size: 1 byte (training_backend) + 4 (n_features) + 4 (n_samples) + 
	 * 8 (bias) + 8 (learning_rate) + 8 (lambda) + 4 (max_iters) + 8 (final_loss) + 8 (accuracy) = 53 bytes */
	header_size = 1 + 4 + 4 + 8 + 8 + 8 + 4 + 8 + 8;

	if (buf.len < header_size)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_model_deserialize: insufficient data in model (got %d bytes, need at least %zu for header)",
						buf.len, (size_t) header_size),
				 errdetail("Model data may be corrupted or in wrong format.")));
		return NULL;
	}

	/* Read training_backend first */
	training_backend = (uint8) pq_getmsgbyte(&buf);

	nalloc(model, LRModel, 1);
	MemSet(model, 0, sizeof(LRModel));
	NDB_CHECK_ALLOC(model, "model");

	model->n_features = pq_getmsgint(&buf, 4);
	model->n_samples = pq_getmsgint(&buf, 4);
	model->bias = pq_getmsgfloat8(&buf);
	model->learning_rate = pq_getmsgfloat8(&buf);
	model->lambda = pq_getmsgfloat8(&buf);
	model->max_iters = pq_getmsgint(&buf, 4);
	model->final_loss = pq_getmsgfloat8(&buf);
	model->accuracy = pq_getmsgfloat8(&buf);

	/* Validate deserialized values - match linear regression */
	if (model->n_features <= 0 || model->n_features > 10000)
	{
		nfree(model);
		model = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr: invalid n_features in deserialized model (corrupted data?)")));
		return NULL;
	}
	if (model != NULL && (model->n_samples < 0 || model->n_samples > 100000000))
	{
		nfree(model);
		model = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr: invalid n_samples in deserialized model (corrupted data?)")));
		return NULL;
	}

	/* Check if we have enough data for weights */
	expected_size = header_size + (size_t) model->n_features * 8;
	if (buf.len < expected_size)
	{
		int			n_features_saved = model->n_features;
		
		nfree(model);
		model = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: lr_model_deserialize: insufficient data for weights (got %d bytes, need %zu for %d features)",
						buf.len, expected_size, n_features_saved),
				 errdetail("Model data may be corrupted or in wrong format. Expected %zu bytes (header %zu + %d weights * 8) but only have %d bytes.",
						   expected_size, (size_t) header_size, n_features_saved, buf.len),
				 errhint("This may indicate the model was stored in GPU format instead of unified format.")));
		return NULL;
	}

	if (model->n_features > 0)
	{
		double *tmp_weights = NULL;
		nalloc(tmp_weights, double, model->n_features);
		model->weights = tmp_weights;
		for (i = 0; i < model->n_features; i++)
			model->weights[i] = pq_getmsgfloat8(&buf);
	}

	/* Return training_backend if output parameter provided */
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;

	return model;
}

/*
 * (removed unused function lr_metadata_is_gpu)
 */
#if 0
/* Removed unused function lr_metadata_is_gpu */
static bool
lr_metadata_is_gpu_removed(Jsonb * metadata)
{
	bool		is_gpu = false;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	JsonbIteratorToken r;

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
					int			backend = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));

					is_gpu = (backend == 1);
				}
			}
			nfree(key);
		}
	}

	return is_gpu;
}
#endif

/*
 * lr_try_gpu_predict_catalog
 *
 * Attempts GPU prediction for a model loaded from the catalog.
 * Returns true if GPU prediction succeeded, false otherwise.
 */
static bool
lr_try_gpu_predict_catalog(int32 model_id,
						   const Vector *feature_vec,
						   double *result_out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	char *gpu_err = NULL;
	double		probability = 0.0;
	bool		success = false;


	/* Check compute mode - only try GPU if compute mode allows it */
	if (!NDB_SHOULD_TRY_GPU())
		return false;

	if (!neurondb_gpu_is_available())
	{
		return false;
	}
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
	{
		return false;
	}

	if (payload == NULL)
	{
		goto cleanup;
	}

	/*
	 * Unified format: any model can be used for GPU prediction if payload
	 * exists
	 */

	/*
	 * training_backend flag is informational only, doesn't restrict
	 * prediction
	 */

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
	/* Convert unified format to GPU format for prediction */
	{
		LRModel *lr_model = NULL;
		uint8		training_backend = 0;

		bytea *gpu_payload = NULL;
		char	   *base = NULL;
		char	   *tmp = NULL;
		NdbCudaLrModelHeader *hdr = NULL;
		float	   *weights_dest = NULL;
		size_t		payload_bytes;
		int			i;

		/* Deserialize unified format */
		lr_model = lr_model_deserialize(payload, &training_backend);
		if (lr_model == NULL)
		{
			goto cleanup;
		}

		/* Convert to GPU format */
		payload_bytes = sizeof(NdbCudaLrModelHeader) +
			sizeof(float) * (size_t) lr_model->n_features;
		tmp = NULL;
		nalloc(tmp, char, VARHDRSZ + payload_bytes);
	gpu_payload = (bytea *) tmp;
		SET_VARSIZE(gpu_payload, VARHDRSZ + payload_bytes);
		base = VARDATA(gpu_payload);

		hdr = (NdbCudaLrModelHeader *) base;
		hdr->feature_dim = lr_model->n_features;
		hdr->n_samples = lr_model->n_samples;
		hdr->max_iters = lr_model->max_iters;
		hdr->learning_rate = lr_model->learning_rate;
		hdr->lambda = lr_model->lambda;
		hdr->bias = lr_model->bias;

		weights_dest = (float *) (base + sizeof(NdbCudaLrModelHeader));
		if (lr_model->weights != NULL)
		{
			for (i = 0; i < lr_model->n_features; i++)
				weights_dest[i] = (float) lr_model->weights[i];
		}

		/* Use GPU format for prediction */
		if (ndb_gpu_lr_predict(gpu_payload,
							   feature_vec->data,
							   feature_vec->dim,
							   &probability,
							   &gpu_err)
			== 0)
		{
			if (result_out != NULL)
				*result_out = probability;
			success = true;
		}
		else if (gpu_err != NULL)
		{
			elog(WARNING,
				 "logistic_regression: GPU prediction failed for model %d (%s)",
				 model_id,
				 gpu_err);
		}

		if (lr_model != NULL)
		{
			if (lr_model->weights != NULL)
			{
				nfree(lr_model->weights);
				lr_model->weights = NULL;
			}
			nfree(lr_model);
			lr_model = NULL;
		}
		if (gpu_payload != NULL)
		{
			nfree(gpu_payload);
			gpu_payload = NULL;
		}
	}
#else
	/* For non-CUDA builds, GPU prediction using CUDA format is not supported */
	success = false;
#endif

	if (success)
	{
		if (result_out != NULL)
			*result_out = probability;
		success = true;
	}
	else if (gpu_err != NULL)
	{
		elog(WARNING,
			 "logistic_regression: GPU prediction failed for model %d (%s)",
			 model_id,
			 gpu_err);
	}

cleanup:
	if (payload != NULL)
	{
		nfree(payload);
		payload = NULL;
	}
	if (metrics != NULL)
	{
		nfree(metrics);
		metrics = NULL;
	}
	if (gpu_err != NULL)
	{
		nfree(gpu_err);
		gpu_err = NULL;
	}

	return success;
}

/*
 * lr_load_model_from_catalog
 *
 * Loads a Logistic Regression model from the catalog by model_id.
 * Returns true on success, false on failure.
 */
static bool
lr_load_model_from_catalog(int32 model_id, LRModel **out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	LRModel    *decoded = NULL;

	if (out == NULL)
		return false;

	*out = NULL;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
	{
		if (metrics != NULL)
		{
			nfree(metrics);
			metrics = NULL;
		}
		return false;
	}

	/*
	 * For CPU evaluation, we can try to load GPU models too if GPU evaluation
	 * failed
	 */

	decoded = lr_model_deserialize(payload, NULL);

	nfree(payload);
	payload = NULL;
	if (metrics != NULL)
	{
		nfree(metrics);
		metrics = NULL;
	}

	if (decoded == NULL)
		return false;

	*out = decoded;
	return true;
}

void
neurondb_gpu_register_lr_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&lr_gpu_model_ops);
	registered = true;
}
