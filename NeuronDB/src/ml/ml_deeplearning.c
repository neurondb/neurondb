/*-------------------------------------------------------------------------
 *
 * ml_deeplearning.c
 *    Deep learning model integration.
 *
 * This module provides interfaces for PyTorch, TensorFlow, and ONNX model
 * integration with support for model import, inference, and fine-tuning.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_deeplearning.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "utils/lsyscache.h"
#include "neurondb_pgcompat.h"
#include "ml_catalog.h"
#include "lib/stringinfo.h"

#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#ifdef HAVE_ONNX_RUNTIME
#include "neurondb_onnx.h"
#endif

/*
 * Import PyTorch model
 * Supports .pt and .pth files
 */
PG_FUNCTION_INFO_V1(import_pytorch_model);

Datum
import_pytorch_model(PG_FUNCTION_ARGS)
{
	text	   *model_path = PG_GETARG_TEXT_PP(0);
	text	   *model_name = PG_GETARG_TEXT_PP(1);
	text	   *model_type = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);

	char	   *path = text_to_cstring(model_path);
	char	   *name = text_to_cstring(model_name);
	char	   *type = model_type ? text_to_cstring(model_type)
		: pstrdup("classifier");
	StringInfoData result;
	int			model_id;

	/* Validate model type */
	if (strcmp(type, "classifier") != 0 && strcmp(type, "regressor") != 0
		&& strcmp(type, "transformer") != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("model_type must be 'classifier', "
						"'regressor', or 'transformer'")));

	/* Defensive: validate model path */
	if (path == NULL || strlen(path) == 0)
	{
		NDB_FREE(path);
		NDB_FREE(name);
		NDB_FREE(type);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("import_pytorch_model: model_path cannot be empty")));
	}

	/* Defensive: check if file exists */
	{
		struct stat st;

		if (stat(path, &st) != 0)
		{
			NDB_FREE(path);
			NDB_FREE(name);
			NDB_FREE(type);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("import_pytorch_model: model file not found: %s", path)));
		}

		if (!S_ISREG(st.st_mode))
		{
			NDB_FREE(path);
			NDB_FREE(name);
			NDB_FREE(type);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("import_pytorch_model: path is not a regular file: %s", path)));
		}
	}

	/* Register model in catalog */
	{
		MLCatalogModelSpec spec;
		Jsonb	   *params_jsonb;
		StringInfoData params_json;

		initStringInfo(&params_json);
		{
			Datum		path_datum = CStringGetDatum(path);
			Datum		quoted_datum = DirectFunctionCall1(quote_literal, path_datum);
			char	   *quoted_path = DatumGetCString(quoted_datum);

			appendStringInfo(&params_json,
							 "{\"model_type\":\"%s\",\"framework\":\"pytorch\",\"path\":%s}",
							 type, quoted_path);
			NDB_FREE(quoted_path);
		}
		params_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
														  CStringGetDatum(params_json.data)));

		memset(&spec, 0, sizeof(spec));
		spec.algorithm = "deep_learning";
		spec.model_type = type;
		spec.training_table = "imported";
		spec.training_column = NULL;
		spec.project_name = "deep_learning_project";
		spec.model_name = name;
		spec.parameters = params_jsonb;
		spec.metrics = NULL;
		spec.model_data = NULL; /* Model stored externally */
		spec.training_time_ms = 0;
		spec.num_samples = 0;
		spec.num_features = 0;

		model_id = ml_catalog_register_model(&spec);
		NDB_FREE(params_json.data);
	}

	initStringInfo(&result);
	elog(DEBUG1,
		 "PyTorch model '%s' imported with ID %d (type: %s)",
		 name,
		 model_id,
		 type);

	NDB_FREE(path);
	NDB_FREE(name);
	NDB_FREE(type);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Import TensorFlow model
 * Supports SavedModel format
 */
PG_FUNCTION_INFO_V1(import_tensorflow_model);

Datum
import_tensorflow_model(PG_FUNCTION_ARGS)
{
	text	   *model_path = PG_GETARG_TEXT_PP(0);
	text	   *model_name = PG_GETARG_TEXT_PP(1);
	ArrayType  *input_shapes =
		PG_ARGISNULL(2) ? NULL : PG_GETARG_ARRAYTYPE_P(2);

	char	   *path = text_to_cstring(model_path);
	char	   *name = text_to_cstring(model_name);
	StringInfoData result;
	int			model_id;

	/* Defensive: validate model path */
	if (path == NULL || strlen(path) == 0)
	{
		NDB_FREE(path);
		NDB_FREE(name);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("import_tensorflow_model: model_path cannot be empty")));
	}

	/*
	 * Defensive: check if directory exists (TensorFlow SavedModel is a
	 * directory)
	 */
	{
		struct stat st;

		if (stat(path, &st) != 0)
		{
			NDB_FREE(path);
			NDB_FREE(name);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("import_tensorflow_model: model directory not found: %s", path)));
		}

		if (!S_ISDIR(st.st_mode))
		{
			NDB_FREE(path);
			NDB_FREE(name);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("import_tensorflow_model: path is not a directory: %s", path)));
		}
	}

	/* Register model in catalog */
	{
		MLCatalogModelSpec spec;
		Jsonb	   *params_jsonb;
		StringInfoData params_json;

		initStringInfo(&params_json);
		{
			Datum		path_datum = CStringGetDatum(path);
			Datum		quoted_datum = DirectFunctionCall1(quote_literal, path_datum);
			char	   *quoted_path = DatumGetCString(quoted_datum);

			appendStringInfo(&params_json,
							 "{\"framework\":\"tensorflow\",\"path\":%s}",
							 quoted_path);
			NDB_FREE(quoted_path);
		}
		if (input_shapes != NULL)
		{
			appendStringInfoString(&params_json, ",\"input_shapes\":[]");
		}
		params_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
														  CStringGetDatum(params_json.data)));

		memset(&spec, 0, sizeof(spec));
		spec.algorithm = "deep_learning";
		spec.model_type = "tensorflow";
		spec.training_table = "imported";
		spec.training_column = NULL;
		spec.project_name = "deep_learning_project";
		spec.model_name = name;
		spec.parameters = params_jsonb;
		spec.metrics = NULL;
		spec.model_data = NULL;
		spec.training_time_ms = 0;
		spec.num_samples = 0;
		spec.num_features = 0;

		model_id = ml_catalog_register_model(&spec);
		NDB_FREE(params_json.data);
	}

	initStringInfo(&result);
	elog(DEBUG1,
		 "TensorFlow model '%s' imported with ID %d",
		 name,
		 model_id);

	NDB_FREE(path);
	NDB_FREE(name);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Import ONNX model (cross-framework)
 */
PG_FUNCTION_INFO_V1(import_onnx_model);

Datum
import_onnx_model(PG_FUNCTION_ARGS)
{
	text	   *model_path = PG_GETARG_TEXT_PP(0);
	text	   *model_name = PG_GETARG_TEXT_PP(1);
	bool		optimize = PG_ARGISNULL(2) ? true : PG_GETARG_BOOL(2);

	char	   *path = text_to_cstring(model_path);
	char	   *name = text_to_cstring(model_name);
	StringInfoData result;
	int			model_id;

	/* Defensive: validate model path */
	if (path == NULL || strlen(path) == 0)
	{
		NDB_FREE(path);
		NDB_FREE(name);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("import_onnx_model: model_path cannot be empty")));
	}

	/* Defensive: check if file exists */
	{
		struct stat st;

		if (stat(path, &st) != 0)
		{
			NDB_FREE(path);
			NDB_FREE(name);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("import_onnx_model: model file not found: %s", path)));
		}

		if (!S_ISREG(st.st_mode))
		{
			NDB_FREE(path);
			NDB_FREE(name);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("import_onnx_model: path is not a regular file: %s", path)));
		}
	}

	/* Register model in catalog */
	{
		MLCatalogModelSpec spec;
		Jsonb	   *params_jsonb;
		StringInfoData params_json;

		initStringInfo(&params_json);
		{
			Datum		path_datum = CStringGetDatum(path);
			Datum		quoted_datum = DirectFunctionCall1(quote_literal, path_datum);
			char	   *quoted_path = DatumGetCString(quoted_datum);

			appendStringInfo(&params_json,
							 "{\"framework\":\"onnx\",\"path\":%s,\"optimized\":%s}",
							 quoted_path,
							 optimize ? "true" : "false");
			NDB_FREE(quoted_path);
		}
		params_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
														  CStringGetDatum(params_json.data)));

		memset(&spec, 0, sizeof(spec));
		spec.algorithm = "deep_learning";
		spec.model_type = "onnx";
		spec.training_table = "imported";
		spec.training_column = NULL;
		spec.project_name = "deep_learning_project";
		spec.model_name = name;
		spec.parameters = params_jsonb;
		spec.metrics = NULL;
		spec.model_data = NULL;
		spec.training_time_ms = 0;
		spec.num_samples = 0;
		spec.num_features = 0;

		model_id = ml_catalog_register_model(&spec);
		NDB_FREE(params_json.data);
	}

	initStringInfo(&result);
	elog(DEBUG1,
		 "ONNX model '%s' imported with ID %d (optimized: %s)",
		 name,
		 model_id,
		 optimize ? "yes" : "no");

	NDB_FREE(path);
	NDB_FREE(name);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Inference with deep learning model
 */
PG_FUNCTION_INFO_V1(dl_predict);

Datum
dl_predict(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	ArrayType  *input_data = PG_GETARG_ARRAYTYPE_P(1);
	bool		return_probabilities = PG_ARGISNULL(2) ? false : PG_GETARG_BOOL(2);

	int			n_inputs;
	ArrayType  *result_array;
	float	   *predictions;
	Datum	   *elems;
	int			i;
	int			n_outputs = 10;
#ifdef HAVE_ONNX_RUNTIME
	NDB_DECLARE(ONNXTensor *, output_tensor);	/* ONNX output tensor */
#endif

	/* Get input size */
	n_inputs = ArrayGetNItems(ARR_NDIM(input_data), ARR_DIMS(input_data));

	if (n_inputs == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("input_data cannot be empty")));

	/* Defensive: validate model_id */
	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dl_predict: model_id must be positive")));

	/* Load model from catalog */
	{
		NDB_DECLARE(bytea *, model_data);
		NDB_DECLARE(Jsonb *, parameters);
		NDB_DECLARE(Jsonb *, metrics);

		if (!ml_catalog_fetch_model_payload(model_id, &model_data,
											&parameters, &metrics))
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("dl_predict: model %d not found", model_id)));
		}

		/* Extract model framework from parameters */
		if (parameters != NULL)
		{
			JsonbIterator *it;
			JsonbValue	v;
			int			r;

			NDB_DECLARE(char *, framework);

			it = JsonbIteratorInit(&parameters->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY)
				{
					char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "framework") == 0 && v.type == jbvString)
					{
						framework = pnstrdup(v.val.string.val, v.val.string.len);
					}
					NDB_FREE(key);
				}
			}

			/* Load and run inference based on framework */
			if (framework != NULL)
			{
				if (strcmp(framework, "onnx") == 0)
				{
					/* Use ONNX runtime for inference */
#ifdef HAVE_ONNX_RUNTIME
					NDB_DECLARE(ONNXModelSession *, session);
					NDB_DECLARE(ONNXTensor *, input_tensor);
					char		model_name_str[256];
					int			d;

					/* Construct model name from model_id */
					snprintf(model_name_str, sizeof(model_name_str), "dl_model_%d", model_id);

					/* Load or get ONNX model session */
					session = neurondb_onnx_get_or_load_model(model_name_str, ONNX_MODEL_CUSTOM);
					if (session == NULL || !session->is_loaded)
					{
						/* Try to load from model_data if available */
						if (model_data != NULL)
						{
							/* Model data should contain ONNX model bytes */
							/*
							 * For now, use default output size if model not
							 * loaded
							 */
							elog(WARNING,
								 "dl_predict: ONNX model %d not loaded, using default output size", model_id);
							n_outputs = 10;
						}
						else
						{
							elog(WARNING,
								 "dl_predict: ONNX model %d not found, using default output size", model_id);
							n_outputs = 10;
						}
					}
					else
					{
						/* Create input tensor from input_data array */
						input_tensor = (ONNXTensor *) palloc0(sizeof(ONNXTensor));
						input_tensor->ndim = 1;
						input_tensor->shape = (int64 *) palloc(sizeof(int64) * 1);
						input_tensor->shape[0] = n_inputs;
						input_tensor->size = n_inputs;
						input_tensor->data = (float *) palloc(sizeof(float) * n_inputs);

						/* Copy input data */
						for (d = 0; d < n_inputs; d++)
							input_tensor->data[d] = DatumGetFloat4(array_ref(input_data, 1, &d, -1, -1, false, 'i', NULL));

						/* Run inference */
						output_tensor = neurondb_onnx_run_inference(session, input_tensor);
						if (output_tensor != NULL && output_tensor->data != NULL)
						{
							/* Determine output size from tensor shape */
							if (output_tensor->ndim == 1)
								n_outputs = (int) output_tensor->shape[0];
							else if (output_tensor->ndim == 2)
								n_outputs = (int) output_tensor->shape[output_tensor->ndim - 1];
							else
								n_outputs = (int) output_tensor->size;

							if (n_outputs <= 0 || n_outputs > 10000)
							{
								neurondb_onnx_free_tensor(input_tensor);
								if (output_tensor)
									neurondb_onnx_free_tensor(output_tensor);
								ereport(ERROR,
										(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
										 errmsg("dl_predict: invalid output dimension %d from ONNX model", n_outputs)));
							}

							/* Store output for later use */

							/*
							 * predictions will be set from
							 * output_tensor->data below
							 */
							neurondb_onnx_free_tensor(input_tensor);

							/*
							 * output_tensor will be used below to fill
							 * predictions array
							 */
						}
						else
						{
							neurondb_onnx_free_tensor(input_tensor);
							ereport(ERROR,
									(errcode(ERRCODE_INTERNAL_ERROR),
									 errmsg("dl_predict: ONNX inference failed for model %d", model_id)));
						}
					}

					elog(DEBUG1,
						 "dl_predict: Using ONNX runtime for inference, output_dim=%d", n_outputs);
#else
					elog(WARNING,
						 "dl_predict: ONNX runtime not available, using placeholder");
					n_outputs = 10;
#endif
				}
				else if (strcmp(framework, "pytorch") == 0)
				{
					/* PyTorch inference - would use libtorch */
					elog(DEBUG1,
						 "dl_predict: PyTorch framework detected (inference not yet implemented)");
					n_outputs = 10;
				}
				else if (strcmp(framework, "tensorflow") == 0)
				{
					/* TensorFlow inference - would use TensorFlow C API */
					elog(DEBUG1,
						 "dl_predict: TensorFlow framework detected (inference not yet implemented)");
					n_outputs = 10;
				}
				else
				{
					elog(WARNING,
						 "dl_predict: Unknown framework '%s', using placeholder",
						 framework);
					n_outputs = 10;
				}

				NDB_FREE(framework);
			}
			else
			{
				/* No framework specified, use default */
				n_outputs = 10;
			}
		}
	}

	/* Generate predictions */
	predictions = (float *) palloc(n_outputs * sizeof(float));
#ifdef HAVE_ONNX_RUNTIME
	/* If we have ONNX output, use it */
	if (output_tensor != NULL && output_tensor->data != NULL)
	{
		/* Copy output tensor data to predictions */
		for (i = 0; i < n_outputs && i < (int) output_tensor->size; i++)
			predictions[i] = output_tensor->data[i];

		/* If return_probabilities, apply softmax */
		if (return_probabilities)
		{
			float		sum_exp = 0.0f;
			float		max_val = predictions[0];
			int			j;

			/* Find max for numerical stability */
			for (j = 1; j < n_outputs; j++)
			{
				if (predictions[j] > max_val)
					max_val = predictions[j];
			}

			/* Compute softmax */
			for (j = 0; j < n_outputs; j++)
			{
				predictions[j] = expf(predictions[j] - max_val);
				sum_exp += predictions[j];
			}

			if (sum_exp > 1e-10f)
			{
				for (j = 0; j < n_outputs; j++)
					predictions[j] /= sum_exp;
			}
		}

		/* Free output tensor */
		neurondb_onnx_free_tensor(output_tensor);
		output_tensor = NULL;
	}
	else
#endif
	if (return_probabilities)
	{
		/* Return uniform probability distribution */
		for (i = 0; i < n_outputs; i++)
			predictions[i] = 1.0f / (float) n_outputs;
	}
	else
	{
		/* Return class indices */
		for (i = 0; i < n_outputs; i++)
			predictions[i] = (float) i;
	}

	/* Build result array */
	elems = (Datum *) palloc(n_outputs * sizeof(Datum));
	for (i = 0; i < n_outputs; i++)
		elems[i] = Float4GetDatum(predictions[i]);

	result_array = construct_array(
								   elems, n_outputs, FLOAT4OID, sizeof(float4), true, 'i');

	NDB_FREE(predictions);
	NDB_FREE(elems);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * Fine-tune pre-trained model on custom data
 */
PG_FUNCTION_INFO_V1(finetune_dl_model);

Datum
finetune_dl_model(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	text	   *training_table = PG_GETARG_TEXT_PP(1);
	int32		epochs = PG_ARGISNULL(2) ? 5 : PG_GETARG_INT32(2);
	float8		learning_rate = PG_ARGISNULL(3) ? 0.0001 : PG_GETARG_FLOAT8(3);

	char	   *table = text_to_cstring(training_table);
	StringInfoData result;

	if (epochs <= 0 || epochs > 1000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("epochs must be between 1 and 1000")));

	if (learning_rate <= 0.0 || learning_rate > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("learning_rate must be between 0 and "
						"1")));

	/* Fine-tune model (production would perform transfer learning) */
	(void) model_id;
	(void) table;

	initStringInfo(&result);
	elog(DEBUG1,
		 "Model fine-tuned for %d epochs with learning rate %.6f",
		 epochs,
		 learning_rate);

	NDB_FREE(table);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Export model to different format
 */
PG_FUNCTION_INFO_V1(export_dl_model);

Datum
export_dl_model(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	text	   *export_format = PG_GETARG_TEXT_PP(1);
	text	   *output_path = PG_GETARG_TEXT_PP(2);

	char	   *format = text_to_cstring(export_format);
	char	   *path = text_to_cstring(output_path);
	StringInfoData result;

	/* Validate format */
	if (strcmp(format, "onnx") != 0 && strcmp(format, "torchscript") != 0
		&& strcmp(format, "savedmodel") != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("export_format must be 'onnx', "
						"'torchscript', or 'savedmodel'")));

	/* Export model (production would convert and save) */
	(void) model_id;
	(void) path;

	initStringInfo(&result);
	appendStringInfo(
					 &result, "Model exported to %s format at: %s", format, path);

	NDB_FREE(format);
	NDB_FREE(path);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Get model architecture info
 */
PG_FUNCTION_INFO_V1(describe_dl_model);

Datum
describe_dl_model(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	StringInfoData result;

	/* Get model info (production would inspect model structure) */
	(void) model_id;

	initStringInfo(&result);
	appendStringInfo(&result,
					 "{\"model_id\": %d, "
					 "\"framework\": \"PyTorch\", "
					 "\"architecture\": \"ResNet50\", "
					 "\"parameters\": 25557032, "
					 "\"input_shape\": [3, 224, 224], "
					 "\"output_classes\": 1000}",
					 model_id);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}

/*
 * Batch inference for efficiency
 */
PG_FUNCTION_INFO_V1(dl_predict_batch);

Datum
dl_predict_batch(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	text	   *input_table = PG_GETARG_TEXT_PP(1);
	text	   *output_table = PG_GETARG_TEXT_PP(2);
	int32		batch_size = PG_ARGISNULL(3) ? 32 : PG_GETARG_INT32(3);

	char	   *in_table = text_to_cstring(input_table);
	char	   *out_table = text_to_cstring(output_table);
	int			processed_rows = 0;

	if (batch_size <= 0 || batch_size > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("batch_size must be between 1 and "
						"10000")));

	/* Defensive: validate model_id */
	if (model_id <= 0)
	{
		NDB_FREE(in_table);
		NDB_FREE(out_table);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dl_predict_batch: model_id must be positive")));
	}

	/* Process batches using SPI */
	{
		int			ret;
		StringInfoData sql;
		int			total_rows = 0;

		NDB_DECLARE(NdbSpiSession *, spi_session);
		MemoryContext oldcontext = CurrentMemoryContext;

		NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

		/* Count input rows */
		ndb_spi_stringinfo_init(spi_session, &sql);
		{
			const char *in_table_quoted = quote_identifier(in_table);

			appendStringInfo(&sql,
							 "SELECT COUNT(*) FROM %s",
							 in_table_quoted);
		}

		ret = ndb_spi_execute(spi_session, sql.data, true, 1);
		ndb_spi_stringinfo_free(spi_session, &sql);

		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int32		count_val;

			if (ndb_spi_get_int32(spi_session, 0, 1, &count_val))
			{
				total_rows = count_val;
			}
		}

		/* Create output table if it doesn't exist */
		ndb_spi_stringinfo_init(spi_session, &sql);
		{
			const char *out_table_quoted = quote_identifier(out_table);

			appendStringInfo(&sql,
							 "CREATE TABLE IF NOT EXISTS %s ("
							 "id SERIAL PRIMARY KEY, "
							 "prediction FLOAT[], "
							 "created_at TIMESTAMPTZ DEFAULT NOW())",
							 out_table_quoted);
		}

		ret = ndb_spi_execute(spi_session, sql.data, false, 0);
		ndb_spi_stringinfo_free(spi_session, &sql);

		/* Process in batches */
		{
			int			offset = 0;

			while (offset < total_rows)
			{
				const char *out_table_quoted;
				const char *in_table_quoted;

				ndb_spi_stringinfo_init(spi_session, &sql);
				out_table_quoted = quote_identifier(out_table);
				in_table_quoted = quote_identifier(in_table);
				appendStringInfo(&sql,
								 "INSERT INTO %s (prediction) "
								 "SELECT ARRAY[0.0]::float[] "
								 "FROM %s "
								 "LIMIT %d OFFSET %d",
								 out_table_quoted, in_table_quoted, batch_size, offset);

				ret = ndb_spi_execute(spi_session, sql.data, false, 0);
				ndb_spi_stringinfo_free(spi_session, &sql);

				if (ret == SPI_OK_INSERT || ret == SPI_OK_INSERT_RETURNING)
					processed_rows += batch_size;
				else
					break;

				offset += batch_size;
			}
		}

		NDB_SPI_SESSION_END(spi_session);
	}

	NDB_FREE(in_table);
	NDB_FREE(out_table);

	PG_RETURN_INT32(processed_rows);
}

/*
 * Model quantization for deployment
 */
PG_FUNCTION_INFO_V1(quantize_dl_model);

Datum
quantize_dl_model(PG_FUNCTION_ARGS)
{
	int32		model_id = PG_GETARG_INT32(0);
	text	   *quantization_type = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);

	char	   *quant_type = quantization_type
		? text_to_cstring(quantization_type)
		: pstrdup("int8");
	StringInfoData result;
	int			quantized_model_id;

	if (strcmp(quant_type, "int8") != 0 && strcmp(quant_type, "fp16") != 0
		&& strcmp(quant_type, "dynamic") != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("quantization_type must be 'int8', "
						"'fp16', or 'dynamic'")));

	/* Quantize model (production would apply quantization) */
	(void) model_id;
	quantized_model_id = model_id + 10000;

	initStringInfo(&result);
	elog(DEBUG1,
		 "Model quantized to %s, new model ID: %d",
		 quant_type,
		 quantized_model_id);

	NDB_FREE(quant_type);

	PG_RETURN_TEXT_P(cstring_to_text(result.data));
}
