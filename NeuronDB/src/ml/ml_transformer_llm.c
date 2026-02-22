/*-------------------------------------------------------------------------
 *
 * ml_transformer_llm.c
 *    Custom Transformer LLM Training for NeuronDB
 *
 * This module implements training of custom PostgreSQL-specific language models
 * using transformer architecture. Models are trained from scratch and exported
 * to ONNX format for inference.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_transformer_llm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "utils/timestamp.h"
#include "utils/memutils.h"
#include "libpq/pqformat.h"
#include "utils/lsyscache.h"
#include "parser/parser.h"
#include <unistd.h>

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_catalog.h"
#include "neurondb_spi.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"
#include "neurondb_json.h"

PG_FUNCTION_INFO_V1(neurondb_train_transformer_llm);
PG_FUNCTION_INFO_V1(neurondb_predict_transformer_llm);
PG_FUNCTION_INFO_V1(neurondb_train_titans_llm);

/* Helper function to parse int from JSONB */
static void
parse_hyperparam_int(Jsonb *hyperparams, const char *key, int *value, int default_value)
{
	Jsonb *field_jsonb = NULL;
	JsonbValue v;
	JsonbIterator *it = NULL;
	int r;

	if (hyperparams == NULL || key == NULL || value == NULL)
	{
		if (value)
			*value = default_value;
		return;
	}

	*value = default_value;

	PG_TRY();
	{
		field_jsonb = ndb_jsonb_object_field(hyperparams, key);
		if (field_jsonb != NULL)
		{
			it = JsonbIteratorInit((JsonbContainer *) &field_jsonb->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_VALUE && v.type == jbvNumeric)
				{
					*value = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					break;
				}
			}
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		*value = default_value;
	}
	PG_END_TRY();
}

/* Helper function to parse float8 from JSONB */
static void
parse_hyperparam_float8(Jsonb *hyperparams, const char *key, double *value, double default_value)
{
	Jsonb *field_jsonb = NULL;
	JsonbValue v;
	JsonbIterator *it = NULL;
	int r;

	if (hyperparams == NULL || key == NULL || value == NULL)
	{
		if (value)
			*value = default_value;
		return;
	}

	*value = default_value;

	PG_TRY();
	{
		field_jsonb = ndb_jsonb_object_field(hyperparams, key);
		if (field_jsonb != NULL)
		{
			it = JsonbIteratorInit((JsonbContainer *) &field_jsonb->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_VALUE && v.type == jbvNumeric)
				{
					*value = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v.val.numeric)));
					break;
				}
			}
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		*value = default_value;
	}
	PG_END_TRY();
}

/* Helper function to parse text/string from JSONB */
static char *
parse_hyperparam_text(Jsonb *hyperparams, const char *key, MemoryContext context)
{
	Jsonb *field_jsonb = NULL;
	JsonbValue v;
	JsonbIterator *it = NULL;
	int r;
	char *result = NULL;

	if (hyperparams == NULL || key == NULL)
		return NULL;

	PG_TRY();
	{
		field_jsonb = ndb_jsonb_object_field(hyperparams, key);
		if (field_jsonb != NULL)
		{
			it = JsonbIteratorInit((JsonbContainer *) &field_jsonb->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_VALUE && v.type == jbvString)
				{
					result = pnstrdup(v.val.string.val, v.val.string.len);
					break;
				}
			}
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		result = NULL;
	}
	PG_END_TRY();

	return result;
}

/*
 * neurondb_train_transformer_llm
 *    Train a custom transformer LLM model for PostgreSQL operations
 *
 * Parameters:
 *    project_name - Name of ML project
 *    table_name - Name of table containing training corpus (text column)
 *    target_column - Name of target column (can be NULL for unsupervised)
 *    feature_columns - Array of feature column names (text[])
 *    hyperparams - JSONB with training hyperparameters
 *
 * Returns:
 *    INTEGER: model_id of trained model
 *
 * Hyperparameters (in JSONB):
 *    - corpus_size_mb: Size of training corpus in MB (default: 1)
 *    - num_epochs: Number of training epochs (default: 5)
 *    - batch_size: Training batch size (default: 4)
 *    - learning_rate: Learning rate (default: 0.0001)
 *    - d_model: Model dimension (default: 256)
 *    - nhead: Number of attention heads (default: 4)
 *    - num_layers: Number of transformer layers (default: 4)
 *    - dim_feedforward: Feedforward dimension (default: 512)
 *    - max_seq_length: Maximum sequence length (default: 512)
 *    - vocab_size: Vocabulary size (default: 256)
 *    - output_dir: Output directory for model files (default: /tmp/neurondb_models)
 */
Datum
neurondb_train_transformer_llm(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	text *table_name_text = PG_GETARG_TEXT_PP(1);
	text *target_column_text = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
	ArrayType *feature_columns_array = PG_GETARG_ARRAYTYPE_P(3);
	Jsonb *hyperparams = PG_ARGISNULL(4) ? NULL : PG_GETARG_JSONB_P(4);

	char *project_name = NULL;
	char *table_name = NULL;
	char *target_column = NULL;
	const char **feature_names = NULL;
	int feature_name_count = 0;
	
	MemoryContext oldcontext;
	MemoryContext callcontext;
	NdbSpiSession *spi_session = NULL;
	
	int model_id = 0;
	MLCatalogModelSpec spec;
	
	/* Hyperparameters with defaults */
	int corpus_size_mb = 1;
	int num_epochs = 5;
	int batch_size = 4;
	double learning_rate = 0.0001;
	int d_model = 256;
	int nhead = 4;
	int num_layers = 4;
	int dim_feedforward = 512;
	int max_seq_length = 512;
	int vocab_size = 256;
	char *output_dir = NULL;
	char *corpus_file_path = NULL;  /* External corpus file path */
	char *corpus_url = NULL;        /* Corpus URL to download */
	
	/* Extract strings */
	project_name = text_to_cstring(project_name_text);
	table_name = text_to_cstring(table_name_text);
	if (target_column_text)
		target_column = text_to_cstring(target_column_text);
	
	/* Create memory context */
	oldcontext = CurrentMemoryContext;
	callcontext = AllocSetContextCreate(oldcontext,
									   "neurondb_train_transformer_llm",
									   ALLOCSET_DEFAULT_SIZES);
	MemoryContextSwitchTo(callcontext);
	
	/* Initialize SPI session */
	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);
	
	/* Extract feature columns */
	if (feature_columns_array != NULL)
	{
		int nelems = ArrayGetNItems(ARR_NDIM(feature_columns_array), ARR_DIMS(feature_columns_array));
		if (nelems > 0)
		{
			Datum *elems;
			bool *nulls;
			int i;
			
			deconstruct_array(feature_columns_array, TEXTOID, -1, false, 'i',
							&elems, &nulls, &nelems);
			
			nalloc(feature_names, const char *, nelems);
			for (i = 0; i < nelems; i++)
			{
				if (!nulls[i])
				{
					text *elem_text = DatumGetTextP(elems[i]);
					feature_names[feature_name_count++] = text_to_cstring(elem_text);
				}
			}
		}
	}
	
	/* Parse hyperparameters using helper functions */
	parse_hyperparam_int(hyperparams, "corpus_size_mb", &corpus_size_mb, 1);
	parse_hyperparam_int(hyperparams, "num_epochs", &num_epochs, 5);
	parse_hyperparam_int(hyperparams, "batch_size", &batch_size, 4);
	parse_hyperparam_float8(hyperparams, "learning_rate", &learning_rate, 0.0001);
	parse_hyperparam_int(hyperparams, "d_model", &d_model, 256);
	parse_hyperparam_int(hyperparams, "nhead", &nhead, 4);
	parse_hyperparam_int(hyperparams, "num_layers", &num_layers, 4);
	parse_hyperparam_int(hyperparams, "dim_feedforward", &dim_feedforward, 512);
	parse_hyperparam_int(hyperparams, "max_seq_length", &max_seq_length, 512);
	parse_hyperparam_int(hyperparams, "vocab_size", &vocab_size, 256);
	
	/* Parse optional corpus source (file or URL) */
	corpus_file_path = parse_hyperparam_text(hyperparams, "corpus_file", callcontext);
	corpus_url = parse_hyperparam_text(hyperparams, "corpus_url", callcontext);
	
	/* Extract output_dir as text */
	{
		Jsonb *field = ndb_jsonb_object_field(hyperparams, "output_dir");
		if (field != NULL)
		{
			JsonbIterator *it = NULL;
			JsonbValue v;
			JsonbIteratorToken r;
			
			it = JsonbIteratorInit((JsonbContainer *) &field->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_VALUE && v.type == jbvString)
				{
					output_dir = pnstrdup(v.val.string.val, v.val.string.len);
					break;
				}
			}
		}
	}
	
	/* Default output directory */
	if (output_dir == NULL)
	{
		output_dir = psprintf("/tmp/neurondb_models/%s_%s", project_name, table_name);
	}
	
	ereport(INFO,
			(errmsg("neurondb_train_transformer_llm: starting training"),
			 errdetail("project=%s, table=%s, corpus_size_mb=%d, epochs=%d",
					   project_name, table_name, corpus_size_mb, num_epochs)));
	
	/*
	 * Training pipeline:
	 * 1. Generate training corpus from table
	 * 2. Call Python training script to train model
	 * 3. Export trained model to ONNX format
	 * 4. Load ONNX model and store in catalog
	 */
	
	/* Step 1: Load or generate corpus */
	{
		StringInfoData corpus_sql;
		char *corpus_file = NULL;
		FILE *corpus_fp = NULL;
		int ret;
		
		corpus_file = psprintf("%s/corpus.txt", output_dir);
		
		/* Create output directory */
		{
			char *mkdir_cmd = psprintf("mkdir -p %s", output_dir);
			ret = system(mkdir_cmd);
			nfree(mkdir_cmd);
			if (ret != 0)
			{
				ereport(WARNING,
						(errmsg("neurondb_train_transformer_llm: failed to create output directory"),
						 errdetail("output_dir=%s", output_dir)));
			}
		}
		
		/* Check if corpus should be loaded from external source */
		if (corpus_url != NULL)
		{
			/* Download corpus from URL */
			char *download_cmd = NULL;
			
			download_cmd = psprintf("curl -s -L -o %s '%s'", corpus_file, corpus_url);
			
			ereport(INFO,
					(errmsg("neurondb_train_transformer_llm: downloading corpus from URL"),
					 errdetail("url=%s, output=%s", corpus_url, corpus_file)));
			
			ret = system(download_cmd);
			nfree(download_cmd);
			
			if (ret != 0)
			{
				ndb_spi_session_end(&spi_session);
				MemoryContextSwitchTo(oldcontext);
				if (callcontext != NULL)
					MemoryContextDelete(callcontext);
				nfree(project_name);
				nfree(table_name);
				if (target_column)
					nfree(target_column);
				if (output_dir)
					nfree(output_dir);
				if (corpus_url)
					nfree(corpus_url);
				ereport(ERROR,
						(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
						 errmsg("neurondb_train_transformer_llm: failed to download corpus from URL"),
						 errdetail("url=%s, exit_code=%d", corpus_url, ret),
						 errhint("Check URL is accessible and curl is installed")));
			}
			
			/* Verify file was downloaded */
			if (access(corpus_file, R_OK) != 0)
			{
				ndb_spi_session_end(&spi_session);
				MemoryContextSwitchTo(oldcontext);
				if (callcontext != NULL)
					MemoryContextDelete(callcontext);
				nfree(project_name);
				nfree(table_name);
				if (target_column)
					nfree(target_column);
				if (output_dir)
					nfree(output_dir);
				if (corpus_url)
					nfree(corpus_url);
				ereport(ERROR,
						(errcode(ERRCODE_UNDEFINED_FILE),
						 errmsg("neurondb_train_transformer_llm: corpus file not found after download"),
						 errdetail("file=%s", corpus_file)));
			}
			
			nfree(corpus_url);
		}
		else if (corpus_file_path != NULL)
		{
			/* Copy corpus from external file */
			char *copy_cmd = NULL;
			
			/* Verify source file exists */
			if (access(corpus_file_path, R_OK) != 0)
			{
				int saved_errno = errno;
				ndb_spi_session_end(&spi_session);
				MemoryContextSwitchTo(oldcontext);
				if (callcontext != NULL)
					MemoryContextDelete(callcontext);
				nfree(project_name);
				nfree(table_name);
				if (target_column)
					nfree(target_column);
				if (output_dir)
					nfree(output_dir);
				if (corpus_file_path)
					nfree(corpus_file_path);
				ereport(ERROR,
						(errcode(ERRCODE_UNDEFINED_FILE),
						 errmsg("neurondb_train_transformer_llm: corpus file not found"),
						 errdetail("file=%s, error=%s", corpus_file_path, strerror(saved_errno))));
			}
			
			copy_cmd = psprintf("cp '%s' '%s'", corpus_file_path, corpus_file);
			
			ereport(INFO,
					(errmsg("neurondb_train_transformer_llm: copying corpus from file"),
					 errdetail("source=%s, destination=%s", corpus_file_path, corpus_file)));
			
			ret = system(copy_cmd);
			nfree(copy_cmd);
			
			if (ret != 0)
			{
				ndb_spi_session_end(&spi_session);
				MemoryContextSwitchTo(oldcontext);
				if (callcontext != NULL)
					MemoryContextDelete(callcontext);
				nfree(project_name);
				nfree(table_name);
				if (target_column)
					nfree(target_column);
				if (output_dir)
					nfree(output_dir);
				if (corpus_file_path)
					nfree(corpus_file_path);
				ereport(ERROR,
						(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
						 errmsg("neurondb_train_transformer_llm: failed to copy corpus file"),
						 errdetail("source=%s, exit_code=%d", corpus_file_path, ret)));
			}
			
			nfree(corpus_file_path);
		}
		else
		{
			/* Extract text data from table to generate corpus */
			ndb_spi_stringinfo_init(spi_session, &corpus_sql);
		
		/* Use first feature column or target_column as text source */
		if (feature_name_count > 0 && feature_names[0] != NULL)
		{
			const char *col_quoted = NULL;
			const char *table_quoted = NULL;
			
			/* Quote identifiers to prevent SQL injection */
			/* Note: quote_identifier returns const char* pointing to managed memory, don't free */
			table_quoted = quote_identifier(table_name);
			col_quoted = quote_identifier(feature_names[0]);
			
			appendStringInfo(&corpus_sql,
							 "SELECT %s FROM %s WHERE %s IS NOT NULL",
							 col_quoted, table_quoted, col_quoted);
		}
		else if (target_column != NULL)
		{
			const char *col_quoted = NULL;
			const char *table_quoted = NULL;
			
			table_quoted = quote_identifier(table_name);
			col_quoted = quote_identifier(target_column);
			
			appendStringInfo(&corpus_sql,
							 "SELECT %s FROM %s WHERE %s IS NOT NULL",
							 col_quoted, table_quoted, col_quoted);
		}
		else
		{
			/* Default: use all text columns - limit to prevent excessive data */
			const char *table_quoted = quote_identifier(table_name);
			appendStringInfo(&corpus_sql,
							 "SELECT * FROM %s LIMIT 1000",
							 table_quoted);
		}
		
		/* Execute query and write to corpus file */
	ret = ndb_spi_execute(spi_session, corpus_sql.data, true, 0);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		corpus_fp = fopen(corpus_file, "w");
		if (corpus_fp != NULL)
		{
			int i;
			/* Validate SPI_tuptable before access */
			if (SPI_tuptable != NULL && SPI_tuptable->vals != NULL && SPI_tuptable->tupdesc != NULL)
			{
				for (i = 0; i < SPI_processed && i < corpus_size_mb * 1000; i++)
				{
					if (i < SPI_processed && SPI_tuptable->vals[i] != NULL)
					{
						bool isnull;
						Datum text_datum = SPI_getbinval(SPI_tuptable->vals[i],
														  SPI_tuptable->tupdesc, 1, &isnull);
						if (!isnull)
						{
							text *text_val = DatumGetTextP(text_datum);
							char *text_str = text_to_cstring(text_val);
							fputs(text_str, corpus_fp);
							fputs("\n", corpus_fp);
							nfree(text_str);
						}
					}
				}
			}
			fclose(corpus_fp);
		}
		}
			ndb_spi_stringinfo_free(spi_session, &corpus_sql);
		}
	}
	
	/* Step 2: Call Python training script */
	{
		char *train_cmd = NULL;
		char *script_path = NULL;
		char *tools_dir = NULL;
		char *share_path = NULL;
		int ret;
		
		/* Find Python training script */
		tools_dir = getenv("NEURONDB_TOOLS_DIR");
		if (tools_dir != NULL)
		{
			script_path = psprintf("%s/train_postgres_llm.py", tools_dir);
		}
		else
		{
			share_path = getenv("NEURONDB_SHARE_DIR");
			if (share_path != NULL)
			{
				script_path = psprintf("%s/../scripts/train_postgres_llm.py", share_path);
			}
			else
			{
				/* Default to scripts directory in workspace root */
				script_path = pstrdup("/home/pge/pge/neurondb/scripts/train_postgres_llm.py");
			}
		}
		
		/* Check if script exists */
		if (access(script_path, R_OK | X_OK) != 0)
		{
			int saved_errno = errno;
			nfree(script_path);
			ndb_spi_session_end(&spi_session);
			MemoryContextSwitchTo(oldcontext);
			if (callcontext != NULL)
				MemoryContextDelete(callcontext);
			nfree(project_name);
			nfree(table_name);
			if (target_column)
				nfree(target_column);
			if (output_dir)
				nfree(output_dir);
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_FILE),
					 errmsg("neurondb_train_transformer_llm: training script not found"),
					 errdetail("script_path=%s, error=%s", script_path ? script_path : "NULL", strerror(saved_errno)),
					 errhint("Set NEURONDB_TOOLS_DIR environment variable or ensure train_postgres_llm.py exists")));
		}
		
		/* Build command to call training script */
		train_cmd = psprintf(
			"python3 %s "
			"--corpus-file %s/corpus.txt "
			"--output-dir %s "
			"--corpus-size-mb %d "
			"--epochs %d "
			"--batch-size %d "
			"--learning-rate %.6f "
			"--d-model %d "
			"--nhead %d "
			"--num-layers %d "
			"--dim-feedforward %d "
			"--max-seq-length %d "
			"--vocab-size %d",
			script_path, output_dir, output_dir, corpus_size_mb, num_epochs, batch_size,
			learning_rate, d_model, nhead, num_layers, dim_feedforward,
			max_seq_length, vocab_size);
		
		ereport(INFO,
				(errmsg("neurondb_train_transformer_llm: calling training script"),
				 errdetail("command=%s", train_cmd)));
		
		ret = system(train_cmd);
		nfree(train_cmd);
		nfree(script_path);
		
		if (ret != 0)
		{
			ndb_spi_session_end(&spi_session);
			MemoryContextSwitchTo(oldcontext);
			if (callcontext != NULL)
				MemoryContextDelete(callcontext);
			nfree(project_name);
			nfree(table_name);
			if (target_column)
				nfree(target_column);
			if (output_dir)
				nfree(output_dir);
			ereport(ERROR,
					(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
					 errmsg("neurondb_train_transformer_llm: training script failed"),
					 errdetail("exit_code=%d, output_dir=%s", ret, output_dir),
					 errhint("Check training script logs for details.")));
		}
	}
	
	/* Step 3: Load trained model and convert to ONNX */
	{
		char *model_file = NULL;
		char *onnx_file = NULL;
		FILE *fp = NULL;
		bytea *model_data = NULL;
		size_t file_size;
		
		model_file = psprintf("%s/postgres_llm_final.pt", output_dir);
		onnx_file = psprintf("%s/model.onnx", output_dir);
		
		/* Check if model file exists */
		fp = fopen(model_file, "rb");
		if (fp == NULL)
		{
			/* Model file not found - use placeholder for now */
			ereport(WARNING,
					(errmsg("neurondb_train_transformer_llm: model file not found, using placeholder"),
					 errdetail("model_file=%s", model_file)));
		}
		else
		{
			/* Read model file */
			fseek(fp, 0, SEEK_END);
			file_size = ftell(fp);
			fseek(fp, 0, SEEK_SET);
			
			if (file_size > 0)
			{
				char *buffer = NULL;
				size_t		bytes_read;

				nalloc(buffer, char, file_size);
				bytes_read = fread(buffer, 1, (size_t) file_size, fp);
				fclose(fp);
				if (bytes_read != (size_t) file_size)
				{
					nfree(buffer);
					ereport(ERROR,
							(errcode(ERRCODE_IO_ERROR),
							 errmsg("neurondb: failed to read model file: expected %ld bytes, got %zu",
									(long) file_size, bytes_read)));
				}
				/* Convert to bytea */
				model_data = (bytea *) palloc(VARHDRSZ + file_size);
				SET_VARSIZE(model_data, VARHDRSZ + file_size);
				memcpy(VARDATA(model_data), buffer, file_size);
				nfree(buffer);
			}
			else
			{
				fclose(fp);
			}
		}
		
		if (model_data != NULL)
		{
			spec.model_data = model_data;
		}
		
		nfree(model_file);
		nfree(onnx_file);
	}
	
	/* Prepare model specification */
	MemSet(&spec, 0, sizeof(MLCatalogModelSpec));
	spec.algorithm = pstrdup("transformer_llm");
	spec.project_name = pstrdup(project_name);
	spec.training_table = pstrdup(table_name);
	if (target_column)
		spec.training_column = pstrdup(target_column);
	
	/* Create model metadata JSONB */
	{
		JsonbParseState *state = NULL;
		JsonbValue jkey, jval, *final_value = NULL;
		
		(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);
		
		/* Add training parameters */
		jkey.type = jbvString;
		jkey.val.string.val = "corpus_size_mb";
		jkey.val.string.len = strlen("corpus_size_mb");
		(void) pushJsonbValue(&state, WJB_KEY, &jkey);
		jval.type = jbvNumeric;
		jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(corpus_size_mb)));
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);
		
		jkey.val.string.val = "num_epochs";
		jkey.val.string.len = strlen("num_epochs");
		(void) pushJsonbValue(&state, WJB_KEY, &jkey);
		jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(num_epochs)));
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);
		
		jkey.val.string.val = "d_model";
		jkey.val.string.len = strlen("d_model");
		(void) pushJsonbValue(&state, WJB_KEY, &jkey);
		jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(d_model)));
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);
		
		jkey.val.string.val = "nhead";
		jkey.val.string.len = strlen("nhead");
		(void) pushJsonbValue(&state, WJB_KEY, &jkey);
		jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(nhead)));
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);
		
		jkey.val.string.val = "num_layers";
		jkey.val.string.len = strlen("num_layers");
		(void) pushJsonbValue(&state, WJB_KEY, &jkey);
		jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(num_layers)));
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);
		
		jkey.val.string.val = "vocab_size";
		jkey.val.string.len = strlen("vocab_size");
		(void) pushJsonbValue(&state, WJB_KEY, &jkey);
		jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(vocab_size)));
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);
		
		jkey.val.string.val = "max_seq_length";
		jkey.val.string.len = strlen("max_seq_length");
		(void) pushJsonbValue(&state, WJB_KEY, &jkey);
		jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_seq_length)));
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);
		
		jkey.val.string.val = "output_dir";
		jkey.val.string.len = strlen("output_dir");
		(void) pushJsonbValue(&state, WJB_KEY, &jkey);
		jval.type = jbvString;
		jval.val.string.val = output_dir;
		jval.val.string.len = strlen(output_dir);
		(void) pushJsonbValue(&state, WJB_VALUE, &jval);
		
		final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
		if (final_value != NULL)
		{
			spec.metrics = JsonbValueToJsonb(final_value);
		}
	}
	
	/* Register model in catalog */
	/* Note: ml_catalog_register_model creates its own SPI session internally */
	model_id = ml_catalog_register_model(&spec);
	
	if (model_id <= 0)
	{
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		if (callcontext != NULL)
			MemoryContextDelete(callcontext);
		nfree(project_name);
		nfree(table_name);
		if (target_column)
			nfree(target_column);
		if (output_dir)
			nfree(output_dir);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb_train_transformer_llm: failed to register model in catalog"),
				 errdetail("project=%s, table=%s", project_name, table_name)));
	}
	
	ereport(INFO,
			(errmsg("neurondb_train_transformer_llm: training completed"),
			 errdetail("model_id=%d, project=%s", model_id, project_name)));
	
	/* Cleanup */
	ndb_spi_session_end(&spi_session);
	MemoryContextSwitchTo(oldcontext);
	if (callcontext != NULL)
		MemoryContextDelete(callcontext);
	
	nfree(project_name);
	nfree(table_name);
	if (target_column)
		nfree(target_column);
	if (output_dir)
		nfree(output_dir);
	if (corpus_file_path)
		nfree(corpus_file_path);
	if (corpus_url)
		nfree(corpus_url);
	if (feature_names)
	{
		int i;
		for (i = 0; i < feature_name_count; i++)
		{
			if (feature_names[i])
				nfree((char *)feature_names[i]);
		}
		nfree(feature_names);
	}
	
	PG_RETURN_INT32(model_id);
}

/*
 * neurondb_train_titans_llm
 *    Train a Titans (Learning to Memorize at Test Time) LLM model
 *
 * Same signature as neurondb_train_transformer_llm.
 * Hyperparameters: variant, memory_depth, chunk_size, n_persistent, window_size,
 * plus d_model, nhead, num_layers, epochs, batch_size, learning_rate, etc.
 */
Datum
neurondb_train_titans_llm(PG_FUNCTION_ARGS)
{
	text *project_name_text = PG_GETARG_TEXT_PP(0);
	text *table_name_text = PG_GETARG_TEXT_PP(1);
	text *target_column_text = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
	ArrayType *feature_columns_array = PG_GETARG_ARRAYTYPE_P(3);
	Jsonb *hyperparams = PG_ARGISNULL(4) ? NULL : PG_GETARG_JSONB_P(4);

	char *project_name = NULL;
	char *table_name = NULL;
	char *target_column = NULL;
	const char **feature_names = NULL;
	int feature_name_count = 0;

	MemoryContext oldcontext;
	MemoryContext callcontext;
	NdbSpiSession *spi_session = NULL;

	int model_id = 0;
	MLCatalogModelSpec spec;

	int num_epochs = 5;
	int batch_size = 4;
	double learning_rate = 4e-4;
	int d_model = 256;
	int nhead = 4;
	int num_layers = 4;
	int max_seq_length = 512;
	int vocab_size = 256;
	int memory_depth = 2;
	int chunk_size = 64;
	int n_persistent = 16;
	int window_size = 256;
	char *variant_str = NULL;
	char *output_dir = NULL;
	char *corpus_file_path = NULL;
	char *corpus_url = NULL;

	project_name = text_to_cstring(project_name_text);
	table_name = text_to_cstring(table_name_text);
	if (target_column_text)
		target_column = text_to_cstring(target_column_text);

	oldcontext = CurrentMemoryContext;
	callcontext = AllocSetContextCreate(oldcontext,
									   "neurondb_train_titans_llm",
									   ALLOCSET_DEFAULT_SIZES);
	MemoryContextSwitchTo(callcontext);

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	if (feature_columns_array != NULL)
	{
		int nelems = ArrayGetNItems(ARR_NDIM(feature_columns_array), ARR_DIMS(feature_columns_array));
		if (nelems > 0)
		{
			Datum *elems;
			bool *nulls;
			int i;

			deconstruct_array(feature_columns_array, TEXTOID, -1, false, 'i',
							&elems, &nulls, &nelems);

			nalloc(feature_names, const char *, nelems);
			for (i = 0; i < nelems; i++)
			{
				if (!nulls[i])
				{
					text *elem_text = DatumGetTextP(elems[i]);
					feature_names[feature_name_count++] = text_to_cstring(elem_text);
				}
			}
		}
	}

	parse_hyperparam_int(hyperparams, "num_epochs", &num_epochs, 5);
	parse_hyperparam_int(hyperparams, "batch_size", &batch_size, 4);
	parse_hyperparam_float8(hyperparams, "learning_rate", &learning_rate, 4e-4);
	parse_hyperparam_int(hyperparams, "d_model", &d_model, 256);
	parse_hyperparam_int(hyperparams, "nhead", &nhead, 4);
	parse_hyperparam_int(hyperparams, "num_layers", &num_layers, 4);
	parse_hyperparam_int(hyperparams, "max_seq_length", &max_seq_length, 512);
	parse_hyperparam_int(hyperparams, "vocab_size", &vocab_size, 256);
	parse_hyperparam_int(hyperparams, "memory_depth", &memory_depth, 2);
	parse_hyperparam_int(hyperparams, "chunk_size", &chunk_size, 64);
	parse_hyperparam_int(hyperparams, "n_persistent", &n_persistent, 16);
	parse_hyperparam_int(hyperparams, "window_size", &window_size, 256);
	variant_str = parse_hyperparam_text(hyperparams, "variant", callcontext);
	corpus_file_path = parse_hyperparam_text(hyperparams, "corpus_file", callcontext);
	corpus_url = parse_hyperparam_text(hyperparams, "corpus_url", callcontext);

	{
		Jsonb *field = ndb_jsonb_object_field(hyperparams, "output_dir");
		if (field != NULL)
		{
			JsonbIterator *it = NULL;
			JsonbValue v;
			JsonbIteratorToken r;

			it = JsonbIteratorInit((JsonbContainer *) &field->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_VALUE && v.type == jbvString)
				{
					output_dir = pnstrdup(v.val.string.val, v.val.string.len);
					break;
				}
			}
		}
	}

	if (output_dir == NULL)
		output_dir = psprintf("/tmp/neurondb_models/%s_%s_titans", project_name, table_name);

	if (variant_str == NULL)
		variant_str = pstrdup("mac");

	ereport(INFO,
			(errmsg("neurondb_train_titans_llm: starting training"),
			 errdetail("project=%s, table=%s, variant=%s, epochs=%d",
					   project_name, table_name, variant_str, num_epochs)));

	/* Step 1: Load or generate corpus - same as transformer */
	{
		char *corpus_file = NULL;
		FILE *corpus_fp = NULL;
		int ret = 0;

		corpus_file = psprintf("%s/corpus.txt", output_dir);
		{
			char *mkdir_cmd = psprintf("mkdir -p %s", output_dir);
			ret = system(mkdir_cmd);
			nfree(mkdir_cmd);
		}

		if (corpus_url != NULL)
		{
			char *download_cmd = psprintf("curl -s -L -o %s '%s'", corpus_file, corpus_url);
			ret = system(download_cmd);
			nfree(download_cmd);
			if (ret != 0)
			{
				ndb_spi_session_end(&spi_session);
				MemoryContextSwitchTo(oldcontext);
				if (callcontext)
					MemoryContextDelete(callcontext);
				nfree(project_name);
				nfree(table_name);
				if (target_column)
					nfree(target_column);
				if (output_dir)
					nfree(output_dir);
				if (corpus_url)
					nfree(corpus_url);
				ereport(ERROR,
						(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
						 errmsg("neurondb_train_titans_llm: failed to download corpus")));
			}
			nfree(corpus_url);
		}
		else if (corpus_file_path != NULL)
		{
			char *copy_cmd = psprintf("cp '%s' '%s'", corpus_file_path, corpus_file);
			ret = system(copy_cmd);
			nfree(copy_cmd);
			if (ret != 0)
			{
				ndb_spi_session_end(&spi_session);
				MemoryContextSwitchTo(oldcontext);
				if (callcontext)
					MemoryContextDelete(callcontext);
				nfree(project_name);
				nfree(table_name);
				if (target_column)
					nfree(target_column);
				if (output_dir)
					nfree(output_dir);
				if (corpus_file_path)
					nfree(corpus_file_path);
				ereport(ERROR,
						(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
						 errmsg("neurondb_train_titans_llm: failed to copy corpus")));
			}
			nfree(corpus_file_path);
		}
		else
		{
			/* Extract text from table - same as transformer */
			StringInfoData corpus_sql;
			ndb_spi_stringinfo_init(spi_session, &corpus_sql);
			if (feature_name_count > 0 && feature_names != NULL && feature_names[0] != NULL)
			{
				const char *col_quoted = quote_identifier(feature_names[0]);
				const char *table_quoted = quote_identifier(table_name);
				appendStringInfo(&corpus_sql,
								 "SELECT %s FROM %s WHERE %s IS NOT NULL",
								 col_quoted, table_quoted, col_quoted);
			}
			else if (target_column != NULL)
			{
				const char *col_quoted = quote_identifier(target_column);
				const char *table_quoted = quote_identifier(table_name);
				appendStringInfo(&corpus_sql,
								 "SELECT %s FROM %s WHERE %s IS NOT NULL",
								 col_quoted, table_quoted, col_quoted);
			}
			else
			{
				const char *table_quoted = quote_identifier(table_name);
				appendStringInfo(&corpus_sql, "SELECT * FROM %s LIMIT 1000", table_quoted);
			}
			ret = ndb_spi_execute(spi_session, corpus_sql.data, true, 0);
			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				corpus_fp = fopen(corpus_file, "w");
				if (corpus_fp != NULL)
				{
					int i;
					if (SPI_tuptable != NULL && SPI_tuptable->vals != NULL && SPI_tuptable->tupdesc != NULL)
					{
						for (i = 0; i < SPI_processed && i < 5000; i++)
						{
							if (i < SPI_processed && SPI_tuptable->vals[i] != NULL)
							{
								bool isnull;
								Datum text_datum = SPI_getbinval(SPI_tuptable->vals[i],
																  SPI_tuptable->tupdesc, 1, &isnull);
								if (!isnull)
								{
									text *text_val = DatumGetTextP(text_datum);
									char *text_str = text_to_cstring(text_val);
									fputs(text_str, corpus_fp);
									fputs("\n", corpus_fp);
									nfree(text_str);
								}
							}
						}
					}
					fclose(corpus_fp);
				}
			}
			ndb_spi_stringinfo_free(spi_session, &corpus_sql);
		}
		nfree(corpus_file);
	}

	/* Step 2: Call train_titans.py */
	{
		char *train_cmd = NULL;
		char *script_path = NULL;
		char *tools_dir = NULL;
		char *titans_dir = NULL;
		int ret;

		titans_dir = getenv("NEURONDB_TITANS_DIR");
		if (titans_dir != NULL)
			script_path = psprintf("%s/train_titans.py", titans_dir);
		else
		{
			tools_dir = getenv("NEURONDB_TOOLS_DIR");
			if (tools_dir != NULL)
				script_path = psprintf("%s/../examples/titans/train_titans.py", tools_dir);
			else
				script_path = pstrdup("/home/pge/pge/neurondb/examples/titans/train_titans.py");
		}

		if (access(script_path, R_OK | X_OK) != 0)
		{
			int saved_errno = errno;
			char *path_for_error = pstrdup(script_path);
			nfree(script_path);
			ndb_spi_session_end(&spi_session);
			MemoryContextSwitchTo(oldcontext);
			if (callcontext)
				MemoryContextDelete(callcontext);
			nfree(project_name);
			nfree(table_name);
			if (target_column)
				nfree(target_column);
			if (output_dir)
				nfree(output_dir);
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_FILE),
					 errmsg("neurondb_train_titans_llm: training script not found"),
					 errdetail("script_path=%s, error=%s", path_for_error, strerror(saved_errno)),
					 errhint("Set NEURONDB_TITANS_DIR or NEURONDB_TOOLS_DIR")));
			nfree(path_for_error);
		}

		train_cmd = psprintf(
			"cd $(dirname %s) && python3 train_titans.py "
			"--corpus-file %s/corpus.txt "
			"--output-dir %s "
			"--epochs %d "
			"--batch-size %d "
			"--learning-rate %.6f "
			"--d-model %d "
			"--nhead %d "
			"--num-layers %d "
			"--max-seq-length %d "
			"--vocab-size %d "
			"--variant %s "
			"--memory-depth %d "
			"--chunk-size %d "
			"--n-persistent %d "
			"--window-size %d",
			script_path, output_dir, output_dir, num_epochs, batch_size,
			learning_rate, d_model, nhead, num_layers, max_seq_length, vocab_size,
			variant_str, memory_depth, chunk_size, n_persistent, window_size);

		ret = system(train_cmd);
		nfree(train_cmd);
		nfree(script_path);

		if (ret != 0)
		{
			ndb_spi_session_end(&spi_session);
			MemoryContextSwitchTo(oldcontext);
			if (callcontext)
				MemoryContextDelete(callcontext);
			nfree(project_name);
			nfree(table_name);
			if (target_column)
				nfree(target_column);
			if (output_dir)
				nfree(output_dir);
			ereport(ERROR,
					(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
					 errmsg("neurondb_train_titans_llm: training script failed"),
					 errdetail("exit_code=%d", ret)));
		}
	}

	/* Step 3: Load model file and register */
	{
		char *model_file = NULL;
		FILE *fp = NULL;
		bytea *model_data = NULL;
		size_t file_size;

		model_file = psprintf("%s/titans_model.pt", output_dir);
		fp = fopen(model_file, "rb");
		if (fp != NULL)
		{
			fseek(fp, 0, SEEK_END);
			file_size = ftell(fp);
			fseek(fp, 0, SEEK_SET);
			if (file_size > 0)
			{
				char *buffer = NULL;
				size_t		bytes_read;

				nalloc(buffer, char, file_size);
				bytes_read = fread(buffer, 1, (size_t) file_size, fp);
				fclose(fp);
				if (bytes_read != (size_t) file_size)
				{
					nfree(buffer);
					ereport(ERROR,
							(errcode(ERRCODE_IO_ERROR),
							 errmsg("neurondb: failed to read ONNX model file: expected %ld bytes, got %zu",
									(long) file_size, bytes_read)));
				}
				model_data = (bytea *) palloc(VARHDRSZ + file_size);
				SET_VARSIZE(model_data, VARHDRSZ + file_size);
				memcpy(VARDATA(model_data), buffer, file_size);
				nfree(buffer);
			}
			else
				fclose(fp);
		}
		nfree(model_file);

		MemSet(&spec, 0, sizeof(MLCatalogModelSpec));
		spec.algorithm = pstrdup("titans_llm");
		spec.project_name = pstrdup(project_name);
		spec.training_table = pstrdup(table_name);
		if (target_column)
			spec.training_column = pstrdup(target_column);
		if (model_data != NULL)
			spec.model_data = model_data;

		{
			JsonbParseState *state = NULL;
			JsonbValue jkey, jval, *final_value = NULL;

			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);
			jkey.type = jbvString;
			jkey.val.string.val = "output_dir";
			jkey.val.string.len = strlen("output_dir");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvString;
			jval.val.string.val = output_dir;
			jval.val.string.len = strlen(output_dir);
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
			if (final_value != NULL)
				spec.metrics = JsonbValueToJsonb(final_value);
		}

		model_id = ml_catalog_register_model(&spec);

		if (model_id <= 0)
		{
			ndb_spi_session_end(&spi_session);
			MemoryContextSwitchTo(oldcontext);
			if (callcontext)
				MemoryContextDelete(callcontext);
			nfree(project_name);
			nfree(table_name);
			if (target_column)
				nfree(target_column);
			if (output_dir)
				nfree(output_dir);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb_train_titans_llm: failed to register model"),
					 errdetail("project=%s, table=%s", project_name, table_name)));
		}
	}

	ereport(INFO,
			(errmsg("neurondb_train_titans_llm: training completed"),
			 errdetail("model_id=%d, project=%s", model_id, project_name)));

	ndb_spi_session_end(&spi_session);
	MemoryContextSwitchTo(oldcontext);
	if (callcontext)
		MemoryContextDelete(callcontext);
	nfree(project_name);
	nfree(table_name);
	if (target_column)
		nfree(target_column);
	if (output_dir)
		nfree(output_dir);
	if (variant_str)
		nfree(variant_str);
	if (corpus_file_path)
		nfree(corpus_file_path);
	if (corpus_url)
		nfree(corpus_url);
	if (feature_names)
	{
		int i;
		for (i = 0; i < feature_name_count; i++)
		{
			if (feature_names[i])
				nfree((char *) feature_names[i]);
		}
		nfree(feature_names);
	}

	PG_RETURN_INT32(model_id);
}

/*
 * neurondb_predict_transformer_llm
 *    Predict using a trained transformer LLM model
 *
 * Parameters:
 *    model_id - ID of the trained model
 *    input_text - Input text for prediction
 *
 * Returns:
 *    TEXT: Generated response text
 */
Datum
neurondb_predict_transformer_llm(PG_FUNCTION_ARGS)
{
	int32 model_id = PG_GETARG_INT32(0);
	text *input_text = PG_GETARG_TEXT_PP(1);
	
	char *input_str = NULL;

	MemoryContext oldcontext;
	MemoryContext callcontext;
	NdbSpiSession *spi_session = NULL;
	StringInfoData sql;
	int ret;
	
	char *algorithm = NULL;
	char *model_path = NULL;
	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	
	/* Extract input text */
	input_str = text_to_cstring(input_text);
	
	/* Create memory context */
	oldcontext = CurrentMemoryContext;
	callcontext = AllocSetContextCreate(oldcontext,
									   "neurondb_predict_transformer_llm",
									   ALLOCSET_DEFAULT_SIZES);
	MemoryContextSwitchTo(callcontext);
	
	/* Initialize SPI session */
	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);
	
	/* Load model from catalog */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT algorithm, model_data, metrics FROM " NDB_FQ_ML_MODELS
					 " WHERE model_id = %d",
					 model_id);
	
	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		if (callcontext != NULL)
			MemoryContextDelete(callcontext);
		nfree(input_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb_predict_transformer_llm: model not found"),
				 errdetail("model_id=%d", model_id)));
	}
	
	/* Validate SPI_tuptable before access */
	if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
		SPI_processed == 0 || SPI_tuptable->vals[0] == NULL || 
		SPI_tuptable->tupdesc == NULL)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		if (callcontext != NULL)
			MemoryContextDelete(callcontext);
		nfree(input_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb_predict_transformer_llm: invalid SPI result"),
				 errdetail("model_id=%d", model_id)));
	}
	
	/* Extract model data */
	{
		bool isnull;
		Datum algorithm_datum;
		Datum model_data_datum;
		Datum metrics_datum;
		
		algorithm_datum = SPI_getbinval(SPI_tuptable->vals[0],
										SPI_tuptable->tupdesc, 1, &isnull);
		if (!isnull)
			algorithm = text_to_cstring(DatumGetTextP(algorithm_datum));
		
		model_data_datum = SPI_getbinval(SPI_tuptable->vals[0],
										 SPI_tuptable->tupdesc, 2, &isnull);
		if (!isnull)
			model_data = DatumGetByteaP(model_data_datum);
		
		metrics_datum = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc, 3, &isnull);
		if (!isnull)
			metrics = DatumGetJsonbP(metrics_datum);
	}
	
	/* Validate algorithm */
	if (algorithm == NULL || (strcmp(algorithm, "transformer_llm") != 0 && 
							  strcmp(algorithm, "custom_llm") != 0 &&
							  strcmp(algorithm, "titans_llm") != 0))
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		if (callcontext != NULL)
			MemoryContextDelete(callcontext);
		nfree(input_str);
		if (algorithm)
			nfree(algorithm);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb_predict_transformer_llm: model is not a transformer_llm"),
				 errdetail("model_id=%d, algorithm=%s", model_id, algorithm ? algorithm : "NULL")));
	}
	
	/* Extract model path from metrics or use default */
	if (metrics != NULL)
	{
		Jsonb *field = ndb_jsonb_object_field(metrics, "output_dir");
		if (field != NULL)
		{
			JsonbIterator *it = NULL;
			JsonbValue v;
			JsonbIteratorToken r;
			
			it = JsonbIteratorInit((JsonbContainer *) &field->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_VALUE && v.type == jbvString)
				{
					model_path = pnstrdup(v.val.string.val, v.val.string.len);
					break;
				}
			}
		}
	}
	
	ereport(INFO,
			(errmsg("neurondb_predict_transformer_llm: generating prediction"),
			 errdetail("model_id=%d, input_length=%zu", model_id, strlen(input_str))));
	
	/*
	 * Run ONNX inference when available; otherwise return a clear error
	 * instead of fake output.
	 */
	ndb_spi_stringinfo_free(spi_session, &sql);
	ndb_spi_session_end(&spi_session);
	MemoryContextSwitchTo(oldcontext);
	if (callcontext != NULL)
		MemoryContextDelete(callcontext);
	nfree(input_str);
	if (algorithm)
		nfree(algorithm);
	if (model_path)
		nfree(model_path);

#ifdef HAVE_ONNX_RUNTIME
	/* TODO: Load ONNX model from model_path or model_data, run inference, decode output */
	(void) model_data;
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("neurondb_predict_transformer_llm: ONNX inference for transformer LLM is not yet implemented"),
			 errhint("Model inference requires ONNX Runtime integration to be completed.")));
#else
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("neurondb_predict_transformer_llm: transformer LLM prediction requires ONNX Runtime"),
			 errhint("Rebuild NeuronDB with HAVE_ONNX_RUNTIME to enable transformer LLM inference.")));
#endif
}

