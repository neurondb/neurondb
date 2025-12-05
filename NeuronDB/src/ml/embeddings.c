/*-------------------------------------------------------------------------
 *
 * embeddings.c
 *    Text and multimodal embedding generation.
 *
 * This module provides embedding generation functions integrated with
 * the LLM runtime infrastructure.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/embeddings.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "utils/lsyscache.h"
#include "utils/fmgroids.h"
#include "utils/syscache.h"
#include "utils/guc.h"
#include "utils/datum.h"
#include "parser/parse_type.h"
#include "parser/parse_func.h"
#include "nodes/makefuncs.h"
#include "catalog/pg_type.h"
#include "lib/stringinfo.h"
#include "executor/spi.h"
#include "access/tupdesc.h"
#include "funcapi.h"
#include "neurondb.h"
#include "neurondb_llm.h"
#include "neurondb_gpu.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_json.h"
#include "neurondb_constants.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

#if PG_VERSION_NUM >= 150000
#include "utils/jsonb.h"
#else
#include "utils/jsonb.h"
#endif

/* ndb_json_quote_string is now replaced by ndb_json_quote_string from neurondb_json.h */

/*
 * parse_vector_from_text
 *    Parse a vector from a text string in format {dim, x1, x2, ...}
 */
static Vector *
parse_vector_from_text(const char *str)
{
	int			dim = 0;
	int			i = 0;

	NDB_DECLARE(char *, dup);
	NDB_DECLARE(char *, token);
	NDB_DECLARE(char *, saveptr);
	NDB_DECLARE(char *, endptr);
	NDB_DECLARE(Vector *, result);
	NDB_DECLARE(float *, data);
	bool		first = true;

	NDB_DECLARE(char *, start);
	NDB_DECLARE(char *, end);

	/* Duplicate input string as strtok modifies it */
	dup = pstrdup(str);

	/* Skip leading '{' and trailing '}' if present */
	start = strchr(dup, '{');
	end = strrchr(dup, '}');
	if (!start || !end || end <= start)
	{
		NDB_FREE(dup);
		return NULL;
	}
	*end = '\0';
	start++;

	/* Count number of commas for dimension */
	{
		int			commas = 0;
		char	   *c;

		for (c = start; *c; c++)
		{
			if (*c == ',')
				commas++;
		}
		dim = commas;			/* There should be dim+1 entries (first: dim) */
	}
	if (dim < 1)
	{
		NDB_FREE(dup);
		return NULL;
	}
	NDB_ALLOC(data, float, dim);

	/* Tokenize for dimension and entries */
	i = 0;
	first = true;
	for (token = strtok_r(start, ",", &saveptr);
		 token != NULL;
		 token = strtok_r(NULL, ",", &saveptr))
	{
		while (*token == ' ')
			token++;			/* trim leading spaces */

		if (first)
		{
			int			dval;

			first = false;
			dval = strtol(token, &endptr, 10);
			if (*endptr != '\0' || dval <= 0)
			{
				NDB_FREE(data);
				NDB_FREE(dup);
				return NULL;
			}
			if (dval != dim)
			{
				/* might be '{384,x,x,...}' not '{n-entries,...}' */
				dim = dval;
				data = repalloc(data, dim * sizeof(float));
			}
		}
		else
		{
			if (i < dim)
			{
				data[i++] = strtof(token, &endptr);
				if (*endptr != '\0' && *endptr != '\n' && *endptr != '\r')
				{
					NDB_FREE(data);
					NDB_FREE(dup);
					return NULL;
				}
			}
		}
	}
	if (i != dim)
	{
		NDB_FREE(data);
		NDB_FREE(dup);
		return NULL;
	}

	/* Allocate Vector with variable-length data array */
	{
		size_t		vec_size = VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4);

		NDB_DECLARE(char *, vec_bytes);
		NDB_ALLOC(vec_bytes, char, vec_size);
		result = (Vector *) vec_bytes;
	}
	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	result->dim = dim;
	for (i = 0; i < dim; i++)
		result->data[i] = data[i];

	NDB_FREE(dup);
	NDB_FREE(data);

	return result;
}

/* SAFE_PFREE macro removed - use NDB_FREE instead */

/*
 * Helper: Build SQL safe literal for string input
 *    Quotes a string for use in SQL with single quotes, escaping embedded quotes.
 */
static inline char *
to_sql_literal(const char *val)
{
	StringInfoData buf;
	const char *p;

	if (val == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("to_sql_literal: cannot quote NULL string")));

	initStringInfo(&buf);
	appendStringInfoCharMacro(&buf, '\'');
	for (p = val; *p; p++)
	{
		if (*p == '\'')
			appendStringInfoString(&buf, "''");
		else
			appendStringInfoChar(&buf, *p);
	}
	appendStringInfoCharMacro(&buf, '\'');
	return buf.data;
}

/*
 * get_embedding_model_config_internal
 *    Retrieve stored configuration for a model from catalog table.
 *    Returns Jsonb* in caller's memory context, or NULL if not found.
 *    Caller must pfree the result.
 */
static Jsonb *
get_embedding_model_config_internal(const char *model_name)
{
	NDB_DECLARE(Jsonb *, result);
	NDB_DECLARE(NdbSpiSession *, spi_session);
	NDB_DECLARE(char *, sql_str);
	NDB_DECLARE(char *, quoted_model_name);
	StringInfoData sql;
	int			spi_ret;
	MemoryContext oldcontext;

	if (model_name == NULL)
		return NULL;

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Quote model name for SQL with single quotes */
	quoted_model_name = to_sql_literal(model_name);

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT config_json FROM neurondb.embedding_model_config "
					 "WHERE model_name = %s",
					 quoted_model_name);
	sql_str = sql.data;

	/*
	 * Free the quoted string after appending (appendStringInfo copies the
	 * content)
	 */
	NDB_FREE(quoted_model_name);

	spi_ret = ndb_spi_execute(spi_session, sql_str, true, 1);

	if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		result = ndb_spi_get_jsonb(spi_session, 0, 0, oldcontext);
	}

	/* sql.data is managed by PostgreSQL's memory context, not freed manually */
	NDB_SPI_SESSION_END(spi_session);

	return result;
}

/*
 * apply_embedding_model_config
 *    Apply stored configuration to NdbLLMConfig structure.
 *    Merges stored config with GUC defaults.
 */
static void
apply_embedding_model_config(NdbLLMConfig *cfg, const char *model_name)
{
	NDB_DECLARE(Jsonb *, config_jsonb);
	NDB_DECLARE(JsonbIterator *, it);
	JsonbValue	v;
	int			r;

	if (cfg == NULL || model_name == NULL)
		return;

	config_jsonb = get_embedding_model_config_internal(model_name);
	if (config_jsonb == NULL)
		return;

	it = JsonbIteratorInit(&config_jsonb->root);
	while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
	{
		if (r == WJB_KEY)
		{
			NDB_DECLARE(char *, key);
			int			key_len = v.val.string.len;

			key = pnstrdup(v.val.string.val, key_len);
			r = JsonbIteratorNext(&it, &v, false);
			if (r == WJB_VALUE)
			{
				if (strcmp(key, "batch_size") == 0)
				{
					/* batch_size is handled by batch API, not config */
				}
				else if (strcmp(key, "normalize") == 0)
				{
					/* normalize is handled by embedding API, not config */
				}
				else if (strcmp(key, "device") == 0)
				{
					/* device is handled by prefer_gpu/require_gpu */
					if (v.type == jbvString)
					{
						NDB_DECLARE(char *, device_str);

						device_str = pnstrdup(v.val.string.val, v.val.string.len);
						if (pg_strcasecmp(device_str, "gpu") == 0 ||
							pg_strcasecmp(device_str, "cuda") == 0)
						{
							cfg->prefer_gpu = true;
						}
						NDB_FREE(device_str);
					}
				}
				else if (strcmp(key, "timeout_ms") == 0)
				{
					if (v.type == jbvNumeric)
					{
						int32		timeout = DatumGetInt32(
															DirectFunctionCall1(numeric_int4,
																				NumericGetDatum(v.val.numeric)));

						if (timeout >= 100 && timeout <= 300000)
							cfg->timeout_ms = timeout;
					}
				}
			}
			NDB_FREE(key);
		}
	}

	NDB_FREE(config_jsonb);
}

PG_FUNCTION_INFO_V1(embed_text);
/*
 * embed_text
 *    Generate text embedding using Hugging Face API
 *
 * input_text: TEXT
 * model_text: TEXT (optional)
 * Returns: vector
 */
Datum
embed_text(PG_FUNCTION_ARGS)
{
	NDB_DECLARE(text *, input_text);
	NDB_DECLARE(text *, model_text);
	NDB_DECLARE(char *, input_str);
	NDB_DECLARE(char *, model_str);
	NdbLLMConfig cfg;
	NdbLLMCallOptions call_opts;

	NDB_DECLARE(float *, vec_data);
	NDB_DECLARE(Vector *, result);
	int			dim = 0;
	int			i;

	input_text = PG_GETARG_TEXT_PP(0);
	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	input_str = text_to_cstring(input_text);

	if (model_text != NULL)
		model_str = text_to_cstring(model_text);
	else
		model_str = pstrdup("sentence-transformers/all-MiniLM-L6-v2");

	memset(&cfg, 0, sizeof(cfg));
	cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
	cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
		neurondb_llm_endpoint :
		"https://api-inference.huggingface.co";
	cfg.model = model_str != NULL ? model_str :
		(neurondb_llm_model != NULL ?
		 neurondb_llm_model :
		 "sentence-transformers/all-MiniLM-L6-v2");
	cfg.api_key = neurondb_llm_api_key;
	cfg.timeout_ms = neurondb_llm_timeout_ms;
	cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
	cfg.require_gpu = NDB_REQUIRE_GPU();
	if (cfg.provider != NULL &&
		(pg_strcasecmp(cfg.provider, "huggingface-local") == 0 ||
		 pg_strcasecmp(cfg.provider, "hf-local") == 0) &&
		!neurondb_llm_fail_open)
		cfg.require_gpu = true;

	/* Apply stored model configuration if available */
	apply_embedding_model_config(&cfg, cfg.model);

	call_opts.task = "embed";
	call_opts.prefer_gpu = cfg.prefer_gpu;
	call_opts.require_gpu = cfg.require_gpu;
	call_opts.fail_open = neurondb_llm_fail_open;

	if (ndb_llm_route_embed(&cfg, &call_opts, input_str, &vec_data, &dim) != 0 ||
		vec_data == NULL || dim <= 0)
	{
		/* Check if we should fail with error or generate fallback data */
		if (!neurondb_llm_fail_open)
		{
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("neurondb: embedding generation failed"),
					 errhint("Configure neurondb.llm_api_key for Hugging Face API access, or set neurondb.llm_provider='onnx' for local models.")));
		}

		/* Fallback: Generate deterministic embeddings from text hash */
		{
			unsigned int hash;
			int			input_len;
			int			model_len;
			int			j;

			dim = 384;
			NDB_ALLOC(vec_data, float, dim);

			/*
			 * Generate deterministic pseudo-random vector using djb2-like
			 * hash
			 */
			hash = 5381;
			input_len = strlen(input_str);
			model_len = model_str ? strlen(model_str) : 0;

			/* Hash input text */
			for (j = 0; j < input_len; j++)
				hash = ((hash << 5) + hash) ^ (unsigned char) input_str[j];

			/* Generate embedding values based on hash */
			for (i = 0; i < dim; i++)
			{
				if (model_len > 0)
					hash = ((hash << 5) + hash) ^ (unsigned char) (model_str[i % model_len]);
				else
					hash = ((hash << 5) + hash) ^ (unsigned char) (i);
				/* Convert to float in range [-1, 1] */
				vec_data[i] = ((float) ((hash % 2000) - 1000)) / 1000.0f;
			}
		}
	}

	/* Allocate Vector with variable-length data array */
	{
		size_t		vec_size = VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4);

		NDB_DECLARE(char *, vec_bytes);
		NDB_ALLOC(vec_bytes, char, vec_size);
		result = (Vector *) vec_bytes;
	}
	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	result->dim = dim;

	for (i = 0; i < dim; i++)
		result->data[i] = vec_data[i];

	NDB_FREE(input_str);
	NDB_FREE(model_str);
	NDB_FREE(vec_data);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(embed_text_batch);
/*
 * embed_text_batch
 *    Batch text embedding. Accepts TEXT[] and optional model.
 *
 * input_array: TEXT[]
 * model_text: TEXT (optional)
 * Returns: vector[]
 */
Datum
embed_text_batch(PG_FUNCTION_ARGS)
{
	ArrayType  *input_array;

	NDB_DECLARE(text *, model_text);
	NDB_DECLARE(Datum *, text_datums);
	NDB_DECLARE(bool *, text_nulls);
	int			nitems = 0;
	int			i;

	NDB_DECLARE(Datum *, result_datums);
	NDB_DECLARE(bool *, result_nulls);
	NDB_DECLARE(ArrayType *, result);
	Oid			array_oid;
	Oid			vector_oid;
	static Oid cached_vector_oid = InvalidOid;

	input_array = PG_GETARG_ARRAYTYPE_P(0);
	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	deconstruct_array(input_array,
					  TEXTOID,
					  -1,
					  false,
					  'i',
					  &text_datums,
					  &text_nulls,
					  &nitems);

	NDB_ALLOC(result_datums, Datum, nitems);
	NDB_ALLOC(result_nulls, bool, nitems);

	/* Use batch API for better performance */
	{
		NDB_DECLARE(char **, text_cstrs);
		NDB_DECLARE(char *, model_str);
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;
		float	  **vecs = NULL;

		NDB_DECLARE(int *, dims);
		int			num_success = 0;
		int			rc;
		int			j;

		/* Convert text datums to C strings */
		NDB_ALLOC(text_cstrs, char *, nitems);
		for (i = 0; i < nitems; i++)
		{
			if (text_nulls[i])
				text_cstrs[i] = NULL;
			else
				text_cstrs[i] = text_to_cstring((text *) DatumGetPointer(text_datums[i]));
		}

		/* Setup config */
		memset(&cfg, 0, sizeof(cfg));
		cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
		cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
			neurondb_llm_endpoint :
			"https://api-inference.huggingface.co";
		if (model_text != NULL)
		{
			model_str = text_to_cstring(model_text);
			cfg.model = model_str;
		}
		else
		{
			cfg.model = (neurondb_llm_model != NULL ?
						 neurondb_llm_model :
						 "sentence-transformers/all-MiniLM-L6-v2");
		}
		cfg.api_key = neurondb_llm_api_key;
		cfg.timeout_ms = neurondb_llm_timeout_ms;
		cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
		cfg.require_gpu = false;

		/* Apply stored model configuration if available */
		apply_embedding_model_config(&cfg, cfg.model);

		call_opts.task = "embed";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Call batch embedding API */
		rc = ndb_llm_route_embed_batch(&cfg, &call_opts,
									   (const char **) text_cstrs, nitems,
									   &vecs, &dims, &num_success);

		/* Convert results to Datum array */
		if (rc == NDB_LLM_ROUTE_SUCCESS && vecs != NULL && dims != NULL && num_success > 0)
		{
			for (i = 0; i < nitems; i++)
			{
				if (text_nulls[i] || vecs[i] == NULL || dims[i] <= 0)
				{
					result_datums[i] = (Datum) 0;
					result_nulls[i] = true;
				}
				else
				{
					NDB_DECLARE(Vector *, result_vec);
					size_t		vec_size = VARHDRSZ + sizeof(int16) * 2 + dims[i] * sizeof(float4);

					NDB_DECLARE(char *, vec_bytes);
					NDB_ALLOC(vec_bytes, char, vec_size);
					result_vec = (Vector *) vec_bytes;
					SET_VARSIZE(result_vec, VARHDRSZ + sizeof(int16) * 2 + dims[i] * sizeof(float4));
					result_vec->dim = dims[i];
					for (j = 0; j < dims[i]; j++)
						result_vec->data[j] = vecs[i][j];
					result_datums[i] = PointerGetDatum(result_vec);
				}
			}

			/* Free batch results */
			for (i = 0; i < nitems; i++)
			{
				if (vecs[i] != NULL)
					NDB_FREE(vecs[i]);
			}
			NDB_FREE(vecs);
			NDB_FREE(dims);
		}
		else
		{
			/* Free vecs and dims if they were allocated but batch failed */
			if (vecs != NULL)
			{
				for (i = 0; i < nitems; i++)
				{
					if (vecs[i] != NULL)
						NDB_FREE(vecs[i]);
				}
				NDB_FREE(vecs);
			}
			if (dims != NULL)
			{
				NDB_FREE(dims);
			}

			/* Look up embed_text function OID - always use 2-argument version */
			{
				List	   *funcname;
				Oid			argtypes[2];
				Oid			embed_text_oid;
				FmgrInfo	flinfo;
				bool		have_oid = false;

				funcname = list_make1(makeString("embed_text"));
				argtypes[0] = TEXTOID;
				argtypes[1] = TEXTOID;
				embed_text_oid = LookupFuncName(funcname, 2, argtypes, false);
				if (OidIsValid(embed_text_oid))
				{
					fmgr_info(embed_text_oid, &flinfo);
					have_oid = true;
				}
				else
				{
					elog(ERROR, "embed_text_batch: embed_text function not found");
				}
				list_free(funcname);

				/* Batch API failed, fall back to individual calls */
				for (i = 0; i < nitems; i++)
				{
					if (text_nulls[i])
					{
						result_datums[i] = (Datum) 0;
						result_nulls[i] = true;
					}
					else
					{
						Datum		embed_result;
						Vector	   *vec_copy;
						text	   *text_copy;

						/*
						 * Copy text to current memory context before calling
						 * embed_text
						 */
						text_copy = (text *) PG_DETOAST_DATUM_COPY(text_datums[i]);
						if (text_copy == NULL)
						{
							elog(WARNING, "embed_text_batch: text_copy is NULL for item %d", i);
							result_datums[i] = (Datum) 0;
							result_nulls[i] = true;
							continue;
						}

						if (have_oid)
						{
							/*
							 * Call embed_text directly via its internal logic
							 * to avoid function lookup issues
							 */
							text	   *input_text_copy = (text *) text_copy;
							char	   *input_str_copy;

							NDB_DECLARE(char *, model_str_copy);
							NdbLLMConfig cfg_copy;
							NdbLLMCallOptions call_opts_copy;

							NDB_DECLARE(float *, vec_data_copy);
							int			dim_copy = 0;
							int			k;

							/* Extract text string */
							input_str_copy = text_to_cstring(input_text_copy);

							/* Setup model */
							if (model_text != NULL)
								model_str_copy = text_to_cstring(model_text);
							else
								model_str_copy = pstrdup("sentence-transformers/all-MiniLM-L6-v2");

							/* Setup config */
							memset(&cfg_copy, 0, sizeof(cfg_copy));
							cfg_copy.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
							cfg_copy.endpoint = (neurondb_llm_endpoint != NULL) ?
								neurondb_llm_endpoint :
								"https://api-inference.huggingface.co";
							cfg_copy.model = model_str_copy;
							cfg_copy.api_key = neurondb_llm_api_key;
							cfg_copy.timeout_ms = neurondb_llm_timeout_ms;
							cfg_copy.prefer_gpu = NDB_SHOULD_TRY_GPU();
							cfg_copy.require_gpu = false;
							if (cfg_copy.provider != NULL &&
								(pg_strcasecmp(cfg_copy.provider, "huggingface-local") == 0 ||
								 pg_strcasecmp(cfg_copy.provider, "hf-local") == 0) &&
								!neurondb_llm_fail_open)
								cfg_copy.require_gpu = true;

							/* Apply stored model configuration if available */
							apply_embedding_model_config(&cfg_copy, cfg_copy.model);

							call_opts_copy.task = "embed";
							call_opts_copy.prefer_gpu = cfg_copy.prefer_gpu;
							call_opts_copy.require_gpu = cfg_copy.require_gpu;
							call_opts_copy.fail_open = neurondb_llm_fail_open;

							if (ndb_llm_route_embed(&cfg_copy, &call_opts_copy, input_str_copy, &vec_data_copy, &dim_copy) != 0 ||
								vec_data_copy == NULL || dim_copy <= 0)
							{
								dim_copy = 384;
								NDB_ALLOC(vec_data_copy, float, dim_copy);
							}

							/* Allocate Vector */
							{
								size_t		vec_size = VARHDRSZ + sizeof(int16) * 2 + dim_copy * sizeof(float4);

								NDB_DECLARE(char *, vec_bytes);
								NDB_DECLARE(Vector *, result_vec);
								NDB_ALLOC(vec_bytes, char, vec_size);
								result_vec = (Vector *) vec_bytes;
								SET_VARSIZE(result_vec, VARHDRSZ + sizeof(int16) * 2 + dim_copy * sizeof(float4));
								result_vec->dim = dim_copy;
								for (k = 0; k < dim_copy; k++)
									result_vec->data[k] = vec_data_copy[k];
								embed_result = PointerGetDatum(result_vec);
							}

							NDB_FREE(input_str_copy);
							NDB_FREE(model_str_copy);
							NDB_FREE(vec_data_copy);
						}
						else
						{
							elog(ERROR, "embed_text_batch: embed_text function not found");
							embed_result = (Datum) 0;	/* not reached */
						}

						/* Copy Vector to current memory context */
						vec_copy = (Vector *) PG_DETOAST_DATUM_COPY(embed_result);
						result_datums[i] = PointerGetDatum(vec_copy);
						result_nulls[i] = false;
					}
				}
			}
		}

		/* Free C strings */
		for (i = 0; i < nitems; i++)
		{
			NDB_FREE(text_cstrs[i]);
		}
		NDB_FREE(text_cstrs);

		/* Free model_str if allocated (only if model_text was provided) */
		if (model_text != NULL)
		{
			NDB_FREE(model_str);
		}
	}

	array_oid = get_fn_expr_rettype(fcinfo->flinfo);
	vector_oid = get_element_type(array_oid);
	if (!OidIsValid(vector_oid))
	{
		if (!OidIsValid(cached_vector_oid))
		{
			cached_vector_oid = LookupTypeNameOid(NULL,
												  makeTypeNameFromNameList(
																		   list_make1(makeString("vector"))),
												  false);
			if (!OidIsValid(cached_vector_oid))
				ereport(ERROR,
						(errcode(ERRCODE_UNDEFINED_OBJECT),
						 errmsg("vector type not found")));
		}
		vector_oid = cached_vector_oid;
	}

	{
		int16		typlen;
		bool		typbyval;
		char		typalign;
		int			dims[1];
		int			lbs[1];

		get_typlenbyvalalign(vector_oid, &typlen, &typbyval, &typalign);
		dims[0] = nitems;
		lbs[0] = 1;
		result = construct_md_array(result_datums, result_nulls, 1, dims, lbs, vector_oid, typlen, typbyval, typalign);
	}

	NDB_FREE(result_datums);
	NDB_FREE(result_nulls);
	NDB_FREE(text_datums);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(embed_image);
/*
 * embed_image
 *    Generate image embedding using Hugging Face API or similar.
 *
 * image_data: BYTEA
 * model_text: TEXT (optional)
 * Returns: vector
 */
Datum
embed_image(PG_FUNCTION_ARGS)
{
	bytea	   *image_data;

	NDB_DECLARE(text *, model_text);
	NDB_DECLARE(Vector *, result);
	NDB_DECLARE(float *, vec_data);
	int			dim = 0;
	int			i;

	NDB_DECLARE(char *, model_str);
	NdbLLMConfig cfg;

	image_data = PG_GETARG_BYTEA_PP(0);

	/* Validate image data */
	if (image_data == NULL || VARSIZE_ANY_EXHDR(image_data) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("embed_image: image_data must not be NULL or empty")));

	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	if (model_text != NULL)
		model_str = text_to_cstring(model_text);
	else
		model_str = pstrdup("sentence-transformers/clip-ViT-B-32");

	memset(&cfg, 0, sizeof(cfg));
	cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
	cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
		neurondb_llm_endpoint :
		"https://api-inference.huggingface.co";
	cfg.model = model_str != NULL ? model_str :
		(neurondb_llm_model != NULL ?
		 neurondb_llm_model :
		 "sentence-transformers/clip-ViT-B-32");
	cfg.api_key = neurondb_llm_api_key;
	cfg.timeout_ms = neurondb_llm_timeout_ms;
	cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
	cfg.require_gpu = NDB_REQUIRE_GPU();
	if (cfg.provider != NULL &&
		(pg_strcasecmp(cfg.provider, "huggingface-local") == 0 ||
		 pg_strcasecmp(cfg.provider, "hf-local") == 0) &&
		!neurondb_llm_fail_open)
		cfg.require_gpu = true;

	{
		NdbLLMCallOptions call_opts;

		NDB_DECLARE(bytea *, detoasted_image);
		size_t		image_size;
		const unsigned char *image_bytes;
		bool		need_free_detoasted = false;

		/* Detoast the bytea to ensure we have uncompressed data */
		detoasted_image = (bytea *) PG_DETOAST_DATUM(PointerGetDatum(image_data));
		/* Check if detoasting created a new copy */
		if (detoasted_image != image_data)
			need_free_detoasted = true;
		image_size = VARSIZE_ANY_EXHDR(detoasted_image);
		image_bytes = (const unsigned char *) VARDATA_ANY(detoasted_image);

		call_opts.task = "embed";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		if (ndb_llm_route_image_embed(&cfg, &call_opts,
									  image_bytes, image_size, &vec_data, &dim) != 0 ||
			vec_data == NULL || dim <= 0)
		{
			dim = 512;
			NDB_ALLOC(vec_data, float, dim);
		}

		/* Free detoasted image if we created a copy */
		if (need_free_detoasted)
			NDB_FREE(detoasted_image);
	}

	/* Allocate Vector with variable-length data array */
	{
		size_t		vec_size = VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4);

		NDB_DECLARE(char *, vec_bytes);
		NDB_ALLOC(vec_bytes, char, vec_size);
		result = (Vector *) vec_bytes;
	}
	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	result->dim = dim;

	for (i = 0; i < dim; i++)
		result->data[i] = vec_data[i];

	NDB_FREE(vec_data);
	NDB_FREE(model_str);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(embed_multimodal);
/*
 * embed_multimodal
 *    Generate a multimodal (text+image) embedding by combining both.
 *
 * input_text: TEXT
 * image_data: BYTEA
 * model_text: TEXT (optional)
 * Returns: vector
 */
Datum
embed_multimodal(PG_FUNCTION_ARGS)
{
	text	   *input_text;
	bytea	   *image_data;

	NDB_DECLARE(text *, model_text);
	NDB_DECLARE(Vector *, result);
	NDB_DECLARE(float *, vec_data);
	int			dim = 0;
	int			i;

	NDB_DECLARE(char *, input_str);
	NDB_DECLARE(char *, model_str);
#ifdef NDB_HAVE_MULTIMODAL_EMBED
	NdbLLMConfig cfg;
#endif

	input_text = PG_GETARG_TEXT_PP(0);
	image_data = PG_GETARG_BYTEA_PP(1);

	if (PG_ARGISNULL(2))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(2);

	input_str = text_to_cstring(input_text);

	/* Validate inputs */
	if (image_data == NULL || VARSIZE_ANY_EXHDR(image_data) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("embed_multimodal: image_data must not be NULL or empty")));

	if (model_text != NULL)
		model_str = text_to_cstring(model_text);
	else
		model_str = pstrdup("sentence-transformers/clip-ViT-B-32");

	{
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;

		NDB_DECLARE(bytea *, detoasted_image);
		size_t		image_size;
		const unsigned char *image_bytes;
		bool		need_free_detoasted = false;

		/* Detoast the bytea to ensure we have uncompressed data */

		/*
		 * PG_GETARG_BYTEA_PP already handles detoasting, but we need to
		 * ensure it's fully detoasted
		 */
		detoasted_image = (bytea *) PG_DETOAST_DATUM(PointerGetDatum(image_data));
		/* Check if detoasting created a new copy */
		if (detoasted_image != image_data)
			need_free_detoasted = true;
		image_size = VARSIZE_ANY_EXHDR(detoasted_image);
		image_bytes = (const unsigned char *) VARDATA_ANY(detoasted_image);

		memset(&cfg, 0, sizeof(cfg));
		cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
		cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
			neurondb_llm_endpoint :
			"https://api-inference.huggingface.co";
		cfg.model = model_str != NULL ? model_str :
			(neurondb_llm_model != NULL ?
			 neurondb_llm_model :
			 "sentence-transformers/clip-ViT-B-32");
		cfg.api_key = neurondb_llm_api_key;
		cfg.timeout_ms = neurondb_llm_timeout_ms;
		cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
		cfg.require_gpu = false;
		if (cfg.provider != NULL &&
			(pg_strcasecmp(cfg.provider, "huggingface-local") == 0 ||
			 pg_strcasecmp(cfg.provider, "hf-local") == 0) &&
			!neurondb_llm_fail_open)
			cfg.require_gpu = true;

		call_opts.task = "embed";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Try true multimodal embedding first */
		if (ndb_llm_route_multimodal_embed(&cfg, &call_opts,
										   input_str, image_bytes, image_size, &vec_data, &dim) != 0 ||
			vec_data == NULL || dim <= 0)
		{
			/* Fallback: concatenate text and image embeddings */
			NDB_DECLARE(float *, text_vec);
			NDB_DECLARE(float *, img_vec);
			int			text_dim = 0;
			int			img_dim = 0;

			if (ndb_llm_route_embed(&cfg, &call_opts, input_str, &text_vec, &text_dim) == 0 &&
				ndb_llm_route_image_embed(&cfg, &call_opts, image_bytes, image_size, &img_vec, &img_dim) == 0 &&
				text_vec != NULL && img_vec != NULL && text_dim > 0 && img_dim > 0)
			{
				dim = text_dim + img_dim;
				NDB_ALLOC(vec_data, float, dim);
				memcpy(vec_data, text_vec, text_dim * sizeof(float));
				memcpy(vec_data + text_dim, img_vec, img_dim * sizeof(float));
				NDB_FREE(text_vec);
				NDB_FREE(img_vec);
			}
			else
			{
				dim = 512;
				NDB_ALLOC(vec_data, float, dim);
				NDB_FREE(text_vec);
				NDB_FREE(img_vec);
			}
		}

		/* Free detoasted image if we created a copy */
		if (need_free_detoasted && detoasted_image != NULL)
			NDB_FREE(detoasted_image);
	}

	/* Allocate Vector with variable-length data array */
	{
		size_t		vec_size = VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4);

		NDB_DECLARE(char *, vec_bytes);
		NDB_ALLOC(vec_bytes, char, vec_size);
		result = (Vector *) vec_bytes;
	}
	SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
	result->dim = dim;

	for (i = 0; i < dim; i++)
		result->data[i] = vec_data[i];

	NDB_FREE(vec_data);
	NDB_FREE(input_str);
	NDB_FREE(model_str);

	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(embed_cached);
/*
 * embed_cached
 *    Cached text embedding.
 *
 * input_text: TEXT
 * model_text: TEXT (optional)
 * Returns: vector using cache if available
 */
Datum
embed_cached(PG_FUNCTION_ARGS)
{
	NDB_DECLARE(text *, input_text);
	NDB_DECLARE(text *, model_text);
	NDB_DECLARE(char *, input_str);
	NDB_DECLARE(char *, model_str);
	NDB_DECLARE(char *, cache_key);
	NDB_DECLARE(char *, cached_text);
	int			max_age;

	NDB_DECLARE(Vector *, result);
	StringInfoData key_buf;

	/* bool hit = false; */
	const char *p;
	uint32		hashval;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	input_text = PG_GETARG_TEXT_PP(0);
	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	input_str = text_to_cstring(input_text);

	initStringInfo(&key_buf);
	appendStringInfo(&key_buf, "embed:");

	hashval = 0;
	for (p = input_str; *p; p++)
		hashval = (hashval * 31) + *p;
	appendStringInfo(&key_buf, "%u:", (unsigned int) hashval);

	if (model_text != NULL)
	{
		model_str = text_to_cstring(model_text);
		appendStringInfoString(&key_buf, model_str);
	}
	else
	{
		appendStringInfoString(&key_buf, "default");
	}

	/* Copy cache_key to current context */
	cache_key = pstrdup(key_buf.data);
	NDB_FREE(key_buf.data);
	max_age = neurondb_llm_cache_ttl;

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Cache lookup (uses SPI) */
	if (ndb_llm_cache_lookup(cache_key, max_age, &cached_text) && cached_text != NULL)
	{
		result = parse_vector_from_text(cached_text);
		NDB_FREE(cached_text);
	}

	/* If cache miss or parse failed, generate embedding */
	if (result == NULL)
	{
		StringInfoData query;
		int			spi_ret;
		bool		isnull;
		Datum		vec_datum;
		MemoryContext session_ctx;

		ndb_spi_stringinfo_init(spi_session, &query);
		if (model_text != NULL)
		{
			appendStringInfo(&query, "SELECT embed_text(%s, %s)",
							 quote_literal_cstr(input_str), quote_literal_cstr(model_str));
		}
		else
		{
			appendStringInfo(&query, "SELECT embed_text(%s)",
							 quote_literal_cstr(input_str));
		}

		spi_ret = ndb_spi_execute(spi_session, query.data, true, 1);
		ndb_spi_stringinfo_free(spi_session, &query);

		if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			/* Extract Vector from SPI result - copy to caller's context */
			if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL && SPI_tuptable->vals != NULL)
			{
				vec_datum = SPI_getbinval(SPI_tuptable->vals[0],
										  SPI_tuptable->tupdesc,
										  1,
										  &isnull);
				if (!isnull)
				{
					session_ctx = MemoryContextSwitchTo(oldcontext);
					result = (Vector *) PG_DETOAST_DATUM_COPY(vec_datum);
					MemoryContextSwitchTo(session_ctx);
				}
			}
		}
	}

	/* Bulletproof fallback: if result is still NULL, return zero vector */
	if (result == NULL)
	{
		int			dim = 384;

		/* Allocate Vector with variable-length data array */
		{
			size_t		vec_size = VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4);

			NDB_DECLARE(char *, vec_bytes);
			NDB_ALLOC(vec_bytes, char, vec_size);
			result = (Vector *) vec_bytes;
		}
		SET_VARSIZE(result, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
		result->dim = dim;
	}

	/* Store in cache if result is valid and not from cache */
	if (result != NULL && cached_text == NULL)
	{
		StringInfoData cache_val;
		int			d;

		initStringInfo(&cache_val);
		appendStringInfo(&cache_val, "{%d", result->dim);
		for (d = 0; d < result->dim; d++)
			appendStringInfo(&cache_val, ",%.7f", result->data[d]);
		appendStringInfoChar(&cache_val, '}');
		ndb_llm_cache_store(cache_key, cache_val.data);
		NDB_FREE(cache_val.data);
	}

	NDB_SPI_SESSION_END(spi_session);
	NDB_FREE(input_str);
	NDB_FREE(model_str);
	NDB_FREE(cache_key);
	PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(configure_embedding_model);
/*
 * configure_embedding_model
 *    Configure embedding model parameters in persistent storage.
 *
 * model_name: TEXT
 * config_json: TEXT
 * Returns: BOOLEAN
 */
Datum
configure_embedding_model(PG_FUNCTION_ARGS)
{
	NDB_DECLARE(text *, model_name);
	NDB_DECLARE(text *, config_json);
	NDB_DECLARE(char *, model_str);
	NDB_DECLARE(char *, cfg_str);
	NDB_DECLARE(Jsonb *, config_jsonb);
	NDB_DECLARE(JsonbIterator *, it);
	JsonbValue	v;
	int			r;
	bool		valid = true;
	int			spi_ret;
	Oid			argtypes[2];
	Datum		values[2];
	char		nulls[2];

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	model_name = PG_GETARG_TEXT_PP(0);
	config_json = PG_GETARG_TEXT_PP(1);

	if (model_name == NULL || config_json == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("configure_embedding_model: model_name and config_json must not be NULL")));

	model_str = text_to_cstring(model_name);
	cfg_str = text_to_cstring(config_json);

	/* Parse and validate JSON */
	{
		Oid			jsonb_oid = JSONBOID;
		Oid			typinput;
		Oid			typioparam;

		getTypeInputInfo(jsonb_oid, &typinput, &typioparam);
		config_jsonb = (Jsonb *) DatumGetPointer(
												 OidFunctionCall3(typinput,
																  CStringGetDatum(cfg_str),
																  ObjectIdGetDatum(InvalidOid),
																  Int32GetDatum(-1)));

		if (config_jsonb == NULL)
		{
			NDB_FREE(model_str);
			NDB_FREE(cfg_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("configure_embedding_model: invalid JSON in config_json")));
		}
	}

	/* Validate JSON structure and values */
	it = JsonbIteratorInit(&config_jsonb->root);
	while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE && valid)
	{
		if (r == WJB_KEY)
		{
			NDB_DECLARE(char *, key);
			int			key_len = v.val.string.len;

			key = pnstrdup(v.val.string.val, key_len);
			r = JsonbIteratorNext(&it, &v, false);
			if (r == WJB_VALUE)
			{
				if (strcmp(key, "batch_size") == 0)
				{
					if (v.type == jbvNumeric)
					{
						int32		batch_size = DatumGetInt32(
															   DirectFunctionCall1(numeric_int4,
																				   NumericGetDatum(v.val.numeric)));

						if (batch_size < 1 || batch_size > 10000)
						{
							valid = false;
							elog(WARNING,
								 "configure_embedding_model: batch_size must be between 1 and 10000, got %d",
								 batch_size);
						}
					}
					else
					{
						valid = false;
						elog(WARNING,
							 "configure_embedding_model: batch_size must be a number");
					}
				}
				else if (strcmp(key, "normalize") == 0)
				{
					if (v.type != jbvBool)
					{
						valid = false;
						elog(WARNING,
							 "configure_embedding_model: normalize must be a boolean");
					}
				}
				else if (strcmp(key, "device") == 0)
				{
					if (v.type != jbvString)
					{
						valid = false;
						elog(WARNING,
							 "configure_embedding_model: device must be a string");
					}
				}
				else if (strcmp(key, "timeout_ms") == 0)
				{
					if (v.type == jbvNumeric)
					{
						int32		timeout = DatumGetInt32(
															DirectFunctionCall1(numeric_int4,
																				NumericGetDatum(v.val.numeric)));

						if (timeout < 100 || timeout > 300000)
						{
							valid = false;
							elog(WARNING,
								 "configure_embedding_model: timeout_ms must be between 100 and 300000, got %d",
								 timeout);
						}
					}
					else
					{
						valid = false;
						elog(WARNING,
							 "configure_embedding_model: timeout_ms must be a number");
					}
				}
			}
			NDB_FREE(key);
		}
	}

	if (!valid)
	{
		NDB_FREE(model_str);
		NDB_FREE(cfg_str);
		PG_RETURN_BOOL(false);
	}

	/* Store configuration in catalog table using SPI */
	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	memset(nulls, ' ', sizeof(nulls));
	argtypes[0] = TEXTOID;
	argtypes[1] = JSONBOID;

	values[0] = CStringGetTextDatum(model_str);
	values[1] = JsonbPGetDatum(config_jsonb);

	spi_ret = ndb_spi_execute_with_args(spi_session,
										"INSERT INTO neurondb.embedding_model_config "
										"(model_name, config_json, created_at, updated_at) "
										"VALUES ($1, $2, now(), now()) "
										"ON CONFLICT (model_name) DO UPDATE "
										"SET config_json = EXCLUDED.config_json, "
										"    updated_at = now()",
										2,
										argtypes,
										values,
										nulls,
										false,
										0);

	NDB_SPI_SESSION_END(spi_session);

	if (spi_ret != SPI_OK_INSERT && spi_ret != SPI_OK_UPDATE)
	{
		NDB_FREE(model_str);
		NDB_FREE(cfg_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("configure_embedding_model: failed to store configuration")));
	}

	elog(DEBUG1,
		 "neurondb: configure_embedding_model: model='%s', config stored successfully",
		 model_str);

	NDB_FREE(model_str);
	NDB_FREE(cfg_str);

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(get_embedding_model_config);
/*
 * get_embedding_model_config
 *    Retrieve stored configuration for a model.
 *
 * model_name: TEXT
 * Returns: JSONB
 */
Datum
get_embedding_model_config(PG_FUNCTION_ARGS)
{
	NDB_DECLARE(text *, model_name);
	NDB_DECLARE(char *, model_str);
	NDB_DECLARE(Jsonb *, result);

	model_name = PG_GETARG_TEXT_PP(0);
	if (model_name == NULL)
		PG_RETURN_NULL();

	model_str = text_to_cstring(model_name);
	result = get_embedding_model_config_internal(model_str);
	NDB_FREE(model_str);

	if (result == NULL)
		PG_RETURN_NULL();

	PG_RETURN_JSONB_P(result);
}

PG_FUNCTION_INFO_V1(list_embedding_model_configs);
/*
 * list_embedding_model_configs
 *    List all stored embedding model configurations.
 *
 * Returns: TABLE(model_name TEXT, config_json JSONB, created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ)
 */
Datum
list_embedding_model_configs(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore;
	MemoryContext per_query_ctx;
	MemoryContext oldcontext;
	StringInfoData sql;
	int			spi_ret;
	int			i;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	NDB_DECLARE(char *, sql_str);

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("set-valued function called in context that cannot accept a set")));

	if (rsinfo->expectedDesc == NULL)
	{
		TupleDesc	desc;

		desc = CreateTemplateTupleDesc(4);
		TupleDescInitEntry(desc, (AttrNumber) 1, "model_name", TEXTOID, -1, 0);
		TupleDescInitEntry(desc, (AttrNumber) 2, "config_json", JSONBOID, -1, 0);
		TupleDescInitEntry(desc, (AttrNumber) 3, "created_at", TIMESTAMPTZOID, -1, 0);
		TupleDescInitEntry(desc, (AttrNumber) 4, "updated_at", TIMESTAMPTZOID, -1, 0);
		tupdesc = BlessTupleDesc(desc);
		rsinfo->returnMode = SFRM_Materialize;
		rsinfo->setDesc = tupdesc;
		rsinfo->expectedDesc = tupdesc;
	}
	else
	{
		tupdesc = rsinfo->expectedDesc;
		rsinfo->returnMode = SFRM_Materialize;
		rsinfo->setDesc = tupdesc;
	}
	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

	/* Get work_mem GUC setting */
	{
		const char *work_mem_str = GetConfigOption("work_mem", true, false);
		int			work_mem_kb = 262144;	/* Default 256MB */

		if (work_mem_str)
		{
			work_mem_kb = atoi(work_mem_str);
			if (work_mem_kb <= 0)
				work_mem_kb = 262144;
		}
		tupstore = tuplestore_begin_heap(true, false, work_mem_kb);
	}
	rsinfo->setResult = tupstore;

	MemoryContextSwitchTo(oldcontext);

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	initStringInfo(&sql);
	appendStringInfo(&sql,
					 "SELECT model_name, config_json, created_at, updated_at "
					 "FROM neurondb.embedding_model_config "
					 "ORDER BY model_name");
	sql_str = sql.data;

	spi_ret = ndb_spi_execute(spi_session, sql_str, true, 0);

	if (spi_ret == SPI_OK_SELECT)
	{
		for (i = 0; i < (int) SPI_processed; i++)
		{
			Datum		values[4];
			bool		nulls[4];
			text	   *model_name_text;
			Jsonb	   *config_jsonb;
			Datum		created_at_datum;
			Datum		updated_at_datum;
			bool		created_at_isnull;
			bool		updated_at_isnull;

			memset(nulls, 0, sizeof(nulls));

			/* Column 0: model_name (text) */
			model_name_text = ndb_spi_get_text(spi_session, i, 1, per_query_ctx);
			if (model_name_text != NULL)
			{
				values[0] = PointerGetDatum(model_name_text);
				nulls[0] = false;
			}
			else
			{
				nulls[0] = true;
			}

			/* Column 1: config_json (jsonb) */
			config_jsonb = ndb_spi_get_jsonb(spi_session, i, 2, per_query_ctx);
			if (config_jsonb != NULL)
			{
				values[1] = JsonbPGetDatum(config_jsonb);
				nulls[1] = false;
			}
			else
			{
				nulls[1] = true;
			}

			/*
			 * Columns 2-3: created_at, updated_at (timestamps) - safe access
			 * pattern
			 */
			if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL && SPI_tuptable->vals != NULL && i < (int) SPI_processed)
			{
				/* Column 2: created_at */
				created_at_datum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 3, &created_at_isnull);
				if (!created_at_isnull)
				{
					Oid			created_at_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 3);
					int16		typlen;
					bool		typbyval;

					{
						char		typalign;

						get_typlenbyvalalign(created_at_type_oid, &typlen, &typbyval, &typalign);
					}
					if (typbyval)
					{
						/*
						 * Pass-by-value type (like TIMESTAMPTZ) - no need to
						 * copy, just use directly
						 */
						values[2] = created_at_datum;
					}
					else
					{
						/*
						 * Pass-by-reference type - need to copy to
						 * destination context
						 */
						MemoryContextSwitchTo(per_query_ctx);
						values[2] = datumCopy(created_at_datum, typlen, typbyval);
						MemoryContextSwitchTo(oldcontext);
					}
					nulls[2] = false;
				}
				else
				{
					nulls[2] = true;
				}

				/* Column 3: updated_at */

				/*
				 * Safe access for updated_at - validate tupdesc has at least
				 * 4 columns
				 */
				if (SPI_tuptable->tupdesc->natts >= 4)
				{
					updated_at_datum = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 4, &updated_at_isnull);
					if (!updated_at_isnull)
					{
						Oid			updated_at_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 4);
						int16		typlen;
						bool		typbyval;

						{
							char		typalign;

							get_typlenbyvalalign(updated_at_type_oid, &typlen, &typbyval, &typalign);
						}
						if (typbyval)
						{
							/*
							 * Pass-by-value type (like TIMESTAMPTZ) - no need
							 * to copy, just use directly
							 */
							values[3] = updated_at_datum;
						}
						else
						{
							/*
							 * Pass-by-reference type - need to copy to
							 * destination context
							 */
							MemoryContextSwitchTo(per_query_ctx);
							values[3] = datumCopy(updated_at_datum, typlen, typbyval);
							MemoryContextSwitchTo(oldcontext);
						}
						nulls[3] = false;
					}
					else
					{
						nulls[3] = true;
					}
				}
				else
				{
					nulls[3] = true;
				}
			}
			else
			{
				nulls[2] = true;
				nulls[3] = true;
			}

			tuplestore_putvalues(tupstore, tupdesc, values, nulls);
		}
	}

	NDB_SPI_SESSION_END(spi_session);

	return (Datum) 0;
}

PG_FUNCTION_INFO_V1(delete_embedding_model_config);
/*
 * delete_embedding_model_config
 *    Delete stored configuration for a model.
 *
 * model_name: TEXT
 * Returns: BOOLEAN
 */
Datum
delete_embedding_model_config(PG_FUNCTION_ARGS)
{
	NDB_DECLARE(text *, model_name);
	NDB_DECLARE(char *, model_str);
	int			spi_ret;
	Oid			argtypes[1];
	Datum		values[1];
	char		nulls[1];

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	model_name = PG_GETARG_TEXT_PP(0);
	if (model_name == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("delete_embedding_model_config: model_name must not be NULL")));

	model_str = text_to_cstring(model_name);

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	memset(nulls, ' ', sizeof(nulls));
	argtypes[0] = TEXTOID;
	values[0] = CStringGetTextDatum(model_str);

	spi_ret = ndb_spi_execute_with_args(spi_session,
										"DELETE FROM neurondb.embedding_model_config WHERE model_name = $1",
										1,
										argtypes,
										values,
										nulls,
										false,
										0);

	NDB_SPI_SESSION_END(spi_session);
	NDB_FREE(model_str);

	if (spi_ret == SPI_OK_DELETE)
		PG_RETURN_BOOL(true);
	else
		PG_RETURN_BOOL(false);
}
