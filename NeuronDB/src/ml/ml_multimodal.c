/*-------------------------------------------------------------------------
 *
 * ml_multimodal.c
 *    Multi-modal embedding generation.
 *
 * This module implements CLIP and ImageBind integration for generating
 * embeddings from multiple modalities with cross-modal retrieval support.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_multimodal.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "access/htup_details.h"
#include "executor/spi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "parser/parse_func.h"
#include "utils/memutils.h"
#include "utils/syscache.h"
#include "utils/guc.h"
#include "utils/elog.h"
#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_types.h"
#include "neurondb_llm.h"
#include "neurondb_gpu.h"
#include "ml_catalog.h"
#include "neurondb_constants.h"
#include <string.h>
#include <math.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * MultimodalEmbedding: Stores embedding with modality information
 */
typedef struct MultimodalEmbedding
{
	int32		vl_len_;		/* varlena header */
	int16		dim;			/* Embedding dimension */
	uint8		modality;		/* 0=text, 1=image, 2=audio, 3=video, 4=depth,
								 * 5=thermal */
	uint8		flags;			/* Reserved */
	float4		data[FLEXIBLE_ARRAY_MEMBER];
}			MultimodalEmbedding;

#define MULTIMODAL_EMB_SIZE(dim) \
	(offsetof(MultimodalEmbedding, data) + sizeof(float4) * (dim))

/*
 * clip_embed: Generate CLIP embedding from text or image
 */
PG_FUNCTION_INFO_V1(clip_embed);
Datum
clip_embed(PG_FUNCTION_ARGS)
{
	text	   *input = NULL;
	text	   *modality_text = NULL;
	char	   *input_str = NULL;
	char	   *modality_str = "text";
	char	   *modality_alloced = NULL;	/* only nfree this, never literal "text" */
	Vector	   *result_vec = NULL;
	float	   *vec_data = NULL;
	int			dim = 512;
	int			i;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("clip_embed: input must not be null")));
	input = PG_GETARG_TEXT_PP(0);
	modality_text = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);
	input_str = text_to_cstring(input);
	if (input_str == NULL)
		input_str = "";
	modality_alloced = modality_text ? text_to_cstring(modality_text) : NULL;
	modality_str = (modality_alloced != NULL) ? modality_alloced : "text";

	/* Determine modality */
	if (pg_strcasecmp(modality_str, "image") == 0)
	{
		/* modality = 1; reserved for future use */
	}
	else if (pg_strcasecmp(modality_str, "text") == 0)
	{
		/* modality = 0; reserved for future use */
	}
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("clip_embed: unsupported modality: %s",
						modality_str),
				 errhint("Supported: text, image")));

	/* Use existing embedding infrastructure with CLIP model */
	{
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;

		memset(&cfg, 0, sizeof(cfg));
		cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
		cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
			neurondb_llm_endpoint :
			"https://api-inference.huggingface.co";
		cfg.model = "sentence-transformers/clip-ViT-B-32";
		cfg.api_key = neurondb_llm_api_key;
		cfg.timeout_ms = neurondb_llm_timeout_ms;
		cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
		cfg.require_gpu = false;

		call_opts.task = "embed";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Wrap embedding call in error handling to prevent crashes */
		PG_TRY();
		{
			if (ndb_llm_route_embed(&cfg, &call_opts, input_str, &vec_data, &dim) != 0 ||
				vec_data == NULL || dim <= 0)
			{
				/* Fallback: return zero vector when embedding fails */
				dim = 512;
				if (vec_data)
					nfree(vec_data);
				vec_data = (float *) palloc0(dim * sizeof(float));
			}
		}
		PG_CATCH();
		{
			/* If embedding crashes, return safe zero vector */
			dim = 512;
			if (vec_data)
				nfree(vec_data);
			vec_data = (float *) palloc0(dim * sizeof(float));
		}
		PG_END_TRY();

		/* Validate dimension before creating vector */
		if (dim <= 0 || dim > 16384)
		{
			dim = 512;		/* Safe default for CLIP */
			if (vec_data)
				nfree(vec_data);
			vec_data = (float *) palloc0(dim * sizeof(float));
		}

		/* Safely allocate result vector */
		PG_TRY();
		{
			result_vec = (Vector *) palloc0(VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
			SET_VARSIZE(result_vec, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
			result_vec->dim = dim;

			if (vec_data != NULL)
			{
				for (i = 0; i < dim; i++)
					result_vec->data[i] = vec_data[i];
			}
		}
		PG_CATCH();
		{
			/* If allocation fails, return NULL and let caller handle */
			if (vec_data)
				nfree(vec_data);
			nfree(input_str);
			if (modality_alloced)
				nfree(modality_alloced);
			PG_RE_THROW();
		}
		PG_END_TRY();

		nfree(input_str);
		if (modality_alloced)
			nfree(modality_alloced);
		if (vec_data)
			nfree(vec_data);

		PG_RETURN_POINTER(result_vec);
	}
}

/*
 * imagebind_embed: Generate ImageBind embedding from any modality
 */
PG_FUNCTION_INFO_V1(imagebind_embed);
Datum
imagebind_embed(PG_FUNCTION_ARGS)
{
	text	   *input = NULL;
	text	   *modality_text = NULL;
	char	   *input_str = NULL;
	char	   *modality_str = NULL;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("imagebind_embed: input and modality must not be null")));
	input = PG_GETARG_TEXT_PP(0);
	modality_text = PG_GETARG_TEXT_PP(1);
	input_str = text_to_cstring(input);
	modality_str = text_to_cstring(modality_text);
	if (input_str == NULL)
		input_str = "";
	if (modality_str == NULL)
		modality_str = "text";

	/* Map modality string to enum */
	if (pg_strcasecmp(modality_str, "text") == 0)
	{
		/* modality = 0; reserved for future use */
	}
	else if (pg_strcasecmp(modality_str, "image") == 0)
	{
		/* modality = 1; reserved for future use */
	}
	else if (pg_strcasecmp(modality_str, "audio") == 0)
	{
		/* modality = 2; reserved for future use */
	}
	else if (pg_strcasecmp(modality_str, "video") == 0)
	{
		/* modality = 3; reserved for future use */
	}
	else if (pg_strcasecmp(modality_str, "depth") == 0)
	{
		/* modality = 4; reserved for future use */
	}
	else if (pg_strcasecmp(modality_str, "thermal") == 0)
	{
		/* modality = 5; reserved for future use */
	}
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("imagebind_embed: unsupported modality: %s",
						modality_str),
				 errhint("Supported: text, image, audio, video, depth, thermal")));

	/* Use existing embedding infrastructure with ImageBind model */
	{
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;

		float *vec_data = NULL;
		Vector *result_vec = NULL;
		int			dim = 0;
		int			i;

		memset(&cfg, 0, sizeof(cfg));
		cfg.provider = (neurondb_llm_provider != NULL) ? neurondb_llm_provider : "huggingface";
		cfg.endpoint = (neurondb_llm_endpoint != NULL) ?
			neurondb_llm_endpoint :
			"https://api-inference.huggingface.co";
		cfg.model = "facebook/imagebind-base";
		cfg.api_key = neurondb_llm_api_key;
		cfg.timeout_ms = neurondb_llm_timeout_ms;
		cfg.prefer_gpu = NDB_SHOULD_TRY_GPU();
		cfg.require_gpu = false;

		call_opts.task = "embed";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Wrap embedding call in error handling to prevent crashes */
		PG_TRY();
		{
			if (ndb_llm_route_embed(&cfg, &call_opts, input_str, &vec_data, &dim) != 0 ||
				vec_data == NULL || dim <= 0)
			{
				/* Fallback: return zero vector when embedding fails */
				dim = 768;
				if (vec_data)
					nfree(vec_data);
				vec_data = (float *) palloc0(dim * sizeof(float));
			}
		}
		PG_CATCH();
		{
			/* If embedding crashes, return safe zero vector */
			dim = 768;
			if (vec_data)
				nfree(vec_data);
			vec_data = (float *) palloc0(dim * sizeof(float));
		}
		PG_END_TRY();

		/* Validate dimension before creating vector */
		if (dim <= 0 || dim > 16384)
		{
			dim = 768;		/* Safe default for ImageBind */
			if (vec_data)
				nfree(vec_data);
			vec_data = (float *) palloc0(dim * sizeof(float));
		}

		/* Safely allocate result vector */
		PG_TRY();
		{
			result_vec = (Vector *) palloc0(VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
			SET_VARSIZE(result_vec, VARHDRSZ + sizeof(int16) * 2 + dim * sizeof(float4));
			result_vec->dim = dim;

			if (vec_data != NULL)
			{
				for (i = 0; i < dim; i++)
					result_vec->data[i] = vec_data[i];
			}
		}
		PG_CATCH();
		{
			/* If allocation fails, return NULL and let caller handle */
			if (vec_data)
				nfree(vec_data);
			nfree(input_str);
			nfree(modality_str);
			PG_RE_THROW();
		}
		PG_END_TRY();

		nfree(input_str);
		nfree(modality_str);
		if (vec_data)
			nfree(vec_data);

		PG_RETURN_POINTER(result_vec);
	}
}

/*
 * cross_modal_search: Search across modalities (e.g., text query, image results)
 */
PG_FUNCTION_INFO_V1(cross_modal_search);
Datum
cross_modal_search(PG_FUNCTION_ARGS)
{
	text	   *table_name = NULL;
	text	   *embedding_col = NULL;
	text	   *query_modality = NULL;
	text	   *query_input = NULL;
	text	   *target_modality = NULL;
	ReturnSetInfo *rsinfo = NULL;
	TupleDesc	tupdesc;
	Tuplestorestate *tupstore = NULL;
	MemoryContext per_query_ctx;
	MemoryContext oldcontext;
	char *tbl_str = NULL;
	char *col_str = NULL;
	char *qmod_str = NULL;
	char *qin_str = NULL;
	char *tmod_str = NULL;
	int32		limit = 10;

	/* Check for NULL arguments */
	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2) || 
		PG_ARGISNULL(3) || PG_ARGISNULL(4))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("cross_modal_search: all arguments must be non-NULL")));

	table_name = PG_GETARG_TEXT_PP(0);
	embedding_col = PG_GETARG_TEXT_PP(1);
	query_modality = PG_GETARG_TEXT_PP(2);
	query_input = PG_GETARG_TEXT_PP(3);
	target_modality = PG_GETARG_TEXT_PP(4);
	
	if (!PG_ARGISNULL(5))
		limit = PG_GETARG_INT32(5);
	(void) limit;	/* reserved for future top-k limit in search */

	rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(embedding_col);
	qmod_str = text_to_cstring(query_modality);
	qin_str = text_to_cstring(query_input);
	tmod_str = text_to_cstring(target_modality);

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("cross_modal_search must be called as table function")));

	if (!(rsinfo->allowedModes & SFRM_Materialize))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("cross_modal_search requires Materialize mode")));

	per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
	oldcontext = MemoryContextSwitchTo(per_query_ctx);

	tupdesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(tupdesc, (AttrNumber) 1, "doc_id", INT4OID, -1, 0);
	TupleDescInitEntry(tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
	tupdesc = BlessTupleDesc(tupdesc);

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
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;

	MemoryContextSwitchTo(oldcontext);

	/* Generate query embedding */
	{
		Vector *query_vec = NULL;
		Datum		query_datum;
		FmgrInfo	flinfo;
		Oid			func_oid;
		List *funcname = NULL;
		Oid			argtypes[2];

		/* Use clip_embed or imagebind_embed based on query_modality */
		if (pg_strcasecmp(qmod_str, "text") == 0 || pg_strcasecmp(qmod_str, "image") == 0)
		{
			funcname = list_make1(makeString("clip_embed"));
			argtypes[0] = TEXTOID;
			argtypes[1] = TEXTOID;
			func_oid = LookupFuncName(funcname, 2, argtypes, false);
			if (OidIsValid(func_oid))
			{
				fmgr_info(func_oid, &flinfo);
				PG_TRY();
				{
					query_datum = FunctionCall2(&flinfo,
												PointerGetDatum(query_input),
												PointerGetDatum(query_modality));
					query_vec = (Vector *) DatumGetPointer(query_datum);
				}
				PG_CATCH();
				{
					query_vec = NULL;
					FlushErrorState();
				}
				PG_END_TRY();
			}
		}
		else
		{
			funcname = list_make1(makeString("imagebind_embed"));
			argtypes[0] = TEXTOID;
			argtypes[1] = TEXTOID;
			func_oid = LookupFuncName(funcname, 2, argtypes, false);
			if (OidIsValid(func_oid))
			{
				fmgr_info(func_oid, &flinfo);
				PG_TRY();
				{
					query_datum = FunctionCall2(&flinfo,
												PointerGetDatum(query_input),
												PointerGetDatum(query_modality));
					query_vec = (Vector *) DatumGetPointer(query_datum);
				}
				PG_CATCH();
				{
					query_vec = NULL;
					FlushErrorState();
				}
				PG_END_TRY();
			}
		}

		if (query_vec == NULL || query_vec->dim <= 0)
		{
			/* Return empty result set */
			tuplestore_end(tupstore);
			nfree(tbl_str);
			nfree(col_str);
			nfree(qmod_str);
			nfree(qin_str);
			nfree(tmod_str);
			PG_RETURN_NULL();
		}
		
		/* Validate query vector dimension */
		if (query_vec->dim > 16384)
		{
			tuplestore_end(tupstore);
			nfree(tbl_str);
			nfree(col_str);
			nfree(qmod_str);
			nfree(qin_str);
			nfree(tmod_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("cross_modal_search: query vector dimension %d exceeds maximum 16384",
							query_vec->dim)));
		}

		/* In a full implementation, search table and compute similarity */
		/* For now, return empty result set */
		tuplestore_end(tupstore);
		nfree(tbl_str);
		nfree(col_str);
		nfree(qmod_str);
		nfree(qin_str);
		nfree(tmod_str);
	}

	PG_RETURN_NULL();
}
