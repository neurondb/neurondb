/*-------------------------------------------------------------------------
 *
 * neurondb_json.h
 *    Centralized JSON handling utilities for NeuronDB
 *
 * Provides unified JSON parsing, extraction, quoting, and generation
 * functions with DirectFunctionCall wrappers for PostgreSQL's jsonb
 * functions. Consolidates all JSON handling logic in one place.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    include/neurondb_json.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_JSON_H
#define NEURONDB_JSON_H

#include "postgres.h"
#include "fmgr.h"
#include "utils/jsonb.h"
#include "common/jsonapi.h"
#include "utils/builtins.h"
#include "lib/stringinfo.h"
#include "utils/memutils.h"
#include "parser/parse_type.h"
#include "parser/parse_func.h"
#include "utils/lsyscache.h"
#include "catalog/pg_proc.h"
#include "utils/array.h"
#include "utils/varlena.h"
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <float.h>
#include "neurondb_constants.h"

/* ========== DirectFunctionCall Wrappers for JSONB Functions ========== */

/*
 * ndb_jsonb_in - Convert text to JSONB
 * Wrapper for DirectFunctionCall1(jsonb_in, text_datum)
 */
extern Jsonb * ndb_jsonb_in(text * json_text);

/*
 * ndb_jsonb_in_cstring - Convert C string to JSONB
 * Convenience wrapper that converts C string to text first
 */
extern Jsonb * ndb_jsonb_in_cstring(const char *json_str);

/*
 * ndb_jsonb_out - Convert JSONB to text
 * Wrapper for DirectFunctionCall1(jsonb_out, jsonb_datum)
 */
extern text * ndb_jsonb_out(Jsonb * jsonb);

/*
 * ndb_jsonb_out_cstring - Convert JSONB to C string
 * Convenience wrapper that returns C string
 */
extern char *ndb_jsonb_out_cstring(Jsonb * jsonb);

/*
 * ndb_jsonb_object_field - Extract field from JSONB object
 * Wrapper for DirectFunctionCall2(jsonb_object_field, jsonb_datum, text_datum)
 */
extern Jsonb * ndb_jsonb_object_field(Jsonb * jsonb, const char *field_name);

/*
 * ndb_jsonb_array_element - Extract element from JSONB array
 * Wrapper for DirectFunctionCall2(jsonb_array_element, jsonb_datum, int_datum)
 */
extern Jsonb * ndb_jsonb_array_element(Jsonb * jsonb, int index);

/*
 * ndb_jsonb_extract_path - Extract value by path
 * Wrapper for DirectFunctionCall2(jsonb_extract_path, jsonb_datum, text_array_datum)
 */
extern Jsonb * ndb_jsonb_extract_path(Jsonb * jsonb, const char **path, int path_len);

/*
 * ndb_jsonb_extract_path_text - Extract text value by path
 * Wrapper for DirectFunctionCall2(jsonb_extract_path_text, jsonb_datum, text_array_datum)
 */
extern text * ndb_jsonb_extract_path_text(Jsonb * jsonb, const char **path, int path_len);

/*
 * ndb_jsonb_extract_path_cstring - Extract C string value by path
 * Convenience wrapper that returns C string
 */
extern char *ndb_jsonb_extract_path_cstring(Jsonb * jsonb, const char **path, int path_len);

/*
 * ndb_jsonb_typeof - Get JSONB type
 * Wrapper for DirectFunctionCall1(jsonb_typeof, jsonb_datum)
 */
extern text * ndb_jsonb_typeof(Jsonb * jsonb);

/*
 * ndb_jsonb_typeof_cstring - Get JSONB type as C string
 * Convenience wrapper that returns C string
 */
extern char *ndb_jsonb_typeof_cstring(Jsonb * jsonb);

/*
 * ndb_jsonb_to_text - Convert JSONB to text (alias for jsonb_out)
 */
extern text * ndb_jsonb_to_text(Jsonb * jsonb);

/*
 * ndb_jsonb_build_object - Build JSONB object from key-value pairs
 * Uses variadic DirectFunctionCall for jsonb_build_object
 */
extern Jsonb * ndb_jsonb_build_object(const char *key1, const char *value1,...);

/*
 * ndb_jsonb_build_array - Build JSONB array from values
 * Uses variadic DirectFunctionCall for jsonb_build_array
 */
extern Jsonb * ndb_jsonb_build_array(const char *value1,...);

/* ========== JSON String Quoting and Escaping ========== */

/*
 * ndb_json_quote_string - Quote and escape a C string for JSON
 * Returns JSON-quoted string with proper escaping
 */
extern char *ndb_json_quote_string(const char *str);

/*
 * ndb_json_quote_string_buf - Quote and escape into StringInfo buffer
 * Appends quoted string to existing buffer
 */
extern void ndb_json_quote_string_buf(StringInfo buf, const char *str);

/*
 * ndb_json_unescape_string - Unescape a JSON string
 * Converts escaped JSON string back to normal string
 */
extern char *ndb_json_unescape_string(const char *json_str);

/* ========== JSON Parsing Utilities ========== */

/*
 * JSON parsing result structure
 */
typedef struct NdbJsonParseResult
{
	char	   *key;
	char	   *value;
	int			value_type;		/* 0=string, 1=number, 2=boolean, 3=null,
								 * 4=object, 5=array */
	double		num_value;
	bool		bool_value;
} NdbJsonParseResult;

/*
 * ndb_json_parse_object - Parse JSON object into key-value pairs
 * Returns array of NdbJsonParseResult structures
 */
extern NdbJsonParseResult *ndb_json_parse_object(const char *json_str, int *count);

/*
 * ndb_json_parse_object_free - Free array parsed by ndb_json_parse_object
 * Caller must free the returned array using this function
 */
extern void ndb_json_parse_object_free(NdbJsonParseResult *arr, int count);

/*
 * ndb_json_find_key - Find value for a key in JSON object
 * Returns pointer to value string or NULL if not found
 */
extern char *ndb_json_find_key(const char *json_str, const char *key);

/*
 * ndb_json_extract_string - Extract string value by key
 * Returns allocated string or NULL
 */
extern char *ndb_json_extract_string(const char *json_str, const char *key);

/*
 * ndb_json_extract_number - Extract numeric value by key
 * Returns numeric value, sets found flag
 */
extern double ndb_json_extract_number(const char *json_str, const char *key, bool *found);

/*
 * ndb_json_extract_bool - Extract boolean value by key
 * Returns boolean value, sets found flag
 */
extern bool ndb_json_extract_bool(const char *json_str, const char *key, bool *found);

/*
 * ndb_json_extract_int - Extract integer value by key
 * Returns integer value, sets found flag
 */
extern int	ndb_json_extract_int(const char *json_str, const char *key, bool *found);

/*
 * ndb_json_extract_float - Extract float value by key
 * Returns float value, sets found flag
 */
extern float ndb_json_extract_float(const char *json_str, const char *key, bool *found);

/* ========== Specialized JSON Parsers ========== */

/*
 * Generation parameters structure (used by LLM functions)
 */
typedef struct NdbGenParams
{
	float		temperature;
	float		top_p;
	int			top_k;
	int			max_tokens;
	int			min_tokens;
	float		repetition_penalty;
	bool		do_sample;
	bool		return_prompt;
	int			seed;
	bool		streaming;
	int			num_stop_sequences;
	char	  **stop_sequences;
	int			num_logit_bias;
	int32	   *logit_bias_tokens;
	float	   *logit_bias_values;
} NdbGenParams;

/*
 * ndb_json_parse_gen_params - Parse generation parameters from JSON
 * Unified parser for LLM generation parameters
 * Returns 0 on success, -1 on error (sets errstr)
 */
extern int	ndb_json_parse_gen_params(const char *params_json,
									  NdbGenParams *gen_params,
									  char **errstr);

/*
 * ndb_json_parse_gen_params_free - Free allocated resources in NdbGenParams
 */
extern void ndb_json_parse_gen_params_free(NdbGenParams *gen_params);

/*
 * OpenAI response extraction structure
 */
typedef struct NdbOpenAIResponse
{
	char	   *text;
	int			tokens_in;
	int			tokens_out;
	char	   *error_message;
} NdbOpenAIResponse;

/*
 * ndb_json_extract_openai_response - Extract OpenAI API response
 * Parses OpenAI chat completion or embedding response
 * Returns 0 on success, -1 on error
 */
extern int	ndb_json_extract_openai_response(const char *json_str,
											 NdbOpenAIResponse *response);

/*
 * ndb_json_extract_openai_response_free - Free OpenAI response structure
 */
extern void ndb_json_extract_openai_response_free(NdbOpenAIResponse *response);

/*
 * ndb_json_parse_openai_embedding - Parse OpenAI embedding vector from JSON
 * Extracts embedding array from OpenAI embedding API response
 * Returns 0 on success, -1 on error
 */
extern int	ndb_json_parse_openai_embedding(const char *json_str,
											float **vec_out,
											int *dim_out);

/*
 * Sparse vector parsing structure
 */
typedef struct NdbSparseVectorParse
{
	int32		vocab_size;
	uint16		model_type;		/* 0=BM25, 1=SPLADE, 2=ColBERTv2 */
	int32		nnz;
	int32	   *token_ids;
	float4	   *weights;
} NdbSparseVectorParse;

/*
 * ndb_json_parse_sparse_vector - Parse sparse vector from JSON
 * Proper JSON parsing for sparse vector format
 * Returns 0 on success, -1 on error (sets errstr)
 */
extern int	ndb_json_parse_sparse_vector(const char *json_str,
										 NdbSparseVectorParse *result,
										 char **errstr);

/*
 * ndb_json_parse_sparse_vector_free - Free sparse vector parse structure
 */
extern void ndb_json_parse_sparse_vector_free(NdbSparseVectorParse *result);

/* ========== JSON Generation Utilities ========== */

/*
 * ndb_json_build_object - Build JSON object string from key-value pairs
 * Returns allocated JSON string
 */
extern char *ndb_json_build_object(const char *key1, const char *value1,...);

/*
 * ndb_json_build_object_buf - Build JSON object into StringInfo buffer
 * Appends JSON object to existing buffer
 */
extern void ndb_json_build_object_buf(StringInfo buf, const char *key1, const char *value1,...);

/*
 * ndb_json_build_array - Build JSON array string from values
 * Returns allocated JSON string
 */
extern char *ndb_json_build_array(const char *value1,...);

/*
 * ndb_json_build_array_buf - Build JSON array into StringInfo buffer
 * Appends JSON array to existing buffer
 */
extern void ndb_json_build_array_buf(StringInfo buf, const char *value1,...);

/*
 * ndb_json_merge_objects - Merge two JSON objects
 * Returns new JSON string with merged content
 */
extern char *ndb_json_merge_objects(const char *json1, const char *json2);

/* ========== JSON Array Utilities ========== */

/*
 * ndb_json_parse_array - Parse JSON array into string array
 * Returns array of strings and count
 */
extern char **ndb_json_parse_array(const char *json_str, int *count);

/*
 * ndb_json_parse_array_free - Free array parsed by ndb_json_parse_array
 */
extern void ndb_json_parse_array_free(char **array, int count);

/*
 * ndb_json_parse_float_array - Parse JSON array of floats
 * Returns array of floats and count
 */
extern float *ndb_json_parse_float_array(const char *json_str, int *count);

/*
 * ndb_json_parse_int_array - Parse JSON array of integers
 * Returns array of integers and count
 */
extern int *ndb_json_parse_int_array(const char *json_str, int *count);

/* ========== Utility Functions ========== */

/*
 * ndb_json_validate - Validate JSON string syntax
 * Returns true if valid JSON, false otherwise
 */
extern bool ndb_json_validate(const char *json_str);

/*
 * ndb_json_is_empty - Check if JSON string is empty object/array
 * Returns true if empty, false otherwise
 */
extern bool ndb_json_is_empty(const char *json_str);

/*
 * ndb_json_strip_whitespace - Remove unnecessary whitespace from JSON
 * Returns new JSON string with minimal whitespace
 */
extern char *ndb_json_strip_whitespace(const char *json_str);

/* ========== JSON Operation Helper Macros (using constants) ========== */

/*
 * NDB_JSON_GET_FIELD - Safe JSON field extraction using constants
 * Usage: Jsonb *field = NDB_JSON_GET_FIELD(jsonb, NDB_JSON_KEY_STORAGE);
 */
#define NDB_JSON_GET_FIELD(jsonb, key_const) \
	ndb_jsonb_object_field((jsonb), (key_const))

/*
 * NDB_JSON_GET_STRING - Extract string value from JSON field using constants
 * Usage: char *value = NDB_JSON_GET_STRING(jsonb, NDB_JSON_KEY_TEXT);
 */
#define NDB_JSON_GET_STRING(jsonb, key_const) \
	({ \
		Jsonb *_j = ndb_jsonb_object_field((jsonb), (key_const)); \
		_j ? ndb_jsonb_out_cstring(_j) : NULL; \
	})

/*
 * NDB_JSON_HAS_FIELD - Check if JSON field exists using constants
 * Usage: if (NDB_JSON_HAS_FIELD(jsonb, NDB_JSON_KEY_METRICS)) { ... }
 */
#define NDB_JSON_HAS_FIELD(jsonb, key_const) \
	(ndb_jsonb_object_field((jsonb), (key_const)) != NULL)

#endif							/* NEURONDB_JSON_H */
