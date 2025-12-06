/*-------------------------------------------------------------------------
 *
 * neurondb_json.c
 *    Centralized JSON handling utilities for NeuronDB
 *
 * Provides unified JSON parsing, extraction, quoting, and generation
 * functions with DirectFunctionCall wrappers for PostgreSQL's jsonb
 * functions. Consolidates all JSON handling logic in one place.
 *
 * Uses PostgreSQL's common/jsonapi.h for robust JSON parsing and
 * utils/jsonb.h for JSONB operations. All functions follow 100%
 * PostgreSQL C coding standards with proper error handling.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/util/neurondb_json.c
 *
 *-------------------------------------------------------------------------
 */

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
#include "utils/numeric.h"
#include "nodes/makefuncs.h"
#include "nodes/pg_list.h"

#include "neurondb_json.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>

static Oid jsonb_object_field_oid = InvalidOid;
static Oid jsonb_array_element_oid = InvalidOid;
static Oid jsonb_extract_path_oid = InvalidOid;
static Oid jsonb_extract_path_text_oid = InvalidOid;
static Oid jsonb_typeof_oid = InvalidOid;

static FmgrInfo jsonb_object_field_flinfo;
static FmgrInfo jsonb_array_element_flinfo;
static FmgrInfo jsonb_extract_path_flinfo;
static FmgrInfo jsonb_extract_path_text_flinfo;
static FmgrInfo jsonb_typeof_flinfo;

static bool jsonb_oids_initialized = false;

static void
ndb_jsonb_init_oids(void)
{
	List	   *funcname;
	Oid			argtypes[2];

	if (jsonb_oids_initialized)
		return;

	funcname = list_make1(makeString("jsonb_object_field"));
	argtypes[0] = JSONBOID;
	argtypes[1] = TEXTOID;
	jsonb_object_field_oid = LookupFuncName(funcname, 2, argtypes, false);
	if (!OidIsValid(jsonb_object_field_oid))
		elog(ERROR, "neurondb: jsonb_object_field function not found");
	fmgr_info_cxt(jsonb_object_field_oid, &jsonb_object_field_flinfo, TopMemoryContext);
	list_free(funcname);

	funcname = list_make1(makeString("jsonb_array_element"));
	argtypes[0] = JSONBOID;
	argtypes[1] = INT4OID;
	jsonb_array_element_oid = LookupFuncName(funcname, 2, argtypes, false);
	if (!OidIsValid(jsonb_array_element_oid))
		elog(ERROR, "neurondb: jsonb_array_element function not found");
	fmgr_info_cxt(jsonb_array_element_oid, &jsonb_array_element_flinfo, TopMemoryContext);
	list_free(funcname);

	funcname = list_make1(makeString("jsonb_extract_path"));
	argtypes[0] = JSONBOID;
	argtypes[1] = TEXTARRAYOID;
	jsonb_extract_path_oid = LookupFuncName(funcname, 2, argtypes, false);
	if (!OidIsValid(jsonb_extract_path_oid))
		elog(ERROR, "neurondb: jsonb_extract_path function not found");
	fmgr_info_cxt(jsonb_extract_path_oid, &jsonb_extract_path_flinfo, TopMemoryContext);
	list_free(funcname);

	funcname = list_make1(makeString("jsonb_extract_path_text"));
	argtypes[0] = JSONBOID;
	argtypes[1] = TEXTARRAYOID;
	jsonb_extract_path_text_oid = LookupFuncName(funcname, 2, argtypes, false);
	if (!OidIsValid(jsonb_extract_path_text_oid))
		elog(ERROR, "neurondb: jsonb_extract_path_text function not found");
	fmgr_info_cxt(jsonb_extract_path_text_oid, &jsonb_extract_path_text_flinfo, TopMemoryContext);
	list_free(funcname);

	funcname = list_make1(makeString("jsonb_typeof"));
	argtypes[0] = JSONBOID;
	jsonb_typeof_oid = LookupFuncName(funcname, 1, argtypes, false);
	if (!OidIsValid(jsonb_typeof_oid))
		elog(ERROR, "neurondb: jsonb_typeof function not found");
	fmgr_info_cxt(jsonb_typeof_oid, &jsonb_typeof_flinfo, TopMemoryContext);
	list_free(funcname);

	jsonb_oids_initialized = true;
}

Jsonb *
ndb_jsonb_in(text * json_text)
{
	char	   *cstr;
	Datum		result_datum;

	NDB_DECLARE(Jsonb *, result);

	if (json_text == NULL)
		return NULL;

	if (!jsonb_oids_initialized)
		ndb_jsonb_init_oids();

	cstr = text_to_cstring(json_text);

	PG_TRY();
	{
		result_datum = DirectFunctionCall1(jsonb_in, CStringGetDatum(cstr));
		result = DatumGetJsonbP(result_datum);
	}
	PG_CATCH();
	{
		FlushErrorState();
		NDB_FREE(cstr);
		return NULL;
	}
	PG_END_TRY();

	NDB_FREE(cstr);
	return result;
}

Jsonb *
ndb_jsonb_in_cstring(const char *json_str)
{
	text	   *json_text;

	NDB_DECLARE(Jsonb *, result);

	if (json_str == NULL)
		return NULL;

	json_text = cstring_to_text(json_str);
	result = ndb_jsonb_in(json_text);
	NDB_FREE(json_text);

	return result;
}

/*
 * ndb_jsonb_out - Convert JSONB to text
 *
 * Converts a JSONB value to a PostgreSQL text value using DirectFunctionCall1
 * to invoke the built-in jsonb_out function. The jsonb_out function returns
 * a cstring which is then converted to a text value.
 *
 * Parameters:
 *   jsonb - Input JSONB value to convert
 *
 * Returns:
 *   Text pointer containing JSON string representation, NULL on error or
 *   if input is NULL
 *
 * Notes:
 *   Memory for the returned text is allocated in CurrentMemoryContext.
 */
text *
ndb_jsonb_out(Jsonb * jsonb)
{
	Datum		jsonb_datum;
	Datum		result_datum;
	char	   *cstr;

	NDB_DECLARE(text *, result);

	if (jsonb == NULL)
		return NULL;

	jsonb_datum = PointerGetDatum(jsonb);

	PG_TRY();
	{
		result_datum = DirectFunctionCall1(jsonb_out, jsonb_datum);
		cstr = DatumGetCString(result_datum);
		result = cstring_to_text(cstr);
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
 * ndb_jsonb_out_cstring - Convert JSONB to C string
 */
char *
ndb_jsonb_out_cstring(Jsonb * jsonb)
{
	text	   *json_text;

	NDB_DECLARE(char *, result);

	if (jsonb == NULL)
		return NULL;

	json_text = ndb_jsonb_out(jsonb);
	if (json_text != NULL)
	{
		result = text_to_cstring(json_text);
		NDB_FREE(json_text);
	}

	return result;
}

/*
 * ndb_jsonb_object_field - Extract field from JSONB object
 */
Jsonb *
ndb_jsonb_object_field(Jsonb * jsonb, const char *field_name)
{
	Datum		jsonb_datum;
	Datum		text_datum;
	Datum		result_datum;
	text	   *field_text;

	NDB_DECLARE(Jsonb *, result);

	if (jsonb == NULL || field_name == NULL)
		return NULL;

	if (!jsonb_oids_initialized)
		ndb_jsonb_init_oids();

	jsonb_datum = PointerGetDatum(jsonb);
	field_text = cstring_to_text(field_name);
	text_datum = PointerGetDatum(field_text);

	PG_TRY();
	{
		result_datum = FunctionCall2(&jsonb_object_field_flinfo,
									 jsonb_datum, text_datum);
		if (!DatumGetPointer(result_datum))
			result = NULL;
		else
		{
			result = DatumGetJsonbP(result_datum);
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		result = NULL;
	}
	PG_END_TRY();

	NDB_FREE(field_text);

	return result;
}

/*
 * ndb_jsonb_array_element - Extract element from JSONB array
 */
Jsonb *
ndb_jsonb_array_element(Jsonb * jsonb, int index)
{
	Datum		jsonb_datum;
	Datum		int_datum;
	Datum		result_datum;

	NDB_DECLARE(Jsonb *, result);

	if (jsonb == NULL || index < 0)
		return NULL;

	if (!jsonb_oids_initialized)
		ndb_jsonb_init_oids();

	jsonb_datum = PointerGetDatum(jsonb);
	int_datum = Int32GetDatum(index);

	PG_TRY();
	{
		result_datum = FunctionCall2(&jsonb_array_element_flinfo,
									 jsonb_datum, int_datum);
		if (!DatumGetPointer(result_datum))
			result = NULL;
		else
		{
			result = DatumGetJsonbP(result_datum);
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
 * ndb_jsonb_extract_path - Extract value by path
 */
Jsonb *
ndb_jsonb_extract_path(Jsonb * jsonb, const char **path, int path_len)
 *   from the path components. The function initializes OID caches if needed.
 */
Jsonb *
ndb_jsonb_extract_path(Jsonb * jsonb, const char **path, int path_len)
{
	Datum		jsonb_datum;
	Datum		array_datum;
	Datum		result_datum;
	ArrayType  *path_array;
	int			i;

	NDB_DECLARE(Jsonb *, result);
	NDB_DECLARE(text * *, path_texts);

	if (jsonb == NULL || path == NULL || path_len <= 0)
		return NULL;

	/* Initialize OIDs if needed */
	if (!jsonb_oids_initialized)
		ndb_jsonb_init_oids();

	/* Build text array from path strings */
	NDB_ALLOC(path_texts, text *, path_len);
	for (i = 0; i < path_len; i++)
	{
		path_texts[i] = cstring_to_text(path[i]);
	}

	path_array = construct_array((Datum *) path_texts, path_len, TEXTOID,
								 -1, false, 'i');

	/* Free individual text elements */
	for (i = 0; i < path_len; i++)
	{
		NDB_FREE(path_texts[i]);
	}
	NDB_FREE(path_texts);

	jsonb_datum = PointerGetDatum(jsonb);
	array_datum = PointerGetDatum(path_array);

	PG_TRY();
	{
		result_datum = FunctionCall2(&jsonb_extract_path_flinfo,
									 jsonb_datum, array_datum);
		if (!DatumGetPointer(result_datum))
		{
			result = NULL;
		}
		else
		{
			result = DatumGetJsonbP(result_datum);
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		result = NULL;
	}
	PG_END_TRY();

	NDB_FREE(path_array);

	return result;
}

/*
 * ndb_jsonb_extract_path_text - Extract text value by path
 */
text *
ndb_jsonb_extract_path_text(Jsonb * jsonb, const char **path, int path_len)
{
	Datum		jsonb_datum;
	Datum		array_datum;
	Datum		result_datum;
	ArrayType  *path_array;
	int			i;

	NDB_DECLARE(text *, result);
	NDB_DECLARE(text * *, path_texts);

	if (jsonb == NULL || path == NULL || path_len <= 0)
		return NULL;

	/* Initialize OIDs if needed */
	if (!jsonb_oids_initialized)
		ndb_jsonb_init_oids();

	/* Build text array from path strings */
	NDB_ALLOC(path_texts, text *, path_len);
	for (i = 0; i < path_len; i++)
	{
		path_texts[i] = cstring_to_text(path[i]);
	}

	path_array = construct_array((Datum *) path_texts, path_len, TEXTOID,
								 -1, false, 'i');

	/* Free individual text elements */
	for (i = 0; i < path_len; i++)
	{
		NDB_FREE(path_texts[i]);
	}
	NDB_FREE(path_texts);

	jsonb_datum = PointerGetDatum(jsonb);
	array_datum = PointerGetDatum(path_array);

	PG_TRY();
	{
		result_datum = FunctionCall2(&jsonb_extract_path_text_flinfo,
									 jsonb_datum, array_datum);
		if (!DatumGetPointer(result_datum))
		{
			result = NULL;
		}
		else
		{
			result = DatumGetTextP(result_datum);
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		result = NULL;
	}
	PG_END_TRY();

	NDB_FREE(path_array);

	return result;
}

/*
 * ndb_jsonb_extract_path_cstring - Extract C string value by path
 */
char *
ndb_jsonb_extract_path_cstring(Jsonb * jsonb, const char **path, int path_len)
{
	text	   *path_text;

	NDB_DECLARE(char *, result);

	if (jsonb == NULL || path == NULL || path_len <= 0)
		return NULL;

	path_text = ndb_jsonb_extract_path_text(jsonb, path, path_len);
	if (path_text != NULL)
	{
		result = text_to_cstring(path_text);
		NDB_FREE(path_text);
	}

	return result;
}

/*
 * ndb_jsonb_typeof - Get JSONB type
 *
 * Returns the type of a JSONB value as a text string. Possible return values
 * are "object", "array", "string", "number", "boolean", or "null". Uses cached
 * function OID and FmgrInfo for efficient function calls.
 *
 * Parameters:
 *   jsonb - Input JSONB value
 *
 * Returns:
 *   Text pointer containing the type name, NULL if input is NULL or on error
 *
 * Notes:
 *   Memory for the returned text is allocated in CurrentMemoryContext.
 *   The function initializes OID caches if needed.
 */
text *
ndb_jsonb_typeof(Jsonb * jsonb)
{
	Datum		jsonb_datum;
	Datum		result_datum;

	NDB_DECLARE(text *, result);

	if (jsonb == NULL)
		return NULL;

	if (!jsonb_oids_initialized)
		ndb_jsonb_init_oids();

	jsonb_datum = PointerGetDatum(jsonb);

	PG_TRY();
	{
		result_datum = FunctionCall1(&jsonb_typeof_flinfo, jsonb_datum);
		result = DatumGetTextP(result_datum);
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
 * ndb_jsonb_typeof_cstring - Get JSONB type as C string
 */
char *
ndb_jsonb_typeof_cstring(Jsonb * jsonb)
{
	text	   *type_text;

	NDB_DECLARE(char *, result);

	if (jsonb == NULL)
		return NULL;

	type_text = ndb_jsonb_typeof(jsonb);
	if (type_text != NULL)
	{
		result = text_to_cstring(type_text);
		NDB_FREE(type_text);
	}

	return result;
}

/*
 * ndb_jsonb_to_text - Convert JSONB to text
 *
 * Alias for ndb_jsonb_out that converts a JSONB value to a text representation.
 * Provided for API consistency and clarity.
 *
 * Parameters:
 *   jsonb - Input JSONB value to convert
 *
 * Returns:
 *   Text pointer containing JSON string representation, NULL on error or
 *   if input is NULL
 */
text *
ndb_jsonb_to_text(Jsonb * jsonb)
{
	return ndb_jsonb_out(jsonb);
}

/*
 * ndb_json_quote_string - Quote and escape a C string for JSON
 *
 * Quotes and escapes a C string according to JSON string encoding rules.
 * Handles all standard JSON escape sequences including control characters,
 * quotes, backslashes, and Unicode escapes.
 *
 * Parameters:
 *   str - Input C string to quote and escape
 *
 * Returns:
 *   Newly allocated C string containing the quoted and escaped JSON string.
 *   Returns "null" (without quotes) if input is NULL.
 *
 * Notes:
 *   Memory for the returned string is allocated in CurrentMemoryContext.
 *   Caller is responsible for freeing using pfree or NDB_FREE.
 *   Control characters (0x00-0x1F) are escaped as \uXXXX sequences.
 */
char *
ndb_json_quote_string(const char *str)
{
	StringInfoData buf;
	const char *p;
	char	   *result;

	if (str == NULL)
		return pstrdup("null");

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '"');

	for (p = str; *p; p++)
	{
		switch (*p)
		{
			case '"':
				appendStringInfoString(&buf, "\\\"");
				break;
			case '\\':
				appendStringInfoString(&buf, "\\\\");
				break;
			case '\b':
				appendStringInfoString(&buf, "\\b");
				break;
			case '\f':
				appendStringInfoString(&buf, "\\f");
				break;
			case '\n':
				appendStringInfoString(&buf, "\\n");
				break;
			case '\r':
				appendStringInfoString(&buf, "\\r");
				break;
			case '\t':
				appendStringInfoString(&buf, "\\t");
				break;
			default:
				if ((unsigned char) *p < 0x20)
				{
					appendStringInfo(&buf, "\\u%04x", (unsigned char) *p);
				}
				else
				{
					appendStringInfoChar(&buf, *p);
				}
				break;
		}
	}

	appendStringInfoChar(&buf, '"');
	result = pstrdup(buf.data);
	NDB_FREE(buf.data);
	return result;
}

/*
 * ndb_json_quote_string_buf - Quote and escape into StringInfo buffer
 */
void
ndb_json_quote_string_buf(StringInfo buf, const char *str)
{
	const char *p;

	if (buf == NULL)
		return;

	if (str == NULL)
	{
		appendStringInfoString(buf, "null");
		return;
	}

	appendStringInfoChar(buf, '"');

	for (p = str; *p; p++)
	{
		switch (*p)
		{
			case '"':
				appendStringInfoString(buf, "\\\"");
				break;
			case '\\':
				appendStringInfoString(buf, "\\\\");
				break;
			case '\b':
				appendStringInfoString(buf, "\\b");
				break;
			case '\f':
				appendStringInfoString(buf, "\\f");
				break;
			case '\n':
				appendStringInfoString(buf, "\\n");
				break;
			case '\r':
				appendStringInfoString(buf, "\\r");
				break;
			case '\t':
				appendStringInfoString(buf, "\\t");
				break;
			default:
				if ((unsigned char) *p < 0x20)
				{
					appendStringInfo(buf, "\\u%04x", (unsigned char) *p);
				}
				else
				{
					appendStringInfoChar(buf, *p);
				}
				break;
		}
	}

	appendStringInfoChar(buf, '"');
}

/*
 * ndb_json_unescape_string - Unescape a JSON string
 *
 * Converts an escaped JSON string back to a normal C string by processing
 * all JSON escape sequences including \n, \t, \r, \\, \/, \", and \uXXXX
 * Unicode escapes. Handles surrogate pairs for proper UTF-8 encoding.
 *
 * Parameters:
 *   json_str - Input JSON string (may include opening/closing quotes)
 *
 * Returns:
 *   Newly allocated C string containing the unescaped result, NULL if input
 *   is NULL or empty
 *
 * Notes:
 *   Memory for the returned string is allocated in CurrentMemoryContext.
 *   Caller is responsible for freeing using pfree or NDB_FREE.
 *   The function automatically skips opening and closing quotes if present.
 *   Invalid Unicode escapes are replaced with the replacement character (U+FFFD).
 */
char *
ndb_json_unescape_string(const char *json_str)
{
	const char *p;
	const char *q;
	size_t		len;

	NDB_DECLARE(char *, result);
	NDB_DECLARE(char *, unescaped);
	int			escape_next = 0;

	if (!json_str || json_str[0] == '\0')
		return NULL;

	p = json_str;

	if (*p == '"')
		p++;

	q = p;
	len = 0;
	while (*q)
	{
		if (escape_next)
		{
			escape_next = 0;
			len++;
			q++;
			continue;
		}
		if (*q == '\\')
		{
			escape_next = 1;
			len++;
			q++;
			continue;
		}
		if (*q == '"')
			break;
		len++;
		q++;
	}

	if (len == 0)
		return pstrdup("");

	NDB_ALLOC(result, char, len + 1);
	unescaped = result;
	q = p;

	while (q < p + len)
	{
		if (*q == '\\' && q + 1 < p + len)
		{
			switch (q[1])
			{
				case 'n':
					*unescaped++ = '\n';
					q += 2;
					break;
				case 't':
					*unescaped++ = '\t';
					q += 2;
					break;
				case 'r':
					*unescaped++ = '\r';
					q += 2;
					break;
				case '\\':
					*unescaped++ = '\\';
					q += 2;
					break;
				case '"':
					*unescaped++ = '"';
					q += 2;
					break;
				case '/':
					*unescaped++ = '/';
					q += 2;
					break;
				case 'u':
					if (q + 5 < p + len && isxdigit((unsigned char) q[2]) &&
						isxdigit((unsigned char) q[3]) &&
						isxdigit((unsigned char) q[4]) &&
						isxdigit((unsigned char) q[5]))
					{
						unsigned int code = 0;

						sscanf(q + 2, "%4x", &code);
						if (code < 0x80)
							*unescaped++ = (char) code;
						else if (code < 0x800)
						{
							*unescaped++ = (char) (0xC0 | (code >> 6));
							*unescaped++ = (char) (0x80 | (code & 0x3F));
						}
						else if (code < 0xD800 || code >= 0xE000)
						{
							*unescaped++ = (char) (0xE0 | (code >> 12));
							*unescaped++ = (char) (0x80 | ((code >> 6) & 0x3F));
							*unescaped++ = (char) (0x80 | (code & 0x3F));
						}
						else
						{
							if (code >= 0xD800 && code < 0xDC00 && q + 11 < p + len &&
								q[6] == '\\' && q[7] == 'u' &&
								isxdigit((unsigned char) q[8]) &&
								isxdigit((unsigned char) q[9]) &&
								isxdigit((unsigned char) q[10]) &&
								isxdigit((unsigned char) q[11]))
							{
								unsigned int code2 = 0;

								sscanf(q + 8, "%4x", &code2);
								if (code2 >= 0xDC00 && code2 < 0xE000)
								{
									unsigned int full_code = 0x10000 + ((code - 0xD800) << 10) + (code2 - 0xDC00);

									*unescaped++ = (char) (0xF0 | (full_code >> 18));
									*unescaped++ = (char) (0x80 | ((full_code >> 12) & 0x3F));
									*unescaped++ = (char) (0x80 | ((full_code >> 6) & 0x3F));
									*unescaped++ = (char) (0x80 | (full_code & 0x3F));
									q += 6;
								}
								else
								{
									*unescaped++ = 0xEF;
									*unescaped++ = 0xBF;
									*unescaped++ = 0xBD;
								}
							}
							else
							{
								*unescaped++ = 0xEF;
								*unescaped++ = 0xBF;
								*unescaped++ = 0xBD;
							}
						}
						q += 6;
					}
					else
					{
						*unescaped++ = *q++;
					}
					break;
				default:
					*unescaped++ = *q++;
					*unescaped++ = *q++;
					break;
			}
		}
		else
		{
			*unescaped++ = *q++;
		}
	}
	*unescaped = '\0';

	return result;
}

/*
 * ndb_json_find_key - Find value for a key in JSON object
 *
 * Searches for a key in a JSON object and returns its value as a C string.
 * First attempts to parse the JSON using JSONB for robust handling, then
 * falls back to simple string-based search if JSONB parsing fails.
 *
 * Parameters:
 *   json_str - Input JSON string
 *   key - Key name to search for
 *
 * Returns:
 *   Newly allocated C string containing the value, NULL if key not found,
 *   input is NULL, or on error
 *
 * Notes:
 *   Memory for the returned string is allocated in CurrentMemoryContext.
 *   Caller is responsible for freeing using pfree or NDB_FREE.
 *   The fallback string search is limited and may not handle all edge cases.
 */
char *
ndb_json_find_key(const char *json_str, const char *key)
{
	volatile	Jsonb *jsonb = NULL;

	NDB_DECLARE(Jsonb *, field);
	volatile char *result = NULL;

	NDB_DECLARE(text *, field_text);

	if (json_str == NULL || key == NULL)
		return NULL;

	PG_TRY();
	{
		jsonb = ndb_jsonb_in_cstring(json_str);
		if (jsonb != NULL)
		{
			field = ndb_jsonb_object_field((Jsonb *) jsonb, key);
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					result = text_to_cstring(field_text);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}
			if (jsonb != NULL)
			{
				Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

				NDB_FREE(jsonb_ptr);
				jsonb = NULL;
			}
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
	}
	PG_END_TRY();

	if (result == NULL)
	{
		const char *p;
		const char *value_start;
		const char *value_end;
		size_t		key_len;

		NDB_DECLARE(char *, key_pattern);

		key_len = strlen(key);
		NDB_ALLOC(key_pattern, char, key_len + 4);
		snprintf(key_pattern, key_len + 4, "\"%s\":", key);

		p = strstr(json_str, key_pattern);
		if (p != NULL)
		{
			p += key_len + 2;
			while (*p && (isspace((unsigned char) *p) || *p == ':'))
				p++;

			if (*p == '"')
			{
				value_start = p + 1;
				value_end = value_start;
				while (*value_end && *value_end != '"')
				{
					if (*value_end == '\\' && value_end[1])
						value_end += 2;
					else
						value_end++;
				}
				if (*value_end == '"')
				{
					char	   *temp_result = pnstrdup(value_start, value_end - value_start);

					if (strchr(temp_result, '\\') != NULL)
					{
						char	   *unescaped = ndb_json_unescape_string(temp_result);

						NDB_FREE(temp_result);
						result = (volatile char *) unescaped;
					}
					else
					{
						result = (volatile char *) temp_result;
					}
				}
			}
			else
			{
				value_start = p;
				value_end = value_start;
				while (*value_end && *value_end != ',' && *value_end != '}' && *value_end != ']')
				{
					if (*value_end == '"')
					{
						value_end++;
						while (*value_end && *value_end != '"')
						{
							if (*value_end == '\\' && value_end[1])
								value_end += 2;
							else
								value_end++;
						}
						if (*value_end == '"')
							value_end++;
					}
					else
						value_end++;
				}
				while (value_end > value_start && isspace((unsigned char) value_end[-1]))
					value_end--;
				result = (volatile char *) pnstrdup(value_start, value_end - value_start);
			}
		}
		NDB_FREE(key_pattern);
	}

	return (char *) result;
}

/*
 * ndb_json_extract_string - Extract string value by key
 *
 * Alias for ndb_json_find_key that extracts a string value from a JSON object
 * by key name. Provided for API clarity and consistency.
 *
 * Parameters:
 *   json_str - Input JSON string
 *   key - Key name to extract
 *
 * Returns:
 *   Newly allocated C string containing the value, NULL if key not found,
 *   input is NULL, or on error
 */
char *
ndb_json_extract_string(const char *json_str, const char *key)
{
	return ndb_json_find_key(json_str, key);
}

/*
 * ndb_json_extract_number - Extract numeric value by key
 *
 * Extracts a numeric value from a JSON object by key name and converts it
 * to a double. Uses strtod for parsing with proper error checking.
 *
 * Parameters:
 *   json_str - Input JSON string
 *   key - Key name to extract
 *   found - Optional pointer to boolean flag set to true if key was found
 *           and value was successfully parsed
 *
 * Returns:
 *   Numeric value as double, 0.0 if key not found, parsing failed, or
 *   input is NULL
 *
 * Notes:
 *   If found is not NULL, it is set to true only if the key exists and
 *   the value is a valid number. The function handles all standard numeric
 *   formats including integers, floats, and scientific notation.
 */
double
ndb_json_extract_number(const char *json_str, const char *key, bool *found)
{
	char	   *value_str;
	double		result = 0.0;
	char	   *endptr;

	if (found != NULL)
		*found = false;

	if (json_str == NULL || key == NULL)
		return 0.0;

	value_str = ndb_json_find_key(json_str, key);
	if (value_str == NULL)
		return 0.0;

	errno = 0;
	result = strtod(value_str, &endptr);
	if (endptr != value_str && *endptr == '\0' && errno == 0)
	{
		if (found != NULL)
			*found = true;
	}

	NDB_FREE(value_str);
	return result;
}

/*
 * ndb_json_extract_bool - Extract boolean value by key
 *
 * Extracts a boolean value from a JSON object by key name. Recognizes
 * "true", "false", and their case variations (TRUE, True, FALSE, False).
 *
 * Parameters:
 *   json_str - Input JSON string
 *   key - Key name to extract
 *   found - Optional pointer to boolean flag set to true if key was found
 *           and value was a valid boolean
 *
 * Returns:
 *   Boolean value, false if key not found, value is not a boolean,
 *   or input is NULL
 *
 * Notes:
 *   If found is not NULL, it is set to true only if the key exists and
 *   the value is a recognized boolean string.
 */
bool
ndb_json_extract_bool(const char *json_str, const char *key, bool *found)
{
	char	   *value_str;
	bool		result = false;

	if (found != NULL)
		*found = false;

	if (json_str == NULL || key == NULL)
		return false;

	value_str = ndb_json_find_key(json_str, key);
	if (value_str == NULL)
		return false;

	if (strcmp(value_str, "true") == 0 || strcmp(value_str, "TRUE") == 0 ||
		strcmp(value_str, "True") == 0)
	{
		result = true;
		if (found != NULL)
			*found = true;
	}
	else if (strcmp(value_str, "false") == 0 || strcmp(value_str, "FALSE") == 0 ||
			 strcmp(value_str, "False") == 0)
	{
		result = false;
		if (found != NULL)
			*found = true;
	}

	NDB_FREE(value_str);
	return result;
}

/*
 * ndb_json_extract_int - Extract integer value by key
 *
 * Extracts a numeric value from a JSON object by key name and converts it
 * to an integer. Uses ndb_json_extract_number internally and casts the
 * result to int.
 *
 * Parameters:
 *   json_str - Input JSON string
 *   key - Key name to extract
 *   found - Optional pointer to boolean flag set to true if key was found
 *           and value was successfully parsed
 *
 * Returns:
 *   Integer value, 0 if key not found, parsing failed, or input is NULL
 *
 * Notes:
 *   Floating point values are truncated to integers. If found is not NULL,
 *   it is set based on whether the numeric extraction succeeded.
 */
int
ndb_json_extract_int(const char *json_str, const char *key, bool *found)
{
	double		num_value;
	int			result;

	if (found != NULL)
		*found = false;

	num_value = ndb_json_extract_number(json_str, key, found);
	result = (int) num_value;

	return result;
}

/*
 * ndb_json_extract_float - Extract float value by key
 *
 * Extracts a numeric value from a JSON object by key name and converts it
 * to a float. Uses ndb_json_extract_number internally and casts the result
 * to float.
 *
 * Parameters:
 *   json_str - Input JSON string
 *   key - Key name to extract
 *   found - Optional pointer to boolean flag set to true if key was found
 *           and value was successfully parsed
 *
 * Returns:
 *   Float value, 0.0 if key not found, parsing failed, or input is NULL
 *
 * Notes:
 *   Precision may be lost when converting from double to float. If found is
 *   not NULL, it is set based on whether the numeric extraction succeeded.
 */
float
ndb_json_extract_float(const char *json_str, const char *key, bool *found)
{
	double		num_value;
	float		result;

	if (found != NULL)
		*found = false;

	num_value = ndb_json_extract_number(json_str, key, found);
	result = (float) num_value;

	return result;
}

/*
 * ndb_json_parse_gen_params - Parse generation parameters from JSON
 */
int
ndb_json_parse_gen_params(const char *params_json,
						  NdbGenParams *gen_params,
						  char **errstr)
{
	NDB_DECLARE(char *, json_copy);
	NDB_DECLARE(char *, p);
	NDB_DECLARE(char *, key);
	char	   *endptr = NULL;
	float		float_val;
	int			int_val;
	volatile	Jsonb *jsonb = NULL;

	if (errstr)
		*errstr = NULL;
	if (!params_json || !gen_params)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for parse_gen_params");
		return -1;
	}

	memset(gen_params, 0, sizeof(NdbGenParams));
	gen_params->temperature = 1.0f;
	gen_params->top_p = 1.0f;
	gen_params->top_k = 0;		/* 0 = disabled */
	gen_params->max_tokens = 100;
	gen_params->min_tokens = 0;
	gen_params->repetition_penalty = 1.0f;
	gen_params->do_sample = false;
	gen_params->return_prompt = false;
	gen_params->seed = 0;
	gen_params->streaming = false;
	gen_params->num_stop_sequences = 0;
	gen_params->stop_sequences = NULL;
	gen_params->num_logit_bias = 0;
	gen_params->logit_bias_tokens = NULL;
	gen_params->logit_bias_values = NULL;

	if (strlen(params_json) == 0 || strcmp(params_json, "{}") == 0)
		return 0;

	PG_TRY();
	{
		jsonb = ndb_jsonb_in_cstring(params_json);

		if (jsonb != NULL)
		{
			Jsonb	   *field;
			text	   *field_text;
			char	   *value_str;

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "temperature");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					float_val = strtof(value_str, &endptr);
					if (endptr != value_str && float_val > 0.0f)
						gen_params->temperature = float_val;
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "top_p");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					float_val = strtof(value_str, &endptr);
					if (endptr != value_str && float_val > 0.0f && float_val <= 1.0f)
						gen_params->top_p = float_val;
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "top_k");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					int_val = (int) strtol(value_str, &endptr, 10);
					if (endptr != value_str && int_val >= 0)
						gen_params->top_k = int_val;
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "max_tokens");
			if (field == NULL)
				field = ndb_jsonb_object_field((Jsonb *) jsonb, "max_length");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					int_val = (int) strtol(value_str, &endptr, 10);
					if (endptr != value_str && int_val > 0)
						gen_params->max_tokens = int_val;
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "min_tokens");
			if (field == NULL)
				field = ndb_jsonb_object_field((Jsonb *) jsonb, "min_length");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					int_val = (int) strtol(value_str, &endptr, 10);
					if (endptr != value_str && int_val >= 0)
						gen_params->min_tokens = int_val;
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "repetition_penalty");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					float_val = strtof(value_str, &endptr);
					if (endptr != value_str && float_val > 0.0f)
						gen_params->repetition_penalty = float_val;
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "do_sample");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					if (strcmp(value_str, "true") == 0 || strcmp(value_str, "TRUE") == 0)
						gen_params->do_sample = true;
					else if (strcmp(value_str, "false") == 0 || strcmp(value_str, "FALSE") == 0)
						gen_params->do_sample = false;
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "return_prompt");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					if (strcmp(value_str, "true") == 0 || strcmp(value_str, "TRUE") == 0)
						gen_params->return_prompt = true;
					else if (strcmp(value_str, "false") == 0 || strcmp(value_str, "FALSE") == 0)
						gen_params->return_prompt = false;
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "seed");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					int_val = (int) strtol(value_str, &endptr, 10);
					if (endptr != value_str)
						gen_params->seed = int_val;
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "streaming");
			if (field == NULL)
				field = ndb_jsonb_object_field((Jsonb *) jsonb, "stream");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					if (strcmp(value_str, "true") == 0 || strcmp(value_str, "TRUE") == 0)
						gen_params->streaming = true;
					else if (strcmp(value_str, "false") == 0 || strcmp(value_str, "FALSE") == 0)
						gen_params->streaming = false;
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "stop_sequences");
			if (field != NULL)
			{
				char	  **stop_seqs = NULL;
				int			count = 0;

				NDB_DECLARE(char *, tmp);

				tmp = ndb_jsonb_out_cstring(field);
				if (tmp != NULL)
				{
					stop_seqs = ndb_json_parse_array(tmp, &count);
					NDB_FREE(tmp);
					if (stop_seqs != NULL && count > 0)
					{
						gen_params->stop_sequences = stop_seqs;
						gen_params->num_stop_sequences = count;
					}
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "logit_bias");
			if (field == NULL)
				field = ndb_jsonb_object_field((Jsonb *) jsonb, "bias");
			if (field != NULL)
			{
				JsonbIterator *it;
				JsonbValue	v;
				JsonbIteratorToken type;
				int			capacity = 16;
				int			count = 0;

				NDB_DECLARE(int32 *, tokens);
				NDB_DECLARE(float *, biases);

				NDB_ALLOC(tokens, int32, capacity);
				NDB_ALLOC(biases, float, capacity);

				it = JsonbIteratorInit(&field->root);
				while ((type = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
				{
					if (type == WJB_KEY && v.type == jbvString)
					{
						char	   *token_str = pnstrdup(v.val.string.val, v.val.string.len);
						int32		token_id = (int32) strtol(token_str, &endptr, 10);

						NDB_FREE(token_str);

						if (endptr != token_str && token_id >= 0)
						{
							type = JsonbIteratorNext(&it, &v, true);
							if (type == WJB_VALUE)
							{
								if (v.type == jbvNumeric)
								{
									Numeric		num = v.val.numeric;
									float		bias_val = DatumGetFloat4(DirectFunctionCall1(numeric_float4,
																							  NumericGetDatum(num)));

									if (count >= capacity)
									{
										capacity *= 2;
										tokens = repalloc(tokens, sizeof(int32) * capacity);
										biases = repalloc(biases, sizeof(float) * capacity);
									}

									tokens[count] = token_id;
									biases[count] = bias_val;
									count++;
								}
							}
						}
					}
				}

				if (count > 0)
				{
					gen_params->logit_bias_tokens = tokens;
					gen_params->logit_bias_values = biases;
					gen_params->num_logit_bias = count;
				}
				else
				{
					NDB_FREE(tokens);
					NDB_FREE(biases);
				}
				NDB_FREE(field);
			}
		}
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
	}
	PG_END_TRY();

	if (gen_params->num_stop_sequences == 0 && gen_params->num_logit_bias == 0)
	{
		json_copy = pstrdup(params_json);
		p = json_copy;

		while (*p && (isspace((unsigned char) *p) || *p == '{'))
			p++;

		while (*p && *p != '}')
		{
			while (*p && (isspace((unsigned char) *p) || *p == ','))
				p++;

			if (*p == '}' || *p == '\0')
				break;

			if (*p != '"')
			{
				NDB_FREE(json_copy);
				if (errstr)
					*errstr = pstrdup("invalid JSON format: expected key");
				return -1;
			}
			p++;
			key = p;
			while (*p && *p != '"')
				p++;
			if (*p != '"')
			{
				NDB_FREE(json_copy);
				if (errstr)
					*errstr = pstrdup("invalid JSON format: unterminated key");
				return -1;
			}
			*p = '\0';
			p++;

			while (*p && (isspace((unsigned char) *p) || *p == ':'))
				p++;

			if (strcmp(key, "temperature") == 0)
			{
				float_val = strtof(p, &endptr);
				if (endptr != p && float_val > 0.0f)
					gen_params->temperature = float_val;
			}
			else if (strcmp(key, "top_p") == 0)
			{
				float_val = strtof(p, &endptr);
				if (endptr != p && float_val > 0.0f && float_val <= 1.0f)
					gen_params->top_p = float_val;
			}
			else if (strcmp(key, "top_k") == 0)
			{
				int_val = (int) strtol(p, &endptr, 10);
				if (endptr != p && int_val >= 0)
					gen_params->top_k = int_val;
			}
			else if (strcmp(key, "max_tokens") == 0 || strcmp(key, "max_length") == 0)
			{
				int_val = (int) strtol(p, &endptr, 10);
				if (endptr != p && int_val > 0)
					gen_params->max_tokens = int_val;
			}
			else if (strcmp(key, "min_tokens") == 0 || strcmp(key, "min_length") == 0)
			{
				int_val = (int) strtol(p, &endptr, 10);
				if (endptr != p && int_val >= 0)
					gen_params->min_tokens = int_val;
			}
			else if (strcmp(key, "repetition_penalty") == 0)
			{
				float_val = strtof(p, &endptr);
				if (endptr != p && float_val > 0.0f)
					gen_params->repetition_penalty = float_val;
			}
			else if (strcmp(key, "do_sample") == 0)
			{
				if (strncmp(p, "true", 4) == 0 || strncmp(p, "TRUE", 4) == 0)
					gen_params->do_sample = true;
				else if (strncmp(p, "false", 5) == 0 || strncmp(p, "FALSE", 5) == 0)
					gen_params->do_sample = false;
			}
			else if (strcmp(key, "return_prompt") == 0)
			{
				if (strncmp(p, "true", 4) == 0 || strncmp(p, "TRUE", 4) == 0)
					gen_params->return_prompt = true;
				else if (strncmp(p, "false", 5) == 0 || strncmp(p, "FALSE", 5) == 0)
					gen_params->return_prompt = false;
			}
			else if (strcmp(key, "seed") == 0)
			{
				int_val = (int) strtol(p, &endptr, 10);
				if (endptr != p)
					gen_params->seed = int_val;
			}
			else if (strcmp(key, "streaming") == 0 || strcmp(key, "stream") == 0)
			{
				if (strncmp(p, "true", 4) == 0 || strncmp(p, "TRUE", 4) == 0)
					gen_params->streaming = true;
				else if (strncmp(p, "false", 5) == 0 || strncmp(p, "FALSE", 5) == 0)
					gen_params->streaming = false;
			}

			while (*p && *p != ',' && *p != '}')
			{
				if (*p == '"')
				{
					p++;
					while (*p && *p != '"')
						p++;
					if (*p == '"')
						p++;
				}
				else if (*p == '[')
				{
					int			depth = 1;

					p++;
					while (*p && depth > 0)
					{
						if (*p == '[')
							depth++;
						else if (*p == ']')
							depth--;
						p++;
					}
				}
				else if (*p == '{')
				{
					int			depth = 1;

					p++;
					while (*p && depth > 0)
					{
						if (*p == '{')
							depth++;
						else if (*p == '}')
							depth--;
						p++;
					}
				}
				else
					p++;
			}
		}

		NDB_FREE(json_copy);
	}

	return 0;
}

/*
 * ndb_json_parse_gen_params_free - Free allocated resources in NdbGenParams
 *
 * Frees all dynamically allocated memory in an NdbGenParams structure,
 * including stop_sequences array and logit_bias arrays. Resets counts to zero.
 *
 * Parameters:
 *   gen_params - Structure to free resources from
 *
 * Notes:
 *   Safe to call with NULL pointer. After calling this function, the structure
 *   should not be used unless re-initialized.
 */
void
ndb_json_parse_gen_params_free(NdbGenParams *gen_params)
{
	int			i;

	if (gen_params == NULL)
		return;

	if (gen_params->stop_sequences != NULL)
	{
		for (i = 0; i < gen_params->num_stop_sequences; i++)
		{
			if (gen_params->stop_sequences[i] != NULL)
				NDB_FREE(gen_params->stop_sequences[i]);
		}
		NDB_FREE(gen_params->stop_sequences);
		gen_params->stop_sequences = NULL;
		gen_params->num_stop_sequences = 0;
	}

	if (gen_params->logit_bias_tokens != NULL)
	{
		NDB_FREE(gen_params->logit_bias_tokens);
		gen_params->logit_bias_tokens = NULL;
	}

	if (gen_params->logit_bias_values != NULL)
	{
		NDB_FREE(gen_params->logit_bias_values);
		gen_params->logit_bias_values = NULL;
	}

	gen_params->num_logit_bias = 0;
}

/*
 * ndb_json_extract_openai_response - Extract OpenAI API response
 *
 * Parses an OpenAI API response (chat completion or embedding) and extracts
 * the generated text, token counts, and error messages. Handles both successful
 * responses and error responses from the OpenAI API.
 *
 * Parameters:
 *   json_str - Input JSON string from OpenAI API
 *   response - Output structure to populate with extracted data
 *
 * Returns:
 *   0 on success, -1 on error
 *
 * Notes:
 *   The function initializes the response structure before parsing.
 *   Memory for text and error_message is allocated in CurrentMemoryContext.
 *   The function first attempts JSONB parsing for robustness, then falls
 *   back to string-based extraction if needed. For chat completions, extracts
 *   text from choices[0].message.content. For embeddings, extracts data array.
 */
int
ndb_json_extract_openai_response(const char *json_str,
								 NdbOpenAIResponse *response)
{
	volatile	Jsonb *jsonb = NULL;

	NDB_DECLARE(Jsonb *, choices);
	NDB_DECLARE(Jsonb *, first_choice);
	NDB_DECLARE(Jsonb *, message);
	NDB_DECLARE(Jsonb *, content);
	NDB_DECLARE(Jsonb *, usage);
	NDB_DECLARE(text *, content_text);
	NDB_DECLARE(char *, content_str);

	if (json_str == NULL || response == NULL)
		return -1;

	memset(response, 0, sizeof(NdbOpenAIResponse));
	response->text = NULL;
	response->tokens_in = 0;
	response->tokens_out = 0;
	response->error_message = NULL;

	if (strncmp(json_str, "{\"error\"", 8) == 0)
	{
		NDB_DECLARE(Jsonb *, error_field);
		NDB_DECLARE(text *, error_text);

		jsonb = ndb_jsonb_in_cstring(json_str);
		if (jsonb != NULL)
		{
			error_field = ndb_jsonb_object_field((Jsonb *) jsonb, "error");
			if (error_field != NULL)
			{
				NDB_DECLARE(Jsonb *, msg_field);
				NDB_DECLARE(text *, msg_text);

				msg_field = ndb_jsonb_object_field(error_field, "message");
				if (msg_field != NULL)
				{
					msg_text = ndb_jsonb_out(msg_field);
					if (msg_text != NULL)
					{
						response->error_message = text_to_cstring(msg_text);
						NDB_FREE(msg_text);
					}
					NDB_FREE(msg_field);
				}
				else
				{
					error_text = ndb_jsonb_out(error_field);
					if (error_text != NULL)
					{
						response->error_message = text_to_cstring(error_text);
						NDB_FREE(error_text);
					}
				}
				NDB_FREE(error_field);
			}
			if (jsonb != NULL)
			{
				Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

				NDB_FREE(jsonb_ptr);
				jsonb = NULL;
			}
		}
		return -1;
	}

	PG_TRY();
	{
		jsonb = ndb_jsonb_in_cstring(json_str);
		if (jsonb != NULL)
		{
			choices = ndb_jsonb_object_field((Jsonb *) jsonb, "choices");
			if (choices != NULL)
			{
				first_choice = ndb_jsonb_array_element(choices, 0);
				if (first_choice != NULL)
				{
					message = ndb_jsonb_object_field(first_choice, "message");
					if (message != NULL)
					{
						content = ndb_jsonb_object_field(message, "content");
						if (content != NULL)
						{
							content_text = ndb_jsonb_out(content);
							if (content_text != NULL)
							{
								content_str = text_to_cstring(content_text);
								if (content_str[0] == '"' && content_str[strlen(content_str) - 1] == '"')
								{
									char	   *temp = pnstrdup(content_str + 1, strlen(content_str) - 2);

									NDB_FREE(content_str);
									content_str = temp;
									if (strchr(content_str, '\\') != NULL)
									{
										char	   *unescaped = ndb_json_unescape_string(content_str);

										NDB_FREE(content_str);
										content_str = unescaped;
									}
								}
								response->text = content_str;
								NDB_FREE(content_text);
							}
							NDB_FREE(content);
						}
						NDB_FREE(message);
					}
					NDB_FREE(first_choice);
				}
				NDB_FREE(choices);
			}

			usage = ndb_jsonb_object_field((Jsonb *) jsonb, "usage");
			if (usage != NULL)
			{
				NDB_DECLARE(Jsonb *, prompt_tokens_field);
				NDB_DECLARE(Jsonb *, completion_tokens_field);

				prompt_tokens_field = ndb_jsonb_object_field(usage, "prompt_tokens");
				completion_tokens_field = ndb_jsonb_object_field(usage, "completion_tokens");

				if (prompt_tokens_field != NULL)
				{
					NDB_DECLARE(text *, tokens_text);
					NDB_DECLARE(char *, tokens_str);

					tokens_text = ndb_jsonb_out(prompt_tokens_field);
					if (tokens_text != NULL)
					{
						tokens_str = text_to_cstring(tokens_text);
						response->tokens_in = (int) strtol(tokens_str, NULL, 10);
						NDB_FREE(tokens_str);
						NDB_FREE(tokens_text);
					}
					NDB_FREE(prompt_tokens_field);
				}

				NDB_DECLARE(text *, tokens_text);
				NDB_DECLARE(char *, tokens_str);

				if (completion_tokens_field != NULL)
				{
					tokens_text = ndb_jsonb_out(completion_tokens_field);
					if (tokens_text != NULL)
					{
						tokens_str = text_to_cstring(tokens_text);
						response->tokens_out = (int) strtol(tokens_str, NULL, 10);
						NDB_FREE(tokens_str);
						NDB_FREE(tokens_text);
					}
					NDB_FREE(completion_tokens_field);
				}
				NDB_FREE(usage);
			}
		}
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
	}
	PG_END_TRY();

	if (response->text == NULL)
	{
		const char *p;
		const char *q;
		size_t		len;

		NDB_DECLARE(char *, result);
		NDB_DECLARE(char *, unescaped);
		int			escape_next = 0;

		p = strstr(json_str, "\"choices\"");
		if (p != NULL)
		{
			p = strchr(p, '[');
			if (p != NULL)
			{
				p++;
				while (*p && (isspace((unsigned char) *p) || *p == '{'))
				{
					if (*p == '{')
						break;
					p++;
				}
				if (*p == '{')
				{
					p = strstr(p, "\"message\"");
					if (p != NULL)
					{
						p = strchr(p, '{');
						if (p != NULL)
						{
							p++;
							p = strstr(p, "\"content\"");
							if (p != NULL)
							{
								p = strchr(p, ':');
								if (p != NULL)
								{
									p++;
									while (*p && isspace((unsigned char) *p))
										p++;

									if (*p == '"')
									{
										p++;

										q = p;
										len = 0;
										while (*q)
										{
											if (escape_next)
											{
												escape_next = 0;
												len++;
												q++;
												continue;
											}
											if (*q == '\\')
											{
												escape_next = 1;
												len++;
												q++;
												continue;
											}
											if (*q == '"')
												break;
											len++;
											q++;
										}

										if (len > 0)
										{
											NDB_ALLOC(result, char, len + 1);
											unescaped = result;
											q = p;

											while (q < p + len)
											{
												if (*q == '\\' && q + 1 < p + len)
												{
													switch (q[1])
													{
														case 'n':
															*unescaped++ = '\n';
															q += 2;
															break;
														case 't':
															*unescaped++ = '\t';
															q += 2;
															break;
														case 'r':
															*unescaped++ = '\r';
															q += 2;
															break;
														case '\\':
															*unescaped++ = '\\';
															q += 2;
															break;
														case '"':
															*unescaped++ = '"';
															q += 2;
															break;
														case '/':
															*unescaped++ = '/';
															q += 2;
															break;
														case 'u':
															if (q + 5 < p + len && isxdigit((unsigned char) q[2]) &&
																isxdigit((unsigned char) q[3]) &&
																isxdigit((unsigned char) q[4]) &&
																isxdigit((unsigned char) q[5]))
															{
																unsigned int code = 0;

																sscanf(q + 2, "%4x", &code);
																if (code < 0x80)
																	*unescaped++ = (char) code;
																else if (code < 0x800)
																{
																	*unescaped++ = (char) (0xC0 | (code >> 6));
																	*unescaped++ = (char) (0x80 | (code & 0x3F));
																}
																else if (code < 0xD800 || code >= 0xE000)
																{
																	*unescaped++ = (char) (0xE0 | (code >> 12));
																	*unescaped++ = (char) (0x80 | ((code >> 6) & 0x3F));
																	*unescaped++ = (char) (0x80 | (code & 0x3F));
																}
																else
																{
																	if (code >= 0xD800 && code < 0xDC00 && q + 11 < p + len &&
																		q[6] == '\\' && q[7] == 'u' &&
																		isxdigit((unsigned char) q[8]) &&
																		isxdigit((unsigned char) q[9]) &&
																		isxdigit((unsigned char) q[10]) &&
																		isxdigit((unsigned char) q[11]))
																	{
																		unsigned int code2 = 0;

																		sscanf(q + 8, "%4x", &code2);
																		if (code2 >= 0xDC00 && code2 < 0xE000)
																		{
																			/*
																			 *
																			 * Valid
																			 *
																			 * surrogate
																			 *
																			 * pair
																			 *
																			 * -
																			 *
																			 * 4-byte
																			 *
																			 * UTF-8
																			 */
																			unsigned int full_code = 0x10000 + ((code - 0xD800) << 10) + (code2 - 0xDC00);

																			*unescaped++ = (char) (0xF0 | (full_code >> 18));
																			*unescaped++ = (char) (0x80 | ((full_code >> 12) & 0x3F));
																			*unescaped++ = (char) (0x80 | ((full_code >> 6) & 0x3F));
																			*unescaped++ = (char) (0x80 | (full_code & 0x3F));
																			q += 6; /* Skip second \uXXXX */
																		}
																		else
																		{
																			/*
																			 *
																			 * Invalid
																			 *
																			 * surrogate
																			 *
																			 * pair
																			 *
																			 * -
																			 *
																			 * use
																			 *
																			 * replacement
																			 *
																			 * character
																			 */
																			*unescaped++ = 0xEF;
																			*unescaped++ = 0xBF;
																			*unescaped++ = 0xBD;
																		}
																	}
																	else
																	{
																		/*
																		 *
																		 * Invalid
																		 *
																		 * surrogate
																		 * -
																		 * use
																		 * replacement
																		 *
																		 * character
																		 */
																		*unescaped++ = 0xEF;
																		*unescaped++ = 0xBF;
																		*unescaped++ = 0xBD;
																	}
																}
																q += 6;
															}
															else
															{
																*unescaped++ = *q++;
															}
															break;
														default:
															*unescaped++ = *q++;
															*unescaped++ = *q++;
															break;
													}
												}
												else
												{
													*unescaped++ = *q++;
												}
											}
											*unescaped = '\0';
											response->text = result;
										}
									}
								}
							}
						}
					}
				}
			}
		}

		p = strstr(json_str, "\"prompt_tokens\":");
		if (p != NULL)
		{
			p = strchr(p, ':');
			if (p != NULL)
			{
				p++;
				while (*p && isspace((unsigned char) *p))
					p++;
				response->tokens_in = (int) strtol(p, NULL, 10);
			}
		}

		p = strstr(json_str, "\"completion_tokens\":");
		if (p != NULL)
		{
			p = strchr(p, ':');
			if (p != NULL)
			{
				p++;
				while (*p && isspace((unsigned char) *p))
					p++;
				response->tokens_out = (int) strtol(p, NULL, 10);
			}
		}
	}

	return (response->text != NULL) ? 0 : -1;
}

/*
 * ndb_json_extract_openai_response_free - Free OpenAI response structure
 */
void
ndb_json_extract_openai_response_free(NdbOpenAIResponse *response)
 *   Safe to call with NULL pointer. After calling this function, the structure
 *   should not be used unless re-initialized.
 */
void
ndb_json_extract_openai_response_free(NdbOpenAIResponse *response)
{
	if (response == NULL)
		return;

	if (response->text != NULL)
	{
		NDB_FREE(response->text);
		response->text = NULL;
	}

	if (response->error_message != NULL)
	{
		NDB_FREE(response->error_message);
		response->error_message = NULL;
	}
}

/*
 * ndb_json_parse_openai_embedding - Parse OpenAI embedding vector from JSON
 */
int
ndb_json_parse_openai_embedding(const char *json_str,
								float **vec_out,
								int *dim_out)
{
	volatile	Jsonb *jsonb = NULL;

	NDB_DECLARE(Jsonb *, data);
	NDB_DECLARE(Jsonb *, first_item);
	NDB_DECLARE(Jsonb *, embedding);
	volatile float *vec = NULL;
	volatile int n = 0;
	volatile int cap = 256;
	volatile int status = -1;

	if (json_str == NULL || vec_out == NULL || dim_out == NULL)
		return -1;

	*vec_out = NULL;
	*dim_out = 0;
	PG_TRY();
	{
		jsonb = ndb_jsonb_in_cstring(json_str);
		if (jsonb != NULL)
		{
			/* Extract data array */
			data = ndb_jsonb_object_field((Jsonb *) jsonb, "data");
			if (data != NULL)
			{
				/* Get first item from data array */
				first_item = ndb_jsonb_array_element(data, 0);
				if (first_item != NULL)
				{
					/* Get embedding array */
					embedding = ndb_jsonb_object_field(first_item, "embedding");
					if (embedding != NULL)
					{
						/* Parse array of floats */
						JsonbIterator *it;
						JsonbValue	v;
						JsonbIteratorToken type;

						NDB_DECLARE(float *, vec);
						NDB_ALLOC(vec, float, (int) cap);

						it = JsonbIteratorInit(&embedding->root);
						while ((type = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
						{
							if (type == WJB_ELEM)
							{
								if (v.type == jbvNumeric)
								{
									Numeric		num;
									float		float_val;

									if ((int) n >= (int) cap)
									{
										cap *= 2;
										vec = (float *) repalloc((float *) vec, sizeof(float) * (int) cap);
									}

									num = v.val.numeric;
									float_val = DatumGetFloat4(DirectFunctionCall1(numeric_float4,
																				   NumericGetDatum(num)));

									((float *) vec)[(int) n++] = float_val;
								}
							}
						}

						if ((int) n > 0)
						{
							/* Trim to actual size */
							if ((int) n < (int) cap)
							{
								vec = (float *) repalloc((float *) vec, sizeof(float) * (int) n);
							}
							*vec_out = (float *) vec;
							*dim_out = (int) n;
							status = 0;
						}
						else
						{
							if (vec != NULL)
							{
								float	   *vec_ptr = (float *) vec;

								NDB_FREE(vec_ptr);
								vec = NULL;
							}
						}
						NDB_FREE(embedding);
					}
					NDB_FREE(first_item);
				}
				NDB_FREE(data);
			}
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		status = -1;
		if (vec != NULL)
		{
			{
				float	   *vec_ptr = (float *) vec;

				NDB_FREE(vec_ptr);
				vec = NULL;
			}
			vec = NULL;
		}
	}
	PG_END_TRY();

	if (jsonb != NULL)
	{
		Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

		NDB_FREE(jsonb_ptr);
		jsonb = NULL;
	}

	if ((int) status == 0)
		return 0;

	/* Fallback to string-based parsing */
	{
		const char *p;
		char	   *endptr;
		double		v;
		bool		in_array = false;

		/* Look for "data" array first, then "embedding" within it */
		p = strstr(json_str, "\"data\"");
		if (p != NULL)
		{
			/* Find opening bracket of data array */
			p = strchr(p, '[');
			if (p != NULL)
			{
				p++;
				/* Find first object in data array */
				p = strchr(p, '{');
				if (p != NULL)
				{
					/* Look for "embedding" within this object */
					p = strstr(p, "\"embedding\"");
				}
			}
		}

		/* If not found in data array, try direct search */
		if (p == NULL || strstr(p, "\"embedding\"") == NULL)
		{
			p = strstr(json_str, "\"embedding\":");
		}

		if (p != NULL)
		{
			/* Find the opening bracket after "embedding": */
			p = strchr(p, '[');
			if (p != NULL)
			{
				NDB_DECLARE(float *, vec);

				p++;
				in_array = true;

				/* Allocate initial vector */
				NDB_ALLOC(vec, float, cap);

				/* Parse array of floats */
				while (*p && in_array)
				{
					/* Skip whitespace and commas */
					while (*p && (isspace((unsigned char) *p) || *p == ','))
						p++;

					if (*p == ']')
					{
						in_array = false;
						break;
					}

					if (!*p)
						break;

					/* Parse float value */
					endptr = NULL;
					errno = 0;
					v = strtod(p, &endptr);

					/* Check for parse error */
					if (endptr == p || errno == ERANGE)
					{
						/* Invalid number, try to continue */
						while (*p && *p != ',' && *p != ']' && !isspace((unsigned char) *p))
							p++;
						continue;
					}

					/* Check for overflow */
					if (v > FLT_MAX || v < -FLT_MAX)
					{
						/* Value out of range, skip */
						p = endptr;
						continue;
					}

					/* Grow array if needed */
					if (n >= cap)
					{
						cap = cap * 2;
						vec = (volatile float *) repalloc((void *) vec, sizeof(float) * cap);
					}

					vec[n++] = (float) v;
					p = endptr;
				}

				if (n > 0)
				{
					/* Trim to actual size */
					if (n < cap)
					{
						vec = (volatile float *) repalloc((void *) vec, sizeof(float) * n);
					}
					*vec_out = (float *) vec;
					*dim_out = n;
					return 0;
				}
				else
				{
					if (vec != NULL)
					{
						pfree((void *) vec);
						vec = NULL;
					}
				}
			}
		}
	}

	return -1;
}

/*
 * ndb_json_parse_sparse_vector - Parse sparse vector from JSON
 *
 * Parses a sparse vector representation from a JSON string into an
 * NdbSparseVectorParse structure. Supports sparse vector formats including
 * vocab_size, model type (BM25, SPLADE, ColBERTv2), tokens array, and
 * weights array.
 *
 * Parameters:
 *   json_str - Input JSON string containing sparse vector data
 *   result - Output structure to populate with parsed sparse vector data
 *   errstr - Optional pointer to error message string (allocated on error)
 *
 * Returns:
 *   0 on success, -1 on error
 *
 * Notes:
 *   The function initializes result with default values (vocab_size=30522,
 *   model_type=SPLADE) before parsing. Memory for token_ids and weights arrays
 *   is allocated in CurrentMemoryContext. If errstr is not NULL and an error
 *   occurs, an error message is allocated and assigned to *errstr. The function
 *   first attempts JSONB parsing for robustness, then falls back to string-based
 *   parsing if needed.
 */
int
ndb_json_parse_sparse_vector(const char *json_str,
							 NdbSparseVectorParse *result,
							 char **errstr)
{
	volatile	Jsonb *jsonb = NULL;

	NDB_DECLARE(Jsonb *, field);
	NDB_DECLARE(text *, field_text);
	NDB_DECLARE(char *, value_str);
	volatile	int32 *token_ids = NULL;
	volatile	float4 *weights = NULL;
	int			capacity = 16;
	volatile int nnz = 0;
	int			i;

	if (errstr)
		*errstr = NULL;
	if (json_str == NULL || result == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for parse_sparse_vector");
		return -1;
	}

	memset(result, 0, sizeof(NdbSparseVectorParse));
	result->vocab_size = 30522;
	result->model_type = 1;
	result->nnz = 0;
	result->token_ids = NULL;
	result->weights = NULL;
	PG_TRY();
	{
		jsonb = ndb_jsonb_in_cstring(json_str);
		if (jsonb != NULL)
		{
			field = ndb_jsonb_object_field((Jsonb *) jsonb, "vocab_size");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					result->vocab_size = (int32) strtol(value_str, NULL, 10);
					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "model");
			if (field != NULL)
			{
				field_text = ndb_jsonb_out(field);
				if (field_text != NULL)
				{
					value_str = text_to_cstring(field_text);
					/* Remove quotes if present */
					if (value_str[0] == '"' && value_str[strlen(value_str) - 1] == '"')
					{
						char	   *temp = pnstrdup(value_str + 1, strlen(value_str) - 2);

						NDB_FREE(value_str);
						value_str = temp;
					}

					if (strcmp(value_str, "BM25") == 0)
						result->model_type = 0;
					else if (strcmp(value_str, "SPLADE") == 0)
						result->model_type = 1;
					else if (strcmp(value_str, "ColBERTv2") == 0)
						result->model_type = 2;

					NDB_FREE(value_str);
					NDB_FREE(field_text);
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "tokens");
			if (field != NULL)
			{
				NDB_DECLARE(char *, tmp);
				int		   *temp_tokens = NULL;
				int			count = 0;

				tmp = ndb_jsonb_out_cstring(field);
				if (tmp != NULL)
				{
					temp_tokens = ndb_json_parse_int_array(tmp, &count);
					NDB_FREE(tmp);
					if (temp_tokens != NULL && count > 0)
					{
						NDB_DECLARE(int32 *, token_ids);
						NDB_ALLOC(token_ids, int32, count);
						for (i = 0; i < count; i++)
							((int32 *) token_ids)[i] = (int32) temp_tokens[i];
						nnz = count;
						NDB_FREE(temp_tokens);
					}
				}
				NDB_FREE(field);
			}

			field = ndb_jsonb_object_field((Jsonb *) jsonb, "weights");
			if (field != NULL)
			{
				NDB_DECLARE(char *, tmp);
				float	   *temp_weights = NULL;
				int			count = 0;

				tmp = ndb_jsonb_out_cstring(field);
				if (tmp != NULL)
				{
					temp_weights = ndb_json_parse_float_array(tmp, &count);
					NDB_FREE(tmp);
					if (temp_weights != NULL && count > 0)
					{
						NDB_DECLARE(float4 *, weights);
						NDB_ALLOC(weights, float4, count);
						for (i = 0; i < count && i < (int) nnz; i++)
							((float4 *) weights)[i] = (float4) temp_weights[i];
						NDB_FREE(temp_weights);
					}
				}
				NDB_FREE(field);
			}
		}
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
	}
	PG_END_TRY();

	if ((int) nnz == 0)
	{
		const char *ptr;
		char	   *tokens_start;
		char	   *tokens_end;
		char	   *weights_start;
		char	   *weights_end;
		char	   *tok_ptr;
		char	   *wgt_ptr;
		int			idx;

		NDB_DECLARE(int32 *, token_ids);
		NDB_DECLARE(float4 *, weights);

		ptr = json_str;
		while (*ptr && *ptr != '{')
			ptr++;

		if (strstr(ptr, "\"vocab_size\":") != NULL || strstr(ptr, "vocab_size:") != NULL)
		{
			const char *vs_ptr = strstr(ptr, "\"vocab_size\":");

			if (vs_ptr == NULL)
				vs_ptr = strstr(ptr, "vocab_size:");
			if (vs_ptr != NULL)
			{
				vs_ptr = strchr(vs_ptr, ':');
				if (vs_ptr != NULL)
				{
					vs_ptr++;
					while (*vs_ptr && isspace((unsigned char) *vs_ptr))
						vs_ptr++;
					result->vocab_size = (int32) strtol(vs_ptr, NULL, 10);
				}
			}
		}

		if (strstr(ptr, "\"model\":\"BM25\"") != NULL || strstr(ptr, "model:BM25") != NULL)
			result->model_type = 0;
		else if (strstr(ptr, "\"model\":\"SPLADE\"") != NULL || strstr(ptr, "model:SPLADE") != NULL)
			result->model_type = 1;
		else if (strstr(ptr, "\"model\":\"ColBERTv2\"") != NULL || strstr(ptr, "model:ColBERTv2") != NULL)
			result->model_type = 2;

		NDB_ALLOC(token_ids, int32, capacity);
		NDB_ALLOC(weights, float4, capacity);

		tokens_start = strstr(ptr, "\"tokens\":[");
		if (tokens_start == NULL)
			tokens_start = strstr(ptr, "tokens:[");
		if (tokens_start != NULL)
		{
			tokens_start = strchr(tokens_start, '[');
			if (tokens_start != NULL)
			{
				tokens_start++;
				tokens_end = strchr(tokens_start, ']');
				if (tokens_end != NULL)
				{
					tok_ptr = tokens_start;

					while (tok_ptr < tokens_end && *tok_ptr)
					{
						if (nnz >= capacity)
						{
							capacity *= 2;
							token_ids = (volatile int32 *) repalloc((void *) token_ids, sizeof(int32) * capacity);
							weights = (volatile float4 *) repalloc((void *) weights, sizeof(float4) * capacity);
						}

						while (*tok_ptr == ' ' || *tok_ptr == ',')
							tok_ptr++;

						if (*tok_ptr == ']')
							break;

						token_ids[nnz] = (int32) strtol(tok_ptr, NULL, 10);
						while (*tok_ptr && *tok_ptr != ',' && *tok_ptr != ']')
							tok_ptr++;
						nnz++;
					}
				}
			}
		}

		weights_start = strstr(ptr, "\"weights\":[");
		if (weights_start == NULL)
			weights_start = strstr(ptr, "weights:[");
		if (weights_start != NULL)
		{
			weights_start = strchr(weights_start, '[');
			if (weights_start != NULL)
			{
				weights_start++;
				weights_end = strchr(weights_start, ']');
				if (weights_end != NULL)
				{
					wgt_ptr = weights_start;
					idx = 0;

					while (wgt_ptr < weights_end && *wgt_ptr && idx < nnz)
					{
						while (*wgt_ptr == ' ' || *wgt_ptr == ',')
							wgt_ptr++;

						if (*wgt_ptr == ']')
							break;

						weights[idx] = (float4) strtof(wgt_ptr, NULL);
						while (*wgt_ptr && *wgt_ptr != ',' && *wgt_ptr != ']')
							wgt_ptr++;
						idx++;
					}

					for (; idx < nnz; idx++)
						weights[idx] = 0.0f;
				}
			}
		}
	}

	if ((int) nnz == 0)
	{
		if (token_ids != NULL)
		{
			int32	   *token_ids_ptr = (int32 *) token_ids;

			NDB_FREE(token_ids_ptr);
			token_ids = NULL;
		}
		if (weights != NULL)
		{
			float4	   *weights_ptr = (float4 *) weights;

			NDB_FREE(weights_ptr);
			weights = NULL;
		}
		if (errstr)
			*errstr = pstrdup("sparse_vector must have at least one token");
		return -1;
	}

	if (result->vocab_size == 0)
		result->vocab_size = 30522;

	if (weights == NULL && (int) nnz > 0)
	{
		NDB_DECLARE(float4 *, weights);
		NDB_ALLOC(weights, float4, (int) nnz);
	}

	result->nnz = (int32) nnz;
	result->token_ids = (int32 *) token_ids;
	result->weights = (float4 *) weights;

	return 0;
}

/*
 * ndb_json_parse_sparse_vector_free - Free sparse vector parse structure
 *
 * Frees all dynamically allocated memory in an NdbSparseVectorParse structure,
 * including token_ids and weights arrays. Resets counts to zero.
 *
 * Parameters:
 *   result - Structure to free resources from
 *
 * Notes:
 *   Safe to call with NULL pointer. After calling this function, the structure
 *   should not be used unless re-initialized.
 */
void
ndb_json_parse_sparse_vector_free(NdbSparseVectorParse *result)
{
	if (result == NULL)
		return;

	if (result->token_ids != NULL)
	{
		NDB_FREE(result->token_ids);
		result->token_ids = NULL;
	}

	if (result->weights != NULL)
	{
		NDB_FREE(result->weights);
		result->weights = NULL;
	}

	result->nnz = 0;
}

/*
 * ndb_json_build_object - Build JSON object string from key-value pairs
 */
char *
ndb_json_build_object(const char *key1, const char *value1,...)
{
	StringInfoData buf;
	va_list		args;
	const char *key;
	const char *value;
	bool		first = true;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '{');

	va_start(args, value1);

	key = key1;
	value = value1;

	while (key != NULL)
	{
		if (!first)
			appendStringInfoChar(&buf, ',');
		first = false;

		ndb_json_quote_string_buf(&buf, key);
		appendStringInfoChar(&buf, ':');
		ndb_json_quote_string_buf(&buf, value);

		key = va_arg(args, const char *);
		if (key != NULL)
			value = va_arg(args, const char *);
	}

	va_end(args);

	appendStringInfoChar(&buf, '}');

	return buf.data;
}

/*
 * ndb_json_build_object_buf - Build JSON object into StringInfo buffer
 */
void
ndb_json_build_object_buf(StringInfo buf, const char *key1, const char *value1,...)
{
	va_list		args;
	const char *key;
	const char *value;
	bool		first = true;

	if (buf == NULL)
		return;

	appendStringInfoChar(buf, '{');

	va_start(args, value1);

	key = key1;
	value = value1;

	while (key != NULL)
	{
		if (!first)
			appendStringInfoChar(buf, ',');
		first = false;

		ndb_json_quote_string_buf(buf, key);
		appendStringInfoChar(buf, ':');
		ndb_json_quote_string_buf(buf, value);

		key = va_arg(args, const char *);
		if (key != NULL)
			value = va_arg(args, const char *);
	}

	va_end(args);

	appendStringInfoChar(buf, '}');
}

/*
 * ndb_json_build_array - Build JSON array string from values
 *
 * Constructs a JSON array string from a variable number of string values
 * provided as arguments. Uses variadic arguments to accept multiple values.
 *
 * Parameters:
 *   value1 - First array element (C string)
 *   ... - Additional array elements, terminated with NULL
 *
 * Returns:
 *   Newly allocated C string containing JSON array
 *
 * Notes:
 *   Memory for the returned string is allocated in CurrentMemoryContext.
 *   Caller is responsible for freeing using pfree or NDB_FREE. All values
 *   are automatically quoted and escaped according to JSON string encoding rules.
 */
char *
ndb_json_build_array(const char *value1,...)
{
	StringInfoData buf;
	va_list		args;
	const char *value;
	bool		first = true;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	va_start(args, value1);

	value = value1;

	while (value != NULL)
	{
		if (!first)
			appendStringInfoChar(&buf, ',');
		first = false;

		ndb_json_quote_string_buf(&buf, value);

		value = va_arg(args, const char *);
	}

	va_end(args);

	appendStringInfoChar(&buf, ']');

	return buf.data;
}

/*
 * ndb_json_build_array_buf - Build JSON array into StringInfo buffer
 *
 * Constructs a JSON array and appends it to an existing StringInfo buffer.
 * Similar to ndb_json_build_array but appends to a buffer instead of
 * allocating a new string.
 *
 * Parameters:
 *   buf - StringInfo buffer to append JSON array to
 *   value1 - First array element (C string)
 *   ... - Additional array elements, terminated with NULL
 *
 * Notes:
 *   If buf is NULL, the function returns without doing anything. All values
 *   are automatically quoted and escaped according to JSON string encoding rules.
 */
void
ndb_json_build_array_buf(StringInfo buf, const char *value1,...)
{
	va_list		args;
	const char *value;
	bool		first = true;

	if (buf == NULL)
		return;

	appendStringInfoChar(buf, '[');

	va_start(args, value1);

	value = value1;

	while (value != NULL)
	{
		if (!first)
			appendStringInfoChar(buf, ',');
		first = false;

		ndb_json_quote_string_buf(buf, value);

		value = va_arg(args, const char *);
	}

	va_end(args);

	appendStringInfoChar(buf, ']');
}

/*
 * ndb_json_merge_objects - Merge two JSON objects
 *
 * Merges two JSON objects into a single JSON object string. Fields from
 * both objects are combined, with fields from json2 potentially overwriting
 * fields from json1 if there are duplicate keys.
 *
 * Parameters:
 *   json1 - First JSON object string
 *   json2 - Second JSON object string
 *
 * Returns:
 *   Newly allocated C string containing merged JSON object, "{}" if both
 *   inputs are NULL, or a copy of the non-NULL input if one is NULL
 *
 * Notes:
 *   Memory for the returned string is allocated in CurrentMemoryContext.
 *   Caller is responsible for freeing using pfree or NDB_FREE. The function
 *   first attempts JSONB parsing for robustness, then falls back to
 *   string-based merging if parsing fails.
 */
char *
ndb_json_merge_objects(const char *json1, const char *json2)
{
	volatile	Jsonb *jsonb1 = NULL;
	volatile	Jsonb *jsonb2 = NULL;
	char	   *result = NULL;

	if (json1 == NULL && json2 == NULL)
		return pstrdup("{}");
	if (json1 == NULL)
		return pstrdup(json2);
	if (json2 == NULL)
		return pstrdup(json1);

	PG_TRY();
	{
		jsonb1 = ndb_jsonb_in_cstring(json1);
		jsonb2 = ndb_jsonb_in_cstring(json2);

		if (jsonb1 != NULL && jsonb2 != NULL)
		{
			char	   *str1 = ndb_jsonb_out_cstring((Jsonb *) jsonb1);
			char	   *str2 = ndb_jsonb_out_cstring((Jsonb *) jsonb2);

			StringInfoData buf;

			initStringInfo(&buf);
			appendStringInfoChar(&buf, '{');

			/* Add fields from first object */
			if (str1 != NULL && strlen(str1) > 2)
			{
				appendStringInfo(&buf, "%.*s", (int) (strlen(str1) - 2), str1 + 1);
			}

			if (str2 != NULL && strlen(str2) > 2)
			{
				if (str1 != NULL && strlen(str1) > 2)
					appendStringInfoChar(&buf, ',');
				appendStringInfo(&buf, "%.*s", (int) (strlen(str2) - 2), str2 + 1);
			}

			appendStringInfoChar(&buf, '}');
			result = buf.data;

			if (str1 != NULL)
				NDB_FREE(str1);
			if (str2 != NULL)
				NDB_FREE(str2);
		}
		if (jsonb1 != NULL)
		{
			Jsonb	   *jsonb1_ptr = (Jsonb *) jsonb1;

			NDB_FREE(jsonb1_ptr);
			jsonb1 = NULL;
		}
		if (jsonb2 != NULL)
		{
			Jsonb	   *jsonb2_ptr = (Jsonb *) jsonb2;

			NDB_FREE(jsonb2_ptr);
			jsonb2 = NULL;
		}
	}
	PG_CATCH();
	{
		StringInfoData buf;

		FlushErrorState();
		if (jsonb1 != NULL)
		{
			Jsonb	   *jsonb1_ptr = (Jsonb *) jsonb1;

			NDB_FREE(jsonb1_ptr);
			jsonb1 = NULL;
		}
		if (jsonb2 != NULL)
		{
			Jsonb	   *jsonb2_ptr = (Jsonb *) jsonb2;

			NDB_FREE(jsonb2_ptr);
			jsonb2 = NULL;
		}

		initStringInfo(&buf);
		appendStringInfoString(&buf, "{");
		if (json1 != NULL && strlen(json1) > 2)
			appendStringInfo(&buf, "%.*s", (int) (strlen(json1) - 2), json1 + 1);
		if (json2 != NULL && strlen(json2) > 2)
		{
			if (json1 != NULL && strlen(json1) > 2)
				appendStringInfoChar(&buf, ',');
			appendStringInfo(&buf, "%.*s", (int) (strlen(json2) - 2), json2 + 1);
		}
		appendStringInfoChar(&buf, '}');
		result = buf.data;
	}
	PG_END_TRY();

	return result;
}

/*
 * ndb_json_parse_array - Parse JSON array into string array
 */
char	  **
ndb_json_parse_array(const char *json_str, int *count)
{
	NDB_DECLARE(Jsonb *, jsonb);
	NDB_DECLARE(Jsonb *, array_field);
	char	  **result = NULL;
	int			n = 0;
	int			capacity = 16;

	if (json_str == NULL || count == NULL)
		return NULL;

	*count = 0;

	PG_TRY();
	{
		jsonb = ndb_jsonb_in_cstring(json_str);
		if (jsonb != NULL)
		{
			array_field = jsonb;

			if (array_field != NULL)
			{
				JsonbIterator *it;
				JsonbValue	v;
				JsonbIteratorToken type;

				NDB_DECLARE(char **, local_result);
				NDB_ALLOC(local_result, char *, (int) capacity);

				it = JsonbIteratorInit(&array_field->root);
				while ((type = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
				{
					if (type == WJB_ELEM)
					{
						if ((int) n >= capacity)
						{
							capacity *= 2;
							local_result = (char **) repalloc((char **) local_result, sizeof(char *) * capacity);
						}

						if (v.type == jbvString)
						{
							local_result[(int) n] = pnstrdup(v.val.string.val, v.val.string.len);
							n++;
						}
						else
						{
							Jsonb	   *elem_jsonb = JsonbValueToJsonb(&v);
							text	   *elem_text = NULL;
							char	   *elem_str = NULL;

							if (elem_jsonb != NULL)
							{
								elem_text = ndb_jsonb_out(elem_jsonb);
								if (elem_text != NULL)
								{
									elem_str = text_to_cstring(elem_text);
									local_result[(int) n] = elem_str;
									n++;
									NDB_FREE(elem_text);
								}
								NDB_FREE(elem_jsonb);
							}
						}
					}
					result = local_result;
				}
			}
		}
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
		if (result != NULL)
		{
			{
				char	  **result_ptr = (char **) result;

				NDB_FREE(result_ptr);
				result = NULL;
			}
			result = NULL;
		}
	}
	PG_END_TRY();

	if (result == NULL || (int) n == 0)
	{
		const char *p;
		const char *start;
		const char *end;
		char	   *value;

		if (result != NULL)
		{
			{
				char	  **result_ptr = (char **) result;

				NDB_FREE(result_ptr);
				result = NULL;
			}
			result = NULL;
		}

		p = json_str;
		while (*p && *p != '[')
			p++;
		if (*p == '[')
		{
			NDB_DECLARE(char **, local_result);

			p++;
			NDB_ALLOC(local_result, char *, capacity);

			while (*p && *p != ']')
			{
				while (*p && (isspace((unsigned char) *p) || *p == ','))
					p++;

				if (*p == ']')
					break;

				start = p;

				if (*p == '"')
				{
					p++;
					while (*p && *p != '"')
					{
						if (*p == '\\' && p[1])
							p += 2;
						else
							p++;
					}
					if (*p == '"')
					{
						end = p;
						value = pnstrdup(start + 1, end - start - 1);
						if (strchr(value, '\\') != NULL)
						{
							char	   *unescaped = ndb_json_unescape_string(value);

							NDB_FREE(value);
							value = unescaped;
						}
					}
					else
					{
						value = pstrdup("");
					}
					p++;
				}
				else
				{
					while (*p && *p != ',' && *p != ']' && !isspace((unsigned char) *p))
						p++;
					end = p;
					value = pnstrdup(start, end - start);
				}

				if (n >= capacity)
				{
					capacity *= 2;
					local_result = repalloc(local_result, sizeof(char *) * capacity);
				}

				local_result[n++] = value;
			}
			result = local_result;
		}
	}

	if (n > 0)
	{
		*count = n;
		return result;
	}

	if (result != NULL)
		NDB_FREE(result);

	return NULL;
}

/*
 * ndb_json_parse_array_free - Free array parsed by ndb_json_parse_array
 *
 * Frees all memory allocated by ndb_json_parse_array, including the array
 * itself and all individual string elements.
 *
 * Parameters:
 *   array - Array of strings to free (from ndb_json_parse_array)
 *   count - Number of elements in the array
 *
 * Notes:
 *   Safe to call with NULL array pointer. This function should be called
 *   to free arrays returned by ndb_json_parse_array.
 */
void
ndb_json_parse_array_free(char **array, int count)
{
	int			i;

	if (array == NULL)
		return;

	for (i = 0; i < count; i++)
	{
		if (array[i] != NULL)
			NDB_FREE(array[i]);
	}

	NDB_FREE(array);
}

/*
 * ndb_json_parse_float_array - Parse JSON array of floats
 *
 * Parses a JSON array string and converts all elements to float values.
 * Uses ndb_json_parse_array internally to extract string values, then
 * converts each to a float using strtof.
 *
 * Parameters:
 *   json_str - Input JSON array string
 *   count - Output pointer to receive the number of elements parsed
 *
 * Returns:
 *   Array of float values, NULL if parsing fails, input is NULL, or
 *   no elements found. Invalid numbers are set to 0.0.
 *
 * Notes:
 *   Memory for the array is allocated in CurrentMemoryContext. Caller is
 *   responsible for freeing using pfree or NDB_FREE.
 */
float *
ndb_json_parse_float_array(const char *json_str, int *count)
{
	char	  **str_array = NULL;
	int			str_count = 0;
	int			i;
	char	   *endptr;

	NDB_DECLARE(float *, result);

	if (json_str == NULL || count == NULL)
		return NULL;

	*count = 0;

	str_array = ndb_json_parse_array(json_str, &str_count);
	if (str_array == NULL || str_count == 0)
		return NULL;

	NDB_ALLOC(result, float, str_count);

	for (i = 0; i < str_count; i++)
	{
		errno = 0;
		result[i] = strtof(str_array[i], &endptr);
		if (endptr == str_array[i] || errno != 0)
		{
			result[i] = 0.0f;
		}
	}

	ndb_json_parse_array_free(str_array, str_count);

	*count = str_count;
	return result;
}

/*
 * ndb_json_parse_int_array - Parse JSON array of integers
 *
 * Parses a JSON array string and converts all elements to integer values.
 * Uses ndb_json_parse_array internally to extract string values, then
 * converts each to an integer using strtol.
 *
 * Parameters:
 *   json_str - Input JSON array string
 *   count - Output pointer to receive the number of elements parsed
 *
 * Returns:
 *   Array of integer values, NULL if parsing fails, input is NULL, or
 *   no elements found. Invalid numbers are set to 0.
 *
 * Notes:
 *   Memory for the array is allocated in CurrentMemoryContext. Caller is
 *   responsible for freeing using pfree or NDB_FREE.
 */
int *
ndb_json_parse_int_array(const char *json_str, int *count)
{
	char	  **str_array = NULL;
	int			str_count = 0;
	int			i;
	char	   *endptr;

	NDB_DECLARE(int *, result);

	if (json_str == NULL || count == NULL)
		return NULL;

	*count = 0;

	str_array = ndb_json_parse_array(json_str, &str_count);
	if (str_array == NULL || str_count == 0)
		return NULL;

	NDB_ALLOC(result, int, str_count);

	for (i = 0; i < str_count; i++)
	{
		errno = 0;
		result[i] = (int) strtol(str_array[i], &endptr, 10);
		if (endptr == str_array[i] || errno != 0)
		{
			result[i] = 0;
		}
	}

	ndb_json_parse_array_free(str_array, str_count);

	*count = str_count;
	return result;
}

/*
 * ndb_json_validate - Validate JSON string syntax
 */
bool
ndb_json_validate(const char *json_str)
{
	volatile	Jsonb *jsonb = NULL;
	volatile bool result = false;

	if (json_str == NULL || json_str[0] == '\0')
		return false;

	PG_TRY();
	{
		jsonb = ndb_jsonb_in_cstring(json_str);
		if (jsonb != NULL)
		{
			result = true;
			if (jsonb != NULL)
			{
				Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

				NDB_FREE(jsonb_ptr);
				jsonb = NULL;
			}
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
		result = false;
	}
	PG_END_TRY();

	return (bool) result;
}

/*
 * ndb_json_is_empty - Check if JSON string is empty object/array
 */
bool
ndb_json_is_empty(const char *json_str)
{
	if (json_str == NULL)
		return true;

	if (strcmp(json_str, "{}") == 0)
		return true;

	if (strcmp(json_str, "[]") == 0)
		return true;

	{
		const char *p = json_str;

		while (*p && isspace((unsigned char) *p))
			p++;

		if (*p == '{')
		{
			p++;
			while (*p && isspace((unsigned char) *p))
				p++;
			if (*p == '}')
				return true;
		}
		else if (*p == '[')
		{
			p++;
			while (*p && isspace((unsigned char) *p))
				p++;
			if (*p == ']')
				return true;
		}
	}

	return false;
}

/*
 * ndb_json_strip_whitespace - Remove unnecessary whitespace from JSON
 */
char *
ndb_json_strip_whitespace(const char *json_str)
{
	volatile	Jsonb *jsonb = NULL;
	char	   *result = NULL;

	if (json_str == NULL)
		return NULL;

	PG_TRY();
	{
		jsonb = ndb_jsonb_in_cstring(json_str);
		if (jsonb != NULL)
		{
			result = ndb_jsonb_out_cstring((Jsonb *) jsonb);
			if (jsonb != NULL)
			{
				Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

				NDB_FREE(jsonb_ptr);
				jsonb = NULL;
			}
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
		result = pstrdup(json_str);
	}
	PG_END_TRY();

	return result;
}

/*
 * ndb_json_parse_object - Parse JSON object into key-value pairs
 *
 * Parses a JSON object string and extracts all key-value pairs into an array
 * of NdbJsonParseResult structures. Each structure contains the key, value
 * type, and value representation.
 *
 * Parameters:
 *   json_str - Input JSON object string
 *   count - Output pointer to receive the number of key-value pairs parsed
 *
 * Returns:
 *   Array of NdbJsonParseResult structures, NULL if parsing fails, input is
 *   NULL, or no pairs found. The count is set to the number of pairs.
 *
 * Notes:
 *   Memory for the array and all key/value strings is allocated in
 *   CurrentMemoryContext. Caller is responsible for freeing using
 *   ndb_json_parse_object_free. The function uses JSONB parsing internally
 *   for robustness.
 */
NdbJsonParseResult *
ndb_json_parse_object(const char *json_str, int *count)
{
	volatile	Jsonb *jsonb = NULL;
	volatile NdbJsonParseResult *result = NULL;
	volatile int n = 0;
	int			capacity = 16;

	if (json_str == NULL || count == NULL)
		return NULL;

	*count = 0;

	PG_TRY();
	{
		jsonb = ndb_jsonb_in_cstring(json_str);
		if (jsonb != NULL)
		{
			JsonbIterator *it;
			JsonbValue	v;
			JsonbIteratorToken type;
			char	   *current_key = NULL;

			NDB_DECLARE(NdbJsonParseResult *, result);
			NDB_ALLOC(result, NdbJsonParseResult, capacity);

			it = JsonbIteratorInit(&((Jsonb *) jsonb)->root);
			while ((type = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
			{
				if (type == WJB_KEY && v.type == jbvString)
				{
					current_key = pnstrdup(v.val.string.val, v.val.string.len);
				}
				else if (type == WJB_VALUE && current_key != NULL)
				{
					if ((int) n >= (int) capacity)
					{
						capacity *= 2;
						result = (NdbJsonParseResult *) repalloc((NdbJsonParseResult *) result, sizeof(NdbJsonParseResult) * (int) capacity);
					}

					((NdbJsonParseResult *) result)[(int) n].key = current_key;
					current_key = NULL;

					switch (v.type)
					{
						case jbvString:
							((NdbJsonParseResult *) result)[(int) n].value_type = 0;
							((NdbJsonParseResult *) result)[(int) n].value = pnstrdup(v.val.string.val, v.val.string.len);
							break;
						case jbvNumeric:
							{
								Numeric		num = v.val.numeric;
								double		num_val = DatumGetFloat8(DirectFunctionCall1(numeric_float8,
																						 NumericGetDatum(num)));

								((NdbJsonParseResult *) result)[(int) n].value_type = 1;
								((NdbJsonParseResult *) result)[(int) n].num_value = num_val;
								((NdbJsonParseResult *) result)[(int) n].value = psprintf("%g", num_val);
							}
							break;
						case jbvBool:
							((NdbJsonParseResult *) result)[(int) n].value_type = 2;
							((NdbJsonParseResult *) result)[(int) n].bool_value = v.val.boolean;
							((NdbJsonParseResult *) result)[(int) n].value = pstrdup(v.val.boolean ? "true" : "false");
							break;
						case jbvNull:
							((NdbJsonParseResult *) result)[(int) n].value_type = 3;
							((NdbJsonParseResult *) result)[(int) n].value = pstrdup("null");
							break;
						default:
							{
								Jsonb	   *value_jsonb = JsonbValueToJsonb(&v);
								text	   *value_text = NULL;
								char	   *value_str = NULL;

								if (value_jsonb != NULL)
								{
									value_text = ndb_jsonb_out(value_jsonb);
									if (value_text != NULL)
									{
										value_str = text_to_cstring(value_text);
										((NdbJsonParseResult *) result)[(int) n].value_type = (v.type == jbvBinary) ? 4 : 5;
										((NdbJsonParseResult *) result)[(int) n].value = value_str;
										NDB_FREE(value_text);
									}
									NDB_FREE(value_jsonb);
								}
								if (value_str == NULL)
								{
									((NdbJsonParseResult *) result)[(int) n].value_type = 0;
									((NdbJsonParseResult *) result)[(int) n].value = pstrdup("");
								}
							}
							break;
					}
					n++;
				}
			}
		}
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		if (jsonb != NULL)
		{
			Jsonb	   *jsonb_ptr = (Jsonb *) jsonb;

			NDB_FREE(jsonb_ptr);
			jsonb = NULL;
		}
		if (result != NULL)
		{
			int			i;

			for (i = 0; i < (int) n; i++)
			{
				if (((NdbJsonParseResult *) result)[i].key != NULL)
					NDB_FREE(((NdbJsonParseResult *) result)[i].key);
				if (((NdbJsonParseResult *) result)[i].value != NULL)
					NDB_FREE(((NdbJsonParseResult *) result)[i].value);
			}
			{
				NdbJsonParseResult *result_ptr = (NdbJsonParseResult *) result;

				NDB_FREE(result_ptr);
				result = NULL;
			}
			result = NULL;
		}
	}
	PG_END_TRY();

	if (n > 0)
	{
		*count = n;
		return (NdbJsonParseResult *) result;
	}

	if (result != NULL)
	{
		NdbJsonParseResult *result_ptr = (NdbJsonParseResult *) result;

		NDB_FREE(result_ptr);
	}

	return NULL;
}

/*
 * ndb_json_parse_object_free - Free array parsed by ndb_json_parse_object
 */
void
ndb_json_parse_object_free(NdbJsonParseResult *arr, int count)
{
	int			i;

	if (arr == NULL)
		return;

	for (i = 0; i < count; i++)
	{
		if (arr[i].key != NULL)
			NDB_FREE(arr[i].key);
		if (arr[i].value != NULL)
			NDB_FREE(arr[i].value);
	}

	NDB_FREE(arr);
}

/*
 * ndb_jsonb_build_object - Build JSONB object from key-value pairs
 */
Jsonb *
ndb_jsonb_build_object(const char *key1, const char *value1,...)
{
	StringInfoData buf;
	va_list		args;
	const char *key;
	const char *value;
	bool		first = true;
	char	   *json_str;
	Jsonb	   *result = NULL;

	if (key1 == NULL)
		return NULL;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '{');

	va_start(args, value1);

	key = key1;
	value = value1;

	while (key != NULL)
	{
		if (!first)
			appendStringInfoChar(&buf, ',');
		first = false;

		ndb_json_quote_string_buf(&buf, key);
		appendStringInfoChar(&buf, ':');
		ndb_json_quote_string_buf(&buf, value);

		key = va_arg(args, const char *);
		if (key != NULL)
			value = va_arg(args, const char *);
	}

	va_end(args);

	appendStringInfoChar(&buf, '}');
	json_str = buf.data;

	result = ndb_jsonb_in_cstring(json_str);

	NDB_FREE(buf.data);

	return result;
}

/*
 * ndb_jsonb_build_array - Build JSONB array from values
 */
Jsonb *
ndb_jsonb_build_array(const char *value1,...)
{
	StringInfoData buf;
	va_list		args;
	const char *value;
	bool		first = true;
	char	   *json_str;
	Jsonb	   *result = NULL;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '[');

	va_start(args, value1);

	value = value1;

	while (value != NULL)
	{
		if (!first)
			appendStringInfoChar(&buf, ',');
		first = false;

		ndb_json_quote_string_buf(&buf, value);

		value = va_arg(args, const char *);
	}

	va_end(args);

	appendStringInfoChar(&buf, ']');
	json_str = buf.data;

	result = ndb_jsonb_in_cstring(json_str);

	NDB_FREE(buf.data);

	return result;
}
