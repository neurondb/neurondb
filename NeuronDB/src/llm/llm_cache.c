/*-------------------------------------------------------------------------
 *
 * llm_cache.c
 *		LLM response cache with TTL and key-based lookup
 *
 * This cache is backed by the neurondb.llm_cache table.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *	  src/llm/llm_cache.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "executor/spi.h"
#include "miscadmin.h"
#include "access/xact.h"
#include "lib/stringinfo.h"
#include "parser/parse_func.h"
#include "neurondb_llm.h"
#include "neurondb.h"
#include "utils/timestamp.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "funcapi.h"
#include "utils/array.h"
#include "parser/parse_type.h"
#include "utils/lsyscache.h"
#include "catalog/pg_proc.h"
#include "nodes/makefuncs.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_constants.h"
#include "neurondb_json.h"

/* ndb_json_quote_string is now replaced by ndb_json_quote_string from neurondb_json.h */
/* Unused function - kept for reference */
static char * __attribute__((unused))
ndb_json_quote_string_OLD_REMOVED(const char *str)
{
	StringInfoData buf;
	const char *p;
	char *result;

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
				} else
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
 * Returns true if a non-stale cache entry is found for 'key' (and text value, if non-NULL), false otherwise.
 * If found and out_text is non-NULL, sets *out_text to a freshly allocated string with the "text" field's value.
 *
 * Note: All variable declarations appear before executable statements for pre-C99 compatibility.
 */
bool
ndb_llm_cache_lookup(const char *key, int max_age_seconds, char **out_text)
{
	bool hit = false;
	int max_age = max_age_seconds > 0 ? max_age_seconds : 600;
	int ret;
	NDB_DECLARE(NdbSpiSession *, session);
	MemoryContext oldcontext;

	Assert(key != NULL);
	Assert(max_age_seconds >= 0);

	oldcontext = CurrentMemoryContext;
	session = ndb_spi_session_begin(oldcontext, false);
	if (session == NULL)
		return false;

	/* Use parameterized query to avoid SQL injection and quoting issues */
	/* Safely extract text field with validation to prevent crashes on corrupted JSONB */
	/* Wrap in PG_TRY to handle any remaining JSONB parsing errors gracefully */
	PG_TRY();
	{
		ret = ndb_spi_execute_with_args(session,
				"SELECT CASE "
				"  WHEN value IS NULL THEN NULL "
				"  WHEN jsonb_typeof(value) != 'object' THEN NULL "
				"  WHEN NOT (value ? '" NDB_JSON_KEY_TEXT "') THEN NULL "
				"  ELSE value->>'" NDB_JSON_KEY_TEXT "' "
				"END AS text_value "
				"FROM " NDB_FQ_LLM_CACHE " "
				"WHERE key = $1 AND now() - created_at < make_interval(secs => $2)",
				2,
				(Oid[]) { TEXTOID, INT4OID },
				(Datum[]) { CStringGetTextDatum(key),
					Int32GetDatum(max_age) },
				NULL,
				true,
				0);

		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			text *spi_text = NULL;
			char *spi_text_cstr = NULL;

			spi_text = ndb_spi_get_text(session, 0, 1, oldcontext);
			if (spi_text != NULL)
			{
				spi_text_cstr = text_to_cstring(spi_text);
				NDB_FREE(spi_text);
				if (out_text)
				{
					*out_text = spi_text_cstr;
					hit = true;
				} else
				{
					NDB_FREE(spi_text_cstr);
				}
			}
		}
	}
	PG_CATCH();
	{
		/* If JSONB is corrupted beyond repair, log warning and return false */
		ndb_spi_session_end(&session);
		FlushErrorState();
		elog(WARNING,
			 "ndb_llm_cache_lookup: Failed to read cache entry for key '%s' (possibly corrupted JSONB)",
			 key);
		hit = false;
		PG_RE_THROW();
	}
	PG_END_TRY();

	ndb_spi_session_end(&session);
	return hit;
}

/*
 * Store a text result in the cache under the given key, replacing any existing entry.
 * Variable declarations have been moved before any executable statements for pre-C99 compatibility.
 */
void
ndb_llm_cache_store(const char *key, const char *text)
{
	Oid argtypes[2];
	Datum values[2];
	StringInfoData val;
	NDB_DECLARE(NdbSpiSession *, session);
	MemoryContext oldcontext;
	int ret;

	/* All declarations above */
	argtypes[0] = TEXTOID;
	argtypes[1] = JSONBOID;

	oldcontext = CurrentMemoryContext;
	session = ndb_spi_session_begin(oldcontext, false);
	if (session == NULL)
		return;

	initStringInfo(&val);

	/* Construct JSON with a single 'text' field. */
	appendStringInfo(&val, "{\"" NDB_JSON_KEY_TEXT "\":%s}", ndb_json_quote_string(text));
	values[0] = CStringGetTextDatum(key);
	/* JSONBOID column; supply as text datum, will be cast to jsonb in SQL. */
	values[1] = CStringGetTextDatum(val.data);

	ret = ndb_spi_execute_with_args(session,
			"INSERT INTO " NDB_FQ_LLM_CACHE "(key, "
			"value, created_at) "
			"VALUES($1, $2::jsonb, now()) "
			"ON CONFLICT (key) DO UPDATE SET "
			"value=EXCLUDED.value, created_at=now()",
			2,
			argtypes,
			values,
			NULL,
			false,
			0);
	if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
	{
		elog(WARNING,
			"neurondb: store_cache_entry: failed to insert/update cache entry: SPI return code %d",
			ret);
	}

	NDB_FREE(val.data);
	ndb_spi_session_end(&session);
}

/*
 * Example test function:
 *   SELECT ndb_llm_cache_test('somekey', 'testvalue', 100);
 *
 * Useful for verifying the detailed implementation.
 * All variable declarations now appear at the top for standards compliance.
 */
PG_FUNCTION_INFO_V1(ndb_llm_cache_test);
Datum
ndb_llm_cache_test(PG_FUNCTION_ARGS)
{
	text *key_in;
	text *val_in;
	int32 max_age;
	char *key;
	char *val;
	char *result;
	bool hit;
	text *out;

	/* All declarations above */
	result = NULL;

	key_in = PG_GETARG_TEXT_PP(0);
	val_in = PG_GETARG_TEXT_PP(1);
	max_age = PG_GETARG_INT32(2);

	key = text_to_cstring(key_in);
	val = text_to_cstring(val_in);

	ndb_llm_cache_store(key, val);

	hit = ndb_llm_cache_lookup(key, max_age, &result);

	if (hit && result)
	{
		out = cstring_to_text(result);
		NDB_FREE(result);
		PG_RETURN_TEXT_P(out);
	} else
		PG_RETURN_NULL();
}

/*
 * ndb_llm_cache_evict_lru
 *    Evict least recently used cache entries when cache exceeds max_size.
 *    Returns number of entries evicted.
 */
static int
ndb_llm_cache_evict_lru(int max_size)
{
	int			spi_ret;
	int			current_count = 0;
	int			evicted = 0;
	StringInfoData sql;
	NDB_DECLARE(NdbSpiSession *, session);
	MemoryContext oldcontext;
	int32		count_val;

	oldcontext = CurrentMemoryContext;
	session = ndb_spi_session_begin(oldcontext, false);
	if (session == NULL)
		return 0;

	/* Get current cache size */
	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT COUNT(*) FROM " NDB_FQ_LLM_CACHE);
	spi_ret = ndb_spi_execute(session, sql.data, true, 1);
	if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		if (ndb_spi_get_int32(session, 0, 1, &count_val))
			current_count = count_val;
	}
	NDB_FREE(sql.data);

	if (current_count <= max_size)
	{
		ndb_spi_session_end(&session);
		return 0;
	}

	/* Evict oldest entries */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"DELETE FROM " NDB_FQ_LLM_CACHE " "
		"WHERE key IN ("
		"  SELECT key FROM " NDB_FQ_LLM_CACHE " "
		"  ORDER BY created_at ASC "
		"  LIMIT %d"
		")",
		current_count - max_size);
	spi_ret = ndb_spi_execute(session, sql.data, false, 0);
	if (spi_ret == SPI_OK_DELETE)
		evicted = current_count - max_size;
	NDB_FREE(sql.data);

	ndb_spi_session_end(&session);
	return evicted;
}

/*
 * ndb_llm_cache_store_with_eviction
 *    Store cache entry with automatic eviction if cache exceeds max_size.
 */
void
ndb_llm_cache_store_with_eviction(const char *key, const char *text, int max_size)
{
	ndb_llm_cache_store(key, text);
	if (max_size > 0)
		ndb_llm_cache_evict_lru(max_size);
}

PG_FUNCTION_INFO_V1(ndb_llm_cache_stats);
/*
 * ndb_llm_cache_stats
 *    Return cache statistics as JSONB.
 */
Datum
ndb_llm_cache_stats(PG_FUNCTION_ARGS)
{
	StringInfoData jsonbuf;
	StringInfoData sql;
	int			spi_ret;
	int			total_entries = 0;
	int			stale_entries = 0;
	int			ttl_seconds = 0;
	TimestampTz oldest_entry = 0;
	TimestampTz newest_entry = 0;
	NDB_DECLARE(NdbSpiSession *, session);
	MemoryContext oldcontext;
	int32		count_val;

	oldcontext = CurrentMemoryContext;
	session = ndb_spi_session_begin(oldcontext, false);
	if (session == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("ndb_llm_cache_stats: failed to begin SPI session")));

	ttl_seconds = neurondb_llm_cache_ttl;

	/* Get total entries */
	initStringInfo(&sql);
	appendStringInfo(&sql, "SELECT COUNT(*) FROM " NDB_FQ_LLM_CACHE);
	spi_ret = ndb_spi_execute(session, sql.data, true, 1);
	if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		if (ndb_spi_get_int32(session, 0, 1, &count_val))
			total_entries = count_val;
	}
	NDB_FREE(sql.data);

	/* Get stale entries */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT COUNT(*) FROM " NDB_FQ_LLM_CACHE " "
		"WHERE now() - created_at >= make_interval(secs => %d)",
		ttl_seconds);
	spi_ret = ndb_spi_execute(session, sql.data, true, 1);
	if (spi_ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		if (ndb_spi_get_int32(session, 0, 1, &count_val))
			stale_entries = count_val;
	}
	NDB_FREE(sql.data);

	/* Get oldest and newest entry timestamps */
	initStringInfo(&sql);
	appendStringInfo(&sql,
		"SELECT MIN(created_at), MAX(created_at) FROM " NDB_FQ_LLM_CACHE);
	spi_ret = ndb_spi_execute(session, sql.data, true, 1);
	if (spi_ret == SPI_OK_SELECT && SPI_processed > 0 && SPI_tuptable != NULL)
	{
		bool		isnull1;
		bool		isnull2;
		Datum		d1;
		Datum		d2;

		d1 = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			1,
			&isnull1);
		d2 = SPI_getbinval(SPI_tuptable->vals[0],
			SPI_tuptable->tupdesc,
			2,
			&isnull2);
		if (!isnull1)
			oldest_entry = DatumGetTimestampTz(d1);
		if (!isnull2)
			newest_entry = DatumGetTimestampTz(d2);
	}
	NDB_FREE(sql.data);

	ndb_spi_session_end(&session);

	/* Build JSON result */
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
		"{\"total_entries\":%d,"
		"\"stale_entries\":%d,"
		"\"valid_entries\":%d,"
		"\"ttl_seconds\":%d",
		total_entries,
		stale_entries,
		total_entries - stale_entries,
		ttl_seconds);
	if (oldest_entry != 0)
	{
		char	   *oldest_str = NULL;

		oldest_str = DatumGetCString(DirectFunctionCall1(timestamptz_out, TimestampTzGetDatum(oldest_entry)));
		appendStringInfo(&jsonbuf, ",\"oldest_entry\":%s", ndb_json_quote_string(oldest_str));
		NDB_FREE(oldest_str);
	}
	if (newest_entry != 0)
	{
		char	   *newest_str = NULL;

		newest_str = DatumGetCString(DirectFunctionCall1(timestamptz_out, TimestampTzGetDatum(newest_entry)));
		appendStringInfo(&jsonbuf, ",\"newest_entry\":%s", ndb_json_quote_string(newest_str));
		NDB_FREE(newest_str);
	}
	appendStringInfoChar(&jsonbuf, '}');

	{
		Jsonb	   *result;
		
		result = ndb_jsonb_in_cstring(jsonbuf.data);
		NDB_FREE(jsonbuf.data);
		PG_RETURN_JSONB_P(result);
	}
}

PG_FUNCTION_INFO_V1(ndb_llm_cache_clear);
/*
 * ndb_llm_cache_clear
 *    Clear all cache entries or entries matching a pattern.
 *
 * pattern: TEXT (optional, NULL clears all)
 * Returns: INTEGER (number of entries cleared)
 */
Datum
ndb_llm_cache_clear(PG_FUNCTION_ARGS)
{
	text	   *pattern = NULL;
	char	   *pattern_str = NULL;
	StringInfoData sql;
	int			spi_ret;
	int			deleted = 0;
	NDB_DECLARE(NdbSpiSession *, session);
	MemoryContext oldcontext;

	if (PG_ARGISNULL(0))
		pattern = NULL;
	else
		pattern = PG_GETARG_TEXT_PP(0);

	oldcontext = CurrentMemoryContext;
	session = ndb_spi_session_begin(oldcontext, false);
	if (session == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("ndb_llm_cache_clear: failed to begin SPI session")));

	initStringInfo(&sql);
	if (pattern == NULL)
	{
		appendStringInfo(&sql, "DELETE FROM " NDB_FQ_LLM_CACHE);
	} else
	{
		pattern_str = text_to_cstring(pattern);
		appendStringInfo(&sql,
			"DELETE FROM " NDB_FQ_LLM_CACHE " "
			"WHERE key LIKE %s",
			ndb_json_quote_string(pattern_str));
		NDB_FREE(pattern_str);
	}

	spi_ret = ndb_spi_execute(session, sql.data, false, 0);
	if (spi_ret == SPI_OK_DELETE)
	{
		/* Get number of deleted rows */
		deleted = SPI_processed;
	}

	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);

	PG_RETURN_INT32(deleted);
}

PG_FUNCTION_INFO_V1(ndb_llm_cache_evict_stale);
/*
 * ndb_llm_cache_evict_stale
 *    Evict stale cache entries (older than TTL).
 *
 * Returns: INTEGER (number of entries evicted)
 */
Datum
ndb_llm_cache_evict_stale(PG_FUNCTION_ARGS)
{
	StringInfoData sql;
	int			spi_ret;
	int			deleted = 0;
	int			ttl_seconds = 0;
	NDB_DECLARE(NdbSpiSession *, session);
	MemoryContext oldcontext;

	ttl_seconds = neurondb_llm_cache_ttl;

	oldcontext = CurrentMemoryContext;
	session = ndb_spi_session_begin(oldcontext, false);
	if (session == NULL)
		ereport(ERROR,
			(errcode(ERRCODE_INTERNAL_ERROR),
			 errmsg("ndb_llm_cache_evict_stale: failed to begin SPI session")));

	initStringInfo(&sql);
	appendStringInfo(&sql,
		"DELETE FROM " NDB_FQ_LLM_CACHE " "
		"WHERE now() - created_at >= make_interval(secs => %d)",
		ttl_seconds);

	spi_ret = ndb_spi_execute(session, sql.data, false, 0);
	if (spi_ret == SPI_OK_DELETE)
		deleted = SPI_processed;

	NDB_FREE(sql.data);
	ndb_spi_session_end(&session);

	PG_RETURN_INT32(deleted);
}

PG_FUNCTION_INFO_V1(ndb_llm_cache_evict_size);
/*
 * ndb_llm_cache_evict_size
 *    Evict cache entries to maintain max_size limit (LRU eviction).
 *
 * max_size: INTEGER
 * Returns: INTEGER (number of entries evicted)
 */
Datum
ndb_llm_cache_evict_size(PG_FUNCTION_ARGS)
{
	int32		max_size;
	int			evicted = 0;

	max_size = PG_GETARG_INT32(0);
	if (max_size < 0)
		ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("ndb_llm_cache_evict_size: max_size must be non-negative")));

	evicted = ndb_llm_cache_evict_lru(max_size);

	PG_RETURN_INT32(evicted);
}

PG_FUNCTION_INFO_V1(ndb_llm_cache_warm);
/*
 * ndb_llm_cache_warm
 *    Pre-populate cache with embeddings for given texts and model.
 *
 * texts: TEXT[]
 * model: TEXT (optional)
 * Returns: INTEGER (number of entries cached)
 */
Datum
ndb_llm_cache_warm(PG_FUNCTION_ARGS)
{
	ArrayType  *texts_array;
	Datum	   *text_datums = NULL;
	bool	   *text_nulls = NULL;
	int			nitems = 0;
	int			i;
	int			cached = 0;
	text	   *model_text = NULL;
	char	   *model_str = NULL;

	texts_array = PG_GETARG_ARRAYTYPE_P(0);
	if (PG_ARGISNULL(1))
		model_text = NULL;
	else
		model_text = PG_GETARG_TEXT_PP(1);

	deconstruct_array(texts_array,
		TEXTOID,
		-1,
		false,
		'i',
		&text_datums,
		&text_nulls,
		&nitems);

	if (model_text != NULL)
		model_str = text_to_cstring(model_text);
	else
		model_str = pstrdup("sentence-transformers/all-MiniLM-L6-v2");

	for (i = 0; i < nitems; i++)
	{
		if (!text_nulls[i])
		{
			text	   *input_text = (text *) DatumGetPointer(text_datums[i]);
			char	   *input_str = NULL;
			char	   *cache_key = NULL;
			StringInfoData key_buf;
			Vector	   *embedding = NULL;
			StringInfoData cache_val;
			int			d;
			const char *p;
			uint32		hashval;

			input_str = text_to_cstring(input_text);

			/* Generate cache key */
			initStringInfo(&key_buf);
			appendStringInfo(&key_buf, "embed:");

			hashval = 0;
			for (p = input_str; *p; p++)
				hashval = (hashval * 31) + *p;
			appendStringInfo(&key_buf, "%u:", (unsigned int) hashval);
			appendStringInfoString(&key_buf, model_str);
			
			/* BULLETPROOF: Copy cache_key to current context before SPI calls */
			/* ndb_llm_cache_lookup/store use SPI which may change memory context */
			cache_key = pstrdup(key_buf.data);
			NDB_FREE(key_buf.data);  /* Free original StringInfo data */

			/* Check if already cached */
			{
				char	   *cached_text = NULL;
				bool		hit = false;

				hit = ndb_llm_cache_lookup(cache_key, neurondb_llm_cache_ttl, &cached_text);
				if (hit && cached_text != NULL)
				{
					NDB_FREE(cached_text);
					NDB_FREE(cache_key);
					NDB_FREE(input_str);
					continue;
				}
			}

			/* Generate embedding via FunctionCall */
			{
				Oid			embed_text_oid;
				FmgrInfo	flinfo;
				List	   *funcname;
				Oid			argtypes[2];

				funcname = list_make1(makeString("embed_text"));
				if (model_text != NULL)
				{
					argtypes[0] = TEXTOID;
					argtypes[1] = TEXTOID;
					embed_text_oid = LookupFuncName(funcname, 2, argtypes, false);
					if (OidIsValid(embed_text_oid))
					{
						fmgr_info(embed_text_oid, &flinfo);
						embedding = (Vector *) DatumGetPointer(
							FunctionCall2(&flinfo,
								PointerGetDatum(input_text),
								PointerGetDatum(model_text)));
					}
				}
				if (embedding == NULL)
				{
					argtypes[0] = TEXTOID;
					embed_text_oid = LookupFuncName(funcname, 1, argtypes, false);
					if (OidIsValid(embed_text_oid))
					{
						fmgr_info(embed_text_oid, &flinfo);
						embedding = (Vector *) DatumGetPointer(
							FunctionCall1(&flinfo,
								PointerGetDatum(input_text)));
					}
				}
			}

			if (embedding != NULL)
			{
				/* Store in cache */
				initStringInfo(&cache_val);
				appendStringInfo(&cache_val, "{%d", embedding->dim);
				for (d = 0; d < embedding->dim; d++)
					appendStringInfo(&cache_val, ",%.7f", embedding->data[d]);
				appendStringInfoChar(&cache_val, '}');

				ndb_llm_cache_store(cache_key, cache_val.data);
				cached++;

				NDB_FREE(cache_val.data);
			}

			NDB_FREE(cache_key);
			NDB_FREE(input_str);
		}
	}

	if (text_datums)
		NDB_FREE(text_datums);
	if (text_nulls)
		NDB_FREE(text_nulls);
	if (model_str)
		NDB_FREE(model_str);

	PG_RETURN_INT32(cached);
}
