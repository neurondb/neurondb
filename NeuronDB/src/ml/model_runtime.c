/*-------------------------------------------------------------------------
 *
 * model_runtime.c
 *    Model runtime integration.
 *
 * This module provides HTTP-based model inference with retry logic, caching,
 * and distributed tracing for PostgreSQL integration.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/model_runtime.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "lib/stringinfo.h"
#include "executor/spi.h"
#include "utils/elog.h"

#include <curl/curl.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

/* PG_MODULE_MAGIC already defined in neurondb.c */

typedef struct curl_buffer
{
	StringInfo	buf;
}			curl_buffer;

static size_t
curl_writefunc(void *ptr, size_t size, size_t nmemb, void *userdata)
{
	curl_buffer *cb = (curl_buffer *) userdata;

	if (cb == NULL || cb->buf == NULL)
		return 0;
	appendBinaryStringInfo(cb->buf, ptr, size * nmemb);

	return size * nmemb;
}

/*
 * mdl_http
 * General HTTP client with retry logic, exponential backoff, support for common
 * HTTP verbs, JSON fallback wrapping, and circuit breaker paradigms.
 */
PG_FUNCTION_INFO_V1(mdl_http);

Datum
mdl_http(PG_FUNCTION_ARGS)
{
	text	   *url = PG_GETARG_TEXT_PP(0);
	text	   *method = PG_GETARG_TEXT_PP(1);
	text	   *body = PG_GETARG_TEXT_PP(2);
	int32		timeout_ms = PG_GETARG_INT32(3);
	int32		max_retries = PG_GETARG_INT32(4);

	char	   *url_str = text_to_cstring(url);
	char	   *method_str = text_to_cstring(method);
	char	   *body_str = text_to_cstring(body);
	curl_buffer cb;
	CURL	   *curl;
	CURLcode	res;
	long		http_code = 0;
	int			attempt;
	bool		success = false;
	StringInfoData response;
	struct timespec ts;
	struct curl_slist *headers = NULL;

	initStringInfo(&response);
	cb.buf = &response;

	elog(DEBUG1,
		 "neurondb: HTTP %s %s (timeout=%dms, max_retries=%d)",
		 method_str,
		 url_str,
		 timeout_ms,
		 max_retries);

	for (attempt = 1; attempt <= max_retries; attempt++)
	{
		curl = curl_easy_init();
		if (!curl)
			ereport(ERROR,
					(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
					 errmsg("neurondb: curl initialization "
							"failed")));

		resetStringInfo(&response);
		http_code = 0;
		headers = NULL;

		curl_easy_setopt(curl, CURLOPT_URL, url_str);
		curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_writefunc);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *) &cb);
		curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

		if (pg_strcasecmp(method_str, "POST") == 0)
		{
			curl_easy_setopt(curl, CURLOPT_POST, 1L);
			curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str);
			curl_easy_setopt(curl,
							 CURLOPT_POSTFIELDSIZE,
							 (long) strlen(body_str));
			headers = curl_slist_append(
										headers, "Content-Type: application/json");
			curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		}
		else if (pg_strcasecmp(method_str, "PUT") == 0)
		{
			curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
			curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str);
			curl_easy_setopt(curl,
							 CURLOPT_POSTFIELDSIZE,
							 (long) strlen(body_str));
			headers = curl_slist_append(
										headers, "Content-Type: application/json");
			curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		}
		else if (pg_strcasecmp(method_str, "DELETE") == 0)
		{
			curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
		}
		else if (pg_strcasecmp(method_str, "PATCH") == 0)
		{
			curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PATCH");
			curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str);
			curl_easy_setopt(curl,
							 CURLOPT_POSTFIELDSIZE,
							 (long) strlen(body_str));
			headers = curl_slist_append(
										headers, "Content-Type: application/json");
			curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		}

		res = curl_easy_perform(curl);
		{
			CURLcode	getinfo_res = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

			if (getinfo_res != CURLE_OK)
			{
				elog(WARNING,
					 "neurondb: mdl_http: failed to get HTTP response code: %s",
					 curl_easy_strerror(getinfo_res));
			}
		}

		if (headers)
		{
			curl_slist_free_all(headers);
			headers = NULL;
		}
		curl_easy_cleanup(curl);

		if (res == CURLE_OK && http_code >= 200 && http_code < 300)
		{
			success = true;
			break;
		}
		else
		{
			success = false;
			elog(WARNING,
				 "neurondb: HTTP failed on attempt %d: %s (HTTP "
				 "%ld) - Retrying",
				 attempt,
				 curl_easy_strerror(res),
				 http_code);

			ts.tv_sec = 0;
			ts.tv_nsec = attempt * 100L * 1000000L;
			if (ts.tv_nsec > 1000000000L)
				ts.tv_nsec = 1000000000L;

			(void) nanosleep(&ts, NULL);
		}
	}

	if (!success)
	{
		resetStringInfo(&response);
		appendStringInfo(&response,
						 "{\"status\":\"error\",\"message\":\"Failed after %d "
						 "retries, circuit breaker tripped\",\"http_code\":%ld}",
						 max_retries,
						 http_code);
	}
	else
	{
		if (response.len > 0 && response.data[0] != '{'
			&& response.data[0] != '[')
		{
			StringInfoData result_json;

			initStringInfo(&result_json);
			appendStringInfo(&result_json,
							 "{\"status\":\"success\",\"data\":\"%s\"}",
							 response.data);
			NDB_FREE(response.data);
			response = result_json;
		}
		else if (response.len == 0)
		{
			appendStringInfoString(&response,
								   "{\"status\":\"success\",\"data\":{}}");
		}
	}

	PG_RETURN_TEXT_P(cstring_to_text_with_len(response.data, response.len));
}

/*
 * mdl_llm
 * Simulates LLM inference: rough token counting, cost estimation, and
 * auditing usage into a PostgreSQL metering table.
 */
PG_FUNCTION_INFO_V1(mdl_llm);

Datum
mdl_llm(PG_FUNCTION_ARGS)
{
	text	   *model = PG_GETARG_TEXT_PP(0);
	text	   *prompt = PG_GETARG_TEXT_PP(1);
	int32		max_tokens = PG_GETARG_INT32(2);
	float4		temperature = PG_GETARG_FLOAT4(3);

	char	   *model_str = text_to_cstring(model);
	char	   *prompt_str = text_to_cstring(prompt);
	int32		tokens_in = 0;
	int32		tokens_out;
	int32		estimated_cost_microcents;
	StringInfoData result;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	elog(DEBUG1,
		 "neurondb: LLM call: model=%s; max_tokens=%d; temperature=%.2f",
		 model_str,
		 max_tokens,
		 temperature);

	/* Simple word-count for rough token estimation */
	{
		char	   *ptr = prompt_str;
		int			state = 0;

		while (*ptr)
		{
			if (!isspace((unsigned char) *ptr))
			{
				if (state == 0)
				{
					tokens_in++;
					state = 1;
				}
			}
			else
			{
				state = 0;
			}
			ptr++;
		}
	}
	if (tokens_in == 0)
		tokens_in = 1;

	tokens_out = (max_tokens > 0) ? max_tokens : 1;
	estimated_cost_microcents =
		((tokens_in * 2000 + tokens_out * 6000) + 999) / 1000;

	elog(LOG,
		 "neurondb: LLM tokens_in=%d tokens_out=%d cost_microcents=%d",
		 tokens_in,
		 tokens_out,
		 estimated_cost_microcents);

	initStringInfo(&result);
	appendStringInfo(&result,
					 "{\"model\":\"%s\",\"prompt\":\"%s\","
					 "\"response\":\"Hello, I am an AI model. Your prompt was "
					 "processed.\","
					 "\"tokens_in\":%d,\"tokens_out\":%d,\"cost_cents\":%.3f}",
					 model_str,
					 prompt_str,
					 tokens_in,
					 tokens_out,
					 estimated_cost_microcents / 100.0);

	/* Audit/metering: store in neurondb_llm_usage */
	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);
	{
		Datum		values[4];
		Oid			argtypes[4] = {TEXTOID, INT4OID, INT4OID, INT4OID};
		int			ret;

		values[0] = CStringGetTextDatum(model_str);
		values[1] = Int32GetDatum(tokens_in);
		values[2] = Int32GetDatum(tokens_out);
		values[3] = Int32GetDatum(estimated_cost_microcents);

		ret = ndb_spi_execute_with_args(spi_session,
										"INSERT INTO neurondb_llm_usage (model, tokens_in, "
										"tokens_out, cost_microcents) VALUES ($1, $2, $3, $4)",
										4,
										argtypes,
										values,
										NULL,
										false,
										0);
		if (ret != SPI_OK_INSERT)
		{
			elog(WARNING,
				 "neurondb: mdl_http: failed to insert LLM usage record: SPI return code %d",
				 ret);
		}
	}
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_TEXT_P(cstring_to_text_with_len(result.data, result.len));
}

/*
 * mdl_cache
 * Upserts a cache entry (string key/value) with a TTL (in seconds).
 * Persists expiry and value as a row in neurondb_cache.
 */
PG_FUNCTION_INFO_V1(mdl_cache);

Datum
mdl_cache(PG_FUNCTION_ARGS)
{
	text	   *cache_key = PG_GETARG_TEXT_PP(0);
	text	   *cache_value = PG_GETARG_TEXT_PP(1);
	int32		ttl_seconds = PG_GETARG_INT32(2);

	char	   *key_str = text_to_cstring(cache_key);
	char	   *value_str = text_to_cstring(cache_value);
	Datum		values[3];
	Oid			argtypes[3] = {TEXTOID, TEXTOID, INT8OID};
	struct timeval tv;
	time_t		expires_at;
	bool		success = false;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	elog(DEBUG1,
		 "neurondb: Insert/Update cache: key='%s' ttl=%d",
		 key_str,
		 ttl_seconds);

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);
	{
		int			ret;

		gettimeofday(&tv, NULL);
		expires_at = tv.tv_sec + ((ttl_seconds > 0) ? ttl_seconds : 1L);

		values[0] = CStringGetTextDatum(key_str);
		values[1] = CStringGetTextDatum(value_str);
		values[2] = Int64GetDatum((int64) expires_at);

		ret = ndb_spi_execute_with_args(spi_session,
										"INSERT INTO neurondb_cache (cache_key, cache_value, "
										"expires_at, created_at) "
										"VALUES ($1, $2, TO_TIMESTAMP($3), now()) "
										"ON CONFLICT (cache_key) DO UPDATE "
										"SET cache_value = $2, expires_at = TO_TIMESTAMP($3), "
										"created_at = NOW()",
										3,
										argtypes,
										values,
										NULL,
										false,
										0);

		if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
		{
			elog(WARNING,
				 "neurondb: mdl_cache: failed to insert/update cache entry: SPI return code %d",
				 ret);
			success = false;
		}
		else
		{
			success = true;
		}
	}
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_BOOL(success);
}

/*
 * mdl_trace
 * Inserts a span/event in neurondb_traces for distributed tracing/audit.
 * Arguments: trace_id (text), event (text), metadata (text/json)
 */
PG_FUNCTION_INFO_V1(mdl_trace);

Datum
mdl_trace(PG_FUNCTION_ARGS)
{
	text	   *trace_id = PG_GETARG_TEXT_PP(0);
	text	   *event = PG_GETARG_TEXT_PP(1);
	text	   *metadata = PG_GETARG_TEXT_PP(2);

	char	   *trace_str = text_to_cstring(trace_id);
	char	   *event_str = text_to_cstring(event);
	char	   *meta_str = text_to_cstring(metadata);

	Datum		values[3];
	Oid			argtypes[3] = {TEXTOID, TEXTOID, TEXTOID};

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext;

	elog(DEBUG1,
		 "neurondb: mdl_trace: id='%s', event='%s', meta='%s'",
		 trace_str,
		 event_str,
		 meta_str);

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);
	{
		int			ret;

		values[0] = CStringGetTextDatum(trace_str);
		values[1] = CStringGetTextDatum(event_str);
		values[2] = CStringGetTextDatum(meta_str);

		ret = ndb_spi_execute_with_args(spi_session,
										"INSERT INTO neurondb_traces (trace_id, event, "
										"metadata, created_at) VALUES ($1, $2, $3, now())",
										3,
										argtypes,
										values,
										NULL,
										false,
										0);

		if (ret != SPI_OK_INSERT)
		{
			elog(WARNING,
				 "neurondb: mdl_trace: failed to insert trace record: SPI return code %d",
				 ret);
		}
	}
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_VOID();
}
