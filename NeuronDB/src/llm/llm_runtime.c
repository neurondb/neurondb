#include "postgres.h"
#include "fmgr.h"
#include "utils/guc.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "executor/spi.h"
#include "access/xact.h"
#include "utils/array.h"
#include "funcapi.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "utils/timestamp.h"
#include "lib/stringinfo.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "openssl/sha.h"
#include "neurondb.h"
#include "neurondb_llm.h"
#include "neurondb_guc.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_onnx.h"
#include "neurondb_constants.h"
#include <ctype.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi_safe.h"
#include "neurondb_json.h"
#include "neurondb_spi.h"

extern ImageMetadata *ndb_validate_image(const unsigned char *data, size_t size, MemoryContext mctx);

typedef struct NdbLLMRateLimiter
{
	double		tokens;
	TimestampTz last_refill;
	int			capacity;		/* tokens per second */
}			NdbLLMRateLimiter;

static NdbLLMRateLimiter * llm_rl = NULL;
static LWLock * llm_rl_lock = NULL;

Size
neurondb_llm_shmem_size(void)
{
	return MAXALIGN(sizeof(NdbLLMRateLimiter));
}

void
neurondb_llm_shmem_init(void)
{
	bool		found = false;

	llm_rl = (NdbLLMRateLimiter *) ShmemInitStruct(
												   "neurondb_llm_rate", sizeof(NdbLLMRateLimiter), &found);
	llm_rl_lock = &(GetNamedLWLockTranche("neurondb_llm")[0].lock);
	if (!found)
	{
		llm_rl->tokens = 0.0;
		llm_rl->last_refill = GetCurrentTimestamp();
		llm_rl->capacity = neurondb_llm_rate_limiter_qps > 0
			? neurondb_llm_rate_limiter_qps
			: 5;
	}
}

static bool
llm_acquire_token(void)
{
	TimestampTz now = GetCurrentTimestamp();
	long		secs;
	int			usecs;
	double		elapsed;
	int			cap;

	if (llm_rl == NULL || llm_rl_lock == NULL)
		return true;

	LWLockAcquire(llm_rl_lock, LW_EXCLUSIVE);

	cap = neurondb_llm_rate_limiter_qps > 0 ? neurondb_llm_rate_limiter_qps
		: llm_rl->capacity;

	TimestampDifference(llm_rl->last_refill, now, &secs, &usecs);
	elapsed = (double) secs + ((double) usecs / 1000000.0);

	if (elapsed > 0.0)
	{
		llm_rl->tokens =
			Min((double) cap, llm_rl->tokens + elapsed * cap);
		llm_rl->last_refill = now;
		llm_rl->capacity = cap;
	}
	if (llm_rl->tokens >= 1.0)
	{
		llm_rl->tokens -= 1.0;
		LWLockRelease(llm_rl_lock);
		return true;
	}
	LWLockRelease(llm_rl_lock);
	return false;
}

static double
ndb_elapsed_ms(TimestampTz start, TimestampTz end)
{
	long		secs;
	int			usecs;

	TimestampDifference(start, end, &secs, &usecs);
	return ((double) secs * 1000.0) + ((double) usecs / 1000.0);
}


static void
compute_cache_key(StringInfo dst,
				  const char *provider,
				  const char *model,
				  const char *endpoint,
				  const char *payload)
{
	unsigned char hash[SHA256_DIGEST_LENGTH];
	StringInfoData src;
	int			i;

	initStringInfo(&src);
	appendStringInfo(&src,
					 "%s|%s|%s|%s",
					 provider ? provider : "",
					 model ? model : "",
					 endpoint ? endpoint : "",
					 payload ? payload : "");
	SHA256((unsigned char *) src.data, src.len, hash);

	initStringInfo(dst);
	for (i = 0; i < SHA256_DIGEST_LENGTH; i++)
		appendStringInfo(&dst[0], "%02x", hash[i]);
}

static bool
cache_lookup_text(const char *key, char **text_out)
{
	bool		hit = false;

	NDB_DECLARE(NdbSpiSession *, session);
	Oid			argtypes[2] = {TEXTOID, INT4OID};
	Datum		values[2];
	const char	nulls[2] = {'i', 'i'};
	int			ret;

	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		return false;

	values[0] = CStringGetTextDatum(key);
	values[1] = Int32GetDatum(neurondb_llm_cache_ttl);

	/*
	 * Safely extract text field with validation to prevent crashes on
	 * corrupted JSONB
	 */
	ret = ndb_spi_execute_with_args(session,
									"SELECT CASE "
									"  WHEN value IS NULL THEN NULL "
									"  WHEN jsonb_typeof(value) != 'object' THEN NULL "
									"  WHEN NOT (value ? 'text') THEN NULL "
									"  ELSE value->>'text' "
									"END AS text_value "
									"FROM neurondb.llm_cache WHERE key = "
									"$1 AND now() - created_at < make_interval(secs => $2)",
									2,
									argtypes,
									values,
									nulls,
									true,
									0);

	if (ret == SPI_OK_SELECT && SPI_processed == 1)
	{
		bool		isnull;
		Datum		d = SPI_getbinval(SPI_tuptable->vals[0],
									  SPI_tuptable->tupdesc,
									  1,
									  &isnull);

		if (!isnull)
		{
			*text_out = TextDatumGetCString(d);
			hit = true;
		}
	}
	ndb_spi_session_end(&session);
	return hit;
}

static void
cache_store_text(const char *key, const char *text)
{
	Oid			argtypes[2];
	Datum		values[2];
	StringInfoData val;

	NDB_DECLARE(NdbSpiSession *, session);
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		return;
	argtypes[0] = TEXTOID;
	argtypes[1] = JSONBOID;
	initStringInfo(&val);
	appendStringInfo(&val, "{\"text\":%s}", quote_literal_cstr(text));
	values[0] = CStringGetTextDatum(key);
	values[1] = CStringGetTextDatum(val.data);
	{
		int			ret = ndb_spi_execute_with_args(session,
													"INSERT INTO neurondb.llm_cache(key,value,created_at) "
													"VALUES($1,$2::jsonb,now()) "
													"ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, "
													"created_at=now()",
													2,
													argtypes,
													values,
													NULL,
													false,
													0);

		if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
		{
			elog(WARNING,
				 "neurondb: store_llm_cache: failed to insert/update cache entry: SPI return code %d",
				 ret);
		}
	}
	NDB_FREE(val.data);
	ndb_spi_session_end(&session);
}

static void
record_llm_stats(const char *model_name,
				 const char *operation,
				 bool success,
				 bool cache_hit,
				 int64 latency_ms,
				 int tokens_in,
				 int tokens_out,
				 const char *error_type)
{
	int			ret;
	Oid			argtypes[8];
	Datum		values[8];
	char		nulls[8];

	NDB_DECLARE(NdbSpiSession *, session);
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		return;

	ret = ndb_spi_execute(session, "SELECT 1 FROM pg_tables WHERE schemaname = "
						  "'neurondb' AND tablename = 'llm_stats'",
						  true,
						  0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		ndb_spi_session_end(&session);
		return;
	}

	argtypes[0] = TEXTOID;
	argtypes[1] = TEXTOID;
	argtypes[2] = BOOLOID;
	argtypes[3] = BOOLOID;
	argtypes[4] = INT8OID;
	argtypes[5] = INT4OID;
	argtypes[6] = INT4OID;
	argtypes[7] = TEXTOID;

	values[0] = CStringGetTextDatum(model_name ? model_name : "unknown");
	values[1] = CStringGetTextDatum(operation ? operation : "unknown");
	nulls[0] = ' ';
	nulls[1] = ' ';

	if (success)
	{
		values[2] = BoolGetDatum(true);
		values[3] = BoolGetDatum(cache_hit);
		values[4] = Int64GetDatum(latency_ms);
		values[5] = Int32GetDatum(tokens_in);
		values[6] = Int32GetDatum(tokens_out);
		values[7] = (Datum) 0;
		nulls[2] = ' ';
		nulls[3] = ' ';
		nulls[4] = ' ';
		nulls[5] = ' ';
		nulls[6] = ' ';
		nulls[7] = 'n';
	}
	else
	{
		values[2] = BoolGetDatum(false);
		values[3] = BoolGetDatum(false);
		values[4] = Int64GetDatum(latency_ms);
		values[5] = Int32GetDatum(tokens_in);
		values[6] = Int32GetDatum(tokens_out);
		values[7] = CStringGetTextDatum(
										error_type ? error_type : "unknown");
		nulls[2] = ' ';
		nulls[3] = ' ';
		nulls[4] = ' ';
		nulls[5] = ' ';
		nulls[6] = ' ';
		nulls[7] = ' ';
	}

	{
		int			stats_ret = ndb_spi_execute_with_args(session,
														  "INSERT INTO neurondb.llm_stats (model_name, "
														  "total_requests, successful_requests, failed_requests, "
														  "cache_hits, total_latency_ms, total_tokens, total_tokens_in, "
														  "total_tokens_out, last_request_at, last_updated) "
														  "VALUES ($1, 1, "
														  "  CASE WHEN $3 THEN 1 ELSE 0 END, "
														  "  CASE WHEN $3 THEN 0 ELSE 1 END, "
														  "  CASE WHEN $4 THEN 1 ELSE 0 END, "
														  "  $5, $6 + $7, $6, $7, now(), now()) "
														  "ON CONFLICT (model_name) DO UPDATE SET "
														  "  total_requests = llm_stats.total_requests + 1, "
														  "  successful_requests = "
														  "llm_stats.successful_requests + CASE WHEN $3 THEN 1 "
														  "ELSE 0 END, "
														  "  failed_requests = llm_stats.failed_requests + CASE "
														  "WHEN $3 THEN 0 ELSE 1 END, "
														  "  cache_hits = llm_stats.cache_hits + CASE WHEN $4 "
														  "THEN 1 ELSE 0 END, "
														  "  total_latency_ms = llm_stats.total_latency_ms + "
														  "$5, "
														  "  total_tokens = llm_stats.total_tokens + $6 + $7, "
														  "  total_tokens_in = llm_stats.total_tokens_in + $6, "
														  "  total_tokens_out = llm_stats.total_tokens_out + "
														  "$7, "
														  "  last_request_at = now(), "
														  "  last_updated = now()",
														  8,
														  argtypes,
														  values,
														  nulls,
														  false,
														  0);

		if (stats_ret != SPI_OK_INSERT && stats_ret != SPI_OK_UPDATE)
		{
			elog(WARNING,
				 "neurondb: record_llm_stats: failed to insert/update stats: SPI return code %d",
				 stats_ret);
		}
	}

	if (!success && error_type)
	{
		Oid			error_argtypes[4] = {TEXTOID, TEXTOID, TEXTOID, INT8OID};
		Datum		error_values[4];
		char		error_nulls[4] = {' ', ' ', ' ', ' '};

		error_values[0] = CStringGetTextDatum(
											  model_name ? model_name : "unknown");
		error_values[1] =
			CStringGetTextDatum(operation ? operation : "unknown");
		error_values[2] = CStringGetTextDatum(error_type);
		error_values[3] = Int64GetDatum(latency_ms);

		ret = ndb_spi_execute(session,
							  "SELECT 1 FROM pg_tables WHERE schemaname = 'neurondb' "
							  "AND tablename = 'llm_errors'",
							  true,
							  0);
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int			error_ret = ndb_spi_execute_with_args(session,
															  "INSERT INTO neurondb.llm_errors "
															  "(model_name, operation, error_type, "
															  "latency_ms, error_timestamp) "
															  "VALUES ($1, $2, $3, $4, now())",
															  4,
															  error_argtypes,
															  error_values,
															  error_nulls,
															  false,
															  0);

			if (error_ret != SPI_OK_INSERT)
			{
				elog(WARNING,
					 "neurondb: record_llm_stats: failed to insert error record: SPI return code %d",
					 error_ret);
			}
		}
	}

	ret = ndb_spi_execute(session, "SELECT 1 FROM pg_tables WHERE schemaname = "
						  "'neurondb' AND tablename = 'histograms'",
						  true,
						  0);
	if (ret == SPI_OK_SELECT && SPI_processed > 0)
	{
		int			bucket_start = ((int) (latency_ms / 10.0)) * 10;
		int			bucket_end = bucket_start + 10;
		Oid			hist_argtypes[4] = {
			TEXTOID, FLOAT4OID, FLOAT4OID, INT8OID
		};
		Datum		hist_values[4];
		char		hist_nulls[4] = {' ', ' ', ' ', ' '};
		StringInfoData metric_name;
		char	   *metric_name_str;

		initStringInfo(&metric_name);
		appendStringInfo(&metric_name,
						 "llm_%s_%s_latency_ms",
						 operation ? operation : "unknown",
						 model_name ? model_name : "unknown");
		metric_name_str = pstrdup(metric_name.data);

		hist_values[0] = CStringGetTextDatum(metric_name_str);
		hist_values[1] = Float4GetDatum((float) bucket_start);
		hist_values[2] = Float4GetDatum((float) bucket_end);
		hist_values[3] = Int64GetDatum(1);

		ndb_spi_execute_with_args(session,
								  "INSERT INTO neurondb.histograms "
								  "(metric_name, bucket_start, bucket_end, count, "
								  "last_updated) "
								  "VALUES ($1, $2, $3, $4, now()) "
								  "ON CONFLICT (metric_name, bucket_start) DO UPDATE SET "
								  "  count = histograms.count + $4, "
								  "  last_updated = now()",
								  4,
								  hist_argtypes,
								  hist_values,
								  hist_nulls,
								  false,
								  0);

		NDB_FREE(metric_name.data);
		NDB_FREE(metric_name_str);
	}

	ndb_spi_session_end(&session);
}

static void
fill_cfg(NdbLLMConfig *cfg)
{
	cfg->provider =
		neurondb_llm_provider ? neurondb_llm_provider : "huggingface";
	if (cfg->provider && (pg_strcasecmp(cfg->provider, "openai") == 0
						  || pg_strcasecmp(cfg->provider, "chatgpt") == 0))
	{
		cfg->endpoint = neurondb_llm_endpoint
			? neurondb_llm_endpoint
			: "https://api.openai.com";
		cfg->model = neurondb_llm_model
			? neurondb_llm_model
			: "gpt-3.5-turbo";
	}
	else
	{
		cfg->endpoint = neurondb_llm_endpoint
			? neurondb_llm_endpoint
			: "https://router.huggingface.co/hf-inference";
		cfg->model = neurondb_llm_model
			? neurondb_llm_model
			: "sentence-transformers/all-MiniLM-L6-v2";
	}
	cfg->api_key = neurondb_llm_api_key;
	cfg->timeout_ms = neurondb_llm_timeout_ms;
	cfg->prefer_gpu = NDB_SHOULD_TRY_GPU();
	cfg->require_gpu = false;
	if (cfg->provider != NULL
		&& (pg_strcasecmp(cfg->provider, "huggingface-local") == 0
			|| pg_strcasecmp(cfg->provider, "hf-local") == 0)
		&& !neurondb_llm_fail_open)
		cfg->require_gpu = true;
}

/*
 * ndb_llm_complete
 *    Calls the configured LLM to complete a prompt, with input params as JSONB string.
 *    Uses rate-limiter, in-memory/DB cache, and supports fail-open/fail-closed.
 */
PG_FUNCTION_INFO_V1(ndb_llm_complete);
Datum
ndb_llm_complete(PG_FUNCTION_ARGS)
{
	text	   *prompt_in = PG_GETARG_TEXT_PP(0);
	text	   *params_in = (PG_NARGS() > 1 && !PG_ARGISNULL(1))
		? PG_GETARG_TEXT_PP(1)
		: NULL;
	char	   *prompt = text_to_cstring(prompt_in);
	char	   *params = params_in ? text_to_cstring(params_in) : "{}";
	NdbLLMConfig cfg;
	NdbLLMResp	resp = {0};
	NdbLLMCallOptions call_opts;
	StringInfoData keysrc,
				keyhex;
	char	   *cached = NULL;
	char	   *model_from_params = NULL;
	int			rc;
	TimestampTz start_time;
	TimestampTz end_time;
	int64		latency_ms = 0;
	bool		cache_hit = false;
	bool		success = false;
	const char *error_type = NULL;
	int			tokens_in = 0;
	int			tokens_out = 0;

	start_time = GetCurrentTimestamp();
	fill_cfg(&cfg);
	
	/* Extract model from params JSON if provided (overrides GUC setting) */
	if (params && strlen(params) > 0 && strcmp(params, "{}") != 0)
	{
		char	   *raw_model = ndb_json_extract_string(params, "model");
		
		if (raw_model && strlen(raw_model) > 0)
		{
			/* Strip quotes if present (ndb_jsonb_out returns JSON representation) */
			size_t		len = strlen(raw_model);
			
			if (len >= 2 && raw_model[0] == '"' && raw_model[len - 1] == '"')
			{
				/* Allocate new string without quotes */
				model_from_params = palloc(len - 1);
				memcpy(model_from_params, raw_model + 1, len - 2);
				model_from_params[len - 2] = '\0';
				NDB_FREE(raw_model);
			}
			else
			{
				model_from_params = raw_model;
			}
			
			if (model_from_params && strlen(model_from_params) > 0)
			{
				/* Override model from params - note: model_from_params is allocated and will be freed later */
				cfg.model = model_from_params;
			}
			else
			{
				NDB_FREE(model_from_params);
				model_from_params = NULL;
			}
		}
		else if (raw_model)
		{
			NDB_FREE(raw_model);
		}
	}
	
	call_opts.task = "complete";
	call_opts.prefer_gpu = cfg.prefer_gpu;
	call_opts.require_gpu = cfg.require_gpu;
	call_opts.fail_open = neurondb_llm_fail_open;

	initStringInfo(&keysrc);
	appendStringInfo(&keysrc, "%s|%s|%s", prompt, params, cfg.model);
	compute_cache_key(
					  &keyhex, cfg.provider, cfg.model, cfg.endpoint, keysrc.data);

	if (cache_lookup_text(keyhex.data, &cached))
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64) ndb_elapsed_ms(start_time, end_time);
		cache_hit = true;
		success = true;
		{
			int32		token_length;
			int32	   *token_ids = neurondb_tokenize_with_model(
																 prompt, 2048, &token_length, cfg.model);

			if (token_ids && token_length > 0)
				tokens_in = token_length;
			else
				tokens_in =
					0;
			if (token_ids)
				NDB_FREE(token_ids);

#ifdef HAVE_ONNX_RUNTIME
			PG_TRY();
			{
				int32		output_token_length;
				int32	   *output_token_ids =
					neurondb_tokenize_with_model(cached,
												 2048,
												 &output_token_length,
												 cfg.model);

				if (output_token_ids && output_token_length > 0)
				{
					tokens_out = output_token_length;
				}
				else
				{
					if (cached && strlen(cached) > 0)
					{
						const char *ptr = cached;
						int			word_count = 0;
						int			in_word = 0;

						while (*ptr)
						{
							if (!isspace((
										  unsigned char) *ptr))
							{
								if (!in_word)
								{
									word_count++;
									in_word =
										1;
								}
							}
							else
							{
								in_word = 0;
							}
							ptr++;
						}
						tokens_out = word_count > 0
							? word_count
							: 1;
					}
					else
					{
						tokens_out = 0;
					}
				}
				if (output_token_ids)
					NDB_FREE(output_token_ids);
			}
			PG_CATCH();
			{
				EmitErrorReport();
				FlushErrorState();

				if (cached && strlen(cached) > 0)
				{
					const char *ptr = cached;
					int			word_count = 0;
					int			in_word = 0;

					while (*ptr)
					{
						if (!isspace((
									  unsigned char) *ptr))
						{
							if (!in_word)
							{
								word_count++;
								in_word = 1;
							}
						}
						else
						{
							in_word = 0;
						}
						ptr++;
					}
					tokens_out =
						word_count > 0 ? word_count : 1;
				}
				else
				{
					tokens_out = 0;
				}
			}
			PG_END_TRY();
#else
			if (cached && strlen(cached) > 0)
			{
				const char *ptr = cached;
				int			word_count = 0;
				int			in_word = 0;

				while (*ptr)
				{
					if (!isspace((unsigned char) *ptr))
					{
						if (!in_word)
						{
							word_count++;
							in_word = 1;
						}
					}
					else
					{
						in_word = 0;
					}
					ptr++;
				}
				tokens_out = word_count > 0 ? word_count : 1;
			}
			else
			{
				tokens_out = 0;
			}
#endif
		}
		record_llm_stats(cfg.model,
						 "complete",
						 success,
						 cache_hit,
						 latency_ms,
						 tokens_in,
						 tokens_out,
						 NULL);
		PG_RETURN_TEXT_P(cstring_to_text(cached));
	}

	if (cfg.endpoint && strncmp(cfg.endpoint, "mock://", 7) == 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Mock endpoints are not supported"),
				 errdetail("Use real API endpoints with valid API keys. Mock endpoints for testing are not allowed."),
				 errhint("Set neurondb.llm_endpoint to a valid API endpoint (e.g., https://api.openai.com or https://api-inference.huggingface.co)")));
	}

	/* Rate limit */
	if (!llm_acquire_token())
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64) ndb_elapsed_ms(start_time, end_time);
		cache_hit = false;
		success = false;
		error_type = "rate_limit";
		if (neurondb_llm_fail_open)
		{
			ereport(ERROR,
					(errmsg("neurondb: LLM rate limited - fail "
							"open")));
			record_llm_stats(cfg.model,
							 "complete",
							 success,
							 cache_hit,
							 latency_ms,
							 tokens_in,
							 tokens_out,
							 error_type);
			PG_RETURN_NULL();
		}
		record_llm_stats(cfg.model,
						 "complete",
						 success,
						 cache_hit,
						 latency_ms,
						 tokens_in,
						 tokens_out,
						 error_type);
		ereport(ERROR, (errmsg("neurondb: LLM rate limited")));
	}

	/* Route to configured provider (remote Hugging Face or local runtime) */
	rc = ndb_llm_route_complete(&cfg, &call_opts, prompt, params, &resp);
	end_time = GetCurrentTimestamp();
	latency_ms = (int64) ndb_elapsed_ms(start_time, end_time);
	tokens_in = resp.tokens_in;
	tokens_out = resp.tokens_out;

	if (rc != 0)
	{
		cache_hit = false;
		success = false;
		error_type = "provider_error";
		
		/* For HTTP errors (>= 400), always return error message to SQL user, even with fail_open */
		if (resp.http_status >= NDB_HTTP_STATUS_ERROR_MIN && resp.json && strlen(resp.json) > 0)
		{
			StringInfoData err_msg;
			char	   *error_text = NULL;
			
			/* Try to extract error message from JSON if it's in {"error":"..."} format */
			if (strncmp(resp.json, "{\"error\"", 8) == 0)
			{
				error_text = ndb_json_extract_string(resp.json, "error");
				/* Strip quotes if present (ndb_json_extract_string may return JSON representation) */
				if (error_text)
				{
					size_t		len = strlen(error_text);
					
					if (len >= 2 && error_text[0] == '"' && error_text[len - 1] == '"')
					{
						/* Remove quotes */
						char	   *unquoted = palloc(len - 1);
						
						memcpy(unquoted, error_text + 1, len - 2);
						unquoted[len - 2] = '\0';
						NDB_FREE(error_text);
						error_text = unquoted;
					}
				}
			}

			/* Build URL for error message */
			{
				StringInfoData url_str;

				initStringInfo(&url_str);
				if (cfg.endpoint && cfg.model)
				{
					/* Construct the URL that was called */
					if (strstr(cfg.endpoint, "router.huggingface.co") != NULL ||
						strstr(cfg.endpoint, "hf-inference") != NULL)
					{
						appendStringInfo(&url_str, "%s/models/%s", cfg.endpoint, cfg.model);
					}
					else
					{
					appendStringInfo(&url_str, "%s/models/%s", cfg.endpoint, cfg.model);
				}
			}
			else if (cfg.endpoint)
			{
				appendStringInfoString(&url_str, cfg.endpoint);
			}
			else
			{
				appendStringInfoString(&url_str, "unknown endpoint");
			}
			
			initStringInfo(&err_msg);
			if (error_text && strlen(error_text) > 0)
			{
				appendStringInfo(&err_msg, "[ERROR: HTTP %d] Model: '%s' | URL: %s | %s", 
								 resp.http_status, 
								 cfg.model ? cfg.model : "unknown",
								 url_str.data,
								 error_text);
				NDB_FREE(error_text);
			}
			else if (resp.json)
			{
				/* Use raw JSON if error extraction failed */
				appendStringInfo(&err_msg, "[ERROR: HTTP %d] Model: '%s' | URL: %s | %s", 
								 resp.http_status,
								 cfg.model ? cfg.model : "unknown",
								 url_str.data,
								 resp.json);
			}
			else
			{
				appendStringInfo(&err_msg, "[ERROR: HTTP %d] Model: '%s' | URL: %s | Provider error - check API key, endpoint, and model availability", 
								 resp.http_status,
								 cfg.model ? cfg.model : "unknown",
								 url_str.data);
				}
				NDB_FREE(url_str.data);
			}

			record_llm_stats(cfg.model,
							 "complete",
							 success,
							 cache_hit,
							 latency_ms,
							 tokens_in,
							 tokens_out,
							 error_type);
			
			PG_RETURN_TEXT_P(cstring_to_text(err_msg.data));
		}
		
		if (neurondb_llm_fail_open)
		{
			StringInfoData err_msg;
			
			record_llm_stats(cfg.model,
							 "complete",
							 success,
							 cache_hit,
							 latency_ms,
							 tokens_in,
							 tokens_out,
							 error_type);
			/* Return error message instead of NULL when fail_open is true */
			initStringInfo(&err_msg);
			if (resp.json && strlen(resp.json) > 0)
			{
				appendStringInfo(&err_msg, "[ERROR: HTTP %d] %s", resp.http_status, resp.json);
			}
			else
			{
				appendStringInfo(&err_msg, "[ERROR: HTTP %d] Provider error - check API key, endpoint, and model availability", resp.http_status);
			}
			
			PG_RETURN_TEXT_P(cstring_to_text(err_msg.data));
		}
		record_llm_stats(cfg.model,
						 "complete",
						 success,
						 cache_hit,
						 latency_ms,
						 tokens_in,
						 tokens_out,
						 error_type);
		if (resp.json && strlen(resp.json) > 0)
		{
			ereport(ERROR,
					(errmsg("neurondb: llm provider error (HTTP %d): %s", resp.http_status, resp.json)));
		}
		else
		{
			ereport(ERROR,
					(errmsg("neurondb: llm provider error (HTTP %d). Check API key, endpoint, and model availability", resp.http_status)));
		}
	}

	if (resp.text)
	{
		cache_store_text(keyhex.data, resp.text);
		cache_hit = false;
		success = true;
		error_type = NULL;
	}
	else
	{
		cache_hit = false;
		success = false;
		error_type = "empty_response";
		
		/* Always log the error for debugging, even if fail_open is true */
		if (resp.json && strlen(resp.json) > 0)
		{
			elog(WARNING,
				 "neurondb: llm returned empty response (HTTP %d): %s", resp.http_status, resp.json);
		}
		else
		{
			elog(WARNING,
				 "neurondb: llm returned empty response (HTTP %d). Check API key, endpoint, and model availability", resp.http_status);
		}
		
		if (!neurondb_llm_fail_open)
		{
			if (resp.json && strlen(resp.json) > 0)
			{
				ereport(ERROR,
						(errmsg("neurondb: llm returned empty response (HTTP %d): %s", resp.http_status, resp.json)));
			}
			else
			{
				ereport(ERROR,
						(errmsg("neurondb: llm returned empty response (HTTP %d). Check API key, endpoint, and model availability", resp.http_status)));
			}
		}
	}

	record_llm_stats(cfg.model,
					 "complete",
					 success,
					 cache_hit,
					 latency_ms,
					 tokens_in,
					 tokens_out,
					 error_type);
	
	/* If fail_open is true and we have an error, return error message instead of empty string */
	if (!resp.text && neurondb_llm_fail_open && (rc != 0 || error_type != NULL))
	{
		StringInfoData err_msg;
		
		initStringInfo(&err_msg);
		if (resp.json && strlen(resp.json) > 0)
		{
			appendStringInfo(&err_msg, "[ERROR: HTTP %d] %s", resp.http_status, resp.json);
		}
		else if (rc != 0)
		{
			appendStringInfo(&err_msg, "[ERROR: HTTP %d] Provider error - check API key, endpoint, and model availability", resp.http_status);
		}
		else
		{
			appendStringInfo(&err_msg, "[ERROR] Empty response - check API key, endpoint, and model availability");
		}
		
		PG_RETURN_TEXT_P(cstring_to_text(err_msg.data));
	}
	
	/* Clean up model_from_params if we allocated it */
	if (model_from_params)
		NDB_FREE(model_from_params);
	
	PG_RETURN_TEXT_P(cstring_to_text(resp.text ? resp.text : ""));
}

/*
 * ndb_llm_image_analyze
 *    Analyzes an image using vision-capable LLM (e.g., GPT-4 Vision).
 *    Returns text description/analysis of the image.
 */
PG_FUNCTION_INFO_V1(ndb_llm_image_analyze);
Datum
ndb_llm_image_analyze(PG_FUNCTION_ARGS)
{
	bytea	   *image_data = PG_GETARG_BYTEA_PP(0);
	text	   *prompt_in = (PG_NARGS() > 1 && !PG_ARGISNULL(1))
		? PG_GETARG_TEXT_PP(1)
		: NULL;
	text	   *params_in = (PG_NARGS() > 2 && !PG_ARGISNULL(2))
		? PG_GETARG_TEXT_PP(2)
		: NULL;
	text	   *model_in = (PG_NARGS() > 3 && !PG_ARGISNULL(3))
		? PG_GETARG_TEXT_PP(3)
		: NULL;

	char	   *prompt = prompt_in ? text_to_cstring(prompt_in) : NULL;
	char	   *params = params_in ? text_to_cstring(params_in) : "{}";
	char	   *model_str = model_in ? text_to_cstring(model_in) : NULL;
	NdbLLMConfig cfg;
	NdbLLMResp	resp = {0};
	NdbLLMCallOptions call_opts;
	StringInfoData keysrc,
				keyhex;
	char	   *cached = NULL;
	int			rc;
	TimestampTz start_time;
	TimestampTz end_time;
	int64		latency_ms = 0;
	bool		cache_hit = false;
	bool		success = false;
	const char *error_type = NULL;
	int			tokens_in = 0;
	int			tokens_out = 0;
	bytea	   *detoasted_image = NULL;
	size_t		image_size;
	const unsigned char *image_bytes;
	bool		need_free_detoasted = false;

	/* Validate image data */
	if (image_data == NULL || VARSIZE_ANY_EXHDR(image_data) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ndb_llm_image_analyze: image_data must not be NULL or empty")));

	/* Detoast and validate image format before processing */
	detoasted_image = (bytea *) PG_DETOAST_DATUM(PointerGetDatum(image_data));
	if (detoasted_image != image_data)
		need_free_detoasted = true;

	image_size = VARSIZE_ANY_EXHDR(detoasted_image);
	image_bytes = (const unsigned char *) VARDATA_ANY(detoasted_image);

	/* Validate image format and size using image utilities */
	{
		ImageMetadata *img_meta = ndb_validate_image(image_bytes, image_size, CurrentMemoryContext);

		if (!img_meta || !img_meta->is_valid)
		{
			const char *error_msg = img_meta && img_meta->error_msg
				? img_meta->error_msg
				: "Invalid image format or corrupted data";

			if (img_meta)
			{
				if (img_meta->mime_type)
					NDB_FREE(img_meta->mime_type);
				if (img_meta->error_msg)
					NDB_FREE(img_meta->error_msg);
				NDB_FREE(img_meta);
			}
			if (need_free_detoasted)
				NDB_FREE(detoasted_image);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("ndb_llm_image_analyze: %s", error_msg),
					 errdetail("Supported formats: JPEG, PNG, GIF, WebP, BMP. "
							   "Maximum size: 100MB. Minimum size: 100 bytes.")));
		}
		/* Free metadata after validation */
		if (img_meta)
		{
			if (img_meta->mime_type)
				NDB_FREE(img_meta->mime_type);
			if (img_meta->error_msg)
				NDB_FREE(img_meta->error_msg);
			NDB_FREE(img_meta);
		}
	}

	start_time = GetCurrentTimestamp();
	fill_cfg(&cfg);

	/* Override model if provided */
	if (model_str != NULL)
		cfg.model = model_str;

	/* Default model based on provider */
	if (cfg.provider == NULL ||
		pg_strcasecmp(cfg.provider, "huggingface") == 0 ||
		pg_strcasecmp(cfg.provider, "hf-http") == 0)
	{
		/* Default to BLIP-2 or InstructBLIP for HuggingFace */
		if (cfg.model == NULL && model_str == NULL)
			cfg.model = "Salesforce/blip-image-captioning-base";
	}
	else if (pg_strcasecmp(cfg.provider, "openai") == 0 ||
			 pg_strcasecmp(cfg.provider, "chatgpt") == 0)
	{
		/* Default to GPT-4o for OpenAI */
		if (cfg.model == NULL && model_str == NULL)
			cfg.model = "gpt-4o";
	}

	call_opts.task = "vision_complete";
	call_opts.prefer_gpu = cfg.prefer_gpu;
	call_opts.require_gpu = false;	/* Vision API is HTTP-based */
	call_opts.fail_open = neurondb_llm_fail_open;

	/* Build cache key */
	initStringInfo(&keysrc);
	appendStringInfo(&keysrc, "image:%zu|%s|%s|%s",
					 image_size,
					 prompt ? prompt : "default",
					 params,
					 cfg.model);
	compute_cache_key(&keyhex, cfg.provider, cfg.model, cfg.endpoint, keysrc.data);

	if (cache_lookup_text(keyhex.data, &cached))
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64) ndb_elapsed_ms(start_time, end_time);
		cache_hit = true;
		success = true;
		if (need_free_detoasted)
			NDB_FREE(detoasted_image);
		NDB_FREE(keysrc.data);
		NDB_FREE(keyhex.data);
		if (prompt)
			NDB_FREE(prompt);
		if (params)
			NDB_FREE(params);
		if (model_str)
			NDB_FREE(model_str);
		PG_RETURN_TEXT_P(cstring_to_text(cached));
	}

	/* Check rate limiter */
	if (!llm_acquire_token())
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64) ndb_elapsed_ms(start_time, end_time);
		cache_hit = false;
		success = false;
		error_type = "rate_limited";
		if (neurondb_llm_fail_open)
		{
			record_llm_stats(cfg.model,
							 "vision_complete",
							 success,
							 cache_hit,
							 latency_ms,
							 tokens_in,
							 tokens_out,
							 error_type);
			if (need_free_detoasted)
				NDB_FREE(detoasted_image);
			NDB_FREE(keysrc.data);
			NDB_FREE(keyhex.data);
			if (prompt)
				NDB_FREE(prompt);
			if (params)
				NDB_FREE(params);
			if (model_str)
				NDB_FREE(model_str);
			PG_RETURN_NULL();
		}
		record_llm_stats(cfg.model,
						 "vision_complete",
						 success,
						 cache_hit,
						 latency_ms,
						 tokens_in,
						 tokens_out,
						 error_type);
		if (need_free_detoasted)
			NDB_FREE(detoasted_image);
		NDB_FREE(keysrc.data);
		NDB_FREE(keyhex.data);
		if (prompt)
			NDB_FREE(prompt);
		if (params)
			NDB_FREE(params);
		if (model_str)
			NDB_FREE(model_str);
		ereport(ERROR, (errmsg("neurondb: LLM rate limited")));
	}

	/* Route vision request through configured provider */
	rc = ndb_llm_route_vision_complete(&cfg, &call_opts,
									   image_bytes, image_size, prompt, params, &resp);

	end_time = GetCurrentTimestamp();
	latency_ms = (int64) ndb_elapsed_ms(start_time, end_time);

	if (rc != 0 || resp.text == NULL)
	{
		cache_hit = false;
		success = false;
		error_type = "vision_failed";
		tokens_in = resp.tokens_in;
		tokens_out = resp.tokens_out;
		if (neurondb_llm_fail_open)
		{
			record_llm_stats(cfg.model,
							 "vision_complete",
							 success,
							 cache_hit,
							 latency_ms,
							 tokens_in,
							 tokens_out,
							 error_type);
			if (need_free_detoasted)
				NDB_FREE(detoasted_image);
			if (resp.text)
				NDB_FREE(resp.text);
			if (resp.json)
				NDB_FREE(resp.json);
			NDB_FREE(keysrc.data);
			NDB_FREE(keyhex.data);
			if (prompt)
				NDB_FREE(prompt);
			if (params)
				NDB_FREE(params);
			if (model_str)
				NDB_FREE(model_str);
			PG_RETURN_NULL();
		}
		record_llm_stats(cfg.model,
						 "vision_complete",
						 success,
						 cache_hit,
						 latency_ms,
						 tokens_in,
						 tokens_out,
						 error_type);
		if (need_free_detoasted)
			NDB_FREE(detoasted_image);
		if (resp.text)
			NDB_FREE(resp.text);
		if (resp.json)
			NDB_FREE(resp.json);
		NDB_FREE(keysrc.data);
		NDB_FREE(keyhex.data);
		if (prompt)
			NDB_FREE(prompt);
		if (params)
			NDB_FREE(params);
		if (model_str)
			NDB_FREE(model_str);
		ereport(ERROR, (errmsg("neurondb: vision API failed")));
	}

	success = true;
	tokens_in = resp.tokens_in;
	tokens_out = resp.tokens_out;

	/* Cache the result */
	if (resp.text && strlen(resp.text) > 0)
		ndb_llm_cache_store(keyhex.data, resp.text);

	record_llm_stats(cfg.model,
					 "vision_complete",
					 success,
					 cache_hit,
					 latency_ms,
					 tokens_in,
					 tokens_out,
					 NULL);

	if (need_free_detoasted)
		NDB_FREE(detoasted_image);
	if (resp.json)
		NDB_FREE(resp.json);
	NDB_FREE(keysrc.data);
	NDB_FREE(keyhex.data);
	if (prompt)
		NDB_FREE(prompt);
	if (params)
		NDB_FREE(params);
	if (model_str)
		NDB_FREE(model_str);

	PG_RETURN_TEXT_P(cstring_to_text(resp.text));
}

/* SQL: llm_embed(txt text, model text default null) RETURNS vector
 *
 * Calls LLM endpoint to embed input text, with optional model override.
 */
PG_FUNCTION_INFO_V1(ndb_llm_embed);
Datum
ndb_llm_embed(PG_FUNCTION_ARGS)
{
	text	   *txt_in = PG_GETARG_TEXT_PP(0);
	char	   *txt = text_to_cstring(txt_in);
	NdbLLMConfig cfg;
	float	   *vec = NULL;
	int			dim = 0;
	Vector	   *v;
	NdbLLMCallOptions call_opts;
	TimestampTz start_time;
	TimestampTz end_time;
	int64		latency_ms = 0;
	bool		cache_hit = false;
	bool		success = false;
	const char *error_type = NULL;
	int			tokens_in = 0;
	int			tokens_out = 0;

	start_time = GetCurrentTimestamp();
	fill_cfg(&cfg);
	if (PG_NARGS() > 1 && !PG_ARGISNULL(1))
		cfg.model = text_to_cstring(PG_GETARG_TEXT_PP(1));
	call_opts.task = "embed";
	call_opts.prefer_gpu = cfg.prefer_gpu;
	call_opts.require_gpu = cfg.require_gpu;
	call_opts.fail_open = neurondb_llm_fail_open;

	/* Mock endpoints are not supported */
	if (cfg.endpoint && strncmp(cfg.endpoint, "mock://", 7) == 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Mock endpoints are not supported"),
				 errdetail("Use real API endpoints with valid API keys. Mock endpoints for testing are not allowed."),
				 errhint("Set neurondb.llm_endpoint to a valid API endpoint (e.g., https://api.openai.com or https://api-inference.huggingface.co)")));
	}

	/* Rate limiting */
	if (!llm_acquire_token())
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64) ndb_elapsed_ms(start_time, end_time);
		cache_hit = false;
		success = false;
		error_type = "rate_limit";
		if (neurondb_llm_fail_open)
		{
			ereport(ERROR,
					(errmsg("neurondb: LLM rate limited - fail "
							"open")));
			record_llm_stats(cfg.model,
							 "embed",
							 success,
							 cache_hit,
							 latency_ms,
							 tokens_in,
							 tokens_out,
							 error_type);
			PG_RETURN_NULL();
		}
		record_llm_stats(cfg.model,
						 "embed",
						 success,
						 cache_hit,
						 latency_ms,
						 tokens_in,
						 tokens_out,
						 error_type);
		ereport(ERROR, (errmsg("neurondb: LLM rate limited")));
	}

	/* Route embedding through configured provider */
	if (ndb_llm_route_embed(&cfg, &call_opts, txt, &vec, &dim) != 0
		|| dim <= 0)
	{
		end_time = GetCurrentTimestamp();
		latency_ms = (int64) ndb_elapsed_ms(start_time, end_time);
		cache_hit = false;
		success = false;
		error_type = "embed_failed";
		if (neurondb_llm_fail_open)
		{
			record_llm_stats(cfg.model,
							 "embed",
							 success,
							 cache_hit,
							 latency_ms,
							 tokens_in,
							 tokens_out,
							 error_type);
			PG_RETURN_NULL();
		}
		record_llm_stats(cfg.model,
						 "embed",
						 success,
						 cache_hit,
						 latency_ms,
						 tokens_in,
						 tokens_out,
						 error_type);
		ereport(ERROR, (errmsg("neurondb: embed failed")));
	}

	end_time = GetCurrentTimestamp();
	latency_ms = (int64) ndb_elapsed_ms(start_time, end_time);
	v = new_vector(dim);
	memcpy(v->data, vec, sizeof(float) * dim);
	cache_hit = false;
	success = true;
	{
		/* Count input tokens from text */
		int32		token_length;
		int32	   *token_ids = neurondb_tokenize_with_model(
															 txt, 2048, &token_length, cfg.model);

		if (token_ids && token_length > 0)
			tokens_in = token_length;
		else
			tokens_in = 0;		/* Fallback: estimate from word count */
		if (token_ids)
			NDB_FREE(token_ids);
	}
	tokens_out = dim;			/* Embedding dimension */
	record_llm_stats(cfg.model,
					 "embed",
					 success,
					 cache_hit,
					 latency_ms,
					 tokens_in,
					 tokens_out,
					 NULL);
	PG_RETURN_VECTOR_P(v);
}

/* SQL: llm_rerank(query text, docs text[], model text default null, top_n int default 10)
 * RETURNS TABLE(idx int, score real)
 *
 * Rerank documents using LLM provider APIs. Requires valid API keys and endpoints.
 */
PG_FUNCTION_INFO_V1(ndb_llm_rerank);
Datum
ndb_llm_rerank(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	typedef struct
	{
		int			ndocs;
		int			i;
	}			Ctx;
	Ctx		   *c;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext old;
		Ctx		   *new_ctx;
		ArrayType  *arr;
		TupleDesc	tupdesc;

		funcctx = SRF_FIRSTCALL_INIT();
		old = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);
		new_ctx = palloc0(sizeof(Ctx));
		arr = PG_GETARG_ARRAYTYPE_P(1);
		new_ctx->ndocs = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
		new_ctx->i = 0;
		funcctx->user_fctx = new_ctx;

		tupdesc = CreateTemplateTupleDesc(2);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 1, "idx", INT4OID, -1, 0);
		TupleDescInitEntry(
						   tupdesc, (AttrNumber) 2, "score", FLOAT4OID, -1, 0);
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		MemoryContextSwitchTo(old);
	}
	funcctx = SRF_PERCALL_SETUP();
	c = (Ctx *) funcctx->user_fctx;

	if (c->i < c->ndocs)
	{
		Datum		values[2];
		bool		nulls[2] = {false, false};
		HeapTuple	tup;

		values[0] = Int32GetDatum(c->i + 1);
		values[1] = Float4GetDatum(
								   1.0f - ((float) c->i / (float) Max(1, c->ndocs)));
		tup = heap_form_tuple(funcctx->tuple_desc, values, nulls);
		c->i++;
		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tup));
	}
	SRF_RETURN_DONE(funcctx);
}

/* SQL: llm_enqueue(action text, payload jsonb) RETURNS bigint
 *
 * Enqueue an LLM job in the neurondb_llm_jobs table.
 */
PG_FUNCTION_INFO_V1(ndb_llm_enqueue);
Datum
ndb_llm_enqueue(PG_FUNCTION_ARGS)
{
	Oid			argtypes[2] = {TEXTOID, JSONBOID};
	Datum		values[2];
	bool		isnull;
	Datum		d;
	text	   *action = PG_GETARG_TEXT_PP(0);
	text	   *payload = PG_GETARG_TEXT_PP(1);

	NDB_DECLARE(NdbSpiSession *, session);
	session = ndb_spi_session_begin(CurrentMemoryContext, false);
	if (session == NULL)
		ereport(ERROR, (errmsg("failed to begin SPI session")));

	values[0] = PointerGetDatum(action);
	values[1] = PointerGetDatum(payload);
	ndb_spi_execute_with_args(session,
							  "INSERT INTO "
							  "neurondb.llm_jobs(operation,input_text,status,model_name,created_at) "
							  "VALUES($1,$2::jsonb::text,'queued','default',now()) RETURNING job_id",
							  2,
							  argtypes,
							  values,
							  NULL,
							  true,
							  0);
	if (SPI_processed != 1)
	{
		ndb_spi_session_end(&session);
		ereport(ERROR, (errmsg("enqueue failed")));
	}

	d = SPI_getbinval(
					  SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
	ndb_spi_session_end(&session);
	PG_RETURN_INT64(DatumGetInt64(d));
}

/*
 * neurondb_llm_gpu_available
 *	  Check if GPU is available for LLM operations
 */
PG_FUNCTION_INFO_V1(neurondb_llm_gpu_available);

Datum
neurondb_llm_gpu_available(PG_FUNCTION_ARGS)
{
	bool		gpu_available = false;

	if (neurondb_gpu_is_available())
	{
		const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();

		if (backend != NULL && backend->hf_embed != NULL)
			gpu_available = true;
	}

	PG_RETURN_BOOL(gpu_available);
}

/*
 * neurondb_llm_gpu_info
 *	  Get GPU information for LLM operations
 */
PG_FUNCTION_INFO_V1(neurondb_llm_gpu_info);

Datum
neurondb_llm_gpu_info(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc	tupdesc;
	AttInMetadata *attinmeta;
	Datum		values[6];
	bool		nulls[6];
	HeapTuple	tuple;
	MemoryContext oldcontext;

	if (SRF_IS_FIRSTCALL())
	{
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record "
							"called in context "
							"that cannot accept type "
							"record")));

		attinmeta = TupleDescGetAttInMetadata(tupdesc);
		funcctx->attinmeta = attinmeta;
		funcctx->user_fctx = NULL;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	attinmeta = funcctx->attinmeta;

	if (funcctx->call_cntr < 1)
	{
		const ndb_gpu_backend *backend;
		GPUDeviceInfo *device_info;
		bool		gpu_available;
		char	   *backend_name = NULL;
		int			device_id = 0;
		char	   *device_name = NULL;
		int64		total_memory_mb = 0;
		int64		free_memory_mb = 0;
		bool		is_available = false;

		gpu_available = neurondb_gpu_is_available();
		backend = ndb_gpu_get_active_backend();

		if (gpu_available && backend != NULL)
		{
			backend_name = backend->name ? pstrdup(backend->name)
				: pstrdup("unknown");
			device_id = neurondb_gpu_device;
			device_info = neurondb_gpu_get_device_info(device_id);
			if (device_info != NULL)
			{
				device_name = pstrdup(device_info->name);
				total_memory_mb = device_info->total_memory_mb;
				free_memory_mb = device_info->free_memory_mb;
				is_available = device_info->is_available;
			}
			else
			{
				device_name = pstrdup("unknown");
				is_available = false;
			}
		}
		else
		{
			backend_name = pstrdup("none");
			device_name = pstrdup("unavailable");
			is_available = false;
		}

		values[0] = CStringGetTextDatum(backend_name);
		values[1] = Int32GetDatum(device_id);
		values[2] = CStringGetTextDatum(device_name);
		values[3] = Int64GetDatum(total_memory_mb);
		values[4] = Int64GetDatum(free_memory_mb);
		values[5] = BoolGetDatum(is_available);

		nulls[0] = false;
		nulls[1] = false;
		nulls[2] = false;
		nulls[3] = false;
		nulls[4] = false;
		nulls[5] = false;

		tuple = heap_form_tuple(tupdesc, values, nulls);
		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * neurondb_llm_gpu_utilization
 *	  Get GPU utilization metrics (utilization percentage, memory usage, temperature, power consumption)
 */
PG_FUNCTION_INFO_V1(neurondb_llm_gpu_utilization);

Datum
neurondb_llm_gpu_utilization(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc	tupdesc;
	AttInMetadata *attinmeta;
	Datum		values[8];
	bool		nulls[8];
	HeapTuple	tuple;
	MemoryContext oldcontext;

	if (SRF_IS_FIRSTCALL())
	{
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record "
							"called in context "
							"that cannot accept type "
							"record")));

		attinmeta = TupleDescGetAttInMetadata(tupdesc);
		funcctx->attinmeta = attinmeta;
		funcctx->user_fctx = NULL;
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	attinmeta = funcctx->attinmeta;

	if (funcctx->call_cntr < 1)
	{
		const ndb_gpu_backend *backend;
		GPUDeviceInfo *device_info;
		bool		gpu_available;
		int			device_id = 0;
		float4		utilization_pct = 0.0;
		int64		memory_used_mb = 0;
		int64		memory_total_mb = 0;
		float4		memory_utilization_pct = 0.0;
		int32		temperature_c = 0;
		float4		power_w = 0.0;
		TimestampTz timestamp;

		gpu_available = neurondb_gpu_is_available();
		backend = ndb_gpu_get_active_backend();

		if (gpu_available && backend != NULL)
		{
			device_id = neurondb_gpu_device;
			device_info = neurondb_gpu_get_device_info(device_id);
			if (device_info != NULL)
			{
				memory_total_mb = device_info->total_memory_mb;
				memory_used_mb = device_info->total_memory_mb - device_info->free_memory_mb;
				if (memory_total_mb > 0)
				{
					memory_utilization_pct = ((float4) memory_used_mb / (float4) memory_total_mb) * 100.0;
				}

				/*
				 * Note: GPU utilization percentage, temperature, and power
				 * require system-level APIs (NVML for CUDA, rocm-smi for
				 * ROCm, IOKit for Metal). These require integration with
				 * backend-specific monitoring APIs. For now, return 0.0 to
				 * indicate metrics are not available.
				 */
				utilization_pct = 0.0;
				temperature_c = 0;
				power_w = 0.0;
			}
		}

		timestamp = GetCurrentTimestamp();

		values[0] = Int32GetDatum(device_id);
		values[1] = Float4GetDatum(utilization_pct);
		values[2] = Int64GetDatum(memory_used_mb);
		values[3] = Int64GetDatum(memory_total_mb);
		values[4] = Float4GetDatum(memory_utilization_pct);
		values[5] = Int32GetDatum(temperature_c);
		values[6] = Float4GetDatum(power_w);
		values[7] = TimestampTzGetDatum(timestamp);

		nulls[0] = false;
		nulls[1] = false;
		nulls[2] = false;
		nulls[3] = false;
		nulls[4] = false;
		nulls[5] = false;
		nulls[6] = false;
		nulls[7] = false;

		tuple = heap_form_tuple(tupdesc, values, nulls);
		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * ndb_llm_complete_batch
 *	  Batch LLM completion with GPU acceleration support
 */
PG_FUNCTION_INFO_V1(ndb_llm_complete_batch);

Datum
ndb_llm_complete_batch(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc	tupdesc;
	AttInMetadata *attinmeta;
	Datum		values[5];
	bool		nulls[5];
	HeapTuple	tuple;
	MemoryContext oldcontext;
	typedef struct
	{
		int			num_prompts;
		int			current_idx;
		NdbLLMBatchResp *batch_resp;
		char	  **prompts;
		TupleDesc	tupdesc;
	}			BatchCompleteState;

	if (SRF_IS_FIRSTCALL())
	{
		ArrayType  *prompts_arr;
		text	   *params_in;
		char	   *params;
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;
		NdbLLMBatchResp batch_resp;
		int			i;
		int			num_prompts;
		char	  **prompts;
		int			rc;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record "
							"called in context "
							"that cannot accept type "
							"record")));

		attinmeta = TupleDescGetAttInMetadata(tupdesc);
		funcctx->attinmeta = attinmeta;

		/* Get prompts array */
		prompts_arr = PG_GETARG_ARRAYTYPE_P(0);
		num_prompts = ArrayGetNItems(
									 ARR_NDIM(prompts_arr), ARR_DIMS(prompts_arr));
		if (num_prompts <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("prompts array must not be "
							"empty")));

		/* Get params */
		params_in = (PG_NARGS() > 1 && !PG_ARGISNULL(1))
			? PG_GETARG_TEXT_PP(1)
			: NULL;
		params = params_in ? text_to_cstring(params_in) : "{}";

		/* Extract prompts from array using deconstruct_array */
		{
			Datum	   *prompt_datums;
			bool	   *prompt_nulls;
			int			nprompt_elems;

			deconstruct_array(prompts_arr,
							  TEXTOID,
							  -1,
							  false,
							  'i',
							  &prompt_datums,
							  &prompt_nulls,
							  &nprompt_elems);

			if (nprompt_elems != num_prompts)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("prompts array size mismatch: expected %d, got %d",
								num_prompts,
								nprompt_elems)));

			prompts = (char **) palloc0(num_prompts * sizeof(char *));
			for (i = 0; i < num_prompts; i++)
			{
				if (prompt_nulls[i])
					continue;
				prompts[i] = text_to_cstring(
											 DatumGetTextP(prompt_datums[i]));
			}
		}

		/* Fill config and call options */
		fill_cfg(&cfg);
		call_opts.task = "complete";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Call batch completion router */
		memset(&batch_resp, 0, sizeof(NdbLLMBatchResp));
		rc = ndb_llm_route_complete_batch(&cfg,
										  &call_opts,
										  (const char **) prompts,
										  num_prompts,
										  params,
										  &batch_resp);
		if (rc != 0)
		{
			/* Free prompts */
			for (i = 0; i < num_prompts; i++)
				if (prompts[i])
					NDB_FREE(prompts[i]);
			NDB_FREE(prompts);
			if (params)
				NDB_FREE(params);
			ereport(ERROR,
					(errcode(ERRCODE_EXTERNAL_ROUTINE_INVOCATION_EXCEPTION),
					 errmsg("neurondb: batch completion "
							"failed")));
		}

		/* Store state */
		{
			BatchCompleteState *state =
				(BatchCompleteState *) palloc0(
											   sizeof(BatchCompleteState));

			state->num_prompts = num_prompts;
			state->current_idx = 0;
			state->batch_resp = (NdbLLMBatchResp *) palloc0(
															sizeof(NdbLLMBatchResp));
			/* Deep copy batch_resp structure */
			state->batch_resp->num_items = batch_resp.num_items;
			state->batch_resp->num_success = batch_resp.num_success;
			if (batch_resp.texts && batch_resp.num_items > 0)
			{
				int			j;

				state->batch_resp->texts = (char **) palloc0(
															 batch_resp.num_items * sizeof(char *));
				for (j = 0; j < batch_resp.num_items; j++)
				{
					if (batch_resp.texts[j])
						state->batch_resp->texts[j] =
							pstrdup(batch_resp.texts[j]);
				}
			}
			if (batch_resp.tokens_in && batch_resp.num_items > 0)
			{
				NDB_DECLARE(int *, tokens_in);
				NDB_ALLOC(tokens_in, int, batch_resp.num_items);
				state->batch_resp->tokens_in = tokens_in;
				memcpy(state->batch_resp->tokens_in,
					   batch_resp.tokens_in,
					   batch_resp.num_items * sizeof(int));
			}
			if (batch_resp.tokens_out && batch_resp.num_items > 0)
			{
				NDB_DECLARE(int *, tokens_out);
				NDB_ALLOC(tokens_out, int, batch_resp.num_items);
				state->batch_resp->tokens_out = tokens_out;
				memcpy(state->batch_resp->tokens_out,
					   batch_resp.tokens_out,
					   batch_resp.num_items * sizeof(int));
			}
			if (batch_resp.http_status && batch_resp.num_items > 0)
			{
				NDB_DECLARE(int *, http_status);
				NDB_ALLOC(http_status, int, batch_resp.num_items);
				state->batch_resp->http_status = http_status;
				memcpy(state->batch_resp->http_status,
					   batch_resp.http_status,
					   batch_resp.num_items * sizeof(int));
			}
			state->prompts = prompts;
			state->tupdesc = tupdesc;
			funcctx->user_fctx = state;
		}

		if (params)
			NDB_FREE(params);
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	attinmeta = funcctx->attinmeta;
	{
		BatchCompleteState *state =
			(BatchCompleteState *) funcctx->user_fctx;
		TupleDesc	tupdesc_local;

		if (!state || !state->batch_resp || !state->tupdesc)
		{
			elog(ERROR, "neurondb: batch completion state is invalid");
			SRF_RETURN_DONE(funcctx);
		}
		tupdesc_local = state->tupdesc;

		if (state->current_idx < state->batch_resp->num_items)
		{
			int			idx = state->current_idx;
			NdbLLMBatchResp *resp = state->batch_resp;

			if (idx < 0 || idx >= resp->num_items)
			{
				elog(ERROR, "neurondb: batch completion index %d out of bounds (0-%d)", idx, resp->num_items - 1);
				SRF_RETURN_DONE(funcctx);
			}

			values[0] = Int32GetDatum(idx);
			if (resp->texts && idx < resp->num_items && resp->texts[idx])
			{
				values[1] =
					CStringGetTextDatum(resp->texts[idx]);
				values[2] = Int32GetDatum(resp->tokens_in
										  && idx < resp->num_items
										  ? resp->tokens_in[idx]
										  : 0);
				values[3] = Int32GetDatum(resp->tokens_out
										  && idx < resp->num_items
										  ? resp->tokens_out[idx]
										  : 0);
				values[4] = Int32GetDatum(resp->http_status
										  && idx < resp->num_items
										  ? resp->http_status[idx]
										  : 200);
				nulls[0] = false;
				nulls[1] = false;
				nulls[2] = false;
				nulls[3] = false;
				nulls[4] = false;
			}
			else
			{
				values[1] = (Datum) 0;
				values[2] = Int32GetDatum(0);
				values[3] = Int32GetDatum(0);
				values[4] = Int32GetDatum(500);
				nulls[0] = false;
				nulls[1] = true;
				nulls[2] = false;
				nulls[3] = false;
				nulls[4] = false;
			}

			tuple = heap_form_tuple(tupdesc_local, values, nulls);
			state->current_idx++;
			SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
		}
		else
		{
			/* Clean up */
			BatchCompleteState *cleanup_state =
				(BatchCompleteState *) funcctx->user_fctx;
			int			i;

			if (cleanup_state->batch_resp)
			{
				if (cleanup_state->batch_resp->texts)
				{
					for (i = 0; i
						 < cleanup_state->batch_resp->num_items;
						 i++)
						if (cleanup_state->batch_resp->texts[i])
							NDB_FREE(cleanup_state->batch_resp->texts
									 [i]);
					NDB_FREE(cleanup_state->batch_resp->texts);
				}
				if (cleanup_state->batch_resp->tokens_in)
					NDB_FREE(cleanup_state->batch_resp->tokens_in);
				if (cleanup_state->batch_resp->tokens_out)
					NDB_FREE(cleanup_state->batch_resp->tokens_out);
				if (cleanup_state->batch_resp->http_status)
					NDB_FREE(cleanup_state->batch_resp->http_status);
				NDB_FREE(cleanup_state->batch_resp);
			}
			if (cleanup_state->prompts)
			{
				for (i = 0; i < cleanup_state->num_prompts; i++)
					if (cleanup_state->prompts[i])
						NDB_FREE(cleanup_state->prompts[i]);
				NDB_FREE(cleanup_state->prompts);
			}
			NDB_FREE(cleanup_state);
			SRF_RETURN_DONE(funcctx);
		}
	}
}

/*
 * ndb_llm_rerank_batch
 *	  Batch LLM reranking with GPU acceleration support
 */
PG_FUNCTION_INFO_V1(ndb_llm_rerank_batch);

Datum
ndb_llm_rerank_batch(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	TupleDesc	tupdesc;
	AttInMetadata *attinmeta;
	Datum		values[3];
	bool		nulls[3];
	HeapTuple	tuple;
	MemoryContext oldcontext;
	typedef struct
	{
		int			num_queries;
		int			current_query_idx;
		int			current_doc_idx;
		float	  **scores;
		int		   *nscores;
		int		   *ndocs_array;
	}			BatchRerankState;

	if (SRF_IS_FIRSTCALL())
	{
		ArrayType  *queries_arr;
		ArrayType  *documents_arr;
		text	   *model_in;
		char	   *model;
		NdbLLMConfig cfg;
		NdbLLMCallOptions call_opts;
		int			num_queries;
		char	  **queries;
		const char ***docs_array;
		int		   *ndocs_array;
		float	  **scores;
		int		   *nscores;
		int			i;
		int			j;
		int			rc;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext =
			MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		if (get_call_result_type(fcinfo, NULL, &tupdesc)
			!= TYPEFUNC_COMPOSITE)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("function returning record "
							"called in context "
							"that cannot accept type "
							"record")));

		attinmeta = TupleDescGetAttInMetadata(tupdesc);
		funcctx->attinmeta = attinmeta;

		/* Get queries array */
		queries_arr = PG_GETARG_ARRAYTYPE_P(0);
		num_queries = ArrayGetNItems(
									 ARR_NDIM(queries_arr), ARR_DIMS(queries_arr));
		if (num_queries <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("queries array must not be "
							"empty")));

		/* Get documents array (2D array or 1D array of arrays) */
		documents_arr = PG_GETARG_ARRAYTYPE_P(1);
		if (ARR_NDIM(documents_arr) == 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("documents_array cannot be empty")));
		if (ARR_NDIM(documents_arr) > 2)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("documents_array must be a 1D or 2D "
							"array")));

		/* Get model */
		model_in = (PG_NARGS() > 2 && !PG_ARGISNULL(2))
			? PG_GETARG_TEXT_PP(2)
			: NULL;
		model = model_in ? text_to_cstring(model_in) : NULL;

		/* Extract queries from array using deconstruct_array */
		{
			Datum	   *query_datums;
			bool	   *query_nulls;
			int			nquery_elems;

			deconstruct_array(queries_arr,
							  TEXTOID,
							  -1,
							  false,
							  'i',
							  &query_datums,
							  &query_nulls,
							  &nquery_elems);

			if (nquery_elems != num_queries)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("queries array size mismatch: expected %d, got %d",
								num_queries,
								nquery_elems)));

			queries = (char **) palloc0(num_queries * sizeof(char *));
			for (i = 0; i < num_queries; i++)
			{
				if (query_nulls[i])
					continue;
				queries[i] = text_to_cstring(
											 DatumGetTextP(query_datums[i]));
			}
		}

		/* Extract documents from 2D array */
		ndocs_array = (int *) palloc0(num_queries * sizeof(int));
		docs_array =
			(const char ***) palloc0(num_queries * sizeof(char **));

		/*
		 * Extract documents from array (handles both 1D array of arrays and
		 * 2D array)
		 */
		{
			Oid			outer_elem_type;
			Oid			inner_elem_type;
			int16		outer_elem_len;
			int16		inner_elem_len;
			bool		outer_elem_byval;
			bool		inner_elem_byval;
			char		outer_elem_align;
			char		inner_elem_align;
			int			ndocs;
			char	  **docs;
			int			indices[1];
			int			indices_2d[2];

			if (ARR_NDIM(documents_arr) == 1)
			{
				/* Handle 1D array of arrays */
				int			nrows = ARR_DIMS(documents_arr)[0];

				if (nrows != num_queries)
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("documents array size mismatch: expected %d rows, got %d",
									num_queries,
									nrows)));

				/* Get element type of outer array for array_ref */
				outer_elem_type = ARR_ELEMTYPE(documents_arr);
				get_typlenbyvalalign(outer_elem_type,
									 &outer_elem_len,
									 &outer_elem_byval,
									 &outer_elem_align);

				for (i = 0; i < num_queries; i++)
				{
					bool		isNull;
					Datum		row_datum;
					ArrayType  *doc_row;
					Datum	   *doc_datums;
					bool	   *doc_nulls;
					int			ndoc_elems;

					indices[0] = i;

					/* Extract row i as a 1D array */
					row_datum = array_ref(documents_arr,
										  1,
										  indices,
										  outer_elem_len,
										  outer_elem_len,
										  outer_elem_byval,
										  outer_elem_align,
										  &isNull);

					if (isNull)
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("documents array row %d is NULL",
										i)));

					doc_row = DatumGetArrayTypeP(row_datum);

					/* Deconstruct the row array */
					deconstruct_array(doc_row,
									  TEXTOID,
									  -1,
									  false,
									  'i',
									  &doc_datums,
									  &doc_nulls,
									  &ndoc_elems);

					ndocs_array[i] = ndoc_elems;
					docs = (char **) palloc0(ndoc_elems * sizeof(char *));
					for (j = 0; j < ndoc_elems; j++)
					{
						if (doc_nulls[j])
							continue;
						docs[j] = text_to_cstring(
												  DatumGetTextP(doc_datums[j]));
					}
					docs_array[i] = (const char **) docs;
				}
			}
			else if (ARR_NDIM(documents_arr) == 2)
			{
				/* Handle true 2D array */
				/* For 2D arrays, ARR_ELEMTYPE returns the element type (text) */
				inner_elem_type = ARR_ELEMTYPE(documents_arr);

				/*
				 * Check if it's text or if we need to get element type of
				 * array type
				 */
				if (inner_elem_type == TEXTOID)
				{
					/* Direct 2D text array */
					get_typlenbyvalalign(inner_elem_type,
										 &inner_elem_len,
										 &inner_elem_byval,
										 &inner_elem_align);
				}
				else
				{
					/*
					 * Might be array of arrays - try to get inner element
					 * type
					 */
					inner_elem_type = get_element_type(inner_elem_type);
					if (!OidIsValid(inner_elem_type) || inner_elem_type != TEXTOID)
					{
						/* Fall back to treating as 1D array of arrays */
						goto handle_as_1d;
					}
					get_typlenbyvalalign(inner_elem_type,
										 &inner_elem_len,
										 &inner_elem_byval,
										 &inner_elem_align);
				}

				ndocs = ARR_DIMS(documents_arr)[1];

				for (i = 0; i < num_queries; i++)
				{
					ndocs_array[i] = ndocs;
					docs = (char **) palloc0(ndocs * sizeof(char *));
					for (j = 0; j < ndocs; j++)
					{
						bool		isNull;
						Datum		elem;

						indices_2d[0] = i;
						indices_2d[1] = j;

						/* Extract element [i][j] from 2D array */
						elem = array_ref(documents_arr,
										 2,
										 indices_2d,
										 inner_elem_len,
										 inner_elem_len,
										 inner_elem_byval,
										 inner_elem_align,
										 &isNull);

						if (!isNull && DatumGetPointer(elem))
						{
							docs[j] = text_to_cstring(
													  DatumGetTextP(elem));
						}
					}
					docs_array[i] = (const char **) docs;
				}
			}
			else
			{
				/* Unexpected dimension - treat as 1D array of arrays */
		handle_as_1d:
				/* Handle as 1D array of arrays */
				{
					int			nrows = ARR_DIMS(documents_arr)[0];

					if (nrows != num_queries)
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("documents array size mismatch: expected %d rows, got %d",
										num_queries,
										nrows)));
				}

				/* Get element type of outer array for array_ref */
				outer_elem_type = ARR_ELEMTYPE(documents_arr);
				get_typlenbyvalalign(outer_elem_type,
									 &outer_elem_len,
									 &outer_elem_byval,
									 &outer_elem_align);

				for (i = 0; i < num_queries; i++)
				{
					bool		isNull;
					Datum		row_datum;
					ArrayType  *doc_row;
					Datum	   *doc_datums;
					bool	   *doc_nulls;
					int			ndoc_elems;

					indices[0] = i;

					/* Extract row i as a 1D array */
					row_datum = array_ref(documents_arr,
										  1,
										  indices,
										  outer_elem_len,
										  outer_elem_len,
										  outer_elem_byval,
										  outer_elem_align,
										  &isNull);

					if (isNull)
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("documents array row %d is NULL",
										i)));

					doc_row = DatumGetArrayTypeP(row_datum);

					/* Deconstruct the row array */
					deconstruct_array(doc_row,
									  TEXTOID,
									  -1,
									  false,
									  'i',
									  &doc_datums,
									  &doc_nulls,
									  &ndoc_elems);

					ndocs_array[i] = ndoc_elems;
					docs = (char **) palloc0(ndoc_elems * sizeof(char *));
					for (j = 0; j < ndoc_elems; j++)
					{
						if (doc_nulls[j])
							continue;
						docs[j] = text_to_cstring(
												  DatumGetTextP(doc_datums[j]));
					}
					docs_array[i] = (const char **) docs;
				}
			}
		}

		/* Fill config and call options */
		fill_cfg(&cfg);
		if (model)
			cfg.model = model;
		call_opts.task = "rerank";
		call_opts.prefer_gpu = cfg.prefer_gpu;
		call_opts.require_gpu = cfg.require_gpu;
		call_opts.fail_open = neurondb_llm_fail_open;

		/* Call batch reranking router */
		scores = NULL;
		nscores = NULL;
		rc = ndb_llm_route_rerank_batch(&cfg,
										&call_opts,
										(const char **) queries,
										docs_array,
										ndocs_array,
										num_queries,
										&scores,
										&nscores);
		if (rc != 0)
		{
			/* Free queries */
			for (i = 0; i < num_queries; i++)
				if (queries[i])
					NDB_FREE(queries[i]);
			NDB_FREE(queries);
			/* Free documents */
			for (i = 0; i < num_queries; i++)
			{
				if (docs_array[i])
				{
					char	  **docs_ptr = (char **) docs_array[i];

					for (j = 0; j < ndocs_array[i]; j++)
					{
						NDB_FREE(docs_ptr[j]);
					}
					NDB_FREE(docs_ptr);
				}
			}
			NDB_FREE(docs_array);
			NDB_FREE(ndocs_array);
			if (model)
				NDB_FREE(model);
			ereport(ERROR,
					(errcode(ERRCODE_EXTERNAL_ROUTINE_INVOCATION_EXCEPTION),
					 errmsg("neurondb: batch reranking "
							"failed")));
		}

		/* Store state */
		{
			BatchRerankState *state = (BatchRerankState *) palloc0(
																   sizeof(BatchRerankState));

			if (!state)
			{
				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("neurondb: failed to allocate batch rerank state")));
			}
			state->num_queries = num_queries;
			state->current_query_idx = 0;
			state->current_doc_idx = 0;
			state->scores = scores;
			state->nscores = nscores;
			state->ndocs_array = ndocs_array;
			funcctx->user_fctx = state;
		}

		/* Free queries and documents arrays (scores are stored in state) */
		for (i = 0; i < num_queries; i++)
			if (queries[i])
				NDB_FREE(queries[i]);
		NDB_FREE(queries);
		for (i = 0; i < num_queries; i++)
		{
			if (docs_array[i])
			{
				char	  **docs_ptr = (char **) docs_array[i];

				for (j = 0; j < ndocs_array[i]; j++)
				{
					NDB_FREE(docs_ptr[j]);
				}
				NDB_FREE(docs_ptr);
			}
		}
		NDB_FREE(docs_array);
		if (model)
			NDB_FREE(model);
		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	attinmeta = funcctx->attinmeta;
	{
		BatchRerankState *state =
			(BatchRerankState *) funcctx->user_fctx;

		if (!state)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: batch rerank state is NULL")));
			SRF_RETURN_DONE(funcctx);
		}

		/* Find next query-document pair */
		while (state->current_query_idx < state->num_queries)
		{
			int			query_idx = state->current_query_idx;
			int			doc_idx = state->current_doc_idx;

			if (doc_idx < state->ndocs_array[query_idx])
			{
				/* Validate state and array bounds */
				if (!state || !state->scores || !state->ndocs_array)
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: batch rerank state is invalid")));
					SRF_RETURN_DONE(funcctx);
				}
				if (query_idx < 0 || query_idx >= state->num_queries)
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: query index %d out of bounds (0-%d)",
									query_idx,
									state->num_queries - 1)));
					SRF_RETURN_DONE(funcctx);
				}
				if (doc_idx < 0 || doc_idx >= state->ndocs_array[query_idx])
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: document index %d out of bounds (0-%d) for query %d",
									doc_idx,
									state->ndocs_array[query_idx] - 1,
									query_idx)));
					SRF_RETURN_DONE(funcctx);
				}

				if (state->scores && state->scores[query_idx] && doc_idx < state->nscores[query_idx])
				{
					values[0] = Int32GetDatum(query_idx);
					values[1] = Int32GetDatum(doc_idx);
					values[2] = Float4GetDatum(
											   state->scores[query_idx][doc_idx]);
					nulls[0] = false;
					nulls[1] = false;
					nulls[2] = false;
				}
				else
				{
					values[0] = Int32GetDatum(query_idx);
					values[1] = Int32GetDatum(doc_idx);
					values[2] = Float4GetDatum(0.0f);
					nulls[0] = false;
					nulls[1] = false;
					nulls[2] = false;
				}

				tuple = heap_form_tuple(tupdesc, values, nulls);
				state->current_doc_idx++;
				SRF_RETURN_NEXT(
								funcctx, HeapTupleGetDatum(tuple));
			}
			else
			{
				state->current_query_idx++;
				state->current_doc_idx = 0;
			}
		}

		/* Clean up */
		{
			int			i;

			if (state->scores)
			{
				for (i = 0; i < state->num_queries; i++)
					if (state->scores[i])
						NDB_FREE(state->scores[i]);
				NDB_FREE(state->scores);
			}
			if (state->nscores)
				NDB_FREE(state->nscores);
			if (state->ndocs_array)
				NDB_FREE(state->ndocs_array);
			NDB_FREE(state);
		}
		SRF_RETURN_DONE(funcctx);
	}
}
