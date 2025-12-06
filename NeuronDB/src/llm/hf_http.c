#include "postgres.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "parser/parse_type.h"
#include "parser/parse_func.h"
#include "utils/lsyscache.h"
#include "catalog/pg_proc.h"
#include "nodes/makefuncs.h"
#include <curl/curl.h>
#include <stdlib.h>
#include <ctype.h>
#include "neurondb_llm.h"
#include "neurondb_json.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"

static text * ndb_encode_base64(bytea * data);
int			http_post_json(const char *url, const char *api_key, const char *body, int timeout_ms, char **resp_out);
static bool handle_http_response(int http_status, char **json_ptr, NdbLLMResp *out);

/* HuggingFace endpoint classification */
typedef enum
{
	HF_EP_GENERIC = 0,
	HF_EP_ROUTER,
	HF_EP_API_INFERENCE
} HfEndpointKind;

static HfEndpointKind
hf_classify_endpoint(const char *endpoint)
{
	if (!endpoint)
		return HF_EP_GENERIC;

	if (strstr(endpoint, "router.huggingface.co") != NULL)
		return HF_EP_ROUTER;

	if (strstr(endpoint, "api-inference.huggingface.co") != NULL)
		return HF_EP_API_INFERENCE;

	return HF_EP_GENERIC;
}

/*
 * ndb_hf_vision_complete - Call HuggingFace vision model for image+prompt completion
 */
int
ndb_hf_vision_complete(const NdbLLMConfig *cfg,
					   const unsigned char *image_data,
					   size_t image_size,
					   const char *prompt,
					   const char *params_json,
					   NdbLLMResp *out)
{
	StringInfoData url,
				body;
	char	   *resp = NULL;
	int			code;
	bool		ok = false;
	char	   *base64_data = NULL;
	text	   *encoded_text = NULL;
	char	   *quoted_prompt = NULL;
	char	   *text_start;
	char	   *text_end;
	size_t		len;
	HfEndpointKind kind;

	if (!cfg || !image_data || image_size == 0 || !prompt || !out)
		return -1;

	/* Validate API key is required for HuggingFace inference API */
	if (!cfg->api_key || cfg->api_key[0] == '\0')
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("API key is required for HuggingFace but was not provided"),
				 errhint("Set neurondb.llm_api_key configuration parameter")));
		return -1;
	}

	initStringInfo(&url);
	initStringInfo(&body);

	/* Base64 encode image */
	{
		NDB_DECLARE(bytea *, image_bytea);
		NDB_DECLARE(char *, image_bytea_raw);
		NDB_ALLOC(image_bytea_raw, char, VARHDRSZ + image_size);
		image_bytea = (bytea *) image_bytea_raw;

		SET_VARSIZE(image_bytea, VARHDRSZ + image_size);
		memcpy(VARDATA(image_bytea), image_data, image_size);
		encoded_text = ndb_encode_base64(image_bytea);
		base64_data = text_to_cstring(encoded_text);
		NDB_FREE(image_bytea);
		NDB_FREE(encoded_text);
	}

	quoted_prompt = ndb_json_quote_string(prompt);

	/* Build URL for HuggingFace vision completion API */
	kind = hf_classify_endpoint(cfg->endpoint);

	switch (kind)
	{
		case HF_EP_ROUTER:
			if (strstr(cfg->endpoint, "/hf-inference") != NULL)
				appendStringInfo(&url,
								 "%s/models/%s/pipeline/image-to-text",
								 cfg->endpoint,
								 cfg->model);
			else
				appendStringInfo(&url,
								 "%s/hf-inference/models/%s/pipeline/image-to-text",
								 cfg->endpoint,
								 cfg->model);
			break;

		case HF_EP_API_INFERENCE:
			appendStringInfo(&url,
							 "%s/models/%s/pipeline/image-to-text",
							 cfg->endpoint,
							 cfg->model);
			break;

		case HF_EP_GENERIC:
		default:
			appendStringInfo(&url,
							 "%s/pipeline/image-to-text/%s",
							 cfg->endpoint,
							 cfg->model);
			break;
	}

	/* Compose JSON body */
	if (params_json && strlen(params_json) > 0)
	{
		appendStringInfo(&body,
						 "{\"inputs\":{\"image\":\"data:image/jpeg;base64,%s\",\"prompt\":%s},%s}",
						 base64_data,
						 quoted_prompt,
						 params_json);
	}
	else
	{
		appendStringInfo(&body,
						 "{\"inputs\":{\"image\":\"data:image/jpeg;base64,%s\",\"prompt\":%s}}",
						 base64_data,
						 quoted_prompt);
	}

	code = http_post_json(
						  url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);

	text_start = NULL;
	text_end = NULL;
	len = 0;
	NDB_FREE(url.data);
	NDB_FREE(body.data);
	NDB_FREE(base64_data);
	NDB_FREE(quoted_prompt);

	/* Handle all HTTP response types */
	if (!handle_http_response(code, &resp, out))
	{
		if (resp)
			NDB_FREE(resp);
		return -1;
	}

	/* Parse response: expect JSON with 'generated_text' or similar */
	text_start = strstr(resp, "generated_text");
	if (text_start)
	{
		text_start = strchr(text_start, ':');
		if (text_start)
		{
			text_start++;
			while (*text_start && (*text_start == ' ' || *text_start == '"'))
				text_start++;
			text_end = strchr(text_start, '"');
			if (text_end)
			{
				len = text_end - text_start;
				NDB_ALLOC(out->text, char, len + 1);
				strncpy(out->text, text_start, len);
				out->text[len] = '\0';
				ok = true;
			}
		}
	}
	out->json = resp;
	out->http_status = code;
	out->tokens_in = 0;
	out->tokens_out = 0;
	return ok ? 0 : -1;
}

/* Helper: Look up and call PostgreSQL's encode() function for base64 encoding */
static text *
ndb_encode_base64(bytea * data)
{
	List	   *funcname;
	Oid			argtypes[2];
	Oid			encode_oid;
	FmgrInfo	flinfo;
	Datum		result;

	/* Look up encode(bytea, text) function */
	funcname = list_make1(makeString("encode"));
	argtypes[0] = BYTEAOID;
	argtypes[1] = TEXTOID;
	encode_oid = LookupFuncName(funcname, 2, argtypes, false);

	if (!OidIsValid(encode_oid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_FUNCTION),
				 errmsg("encode function not found")));

	fmgr_info(encode_oid, &flinfo);
	result = FunctionCall2(&flinfo,
						   PointerGetDatum(data),
						   CStringGetDatum("base64"));

	return DatumGetTextP(result);
}

/* Helper for dynamic memory buffer for curl writes */
typedef struct
{
	char	   *data;
	size_t		len;
}			MemBuf;

static size_t
write_cb(void *ptr, size_t size, size_t nmemb, void *userdata)
{
	MemBuf	   *m = (MemBuf *) userdata;
	size_t		n = size * nmemb;

	m->data = repalloc(m->data, m->len + n + 1);
	memcpy(m->data + m->len, ptr, n);
	m->len += n;
	m->data[m->len] = '\0';
	return n;
}

/* HTTP POST with JSON body, outputs body and HTTP status code */
int
http_post_json(const char *url,
			   const char *api_key,
			   const char *json_body,
			   int timeout_ms,
			   char **out)
{
	CURL	   *curl = curl_easy_init();
	struct curl_slist *headers = NULL;
	MemBuf		buf = {palloc0(1), 0};
	long		code = 0;
	CURLcode	res;

	if (!curl)
	{
		*out = NULL;
		return -1;
	}

	headers = curl_slist_append(headers, "Content-Type: application/json");
	if (api_key && api_key[0])
	{
		StringInfoData h;

		initStringInfo(&h);
		appendStringInfo(&h, "Authorization: Bearer %s", api_key);
		headers = curl_slist_append(headers, h.data);
		NDB_FREE(h.data);
	}
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);
	curl_easy_setopt(curl, CURLOPT_USERAGENT, "neurondb-llm/1.0");

	res = curl_easy_perform(curl);
	if (res != CURLE_OK)
	{
		curl_slist_free_all(headers);
		curl_easy_cleanup(curl);
		if (buf.data)
			NDB_FREE(buf.data);
		*out = NULL;
		return -1;
	}
	curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);

	*out = buf.data;
	return (int) code;
}

/*
 * Helper function to handle all HTTP responses consistently
 * Wraps non-JSON error responses in JSON format and stores in out->json
 * Returns true if response should be treated as success, false otherwise
 */
static bool
handle_http_response(int http_status, char **json_ptr, NdbLLMResp *out)
{
	if (!json_ptr || !*json_ptr)
	{
		if (out)
		{
			out->http_status = http_status;
			out->json = NULL;
		}
		return (http_status >= NDB_HTTP_STATUS_OK_MIN && 
				http_status <= NDB_HTTP_STATUS_OK_MAX);
	}

	if (out)
		out->http_status = http_status;

	/* Handle non-JSON error responses (e.g., plain text "Not Found" from router) */
	if (http_status >= NDB_HTTP_STATUS_ERROR_MIN)
	{
		const char *json_ptr_check = *json_ptr;
		
		/* Skip whitespace */
		while (*json_ptr_check && isspace((unsigned char) *json_ptr_check))
			json_ptr_check++;
		
		/* Check if response is JSON (starts with { or [) */
		if (*json_ptr_check != '{' && *json_ptr_check != '[')
		{
			/* Non-JSON response (e.g., plain text "Not Found") - wrap in JSON error format */
			StringInfoData error_json;
			StringInfoData error_msg;
			char	   *quoted_error;
			
			initStringInfo(&error_json);
			initStringInfo(&error_msg);
			
			/* Build error message: "HTTP 404: Not Found" */
			appendStringInfo(&error_msg, "HTTP %d: %s", http_status, *json_ptr);
			
			/* Quote the error message (ndb_json_quote_string adds quotes) */
			quoted_error = ndb_json_quote_string(error_msg.data);
			
			/* Build proper JSON: {"error":"HTTP 404: Not Found"} */
			appendStringInfo(&error_json, "{\"error\":%s}", quoted_error);
			
			NDB_FREE(quoted_error);
			NDB_FREE(error_msg.data);
			NDB_FREE(*json_ptr);
			*json_ptr = error_json.data;
			if (out)
				out->json = *json_ptr;
		}
		else if (out)
		{
			out->json = *json_ptr;
		}
		return false;
	}
	
	/* Success responses (2xx) */
	if (http_status >= NDB_HTTP_STATUS_OK_MIN && http_status <= NDB_HTTP_STATUS_OK_MAX)
	{
		if (out)
			out->json = *json_ptr;
		return true;
	}
	
	/* Other responses (1xx, 3xx) */
	if (out)
		out->json = *json_ptr;
	return false;
}

/* Extracts text field from HuggingFace inference API responses */
static char *
extract_hf_text(const char *json)
{
	/*
	 * The text generation output is a top-level list of { "generated_text":
	 * ... } objects. Example: [{"generated_text":"result"}], so we parse it.
	 * The response might also be { "error": ... }.
	 */
	const char *key;
	char	   *p;
	char	   *q;
	size_t		len;
	NDB_DECLARE(char *, out);
	const char *json_trimmed;

	if (!json || json[0] == '\0')
		return NULL;
	
	/* Trim leading whitespace */
	json_trimmed = json;
	while (*json_trimmed && isspace((unsigned char) *json_trimmed))
		json_trimmed++;
	
	if (strncmp(json_trimmed, "{\"error\"", 8) == 0)
		return NULL;

	/* Try OpenAI-compatible format first: choices[0].message.content */
	key = "\"content\":";
	p = strstr(json_trimmed, key);
	if (p)
	{
		/* Found OpenAI format, extract content */
		p = strchr(p + strlen(key), '"');
		if (!p)
			return NULL;
		p++;
		q = strchr(p, '"');
		if (!q)
			return NULL;
		len = q - p;
		NDB_ALLOC(out, char, len + 1);
		memcpy(out, p, len);
		out[len] = '\0';
		return out;
	}

	/* Fall back to legacy format: generated_text */
	key = "\"generated_text\":";
	p = strstr(json_trimmed, key);
	if (!p)
		return NULL;
	/* Find the first quote after the key */
	p = strchr(p + strlen(key), '"');
	if (!p)
		return NULL;
	p++;
	q = strchr(p, '"');
	if (!q)
		return NULL;
	len = q - p;
	NDB_ALLOC(out, char, len + 1);
	memcpy(out, p, len);
	out[len] = '\0';
	return out;
}

int
ndb_hf_complete(const NdbLLMConfig *cfg,
				const char *prompt,
				const char *params_json,
				NdbLLMResp *out)
{
	StringInfoData url,
				body;
	HfEndpointKind kind;
	int			status;
	int			rc;
	bool		tried_fallback = false;
	bool		tried_chat_format = false;
	bool		use_chat_format = false;

	initStringInfo(&url);
	initStringInfo(&body);

	if (prompt == NULL)
	{
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		return -1;
	}

	/* Validate API key is required for HuggingFace inference API */
	if (!cfg->api_key || cfg->api_key[0] == '\0')
	{
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("API key is required for HuggingFace but was not provided"),
				 errhint("Set neurondb.llm_api_key configuration parameter")));
		return -1;
	}

	kind = hf_classify_endpoint(cfg->endpoint);
	
	/* For router endpoints, try chat format first */
	if (kind == HF_EP_ROUTER && !tried_chat_format)
	{
		use_chat_format = true;
		tried_chat_format = true;
	}

build_url:
	{
		const char *endpoint_to_use = cfg->endpoint;

		resetStringInfo(&url);
		resetStringInfo(&body);

		switch (kind)
		{
			case HF_EP_ROUTER:
				/*
				 * Router style: 
				 * Chat format: clean_base + "/v1/chat/completions"
				 * Non-chat format: clean_base + "/hf-inference/models/{model}"
				 */
				{
					StringInfoData clean_endpoint;
					const char *base = endpoint_to_use;
					
					initStringInfo(&clean_endpoint);
					/* Remove /hf-inference if present to get base router URL */
					if (strstr(base, "/hf-inference") != NULL)
					{
						size_t len = strstr(base, "/hf-inference") - base;
						appendStringInfo(&clean_endpoint, "%.*s", (int)len, base);
						if (use_chat_format)
						{
							/* Chat format: clean_base + "/v1/chat/completions" */
							appendStringInfo(&url, "%s/v1/chat/completions",
											 clean_endpoint.data);
						}
						else
						{
							/* Non-chat format: clean_base + "/hf-inference/models/{model}" */
							appendStringInfo(&url, "%s/hf-inference/models/%s",
											 clean_endpoint.data, cfg->model);
						}
						NDB_FREE(clean_endpoint.data);
					}
					else
					{
						/* Base router endpoint (no /hf-inference in original) */
						if (use_chat_format)
						{
							appendStringInfo(&url, "%s/v1/chat/completions",
											 endpoint_to_use);
						}
						else
						{
							/* Add /hf-inference for non-chat format */
							appendStringInfo(&url, "%s/hf-inference/models/%s",
											 endpoint_to_use, cfg->model);
						}
					}
				}
				break;

			case HF_EP_API_INFERENCE:
				/*
				 * Legacy api-inference, single model endpoint.
				 */
				appendStringInfo(&url, "%s/models/%s",
								 endpoint_to_use, cfg->model);
				break;

			case HF_EP_GENERIC:
			default:
				/*
				 * Generic HF style, assume base already points at correct
				 * location. Keep simple and let admin set a sensible value.
				 */
				appendStringInfo(&url, "%s/models/%s",
								 endpoint_to_use, cfg->model);
				break;
		}
	}

	/* For router endpoints, use OpenAI-compatible format only if use_chat_format is true */
	if (kind == HF_EP_ROUTER && use_chat_format)
	{
		char	   *model_quoted = ndb_json_quote_string(cfg->model);
		char	   *prompt_quoted = ndb_json_quote_string(prompt);
		
		/* Build OpenAI-compatible request body */
		appendStringInfo(&body,
						 "{\"model\":%s,\"messages\":[{\"role\":\"user\",\"content\":%s}]",
						 model_quoted,
						 prompt_quoted);
		
		/* Append params_json if provided (temperature, max_tokens, etc.) */
		/* Filter out "model" field to avoid duplication */
		if (params_json && params_json[0] != '\0' && strcmp(params_json, "{}") != 0)
		{
			const char *p;
			const char *end;
			bool		has_model_field = false;
			
			/* Check if params_json contains "model" field */
			if (strstr(params_json, "\"model\"") != NULL)
			{
				has_model_field = true;
			}
			
			/* Remove outer braces */
			p = params_json;
			while (*p && (*p == '{' || isspace((unsigned char) *p)))
				p++;
			end = params_json + strlen(params_json) - 1;
			while (end > p && (*end == '}' || isspace((unsigned char) *end)))
				end--;
			
			if (end > p)
			{
				appendStringInfoChar(&body, ',');
				
				if (has_model_field)
				{
					/* Filter out "model" field by skipping it during copy */
					const char *current = p;
					while (current <= end)
					{
						if (strncmp(current, "\"model\"", 7) == 0)
						{
							/* Skip the model field */
							const char *skip_start = current;
							const char *skip_end = strchr(skip_start + 7, ':');
							if (skip_end)
							{
								skip_end++; /* Skip ':' */
								while (*skip_end && isspace((unsigned char) *skip_end))
									skip_end++;
								
								/* Skip the value */
								if (*skip_end == '"')
								{
									skip_end = strchr(skip_end + 1, '"');
									if (skip_end)
										skip_end++;
								}
								else
								{
									while (*skip_end && *skip_end != ',' && *skip_end != '}' && !isspace((unsigned char) *skip_end))
										skip_end++;
								}
								
								/* Skip comma if present */
								if (*skip_end == ',')
									skip_end++;
								while (*skip_end && isspace((unsigned char) *skip_end))
									skip_end++;
								
								current = skip_end;
							}
							else
							{
								current += 7;
							}
						}
						else
						{
							/* Find next comma or end */
							const char *next_comma = strchr(current, ',');
							const char *next_brace = strchr(current, '}');
							const char *next = next_comma;
							
							if (next_brace && (!next_comma || next_brace < next_comma))
								next = next_brace;
							
							if (next && next <= end)
							{
								size_t		len = next - current;
								
								appendStringInfo(&body, "%.*s", (int) len, current);
								current = next;
								if (*current == ',')
								{
									appendStringInfoChar(&body, ',');
									current++;
								}
								while (*current && isspace((unsigned char) *current))
									current++;
							}
							else
							{
								/* Copy rest */
								size_t		len = end - current + 1;
								
								appendStringInfo(&body, "%.*s", (int) len, current);
								break;
							}
						}
					}
				}
				else
				{
					/* No model field, just append */
					size_t		len = end - p + 1;
					appendStringInfo(&body, "%.*s", (int) len, p);
				}
			}
		}
		
		appendStringInfoChar(&body, '}');
		
		NDB_FREE(model_quoted);
		NDB_FREE(prompt_quoted);
	}
	else
	{
		/* Legacy inference API format */
		/* Filter out "model" field from params_json since model is in URL path */
		char	   *filtered_params = NULL;
		
		if (params_json && params_json[0] != '\0' && strcmp(params_json, "{}") != 0)
		{
			const char *p;
			const char *end;
			bool		has_model_field = false;
			StringInfoData filtered;
			
			/* Check if params_json contains "model" field */
			if (strstr(params_json, "\"model\"") != NULL)
			{
				has_model_field = true;
			}
			
			initStringInfo(&filtered);
			appendStringInfoChar(&filtered, '{');
			
			/* Remove outer braces */
			p = params_json;
			while (*p && (*p == '{' || isspace((unsigned char) *p)))
				p++;
			end = params_json + strlen(params_json) - 1;
			while (end > p && (*end == '}' || isspace((unsigned char) *end)))
				end--;
			
			if (end > p)
			{
				if (has_model_field)
				{
					/* Filter out "model" field */
					const char *current = p;
					while (current <= end)
					{
						if (strncmp(current, "\"model\"", 7) == 0)
						{
							/* Skip the model field */
							const char *skip_start = current;
							const char *skip_end = strchr(skip_start + 7, ':');
							if (skip_end)
							{
								skip_end++; /* Skip ':' */
								while (*skip_end && isspace((unsigned char) *skip_end))
									skip_end++;
								
								/* Skip the value */
								if (*skip_end == '"')
								{
									skip_end = strchr(skip_end + 1, '"');
									if (skip_end)
										skip_end++;
								}
								else
								{
									while (*skip_end && *skip_end != ',' && *skip_end != '}' && !isspace((unsigned char) *skip_end))
										skip_end++;
								}
								
								/* Skip comma if present */
								if (*skip_end == ',')
									skip_end++;
								while (*skip_end && isspace((unsigned char) *skip_end))
									skip_end++;
								
								current = skip_end;
							}
							else
							{
								current += 7;
							}
						}
						else
						{
							/* Find next comma or end */
							const char *next_comma = strchr(current, ',');
							const char *next_brace = strchr(current, '}');
							const char *next = next_comma;
							
							if (next_brace && (!next_comma || next_brace < next_comma))
								next = next_brace;
							
							if (next && next <= end)
							{
								size_t		len = next - current;
								
								if (filtered.len > 1) /* Already has content */
									appendStringInfoChar(&filtered, ',');
								appendStringInfo(&filtered, "%.*s", (int) len, current);
								current = next;
								if (*current == ',')
									current++;
								while (*current && isspace((unsigned char) *current))
									current++;
							}
							else
							{
								/* Copy rest */
								if (filtered.len > 1)
									appendStringInfoChar(&filtered, ',');
								size_t		len = end - current + 1;
								appendStringInfo(&filtered, "%.*s", (int) len, current);
								break;
							}
						}
					}
				}
				else
				{
					/* No model field, just copy */
					size_t		len = end - p + 1;
					appendStringInfo(&filtered, "%.*s", (int) len, p);
				}
			}
			
			appendStringInfoChar(&filtered, '}');
			filtered_params = filtered.data;
		}
		else
		{
			filtered_params = "{}";
		}
		
		appendStringInfo(&body,
						 "{\"inputs\":%s,\"parameters\":%s}",
						 ndb_json_quote_string(prompt),
						 filtered_params);
		
		/* Free filtered_params if we allocated it */
		if (filtered_params && filtered_params != "{}" && strcmp(filtered_params, "{}") != 0)
		{
			NDB_FREE(filtered_params);
		}
	}

	/* Use local resp variable and handle_http_response pattern like embed functions */
	char	   *resp = NULL;
	NdbLLMResp temp_resp = {0};
	int			code;

	code = http_post_json(url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);
	status = code;

	/*
	 * For router endpoints, handle different error cases:
	 * 1. If "not a chat model" error, retry with non-chat format
	 * Note: We do NOT fall back to api-inference.huggingface.co (it's deprecated)
	 */
	if (kind == HF_EP_ROUTER && !tried_fallback && resp)
	{
		bool		retry_needed = false;
		
		/* Check for "not a chat model" or "model_not_supported" errors */
		if (status == NDB_HTTP_STATUS_BAD_REQUEST &&
			(strstr(resp, "not a chat model") != NULL ||
			 strstr(resp, "model_not_supported") != NULL ||
			 strstr(resp, "not supported by any provider") != NULL))
		{
			ereport(WARNING,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("HuggingFace model '%s' is not supported as a chat model", cfg->model ? cfg->model : "unknown"),
					 errhint("Retrying with inference API format")));
			use_chat_format = false;
			retry_needed = true;
		}
		/* Check for 404 - model not found */
		else if (status == NDB_HTTP_STATUS_NOT_FOUND)
		{
			ereport(WARNING,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("HuggingFace model '%s' not found on router endpoint", cfg->model ? cfg->model : "unknown"),
					 errhint("Model may not be available. Check your HuggingFace account for available models.")));
			/* Do not retry - model is not available on router endpoint */
			retry_needed = false;
		}
		
		if (retry_needed)
		{
			/* Free old JSON buffer from failed attempt */
			if (resp)
			{
				NDB_FREE(resp);
				resp = NULL;
			}
			goto build_url;
		}
	}

	/* Use handle_http_response to process response consistently with other functions */
	if (!handle_http_response(code, &resp, &temp_resp))
	{
		if (resp)
			NDB_FREE(resp);
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		return -1;
	}

	/* Note: api-inference.huggingface.co is no longer supported - all requests should use router.huggingface.co */

	/* Assign response to output */
	out->http_status = code;
	out->json = resp;

	/* Extract text from response if successful */
	out->text = NULL;
	if (code >= NDB_HTTP_STATUS_OK_MIN && code <= NDB_HTTP_STATUS_OK_MAX && resp)
	{
		char	   *t;

		/* Check for error in response first */
		if (strncmp(resp, "{\"error\"", 8) == 0)
		{
			/* Error response from API */
			ereport(WARNING,
					(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
					 errmsg("HuggingFace API returned error in response"),
					 errhint("Check the API response for details")));
			NDB_FREE(url.data);
			NDB_FREE(body.data);
			return -1;
		}

		/* Try to parse a "generated_text" or "content" value out */
		t = extract_hf_text(resp);

		if (t)
		{
			out->text = t;
		}
		else
		{
			/* Could not extract generated text - this might be an error or unexpected format */
			ereport(WARNING,
					(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
					 errmsg("Could not extract generated text from HuggingFace API response"),
					 errhint("The response format may be unexpected")));
			NDB_FREE(url.data);
			NDB_FREE(body.data);
			return -1;
		}
	}
	else if (code >= NDB_HTTP_STATUS_ERROR_MIN)
	{
		/* HTTP error response */
		ereport(WARNING,
				(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
				 errmsg("HuggingFace API returned HTTP error %d", code),
				 errhint("Check API key, endpoint, and model availability")));
	}

	rc = (code >= NDB_HTTP_STATUS_OK_MIN && 
		  code <= NDB_HTTP_STATUS_OK_MAX && 
		  out->text != NULL) ? 0 : -1;
	
	NDB_FREE(url.data);
	NDB_FREE(body.data);
	return rc;
}

/* Extracts a flat float vector from HF embedding API JSON response */
static bool
parse_hf_emb_vector(const char *json, float **vec_out, int *dim_out)
{
	/* Response is: [[float, float, ...]] */
	/* Error response is: {"error":"..."} */
	const char *p;
	float	   *vec = NULL;
	int			n = 0;
	int			cap = 32;
	char	   *endptr;
	double		v;

	if (!json)
		return false;

	/* Check for error response first */
	if (strncmp(json, "{\"error\"", 8) == 0)
	{
		/* Extract error message for logging */
		const char *err_start = strstr(json, "\"error\":");
		const char *err_end;

		if (err_start)
		{
			err_start = strchr(err_start, '"');
			if (err_start)
			{
				err_start++;
				err_end = strchr(err_start, '"');
				if (err_end)
				{
					size_t		err_len = err_end - err_start;

					NDB_DECLARE(char *, err_msg);

					NDB_ALLOC(err_msg, char, err_len + 1);

					memcpy(err_msg, err_start, err_len);
					err_msg[err_len] = '\0';
					elog(DEBUG1, "neurondb: HF API error: %s", err_msg);
					NDB_FREE(err_msg);
				}
			}
		}
		return false;
	}

	p = json;
	while (*p && *p != '[')
		p++;
	if (!*p)
		return false;
	p++;
	while (*p && isspace(*p))
		p++;

	/*
	 * Router endpoint returns flat array [...], old endpoint returns nested
	 * [[...]]
	 */
	/* Check if next char is '[' (nested) or a number/digit (flat) */
	if (*p == '[')
	{
		/* Nested array format: [[...]] */
		p++;
	}
	else if (*p == '-' || (*p >= '0' && *p <= '9'))
	{
		/* Flat array format: [...] - already at start of numbers */
		/* p stays where it is */
	}
	else
	{
		return false;
	}

	NDB_ALLOC(vec, float, cap);

	while (*p && *p != ']')
	{
		while (*p && (isspace(*p) || *p == ','))
			p++;
		endptr = NULL;
		v = strtod(p, &endptr);
		if (endptr == p)
			break;
		if (n == cap)
		{
			cap *= 2;
			vec = repalloc(vec, sizeof(float) * cap);
		}
		vec[n++] = (float) v;
		p = endptr;
	}
	if (n > 0)
	{
		*vec_out = vec;
		*dim_out = n;
		return true;
	}
	else
	{
		NDB_FREE(vec);
		return false;
	}
}

int
ndb_hf_embed(const NdbLLMConfig *cfg,
			 const char *text,
			 float **vec_out,
			 int *dim_out)
{
	StringInfoData url,
				body;
	char	   *resp = NULL;
	int			code;
	bool		ok;
	NdbLLMResp	temp_resp = {0};
	HfEndpointKind kind;

	initStringInfo(&url);
	initStringInfo(&body);

	if (text == NULL)
	{
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		return -1;
	}

	/* Validate API key is required for HuggingFace inference API */
	if (!cfg->api_key || cfg->api_key[0] == '\0')
	{
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("API key is required for HuggingFace but was not provided"),
				 errhint("Set neurondb.llm_api_key configuration parameter")));
		return -1;
	}

	kind = hf_classify_endpoint(cfg->endpoint);

	switch (kind)
	{
		case HF_EP_ROUTER:
			if (strstr(cfg->endpoint, "/hf-inference") != NULL)
				appendStringInfo(&url,
								 "%s/models/%s/pipeline/feature-extraction",
								 cfg->endpoint,
								 cfg->model);
			else
				appendStringInfo(&url,
								 "%s/hf-inference/models/%s/pipeline/feature-extraction",
								 cfg->endpoint,
								 cfg->model);
			break;

		case HF_EP_API_INFERENCE:
			/*
			 * api-inference feature extraction often uses the same
			 * /models/MODEL/pipeline/feature-extraction layout.
			 */
			appendStringInfo(&url,
							 "%s/models/%s/pipeline/feature-extraction",
							 cfg->endpoint,
							 cfg->model);
			break;

		case HF_EP_GENERIC:
		default:
			appendStringInfo(&url,
							 "%s/pipeline/feature-extraction/%s",
							 cfg->endpoint,
							 cfg->model);
			break;
	}
	appendStringInfo(&body,
					 "{\"inputs\":%s,\"truncate\":true}",
					 ndb_json_quote_string(text));
	code = http_post_json(
						  url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);
	
	/* Handle all HTTP response types */
	if (!handle_http_response(code, &resp, &temp_resp))
	{
		if (resp)
			NDB_FREE(resp);
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		return -1;
	}
	
	/* Check for error in response body (API may return 200 with error JSON) */
	if (resp && strncmp(resp, "{\"error\"", 8) == 0)
	{
		NDB_FREE(resp);
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		return -1;
	}
	ok = parse_hf_emb_vector(resp, vec_out, dim_out);
	if (!ok)
	{
		NDB_FREE(resp);
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		return -1;
	}
	NDB_FREE(resp);
	NDB_FREE(url.data);
	NDB_FREE(body.data);
	return 0;
}

/* Parse batch embedding response: [[emb1...], [emb2...], ...] */
static bool
parse_hf_emb_batch(const char *json,
				   float ***vecs_out,
				   int **dims_out,
				   int *num_vecs_out)
{
	const char *p;
	float	  **vecs = NULL;
	int		   *dims = NULL;
	int			num_vecs = 0;
	int			cap = 16;
	char	   *endptr;
	double		v;
	float	   *vec = NULL;
	int			vec_dim = 0;
	int			vec_cap = 32;

	if (!json)
		return false;

	p = json;
	/* Skip to first '[' (outer array) */
	while (*p && *p != '[')
		p++;
	if (!*p)
		return false;
	p++;
	while (*p && isspace(*p))
		p++;

	NDB_ALLOC(vecs, float *, cap);
	NDB_ALLOC(dims, int, cap);

	/* Parse array of arrays */
	while (*p && *p != ']')
	{
		/* Skip whitespace and commas */
		while (*p && (isspace(*p) || *p == ','))
			p++;
		if (*p == ']')
			break;

		/* Expect '[' for start of inner array (vector) */
		if (*p != '[')
			break;
		p++;

		/* Parse vector elements */
		NDB_ALLOC(vec, float, vec_cap);
		vec_dim = 0;
		while (*p && *p != ']')
		{
			/* Skip whitespace and commas */
			while (*p && (isspace(*p) || *p == ','))
				p++;
			if (*p == ']')
				break;

			/* Parse float value */
			endptr = NULL;
			v = strtod(p, &endptr);
			if (endptr == p)
				break;
			if (vec_dim == vec_cap)
			{
				vec_cap *= 2;
				vec = repalloc(vec, sizeof(float) * vec_cap);
			}
			vec[vec_dim++] = (float) v;
			p = endptr;
		}

		/* Skip closing ']' of inner array */
		if (*p == ']')
			p++;

		/* Store vector if valid */
		if (vec_dim > 0)
		{
			if (num_vecs == cap)
			{
				cap *= 2;
				vecs = repalloc(vecs, sizeof(float *) * cap);
				dims = repalloc(dims, sizeof(int) * cap);
			}
			vecs[num_vecs] = vec;
			dims[num_vecs] = vec_dim;
			num_vecs++;
			vec = NULL;
			vec_dim = 0;
			vec_cap = 32;
		}
		else if (vec)
		{
			NDB_FREE(vec);
			vec = NULL;
		}
	}

	if (num_vecs > 0)
	{
		*vecs_out = vecs;
		*dims_out = dims;
		*num_vecs_out = num_vecs;
		return true;
	}
	else
	{
		if (vecs)
			NDB_FREE(vecs);
		if (dims)
			NDB_FREE(dims);
		return false;
	}
}

int
ndb_hf_embed_batch(const NdbLLMConfig *cfg,
				   const char **texts,
				   int num_texts,
				   float ***vecs_out,
				   int **dims_out,
				   int *num_success_out)
{
	StringInfoData url,
				body,
				inputs_json;
	char	   *resp = NULL;
	int			code;
	bool		ok;
	int			i;
	float	  **vecs = NULL;
	int		   *dims = NULL;
	int			num_vecs = 0;
	NdbLLMResp	temp_resp = {0};
	HfEndpointKind kind;

	initStringInfo(&url);
	initStringInfo(&body);
	initStringInfo(&inputs_json);

	if (texts == NULL || num_texts <= 0)
	{
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		NDB_FREE(inputs_json.data);
		return -1;
	}

	/* Build JSON array of input texts */
	appendStringInfoChar(&inputs_json, '[');
	for (i = 0; i < num_texts; i++)
	{
		if (i > 0)
			appendStringInfoChar(&inputs_json, ',');
		if (texts[i] != NULL)
		{
			char	   *quoted = ndb_json_quote_string(texts[i]);

			appendStringInfoString(&inputs_json, quoted);
			NDB_FREE(quoted);
		}
		else
		{
			appendStringInfoString(&inputs_json, "null");
		}
	}
	appendStringInfoChar(&inputs_json, ']');

	kind = hf_classify_endpoint(cfg->endpoint);

	switch (kind)
	{
		case HF_EP_ROUTER:
			if (strstr(cfg->endpoint, "/hf-inference") != NULL)
				appendStringInfo(&url,
								 "%s/models/%s/pipeline/feature-extraction",
								 cfg->endpoint,
								 cfg->model);
			else
				appendStringInfo(&url,
								 "%s/hf-inference/models/%s/pipeline/feature-extraction",
								 cfg->endpoint,
								 cfg->model);
			break;

		case HF_EP_API_INFERENCE:
			appendStringInfo(&url,
							 "%s/models/%s/pipeline/feature-extraction",
							 cfg->endpoint,
							 cfg->model);
			break;

		case HF_EP_GENERIC:
		default:
			appendStringInfo(&url,
							 "%s/pipeline/feature-extraction/%s",
							 cfg->endpoint,
							 cfg->model);
			break;
	}
	appendStringInfo(&body,
					 "{\"inputs\":%s,\"truncate\":true}",
					 inputs_json.data);

	code = http_post_json(
						  url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);

	NDB_FREE(url.data);
	NDB_FREE(body.data);
	NDB_FREE(inputs_json.data);

	/* Handle all HTTP response types */
	if (!handle_http_response(code, &resp, &temp_resp))
	{
		if (resp)
			NDB_FREE(resp);
		return -1;
	}

	ok = parse_hf_emb_batch(resp, &vecs, &dims, &num_vecs);
	NDB_FREE(resp);

	if (!ok)
	{
		if (vecs)
		{
			for (i = 0; i < num_vecs; i++)
			{
				if (vecs[i])
					NDB_FREE(vecs[i]);
			}
			NDB_FREE(vecs);
		}
		if (dims)
			NDB_FREE(dims);
		return -1;
	}

	*vecs_out = vecs;
	*dims_out = dims;
	*num_success_out = num_vecs;
	return 0;
}

int
ndb_hf_image_embed(const NdbLLMConfig *cfg,
				   const unsigned char *image_data,
				   size_t image_size,
				   float **vec_out,
				   int *dim_out)
{
	StringInfoData url,
				body;
	char	   *resp = NULL;
	int			code;
	bool		ok;
	char	   *base64_data = NULL;
	text	   *encoded_text = NULL;
	NdbLLMResp	temp_resp = {0};
	HfEndpointKind kind;

	initStringInfo(&url);
	initStringInfo(&body);

	if (image_data == NULL || image_size == 0)
	{
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		return -1;
	}

	/*
	 * Convert image data to bytea, then base64 encode using PostgreSQL's
	 * encode()
	 */
	{
		bytea	   *image_bytea = NULL;

		NDB_DECLARE(char *, image_bytea_raw);
		NDB_ALLOC(image_bytea_raw, char, VARHDRSZ + image_size);
		image_bytea = (bytea *) image_bytea_raw;
		SET_VARSIZE(image_bytea, VARHDRSZ + image_size);
		memcpy(VARDATA(image_bytea), image_data, image_size);

		/* Use PostgreSQL's encode() function for base64 */
		encoded_text = ndb_encode_base64(image_bytea);
		base64_data = text_to_cstring(encoded_text);

		NDB_FREE(image_bytea);
		NDB_FREE(encoded_text);
	}

	/* Build URL and JSON body for HuggingFace CLIP API */
	kind = hf_classify_endpoint(cfg->endpoint);

	switch (kind)
	{
		case HF_EP_ROUTER:
			if (strstr(cfg->endpoint, "/hf-inference") != NULL)
				appendStringInfo(&url,
								 "%s/models/%s/pipeline/feature-extraction",
								 cfg->endpoint,
								 cfg->model);
			else
				appendStringInfo(&url,
								 "%s/hf-inference/models/%s/pipeline/feature-extraction",
								 cfg->endpoint,
								 cfg->model);
			break;

		case HF_EP_API_INFERENCE:
			appendStringInfo(&url,
							 "%s/models/%s/pipeline/feature-extraction",
							 cfg->endpoint,
							 cfg->model);
			break;

		case HF_EP_GENERIC:
		default:
			appendStringInfo(&url,
							 "%s/pipeline/feature-extraction/%s",
							 cfg->endpoint,
							 cfg->model);
			break;
	}

	/* HuggingFace expects image in data URI format */
	appendStringInfo(&body,
					 "{\"inputs\":{\"image\":\"data:image/jpeg;base64,%s\"}}",
					 base64_data);

	code = http_post_json(
						  url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);

	NDB_FREE(url.data);
	NDB_FREE(body.data);
	NDB_FREE(base64_data);

	/* Handle all HTTP response types */
	if (!handle_http_response(code, &resp, &temp_resp))
	{
		if (resp)
			NDB_FREE(resp);
		return -1;
	}

	ok = parse_hf_emb_vector(resp, vec_out, dim_out);
	NDB_FREE(resp);

	if (!ok)
		return -1;
	return 0;
}

int
ndb_hf_multimodal_embed(const NdbLLMConfig *cfg,
						const char *text_input,
						const unsigned char *image_data,
						size_t image_size,
						float **vec_out,
						int *dim_out)
{
	StringInfoData url,
				body;
	char	   *resp = NULL;
	int			code;
	bool		ok;
	char	   *base64_data = NULL;
	text	   *encoded_text = NULL;
	char	   *quoted_text = NULL;
	NdbLLMResp	temp_resp = {0};
	HfEndpointKind kind;

	initStringInfo(&url);
	initStringInfo(&body);

	if (text_input == NULL || image_data == NULL || image_size == 0)
	{
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		return -1;
	}

	/* Base64 encode image */
	{
		bytea	   *image_bytea = NULL;

		NDB_DECLARE(char *, image_bytea_raw);
		NDB_ALLOC(image_bytea_raw, char, VARHDRSZ + image_size);
		image_bytea = (bytea *) image_bytea_raw;
		SET_VARSIZE(image_bytea, VARHDRSZ + image_size);
		memcpy(VARDATA(image_bytea), image_data, image_size);

		encoded_text = ndb_encode_base64(image_bytea);
		base64_data = text_to_cstring(encoded_text);

		NDB_FREE(image_bytea);
		NDB_FREE(encoded_text);
	}

	/* Quote text for JSON */
	quoted_text = ndb_json_quote_string(text_input);

	/* Build URL and JSON body for HuggingFace CLIP multimodal API */
	kind = hf_classify_endpoint(cfg->endpoint);

	switch (kind)
	{
		case HF_EP_ROUTER:
			if (strstr(cfg->endpoint, "/hf-inference") != NULL)
				appendStringInfo(&url,
								 "%s/models/%s/pipeline/feature-extraction",
								 cfg->endpoint,
								 cfg->model);
			else
				appendStringInfo(&url,
								 "%s/hf-inference/models/%s/pipeline/feature-extraction",
								 cfg->endpoint,
								 cfg->model);
			break;

		case HF_EP_API_INFERENCE:
			appendStringInfo(&url,
							 "%s/models/%s/pipeline/feature-extraction",
							 cfg->endpoint,
							 cfg->model);
			break;

		case HF_EP_GENERIC:
		default:
			appendStringInfo(&url,
							 "%s/pipeline/feature-extraction/%s",
							 cfg->endpoint,
							 cfg->model);
			break;
	}

	/* HuggingFace CLIP expects both text and image in inputs */
	appendStringInfo(&body,
					 "{\"inputs\":{\"text\":%s,\"image\":\"data:image/jpeg;base64,%s\"}}",
					 quoted_text,
					 base64_data);

	code = http_post_json(
						  url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);

	NDB_FREE(url.data);
	NDB_FREE(body.data);
	NDB_FREE(base64_data);
	NDB_FREE(quoted_text);

	/* Handle all HTTP response types */
	if (!handle_http_response(code, &resp, &temp_resp))
	{
		if (resp)
			NDB_FREE(resp);
		return -1;
	}

	ok = parse_hf_emb_vector(resp, vec_out, dim_out);
	NDB_FREE(resp);

	if (!ok)
		return -1;
	return 0;
}

static bool
parse_hf_scores(const char *json, float **scores_out, int ndocs)
{
	/*
	 * The response is [{"scores":[float, float,...]}] or similar; We will
	 * parse for the first float array in the string.
	 */
	const char *scores_key = "\"scores\"";
	char	   *ps;
	float	   *scores;
	int			n = 0;
	char	   *endptr;
	double		v;

	if (!json)
		return false;
	ps = strstr(json, scores_key);
	if (!ps)
		return false;
	ps = strchr(ps, '[');
	if (!ps)
		return false;
	ps++;
	NDB_ALLOC(scores, float, ndocs);
	while (*ps && *ps != ']' && n < ndocs)
	{
		while (*ps && (isspace(*ps) || *ps == ','))
			ps++;
		endptr = NULL;
		v = strtod(ps, &endptr);
		if (endptr == ps)
			break;
		scores[n++] = (float) v;
		ps = endptr;
	}
	if (n == ndocs)
	{
		*scores_out = scores;
		return true;
	}
	NDB_FREE(scores);
	return false;
}

int
ndb_hf_rerank(const NdbLLMConfig *cfg,
			  const char *query,
			  const char **docs,
			  int ndocs,
			  float **scores_out)
{
	StringInfoData url,
				body;
	StringInfoData docs_json;
	char	   *resp = NULL;
	int			code;
	int			i;
	bool		ok;
	NdbLLMResp	temp_resp = {0};
	HfEndpointKind kind;

	initStringInfo(&url);
	initStringInfo(&body);

	/* Validate inputs */
	if (query == NULL || docs == NULL || ndocs <= 0)
	{
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		return -1;
	}

	/* Validate API key is required for HuggingFace inference API */
	if (!cfg->api_key || cfg->api_key[0] == '\0')
	{
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("API key is required for HuggingFace but was not provided"),
				 errhint("Set neurondb.llm_api_key configuration parameter")));
		return -1;
	}

	/* Compose the docs JSON array */
	initStringInfo(&docs_json);
	appendStringInfoChar(&docs_json, '[');
	for (i = 0; i < ndocs; ++i)
	{
		if (i > 0)
			appendStringInfoChar(&docs_json, ',');
		if (docs[i] != NULL)
			appendStringInfoString(&docs_json, ndb_json_quote_string(docs[i]));
		else
			appendStringInfoString(&docs_json, "null");
	}
	appendStringInfoChar(&docs_json, ']');

	kind = hf_classify_endpoint(cfg->endpoint);

	switch (kind)
	{
		case HF_EP_ROUTER:
			if (strstr(cfg->endpoint, "/hf-inference") != NULL)
				appendStringInfo(&url, "%s/models/%s",
								 cfg->endpoint, cfg->model);
			else
				appendStringInfo(&url, "%s/hf-inference/models/%s",
								 cfg->endpoint, cfg->model);
			break;

		case HF_EP_API_INFERENCE:
			appendStringInfo(&url, "%s/models/%s",
							 cfg->endpoint, cfg->model);
			break;

		case HF_EP_GENERIC:
		default:
			/* Generic old style: keep original format for custom endpoints */
			appendStringInfo(&url,
							 "%s/pipeline/token-classification/%s",
							 cfg->endpoint,
							 cfg->model);
			break;
	}

	appendStringInfo(&body,
					 "{\"inputs\":{\"query\":%s,\"documents\":%s}}",
					 ndb_json_quote_string(query),
					 docs_json.data);

	code = http_post_json(
						  url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);

	/* Handle all HTTP response types */
	if (!handle_http_response(code, &resp, &temp_resp))
	{
		if (resp)
			NDB_FREE(resp);
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		NDB_FREE(docs_json.data);
		return -1;
	}

	ok = parse_hf_scores(resp, scores_out, ndocs);
	NDB_FREE(resp);
	NDB_FREE(url.data);
	NDB_FREE(body.data);
	NDB_FREE(docs_json.data);
	if (!ok)
		return -1;
	return 0;
}
