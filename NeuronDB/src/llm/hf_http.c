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

/* Function prototypes to fix implicit declaration errors */
/* ndb_json_quote_string is now replaced by ndb_json_quote_string from neurondb_json.h */
static text * ndb_encode_base64(bytea * data);
int			http_post_json(const char *url, const char *api_key, const char *body, int timeout_ms, char **resp_out);

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
	if (cfg->endpoint && (strstr(cfg->endpoint, "router.huggingface.co") != NULL ||
						  strstr(cfg->endpoint, "hf-inference") != NULL))
	{
		appendStringInfo(&url,
						 "%s/models/%s/pipeline/image-to-text",
						 cfg->endpoint,
						 cfg->model);
	}
	else
	{
		appendStringInfo(&url,
						 "%s/pipeline/image-to-text/%s",
						 cfg->endpoint,
						 cfg->model);
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

	if (code < 200 || code >= 300 || !resp)
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
		return -1;

	headers = curl_slist_append(headers, "Content-Type: application/json");
	if (api_key && api_key[0])
	{
		StringInfoData h;

		initStringInfo(&h);
		appendStringInfo(&h, "Authorization: Bearer %s", api_key);
		headers = curl_slist_append(headers, h.data);
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
		return -1;
	}
	curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);

	*out = buf.data;
	return (int) code;
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
	char	   *out;

	if (!json || json[0] == '\0')
		return NULL;
	if (strncmp(json, "{\"error\"", 8) == 0)
		return NULL;

	/* Find first "generated_text":"..." pattern */
	key = "\"generated_text\":";

	p = strstr(json, key);
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

	appendStringInfo(&url, "%s/models/%s", cfg->endpoint, cfg->model);
	appendStringInfo(&body,
					 "{\"inputs\":%s,\"parameters\":%s}",
					 ndb_json_quote_string(prompt),
					 params_json ? params_json : "{}");

	out->http_status = http_post_json(
									  url.data, cfg->api_key, body.data, cfg->timeout_ms, &out->json);

	out->text = NULL;

	if (out->json && out->http_status >= 200 && out->http_status < 300)
	{
		/* Try to parse a "generated_text" value out */
		char	   *t = extract_hf_text(out->json);

		if (t)
			out->text = t;
		else
			out->text = pstrdup(out->json);
	}
	else if (out->json)
	{
		out->text = NULL;
	}

	return (out->http_status >= 200 && out->http_status < 300) ? 0 : -1;
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

	/*
	 * Router endpoint (router.huggingface.co) uses new path format for
	 * embeddings
	 */
	/* New format: /hf-inference/models/<model>/pipeline/feature-extraction */
	/* Old format: /pipeline/feature-extraction/<model> */
	if (cfg->endpoint && strstr(cfg->endpoint, "router.huggingface.co") != NULL)
	{
		/*
		 * Router endpoint: add /hf-inference if missing, use new format with
		 * models path
		 */
		if (strstr(cfg->endpoint, "/hf-inference") != NULL)
		{
			/* Endpoint already includes /hf-inference */
			appendStringInfo(&url,
							 "%s/models/%s/pipeline/feature-extraction",
							 cfg->endpoint,
							 cfg->model);
		}
		else
		{
			/* Router endpoint without /hf-inference - add it */
			appendStringInfo(&url,
							 "%s/hf-inference/models/%s/pipeline/feature-extraction",
							 cfg->endpoint,
							 cfg->model);
		}
	}
	else if (cfg->endpoint && strstr(cfg->endpoint, "hf-inference") != NULL)
	{
		/* Endpoint contains hf-inference (could be dedicated endpoint) */
		appendStringInfo(&url,
						 "%s/models/%s/pipeline/feature-extraction",
						 cfg->endpoint,
						 cfg->model);
	}
	else
	{
		/*
		 * Old endpoint format (deprecated but kept for backward
		 * compatibility)
		 */
		appendStringInfo(&url,
						 "%s/pipeline/feature-extraction/%s",
						 cfg->endpoint,
						 cfg->model);
	}
	appendStringInfo(&body,
					 "{\"inputs\":%s,\"truncate\":true}",
					 ndb_json_quote_string(text));
	code = http_post_json(
						  url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);
	if (code < 200 || code >= 300 || !resp)
	{
		if (resp)
			NDB_FREE(resp);
		NDB_FREE(url.data);
		NDB_FREE(body.data);
		return -1;
	}
	/* Check for error in response body (API may return 200 with error JSON) */
	if (strncmp(resp, "{\"error\"", 8) == 0)
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

	/* Router endpoint uses new path format for embeddings */
	if (cfg->endpoint && (strstr(cfg->endpoint, "router.huggingface.co") != NULL ||
						  strstr(cfg->endpoint, "hf-inference") != NULL))
	{
		/* Router endpoint: use new format with models path */
		appendStringInfo(&url,
						 "%s/models/%s/pipeline/feature-extraction",
						 cfg->endpoint,
						 cfg->model);
	}
	else
	{
		/* Old endpoint format */
		appendStringInfo(&url,
						 "%s/pipeline/feature-extraction/%s",
						 cfg->endpoint,
						 cfg->model);
	}
	appendStringInfo(&body,
					 "{\"inputs\":%s,\"truncate\":true}",
					 inputs_json.data);

	code = http_post_json(
						  url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);

	NDB_FREE(url.data);
	NDB_FREE(body.data);
	NDB_FREE(inputs_json.data);

	if (code < 200 || code >= 300 || !resp)
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
	/* Router endpoint uses new path format for embeddings */
	if (cfg->endpoint && (strstr(cfg->endpoint, "router.huggingface.co") != NULL ||
						  strstr(cfg->endpoint, "hf-inference") != NULL))
	{
		/* Router endpoint: use new format with models path */
		appendStringInfo(&url,
						 "%s/models/%s/pipeline/feature-extraction",
						 cfg->endpoint,
						 cfg->model);
	}
	else
	{
		/* Old endpoint format */
		appendStringInfo(&url,
						 "%s/pipeline/feature-extraction/%s",
						 cfg->endpoint,
						 cfg->model);
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

	if (code < 200 || code >= 300 || !resp)
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
	/* Router endpoint uses new path format for embeddings */
	if (cfg->endpoint && (strstr(cfg->endpoint, "router.huggingface.co") != NULL ||
						  strstr(cfg->endpoint, "hf-inference") != NULL))
	{
		/* Router endpoint: use new format with models path */
		appendStringInfo(&url,
						 "%s/models/%s/pipeline/feature-extraction",
						 cfg->endpoint,
						 cfg->model);
	}
	else
	{
		/* Old endpoint format */
		appendStringInfo(&url,
						 "%s/pipeline/feature-extraction/%s",
						 cfg->endpoint,
						 cfg->model);
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

	if (code < 200 || code >= 300 || !resp)
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

	/* URL: .../pipeline/text-classification/model */
	appendStringInfo(&url,
					 "%s/pipeline/token-classification/%s",
					 cfg->endpoint,
					 cfg->model);
	/* Use models endpoint if above fails for reranking */
	appendStringInfo(&body,
					 "{\"inputs\":{\"query\":%s,\"documents\":%s}}",
					 ndb_json_quote_string(query),
					 docs_json.data);

	code = http_post_json(
						  url.data, cfg->api_key, body.data, cfg->timeout_ms, &resp);

	if (code < 200 || code >= 300 || !resp)
	{
		if (resp)
			NDB_FREE(resp);
		return -1;
	}

	ok = parse_hf_scores(resp, scores_out, ndocs);
	NDB_FREE(resp);
	if (!ok)
		return -1;
	return 0;
}
