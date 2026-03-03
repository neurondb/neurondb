#include "postgres.h"
#include "neurondb_macros.h"

#include "utils/builtins.h"
#include <ctype.h>
#include <math.h>
#include <string.h>

#include "neurondb_llm.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_hf.h"
#endif

#ifdef HAVE_ONNX_RUNTIME
#include "neurondb_onnx.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#endif

static bool provider_is(const char *provider, const char *name);
static int	fallback_complete(const NdbLLMConfig *cfg,
							  const NdbLLMCallOptions *opts,
							  const char *reason,
							  const char *prompt,
							  const char *params_json,
							  NdbLLMResp *out);
static int	fallback_embed(const NdbLLMConfig *cfg,
						   const NdbLLMCallOptions *opts,
						   const char *reason,
						   const char *text,
						   float **vec_out,
						   int *dim_out);
static int	fallback_rerank(const NdbLLMConfig *cfg,
							const NdbLLMCallOptions *opts,
							const char *reason,
							const char *query,
							const char **docs,
							int ndocs,
							float **scores_out);

static bool
provider_is(const char *provider, const char *name)
{
	if (provider == NULL)
		return false;
	return pg_strcasecmp(provider, name) == 0;
}

static int
fallback_complete(const NdbLLMConfig *cfg,
				  const NdbLLMCallOptions *opts,
				  const char *reason,
				  const char *prompt,
				  const char *params_json,
				  NdbLLMResp *out)
{
	/* Validate inputs */
	if (cfg == NULL || prompt == NULL || out == NULL)
	{
		elog(WARNING, "neurondb: fallback_complete called with NULL parameters");
		return NDB_LLM_ROUTE_ERROR;
	}

	if (opts != NULL && opts->require_gpu)
	{
		ereport(ERROR,
				(errmsg("neurondb: LLM provider \"%s\" requires GPU "
						"but is unavailable locally (%s)",
						cfg->provider ? cfg->provider
						: "huggingface-local",
						reason ? reason : "not supported")));
		return NDB_LLM_ROUTE_ERROR;
	}
	if (opts != NULL && opts->fail_open)
	{
		NdbLLMConfig fallback = *cfg;

		fallback.provider = "huggingface";
		return ndb_hf_complete(&fallback, prompt, params_json, out);
	}

	ereport(ERROR,
			(errmsg("neurondb: LLM provider \"%s\" unavailable locally "
					"(%s)",
					cfg->provider ? cfg->provider : "huggingface-local",
					reason ? reason : "not supported")));
	return NDB_LLM_ROUTE_ERROR; /* unreachable */
}

static int
fallback_embed(const NdbLLMConfig *cfg,
			   const NdbLLMCallOptions *opts,
			   const char *reason,
			   const char *text,
			   float **vec_out,
			   int *dim_out)
{
	/* Validate inputs */
	if (cfg == NULL || text == NULL || vec_out == NULL || dim_out == NULL)
	{
		elog(WARNING, "neurondb: fallback_embed called with NULL parameters");
		return NDB_LLM_ROUTE_ERROR;
	}

	if (opts != NULL && opts->require_gpu)
	{
		ereport(ERROR,
				(errmsg("neurondb: LLM provider \"%s\" requires GPU "
						"but is unavailable locally (%s)",
						cfg->provider ? cfg->provider
						: "huggingface-local",
						reason ? reason : "not supported")));
		return NDB_LLM_ROUTE_ERROR;
	}
	if (opts != NULL && opts->fail_open)
	{
		NdbLLMConfig fallback = *cfg;

		fallback.provider = "huggingface";
		return ndb_hf_embed(&fallback, text, vec_out, dim_out);
	}

	ereport(ERROR,
			(errmsg("neurondb: LLM provider \"%s\" unavailable locally "
					"(%s)",
					cfg->provider ? cfg->provider : "huggingface-local",
					reason ? reason : "not supported")));
	return NDB_LLM_ROUTE_ERROR; /* unreachable */
}

static int
fallback_rerank(const NdbLLMConfig *cfg,
				const NdbLLMCallOptions *opts,
				const char *reason,
				const char *query,
				const char **docs,
				int ndocs,
				float **scores_out)
{
	/* Validate inputs */
	if (cfg == NULL || query == NULL || docs == NULL || scores_out == NULL || ndocs <= 0)
	{
		elog(WARNING, "neurondb: fallback_rerank called with NULL or invalid parameters");
		return NDB_LLM_ROUTE_ERROR;
	}

	if (opts != NULL && opts->require_gpu)
	{
		ereport(ERROR,
				(errmsg("neurondb: LLM provider \"%s\" requires GPU "
						"but is unavailable locally (%s)",
						cfg->provider ? cfg->provider
						: "huggingface-local",
						reason ? reason : "not supported")));
		return NDB_LLM_ROUTE_ERROR;
	}
	if (opts != NULL && opts->fail_open)
	{
		NdbLLMConfig fallback = *cfg;

		fallback.provider = "huggingface";
		return ndb_hf_rerank(&fallback, query, docs, ndocs, scores_out);
	}

	ereport(ERROR,
			(errmsg("neurondb: LLM provider \"%s\" unavailable locally "
					"(%s)",
					cfg->provider ? cfg->provider : "huggingface-local",
					reason ? reason : "not supported")));
	return NDB_LLM_ROUTE_ERROR; /* unreachable */
}

int
ndb_llm_route_complete(const NdbLLMConfig *cfg,
					   const NdbLLMCallOptions *opts,
					   const char *prompt,
					   const char *params_json,
					   NdbLLMResp *out)
{
	if (cfg == NULL || prompt == NULL || out == NULL)
	{
		elog(WARNING, "neurondb: ndb_llm_route_complete called with NULL parameters");
		return NDB_LLM_ROUTE_ERROR;
	}

	/* Validate prompt is not empty */
	if (strlen(prompt) == 0)
	{
		elog(WARNING, "neurondb: ndb_llm_route_complete called with empty prompt");
		return NDB_LLM_ROUTE_ERROR;
	}

	if (provider_is(cfg->provider, "openai") || provider_is(cfg->provider, "chatgpt"))
	{
		return ndb_openai_complete(cfg, prompt, params_json, out);
	}

	if (cfg->provider == NULL || provider_is(cfg->provider, "huggingface")
		|| provider_is(cfg->provider, "hf-http"))
	{
		return ndb_hf_complete(cfg, prompt, params_json, out);
	}

	if (provider_is(cfg->provider, "huggingface-local")
		|| provider_is(cfg->provider, "hf-local"))
	{
		if (neurondb_gpu_is_available()
			&& (opts == NULL || opts->prefer_gpu
				|| opts->require_gpu))
		{
			char	   *gpu_err = NULL;
			char	   *gpu_text = NULL;
			int			rc;

			rc = neurondb_gpu_hf_complete(cfg->model,
										  prompt,
										  params_json,
										  &gpu_text,
										  &gpu_err);
			if (rc == 0)
			{

				/* Validate gpu_text before use */
				if (gpu_text == NULL)
				{
					elog(WARNING, "neurondb: GPU completion returned NULL text");
					if (gpu_err)
						pfree(gpu_err);
					return NDB_LLM_ROUTE_ERROR;
				}

#ifdef HAVE_ONNX_RUNTIME
			{
				int32		token_length;
				int32	   *token_ids = NULL;

				token_ids = neurondb_tokenize_with_model(prompt,
														 2048,
														 &token_length,
														 cfg->model);
				if (token_ids && token_length > 0)
					out->tokens_in = token_length;
				else
					out->tokens_in =
						0;
				if (token_ids)
					pfree(token_ids);
			}
#else
			out->tokens_in = 0;
#endif

#ifdef HAVE_ONNX_RUNTIME
				PG_TRY();
				{
					int32		output_token_length;
					int32	   *output_token_ids =
						neurondb_tokenize_with_model(
													 gpu_text,
													 2048,
													 &output_token_length,
													 cfg->model);

					if (output_token_ids
						&& output_token_length > 0)
					{
						out->tokens_out =
							output_token_length;
					}
					else
					{
					if (gpu_text
						&& strlen(gpu_text) > 0)
					{
						const char *ptr = gpu_text;
						int			in_word = 0;
						int			word_count = 0;

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
								in_word =
									0;
							}
							ptr++;
							}
							out->tokens_out =
								word_count > 0
								? word_count
								: 1;
						}
						else
						{
							out->tokens_out = 0;
						}
					}
					if (output_token_ids)
						pfree(output_token_ids);
				}
				PG_CATCH();
				{
					EmitErrorReport();
					FlushErrorState();

					if (gpu_text && strlen(gpu_text) > 0)
					{
						const char *ptr = gpu_text;
						int			in_word = 0;
						int			word_count = 0;

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
						out->tokens_out = word_count > 0
							? word_count
							: 1;
					}
					else
					{
						out->tokens_out = 0;
					}
				}
				PG_END_TRY();
#else
			if (gpu_text && strlen(gpu_text) > 0)
			{
				const char *ptr = gpu_text;
				int			in_word = 0;
				int			word_count = 0;

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
					out->tokens_out =
						word_count > 0 ? word_count : 1;
				}
				else
				{
					out->tokens_out = 0;
				}
#endif

				out->text = gpu_text ? gpu_text : pstrdup("");
				out->json = NULL;
				out->http_status = 200;
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (gpu_err)
				{
					elog(ERROR,
						 "neurondb: GPU HF completion failed: %s",
						 gpu_err);
					pfree(gpu_err);
				}
				else
				{
					elog(ERROR,
						 "neurondb: GPU HF completion failed (no error message)");
				}
				if (gpu_text)
					pfree(gpu_text);
				return NDB_LLM_ROUTE_ERROR;
			}
			/* GPU not required, log warning and continue */
			if (gpu_err)
			{
				pfree(gpu_err);
			}
			if (gpu_text)
				pfree(gpu_text);
		}
#ifdef HAVE_ONNX_RUNTIME
		if (!neurondb_onnx_available())
			return fallback_complete(cfg,
									 opts,
									 "ONNX runtime not available",
									 prompt,
									 params_json,
									 out);
		{
			char *onnx_text = NULL;
			char *onnx_err = NULL;
			int			rc;

			rc = ndb_onnx_hf_complete(cfg->model,
									  prompt,
									  params_json,
									  &onnx_text,
									  &onnx_err);
			if (rc == 0 && onnx_text != NULL)
			{
				int32	   *token_ids = NULL;
				int			token_length = 0;

				/* Validate onnx_text before use */
				if (strlen(onnx_text) == 0)
				{
					elog(WARNING, "neurondb: ONNX completion returned empty text");
					pfree(onnx_text);
					onnx_text = NULL;
					if (onnx_err)
						pfree(onnx_err);
					return fallback_complete(cfg,
											 opts,
											 "ONNX completion returned empty text",
											 prompt,
											 params_json,
											 out);
				}

				token_ids = neurondb_tokenize_with_model(prompt,
														 2048,
														 &token_length,
														 cfg->model);
				if (token_ids && token_length > 0)
					out->tokens_in = token_length;
				else
					out->tokens_in =
						0;
				if (token_ids)
					pfree(token_ids);

#ifdef HAVE_ONNX_RUNTIME
				PG_TRY();
				{
					int			output_token_length;
					int32	   *output_token_ids = NULL;

					output_token_ids = neurondb_tokenize_with_model(
																   onnx_text,
																   2048,
																   &output_token_length,
																   cfg->model);

					if (output_token_ids
						&& output_token_length > 0)
					{
						out->tokens_out =
							output_token_length;
					}
					else
					{
						if (onnx_text
							&& strlen(onnx_text)
							> 0)
						{
							const char *ptr = onnx_text;
							int			in_word = 0;
							int			word_count = 0;

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
									in_word =
										0;
								}
								ptr++;
							}
							out->tokens_out =
								word_count > 0
								? word_count
								: 1;
						}
						else
						{
							out->tokens_out = 0;
						}
					}
					if (output_token_ids)
						pfree(output_token_ids);
				}
				PG_CATCH();
				{
					EmitErrorReport();
					FlushErrorState();

				if (onnx_text && strlen(onnx_text) > 0)
				{
					const char *ptr = onnx_text;
					int			in_word = 0;
					int			word_count = 0;

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
						out->tokens_out = word_count > 0
							? word_count
							: 1;
					}
					else
					{
						out->tokens_out = 0;
					}
				}
				PG_END_TRY();
#else
				if (onnx_text && strlen(onnx_text) > 0)
				{
					const char *ptr = onnx_text;
					int			in_word = 0;
					int			word_count = 0;

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
					out->tokens_out =
						word_count > 0 ? word_count : 1;
				}
				else
				{
					out->tokens_out = 0;
				}
#endif

				out->text = onnx_text;
				out->json = NULL;
				out->http_status = 200;
				if (onnx_err)
					pfree(onnx_err);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (onnx_err)
				{
					elog(ERROR,
						 "neurondb: ONNX HF completion failed: %s",
						 onnx_err);
					pfree(onnx_err);
				}
				else
				{
					elog(ERROR,
						 "neurondb: ONNX HF completion failed (no error message)");
				}
				if (onnx_text)
					pfree(onnx_text);
				return NDB_LLM_ROUTE_ERROR;
			}
			/* ONNX not required, log warning and fallback */
			{
				const char *fallback_reason = "ONNX completion failed";

				if (onnx_err)
				{
					fallback_reason = onnx_err;
				}
				if (onnx_text)
					pfree(onnx_text);
				if (onnx_err)
					pfree(onnx_err);
				return fallback_complete(cfg,
										 opts,
										 fallback_reason,
										 prompt,
										 params_json,
										 out);
			}
		}
#else
		return fallback_complete(cfg,
								 opts,
								 "compiled without ONNX runtime",
								 prompt,
								 params_json,
								 out);
#endif
	}

	ereport(ERROR,
			(errmsg("neurondb: unknown LLM provider \"%s\", using remote "
					"Hugging Face",
					cfg->provider)));
	return ndb_hf_complete(cfg, prompt, params_json, out);
}

/*
 * ndb_llm_route_vision_complete
 *	  Route vision (image-to-text) requests to appropriate backend
 */
int
ndb_llm_route_vision_complete(const NdbLLMConfig *cfg,
							  const NdbLLMCallOptions *opts,
							  const unsigned char *image_data,
							  size_t image_size,
							  const char *prompt,
							  const char *params_json,
							  NdbLLMResp *out)
{
	if (cfg == NULL || image_data == NULL || image_size == 0 || out == NULL)
	{
		elog(WARNING, "neurondb: ndb_llm_route_vision_complete called with NULL or invalid parameters");
		return NDB_LLM_ROUTE_ERROR;
	}

	/* Validate prompt if provided (it's optional for vision) */
	if (prompt != NULL && strlen(prompt) == 0)
	{
		prompt = NULL; /* Will use default in backend */
	}

	if (provider_is(cfg->provider, "openai") || provider_is(cfg->provider, "chatgpt"))
		return ndb_openai_vision_complete(cfg, image_data, image_size, prompt, params_json, out);

	/* For HuggingFace, use vision models (BLIP, InstructBLIP, etc.) */
	if (cfg->provider == NULL || provider_is(cfg->provider, "huggingface")
		|| provider_is(cfg->provider, "hf-http"))
	{
		return ndb_hf_vision_complete(cfg, image_data, image_size, prompt, params_json, out);
	}

	/* Try local GPU vision models if available */
	if (provider_is(cfg->provider, "huggingface-local") || provider_is(cfg->provider, "hf-local"))
	{
		if (neurondb_gpu_is_available()
			&& (opts == NULL || opts->prefer_gpu || opts->require_gpu))
		{
			const ndb_gpu_backend *backend;
			char	   *gpu_text = NULL;
			char	   *gpu_err = NULL;
			int			gpu_rc;

			/* Try direct GPU backend implementation */
			backend = ndb_gpu_get_active_backend();
			if (backend != NULL && backend->hf_vision_complete != NULL)
			{
				gpu_rc = backend->hf_vision_complete(
													 cfg->model, image_data, image_size, prompt, params_json,
													 &gpu_text, &gpu_err);
				if (gpu_rc == 0 && gpu_text != NULL)
				{
					out->text = gpu_text;
					out->json = NULL;
					out->http_status = 200;
					out->tokens_in = 0; /* GPU doesn't provide token counts
										 * yet */
					out->tokens_out = 0;
					if (gpu_err)
						pfree(gpu_err);
					return NDB_LLM_ROUTE_SUCCESS;
				}
				if (gpu_text)
					pfree(gpu_text);
				if (gpu_err)
					pfree(gpu_err);
				if (opts != NULL && opts->require_gpu)
				{
					return NDB_LLM_ROUTE_ERROR;
				}
			}
			/* GPU not available or failed, fall back to HTTP API */
		}
		return ndb_hf_vision_complete(cfg, image_data, image_size, prompt, params_json, out);
	}

	/* Unknown provider */
	return NDB_LLM_ROUTE_ERROR;
}

int
ndb_llm_route_embed(const NdbLLMConfig *cfg,
					const NdbLLMCallOptions *opts,
					const char *text,
					float **vec_out,
					int *dim_out)
{
	if (cfg == NULL || text == NULL || vec_out == NULL || dim_out == NULL)
	{
		elog(WARNING, "neurondb: ndb_llm_route_embed called with NULL parameters");
		return NDB_LLM_ROUTE_ERROR;
	}

	/* Validate text is not empty */
	if (strlen(text) == 0)
	{
		elog(WARNING, "neurondb: ndb_llm_route_embed called with empty text");
		return NDB_LLM_ROUTE_ERROR;
	}

	if (provider_is(cfg->provider, "openai") || provider_is(cfg->provider, "chatgpt"))
		return ndb_openai_embed(cfg, text, vec_out, dim_out);

	if (cfg->provider == NULL || provider_is(cfg->provider, "huggingface")
		|| provider_is(cfg->provider, "hf-http"))
	{

		if (neurondb_gpu_is_available()
			&& (opts == NULL || opts->prefer_gpu || opts->require_gpu))
		{
			float	   *gpu_vec = NULL;
			int			gpu_dim = 0;
			char	   *gpu_err = NULL;
			int			rc;

			rc = neurondb_gpu_hf_embed(
									   cfg->model, text, &gpu_vec, &gpu_dim, &gpu_err);
			if (rc == 0 && gpu_vec != NULL && gpu_dim > 0)
			{
				*vec_out = gpu_vec;
				*dim_out = gpu_dim;
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (gpu_err)
					ereport(ERROR,
							(errmsg("neurondb: GPU HF "
									"embedding failed: %s",
									gpu_err)));
				ereport(ERROR,
						(errmsg("neurondb: GPU HF embedding "
								"failed")));
				if (gpu_vec)
					pfree(gpu_vec);
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_ERROR;
			}
			if (gpu_vec)
				pfree(gpu_vec);
			if (gpu_err)
				pfree(gpu_err);
		}
		/* Fall back to HTTP API */
		return ndb_hf_embed(cfg, text, vec_out, dim_out);
	}

	if (provider_is(cfg->provider, "huggingface-local")
		|| provider_is(cfg->provider, "hf-local"))
	{
		if (neurondb_gpu_is_available()
			&& (opts == NULL || opts->prefer_gpu
				|| opts->require_gpu))
		{
			float	   *gpu_vec = NULL;
			int			gpu_dim = 0;
			char	   *gpu_err = NULL;
			int			rc;

			rc = neurondb_gpu_hf_embed(
									   cfg->model, text, &gpu_vec, &gpu_dim, &gpu_err);
			if (rc == 0 && gpu_vec != NULL && gpu_dim > 0)
			{
				*vec_out = gpu_vec;
				*dim_out = gpu_dim;
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (gpu_err)
					ereport(ERROR,
							(errmsg("neurondb: GPU HF "
									"embedding failed: %s",
									gpu_err)));
				ereport(ERROR,
						(errmsg("neurondb: GPU HF embedding "
								"failed")));
				if (gpu_vec)
					pfree(gpu_vec);
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_ERROR;
			}
			if (gpu_vec)
				pfree(gpu_vec);
			if (gpu_err)
				pfree(gpu_err);
		}
#ifdef HAVE_ONNX_RUNTIME
		if (!neurondb_onnx_available())
			return fallback_embed(cfg,
								  opts,
								  "ONNX runtime not available",
								  text,
								  vec_out,
								  dim_out);
		{
			ONNXModelSession *session = NULL;
			int32 *token_ids = NULL;
			int32		token_length;
			ONNXTensor *input_tensor = NULL;
			ONNXTensor *output_tensor = NULL;
			float *input_data = NULL;
			int			i;
			int64		input_shape[2];
			int			rc = NDB_LLM_ROUTE_ERROR;

			if (!text || strlen(text) == 0)
				return fallback_embed(cfg,
									  opts,
									  "empty input text",
									  text,
									  vec_out,
									  dim_out);

			PG_TRY();
			{
				session = neurondb_onnx_get_or_load_model(
														  cfg->model, ONNX_MODEL_EMBEDDING);
				if (!session)
				{
					rc = fallback_embed(cfg,
										opts,
										"failed to load ONNX model",
										text,
										vec_out,
										dim_out);
				}
				else
				{
					token_ids =
						neurondb_tokenize_with_model(
													 text,
													 128,
													 &token_length,
													 cfg->model);
					if (!token_ids)
					{
						rc = fallback_embed(cfg,
											opts,
											"tokenization failed: "
											"returned NULL",
											text,
											vec_out,
											dim_out);
					}
					else if (token_length <= 0)
					{
						pfree(token_ids);
						token_ids = NULL;
						rc = fallback_embed(cfg,
											opts,
											"tokenization failed: "
											"empty result",
											text,
											vec_out,
											dim_out);
					}
					else
					{
						nalloc(input_data, float, token_length);
						for (i = 0; i < token_length;
							 i++)
							input_data[i] = (float)
								token_ids[i];

						input_shape[0] = 1;
						input_shape[1] = token_length;
						input_tensor =
							neurondb_onnx_create_tensor(
														input_data,
														input_shape,
														2);

						/*
						 * Note: neurondb_onnx_create_tensor uses Assert() and
						 * palloc(), so it will either succeed or throw an
						 * error. No NULL check needed.
						 */

						output_tensor =
							neurondb_onnx_run_inference(
														session,
														input_tensor);
						if (!output_tensor
							|| output_tensor->size
							<= 0)
						{
							if (input_tensor)
								neurondb_onnx_free_tensor(
														  input_tensor);
							if (input_data)
								pfree(input_data);
							if (token_ids)
								pfree(token_ids);
							rc = fallback_embed(cfg,
												opts,
												"ONNX "
												"inference "
												"failed: "
												"invalid "
												"output",
												text,
												vec_out,
												dim_out);
						}
						else
						{
							*dim_out =
								output_tensor
								->size;
							nalloc(*vec_out, float, *dim_out);
							memcpy(*vec_out,
								   output_tensor
								   ->data,
								   *dim_out
								   * sizeof(
											float));

							{
								float		sum =
									0.0f;

								for (i = 0; i
									 < *dim_out;
									 i++)
									sum += (*vec_out)
										[i]
										* (*vec_out)
										[i];
								sum = sqrtf(
											sum);
								if (sum > 0.0f)
								{
									for (i = 0;
										 i
										 < *dim_out;
										 i++)
										(*vec_out)
											[i] /=
											sum;
								}
							}

							if (input_tensor)
								neurondb_onnx_free_tensor(
														  input_tensor);
							if (output_tensor)
								neurondb_onnx_free_tensor(
														  output_tensor);
							if (input_data)
								pfree(input_data);
							if (token_ids)
								pfree(token_ids);

							rc = NDB_LLM_ROUTE_SUCCESS;
						}
					}
				}
			}
			PG_CATCH();
			{
				/* Cleanup on error */
				EmitErrorReport();
				FlushErrorState();

				if (input_tensor)
					neurondb_onnx_free_tensor(input_tensor);
				if (output_tensor)
					neurondb_onnx_free_tensor(
											  output_tensor);
				if (input_data)
					pfree(input_data);
				if (token_ids)
					pfree(token_ids);
				if (*vec_out)
				{
					pfree(*vec_out);
					*vec_out = NULL;
					*dim_out = 0;
				}

				/* Fall back to HTTP or error */
				rc = fallback_embed(cfg,
									opts,
									"ONNX inference error",
									text,
									vec_out,
									dim_out);
			}
			PG_END_TRY();

			return rc;
		}
#else
		return fallback_embed(cfg,
							  opts,
							  "compiled without ONNX runtime",
							  text,
							  vec_out,
							  dim_out);
#endif
	}

	ereport(ERROR,
			(errmsg("neurondb: unknown LLM provider \"%s\", using remote "
					"Hugging Face",
					cfg->provider)));
	return ndb_hf_embed(cfg, text, vec_out, dim_out);
}

int
ndb_llm_route_rerank(const NdbLLMConfig *cfg,
					 const NdbLLMCallOptions *opts,
					 const char *query,
					 const char **docs,
					 int ndocs,
					 float **scores_out)
{
	if (cfg == NULL || query == NULL || docs == NULL || scores_out == NULL)
	{
		elog(WARNING, "neurondb: ndb_llm_route_rerank called with NULL parameters");
		return NDB_LLM_ROUTE_ERROR;
	}

	/* Validate query is not empty */
	if (strlen(query) == 0)
	{
		elog(WARNING, "neurondb: ndb_llm_route_rerank called with empty query");
		return NDB_LLM_ROUTE_ERROR;
	}

	/* Validate ndocs is positive */
	if (ndocs <= 0)
	{
		elog(WARNING, "neurondb: ndb_llm_route_rerank called with invalid ndocs: %d", ndocs);
		return NDB_LLM_ROUTE_ERROR;
	}

	/* Validate docs array elements are not NULL */
	for (int i = 0; i < ndocs; i++)
	{
		if (docs[i] == NULL)
		{
			elog(WARNING, "neurondb: ndb_llm_route_rerank called with NULL doc at index %d", i);
			return NDB_LLM_ROUTE_ERROR;
		}
		if (strlen(docs[i]) == 0)
		{
		}
	}

	if (provider_is(cfg->provider, "openai") || provider_is(cfg->provider, "chatgpt"))
	{
		int			rc = ndb_openai_rerank(cfg, query, docs, ndocs, scores_out);

		return (rc == 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
	}

	if (cfg->provider == NULL || provider_is(cfg->provider, "huggingface")
		|| provider_is(cfg->provider, "hf-http"))
	{
		int			rc = ndb_hf_rerank(cfg, query, docs, ndocs, scores_out);

		return (rc == 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
	}

	if (provider_is(cfg->provider, "huggingface-local")
		|| provider_is(cfg->provider, "hf-local"))
	{
		if (neurondb_gpu_is_available()
			&& (opts == NULL || opts->prefer_gpu
				|| opts->require_gpu))
		{
			float	   *gpu_scores = NULL;
			char	   *gpu_err = NULL;
			int			rc;

			rc = neurondb_gpu_hf_rerank(cfg->model,
										query,
										docs,
										ndocs,
										&gpu_scores,
										&gpu_err);
			if (rc == 0 && gpu_scores != NULL)
			{
				*scores_out = gpu_scores;
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (gpu_err)
					ereport(ERROR,
							(errmsg("neurondb: GPU HF "
									"reranking failed: %s",
									gpu_err)));
				ereport(ERROR,
						(errmsg("neurondb: GPU HF reranking "
								"failed")));
				if (gpu_scores)
					pfree(gpu_scores);
				if (gpu_err)
					pfree(gpu_err);
				return NDB_LLM_ROUTE_ERROR;
			}
			if (gpu_scores)
				pfree(gpu_scores);
			if (gpu_err)
				pfree(gpu_err);
		}
#ifdef HAVE_ONNX_RUNTIME
		if (!neurondb_onnx_available())
			return fallback_rerank(cfg,
								   opts,
								   "ONNX runtime not available",
								   query,
								   docs,
								   ndocs,
								   scores_out);
		/* Try ONNX reranking */
		{
			float *onnx_scores = NULL;
			char *onnx_err = NULL;
			int			rc;

			rc = ndb_onnx_hf_rerank(cfg->model,
									query,
									docs,
									ndocs,
									&onnx_scores,
									&onnx_err);
			if (rc == 0 && onnx_scores != NULL)
			{
				*scores_out = onnx_scores;
				if (onnx_err)
					pfree(onnx_err);
				return NDB_LLM_ROUTE_SUCCESS;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (onnx_err)
					ereport(ERROR,
							(errmsg("neurondb: ONNX HF "
									"reranking failed: %s",
									onnx_err)));
				ereport(ERROR,
						(errmsg("neurondb: ONNX HF reranking "
								"failed")));
				if (onnx_scores)
					pfree(onnx_scores);
				if (onnx_err)
					pfree(onnx_err);
				return NDB_LLM_ROUTE_ERROR;
			}
			if (onnx_scores)
				pfree(onnx_scores);
			if (onnx_err)
				pfree(onnx_err);
			/* Fall back to HTTP */
			return fallback_rerank(cfg,
								   opts,
								   "ONNX reranking failed, falling back to HTTP",
								   query,
								   docs,
								   ndocs,
								   scores_out);
		}
#else
		return fallback_rerank(cfg,
							   opts,
							   "compiled without ONNX runtime",
							   query,
							   docs,
							   ndocs,
							   scores_out);
#endif
	}

	ereport(ERROR,
			(errmsg("neurondb: unknown LLM provider \"%s\", using remote "
					"Hugging Face",
					cfg->provider)));
	return ndb_hf_rerank(cfg, query, docs, ndocs, scores_out);
}

/*
 * ndb_llm_route_complete_batch
 *	  Route batch completion requests to appropriate backend (GPU, ONNX, HTTP)
 */
int
ndb_llm_route_complete_batch(const NdbLLMConfig *cfg,
							 const NdbLLMCallOptions *opts,
							 const char **prompts,
							 int num_prompts,
							 const char *params_json,
							 NdbLLMBatchResp *out)
{
	int			i;
	int			num_success = 0;

	if (cfg == NULL || prompts == NULL || out == NULL || num_prompts <= 0)
	{
		elog(WARNING, "neurondb: ndb_llm_route_complete_batch called with NULL or invalid parameters");
		return NDB_LLM_ROUTE_ERROR;
	}

	/* Validate prompts array elements */
	for (i = 0; i < num_prompts; i++)
	{
		if (prompts[i] == NULL)
		{
			elog(WARNING, "neurondb: ndb_llm_route_complete_batch called with NULL prompt at index %d", i);
			return NDB_LLM_ROUTE_ERROR;
		}
		if (strlen(prompts[i]) == 0)
		{
		}
	}

	/* Initialize output */
	out->num_items = num_prompts;
	out->num_success = 0;
		nalloc(out->texts, char *, num_prompts);
		MemSet(out->texts, 0, sizeof(char *) * num_prompts);
		nalloc(out->tokens_in, int, num_prompts);
		MemSet(out->tokens_in, 0, sizeof(int) * num_prompts);
		nalloc(out->tokens_out, int, num_prompts);
		MemSet(out->tokens_out, 0, sizeof(int) * num_prompts);
		nalloc(out->http_status, int, num_prompts);
		MemSet(out->http_status, 0, sizeof(int) * num_prompts);

	if (cfg->provider == NULL || provider_is(cfg->provider, "huggingface")
		|| provider_is(cfg->provider, "hf-http")
		|| provider_is(cfg->provider, "huggingface-local")
		|| provider_is(cfg->provider, "hf-local"))
	{
		/* Try GPU-accelerated batch inference first if GPU is available */
		if (neurondb_gpu_is_available()
			&& (opts == NULL || opts->prefer_gpu
				|| opts->require_gpu))
		{
#ifdef NDB_GPU_CUDA
			NdbCudaHfBatchResult *batch_results = NULL;
			int			rc;
			char	   *gpu_err = NULL;


			nalloc(batch_results, NdbCudaHfBatchResult, num_prompts);
			MemSet(batch_results, 0, sizeof(NdbCudaHfBatchResult) * num_prompts);
			rc = neurondb_gpu_hf_complete_batch(cfg->model,
												prompts,
												num_prompts,
												params_json,
												batch_results,
												&gpu_err);
			if (rc == 0)
			{
				/* Copy results to output */
				for (i = 0; i < num_prompts; i++)
				{
					if (batch_results[i].status == 0)
					{
						int32		token_length;
						int32 *token_ids = NULL;

						/* Validate prompt before tokenization */
						if (prompts[i] == NULL || strlen(prompts[i]) == 0)
						{
							elog(WARNING, "neurondb: batch completion: invalid prompt at index %d", i);
							out->texts[i] = NULL;
							out->tokens_in[i] = 0;
							out->tokens_out[i] = 0;
							out->http_status[i] = 500;
							continue;
						}

						token_ids = neurondb_tokenize_with_model(
														 prompts[i],
														 2048,
														 &token_length,
														 cfg->model);

						if (token_ids
							&& token_length > 0)
							out->tokens_in[i] =
								token_length;
						else
							out->tokens_in[i] =
								0;
						if (token_ids)
							pfree(token_ids);

						out->texts[i] =
							batch_results[i].text
							? batch_results[i].text
							: pstrdup("");
						out->tokens_out[i] =
							batch_results[i]
							.num_tokens;
						out->http_status[i] = 200;
						num_success++;
					}
					else
					{
						out->texts[i] = NULL;
						out->tokens_in[i] = 0;
						out->tokens_out[i] = 0;
						out->http_status[i] = 500;
						if (batch_results[i].error)
							pfree(batch_results[i]
									 .error);
					}
				}
				out->num_success = num_success;
				if (gpu_err)
					pfree(gpu_err);
				pfree(batch_results);
				return (num_success > 0) ? NDB_LLM_ROUTE_SUCCESS
					: NDB_LLM_ROUTE_ERROR;
			}
			if (opts != NULL && opts->require_gpu)
			{
				if (gpu_err)
					ereport(ERROR,
							(errmsg("neurondb: GPU HF "
									"batch completion "
									"failed: %s",
									gpu_err)));
				ereport(ERROR,
						(errmsg("neurondb: GPU HF batch "
								"completion failed")));
				if (gpu_err)
					pfree(gpu_err);
				pfree(batch_results);
				return NDB_LLM_ROUTE_ERROR;
			}
			if (gpu_err)
				pfree(gpu_err);
			pfree(batch_results);
#endif
		}
		else
		{
		}
		/* Fall back to sequential processing or ONNX */
		/* For now, process sequentially */
		for (i = 0; i < num_prompts; i++)
		{
			NdbLLMResp	resp;
			int			rc;

			/* Validate prompt before processing */
			if (prompts[i] == NULL || strlen(prompts[i]) == 0)
			{
				out->texts[i] = NULL;
				out->tokens_in[i] = 0;
				out->tokens_out[i] = 0;
				out->http_status[i] = 500;
				continue;
			}

			memset(&resp, 0, sizeof(NdbLLMResp));
			rc = ndb_llm_route_complete(
										cfg, opts, prompts[i], params_json, &resp);
			if (rc == 0)
			{
				out->texts[i] = resp.text ? resp.text : pstrdup("");
				out->tokens_in[i] = resp.tokens_in;
				out->tokens_out[i] = resp.tokens_out;
				out->http_status[i] = resp.http_status;
				num_success++;
			}
			else
			{
				out->texts[i] = NULL;
				out->tokens_in[i] = 0;
				out->tokens_out[i] = 0;
				out->http_status[i] = 500;
			}
		}
		out->num_success = num_success;
		return (num_success > 0) ? NDB_LLM_ROUTE_SUCCESS
			: NDB_LLM_ROUTE_ERROR;
	}

	/* For remote providers, process sequentially */
	for (i = 0; i < num_prompts; i++)
	{
		NdbLLMResp	resp;
		int			rc;

		/* Validate prompt before processing */
		if (prompts[i] == NULL || strlen(prompts[i]) == 0)
		{
			out->texts[i] = NULL;
			out->tokens_in[i] = 0;
			out->tokens_out[i] = 0;
			out->http_status[i] = 500;
			continue;
		}

		memset(&resp, 0, sizeof(NdbLLMResp));
		rc = ndb_hf_complete(cfg, prompts[i], params_json, &resp);
		if (rc == 0)
		{
			out->texts[i] = resp.text ? resp.text : pstrdup("");
			out->tokens_in[i] = resp.tokens_in;
			out->tokens_out[i] = resp.tokens_out;
			out->http_status[i] = resp.http_status;
			num_success++;
		}
		else
		{
			out->texts[i] = NULL;
			out->tokens_in[i] = 0;
			out->tokens_out[i] = 0;
			out->http_status[i] = 500;
		}
	}
	out->num_success = num_success;
	return (num_success > 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
}

/*
 * ndb_llm_route_rerank_batch
 *	  Route batch reranking requests to appropriate backend (GPU, ONNX, HTTP)
 */
int
ndb_llm_route_rerank_batch(const NdbLLMConfig *cfg,
						   const NdbLLMCallOptions *opts,
						   const char **queries,
						   const char ***docs_array,
						   int *ndocs_array,
						   int num_queries,
						   float ***scores_out,
						   int **nscores_out)
{
	int			i;
	int			num_success = 0;

	if (cfg == NULL || queries == NULL || docs_array == NULL
		|| ndocs_array == NULL || scores_out == NULL
		|| nscores_out == NULL || num_queries <= 0)
	{
		elog(WARNING, "neurondb: ndb_llm_route_rerank_batch called with NULL or invalid parameters");
		return NDB_LLM_ROUTE_ERROR;
	}

	/* Validate array dimensions */
	for (i = 0; i < num_queries; i++)
	{
		if (ndocs_array[i] < 0)
		{
			elog(WARNING, "neurondb: ndb_llm_route_rerank_batch: invalid ndocs_array[%d] = %d", i, ndocs_array[i]);
			return NDB_LLM_ROUTE_ERROR;
		}
		if (ndocs_array[i] > 0 && docs_array[i] == NULL)
		{
			elog(WARNING, "neurondb: ndb_llm_route_rerank_batch: NULL docs_array[%d] with ndocs_array[%d] = %d", i, i, ndocs_array[i]);
			return NDB_LLM_ROUTE_ERROR;
		}
	}

	/* Initialize output */
		nalloc(*scores_out, float *, num_queries);
		MemSet(*scores_out, 0, sizeof(float *) * num_queries);
		nalloc(*nscores_out, int, num_queries);
		MemSet(*nscores_out, 0, sizeof(int) * num_queries);

	/* For now, process sequentially */
	for (i = 0; i < num_queries; i++)
	{
		float	   *scores = NULL;
		int			rc;

		/* Skip if query or docs are NULL */
		if (queries[i] == NULL || docs_array[i] == NULL
			|| ndocs_array[i] <= 0)
		{
			(*scores_out)[i] = NULL;
			(*nscores_out)[i] = 0;
			continue;
		}

		rc = ndb_llm_route_rerank(cfg,
								  opts,
								  queries[i],
								  docs_array[i],
								  ndocs_array[i],
								  &scores);
		if (rc == NDB_LLM_ROUTE_SUCCESS && scores != NULL)
		{
			(*scores_out)[i] = scores;
			(*nscores_out)[i] = ndocs_array[i];
			num_success++;
		}
		else
		{
			(*scores_out)[i] = NULL;
			(*nscores_out)[i] = 0;
		}
	}

	return (num_success > 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
}

/*
 * ndb_llm_route_embed_batch
 *	  Route batch embedding requests to appropriate backend (GPU, HTTP)
 */
int
ndb_llm_route_embed_batch(const NdbLLMConfig *cfg,
						  const NdbLLMCallOptions *opts,
						  const char **texts,
						  int num_texts,
						  float ***vecs_out,
						  int **dims_out,
						  int *num_success_out)
{
	int			i;
	int			num_success = 0;
	float	  **vecs = NULL;
	int		   *dims = NULL;

	if (cfg == NULL || texts == NULL || vecs_out == NULL
		|| dims_out == NULL || num_success_out == NULL || num_texts <= 0)
	{
		elog(WARNING, "neurondb: ndb_llm_route_embed_batch called with NULL or invalid parameters");
		return NDB_LLM_ROUTE_ERROR;
	}

	/* Validate texts array elements */
	for (i = 0; i < num_texts; i++)
	{
		if (texts[i] == NULL)
		{
		}
		else if (strlen(texts[i]) == 0)
		{
		}
	}

	if (provider_is(cfg->provider, "openai") || provider_is(cfg->provider, "chatgpt"))
	{
		int			rc = ndb_openai_embed_batch(cfg, texts, num_texts, vecs_out, dims_out, num_success_out);

		return (rc == 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
	}

	/* Initialize output */
		nalloc(vecs, float *, num_texts);
		MemSet(vecs, 0, sizeof(float *) * num_texts);
		nalloc(dims, int, num_texts);
		MemSet(dims, 0, sizeof(int) * num_texts);

	if (cfg->provider == NULL || provider_is(cfg->provider, "huggingface")
		|| provider_is(cfg->provider, "hf-http"))
	{
		/* Try GPU-accelerated batch if GPU is available and preferred */
		if (neurondb_gpu_is_available()
			&& (opts == NULL || opts->prefer_gpu || opts->require_gpu))
		{
			/* For GPU, process sequentially but efficiently */
			for (i = 0; i < num_texts; i++)
			{
				float *gpu_vec = NULL;
				int			gpu_dim = 0;
				char *gpu_err = NULL;
				int			rc;

				if (texts[i] == NULL)
				{
					vecs[i] = NULL;
					dims[i] = 0;
					continue;
				}

				rc = neurondb_gpu_hf_embed(
										   cfg->model, texts[i], &gpu_vec, &gpu_dim, &gpu_err);
				if (rc == 0 && gpu_vec != NULL && gpu_dim > 0)
				{
					vecs[i] = gpu_vec;
					dims[i] = gpu_dim;
					num_success++;
					if (gpu_err)
						pfree(gpu_err);
				}
				else
				{
					vecs[i] = NULL;
					dims[i] = 0;
					if (gpu_vec)
						pfree(gpu_vec);
					if (gpu_err)
						pfree(gpu_err);
					/* If GPU required, fail entire batch */
					if (opts != NULL && opts->require_gpu)
					{
						/* Clean up and return error */
						for (i = 0; i < num_texts; i++)
						{
							if (vecs[i])
								pfree(vecs[i]);
						}
						pfree(vecs);
						pfree(dims);
						return NDB_LLM_ROUTE_ERROR;
					}
				}
			}

			/*
			 * If all succeeded or some succeeded and GPU not required, use
			 * results
			 */
			if (num_success > 0 || (opts != NULL && !opts->require_gpu))
			{
				/* Fall back to HTTP for failed items if GPU not required */
				if (num_success < num_texts && (opts == NULL || !opts->require_gpu))
				{
					for (i = 0; i < num_texts; i++)
					{
						if (vecs[i] == NULL && texts[i] != NULL)
						{
							float	   *http_vec = NULL;
							int			http_dim = 0;

							if (ndb_hf_embed(cfg, texts[i], &http_vec, &http_dim) == 0)
							{
								vecs[i] = http_vec;
								dims[i] = http_dim;
								num_success++;
							}
						}
					}
				}
			}
			else
			{
				/* GPU failed entirely, fall through to HTTP */
				for (i = 0; i < num_texts; i++)
				{
					if (vecs[i])
						pfree(vecs[i]);
				}
				pfree(vecs);
				pfree(dims);
				vecs = NULL;
				dims = NULL;
				num_success = 0;
			}
		}

		/* Use HTTP batch API if GPU not used or failed */
		if (vecs == NULL || num_success < num_texts)
		{
			float	  **http_vecs = NULL;
			int		   *http_dims = NULL;
			int			http_num_success = 0;
			int			rc;

			/* Free any partial GPU results if switching to HTTP */
			if (vecs != NULL)
			{
				for (i = 0; i < num_texts; i++)
				{
					if (vecs[i])
						pfree(vecs[i]);
				}
				pfree(vecs);
				pfree(dims);
			}

			rc = ndb_hf_embed_batch(cfg, texts, num_texts,
									&http_vecs, &http_dims, &http_num_success);
			if (rc == 0 && http_vecs != NULL)
			{
				vecs = http_vecs;
				dims = http_dims;
				num_success = http_num_success;
			}
			else
			{
				/* HTTP batch failed, allocate empty results */
				vecs = (float **) palloc0(num_texts * sizeof(float *));
				dims = (int *) palloc0(num_texts * sizeof(int));
				num_success = 0;
			}
		}
	}
	else
	{
		/* Unknown provider */
		pfree(vecs);
		pfree(dims);
		return NDB_LLM_ROUTE_ERROR;
	}

	*vecs_out = vecs;
	*dims_out = dims;
	*num_success_out = num_success;
	return (num_success > 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
}

/*
 * ndb_llm_route_image_embed
 *	  Route image embedding requests to appropriate backend (GPU, HTTP)
 */
int
ndb_llm_route_image_embed(const NdbLLMConfig *cfg,
						  const NdbLLMCallOptions *opts,
						  const unsigned char *image_data,
						  size_t image_size,
						  float **vec_out,
						  int *dim_out)
{
	if (cfg == NULL || image_data == NULL || image_size == 0
		|| vec_out == NULL || dim_out == NULL)
		return NDB_LLM_ROUTE_ERROR;

	if (provider_is(cfg->provider, "openai") || provider_is(cfg->provider, "chatgpt"))
	{
		int			rc = ndb_openai_image_embed(cfg, image_data, image_size, vec_out, dim_out);

		return (rc == 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
	}

	if (cfg->provider == NULL || provider_is(cfg->provider, "huggingface")
		|| provider_is(cfg->provider, "hf-http"))
	{
		if (neurondb_gpu_is_available()
			&& (opts == NULL || opts->prefer_gpu || opts->require_gpu))
		{
			const ndb_gpu_backend *backend;
			float	   *gpu_vec = NULL;
			int			gpu_dim = 0;
			char	   *gpu_err = NULL;

			/* Try direct GPU backend implementation first */
			backend = ndb_gpu_get_active_backend();
			if (backend != NULL && backend->hf_image_embed != NULL)
			{
				int			gpu_rc;

				gpu_rc = backend->hf_image_embed(
												 cfg->model, image_data, image_size,
												 &gpu_vec, &gpu_dim, &gpu_err);
				if (gpu_rc == 0 && gpu_vec != NULL && gpu_dim > 0)
				{
					*vec_out = gpu_vec;
					*dim_out = gpu_dim;
					if (gpu_err)
						pfree(gpu_err);
					return NDB_LLM_ROUTE_SUCCESS;
				}
				if (gpu_vec)
					pfree(gpu_vec);
				if (gpu_err)
					pfree(gpu_err);
				if (opts != NULL && opts->require_gpu)
				{
					return NDB_LLM_ROUTE_ERROR;
				}
			}

#ifdef HAVE_ONNX_RUNTIME
			/* Try ONNX CLIP image embedding (uses GPU provider if available) */
			if (neurondb_onnx_available())
			{
				char *onnx_err = NULL;
				int			onnx_rc;

				onnx_rc = ndb_onnx_hf_image_embed(
												  cfg->model, image_data, image_size,
												  vec_out, dim_out, &onnx_err);
				if (onnx_rc == 0 && *vec_out != NULL && *dim_out > 0)
				{
					if (onnx_err)
						pfree(onnx_err);
					return NDB_LLM_ROUTE_SUCCESS;
				}
				/* ONNX failed, fall through to HTTP */
				if (onnx_err)
					pfree(onnx_err);
				if (opts != NULL && opts->require_gpu)
				{
					/* GPU required but failed */
					return NDB_LLM_ROUTE_ERROR;
				}
			}
#endif
		}
		/* Fall back to HTTP API */
		return ndb_hf_image_embed(cfg, image_data, image_size, vec_out, dim_out);
	}

	/* Unknown provider */
	return NDB_LLM_ROUTE_ERROR;
}

/*
 * ndb_llm_route_multimodal_embed
 *	  Route multimodal (text+image) embedding requests to appropriate backend (GPU, HTTP)
 */
int
ndb_llm_route_multimodal_embed(const NdbLLMConfig *cfg,
							   const NdbLLMCallOptions *opts,
							   const char *text,
							   const unsigned char *image_data,
							   size_t image_size,
							   float **vec_out,
							   int *dim_out)
{
	if (cfg == NULL || text == NULL || image_data == NULL || image_size == 0
		|| vec_out == NULL || dim_out == NULL)
		return NDB_LLM_ROUTE_ERROR;

	if (provider_is(cfg->provider, "openai") || provider_is(cfg->provider, "chatgpt"))
	{
		int			rc = ndb_openai_multimodal_embed(cfg, text, image_data, image_size, vec_out, dim_out);

		return (rc == 0) ? NDB_LLM_ROUTE_SUCCESS : NDB_LLM_ROUTE_ERROR;
	}

	if (cfg->provider == NULL || provider_is(cfg->provider, "huggingface")
		|| provider_is(cfg->provider, "hf-http"))
	{
		if (neurondb_gpu_is_available()
			&& (opts == NULL || opts->prefer_gpu || opts->require_gpu))
		{
			const ndb_gpu_backend *backend;
			float	   *gpu_vec = NULL;
			int			gpu_dim = 0;
			char	   *gpu_err = NULL;
			int			gpu_rc;

			/* Try direct GPU backend implementation first */
			backend = ndb_gpu_get_active_backend();
			if (backend != NULL && backend->hf_multimodal_embed != NULL)
			{
				gpu_rc = backend->hf_multimodal_embed(
													  cfg->model, text, image_data, image_size,
													  &gpu_vec, &gpu_dim, &gpu_err);
				if (gpu_rc == 0 && gpu_vec != NULL && gpu_dim > 0)
				{
					*vec_out = gpu_vec;
					*dim_out = gpu_dim;
					if (gpu_err)
						pfree(gpu_err);
					return NDB_LLM_ROUTE_SUCCESS;
				}
				if (gpu_vec)
					pfree(gpu_vec);
				if (gpu_err)
					pfree(gpu_err);
				if (opts != NULL && opts->require_gpu)
				{
					return NDB_LLM_ROUTE_ERROR;
				}
			}

#ifdef HAVE_ONNX_RUNTIME

			/*
			 * Try ONNX CLIP multimodal embedding (uses GPU provider if
			 * available)
			 */
			if (neurondb_onnx_available())
			{
				char *onnx_err = NULL;
				int			onnx_rc;

				onnx_rc = ndb_onnx_hf_multimodal_embed(
													   cfg->model, text, image_data, image_size,
													   vec_out, dim_out, &onnx_err);
				if (onnx_rc == 0 && *vec_out != NULL && *dim_out > 0)
				{
					if (onnx_err)
						pfree(onnx_err);
					return NDB_LLM_ROUTE_SUCCESS;
				}
				/* ONNX failed, fall through to HTTP */
				if (onnx_err)
					pfree(onnx_err);
				if (opts != NULL && opts->require_gpu)
				{
					/* GPU required but failed */
					return NDB_LLM_ROUTE_ERROR;
				}
			}
#endif
		}
		/* Fall back to HTTP API */
		return ndb_hf_multimodal_embed(cfg, text, image_data, image_size, vec_out, dim_out);
	}

	/* Unknown provider */
	return NDB_LLM_ROUTE_ERROR;
}
