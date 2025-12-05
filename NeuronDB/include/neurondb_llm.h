/*-------------------------------------------------------------------------
 * neurondb_llm.h
 *   Hugging Face / LLM provider integration for NeurondB
 *-------------------------------------------------------------------------*/

#ifndef NEURONDB_LLM_H
#define NEURONDB_LLM_H

#include "postgres.h"
#include "fmgr.h"

/* GUC variables are now in neurondb_guc.h */
#include "neurondb_guc.h"

typedef struct NdbLLMConfig
{
	const char *provider;
	const char *endpoint;
	const char *model;
	const char *api_key;
	int			timeout_ms;
	bool		prefer_gpu;
	bool		require_gpu;
} NdbLLMConfig;

typedef struct NdbLLMResp
{
	char	   *text;			/* generation */
	char	   *json;			/* raw */
	int			tokens_in;
	int			tokens_out;
	int			http_status;
} NdbLLMResp;

typedef struct NdbLLMCallOptions
{
	const char *task;			/* "complete", "embed", "rerank" */
	bool		prefer_gpu;		/* caller would like GPU */
	bool		require_gpu;	/* hard fail if GPU not available */
	bool		fail_open;		/* mirror neurondb_llm_fail_open */
} NdbLLMCallOptions;

typedef enum NdbLLMRouteStatus
{
	NDB_LLM_ROUTE_SUCCESS = 0,
	NDB_LLM_ROUTE_ERROR = -1,
	NDB_LLM_ROUTE_BACKEND_UNAVAILABLE = 1
} NdbLLMRouteStatus;

/* Provider calls */
int			ndb_hf_complete(const NdbLLMConfig *cfg,
							const char *prompt,
							const char *params_json,
							NdbLLMResp *out);

int			ndb_hf_embed(const NdbLLMConfig *cfg,
						 const char *text,
						 float **vec_out,
						 int *dim_out);

int			ndb_hf_embed_batch(const NdbLLMConfig *cfg,
							   const char **texts,
							   int num_texts,
							   float ***vecs_out,
							   int **dims_out,
							   int *num_success_out);

int			ndb_hf_image_embed(const NdbLLMConfig *cfg,
							   const unsigned char *image_data,
							   size_t image_size,
							   float **vec_out,
							   int *dim_out);

int			ndb_hf_multimodal_embed(const NdbLLMConfig *cfg,
									const char *text,
									const unsigned char *image_data,
									size_t image_size,
									float **vec_out,
									int *dim_out);

int			ndb_hf_vision_complete(const NdbLLMConfig *cfg,
								   const unsigned char *image_data,
								   size_t image_size,
								   const char *prompt,
								   const char *params_json,
								   NdbLLMResp *out);

int			ndb_hf_rerank(const NdbLLMConfig *cfg,
						  const char *query,
						  const char **docs,
						  int ndocs,
						  float **scores_out);

/* OpenAI provider calls */
int			ndb_openai_complete(const NdbLLMConfig *cfg,
								const char *prompt,
								const char *params_json,
								NdbLLMResp *out);

int			ndb_openai_vision_complete(const NdbLLMConfig *cfg,
									   const unsigned char *image_data,
									   size_t image_size,
									   const char *prompt,
									   const char *params_json,
									   NdbLLMResp *out);

int			ndb_openai_embed(const NdbLLMConfig *cfg,
							 const char *text,
							 float **vec_out,
							 int *dim_out);

int			ndb_openai_embed_batch(const NdbLLMConfig *cfg,
								   const char **texts,
								   int num_texts,
								   float ***vecs_out,
								   int **dims_out,
								   int *num_success_out);

int			ndb_openai_image_embed(const NdbLLMConfig *cfg,
								   const unsigned char *image_data,
								   size_t image_size,
								   float **vec_out,
								   int *dim_out);

int			ndb_openai_multimodal_embed(const NdbLLMConfig *cfg,
										const char *text,
										const unsigned char *image_data,
										size_t image_size,
										float **vec_out,
										int *dim_out);

int			ndb_openai_rerank(const NdbLLMConfig *cfg,
							  const char *query,
							  const char **docs,
							  int ndocs,
							  float **scores_out);

/* Batch operations */
typedef struct NdbLLMBatchResp
{
	char	  **texts;			/* Generated texts */
	int		   *tokens_in;		/* Input tokens per item */
	int		   *tokens_out;		/* Output tokens per item */
	int		   *http_status;	/* HTTP status per item */
	int			num_items;		/* Number of items */
	int			num_success;	/* Number of successful items */
} NdbLLMBatchResp;

/* Provider router */
int			ndb_llm_route_complete(const NdbLLMConfig *cfg,
								   const NdbLLMCallOptions *opts,
								   const char *prompt,
								   const char *params_json,
								   NdbLLMResp *out);

int			ndb_llm_route_vision_complete(const NdbLLMConfig *cfg,
										  const NdbLLMCallOptions *opts,
										  const unsigned char *image_data,
										  size_t image_size,
										  const char *prompt,
										  const char *params_json,
										  NdbLLMResp *out);

int			ndb_llm_route_embed(const NdbLLMConfig *cfg,
								const NdbLLMCallOptions *opts,
								const char *text,
								float **vec_out,
								int *dim_out);

int			ndb_llm_route_embed_batch(const NdbLLMConfig *cfg,
									  const NdbLLMCallOptions *opts,
									  const char **texts,
									  int num_texts,
									  float ***vecs_out,
									  int **dims_out,
									  int *num_success_out);

int			ndb_llm_route_image_embed(const NdbLLMConfig *cfg,
									  const NdbLLMCallOptions *opts,
									  const unsigned char *image_data,
									  size_t image_size,
									  float **vec_out,
									  int *dim_out);

int			ndb_llm_route_multimodal_embed(const NdbLLMConfig *cfg,
										   const NdbLLMCallOptions *opts,
										   const char *text,
										   const unsigned char *image_data,
										   size_t image_size,
										   float **vec_out,
										   int *dim_out);

int			ndb_llm_route_rerank(const NdbLLMConfig *cfg,
								 const NdbLLMCallOptions *opts,
								 const char *query,
								 const char **docs,
								 int ndocs,
								 float **scores_out);

/* Batch router */
int			ndb_llm_route_complete_batch(const NdbLLMConfig *cfg,
										 const NdbLLMCallOptions *opts,
										 const char **prompts,
										 int num_prompts,
										 const char *params_json,
										 NdbLLMBatchResp *out);

int			ndb_llm_route_rerank_batch(const NdbLLMConfig *cfg,
									   const NdbLLMCallOptions *opts,
									   const char **queries,
									   const char ***docs_array,
									   int *ndocs_array,
									   int num_queries,
									   float ***scores_out,
									   int **nscores_out);

/* SQL-callable */
extern Datum ndb_llm_complete(PG_FUNCTION_ARGS);
extern Datum ndb_llm_image_analyze(PG_FUNCTION_ARGS);
extern Datum ndb_llm_embed(PG_FUNCTION_ARGS);
extern Datum ndb_llm_rerank(PG_FUNCTION_ARGS);
extern Datum ndb_llm_enqueue(PG_FUNCTION_ARGS);
extern Datum ndb_llm_complete_batch(PG_FUNCTION_ARGS);
extern Datum ndb_llm_rerank_batch(PG_FUNCTION_ARGS);

/* GUC init and shared memory */
void		neurondb_llm_init_guc(void);
Size		neurondb_llm_shmem_size(void);
void		neurondb_llm_shmem_init(void);

/* Cache functions (from llm_cache.c) */
bool
			ndb_llm_cache_lookup(const char *key, int max_age_seconds, char **out_text);
void		ndb_llm_cache_store(const char *key, const char *text);
void		ndb_llm_cache_store_with_eviction(const char *key, const char *text, int max_size);

/* Job functions (from llm_jobs.c) */
int			ndb_llm_job_enqueue(const char *job_type, const char *payload);
bool		ndb_llm_job_acquire(int *job_id, char **job_type, char **payload);
bool		ndb_llm_job_update(int job_id,
							   const char *status,
							   const char *result,
							   const char *error);
int			ndb_llm_job_prune(int max_age_days);
int			ndb_llm_job_count_status(const char *status);
int			ndb_llm_job_retry_failed(int max_retries);
void		ndb_llm_job_clear(void);

/* Image processing utilities (from llm_image_utils.c) */
typedef enum
{
	IMAGE_FORMAT_UNKNOWN = 0,
	IMAGE_FORMAT_JPEG,
	IMAGE_FORMAT_PNG,
	IMAGE_FORMAT_GIF,
	IMAGE_FORMAT_WEBP,
	IMAGE_FORMAT_BMP
} ImageFormat;

typedef struct ImageMetadata
{
	ImageFormat format;
	size_t		size;
	int			width;
	int			height;
	int			channels;
	bool		is_valid;
	char	   *mime_type;
	char	   *error_msg;
} ImageMetadata;

ImageMetadata *ndb_validate_image(const unsigned char *data, size_t size, MemoryContext mctx);
const char *ndb_get_image_format_name(ImageFormat format);
char	   *ndb_image_metadata_to_json(const ImageMetadata *meta);

#endif							/* NEURONDB_LLM_H */
