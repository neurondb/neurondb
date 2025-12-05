/*-------------------------------------------------------------------------
 *
 * neurondb_rocm_hf.h
 *	  ROCm-backed Hugging Face / LLM data structures and API surface.
 *
 * Defines host-side representations for GPU Hugging Face operations and the
 * entry points used by the ROCm backend. Implementations live under
 * src/gpu/rocm.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ROCM_HF_H
#define NEURONDB_ROCM_HF_H

#ifndef __HIPCC__
#include "postgres.h"
#include <stdint.h>
#else
struct varlena;
typedef struct varlena bytea;
#include <stdint.h>
#endif

/* Maximum sequence length for tokenization */
#define NDB_HF_MAX_SEQ_LEN 512
#define NDB_HF_MAX_VOCAB_SIZE 30522
#define NDB_HF_MAX_EMBED_DIM 768
#define NDB_HF_MAX_MODEL_NAME 256
#define NDB_HF_MAX_GEN_TOKENS 2048
#define NDB_HF_MAX_STOP_SEQUENCES 16
#define NDB_HF_MAX_STOP_SEQ_LEN 32

/* Hugging Face model types */
typedef enum
{
	NDB_HF_MODEL_EMBEDDING,		/* Text embeddings (BERT, etc.) */
	NDB_HF_MODEL_GENERATION,	/* Text generation (GPT, etc.) */
	NDB_HF_MODEL_RERANK,		/* Reranking (cross-encoder) */
	NDB_HF_MODEL_CLASSIFICATION /* Classification */
} NdbHfModelType;

/* Model configuration */
typedef struct NdbRocmHfModelConfig
{
	char		model_name[NDB_HF_MAX_MODEL_NAME];
	NdbHfModelType model_type;
	int			vocab_size;
	int			embed_dim;
	int			max_seq_len;
	int			num_layers;
	int			num_heads;
	int			hidden_dim;
	bool		use_gpu;
} NdbRocmHfModelConfig;

/* Tokenization result */
typedef struct NdbRocmHfTokens
{
	int32_t    *token_ids;
	int32_t    *attention_mask;
	int			seq_len;
	int			vocab_size;
} NdbRocmHfTokens;

/* Model weights structure (placeholder for actual model loading) */
typedef struct NdbRocmHfModelWeights
{
	float	   *embedding_table;	/* Vocabulary embeddings */
	float	   *position_embeddings;	/* Position embeddings */
	float	   *layer_norm_gamma;	/* Layer norm parameters */
	float	   *layer_norm_beta;
	float	   *query_weights;	/* Attention query weights */
	float	   *key_weights;	/* Attention key weights */
	float	   *value_weights;	/* Attention value weights */
	float	   *output_weights; /* Attention output weights */
	float	   *ffn_weights1;	/* Feed-forward weights 1 */
	float	   *ffn_weights2;	/* Feed-forward weights 2 */
	float	   *lm_head_weights;	/* Language model head weights */
	float	   *classification_weights; /* Classification head weights for
										 * reranking */
	float		classification_bias;	/* Classification head bias */
	size_t		total_bytes;
} NdbRocmHfModelWeights;

/* Generation parameters */
typedef struct NdbRocmHfGenParams
{
	float		temperature;	/* Sampling temperature (0.0 = greedy, >1.0 =
								 * more random) */
	float		top_p;			/* Nucleus sampling (0.0-1.0) */
	int			top_k;			/* Top-k sampling (0 = disabled) */
	int			max_tokens;		/* Maximum tokens to generate */
	int			min_tokens;		/* Minimum tokens to generate */
	float		repetition_penalty; /* Repetition penalty (1.0 = no penalty,
									 * >1.0 = penalty) */
	bool		do_sample;		/* Use sampling (true) or greedy (false) */
	int			num_stop_sequences; /* Number of stop sequences */
	int32_t		stop_sequences
	[			NDB_HF_MAX_STOP_SEQUENCES]
	[			NDB_HF_MAX_STOP_SEQ_LEN];	/* Stop sequence token IDs */
	int			stop_seq_lens
	[			NDB_HF_MAX_STOP_SEQUENCES]; /* Length of each stop sequence */
	bool		return_prompt;	/* Include prompt in output */
	int			seed;			/* Random seed for reproducibility */
	bool		streaming;		/* Enable streaming mode (return tokens as
								 * generated) */
	int			num_logit_bias; /* Number of logit bias entries */
	int32_t		logit_bias_tokens[256]; /* Token IDs for logit bias */
	float		logit_bias_values[256]; /* Bias values for tokens */
} NdbRocmHfGenParams;

/* KV cache entry for autoregressive generation */
typedef struct NdbRocmHfKVCache
{
	float	   *key_cache;		/* Key cache [num_layers, seq_len, num_heads,
								 * head_dim] */
	float	   *value_cache;	/* Value cache [num_layers, seq_len,
								 * num_heads, head_dim] */
	int			current_pos;	/* Current position in cache */
	int			max_pos;		/* Maximum position in cache */
	int			num_layers;		/* Number of transformer layers */
	int			num_heads;		/* Number of attention heads */
	int			head_dim;		/* Dimension per attention head */
	bool		allocated;		/* Whether cache is allocated */
} NdbRocmHfKVCache;

#ifdef __cplusplus
extern "C"
{
#endif

/* Model management */
extern int	ndb_rocm_hf_load_model(const char *model_name,
								   const char *model_path,
								   NdbHfModelType model_type,
								   NdbRocmHfModelConfig *config,
								   char **errstr);

extern int	ndb_rocm_hf_unload_model(const char *model_name, char **errstr);

extern bool ndb_rocm_hf_model_loaded(const char *model_name);

extern int	ndb_rocm_hf_get_model_config(const char *model_name,
										 NdbRocmHfModelConfig *config,
										 char **errstr);

/* Tokenization */
extern int	ndb_rocm_hf_tokenize(const char *text,
								 const char *model_name,
								 int32_t * token_ids_out,
								 int32_t * attention_mask_out,
								 int *seq_len_out,
								 char **errstr);

/* Internal HIP kernel function for embedding inference */
extern int	ndb_rocm_hf_embed_inference(const char *model_name,
										const int32_t * token_ids,
										const int32_t * attention_mask,
										int seq_len,
										const float *embedding_table,
										int vocab_size,
										int embed_dim,
										float *output_embedding,
										char **errstr);

/* Internal HIP kernel function for text generation inference */
extern int	ndb_rocm_hf_generate_inference(const char *model_name,
										   const int32_t * input_token_ids,
										   int input_seq_len,
										   const float *embedding_table,
										   const float *position_embeddings,
										   const float *lm_head_weights,
										   const NdbRocmHfModelWeights *weights,
										   const NdbRocmHfModelConfig *config,
										   const NdbRocmHfGenParams *gen_params,
										   NdbRocmHfKVCache *kv_cache,
										   int32_t * output_token_ids,
										   int *output_seq_len,
										   char **errstr);

/* Internal HIP kernel function for cross-encoder reranking inference */
extern int	ndb_rocm_hf_cross_encoder_rerank_inference(
													   const char *model_name,
													   const int32_t * token_ids_batch,
													   const int32_t * attention_mask_batch,
													   int batch_size,
													   int seq_len,
													   const float *embedding_table,
													   int vocab_size,
													   int embed_dim,
													   const float *classification_weights,
													   float classification_bias,
													   float *scores_out,
													   char **errstr);

/* Streaming generation callback */
typedef void (*ndb_rocm_hf_stream_callback) (int32_t token_id,
											 const char *token_text,
											 int position,
											 void *user_data);

/* Batch generation result */
typedef struct NdbRocmHfBatchResult
{
	char	   *text;			/* Generated text */
	int32_t		token_ids[NDB_HF_MAX_GEN_TOKENS];	/* Generated token IDs */
	int			num_tokens;		/* Number of generated tokens */
	int			status;			/* 0 = success, <0 = error */
	char	   *error;			/* Error message if status < 0 */
} NdbRocmHfBatchResult;

/* Streaming generation with callback */
extern int	ndb_rocm_hf_generate_stream(const char *model_name,
										const char *prompt,
										const char *params_json,
										ndb_rocm_hf_stream_callback callback,
										void *user_data,
										char **errstr);

/* Batch generation */
extern int	ndb_rocm_hf_generate_batch(const char *model_name,
									   const char **prompts,
									   int num_prompts,
									   const char *params_json,
									   NdbRocmHfBatchResult *results,
									   char **errstr);

/* ROCm-backed Hugging Face embedding generation */
extern int	ndb_rocm_hf_embed(const char *model_name,
							  const char *text,
							  float **vec_out,
							  int *dim_out,
							  char **errstr);

/* ROCm-backed Hugging Face text completion */
extern int	ndb_rocm_hf_complete(const char *model_name,
								 const char *prompt,
								 const char *params_json,
								 char **text_out,
								 char **errstr);

/* ROCm-backed Hugging Face reranking */
extern int	ndb_rocm_hf_rerank(const char *model_name,
							   const char *query,
							   const char **docs,
							   int ndocs,
							   float **scores_out,
							   char **errstr);

#ifdef __cplusplus
}
#endif

#endif							/* NEURONDB_ROCM_HF_H */
