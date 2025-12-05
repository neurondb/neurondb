/*-------------------------------------------------------------------------
 *
 * neurondb_onnx.h
 *    ONNX Runtime integration for NeuronDB
 *    Provides C API wrapper around ONNX Runtime for HuggingFace model inference
 *
 * Architecture:
 *   HuggingFace Models -> ONNX Export -> ONNX Runtime C API -> PostgreSQL
 *
 * Supported Backends:
 *   - CPU (default)
 *   - CUDA (NVIDIA GPUs)
 *   - TensorRT (NVIDIA optimized)
 *   - CoreML (Apple Silicon)
 *   - DirectML (Windows)
 *
 * SPDX-License-Identifier: PostgreSQL
 *
 *-------------------------------------------------------------------------
 */
#ifndef NEURONDB_ONNX_H
#define NEURONDB_ONNX_H

#include "postgres.h"
#include "fmgr.h"

/* Include ONNX Runtime C API */
#ifdef HAVE_ONNX_RUNTIME
#include <onnxruntime_c_api.h>
#endif

extern PGDLLIMPORT char *neurondb_onnx_model_path;
extern PGDLLIMPORT bool neurondb_onnx_use_gpu;
extern PGDLLIMPORT int neurondb_onnx_threads;
extern PGDLLIMPORT int neurondb_onnx_cache_size;

/*
 * ONNX Model Types
 */
typedef enum
{
	ONNX_MODEL_EMBEDDING,		/* Text/image embeddings */
	ONNX_MODEL_CLASSIFICATION,	/* Text classification */
	ONNX_MODEL_NER,				/* Named Entity Recognition */
	ONNX_MODEL_QA,				/* Question Answering */
	ONNX_MODEL_GENERATION,		/* Text generation */
	ONNX_MODEL_CUSTOM			/* Custom models */
} ONNXModelType;

/*
 * ONNX Execution Provider (Backend)
 */
typedef enum
{
	ONNX_PROVIDER_CPU,
	ONNX_PROVIDER_CUDA,
	ONNX_PROVIDER_TENSORRT,
	ONNX_PROVIDER_COREML,
	ONNX_PROVIDER_DIRECTML
} ONNXProvider;

/*
 * ONNX Model Session
 * Wraps OrtSession* from ONNX Runtime
 */
typedef struct ONNXModelSession
{
#ifdef HAVE_ONNX_RUNTIME
	OrtSession *session;
	OrtEnv	   *env;
	OrtSessionOptions *session_options;
#else
	void	   *session;
	void	   *env;
	void	   *session_options;
#endif
	char	   *model_path;
	ONNXModelType model_type;
	ONNXProvider provider;
	int32		input_dim;
	int32		output_dim;
	bool		is_loaded;
} ONNXModelSession;

/*
 * ONNX Tensor
 * Wrapper for model inputs/outputs
 */
typedef struct ONNXTensor
{
	float	   *data;
	int64	   *shape;
	int32		ndim;
	size_t		size;
} ONNXTensor;

/*
 * Core ONNX Runtime Functions
 */

/* Initialize ONNX Runtime environment */
extern void neurondb_onnx_init(void);

/* Cleanup ONNX Runtime environment */
extern void neurondb_onnx_cleanup(void);

/* GUC initialization is now centralized in neurondb_guc.c */

/* Load ONNX model from file */
extern ONNXModelSession *neurondb_onnx_load_model(const char *model_path,
												  ONNXModelType model_type,
												  ONNXProvider provider);

/* Unload ONNX model */
extern void neurondb_onnx_unload_model(ONNXModelSession *session);

/* Run inference on ONNX model */
extern ONNXTensor *neurondb_onnx_run_inference(ONNXModelSession *session,
											   ONNXTensor *input);

/* Free ONNX tensor */
extern void neurondb_onnx_free_tensor(ONNXTensor *tensor);

/*
 * HuggingFace Model Functions (via ONNX)
 */

/* C-callable ONNX inference functions (for LLM router integration) */
extern int	ndb_onnx_hf_embed(const char *model_name,
							  const char *text,
							  float **vec_out,
							  int *dim_out,
							  char **errstr);

extern int	ndb_onnx_hf_complete(const char *model_name,
								 const char *prompt,
								 const char *params_json,
								 char **text_out,
								 char **errstr);

extern int	ndb_onnx_hf_rerank(const char *model_name,
							   const char *query,
							   const char **docs,
							   int ndocs,
							   float **scores_out,
							   char **errstr);

extern int	ndb_onnx_hf_image_embed(const char *model_name,
									const unsigned char *image_data,
									size_t image_size,
									float **vec_out,
									int *dim_out,
									char **errstr);

extern int	ndb_onnx_hf_multimodal_embed(const char *model_name,
										 const char *text,
										 const unsigned char *image_data,
										 size_t image_size,
										 float **vec_out,
										 int *dim_out,
										 char **errstr);

/* SQL-callable functions */
extern Datum neurondb_hf_embedding(PG_FUNCTION_ARGS);
extern Datum neurondb_hf_classify(PG_FUNCTION_ARGS);
extern Datum neurondb_hf_ner(PG_FUNCTION_ARGS);
extern Datum neurondb_hf_qa(PG_FUNCTION_ARGS);
extern Datum neurondb_hf_tokenize(PG_FUNCTION_ARGS);
extern Datum neurondb_hf_detokenize(PG_FUNCTION_ARGS);

/*
 * Utility Functions
 */

/* Get ONNX Runtime version */
extern const char *neurondb_onnx_version(void);

/* Get available execution providers */
extern char **neurondb_onnx_get_providers(int *num_providers);

/* Check if ONNX Runtime is available */
extern bool neurondb_onnx_available(void);

/* Convert provider enum to string */
extern const char *neurondb_onnx_provider_name(ONNXProvider provider);

/* Convert model type enum to string */
extern const char *neurondb_onnx_model_type_name(ONNXModelType type);

/* Get or load model from cache */
extern ONNXModelSession *neurondb_onnx_get_or_load_model(const char *model_name,
														 ONNXModelType model_type);

/* Create ONNX tensor from float array */
extern ONNXTensor *neurondb_onnx_create_tensor(float *data, int64 * shape, int32 ndim);

/*
 * Tokenizer Functions
 */

/* Tokenize text to token IDs */
extern int32 *
neurondb_tokenize(const char *text, int32 max_length, int32 * output_length);

/* Tokenize text with model-specific tokenizer */
extern int32 * neurondb_tokenize_with_model(const char *text,
											int32 max_length,
											int32 * output_length,
											const char *model_name);

/* Detokenize token IDs back to text */
extern char *neurondb_detokenize(const int32 * token_ids,
								 int32 length,
								 const char *model_name);

/* Create attention mask from token IDs */
extern int32 * neurondb_create_attention_mask(int32 * token_ids, int32 length);

#endif							/* NEURONDB_ONNX_H */
