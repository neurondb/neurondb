/*-------------------------------------------------------------------------
 *
 * gpu_stubs.c
 *    Stub implementations for GPU backend functions
 *    Always compiled to ensure symbols are available in CPU-only builds
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_cuda_hf.h"

/* GPU backend registration stubs */
__attribute__((weak)) void
neurondb_gpu_register_cuda_backend(void)
{
	/* No-op stub - overridden by real implementation if CUDA is compiled */
}

__attribute__((weak)) void
neurondb_gpu_register_rocm_backend(void)
{
	/* No-op stub - overridden by real implementation if ROCm is compiled */
}

__attribute__((weak)) void
neurondb_gpu_register_metal_backend(void)
{
	/* No-op stub - overridden by real implementation if Metal is compiled */
}

/* CUDA HF function stubs */
__attribute__((weak)) int
ndb_cuda_hf_generate_batch(const char *model_name,
						   const char **prompts,
						   int num_prompts,
						   const char *params_json,
						   NdbCudaHfBatchResult *results,
						   char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA backend not available - GPU support not compiled");
	return -1;
}

__attribute__((weak)) int
ndb_cuda_hf_complete(const char *model_name,
					 const char *prompt,
					 const char *params_json,
					 char **text_out,
					 char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA backend not available - GPU support not compiled");
	return -1;
}

__attribute__((weak)) int
ndb_cuda_hf_embed(const char *model_name,
				  const char *text,
				  float **vec_out,
				  int *dim_out,
				  char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA backend not available - GPU support not compiled");
	return -1;
}

__attribute__((weak)) int
ndb_cuda_hf_rerank(const char *model_name,
				   const char *query,
				   const char **docs,
				   int ndocs,
				   float **scores_out,
				   char **errstr)
{
	if (errstr)
		*errstr = pstrdup("CUDA backend not available - GPU support not compiled");
	return -1;
}

