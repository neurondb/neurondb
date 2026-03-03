/*-------------------------------------------------------------------------
 *
 * ml_gpu_context.h
 *	  GPU context management helpers for NeurondB ML routines.
 *
 * Provides a lightweight wrapper around neurondb_gpu.c for per-call GPU
 * usage, with graceful CPU fallback when GPU resources are unavailable.
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_GPU_CONTEXT_H
#define ML_GPU_CONTEXT_H

#include "postgres.h"
#include "utils/memutils.h"

#include "neurondb_gpu.h"

typedef struct MLGpuContext
{
	MemoryContext memcxt;
	GPUBackend	backend;
	int			device_id;
	bool		gpu_available;
	char	   *tag;
} MLGpuContext;

extern MLGpuContext *ml_gpu_context_acquire(const char *tag);
extern void ml_gpu_context_release(MLGpuContext *ctx);
extern MemoryContext ml_gpu_context_memory(const MLGpuContext *ctx);
extern bool ml_gpu_context_ready(const MLGpuContext *ctx);
extern GPUBackend ml_gpu_context_backend(const MLGpuContext *ctx);
extern int	ml_gpu_context_device(const MLGpuContext *ctx);
extern const char *ml_gpu_context_tag(const MLGpuContext *ctx);

#endif							/* ML_GPU_CONTEXT_H */
