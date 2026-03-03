/*-------------------------------------------------------------------------
 *
 * ml_gpu_support.h
 *	  Helpers for consistent GPU usage across NeurondB ML routines.
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_GPU_SUPPORT_H
#define ML_GPU_SUPPORT_H

#include "ml_gpu_context.h"

#include "neurondb_gpu.h"

typedef struct MLGpuCallState
{
	MLGpuContext *ctx;
	bool		gpu_enabled;
	bool		gpu_ready;
	bool		kernel_allowed;
	bool		use_gpu;
	const char *kernel_name;
	bool		must_have;
	const char *tag;
} MLGpuCallState;

extern void ml_gpu_call_begin(MLGpuCallState *state,
							  const char *tag,
							  const char *kernel_name,
							  bool must_have_kernel);

extern bool ml_gpu_call_use_gpu(const MLGpuCallState *state);
extern MLGpuContext *ml_gpu_call_context(const MLGpuCallState *state);
extern void ml_gpu_call_end(MLGpuCallState *state);

#endif							/* ML_GPU_SUPPORT_H */
