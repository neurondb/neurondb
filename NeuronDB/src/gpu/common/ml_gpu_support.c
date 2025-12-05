/*-------------------------------------------------------------------------
 *
 * ml_gpu_support.c
 *    Helpers for consistent usage across ML routines.
 *
 * This module provides helper functions for consistent resource management
 * in ML operations.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/ml_gpu_support.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "ml_gpu_support.h"

#include "utils/builtins.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_constants.h"

void
ml_gpu_call_begin(MLGpuCallState *state,
				  const char *tag,
				  const char *kernel_name,
				  bool must_have_kernel)
{
	Assert(state != NULL);

	memset(state, 0, sizeof(MLGpuCallState));
	state->tag = tag;
	state->kernel_name = kernel_name;
	state->must_have = must_have_kernel;
	state->gpu_enabled = NDB_SHOULD_TRY_GPU();

	state->ctx = ml_gpu_context_acquire(tag);
	state->gpu_ready = ml_gpu_context_ready(state->ctx);
	state->kernel_allowed = (kernel_name == NULL)
		? true
		: ndb_gpu_kernel_enabled(kernel_name);
	state->use_gpu = state->gpu_ready && state->kernel_allowed;

	if (state->must_have && state->gpu_enabled && !state->use_gpu
		&& NDB_REQUIRE_GPU())
	{
		if (kernel_name)
			ereport(ERROR,
					(errmsg("GPU kernel \"%s\" unavailable for %s",
							kernel_name,
							tag ? tag : "ML call")));
		else
			ereport(ERROR,
					(errmsg("GPU unavailable for %s",
							tag ? tag : "ML call")));
	}
}

bool
ml_gpu_call_use_gpu(const MLGpuCallState *state)
{
	return state && state->use_gpu;
}

MLGpuContext *
ml_gpu_call_context(const MLGpuCallState *state)
{
	if (state == NULL)
		return NULL;
	return state->ctx;
}

void
ml_gpu_call_end(MLGpuCallState *state)
{
	if (state == NULL)
		return;

	if (state->ctx)
		ml_gpu_context_release(state->ctx);

	memset(state, 0, sizeof(MLGpuCallState));
}
