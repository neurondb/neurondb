/*-------------------------------------------------------------------------
 *
 * ml_gpu_context.c
 *    Context helpers for ML subsystems.
 *
 * This module provides resource-scoped context management for ML operations
 * with deterministic cleanup.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/ml_gpu_context.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "ml_gpu_context.h"

#include "miscadmin.h"
#include "utils/memutils.h"
#include "utils/builtins.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_constants.h"

static const char *
ml_gpu_context_default_tag(void)
{
	return "ML GPU context";
}

MLGpuContext *
ml_gpu_context_acquire(const char *tag)
{
	MLGpuContext *ctx;
	MemoryContext parent;
	MemoryContext local;
	const char *tagname;

	parent = CurrentMemoryContext;
	tagname = (tag != NULL && tag[0] != '\0')
		? tag
		: ml_gpu_context_default_tag();
	local = AllocSetContextCreate(
								  parent, "ML GPU context", ALLOCSET_SMALL_SIZES);

	ctx = NULL;

	PG_TRY();
	{
		MemoryContext oldcx;

		oldcx = MemoryContextSwitchTo(local);

		ctx = (MLGpuContext *) palloc0(sizeof(MLGpuContext));
		ctx->memcxt = local;
		ctx->backend = GPU_BACKEND_NONE;
		ctx->device_id = -1;
		ctx->gpu_available = false;
		ctx->tag = pstrdup(tagname);

		if (NDB_SHOULD_TRY_GPU())
		{
			ndb_gpu_init_if_needed();
			if (neurondb_gpu_is_available())
			{
				ctx->gpu_available = true;
				ctx->backend = neurondb_gpu_get_backend();
				ctx->device_id = neurondb_gpu_device;
			}
		}

		MemoryContextSwitchTo(oldcx);
	}
	PG_CATCH();
	{
		MemoryContextSwitchTo(parent);
		MemoryContextDelete(local);
		PG_RE_THROW();
	}
	PG_END_TRY();

	MemoryContextSwitchTo(parent);

	return ctx;
}

void
ml_gpu_context_release(MLGpuContext * ctx)
{
	if (ctx == NULL)
		return;

	MemoryContextDelete(ctx->memcxt);
}

MemoryContext
ml_gpu_context_memory(const MLGpuContext * ctx)
{
	if (ctx == NULL)
		return CurrentMemoryContext;
	return ctx->memcxt;
}

bool
ml_gpu_context_ready(const MLGpuContext * ctx)
{
	return (ctx != NULL && ctx->gpu_available);
}

GPUBackend
ml_gpu_context_backend(const MLGpuContext * ctx)
{
	if (ctx == NULL)
		return GPU_BACKEND_NONE;
	return ctx->backend;
}

int
ml_gpu_context_device(const MLGpuContext * ctx)
{
	if (ctx == NULL)
		return -1;
	return ctx->device_id;
}

const char *
ml_gpu_context_tag(const MLGpuContext * ctx)
{
	if (ctx == NULL || ctx->tag == NULL)
		return ml_gpu_context_default_tag();
	return ctx->tag;
}
