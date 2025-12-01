/*-------------------------------------------------------------------------
 *
 * ml_gpu_buffer.c
 *    Shared buffer abstraction for ML pipelines.
 *
 * This module provides buffer management for ML operations with consistent
 * bookkeeping and resource management.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/ml_gpu_buffer.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "ml_gpu_buffer.h"

#include "utils/memutils.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

Size
ml_gpu_dtype_size(MLGpuDType dtype)
{
	switch (dtype)
	{
		case MLGPU_DTYPE_FLOAT32:
			return sizeof(float);
		case MLGPU_DTYPE_FLOAT64:
			return sizeof(double);
		case MLGPU_DTYPE_INT32:
			return sizeof(int32);
		case MLGPU_DTYPE_UINT8:
			return sizeof(uint8);
		case MLGPU_DTYPE_INVALID:
		default:
			break;
	}

	elog(ERROR, "ml_gpu_buffer: unknown dtype %d", dtype);
	pg_unreachable();
}

static void
ml_gpu_buffer_reset(MLGpuBuffer * buf)
{
	Assert(buf != NULL);

	buf->context = NULL;
	buf->host_ptr = NULL;
	buf->host_bytes = 0;
	buf->host_owner = false;
	buf->device_ptr = NULL;
	buf->device_bytes = 0;
	buf->device_owner = false;
	buf->elem_count = 0;
	buf->dtype = MLGPU_DTYPE_INVALID;
	buf->host_valid = false;
	buf->device_valid = false;
}

void
ml_gpu_buffer_init_owner(MLGpuBuffer * buf,
						 MLGpuContext * ctx,
						 MLGpuDType dtype,
						 int64 elem_count,
						 bool zero)
{
	MemoryContext oldcx;
	Size		elem_size;
	Size		total_bytes;

	Assert(buf != NULL);
	Assert(ctx != NULL);
	Assert(elem_count >= 0);

	ml_gpu_buffer_reset(buf);

	elem_size = ml_gpu_dtype_size(dtype);
	total_bytes = (Size) elem_count * elem_size;

	oldcx = MemoryContextSwitchTo(ml_gpu_context_memory(ctx));

	buf->context = ctx;
	buf->dtype = dtype;
	buf->elem_count = elem_count;
	buf->host_bytes = total_bytes;
	buf->host_owner = true;
	buf->host_valid = true;

	if (total_bytes > 0)
	{
		buf->host_ptr =
			zero ? palloc0(total_bytes) : palloc(total_bytes);
	}

	MemoryContextSwitchTo(oldcx);
}

void
ml_gpu_buffer_bind_host(MLGpuBuffer * buf,
						MLGpuContext * ctx,
						void *host_ptr,
						Size host_bytes,
						int64 elem_count,
						MLGpuDType dtype)
{
	Assert(buf != NULL);

	ml_gpu_buffer_reset(buf);

	buf->context = ctx;
	buf->host_ptr = host_ptr;
	buf->host_bytes = host_bytes;
	buf->host_owner = false;
	buf->host_valid = (host_ptr != NULL);
	buf->elem_count = elem_count;
	buf->dtype = dtype;
}

void
ml_gpu_buffer_invalidate_device(MLGpuBuffer * buf)
{
	if (buf == NULL)
		return;

	buf->device_valid = false;
}

bool
ml_gpu_buffer_ensure_device(MLGpuBuffer * buf, bool copy_from_host)
{
	const		ndb_gpu_backend *backend;
	Size		required;
	void	   *dev_ptr;

	if (buf == NULL || buf->context == NULL)
		return false;
	if (!ml_gpu_context_ready(buf->context))
		return false;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->mem_alloc == NULL)
		return false;

	required = buf->device_bytes ? buf->device_bytes : buf->host_bytes;
	if (required == 0)
		return false;

	if (buf->device_ptr == NULL)
	{
		dev_ptr = NULL;
		if (backend->mem_alloc(&dev_ptr, required) != 0
			|| dev_ptr == NULL)
			return false;
		buf->device_ptr = dev_ptr;
		buf->device_bytes = required;
		buf->device_owner = true;
		buf->device_valid = false;
	}

	if (copy_from_host && buf->host_valid && buf->host_ptr != NULL)
	{
		if (backend->memcpy_h2d == NULL)
			return false;
		if (backend->memcpy_h2d(
								buf->device_ptr, buf->host_ptr, buf->host_bytes)
			!= 0)
			return false;
		buf->device_valid = true;
	}

	return true;
}

bool
ml_gpu_buffer_copy_to_host(MLGpuBuffer * buf)
{
	const		ndb_gpu_backend *backend;

	if (buf == NULL || buf->context == NULL)
		return false;
	if (!ml_gpu_context_ready(buf->context))
		return false;
	if (!buf->device_valid || buf->device_ptr == NULL
		|| buf->host_ptr == NULL)
		return false;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL || backend->memcpy_d2h == NULL)
		return false;

	if (backend->memcpy_d2h(buf->host_ptr, buf->device_ptr, buf->host_bytes)
		!= 0)
		return false;

	buf->host_valid = true;
	return true;
}

void
ml_gpu_buffer_release(MLGpuBuffer * buf)
{
	const		ndb_gpu_backend *backend;

	if (buf == NULL)
		return;

	if (buf->host_owner && buf->host_ptr != NULL)
		NDB_FREE(buf->host_ptr);

	backend = ndb_gpu_get_active_backend();
	if (buf->device_owner && buf->device_ptr != NULL && backend
		&& backend->mem_free)
	{
		backend->mem_free(buf->device_ptr);
	}

	ml_gpu_buffer_reset(buf);
}
