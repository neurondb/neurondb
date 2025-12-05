/*-------------------------------------------------------------------------
 *
 * ml_gpu_buffer.h
 *	  Shared GPU buffer abstraction for NeurondB ML components.
 *
 * Buffers provide a consistent way to reference host data, optional device
 * storage, and state flags for synchronization. Device transfers are wired
 * through gpu_core.c; callers can gracefully fall back to CPU when GPU is
 * unavailable.
 *
 *-------------------------------------------------------------------------
 */

#ifndef ML_GPU_BUFFER_H
#define ML_GPU_BUFFER_H

#include "postgres.h"

#include "ml_gpu_context.h"

typedef enum MLGpuDType
{
	MLGPU_DTYPE_INVALID = 0,
	MLGPU_DTYPE_FLOAT32,
	MLGPU_DTYPE_FLOAT64,
	MLGPU_DTYPE_INT32,
	MLGPU_DTYPE_UINT8
} MLGpuDType;

typedef struct MLGpuBuffer
{
	MLGpuContext *context;
	void	   *host_ptr;
	Size		host_bytes;
	bool		host_owner;
	void	   *device_ptr;
	Size		device_bytes;
	bool		device_owner;
	int64		elem_count;
	MLGpuDType	dtype;
	bool		host_valid;
	bool		device_valid;
} MLGpuBuffer;

extern Size ml_gpu_dtype_size(MLGpuDType dtype);

extern void ml_gpu_buffer_init_owner(MLGpuBuffer *buf,
									 MLGpuContext *ctx,
									 MLGpuDType dtype,
									 int64 elem_count,
									 bool zero);

extern void ml_gpu_buffer_bind_host(MLGpuBuffer *buf,
									MLGpuContext *ctx,
									void *host_ptr,
									Size host_bytes,
									int64 elem_count,
									MLGpuDType dtype);

extern void ml_gpu_buffer_invalidate_device(MLGpuBuffer *buf);
extern bool ml_gpu_buffer_ensure_device(MLGpuBuffer *buf, bool copy_from_host);
extern bool ml_gpu_buffer_copy_to_host(MLGpuBuffer *buf);
extern void ml_gpu_buffer_release(MLGpuBuffer *buf);

#endif							/* ML_GPU_BUFFER_H */
