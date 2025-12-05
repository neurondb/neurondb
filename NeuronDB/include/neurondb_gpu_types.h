#ifndef NEURONDB_GPU_TYPES_H
#define NEURONDB_GPU_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/*
 * Generic GPU stream/queue handle used across backends. Each backend may map
 * this to its native stream type (CUDA stream, HIP stream, Metal command queue,
 * etc.). Common code treats it as an opaque pointer.
 */
typedef void *ndb_stream_t;

/*
 * Identifier for GPU backend implementations. This is kept small and stable so
 * common code can perform lightweight comparisons or logging without pulling in
 * backend-specific headers.
 */
typedef enum NDBGpuBackendKind
{
	NDB_GPU_BACKEND_NONE = 0,
	NDB_GPU_BACKEND_CUDA,
	NDB_GPU_BACKEND_ROCM,
	NDB_GPU_BACKEND_METAL
} NDBGpuBackendKind;

/*
 * Compact capability flags describing what an implementation provides. Backends
 * may set additional feature bits over time but the common code only relies on
 * the canonical ones.
 */
typedef enum NDBGpuBackendFeature
{
	NDB_GPU_FEATURE_DISTANCE = (1U << 0),
	NDB_GPU_FEATURE_QUANTIZE = (1U << 1),
	NDB_GPU_FEATURE_CLUSTERING = (1U << 2),
	NDB_GPU_FEATURE_INFERENCE = (1U << 3)
} NDBGpuBackendFeature;

/*
 * Basic device information shared with SQL-visible layers. The structure is
 * intentionally plain C to avoid PostgreSQL type dependencies in GPU code.
 */
typedef struct NDBGpuDeviceInfo
{
	int			device_id;
	char		name[256];
	size_t		total_memory_bytes;
	size_t		free_memory_bytes;
	int			compute_major;
	int			compute_minor;
	bool		is_available;
} NDBGpuDeviceInfo;

#endif							/* NEURONDB_GPU_TYPES_H */
