/*-------------------------------------------------------------------------
 *
 * gpu_inference.c
 *    Accelerated ML inference.
 *
 * This module implements ONNX Runtime inference and embedding generation
 * for high-throughput batch inference operations.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_inference.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_constants.h"

#ifdef HAVE_ONNXRUNTIME_GPU
#include <onnxruntime_cxx_api.h>
#endif

/*
 * GPU ONNX inference
 */
void
neurondb_gpu_onnx_inference(void *model_handle,
							const float *input,
							int input_size,
							float *output,
							int output_size)
{
	const ndb_gpu_backend *backend;

	(void) model_handle;
	(void) input;
	(void) input_size;
	(void) output;
	(void) output_size;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		return;

	if (!neurondb_gpu_is_available())
		return;

	backend = ndb_gpu_get_active_backend();
	if (backend != NULL)
	{
		elog(DEBUG1,
			 "neurondb: ONNX GPU inference not implemented for "
			 "backend %s; using CPU fallback",
			 backend->name ? backend->name : "unknown");
	}
}
