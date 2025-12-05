#ifndef NEURONDB_CUDA_RUNTIME_H
#define NEURONDB_CUDA_RUNTIME_H

#include "neurondb_config.h"

#if defined(NDB_GPU_CUDA)
#if !defined(__CUDACC__)
#define NEURONDB_CUDA_FLOAT4_REMAP
#define float4 neurondb_cuda_float4
#endif
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if !defined(__CUDACC__)
#undef float4
#undef NEURONDB_CUDA_FLOAT4_REMAP
#endif
#else
/* Minimal shim so host code can build when CUDA support is disabled */
typedef void *cudaStream_t;
#endif

#endif							/* NEURONDB_CUDA_RUNTIME_H */
