/*-------------------------------------------------------------------------
 * neurondb_guc.h
 *   Centralized GUC (Grand Unified Configuration) handling for NeuronDB
 *
 * This header defines the NeuronDBConfig structure that holds all
 * configuration values and provides access to them throughout the system.
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *-------------------------------------------------------------------------*/

#ifndef NEURONDB_GUC_H
#define NEURONDB_GUC_H

#include "postgres.h"

/* NeuronDBConfig structure - holds all GUC values organized by category */
typedef struct NeuronDBConfig
{
	/* Core/index settings */
	struct
	{
		int			hnsw_ef_search;
		int			ivf_probes;
		int			ef_construction;
	}			core;

	/* GPU settings */
	struct
	{
		int			compute_mode;	/* 0=CPU, 1=GPU, 2=AUTO */
		int			backend_type;	/* 0=CUDA, 1=ROCm, 2=Metal */
		int			device;
		int			batch_size;
		int			streams;
		double		memory_pool_mb;
		char	   *kernels;
		int			timeout_ms;
	}			gpu;

	/* LLM settings */
	struct
	{
		char	   *provider;
		char	   *model;
		char	   *endpoint;
		char	   *api_key;
		int			timeout_ms;
		int			cache_ttl;
		int			rate_limiter_qps;
		bool		fail_open;
	}			llm;

	/* Worker: neuranq (queue executor) settings */
	struct
	{
		int			naptime;
		int			queue_depth;
		int			batch_size;
		int			timeout;
		int			max_retries;
		bool		enabled;
	}			neuranq;

	/* Worker: neuranmon (auto-tuner) settings */
	struct
	{
		int			naptime;
		int			sample_size;
		double		target_latency;
		double		target_recall;
		bool		enabled;
	}			neuranmon;

	/* Worker: neurandefrag (index maintenance) settings */
	struct
	{
		int			naptime;
		int			compact_threshold;
		double		fragmentation_threshold;
		char	   *maintenance_window;
		bool		enabled;
	}			neurandefrag;

	/* ONNX Runtime settings */
	struct
	{
		char	   *model_path;
		bool		use_gpu;
		int			threads;
		int			cache_size;
	}			onnx;

	/* Quota settings */
	struct
	{
		int64		default_max_vectors;
		int64		default_max_storage_mb;
		int			default_max_qps;
		bool		enforce_quotas;
	}			quota;

	/* AutoML settings */
	struct
	{
		bool		use_gpu;
	}			automl;
}			NeuronDBConfig;

/* Global config structure instance */
extern NeuronDBConfig * neurondb_config;

/* Function declarations */
extern void neurondb_init_all_gucs(void);
extern void neurondb_sync_config_from_gucs(void);

/* Legacy GUC variable declarations (for backward compatibility) */
extern int	neurondb_hnsw_ef_search;
extern int	neurondb_ivf_probes;
extern int	neurondb_ef_construction;
extern int	neurondb_compute_mode;
extern int	neurondb_gpu_backend_type;
extern int	neurondb_gpu_device;
extern int	neurondb_gpu_batch_size;
extern int	neurondb_gpu_streams;
extern char *neurondb_gpu_kernels;
extern int	neurondb_gpu_timeout_ms;
extern char *neurondb_llm_provider;
extern char *neurondb_llm_model;
extern char *neurondb_llm_endpoint;
extern char *neurondb_llm_api_key;
extern int	neurondb_llm_timeout_ms;
extern int	neurondb_llm_cache_ttl;
extern int	neurondb_llm_rate_limiter_qps;
extern bool neurondb_llm_fail_open;
extern bool neurondb_automl_use_gpu;
extern double neurondb_gpu_memory_pool_mb;
extern char *neurondb_onnx_model_path;
extern bool neurondb_onnx_use_gpu;
extern int	neurondb_onnx_threads;
extern int	neurondb_onnx_cache_size;

#endif							/* NEURONDB_GUC_H */
