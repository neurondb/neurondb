/*-------------------------------------------------------------------------
 *
 * gpu_rf_cuda.c
 *	  CUDA backend bridge for Random Forest training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/cuda/gpu_rf_cuda.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_CUDA

#include <float.h>
#include <math.h>
#include <string.h>

#include "neurondb_cuda_runtime.h"
#include "common/pg_prng.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "miscadmin.h"
#include "common/hashfn.h"
#include "storage/ipc.h"

#include "ml_random_forest_internal.h"
#include "ml_random_forest_shared.h"
#include "neurondb_cuda_rf.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/* Forward declarations for kernel launchers */
extern int	launch_rf_predict_batch_kernel(const NdbCudaRfNode * d_nodes,
										   const NdbCudaRfTreeHeader * d_trees,
										   int tree_count,
										   const float *d_features,
										   int n_samples,
										   int feature_dim,
										   int class_count,
										   int *d_votes);
extern int	ndb_cuda_rf_infer(const NdbCudaRfNode * nodes,
							  const NdbCudaRfTreeHeader * trees,
							  int tree_count,
							  const float *input,
							  int feature_dim,
							  int class_count,
							  int *votes);

/* GPU model cache entry - keyed by hash of model_data */
typedef struct RfGpuModelCacheKey
{
	uint32		model_hash;		/* Hash of model_data bytea */
	uint32		model_size;		/* Size of model_data bytea */
}			RfGpuModelCacheKey;

typedef struct RfGpuModelCacheEntry
{
	RfGpuModelCacheKey key;		/* Compound key */
	NdbCudaRfGpuModel *gpu_model;	/* GPU-resident model */
}			RfGpuModelCacheEntry;

/* Backend-local hash table for GPU model cache */
static HTAB * rf_gpu_model_cache = NULL;
static bool rf_cache_initialized = false;

/* Forward declarations */
static void rf_gpu_cache_init(void);
static void rf_gpu_cache_cleanup(int code, Datum arg);

/* Hash function for model_data bytea */
static uint32
rf_model_hash(const void *key, Size keysize)
{
	uint32		hash_val;
	const		RfGpuModelCacheKey *cache_key = (const RfGpuModelCacheKey *) key;

	(void) keysize;
	/* Combine hash and size for better collision resistance */
	hash_val = cache_key->model_hash ^ cache_key->model_size;
	return hash_val;
}

static void
rf_copy_tree_nodes(const GTree * tree, NdbCudaRfNode * dest,
				   int *node_offset, int *max_feat_idx)
{
	const		GTreeNode *src_nodes;
	int			count;
	int			i;
	int			max_idx = -1;

	if (tree == NULL || dest == NULL || node_offset == NULL)
		return;

	src_nodes = gtree_nodes(tree);
	count = tree->count;
	for (i = 0; i < count; i++)
	{
		const		GTreeNode *src = &src_nodes[i];
		NdbCudaRfNode *dst = &dest[*node_offset + i];

		dst->feature_idx = src->feature_idx;
		dst->threshold = (float) src->threshold;
		if (src->is_leaf)
		{
			dst->left_child = -1;
			dst->right_child = -1;
		}
		else
		{
			dst->left_child = src->left;
			dst->right_child = src->right;
		}
		dst->value = (float) src->value;

		/* Track maximum feature index */
		if (src->feature_idx >= 0 && src->feature_idx > max_idx)
			max_idx = src->feature_idx;
	}
	*node_offset += count;
	if (max_feat_idx != NULL)
		*max_feat_idx = max_idx;
}

static void
rf_fill_single_node_tree(NdbCudaRfNode * node, int majority_class)
{
	node->feature_idx = -1;
	node->threshold = 0.0f;
	node->left_child = -1;
	node->right_child = -1;
	node->value = (float) majority_class;
}

int
ndb_cuda_rf_pack_model(const RFModel * model,
					   bytea * *model_data,
					   Jsonb * *metrics,
					   char **errstr)
{
	int			tree_count = 0;
	int			total_nodes = 0;
	int			i;
	size_t		header_bytes;
	size_t		nodes_bytes;
	size_t		payload_bytes;
	bytea	   *blob;
	char	   *base;
	NdbCudaRfModelHeader *model_hdr;
	NdbCudaRfTreeHeader *tree_hdrs;
	NdbCudaRfNode *nodes;
	int			node_cursor = 0;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid RF model for CUDA pack");
		return -1;
	}

	if (model->tree_count > 0 && model->trees != NULL)
	{
		tree_count = model->tree_count;
		for (i = 0; i < model->tree_count; i++)
		{
			const		GTree *tree = model->trees[i];

			if (tree != NULL)
				total_nodes += tree->count;
		}
	}
	else if (model->tree != NULL)
	{
		tree_count = 1;
		total_nodes = model->tree->count;
	}
	else
	{
		if (errstr)
			*errstr = pstrdup("random_forest model has no trees");
		return -1;
	}

	if (tree_count <= 0 || total_nodes <= 0)
	{
		if (errstr)
			*errstr = pstrdup("random_forest model empty");
		return -1;
	}

	header_bytes = sizeof(NdbCudaRfModelHeader)
		+ (sizeof(NdbCudaRfTreeHeader) * tree_count);
	nodes_bytes = sizeof(NdbCudaRfNode) * total_nodes;
	payload_bytes = header_bytes + nodes_bytes;

	/* Defensive check for integer overflow and MaxAllocSize */
	if (payload_bytes > MaxAllocSize || header_bytes > MaxAllocSize ||
		nodes_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("CUDA RF pack: payload size exceeds MaxAllocSize");
		return -1;
	}
	if (VARHDRSZ + payload_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("CUDA RF pack: total size exceeds MaxAllocSize");
		return -1;
	}

	blob = (bytea *) palloc(VARHDRSZ + payload_bytes);
	if (blob == NULL)
	{
		if (errstr)
			*errstr = pstrdup("CUDA RF pack: palloc failed");
		return -1;
	}
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	model_hdr = (NdbCudaRfModelHeader *) base;
	model_hdr->tree_count = tree_count;
	model_hdr->feature_dim = model->n_features;
	model_hdr->class_count = model->n_classes;
	model_hdr->sample_count = model->n_samples;
	model_hdr->majority_class = (int) rint(model->majority_value);
	model_hdr->majority_fraction = model->majority_fraction;

	tree_hdrs =
		(NdbCudaRfTreeHeader *) (base + sizeof(NdbCudaRfModelHeader));
	nodes = (NdbCudaRfNode *) (base + header_bytes);

	node_cursor = 0;
	for (i = 0; i < tree_count; i++)
	{
		const		GTree *tree =
			(model->tree_count > 0 && model->trees != NULL)
			? model->trees[i]
			: model->tree;
		int			node_count = tree ? tree->count : 0;

		tree_hdrs[i].node_count = node_count;
		tree_hdrs[i].nodes_start = node_cursor;
		tree_hdrs[i].root_index = tree ? tree->root : 0;
		tree_hdrs[i].max_feature_index = -1;

		if (node_count > 0)
			rf_copy_tree_nodes(tree, nodes, &node_cursor,
							   &tree_hdrs[i].max_feature_index);
	}

	*model_data = blob;

	if (metrics != NULL)
	{
		RFMetricsSpec spec;

		memset(&spec, 0, sizeof(spec));
		spec.storage = "gpu";
		spec.algorithm = "random_forest";
		spec.tree_count = tree_count;
		spec.majority_class = model_hdr->majority_class;
		spec.majority_fraction = model_hdr->majority_fraction;
		spec.gini = model->gini_impurity;
		spec.oob_accuracy = model->oob_accuracy;
		*metrics = rf_build_metrics_json(&spec);
	}

	return 0;
}

int
ndb_cuda_rf_train(const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  int class_count,
				  const Jsonb * hyperparams,
				  bytea * *model_data,
				  Jsonb * *metrics,
				  char **errstr)
{
	const int	default_n_trees = 32;
	int			n_trees = default_n_trees;
	int		   *label_ints = NULL;
	int		   *class_counts = NULL;
	int		   *best_left_counts = NULL;
	int		   *best_right_counts = NULL;
	int		   *tmp_left_counts = NULL;
	int		   *tmp_right_counts = NULL;
	bytea	   *payload = NULL;
	Jsonb	   *metrics_json = NULL;
	float	   *d_features = NULL;
	int		   *d_labels = NULL;
	double	   *d_feature_sum = NULL;
	double	   *d_feature_sumsq = NULL;
	int		   *d_left_counts = NULL;
	int		   *d_right_counts = NULL;
	pg_prng_state rng;
	bool		seeded = false;
	cudaError_t status = cudaSuccess;
	double		gini_accumulator = 0.0;
	size_t		feature_bytes;
	size_t		label_bytes;
	size_t		class_bytes;
	int			i;
	int			j;
	int			rc = -1;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || labels == NULL || n_samples <= 0
		|| feature_dim <= 0 || class_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup(
							  "invalid input parameters for CUDA RF train");
		return -1;
	}

	(void) hyperparams;
	if (n_trees <= 0)
		n_trees = default_n_trees;
	if (class_count > 4096)
	{
		if (errstr)
			*errstr = pstrdup("CUDA RF training: class_count exceeds maximum of 4096");
		return -1;
	}

	{
		size_t		label_size = sizeof(int) * (size_t) n_samples;
		size_t		class_size = sizeof(int) * (size_t) class_count;

		if (label_size > MaxAllocSize || class_size > MaxAllocSize)
		{
			if (errstr)
				*errstr = pstrdup("CUDA RF train: allocation size exceeds MaxAllocSize");
			return -1;
		}
		label_ints = (int *) palloc(label_size);
		class_counts = (int *) palloc0(class_size);
		tmp_left_counts = (int *) palloc(class_size);
		tmp_right_counts = (int *) palloc(class_size);
		best_left_counts = (int *) palloc(class_size);
		best_right_counts = (int *) palloc(class_size);
		if (label_ints == NULL || class_counts == NULL ||
			tmp_left_counts == NULL || tmp_right_counts == NULL ||
			best_left_counts == NULL || best_right_counts == NULL)
		{
			if (errstr)
				*errstr = pstrdup("CUDA RF train: palloc failed");
			if (label_ints)
				NDB_FREE(label_ints);
			if (class_counts)
				NDB_FREE(class_counts);
			if (tmp_left_counts)
				NDB_FREE(tmp_left_counts);
			if (tmp_right_counts)
				NDB_FREE(tmp_right_counts);
			if (best_left_counts)
				NDB_FREE(best_left_counts);
			if (best_right_counts)
				NDB_FREE(best_right_counts);
			return -1;
		}
	}
	for (i = 0; i < n_samples; i++)
	{
		double		val = labels[i];

		label_ints[i] = (int) rint(val);
		if (label_ints[i] < 0 || label_ints[i] >= class_count)
			label_ints[i] = 0;
	}

	if (ndb_cuda_rf_histogram(
							  label_ints, n_samples, class_count, class_counts)
		!= 0)
	{
		memset(class_counts, 0, sizeof(int) * class_count);
		for (i = 0; i < n_samples; i++)
			class_counts[label_ints[i]]++;
	}

	{
		NdbCudaRfModelHeader model_hdr;
		NdbCudaRfTreeHeader *tree_hdrs;
		NdbCudaRfNode *nodes;
		int			total_nodes = n_trees * 3;
		size_t		header_bytes;
		size_t		payload_bytes;
		char	   *base;
		int			majority_class = 0;
		int			best_count = class_counts[0];
		double		majority_fraction;

		feature_bytes =
			sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
		label_bytes = sizeof(int) * (size_t) n_samples;
		class_bytes = sizeof(int) * (size_t) class_count;

		status = cudaMalloc((void **) &d_features, feature_bytes);
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMalloc((void **) &d_labels, label_bytes);
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMalloc((void **) &d_feature_sum, sizeof(double));
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMalloc((void **) &d_feature_sumsq, sizeof(double));
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMalloc((void **) &d_left_counts, class_bytes);
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMalloc((void **) &d_right_counts, class_bytes);
		if (status != cudaSuccess)
			goto gpu_fail;

		status = cudaMemcpy(d_features,
							features,
							feature_bytes,
							cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
			goto gpu_fail;
		status = cudaMemcpy(d_labels,
							label_ints,
							label_bytes,
							cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
			goto gpu_fail;

		if (!seeded)
		{
			if (!pg_prng_strong_seed(&rng))
				pg_prng_seed(&rng,
							 (uint64) n_samples
							 ^ (uint64) feature_dim);
			seeded = true;
		}

		for (i = 1; i < class_count; i++)
		{
			if (class_counts[i] > best_count)
			{
				best_count = class_counts[i];
				majority_class = i;
			}
		}
		majority_fraction = (n_samples > 0)
			? ((double) best_count / (double) n_samples)
			: 0.0;

		header_bytes = sizeof(NdbCudaRfModelHeader)
			+ sizeof(NdbCudaRfTreeHeader) * n_trees;
		payload_bytes =
			header_bytes + sizeof(NdbCudaRfNode) * total_nodes;
		/* Defensive check for size */
		if (payload_bytes > MaxAllocSize || VARHDRSZ + payload_bytes > MaxAllocSize)
		{
			if (errstr)
				*errstr = pstrdup("CUDA RF train: payload size exceeds MaxAllocSize");
			goto gpu_fail;
		}
		payload = (bytea *) palloc(VARHDRSZ + payload_bytes);
		if (payload == NULL)
		{
			if (errstr)
				*errstr = pstrdup("CUDA RF train: palloc failed for payload");
			goto gpu_fail;
		}
		SET_VARSIZE(payload, VARHDRSZ + payload_bytes);
		base = VARDATA(payload);

		model_hdr.tree_count = n_trees;
		model_hdr.feature_dim = feature_dim;
		model_hdr.class_count = class_count;
		model_hdr.sample_count = n_samples;
		model_hdr.majority_class = majority_class;
		model_hdr.majority_fraction = majority_fraction;

		memcpy(base, &model_hdr, sizeof(model_hdr));
		tree_hdrs = (NdbCudaRfTreeHeader *) (base + sizeof(model_hdr));
		nodes = (NdbCudaRfNode *) (base + header_bytes);

		for (i = 0; i < n_trees; i++)
		{
			double		best_gini = DBL_MAX;
			float		best_threshold = 0.0f;
			int			best_feature = -1;
			int			left_majority = majority_class;
			int			right_majority = majority_class;
			int			left_total = 0;
			int			right_total = 0;
			int			node_offset = i * 3;
			double		noise = pg_prng_double(&rng) - 0.5;

			memset(best_left_counts, 0, class_bytes);
			memset(best_right_counts, 0, class_bytes);

			for (j = 0; j < feature_dim; j++)
			{
				double		sum_host = 0.0;
				double		sumsq_host = 0.0;
				double		variance;
				float		threshold;

				if (ndb_cuda_rf_launch_feature_stats(d_features,
													 n_samples,
													 feature_dim,
													 j,
													 d_feature_sum,
													 d_feature_sumsq)
					!= 0)
					continue;

				status = cudaMemcpy(&sum_host,
									d_feature_sum,
									sizeof(double),
									cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
					goto gpu_fail;
				status = cudaMemcpy(&sumsq_host,
									d_feature_sumsq,
									sizeof(double),
									cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
					goto gpu_fail;

				if (sum_host == 0.0 && sumsq_host == 0.0)
					continue;

				threshold =
					(float) (sum_host / (double) n_samples);
				variance = (sumsq_host / (double) n_samples)
					- ((double) threshold
					   * (double) threshold);
				if (variance < 0.0)
					variance = 0.0;
				if (variance > 0.0)
					threshold += (float) (noise
										  * sqrt(variance) * 0.25);

				if (ndb_cuda_rf_launch_split_counts(d_features,
													d_labels,
													n_samples,
													feature_dim,
													j,
													threshold,
													class_count,
													d_left_counts,
													d_right_counts)
					!= 0)
					continue;

				status = cudaMemcpy(tmp_left_counts,
									d_left_counts,
									class_bytes,
									cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
					goto gpu_fail;
				status = cudaMemcpy(tmp_right_counts,
									d_right_counts,
									class_bytes,
									cudaMemcpyDeviceToHost);
				if (status != cudaSuccess)
					goto gpu_fail;

				{
					double		gini =
						rf_split_gini(tmp_left_counts,
									  tmp_right_counts,
									  class_count,
									  &left_total,
									  &right_total,
									  NULL,
									  NULL);

					if (gini < best_gini && gini >= 0.0)
					{
						best_gini = gini;
						best_feature = j;
						best_threshold = threshold;
						memcpy(best_left_counts,
							   tmp_left_counts,
							   class_bytes);
						memcpy(best_right_counts,
							   tmp_right_counts,
							   class_bytes);
					}
				}
			}

			if (best_feature < 0)
			{
				tree_hdrs[i].node_count = 1;
				tree_hdrs[i].nodes_start = node_offset;
				tree_hdrs[i].root_index = 0;
				tree_hdrs[i].max_feature_index = -1;
				rf_fill_single_node_tree(
										 &nodes[node_offset], majority_class);
				continue;
			}

			left_total = 0;
			right_total = 0;
			for (j = 0; j < class_count; j++)
			{
				if (best_left_counts[j] > left_total)
				{
					left_total = best_left_counts[j];
					left_majority = j;
				}
				if (best_right_counts[j] > right_total)
				{
					right_total = best_right_counts[j];
					right_majority = j;
				}
			}

			tree_hdrs[i].node_count = 3;
			tree_hdrs[i].nodes_start = node_offset;
			tree_hdrs[i].root_index = 0;
			tree_hdrs[i].max_feature_index = best_feature;

			nodes[node_offset].feature_idx = best_feature;
			nodes[node_offset].threshold = best_threshold;
			nodes[node_offset].left_child = 1;
			nodes[node_offset].right_child = 2;
			nodes[node_offset].value = (float) majority_class;

			rf_fill_single_node_tree(
									 &nodes[node_offset + 1], left_majority);
			rf_fill_single_node_tree(
									 &nodes[node_offset + 2], right_majority);

			if (best_gini > 0.0 && best_gini < DBL_MAX / 4.0)
				gini_accumulator += best_gini;
		}

		{
			RFMetricsSpec spec;

			memset(&spec, 0, sizeof(spec));
			spec.storage = "gpu";
			spec.algorithm = "random_forest";
			spec.tree_count = n_trees;
			spec.majority_class = majority_class;
			spec.majority_fraction = majority_fraction;
			spec.gini = (gini_accumulator > 0.0)
				? (gini_accumulator / (double) n_trees)
				: 0.0;
			spec.oob_accuracy = 0.0;
			metrics_json = rf_build_metrics_json(&spec);
		}
	}

	*model_data = payload;
	if (metrics != NULL)
	{
		*metrics = metrics_json;
		metrics_json = NULL;
	}
	rc = 0;

gpu_cleanup:
	if (d_features)
		cudaFree(d_features);
	if (d_labels)
		cudaFree(d_labels);
	if (d_feature_sum)
		cudaFree(d_feature_sum);
	if (d_feature_sumsq)
		cudaFree(d_feature_sumsq);
	if (d_left_counts)
		cudaFree(d_left_counts);
	if (d_right_counts)
		cudaFree(d_right_counts);
	if (label_ints)
		NDB_FREE(label_ints);
	if (class_counts)
		NDB_FREE(class_counts);
	if (tmp_left_counts)
		NDB_FREE(tmp_left_counts);
	if (tmp_right_counts)
		NDB_FREE(tmp_right_counts);
	if (best_left_counts)
		NDB_FREE(best_left_counts);
	if (best_right_counts)
		NDB_FREE(best_right_counts);
	if (metrics_json)
		NDB_FREE(metrics_json);

	return rc;

gpu_fail:
	if (errstr != NULL)
	{
		if (status != cudaSuccess)
			*errstr = pstrdup(cudaGetErrorString(status));
		else
			*errstr = pstrdup("cuda random_forest training failed");
	}
	rc = -1;
	goto gpu_cleanup;
}

int
ndb_cuda_rf_predict(const bytea * model_data,
					const float *input,
					int feature_dim,
					int *class_out,
					char **errstr)
{
	const char *base;
	const		NdbCudaRfModelHeader *model_hdr;
	const		NdbCudaRfTreeHeader *tree_hdrs;
	const		NdbCudaRfNode *nodes_base;
	size_t		header_bytes;
	int		   *votes = NULL;
	int			rc;
	int			i;
	int			best_class;
	int			best_votes;
	int			effective_dim;
	RfGpuModelCacheEntry *cache_entry;
	RfGpuModelCacheKey cache_key;
	bool		found;
	NdbCudaRfGpuModel *gpu_model = NULL;

	if (errstr)
		*errstr = NULL;
	if (class_out == NULL || model_data == NULL || input == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid inputs for CUDA RF predict");
		return -1;
	}

	/* Initialize cache if needed */
	if (!rf_cache_initialized)
		rf_gpu_cache_init();

	base = VARDATA(model_data);
	model_hdr = (const NdbCudaRfModelHeader *) base;
	if (model_hdr->class_count <= 0 || model_hdr->tree_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA RF model missing metadata");
		return -1;
	}

	/* Require feature_dim to match model expectations */
	if (model_hdr->feature_dim > 0 && model_hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("CUDA RF predict: feature dimension mismatch (model expects %d, got %d)",
							   model_hdr->feature_dim, feature_dim);
		return -1;
	}
	if (feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA RF predict: invalid feature dimension");
		return -1;
	}
	effective_dim = feature_dim;

	/* Compute hash and size of model_data for cache lookup */
	cache_key.model_hash = hash_any((const unsigned char *) VARDATA(model_data),
									VARSIZE(model_data) - VARHDRSZ);
	cache_key.model_size = VARSIZE(model_data) - VARHDRSZ;

	/* Look up in cache */
	if (rf_gpu_model_cache != NULL)
	{
		cache_entry = (RfGpuModelCacheEntry *) hash_search(
														   rf_gpu_model_cache,
														   &cache_key,
														   HASH_FIND,
														   &found);

		if (found && cache_entry != NULL && cache_entry->gpu_model != NULL
			&& cache_entry->gpu_model->is_valid)
		{
			gpu_model = cache_entry->gpu_model;
		}
	}

	/* If not in cache, upload model to GPU and cache it */
	if (gpu_model == NULL)
	{
		header_bytes = sizeof(NdbCudaRfModelHeader)
			+ sizeof(NdbCudaRfTreeHeader) * model_hdr->tree_count;
		tree_hdrs = (const NdbCudaRfTreeHeader *) (base
												   + sizeof(NdbCudaRfModelHeader));
		nodes_base = (const NdbCudaRfNode *) (base + header_bytes);

		rc = ndb_cuda_rf_model_upload(nodes_base,
									  tree_hdrs,
									  model_hdr->tree_count,
									  effective_dim,
									  model_hdr->class_count,
									  model_hdr->majority_class,
									  &gpu_model);

		if (rc == 0 && gpu_model != NULL && rf_gpu_model_cache != NULL)
		{
			/* Store in cache */
			cache_entry = (RfGpuModelCacheEntry *) hash_search(
															   rf_gpu_model_cache,
															   &cache_key,
															   HASH_ENTER,
															   &found);

			if (cache_entry != NULL)
			{
				/* If entry already existed, free old model */
				if (found && cache_entry->gpu_model != NULL
					&& cache_entry->gpu_model != gpu_model)
					ndb_cuda_rf_model_free(cache_entry->gpu_model);

				cache_entry->key.model_hash = cache_key.model_hash;
				cache_entry->gpu_model = gpu_model;
			}
		}
	}

	/* Use cached GPU model for inference */
	{
		size_t		votes_size = sizeof(int) * (size_t) model_hdr->class_count;

		if (votes_size > MaxAllocSize)
		{
			if (errstr)
				*errstr = pstrdup(
								  "CUDA RF predict: votes allocation size exceeds MaxAllocSize");
			return -1;
		}
		votes = (int *) palloc0(votes_size);
		if (votes == NULL)
		{
			if (errstr)
				*errstr = pstrdup("CUDA RF predict: palloc0 failed for votes");
			return -1;
		}
	}
	if (gpu_model != NULL)
	{
		rc = ndb_cuda_rf_infer_model(gpu_model,
									 input,
									 effective_dim,
									 votes);
	}
	else
	{
		/* Fallback to old method if cache failed */
		header_bytes = sizeof(NdbCudaRfModelHeader)
			+ sizeof(NdbCudaRfTreeHeader) * model_hdr->tree_count;
		tree_hdrs = (const NdbCudaRfTreeHeader *) (base
												   + sizeof(NdbCudaRfModelHeader));
		nodes_base = (const NdbCudaRfNode *) (base + header_bytes);

		rc = ndb_cuda_rf_infer(nodes_base,
							   tree_hdrs,
							   model_hdr->tree_count,
							   input,
							   effective_dim,
							   model_hdr->class_count,
							   votes);
	}

	if (rc == 0)
	{
		best_class = model_hdr->majority_class;
		best_votes = -1;
		for (i = 0; i < model_hdr->class_count; i++)
		{
			if (votes[i] > best_votes)
			{
				best_votes = votes[i];
				best_class = i;
			}
		}
		*class_out = best_class;
	}
	else
	{
		if (errstr != NULL && *errstr == NULL)
			*errstr = pstrdup("CUDA RF inference failed");
		*class_out = model_hdr->majority_class;
	}

	if (votes)
		NDB_FREE(votes);

	return 0;
}

/*
 * Initialize GPU model cache hash table
 */
static void
rf_gpu_cache_init(void)
{
	HASHCTL		info;

	if (rf_cache_initialized)
		return;

	memset(&info, 0, sizeof(info));
	info.keysize = sizeof(uint32);
	info.entrysize = sizeof(RfGpuModelCacheEntry);
	info.hash = rf_model_hash;
	info.hcxt = TopMemoryContext;

	rf_gpu_model_cache = hash_create("RF GPU Model Cache",
									 16,
									 &info,
									 HASH_ELEM | HASH_FUNCTION | HASH_CONTEXT);

	/* Register cleanup callback */
	on_proc_exit(rf_gpu_cache_cleanup, 0);

	rf_cache_initialized = true;
}

/*
 * Cleanup callback: free all cached GPU models on backend exit
 */
static void
rf_gpu_cache_cleanup(int code, Datum arg)
{
	HASH_SEQ_STATUS status;
	RfGpuModelCacheEntry *entry;

	(void) code;
	(void) arg;

	if (!rf_cache_initialized || rf_gpu_model_cache == NULL)
		return;

	hash_seq_init(&status, rf_gpu_model_cache);
	while ((entry = (RfGpuModelCacheEntry *) hash_seq_search(&status)) != NULL)
	{
		if (entry->gpu_model != NULL)
			ndb_cuda_rf_model_free(entry->gpu_model);
	}

	hash_destroy(rf_gpu_model_cache);
	rf_gpu_model_cache = NULL;
	rf_cache_initialized = false;
}

/*
 * Upload RF model to GPU and cache it
 */
int
ndb_cuda_rf_model_upload(const NdbCudaRfNode * nodes,
						 const NdbCudaRfTreeHeader * trees,
						 int tree_count,
						 int feature_dim,
						 int class_count,
						 int majority_class,
						 NdbCudaRfGpuModel * *out_model)
{
	NdbCudaRfGpuModel *model;
	cudaError_t status;
	int			total_nodes = 0;
	int			i;

	if (nodes == NULL || trees == NULL || tree_count <= 0
		|| feature_dim <= 0 || class_count <= 0 || out_model == NULL)
		return -1;

	for (i = 0; i < tree_count; i++)
		total_nodes += trees[i].node_count;

	if (total_nodes <= 0)
	{
		return -1;
	}

	model = (NdbCudaRfGpuModel *) palloc0(sizeof(NdbCudaRfGpuModel));
	if (model == NULL)
		return -1;

	/* Allocate device memory for nodes */
	status = cudaMalloc((void **) &model->d_nodes,
						sizeof(NdbCudaRfNode) * (size_t) total_nodes);
	if (status != cudaSuccess)
	{
		NDB_FREE(model);
		return -1;
	}

	/* Allocate device memory for tree headers */
	status = cudaMalloc((void **) &model->d_trees,
						sizeof(NdbCudaRfTreeHeader) * (size_t) tree_count);
	if (status != cudaSuccess)
	{
		cudaFree(model->d_nodes);
		NDB_FREE(model);
		return -1;
	}

	/* Copy nodes to device */
	status = cudaMemcpy(model->d_nodes,
						nodes,
						sizeof(NdbCudaRfNode) * (size_t) total_nodes,
						cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(model->d_nodes);
		cudaFree(model->d_trees);
		NDB_FREE(model);
		return -1;
	}

	/* Copy tree headers to device */
	status = cudaMemcpy(model->d_trees,
						trees,
						sizeof(NdbCudaRfTreeHeader) * (size_t) tree_count,
						cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(model->d_nodes);
		cudaFree(model->d_trees);
		NDB_FREE(model);
		return -1;
	}

	model->total_nodes = total_nodes;
	model->tree_count = tree_count;
	model->feature_dim = feature_dim;
	model->class_count = class_count;
	model->majority_class = majority_class;
	model->is_valid = true;

	*out_model = model;
	return 0;
}

/*
 * Free GPU model
 */
void
ndb_cuda_rf_model_free(NdbCudaRfGpuModel * model)
{
	if (model == NULL)
		return;

	if (model->d_nodes != NULL)
		cudaFree(model->d_nodes);
	if (model->d_trees != NULL)
		cudaFree(model->d_trees);

	model->is_valid = false;
	NDB_FREE(model);
}

/*
 * Infer using cached GPU model (no upload overhead)
 */
int
ndb_cuda_rf_infer_model(const NdbCudaRfGpuModel * model,
						const float *input,
						int feature_dim,
						int *votes)
{
	float	   *d_input = NULL;
	int		   *d_votes = NULL;
	cudaError_t status;
	int			rc;

	if (model == NULL || !model->is_valid || input == NULL || votes == NULL)
		return -1;

	if (model->feature_dim != feature_dim)
		return -1;

	/* Allocate temporary device buffers for input and votes */
	status = cudaMalloc((void **) &d_input, sizeof(float) * (size_t) feature_dim);
	if (status != cudaSuccess)
		return -1;

	status = cudaMalloc((void **) &d_votes,
						sizeof(int) * (size_t) model->class_count);
	if (status != cudaSuccess)
	{
		cudaFree(d_input);
		return -1;
	}

	/* Copy input to device */
	status = cudaMemcpy(d_input,
						input,
						sizeof(float) * (size_t) feature_dim,
						cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
		goto cleanup;

	/* Clear votes */
	status = cudaMemset(d_votes, 0, sizeof(int) * (size_t) model->class_count);
	if (status != cudaSuccess)
		goto cleanup;

	/* Launch kernel using batch kernel with n_samples=1 */
	rc = launch_rf_predict_batch_kernel(model->d_nodes,
										model->d_trees,
										model->tree_count,
										d_input,
										1,	/* n_samples = 1 for single
											 * inference */
										feature_dim,
										model->class_count,
										d_votes);

	if (rc != 0)
		goto cleanup;

	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
		goto cleanup;

	/* Copy votes back */
	status = cudaMemcpy(votes,
						d_votes,
						sizeof(int) * (size_t) model->class_count,
						cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
		goto cleanup;

	cudaFree(d_input);
	cudaFree(d_votes);
	return 0;

cleanup:
	if (d_input)
		cudaFree(d_input);
	if (d_votes)
		cudaFree(d_votes);
	return -1;
}

/* Forward declaration for GPU-only batch inference */
static int	ndb_cuda_rf_infer_batch_gpu(const NdbCudaRfGpuModel * model,
										const float *d_features,
										int n_samples,
										int feature_dim,
										int *d_votes);

/*
 * GPU-only batch inference: assumes model already on device
 * Clears d_votes, launches kernel (launcher handles sync)
 */
static int
ndb_cuda_rf_infer_batch_gpu(const NdbCudaRfGpuModel * model,
							const float *d_features,
							int n_samples,
							int feature_dim,
							int *d_votes)
{
	cudaError_t status;
	size_t		votes_bytes;

	if (model == NULL || !model->is_valid || d_features == NULL
		|| d_votes == NULL)
		return -1;

	if (n_samples <= 0 || model->tree_count <= 0 || feature_dim <= 0
		|| model->class_count <= 0)
		return -1;

	if (model->feature_dim != feature_dim)
		return -1;

	votes_bytes = sizeof(int) * (size_t) n_samples * (size_t) model->class_count;

	/* Clear votes */
	status = cudaMemset(d_votes, 0, votes_bytes);
	if (status != cudaSuccess)
		return -1;

	/* Launch batch kernel (launcher handles error clearing and sync) */
	if (launch_rf_predict_batch_kernel(model->d_nodes,
									   model->d_trees,
									   model->tree_count,
									   d_features,
									   n_samples,
									   feature_dim,
									   model->class_count,
									   d_votes) != 0)
		return -1;

	return 0;
}

/*
 * Batch prediction: predict multiple samples at once
 */
int
ndb_cuda_rf_predict_batch(const bytea * model_data,
						  const float *features,
						  int n_samples,
						  int feature_dim,
						  int *predictions_out,
						  char **errstr)
{
	const char *base;
	const		NdbCudaRfModelHeader *model_hdr;
	const		NdbCudaRfTreeHeader *tree_hdrs;
	const		NdbCudaRfNode *nodes_base;
	size_t		header_bytes;
	int			effective_dim;
	RfGpuModelCacheEntry *cache_entry;
	RfGpuModelCacheKey cache_key;
	bool		found;
	NdbCudaRfGpuModel *gpu_model = NULL;
	float	   *d_features = NULL;
	int		   *d_votes = NULL;
	int		   *h_votes = NULL;
	cudaError_t status;
	int			i,
				j;
	int			best_class;
	int			best_votes;
	int			rc = -1;

	if (errstr)
		*errstr = NULL;

	if (model_data == NULL || features == NULL || predictions_out == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid inputs for CUDA RF batch predict");
		return -1;
	}

	/* Initialize cache if needed */
	if (!rf_cache_initialized)
		rf_gpu_cache_init();

	base = VARDATA(model_data);
	model_hdr = (const NdbCudaRfModelHeader *) base;
	if (model_hdr->class_count <= 0 || model_hdr->tree_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA RF model missing metadata");
		return -1;
	}

	/* Require feature_dim to match model expectations */
	if (model_hdr->feature_dim > 0 && model_hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("CUDA RF batch predict: feature dimension mismatch (model expects %d, got %d)",
							   model_hdr->feature_dim, feature_dim);
		return -1;
	}
	if (feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("CUDA RF batch predict: invalid feature dimension");
		return -1;
	}
	effective_dim = feature_dim;

	/* Compute hash and size for cache lookup */
	cache_key.model_hash = hash_any((const unsigned char *) VARDATA(model_data),
									VARSIZE(model_data) - VARHDRSZ);
	cache_key.model_size = VARSIZE(model_data) - VARHDRSZ;

	if (rf_gpu_model_cache != NULL)
	{
		cache_entry = (RfGpuModelCacheEntry *) hash_search(
														   rf_gpu_model_cache,
														   &cache_key,
														   HASH_FIND,
														   &found);

		if (found && cache_entry != NULL && cache_entry->gpu_model != NULL
			&& cache_entry->gpu_model->is_valid)
		{
			gpu_model = cache_entry->gpu_model;
		}
	}

	/* Upload model if not cached */
	if (gpu_model == NULL)
	{
		header_bytes = sizeof(NdbCudaRfModelHeader)
			+ sizeof(NdbCudaRfTreeHeader) * model_hdr->tree_count;
		tree_hdrs = (const NdbCudaRfTreeHeader *) (base
												   + sizeof(NdbCudaRfModelHeader));
		nodes_base = (const NdbCudaRfNode *) (base + header_bytes);

		rc = ndb_cuda_rf_model_upload(nodes_base,
									  tree_hdrs,
									  model_hdr->tree_count,
									  effective_dim,
									  model_hdr->class_count,
									  model_hdr->majority_class,
									  &gpu_model);

		if (rc == 0 && gpu_model != NULL && rf_gpu_model_cache != NULL)
		{
			cache_entry = (RfGpuModelCacheEntry *) hash_search(
															   rf_gpu_model_cache,
															   &cache_key,
															   HASH_ENTER,
															   &found);

			if (cache_entry != NULL)
			{
				if (found && cache_entry->gpu_model != NULL
					&& cache_entry->gpu_model != gpu_model)
					ndb_cuda_rf_model_free(cache_entry->gpu_model);

				cache_entry->key = cache_key;
				cache_entry->gpu_model = gpu_model;
			}
		}
	}

	if (gpu_model == NULL)
	{
		if (errstr)
			*errstr = pstrdup("failed to upload RF model to GPU");
		return -1;
	}

	/* Allocate device memory for features */
	status = cudaMalloc((void **) &d_features,
						sizeof(float) * (size_t) n_samples * (size_t) effective_dim);
	if (status != cudaSuccess)
	{
		if (errstr)
			*errstr = psprintf("failed to allocate GPU memory for features: %s",
							   cudaGetErrorString(status));
		return -1;
	}

	/* Allocate device memory for votes (n_samples x class_count) */
	status = cudaMalloc((void **) &d_votes,
						sizeof(int) * (size_t) n_samples * (size_t) model_hdr->class_count);
	if (status != cudaSuccess)
	{
		cudaFree(d_features);
		if (errstr)
			*errstr = psprintf("failed to allocate GPU memory for votes: %s",
							   cudaGetErrorString(status));
		return -1;
	}

	/* Allocate host memory for votes */
	h_votes = (int *) palloc0(sizeof(int) * (size_t) n_samples
							  * (size_t) model_hdr->class_count);

	/* Copy features to device */
	status = cudaMemcpy(d_features,
						features,
						sizeof(float) * (size_t) n_samples * (size_t) effective_dim,
						cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_votes);
		NDB_FREE(h_votes);
		if (errstr)
			*errstr = psprintf("failed to copy features to GPU: %s",
							   cudaGetErrorString(status));
		return -1;
	}

	/* Use GPU-only batch inference (assumes cached model) */
	rc = ndb_cuda_rf_infer_batch_gpu(gpu_model,
									 d_features,
									 n_samples,
									 effective_dim,
									 d_votes);

	if (rc != 0)
	{
		cudaFree(d_features);
		cudaFree(d_votes);
		NDB_FREE(h_votes);
		if (errstr)
			*errstr = psprintf("batch prediction failed: %s",
							   cudaGetErrorString(cudaGetLastError()));
		return -1;
	}

	/* Copy votes back */
	status = cudaMemcpy(h_votes,
						d_votes,
						sizeof(int) * (size_t) n_samples * (size_t) model_hdr->class_count,
						cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		cudaFree(d_features);
		cudaFree(d_votes);
		NDB_FREE(h_votes);
		if (errstr)
			*errstr = psprintf("failed to copy votes from GPU: %s",
							   cudaGetErrorString(status));
		return -1;
	}

	/* Find best class for each sample */
	for (i = 0; i < n_samples; i++)
	{
		int		   *sample_votes = h_votes + (i * model_hdr->class_count);

		best_class = model_hdr->majority_class;
		best_votes = -1;

		for (j = 0; j < model_hdr->class_count; j++)
		{
			if (sample_votes[j] > best_votes)
			{
				best_votes = sample_votes[j];
				best_class = j;
			}
		}
		predictions_out[i] = best_class;
	}

	cudaFree(d_features);
	cudaFree(d_votes);
	NDB_FREE(h_votes);

	return 0;
}

/*
 * Batch evaluation: compute metrics for multiple samples
 */
int
ndb_cuda_rf_evaluate_batch(const bytea * model_data,
						   const float *features,
						   const int *labels,
						   int n_samples,
						   int feature_dim,
						   double *accuracy_out,
						   double *precision_out,
						   double *recall_out,
						   double *f1_out,
						   char **errstr)
{
	int		   *predictions = NULL;
	int		   *tp = NULL;
	int		   *fp = NULL;
	int		   *fn = NULL;
	int			class_count;
	int			i;
	int			total_correct = 0;
	double		precision_sum = 0.0;
	double		recall_sum = 0.0;
	double		f1_sum = 0.0;
	int			classes_with_predictions = 0;
	int			rc;

	if (errstr)
		*errstr = NULL;

	if (model_data == NULL || features == NULL || labels == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid inputs for CUDA RF batch evaluate");
		return -1;
	}

	if (accuracy_out == NULL || precision_out == NULL
		|| recall_out == NULL || f1_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("output pointers are NULL");
		return -1;
	}

	/* Get class count from model */
	{
		const		NdbCudaRfModelHeader *model_hdr =
			(const NdbCudaRfModelHeader *) VARDATA(model_data);

		class_count = model_hdr->class_count;
	}

	if (class_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid class count in model");
		return -1;
	}

	/* Allocate predictions array */
	predictions = (int *) palloc(sizeof(int) * (size_t) n_samples);
	tp = (int *) palloc0(sizeof(int) * (size_t) class_count);
	fp = (int *) palloc0(sizeof(int) * (size_t) class_count);
	fn = (int *) palloc0(sizeof(int) * (size_t) class_count);

	/* Batch predict */
	rc = ndb_cuda_rf_predict_batch(model_data,
								   features,
								   n_samples,
								   feature_dim,
								   predictions,
								   errstr);

	if (rc != 0)
	{
		NDB_FREE(predictions);
		NDB_FREE(tp);
		NDB_FREE(fp);
		NDB_FREE(fn);
		return -1;
	}

	/* Compute confusion matrix */
	for (i = 0; i < n_samples; i++)
	{
		int			true_label = labels[i];
		int			pred_label = predictions[i];

		if (true_label < 0 || true_label >= class_count)
			continue;
		if (pred_label < 0 || pred_label >= class_count)
			continue;

		if (true_label == pred_label)
		{
			total_correct++;
			tp[true_label]++;
		}
		else
		{
			fp[pred_label]++;
			fn[true_label]++;
		}
	}

	/* Compute metrics per class */
	for (i = 0; i < class_count; i++)
	{
		double		prec = 0.0;
		double		rec = 0.0;
		double		f1 = 0.0;

		if (tp[i] + fp[i] > 0)
			prec = (double) tp[i] / (double) (tp[i] + fp[i]);
		if (tp[i] + fn[i] > 0)
			rec = (double) tp[i] / (double) (tp[i] + fn[i]);
		if (prec + rec > 0.0)
			f1 = 2.0 * (prec * rec) / (prec + rec);

		if (tp[i] + fp[i] > 0 || tp[i] + fn[i] > 0)
		{
			precision_sum += prec;
			recall_sum += rec;
			f1_sum += f1;
			classes_with_predictions++;
		}
	}

	/* Compute macro-averaged metrics */
	*accuracy_out = (n_samples > 0)
		? ((double) total_correct / (double) n_samples)
		: 0.0;

	if (classes_with_predictions > 0)
	{
		*precision_out = precision_sum / (double) classes_with_predictions;
		*recall_out = recall_sum / (double) classes_with_predictions;
		*f1_out = f1_sum / (double) classes_with_predictions;
	}
	else
	{
		*precision_out = 0.0;
		*recall_out = 0.0;
		*f1_out = 0.0;
	}

	NDB_FREE(predictions);
	NDB_FREE(tp);
	NDB_FREE(fp);
	NDB_FREE(fn);

	return 0;
}

#endif							/* NDB_GPU_CUDA */
