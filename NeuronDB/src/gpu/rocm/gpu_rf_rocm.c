/*-------------------------------------------------------------------------
 *
 * gpu_rf_rocm.c
 *    ROCm backend bridge for Random Forest training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_rf_rocm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_HIP

#include <float.h>
#include <math.h>
#include <string.h>

#include <hip/hip_runtime.h>
#include "neurondb_rocm_launchers.h"
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

extern int	launch_rf_predict_batch_kernel_hip(const NdbCudaRfNode *d_nodes,
											   const NdbCudaRfTreeHeader *d_trees,
											   int tree_count,
											   const float *d_features,
											   int n_samples,
											   int feature_dim,
											   int class_count,
											   int *d_votes);
extern int	ndb_rocm_rf_infer(const NdbCudaRfNode *nodes,
							  const NdbCudaRfTreeHeader *trees,
							  int tree_count,
							  const float *input,
							  int feature_dim,
							  int class_count,
							  int *votes);

static void
rf_copy_tree_nodes_rocm(const GTree *tree, NdbCudaRfNode *dest,
						int *node_offset, int *max_feat_idx)
{
	const GTreeNode *src_nodes;
	int			count;
	int			i;
	int			max_idx = -1;

	if (tree == NULL || dest == NULL || node_offset == NULL)
		return;

	src_nodes = gtree_nodes(tree);
	count = tree->count;
	for (i = 0; i < count; i++)
	{
		const GTreeNode *src = &src_nodes[i];
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

		if (src->feature_idx >= 0 && src->feature_idx > max_idx)
			max_idx = src->feature_idx;
	}
	*node_offset += count;
	if (max_feat_idx != NULL)
		*max_feat_idx = max_idx;
}

int
ndb_rocm_rf_pack_model(const struct RFModel *model,
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
			*errstr = pstrdup("invalid RF model for ROCm pack");
		return -1;
	}

	if (model->tree_count > 0 && model->trees != NULL)
	{
		tree_count = model->tree_count;
		for (i = 0; i < model->tree_count; i++)
		{
			const GTree *tree = model->trees[i];

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
			*errstr = pstrdup("ROCm RF pack: payload size exceeds MaxAllocSize");
		return -1;
	}
	NDB_DECLARE(bytea *, blob);
	NDB_DECLARE(char *, blob_raw);

	if (VARHDRSZ + payload_bytes > MaxAllocSize)
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF pack: total size exceeds MaxAllocSize");
		return -1;
	}
	NDB_ALLOC(blob_raw, char, VARHDRSZ + payload_bytes);
	blob = (bytea *) blob_raw;
	if (blob == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF pack: palloc failed");
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
		const GTree *tree =
			(model->tree_count > 0 && model->trees != NULL)
			? model->trees[i]
			: model->tree;
		int			node_count = tree ? tree->count : 0;

		tree_hdrs[i].node_count = node_count;
		tree_hdrs[i].nodes_start = node_cursor;
		tree_hdrs[i].root_index = tree ? tree->root : 0;
		tree_hdrs[i].max_feature_index = -1;

		if (node_count > 0)
			rf_copy_tree_nodes_rocm(tree, nodes, &node_cursor,
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
ndb_rocm_rf_predict(const bytea * model_data,
					const float *input,
					int feature_dim,
					int *class_out,
					char **errstr)
{
	const NdbCudaRfModelHeader *model_hdr;
	const NdbCudaRfTreeHeader *tree_hdrs;
	const NdbCudaRfNode *nodes;
	int		   *votes = NULL;
	int			i;
	int			best_class = 0;
	int			max_votes = 0;
	int			rc = -1;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || class_out == NULL
		|| feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid parameters for ROCm RF predict");
		return -1;
	}

	if (VARSIZE(model_data) - VARHDRSZ < (int) sizeof(NdbCudaRfModelHeader))
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF model data too small");
		return -1;
	}

	model_hdr = (const NdbCudaRfModelHeader *) VARDATA(model_data);
	if (model_hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = pstrdup("feature dimension mismatch in ROCm RF predict");
		return -1;
	}

	tree_hdrs = (const NdbCudaRfTreeHeader *) ((const char *) model_hdr
											   + sizeof(NdbCudaRfModelHeader));
	nodes = (const NdbCudaRfNode *) ((const char *) tree_hdrs
									 + sizeof(NdbCudaRfTreeHeader) * model_hdr->tree_count);

	votes = (int *) palloc0(sizeof(int) * model_hdr->class_count);
	if (votes == NULL)
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF predict: failed to allocate votes");
		return -1;
	}

	rc = ndb_rocm_rf_infer(nodes,
						   tree_hdrs,
						   model_hdr->tree_count,
						   input,
						   feature_dim,
						   model_hdr->class_count,
						   votes);

	if (rc != 0)
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF inference failed");
		NDB_FREE(votes);
		return -1;
	}

	/* Find class with most votes */
	for (i = 0; i < model_hdr->class_count; i++)
	{
		if (votes[i] > max_votes)
		{
			max_votes = votes[i];
			best_class = i;
		}
	}

	/* Fallback to majority class if no votes */
	if (max_votes == 0)
		best_class = model_hdr->majority_class;

	*class_out = best_class;
	NDB_FREE(votes);
	return 0;
}

int
ndb_rocm_rf_train(const float *features,
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
	NDB_DECLARE(int *, label_ints);
	NDB_DECLARE(int *, class_counts);
	NDB_DECLARE(int *, best_left_counts);
	NDB_DECLARE(int *, best_right_counts);
	NDB_DECLARE(int *, tmp_left_counts);
	NDB_DECLARE(int *, tmp_right_counts);
	NDB_DECLARE(bytea *, payload);
	NDB_DECLARE(char *, payload_raw);
	Jsonb	   *metrics_json = NULL;
	float	   *d_features = NULL;
	int		   *d_labels = NULL;
	hipError_t	status = hipSuccess;
	double		gini_accumulator = 0.0;
	size_t		feature_bytes;
	size_t		label_bytes;
	size_t		class_bytes;
	pg_prng_state rng;
	bool		seeded = false;
	int			i;
	int			j;
	int			rc = -1;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || labels == NULL || n_samples <= 0
		|| feature_dim <= 0 || class_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid input parameters for ROCm RF train");
		return -1;
	}

	/* Parse hyperparameters if provided */
	if (hyperparams != NULL)
	{
		/* Extract n_trees from JSONB if present */
		/* For now, use default */
		(void) hyperparams;
	}

	if (n_trees <= 0)
		n_trees = default_n_trees;
	if (class_count > 4096)
	{
		if (errstr)
			*errstr = pstrdup("ROCm RF training: class_count exceeds maximum of 4096");
		return -1;
	}

	/* Allocate host memory */
	{
		size_t		label_size = sizeof(int) * (size_t) n_samples;
		size_t		class_size = sizeof(int) * (size_t) class_count;

		if (label_size > MaxAllocSize || class_size > MaxAllocSize)
		{
			if (errstr)
				*errstr = pstrdup("ROCm RF train: allocation size exceeds MaxAllocSize");
			return -1;
		}

		NDB_ALLOC(label_ints, int, n_samples);
		NDB_ALLOC(class_counts, int, class_count);
		NDB_ALLOC(tmp_left_counts, int, class_count);
		NDB_ALLOC(tmp_right_counts, int, class_count);
		NDB_ALLOC(best_left_counts, int, class_count);
		NDB_ALLOC(best_right_counts, int, class_count);
		if (label_ints == NULL || class_counts == NULL ||
			tmp_left_counts == NULL || tmp_right_counts == NULL ||
			best_left_counts == NULL || best_right_counts == NULL)
		{
			if (errstr)
				*errstr = pstrdup("ROCm RF train: palloc failed");
			goto cleanup;
		}
	}

	/* Convert labels to integers */
	for (i = 0; i < n_samples; i++)
	{
		double		val = labels[i];

		label_ints[i] = (int) rint(val);
		if (label_ints[i] < 0 || label_ints[i] >= class_count)
			label_ints[i] = 0;
	}

	/* Compute class distribution */
	for (i = 0; i < n_samples; i++)
		class_counts[label_ints[i]]++;

	/* Allocate GPU memory */
	feature_bytes = sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
	label_bytes = sizeof(int) * (size_t) n_samples;
	class_bytes = sizeof(int) * (size_t) class_count;

	status = hipMalloc((void **) &d_features, feature_bytes);
	if (status != hipSuccess)
		goto gpu_fail;
	status = hipMalloc((void **) &d_labels, label_bytes);
	if (status != hipSuccess)
		goto gpu_fail;

	/* Copy data to GPU */
	status = hipMemcpy(d_features, features, feature_bytes, hipMemcpyHostToDevice);
	if (status != hipSuccess)
		goto gpu_fail;
	status = hipMemcpy(d_labels, label_ints, label_bytes, hipMemcpyHostToDevice);
	if (status != hipSuccess)
		goto gpu_fail;

	/* Initialize RNG */
	if (!seeded)
	{
		if (!pg_prng_strong_seed(&rng))
			pg_prng_seed(&rng, (uint64) n_samples ^ (uint64) feature_dim);
		seeded = true;
	}

	/* Find majority class */
	int			majority_class = 0;
	int			best_count = class_counts[0];
	double		majority_fraction;

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

	/* Allocate model payload */
	{
		int			total_nodes = n_trees * 3;
		size_t		header_bytes = sizeof(NdbCudaRfModelHeader)
			+ sizeof(NdbCudaRfTreeHeader) * n_trees;
		size_t		payload_bytes = header_bytes + sizeof(NdbCudaRfNode) * total_nodes;
		char	   *base;
		NdbCudaRfModelHeader model_hdr;
		NdbCudaRfTreeHeader *tree_hdrs;
		NdbCudaRfNode *nodes;

		if (payload_bytes > MaxAllocSize || VARHDRSZ + payload_bytes > MaxAllocSize)
		{
			if (errstr)
				*errstr = pstrdup("ROCm RF train: payload size exceeds MaxAllocSize");
			goto gpu_fail;
		}

		NDB_ALLOC(payload_raw, char, VARHDRSZ + payload_bytes);
		payload = (bytea *) payload_raw;
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

		/* Build trees */
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

			/* Find best split (CPU computation for now, can be moved to GPU) */
			for (j = 0; j < feature_dim; j++)
			{
				double		sum = 0.0;
				double		sumsq = 0.0;
				double		variance;
				float		threshold;
				int			k;

				/* Compute feature statistics */
				for (k = 0; k < n_samples; k++)
				{
					float		val = features[k * feature_dim + j];

					sum += val;
					sumsq += val * val;
				}

				if (sum == 0.0 && sumsq == 0.0)
					continue;

				threshold = (float) (sum / (double) n_samples);
				variance = (sumsq / (double) n_samples) - (threshold * threshold);
				if (variance < 0.0)
					variance = 0.0;
				if (variance > 0.0)
					threshold += (float) (noise * sqrt(variance) * 0.25);

				/* Compute split counts */
				memset(tmp_left_counts, 0, class_bytes);
				memset(tmp_right_counts, 0, class_bytes);
				for (k = 0; k < n_samples; k++)
				{
					float		val = features[k * feature_dim + j];
					int			label = label_ints[k];

					if (val <= threshold)
						tmp_left_counts[label]++;
					else
						tmp_right_counts[label]++;
				}

				/* Compute Gini impurity using shared function */
				{
					double		gini;
					int			left_tot;
					int			right_tot;

					gini = rf_split_gini(tmp_left_counts,
										 tmp_right_counts,
										 class_count,
										 &left_tot,
										 &right_tot,
										 NULL,
										 NULL);

					if (left_tot == 0 || right_tot == 0)
						continue;

					if (gini < best_gini && gini >= 0.0)
					{
						best_gini = gini;
						best_feature = j;
						best_threshold = threshold;
						memcpy(best_left_counts, tmp_left_counts, class_bytes);
						memcpy(best_right_counts, tmp_right_counts, class_bytes);
					}
				}
			}

			/* Create tree structure */
			if (best_feature < 0)
			{
				tree_hdrs[i].node_count = 1;
				tree_hdrs[i].nodes_start = node_offset;
				tree_hdrs[i].root_index = 0;
				tree_hdrs[i].max_feature_index = -1;
				nodes[node_offset].feature_idx = -1;
				nodes[node_offset].threshold = 0.0f;
				nodes[node_offset].left_child = -1;
				nodes[node_offset].right_child = -1;
				nodes[node_offset].value = (float) majority_class;
				continue;
			}

			/* Find majority classes for left and right */
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

			nodes[node_offset + 1].feature_idx = -1;
			nodes[node_offset + 1].threshold = 0.0f;
			nodes[node_offset + 1].left_child = -1;
			nodes[node_offset + 1].right_child = -1;
			nodes[node_offset + 1].value = (float) left_majority;

			nodes[node_offset + 2].feature_idx = -1;
			nodes[node_offset + 2].threshold = 0.0f;
			nodes[node_offset + 2].left_child = -1;
			nodes[node_offset + 2].right_child = -1;
			nodes[node_offset + 2].value = (float) right_majority;

			if (best_gini > 0.0 && best_gini < DBL_MAX / 4.0)
				gini_accumulator += best_gini;
		}
	}

	/* Build metrics */
	{
		RFMetricsSpec spec;

		memset(&spec, 0, sizeof(spec));
		spec.storage = "gpu";
		spec.algorithm = "random_forest";
		spec.tree_count = n_trees;
		spec.majority_class = majority_class;
		spec.majority_fraction = majority_fraction;
		spec.gini = (n_trees > 0) ? (gini_accumulator / (double) n_trees) : 0.0;
		spec.oob_accuracy = 0.0;	/* OOB not computed in simplified version */
		metrics_json = rf_build_metrics_json(&spec);
	}

	*model_data = payload;
	if (metrics != NULL)
		*metrics = metrics_json;

	rc = 0;

gpu_fail:
	if (d_features != NULL)
		hipFree(d_features);
	if (d_labels != NULL)
		hipFree(d_labels);

cleanup:
	if (rc != 0)
	{
		if (payload != NULL)
			NDB_FREE(payload);
		if (metrics_json != NULL)
			NDB_FREE(metrics_json);
	}
	if (label_ints != NULL)
		NDB_FREE(label_ints);
	if (class_counts != NULL)
		NDB_FREE(class_counts);
	if (tmp_left_counts != NULL)
		NDB_FREE(tmp_left_counts);
	if (tmp_right_counts != NULL)
		NDB_FREE(tmp_right_counts);
	if (best_left_counts != NULL)
		NDB_FREE(best_left_counts);
	if (best_right_counts != NULL)
		NDB_FREE(best_right_counts);

	return rc;
}

#endif							/* NDB_GPU_HIP */
