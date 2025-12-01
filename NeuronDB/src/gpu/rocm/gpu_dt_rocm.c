/*-------------------------------------------------------------------------
 *
 * gpu_dt_rocm.c
 *    ROCm backend bridge for Decision Tree training and prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/rocm/gpu_dt_rocm.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#ifdef NDB_GPU_HIP

#include <float.h>
#include <math.h>
#include <string.h>

#include "neurondb_rocm_runtime.h"
#include "lib/stringinfo.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"

#include "ml_decision_tree_internal.h"
#include "neurondb_rocm_dt.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * Helper: Free tree recursively
 */
static void
dt_free_tree(DTNode * node)
{
	if (node == NULL)
		return;
	if (!node->is_leaf)
	{
		dt_free_tree(node->left);
		dt_free_tree(node->right);
	}
	NDB_FREE(node);
}

/*
 * Helper function to serialize tree nodes recursively
 */
static int
dt_serialize_node_recursive(const DTNode * node,
							NdbCudaDtNode * dest,
							int *node_idx,
							int *max_idx)
{
	int			current_idx;
	int			left_idx = -1;
	int			right_idx = -1;

	if (node == NULL || dest == NULL || node_idx == NULL)
		return -1;

	current_idx = *node_idx;
	if (current_idx >= *max_idx)
		return -1;

	dest[current_idx].is_leaf = node->is_leaf;
	dest[current_idx].value = (float) node->leaf_value;
	dest[current_idx].feature_idx = node->feature_idx;
	dest[current_idx].threshold = node->threshold;
	dest[current_idx].left_child = -1;
	dest[current_idx].right_child = -1;

	if (!node->is_leaf)
	{
		(*node_idx)++;
		left_idx = *node_idx;
		if (dt_serialize_node_recursive(
										node->left, dest, node_idx, max_idx)
			< 0)
			return -1;

		(*node_idx)++;
		right_idx = *node_idx;
		if (dt_serialize_node_recursive(
										node->right, dest, node_idx, max_idx)
			< 0)
			return -1;

		dest[current_idx].left_child = left_idx;
		dest[current_idx].right_child = right_idx;
	}

	return 0;
}

/*
 * Count nodes in tree recursively
 */
static int
dt_count_nodes(const DTNode * node)
{
	if (node == NULL)
		return 0;
	if (node->is_leaf)
		return 1;
	return 1 + dt_count_nodes(node->left) + dt_count_nodes(node->right);
}

int
ndb_rocm_dt_pack_model(const DTModel * model,
					   bytea * *model_data,
					   Jsonb * *metrics,
					   char **errstr)
{
	int			node_count = 0;
	size_t		header_bytes;
	size_t		nodes_bytes;
	size_t		payload_bytes;
	bytea	   *blob;
	char	   *base;
	NdbCudaDtModelHeader *hdr;
	NdbCudaDtNode *nodes;
	int			node_idx = 0;

	if (errstr)
		*errstr = NULL;
	if (model == NULL || model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("invalid DT model for HIP pack");
		return -1;
	}

	if (model->root == NULL)
	{
		if (errstr)
			*errstr =
				pstrdup("decision_tree model has no root node");
		return -1;
	}

	node_count = dt_count_nodes(model->root);
	if (node_count <= 0)
	{
		if (errstr)
			*errstr = pstrdup("decision_tree model empty");
		return -1;
	}

	header_bytes = sizeof(NdbCudaDtModelHeader);
	nodes_bytes = sizeof(NdbCudaDtNode) * (size_t) node_count;
	payload_bytes = header_bytes + nodes_bytes;

	blob = (bytea *) palloc(VARHDRSZ + payload_bytes);
	SET_VARSIZE(blob, VARHDRSZ + payload_bytes);
	base = VARDATA(blob);

	hdr = (NdbCudaDtModelHeader *) base;
	hdr->feature_dim = model->n_features;
	hdr->n_samples = model->n_samples;
	hdr->max_depth = model->max_depth;
	hdr->min_samples_split = model->min_samples_split;
	hdr->node_count = node_count;

	nodes = (NdbCudaDtNode *) (base + sizeof(NdbCudaDtModelHeader));
	if (dt_serialize_node_recursive(
									model->root, nodes, &node_idx, &node_count)
		< 0)
	{
		NDB_FREE(blob);
		if (errstr)
			*errstr = pstrdup(
							  "failed to serialize decision tree nodes");
		return -1;
	}

	if (metrics != NULL)
	{
		/*
		 * Don't use DirectFunctionCall - it crashes in HIP context. Build
		 * JSONB manually using JsonbBuilder API.
		 */
		JsonbParseState *state = NULL;
		JsonbValue	k,
					v;
		Jsonb	   *metrics_json;

		(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

		/* Add "algorithm": "decision_tree" */
		k.type = jbvString;
		k.val.string.len = strlen("algorithm");
		k.val.string.val = "algorithm";
		(void) pushJsonbValue(&state, WJB_KEY, &k);

		v.type = jbvString;
		v.val.string.len = strlen("decision_tree");
		v.val.string.val = "decision_tree";
		(void) pushJsonbValue(&state, WJB_VALUE, &v);

		/* Add "storage": "gpu" */
		k.type = jbvString;
		k.val.string.len = strlen("storage");
		k.val.string.val = "storage";
		(void) pushJsonbValue(&state, WJB_KEY, &k);

		v.type = jbvString;
		v.val.string.len = strlen("gpu");
		v.val.string.val = "gpu";
		(void) pushJsonbValue(&state, WJB_VALUE, &v);

		/* Add "n_features": model->n_features */
		k.type = jbvString;
		k.val.string.len = strlen("n_features");
		k.val.string.val = "n_features";
		(void) pushJsonbValue(&state, WJB_KEY, &k);

		v.type = jbvNumeric;
		v.val.numeric = int64_to_numeric(model->n_features);
		(void) pushJsonbValue(&state, WJB_VALUE, &v);

		/* Add "n_samples": model->n_samples */
		k.type = jbvString;
		k.val.string.len = strlen("n_samples");
		k.val.string.val = "n_samples";
		(void) pushJsonbValue(&state, WJB_KEY, &k);

		v.type = jbvNumeric;
		v.val.numeric = int64_to_numeric(model->n_samples);
		(void) pushJsonbValue(&state, WJB_VALUE, &v);

		/* Add "max_depth": model->max_depth */
		k.type = jbvString;
		k.val.string.len = strlen("max_depth");
		k.val.string.val = "max_depth";
		(void) pushJsonbValue(&state, WJB_KEY, &k);

		v.type = jbvNumeric;
		v.val.numeric = int64_to_numeric(model->max_depth);
		(void) pushJsonbValue(&state, WJB_VALUE, &v);

		/* Add "min_samples_split": model->min_samples_split */
		k.type = jbvString;
		k.val.string.len = strlen("min_samples_split");
		k.val.string.val = "min_samples_split";
		(void) pushJsonbValue(&state, WJB_KEY, &k);

		v.type = jbvNumeric;
		v.val.numeric = int64_to_numeric(model->min_samples_split);
		(void) pushJsonbValue(&state, WJB_VALUE, &v);

		/* Add "node_count": node_count */
		k.type = jbvString;
		k.val.string.len = strlen("node_count");
		k.val.string.val = "node_count";
		(void) pushJsonbValue(&state, WJB_KEY, &k);

		v.type = jbvNumeric;
		v.val.numeric = int64_to_numeric(node_count);
		(void) pushJsonbValue(&state, WJB_VALUE, &v);

		metrics_json = JsonbValueToJsonb(pushJsonbValue(&state, WJB_END_OBJECT, NULL));
		*metrics = metrics_json;
	}

	*model_data = blob;
	return 0;
}

/*
 * Helper: Compute Gini impurity from class counts
 */
static double
dt_compute_gini(const int *class_counts, int class_count, int total)
{
	double		gini = 1.0;
	int			i;

	if (total <= 0)
		return 0.0;

	for (i = 0; i < class_count; i++)
	{
		double		p = (double) class_counts[i] / (double) total;

		gini -= p * p;
	}

	return gini;
}

/*
 * Helper: Compute variance from sum and sum of squares
 */
static double
dt_compute_variance(double sum, double sumsq, int count)
{
	double		mean;
	double		variance;

	if (count <= 0)
		return 0.0;

	mean = sum / (double) count;
	variance = (sumsq / (double) count) - (mean * mean);

	return (variance > 0.0) ? variance : 0.0;
}

/*
 * Helper: Build tree node recursively with GPU-accelerated split finding
 */
static DTNode *
dt_build_tree_gpu(const float *features,
				  const double *labels,
				  const int *indices,
				  int n_samples,
				  int feature_dim,
				  int max_depth,
				  int min_samples_split,
				  bool is_classification,
				  int class_count,
				  char **errstr)
{
	DTNode	   *node;
	int			i;
	int			best_feature = -1;
	float		best_threshold = 0.0f;
	double		best_gain = -DBL_MAX;
	int		   *left_indices = NULL;
	int		   *right_indices = NULL;
	int			left_count = 0;
	int			right_count = 0;
	int		   *label_ints = NULL;
	int		   *left_counts = NULL;
	int		   *right_counts = NULL;
	double		left_sum = 0.0;
	double		left_sumsq = 0.0;
	int			left_count_reg = 0;
	double		right_sum = 0.0;
	double		right_sumsq = 0.0;
	int			right_count_reg = 0;
	int			majority;
	int		   *parent_counts;
	double		parent_imp;
	double		left_var;
	double		right_var;
	double		parent_var;

	if (errstr)
		*errstr = NULL;

	/* Allocate node */
	node = (DTNode *) palloc0(sizeof(DTNode));
	if (node == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP DT train: failed to allocate node");
		return NULL;
	}

	/* Stopping criteria */
	if (max_depth <= 0 || n_samples < min_samples_split)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			/* Count classes for majority vote */
			int		   *class_counts = (int *) palloc0(sizeof(int) * class_count);

			for (i = 0; i < n_samples; i++)
			{
				int			label = (int) labels[indices[i]];

				if (label >= 0 && label < class_count)
					class_counts[label]++;
			}
			majority = 0;
			for (i = 1; i < class_count; i++)
			{
				if (class_counts[i] > class_counts[majority])
					majority = i;
			}
			node->leaf_value = (double) majority;
			NDB_FREE(class_counts);
		}
		else
		{
			/* Compute mean for regression */
			double		sum = 0.0;

			for (i = 0; i < n_samples; i++)
				sum += labels[indices[i]];
			node->leaf_value = (n_samples > 0) ? (sum / (double) n_samples) : 0.0;
		}
		return node;
	}

	/* Allocate working arrays */
	left_indices = (int *) palloc(sizeof(int) * n_samples);
	right_indices = (int *) palloc(sizeof(int) * n_samples);
	if (left_indices == NULL || right_indices == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP DT train: failed to allocate index arrays");
		if (left_indices)
			NDB_FREE(left_indices);
		if (right_indices)
			NDB_FREE(right_indices);
		NDB_FREE(node);
		return NULL;
	}

	if (is_classification)
	{
		label_ints = (int *) palloc(sizeof(int) * n_samples);
		left_counts = (int *) palloc0(sizeof(int) * class_count);
		right_counts = (int *) palloc0(sizeof(int) * class_count);
		if (label_ints == NULL || left_counts == NULL || right_counts == NULL)
		{
			if (errstr)
				*errstr = pstrdup("HIP DT train: failed to allocate classification arrays");
			if (label_ints)
				NDB_FREE(label_ints);
			if (left_counts)
				NDB_FREE(left_counts);
			if (right_counts)
				NDB_FREE(right_counts);
			NDB_FREE(left_indices);
			NDB_FREE(right_indices);
			NDB_FREE(node);
			return NULL;
		}

		/* Convert labels to integers */
		for (i = 0; i < n_samples; i++)
		{
			int			label = (int) labels[indices[i]];

			if (label < 0 || label >= class_count)
				label = 0;
			label_ints[i] = label;
		}
	}

	/* Find best split using GPU-accelerated kernels */
	for (int feat = 0; feat < feature_dim; feat++)
	{
		float		min_val = FLT_MAX;
		float		max_val = -FLT_MAX;
		double		sum = 0.0;
		double		sumsq = 0.0;

		/* Compute feature statistics using GPU */
		if (ndb_rocm_dt_launch_feature_stats(features, (int *) indices, n_samples,
											 feature_dim, feat, &min_val, &max_val, &sum, &sumsq) != 0)
			continue;

		if (min_val == max_val || !isfinite(min_val) || !isfinite(max_val))
			continue;

		/* Try candidate thresholds (10 uniformly spaced) */
		for (int thresh_idx = 1; thresh_idx < 10; thresh_idx++)
		{
			float		threshold = min_val + (max_val - min_val) * (float) thresh_idx / 10.0f;
			double		gain = 0.0;
			double		left_imp = 0.0;
			double		right_imp = 0.0;
			int			left_total = 0;
			int			right_total = 0;

			if (is_classification)
			{
				/* Use GPU kernel for classification */
				memset(left_counts, 0, sizeof(int) * class_count);
				memset(right_counts, 0, sizeof(int) * class_count);

				if (ndb_rocm_dt_launch_split_counts_classification(features, label_ints,
																   (int *) indices, n_samples, feature_dim, feat, threshold,
																   class_count, left_counts, right_counts) != 0)
					continue;

				/* Compute totals */
				left_total = 0;
				right_total = 0;
				for (i = 0; i < class_count; i++)
				{
					left_total += left_counts[i];
					right_total += right_counts[i];
				}

				if (left_total <= 0 || right_total <= 0)
					continue;

				/* Compute parent impurity from all samples in current node */
				parent_counts = (int *) palloc0(sizeof(int) * class_count);
				for (i = 0; i < n_samples; i++)
				{
					int			label = label_ints[i];

					if (label >= 0 && label < class_count)
						parent_counts[label]++;
				}
				parent_imp = dt_compute_gini(parent_counts, class_count, n_samples);
				NDB_FREE(parent_counts);

				/* Compute Gini impurity */
				left_imp = dt_compute_gini(left_counts, class_count, left_total);
				right_imp = dt_compute_gini(right_counts, class_count, right_total);

				/* Information gain */
				gain = parent_imp - (((double) left_total / (double) n_samples) * left_imp +
									 ((double) right_total / (double) n_samples) * right_imp);
			}
			else
			{
				/* Use GPU kernel for regression */
				left_sum = 0.0;
				left_sumsq = 0.0;
				left_count_reg = 0;
				right_sum = 0.0;
				right_sumsq = 0.0;
				right_count_reg = 0;

				if (ndb_rocm_dt_launch_split_stats_regression(features, labels,
															  (int *) indices, n_samples, feature_dim, feat, threshold,
															  &left_sum, &left_sumsq, &left_count_reg,
															  &right_sum, &right_sumsq, &right_count_reg) != 0)
					continue;

				if (left_count_reg <= 0 || right_count_reg <= 0)
					continue;

				/* Compute variance */
				left_var = dt_compute_variance(left_sum, left_sumsq, left_count_reg);
				right_var = dt_compute_variance(right_sum, right_sumsq, right_count_reg);

				/* Variance reduction */
				parent_var = dt_compute_variance(left_sum + right_sum,
												 left_sumsq + right_sumsq, left_count_reg + right_count_reg);
				gain = parent_var - (((double) left_count_reg / (double) n_samples) * left_var +
									 ((double) right_count_reg / (double) n_samples) * right_var);
				left_total = left_count_reg;
				right_total = right_count_reg;
			}

			/* Update best split if this is better */
			if (gain > best_gain && isfinite(gain))
			{
				best_gain = gain;
				best_feature = feat;
				best_threshold = threshold;
			}
		}
	}

	/* Clean up working arrays */
	if (label_ints)
		NDB_FREE(label_ints);
	if (left_counts)
		NDB_FREE(left_counts);
	if (right_counts)
		NDB_FREE(right_counts);

	/* If no good split found, make leaf */
	if (best_feature < 0 || best_gain <= 0.0)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int		   *class_counts = (int *) palloc0(sizeof(int) * class_count);

			for (i = 0; i < n_samples; i++)
			{
				int			label = (int) labels[indices[i]];

				if (label >= 0 && label < class_count)
					class_counts[label]++;
			}
			majority = 0;
			for (i = 1; i < class_count; i++)
			{
				if (class_counts[i] > class_counts[majority])
					majority = i;
			}
			node->leaf_value = (double) majority;
			NDB_FREE(class_counts);
		}
		else
		{
			double		sum = 0.0;

			for (i = 0; i < n_samples; i++)
				sum += labels[indices[i]];
			node->leaf_value = (n_samples > 0) ? (sum / (double) n_samples) : 0.0;
		}
		NDB_FREE(left_indices);
		NDB_FREE(right_indices);
		return node;
	}

	/* Partition indices based on best split */
	for (i = 0; i < n_samples; i++)
	{
		float		val = features[indices[i] * feature_dim + best_feature];

		if (isfinite(val) && val <= best_threshold)
			left_indices[left_count++] = indices[i];
		else
			right_indices[right_count++] = indices[i];
	}

	/* Validate split */
	if (left_count <= 0 || right_count <= 0)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int		   *class_counts = (int *) palloc0(sizeof(int) * class_count);

			for (i = 0; i < n_samples; i++)
			{
				int			label = (int) labels[indices[i]];

				if (label >= 0 && label < class_count)
					class_counts[label]++;
			}
			majority = 0;
			for (i = 1; i < class_count; i++)
			{
				if (class_counts[i] > class_counts[majority])
					majority = i;
			}
			node->leaf_value = (double) majority;
			NDB_FREE(class_counts);
		}
		else
		{
			double		sum = 0.0;

			for (i = 0; i < n_samples; i++)
				sum += labels[indices[i]];
			node->leaf_value = (n_samples > 0) ? (sum / (double) n_samples) : 0.0;
		}
		NDB_FREE(left_indices);
		NDB_FREE(right_indices);
		return node;
	}

	/* Build left and right subtrees recursively */
	node->is_leaf = false;
	node->feature_idx = best_feature;
	node->threshold = best_threshold;

	node->left = dt_build_tree_gpu(features, labels, left_indices, left_count,
								   feature_dim, max_depth - 1, min_samples_split, is_classification,
								   class_count, errstr);
	if (node->left == NULL)
	{
		NDB_FREE(left_indices);
		NDB_FREE(right_indices);
		NDB_FREE(node);
		return NULL;
	}

	node->right = dt_build_tree_gpu(features, labels, right_indices, right_count,
									feature_dim, max_depth - 1, min_samples_split, is_classification,
									class_count, errstr);
	if (node->right == NULL)
	{
		dt_free_tree(node->left);
		NDB_FREE(left_indices);
		NDB_FREE(right_indices);
		NDB_FREE(node);
		return NULL;
	}

	NDB_FREE(left_indices);
	NDB_FREE(right_indices);
	return node;
}

int
ndb_rocm_dt_train(const float *features,
				  const double *labels,
				  int n_samples,
				  int feature_dim,
				  const Jsonb * hyperparams,
				  bytea * *model_data,
				  Jsonb * *metrics,
				  char **errstr)
{
	int			max_depth = 10;
	int			min_samples_split = 2;
	bool		is_classification = true;
	int			class_count = 2;
	DTModel    *model = NULL;
	DTNode	   *root = NULL;
	int		   *indices = NULL;
	int			i;
	int			rc = -1;

	if (errstr)
		*errstr = NULL;

	/* Comprehensive input validation */
	if (features == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP DT train: features array is NULL");
		return -1;
	}
	if (labels == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP DT train: labels array is NULL");
		return -1;
	}
	if (model_data == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP DT train: model_data output pointer is NULL");
		return -1;
	}
	if (n_samples <= 0 || n_samples > 100000000)
	{
		if (errstr)
			*errstr = pstrdup("HIP DT train: n_samples must be between 1 and 100000000");
		return -1;
	}
	if (feature_dim <= 0 || feature_dim > 1000000)
	{
		if (errstr)
			*errstr = pstrdup("HIP DT train: feature_dim must be between 1 and 1000000");
		return -1;
	}

	/* Extract hyperparameters - wrap in PG_TRY to handle JSONB parsing errors */
	if (hyperparams != NULL)
	{
		PG_TRY();
		{
			Datum		max_depth_datum;
			Datum		min_samples_split_datum;
			Datum		is_classification_datum;
			Datum		class_count_datum;
			Datum		numeric_datum;
			Numeric		num;

			max_depth_datum = DirectFunctionCall2(
												  jsonb_object_field,
												  JsonbPGetDatum(hyperparams),
												  CStringGetTextDatum("max_depth"));
			if (DatumGetPointer(max_depth_datum) != NULL)
			{
				numeric_datum = DirectFunctionCall1(
													jsonb_numeric, max_depth_datum);
				if (DatumGetPointer(numeric_datum) != NULL)
				{
					num = DatumGetNumeric(numeric_datum);
					max_depth = DatumGetInt32(
											  DirectFunctionCall1(numeric_int4,
																  NumericGetDatum(num)));
					if (max_depth <= 0)
						max_depth = 10;
					if (max_depth > 100)
						max_depth = 100;
				}
			}

			min_samples_split_datum = DirectFunctionCall2(
														  jsonb_object_field,
														  JsonbPGetDatum(hyperparams),
														  CStringGetTextDatum("min_samples_split"));
			if (DatumGetPointer(min_samples_split_datum) != NULL)
			{
				numeric_datum = DirectFunctionCall1(
													jsonb_numeric, min_samples_split_datum);
				if (DatumGetPointer(numeric_datum) != NULL)
				{
					num = DatumGetNumeric(numeric_datum);
					min_samples_split = DatumGetInt32(
													  DirectFunctionCall1(numeric_int4,
																		  NumericGetDatum(num)));
					if (min_samples_split < 2)
						min_samples_split = 2;
				}
			}

			is_classification_datum = DirectFunctionCall2(
														  jsonb_object_field,
														  JsonbPGetDatum(hyperparams),
														  CStringGetTextDatum("is_classification"));
			if (DatumGetPointer(is_classification_datum) != NULL)
			{
				bool		val = DatumGetBool(
											   DirectFunctionCall1(jsonb_bool, is_classification_datum));

				is_classification = val;
			}

			class_count_datum = DirectFunctionCall2(
													jsonb_object_field,
													JsonbPGetDatum(hyperparams),
													CStringGetTextDatum("class_count"));
			if (DatumGetPointer(class_count_datum) != NULL)
			{
				numeric_datum = DirectFunctionCall1(
													jsonb_numeric, class_count_datum);
				if (DatumGetPointer(numeric_datum) != NULL)
				{
					num = DatumGetNumeric(numeric_datum);
					class_count = DatumGetInt32(
												DirectFunctionCall1(numeric_int4,
																	NumericGetDatum(num)));
					if (class_count < 2)
						class_count = 2;
					if (class_count > 1000)
						class_count = 1000;
				}
			}
		}
		PG_CATCH();
		{
			/* If JSONB parsing fails, just use defaults */
			FlushErrorState();
		}
		PG_END_TRY();
	}

	/* Validate input data for NaN/Inf */
	for (i = 0; i < n_samples; i++)
	{
		if (!isfinite(labels[i]))
		{
			if (errstr)
				*errstr = pstrdup("HIP DT train: non-finite value in labels array");
			return -1;
		}
		for (int j = 0; j < feature_dim; j++)
		{
			if (!isfinite(features[i * feature_dim + j]))
			{
				if (errstr)
					*errstr = pstrdup("HIP DT train: non-finite value in features array");
				return -1;
			}
		}
	}

	/* Create index array */
	indices = (int *) palloc(sizeof(int) * n_samples);
	if (indices == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP DT train: failed to allocate indices array");
		return -1;
	}
	for (i = 0; i < n_samples; i++)
		indices[i] = i;

	/* Build tree using GPU-accelerated split finding */
	root = dt_build_tree_gpu(features, labels, indices, n_samples, feature_dim,
							 max_depth, min_samples_split, is_classification, class_count, errstr);
	if (root == NULL)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("HIP DT train: failed to build tree");
		NDB_FREE(indices);
		return -1;
	}

	/* Create model structure */
	model = (DTModel *) palloc0(sizeof(DTModel));
	if (model == NULL)
	{
		if (errstr)
			*errstr = pstrdup("HIP DT train: failed to allocate model");
		dt_free_tree(root);
		NDB_FREE(indices);
		return -1;
	}

	model->model_id = 0;		/* Will be set by catalog */
	model->n_features = feature_dim;
	model->n_samples = n_samples;
	model->max_depth = max_depth;
	model->min_samples_split = min_samples_split;
	model->root = root;

	/* Pack model */
	if (ndb_rocm_dt_pack_model(model, model_data, metrics, errstr) != 0)
	{
		if (errstr && *errstr == NULL)
			*errstr = pstrdup("HIP DT train: model packing failed");
		dt_free_tree(root);
		NDB_FREE(model);
		NDB_FREE(indices);
		return -1;
	}

	NDB_FREE(indices);
	NDB_FREE(model); /* Note: root is now owned by packed model */

	rc = 0;
	return rc;
}

int
ndb_rocm_dt_predict(const bytea * model_data,
					const float *input,
					int feature_dim,
					double *prediction_out,
					char **errstr)
{
	const		NdbCudaDtModelHeader *hdr;
	const		NdbCudaDtNode *nodes;
	const		bytea *detoasted;
	int			node_idx = 0;

	if (errstr)
		*errstr = NULL;
	if (model_data == NULL || input == NULL || prediction_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup(
							  "invalid parameters for HIP DT predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted =
		(const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t		min_size = sizeof(NdbCudaDtModelHeader);
		size_t		actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < min_size)
		{
			if (errstr)
				*errstr = psprintf(
								   "model data too small: expected at "
								   "least %zu bytes, got %zu",
								   min_size,
								   actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaDtModelHeader *) VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model "
							   "has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		return -1;
	}

	nodes = (const NdbCudaDtNode *) ((const char *) hdr
									 + sizeof(NdbCudaDtModelHeader));

	/* Traverse tree to make prediction */
	while (node_idx >= 0 && node_idx < hdr->node_count)
	{
		const		NdbCudaDtNode *node = &nodes[node_idx];

		if (node->is_leaf)
		{
			*prediction_out = (double) node->value;
			return 0;
		}

		if (input[node->feature_idx] <= node->threshold)
			node_idx = node->left_child;
		else
			node_idx = node->right_child;
	}

	if (errstr)
		*errstr = pstrdup(
						  "tree traversal failed - invalid tree structure");
	return -1;
}

/*
 * Batch prediction: predict for multiple samples
 */
int
ndb_rocm_dt_predict_batch(const bytea * model_data,
						  const float *features,
						  int n_samples,
						  int feature_dim,
						  int *predictions_out,
						  char **errstr)
{
	const		NdbCudaDtModelHeader *hdr;
	const		NdbCudaDtNode *nodes;
	const		bytea *detoasted;
	int			i;
	int			node_idx;

	if (errstr)
		*errstr = NULL;

	if (model_data == NULL || features == NULL || predictions_out == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid inputs for HIP DT batch predict");
		return -1;
	}

	/* Detoast the bytea to ensure we have the full data */
	detoasted =
		(const bytea *) PG_DETOAST_DATUM(PointerGetDatum(model_data));

	/* Validate bytea size */
	{
		size_t		min_size = sizeof(NdbCudaDtModelHeader);
		size_t		actual_size = VARSIZE(detoasted) - VARHDRSZ;

		if (actual_size < min_size)
		{
			if (errstr)
				*errstr = psprintf(
								   "model data too small: expected at "
								   "least %zu bytes, got %zu",
								   min_size,
								   actual_size);
			return -1;
		}
	}

	hdr = (const NdbCudaDtModelHeader *) VARDATA(detoasted);
	if (hdr->feature_dim != feature_dim)
	{
		if (errstr)
			*errstr = psprintf("feature dimension mismatch: model "
							   "has %d, input has %d",
							   hdr->feature_dim,
							   feature_dim);
		return -1;
	}

	nodes = (const NdbCudaDtNode *) ((const char *) hdr
									 + sizeof(NdbCudaDtModelHeader));

	/* Predict for each sample */
	for (i = 0; i < n_samples; i++)
	{
		const float *input = features + (i * feature_dim);

		node_idx = 0;

		/* Traverse tree to make prediction */
		while (node_idx >= 0 && node_idx < hdr->node_count)
		{
			const		NdbCudaDtNode *node = &nodes[node_idx];

			if (node->is_leaf)
			{
				predictions_out[i] = (int) rint(node->value);
				break;
			}

			if (input[node->feature_idx] <= node->threshold)
				node_idx = node->left_child;
			else
				node_idx = node->right_child;
		}

		/* If traversal failed, use default prediction */
		if (node_idx < 0 || node_idx >= hdr->node_count)
		{
			if (errstr && *errstr == NULL)
				*errstr = pstrdup(
								  "tree traversal failed - invalid tree structure");
			predictions_out[i] = 0;
		}
	}

	return 0;
}

/*
 * Batch evaluation: compute metrics for multiple samples
 */
int
ndb_rocm_dt_evaluate_batch(const bytea * model_data,
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
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;
	int			i;
	int			total_correct = 0;
	int			rc;

	if (errstr)
		*errstr = NULL;

	if (model_data == NULL || features == NULL || labels == NULL
		|| n_samples <= 0 || feature_dim <= 0)
	{
		if (errstr)
			*errstr = pstrdup("invalid inputs for HIP DT batch evaluate");
		return -1;
	}

	if (accuracy_out == NULL || precision_out == NULL
		|| recall_out == NULL || f1_out == NULL)
	{
		if (errstr)
			*errstr = pstrdup("output pointers are NULL");
		return -1;
	}

	/* Allocate predictions array */
	predictions = (int *) palloc(sizeof(int) * (size_t) n_samples);
	if (predictions == NULL)
	{
		if (errstr)
			*errstr = pstrdup("failed to allocate predictions array");
		return -1;
	}

	/* Batch predict */
	rc = ndb_rocm_dt_predict_batch(model_data,
								   features,
								   n_samples,
								   feature_dim,
								   predictions,
								   errstr);

	if (rc != 0)
	{
		NDB_FREE(predictions);
		return -1;
	}

	/* Compute confusion matrix for binary classification */
	for (i = 0; i < n_samples; i++)
	{
		int			true_label = labels[i];
		int			pred_label = predictions[i];

		if (true_label < 0 || true_label > 1)
			continue;
		if (pred_label < 0 || pred_label > 1)
			continue;

		if (true_label == 1 && pred_label == 1)
		{
			tp++;
			total_correct++;
		}
		else if (true_label == 0 && pred_label == 0)
		{
			tn++;
			total_correct++;
		}
		else if (true_label == 0 && pred_label == 1)
			fp++;
		else if (true_label == 1 && pred_label == 0)
			fn++;
	}

	/* Compute metrics */
	*accuracy_out = (n_samples > 0)
		? ((double) total_correct / (double) n_samples)
		: 0.0;

	if ((tp + fp) > 0)
		*precision_out = (double) tp / (double) (tp + fp);
	else
		*precision_out = 0.0;

	if ((tp + fn) > 0)
		*recall_out = (double) tp / (double) (tp + fn);
	else
		*recall_out = 0.0;

	if ((*precision_out + *recall_out) > 0.0)
		*f1_out = 2.0 * ((*precision_out) * (*recall_out))
			/ ((*precision_out) + (*recall_out));
	else
		*f1_out = 0.0;

	NDB_FREE(predictions);

	return 0;
}

#endif							/* NDB_GPU_HIP */
