/*-------------------------------------------------------------------------
 *
 * ml_random_forest.c
 *    Random forest ensemble learning.
 *
 * This module implements random forest for classification and regression using
 * bootstrap aggregating and random feature selection. Models are serialized
 * and stored in the catalog for prediction.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_random_forest.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/array.h"
#include "utils/memutils.h"
#include "utils/jsonb.h"
#include "common/pg_prng.h"
#include "libpq/pqformat.h"
#include "access/xact.h"

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdint.h>

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu_bridge.h"
#include "ml_gpu_random_forest.h"
#include "neurondb_gpu.h"
#include "ml_catalog.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"
#include "neurondb_spi.h"
#include "gtree.h"
#include "ml_random_forest_internal.h"
#include "ml_random_forest_shared.h"
#include "neurondb_cuda_rf.h"
#include "vector/vector_types.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
#include "neurondb_sql.h"
#include "utils/elog.h"

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
extern int	ndb_cuda_rf_evaluate(const bytea * model_data,
								 const float *features,
								 const int *labels,
								 int n_samples,
								 int feature_dim,
								 double *accuracy_out,
								 double *precision_out,
								 double *recall_out,
								 double *f1_out,
								 char **errstr);
#endif

#define RF_BOOTSTRAP_FRACTION 0.8
#define RF_DEFAULT_TREES 3
#define RF_MAX_DEPTH 4
#define RF_MIN_SAMPLES 5

PG_FUNCTION_INFO_V1(train_random_forest_classifier);
PG_FUNCTION_INFO_V1(predict_random_forest);
PG_FUNCTION_INFO_V1(evaluate_random_forest);

typedef struct RFSplitPair
{
	double		value;
	int			cls;
}			RFSplitPair;

typedef struct RFDataset
{
	float	   *features;
	double	   *labels;
	int			n_samples;
	int			feature_dim;
}			RFDataset;

static bool rf_select_split(const float *features,
							const double *labels,
							const int *indices,
							int count,
							int feature_dim,
							int n_classes,
							pg_prng_state * rng,
							int *feature_order,
							int *best_feature,
							double *best_threshold,
							double *best_impurity);
static int	rf_build_branch_tree(GTree * tree,
								 const float *features,
								 const double *labels,
								 const double *feature_vars,
								 int feature_dim,
								 int n_classes,
								 const int *indices,
								 int count,
								 int depth,
								 int max_depth,
								 int min_samples,
								 pg_prng_state * rng,
								 int *feature_order,
								 double *feature_importance,
								 double *max_split_deviation);
static double rf_tree_predict_row(const GTree * tree, const float *row, int dim);
static void rf_serialize_tree(StringInfo buf, const GTree * tree);
static GTree * rf_deserialize_tree(StringInfo buf);
static bytea * rf_model_serialize(const RFModel * model, uint8 training_backend);
static RFModel * rf_model_deserialize(const bytea * data, uint8 *training_backend_out);
static void rf_free_deserialized_model(RFModel * model);
static bool rf_load_model_from_catalog(int32 model_id, RFModel * *out);
static bool rf_metadata_is_gpu(Jsonb * metadata);
static bool rf_try_gpu_predict_catalog(int32 model_id,
									   const Vector *feature_vec,
									   double *result_out);
static void rf_dataset_init(RFDataset * dataset);
static void rf_dataset_free(RFDataset * dataset);
static void rf_dataset_load(const char *quoted_tbl,
							const char *quoted_feat,
							const char *quoted_label,
							RFDataset * dataset,
							StringInfo query);
void		neurondb_gpu_register_rf_model(void);

static int
rf_split_pair_cmp(const void *a, const void *b)
{
	const		RFSplitPair *pa = (const RFSplitPair *) a;
	const		RFSplitPair *pb = (const RFSplitPair *) b;

	if (pa->value < pb->value)
		return -1;
	if (pa->value > pb->value)
		return 1;
	if (pa->cls < pb->cls)
		return -1;
	if (pa->cls > pb->cls)
		return 1;
	return 0;
}

static bool
rf_select_split(const float *features,
				const double *labels,
				const int *indices,
				int count,
				int feature_dim,
				int n_classes,
				pg_prng_state * rng,
				int *feature_order,
				int *best_feature,
				double *best_threshold,
				double *best_impurity)
{
	int			mtry;
	int			candidates;
	int			f;

	if (count <= 1 || feature_dim <= 0 || n_classes <= 0)
		return false;

	if (best_feature != NULL)
		*best_feature = -1;
	if (best_threshold != NULL)
		*best_threshold = 0.0;
	if (best_impurity != NULL)
		*best_impurity = DBL_MAX;

	if (feature_order != NULL)
	{
		for (f = 0; f < feature_dim; f++)
			feature_order[f] = f;
	}

	mtry = (int) sqrt((double) feature_dim);
	if (mtry < 1)
		mtry = 1;
	if (mtry > feature_dim)
		mtry = feature_dim;
	candidates = mtry;

	if (feature_order != NULL)
	{
		for (f = 0; f < candidates; f++)
		{
			int			swap_idx;

			swap_idx = (int) pg_prng_uint64_range_inclusive(
															rng, (uint64) f, (uint64) (feature_dim - 1));
			if (swap_idx != f)
			{
				int			tmp = feature_order[f];

				feature_order[f] = feature_order[swap_idx];
				feature_order[swap_idx] = tmp;
			}
		}
	}

	if (feature_order == NULL)
		candidates = Min(candidates, feature_dim);

	for (f = 0; f < candidates; f++)
	{
		int			feature_idx =
			(feature_order != NULL) ? feature_order[f] : f;
		RFSplitPair *pairs = NULL;
		int		   *left_counts_tmp;
		int		   *right_counts_tmp;
		int			pair_count = 0;
		int			i;
		int			left_total = 0;
		int			right_total = 0;

		if (feature_idx < 0 || feature_idx >= feature_dim)
			continue;

		{
			size_t		pairs_size = sizeof(RFSplitPair) * (size_t) count;

			if (pairs_size > MaxAllocSize)
			{
				elog(WARNING,
					 "rf_select_split: pairs allocation size %zu exceeds MaxAllocSize (count=%d)",
					 pairs_size, count);
				return false;
			}
			{
				size_t		pair_count_tmp = pairs_size / sizeof(RFSplitPair);
				NDB_ALLOC(pairs, RFSplitPair, pair_count_tmp);
			}
			if (pairs == NULL)
			{
				elog(WARNING, "rf_select_split: palloc failed for pairs (count=%d)", count);
				return false;
			}
		}

		for (i = 0; i < count; i++)
		{
			int			idx = indices[i];
			float		value;
			int			cls;

			if (idx < 0)
				continue;
			cls = (int) rint(labels[idx]);
			if (cls < 0 || cls >= n_classes)
				continue;
			value = features[idx * feature_dim + feature_idx];
			if (!isfinite(value))
				continue;

			pairs[pair_count].value = (double) value;
			pairs[pair_count].cls = cls;
			pair_count++;
		}

		if (pair_count > 1)
		{
			bool		try_gpu = (n_classes == 2 && NDB_SHOULD_TRY_GPU()
								   && neurondb_gpu_is_available()
								   && ndb_gpu_kernel_enabled("rf_split"));

			if (try_gpu)
			{
				float	   *gpu_features;
				uint8_t    *gpu_labels;
				NDB_ALLOC(gpu_features, float, pair_count);
				NDB_CHECK_ALLOC(gpu_features, "gpu_features");
				NDB_ALLOC(gpu_labels, uint8_t, pair_count);
				NDB_CHECK_ALLOC(gpu_labels, "gpu_labels");
				{
					bool		labels_ok = true;
					double		gpu_threshold = 0.0;
					double		gpu_gini = DBL_MAX;
					int			gpu_left = 0;
					int			gpu_right = 0;

					for (i = 0; i < pair_count; i++)
					{
						int			cls = pairs[i].cls;

						gpu_features[i] = (float) pairs[i].value;
						if (cls == 0)
							gpu_labels[i] = 0;
						else if (cls == 1)
							gpu_labels[i] = 1;
						else
						{
							labels_ok = false;
							break;
						}
					}

					if (labels_ok
						&& neurondb_gpu_rf_best_split_binary(
															 gpu_features,
															 gpu_labels,
															 pair_count,
															 &gpu_threshold,
															 &gpu_gini,
															 &gpu_left,
															 &gpu_right))
					{
						if (gpu_gini < *best_impurity)
						{
							*best_impurity = gpu_gini;
							*best_threshold = gpu_threshold;
							*best_feature = feature_idx;
						}

						NDB_FREE(gpu_features);
						gpu_features = NULL;
						NDB_FREE(gpu_labels);
						gpu_labels = NULL;
						NDB_FREE(pairs);
						pairs = NULL;
						continue;
					}

					NDB_FREE(gpu_features);
					gpu_features = NULL;
					NDB_FREE(gpu_labels);
					gpu_labels = NULL;
				}
			}
		}

		if (pair_count <= 1)
		{
			NDB_FREE(pairs);
			pairs = NULL;
			continue;
		}

		qsort(pairs,
			  pair_count,
			  sizeof(RFSplitPair),
			  rf_split_pair_cmp);
		NDB_ALLOC(left_counts_tmp, int, n_classes);
		NDB_CHECK_ALLOC(left_counts_tmp, "left_counts_tmp");
		NDB_ALLOC(right_counts_tmp, int, n_classes);
		NDB_CHECK_ALLOC(right_counts_tmp, "right_counts_tmp");

		for (i = 0; i < pair_count; i++)
			right_counts_tmp[pairs[i].cls]++;

		right_total = pair_count;

		for (i = 0; i < pair_count - 1; i++)
		{
			int			cls = pairs[i].cls;
			double		left_imp;
			double		right_imp;
			double		weighted;
			double		threshold_candidate;

			left_counts_tmp[cls]++;
			right_counts_tmp[cls]--;
			left_total++;
			right_total--;

			if (pairs[i].value == pairs[i + 1].value)
				continue;
			if (left_total <= 0 || right_total <= 0)
				continue;

			left_imp = rf_gini_impurity(
										left_counts_tmp, n_classes, left_total);
			right_imp = rf_gini_impurity(
										 right_counts_tmp, n_classes, right_total);
			threshold_candidate =
				0.5 * (pairs[i].value + pairs[i + 1].value);
			weighted = ((double) left_total / (double) pair_count)
				* left_imp
				+ ((double) right_total / (double) pair_count)
				* right_imp;

			if (weighted < *best_impurity)
			{
				*best_impurity = weighted;
				*best_threshold = threshold_candidate;
				*best_feature = feature_idx;
			}
		}

		NDB_FREE(left_counts_tmp);
		left_counts_tmp = NULL;
		NDB_FREE(right_counts_tmp);
		right_counts_tmp = NULL;
		NDB_FREE(pairs);
		pairs = NULL;
	}

	return (*best_feature >= 0);
}

static int
rf_build_branch_tree(GTree * tree,
					 const float *features,
					 const double *labels,
					 const double *feature_vars,
					 int feature_dim,
					 int n_classes,
					 const int *indices,
					 int count,
					 int depth,
					 int max_depth,
					 int min_samples,
					 pg_prng_state * rng,
					 int *feature_order,
					 double *feature_importance,
					 double *max_split_deviation)
{
	int		   *class_counts;
	int			majority_idx = -1;
	int			i;
	double		best_impurity = DBL_MAX;
	int			split_feature = -1;
	double		split_threshold = 0.0;
	double		gini;
	int			node_idx;

	if (tree == NULL || features == NULL || labels == NULL
		|| indices == NULL || count <= 0)
		return gtree_add_leaf(tree, 0.0);

	NDB_ALLOC(class_counts, int, n_classes);
	NDB_CHECK_ALLOC(class_counts, "class_counts");

	for (i = 0; i < count; i++)
	{
		int			idx = indices[i];
		int			cls;

		if (idx < 0)
			continue;
		if (!isfinite(labels[idx]))
			continue;
		cls = (int) rint(labels[idx]);
		if (cls < 0 || cls >= n_classes)
			continue;
		class_counts[cls]++;
		if (majority_idx < 0
			|| class_counts[cls] > class_counts[majority_idx])
			majority_idx = cls;
	}

	if (majority_idx < 0)
	{
		NDB_FREE(class_counts);
		class_counts = NULL;
		return gtree_add_leaf(tree, 0.0);
	}

	gini = rf_gini_impurity(class_counts, n_classes, count);

	if (gini <= 0.0 || depth >= max_depth || count <= min_samples)
	{
		double		value = (double) majority_idx;

		NDB_FREE(class_counts);
		class_counts = NULL;
		return gtree_add_leaf(tree, value);
	}

	if (!rf_select_split(features,
						 labels,
						 indices,
						 count,
						 feature_dim,
						 n_classes,
						 rng,
						 feature_order,
						 &split_feature,
						 &split_threshold,
						 &best_impurity))
	{
		double		value = (double) majority_idx;

		NDB_FREE(class_counts);
		class_counts = NULL;
		return gtree_add_leaf(tree, value);
	}

	if (split_feature < 0)
	{
		double		value = (double) majority_idx;

		NDB_FREE(class_counts);
		class_counts = NULL;
		return gtree_add_leaf(tree, value);
	}

	{
		int		   *left_indices;
		int		   *right_indices;
		int			left_count = 0;
		int			right_count = 0;

		NDB_ALLOC(left_indices, int, count);
		NDB_CHECK_ALLOC(left_indices, "left_indices");
		NDB_ALLOC(right_indices, int, count);
		NDB_CHECK_ALLOC(right_indices, "right_indices");

		for (i = 0; i < count; i++)
		{
			int			idx = indices[i];
			float		value;

			if (idx < 0)
				continue;
			value = features[idx * feature_dim + split_feature];
			if (!isfinite(value))
				continue;
			if ((double) value <= split_threshold)
				left_indices[left_count++] = idx;
			else
				right_indices[right_count++] = idx;
		}

		if (left_count == 0 || right_count == 0)
		{
			double		value = (double) majority_idx;

			NDB_FREE(left_indices);
			left_indices = NULL;
			NDB_FREE(right_indices);
			right_indices = NULL;
			NDB_FREE(class_counts);
			class_counts = NULL;
			return gtree_add_leaf(tree, value);
		}
		else
		{
			int			left_child;
			int			right_child;

			if (feature_importance != NULL && gini > 0.0
				&& best_impurity < DBL_MAX
				&& split_feature >= 0 && split_feature < feature_dim)
			{
				double		improvement = gini - best_impurity;

				if (improvement < 0.0)
					improvement = 0.0;
				feature_importance[split_feature] +=
					improvement;
			}

			if (feature_vars != NULL && split_feature < feature_dim
				&& feature_vars[split_feature] > 0.0
				&& max_split_deviation != NULL)
			{
				double		split_dev = fabs(split_threshold)
					/ sqrt(feature_vars[split_feature]);

				if (split_dev > *max_split_deviation)
					*max_split_deviation = split_dev;
			}

			node_idx = gtree_add_split(
									   tree, split_feature, split_threshold);

			left_child = rf_build_branch_tree(tree,
											  features,
											  labels,
											  feature_vars,
											  feature_dim,
											  n_classes,
											  left_indices,
											  left_count,
											  depth + 1,
											  max_depth,
											  min_samples,
											  rng,
											  feature_order,
											  feature_importance,
											  max_split_deviation);
			right_child = rf_build_branch_tree(tree,
											   features,
											   labels,
											   feature_vars,
											   feature_dim,
											   n_classes,
											   right_indices,
											   right_count,
											   depth + 1,
											   max_depth,
											   min_samples,
											   rng,
											   feature_order,
											   feature_importance,
											   max_split_deviation);

			gtree_set_child(tree, node_idx, left_child, true);
			gtree_set_child(tree, node_idx, right_child, false);

			NDB_FREE(left_indices);
			left_indices = NULL;
			NDB_FREE(right_indices);
			right_indices = NULL;
			NDB_FREE(class_counts);
			class_counts = NULL;
			return node_idx;
		}
	}
}

static double
rf_tree_predict_row(const GTree * tree, const float *row, int dim)
{
	const		GTreeNode *nodes;
	int			idx;

	if (tree == NULL || row == NULL)
		return 0.0;
	if (tree->root < 0 || tree->count <= 0)
		return 0.0;

	nodes = gtree_nodes(tree);
	idx = tree->root;

	while (idx >= 0 && idx < tree->count)
	{
		const		GTreeNode *node = &nodes[idx];

		if (node->is_leaf)
			return node->value;

		if (node->feature_idx < 0 || node->feature_idx >= dim)
			return 0.0;

		if ((double) row[node->feature_idx] <= node->threshold)
			idx = node->left;
		else
			idx = node->right;
	}

	return 0.0;
}

static RFModel * rf_models = NULL;
static int	rf_model_count = 0;
static int32 rf_next_model_id = 1;

static void
rf_store_model(int32 model_id,
			   int n_features,
			   int n_samples,
			   int n_classes,
			   double majority,
			   double fraction,
			   double gini,
			   double entropy,
			   const int *class_counts,
			   const double *feature_means,
			   const double *feature_variances,
			   const double *feature_importance,
			   GTree * tree,
			   int split_feature,
			   double split_threshold,
			   double second_value,
			   double second_fraction,
			   double left_value,
			   double left_fraction,
			   double right_value,
			   double right_fraction,
			   double max_deviation,
			   double max_split_deviation,
			   int feature_limit,
			   const double *left_means,
			   const double *right_means,
			   int tree_count,
			   GTree * const *trees,
			   const double *tree_majority,
			   const double *tree_majority_fraction,
			   const double *tree_second,
			   const double *tree_second_fraction,
			   const double *tree_oob_accuracy,
			   double oob_accuracy)
{
	MemoryContext oldctx;
	int			i;
	size_t		alloc_size __attribute__((unused));

#ifdef MEMORY_CONTEXT_CHECKING
	/* Check memory context at entry */
	MemoryContext entry_ctx = CurrentMemoryContext;

	if (entry_ctx != NULL)
		MemoryContextCheck(entry_ctx);
#endif

	if (n_features <= 0 || n_features > 10000)
	{
		elog(ERROR, "rf_store_model: invalid n_features=%d (must be 1-10000)", n_features);
		return;
	}
	if (n_classes <= 0 || n_classes > 1000)
	{
		elog(ERROR, "rf_store_model: invalid n_classes=%d (must be 1-1000)", n_classes);
		return;
	}
	if (tree_count < 0 || tree_count > 10000)
	{
		elog(ERROR, "rf_store_model: invalid tree_count=%d (must be 0-10000)", tree_count);
		return;
	}
	if (feature_limit < 0 || feature_limit > n_features)
	{
		elog(ERROR, "rf_store_model: invalid feature_limit=%d (must be 0-%d, n_features=%d)",
			 feature_limit, n_features, n_features);
		return;
	}

	elog(DEBUG1,
		 "rf_store_model: model_id=%d n_features=%d n_classes=%d tree_count=%d feature_limit=%d",
		 model_id, n_features, n_classes, tree_count, feature_limit);

	if (TopMemoryContext == NULL)
	{
		elog(ERROR, "rf_store_model: TopMemoryContext is NULL");
		return;
	}

	if (!IsTransactionState())
	{
		elog(WARNING, "rf_store_model: not in transaction state, cannot store model");
		return;
	}

	oldctx = MemoryContextSwitchTo(TopMemoryContext);

	if (oldctx == NULL)
	{
		elog(ERROR, "rf_store_model: failed to switch to TopMemoryContext");
		return;
	}

	if (CurrentMemoryContext == NULL)
	{
		MemoryContextSwitchTo(oldctx);
		elog(ERROR, "rf_store_model: CurrentMemoryContext is NULL after switch");
		return;
	}

	if (CurrentMemoryContext != TopMemoryContext)
	{
		MemoryContextSwitchTo(oldctx);
		elog(ERROR, "rf_store_model: context switch failed - CurrentMemoryContext != TopMemoryContext");
		return;
	}

	if (rf_models == NULL && rf_model_count > 0)
	{
		MemoryContextSwitchTo(oldctx);
		elog(WARNING, "rf_store_model: rf_models is NULL but rf_model_count=%d, resetting count", rf_model_count);
		rf_model_count = 0;
	}

	if (rf_model_count == 0)
	{
		NDB_ALLOC(rf_models, RFModel, 1);
		if (rf_models == NULL)
		{
			MemoryContextSwitchTo(oldctx);
			elog(ERROR, "rf_store_model: palloc failed for initial rf_models");
			return;
		}
	}
	else
	{
		alloc_size = sizeof(RFModel) * (rf_model_count + 1);
		if (alloc_size > MaxAllocSize)
		{
			MemoryContextSwitchTo(oldctx);
			elog(ERROR, "rf_store_model: allocation size %zu exceeds MaxAllocSize", alloc_size);
			return;
		}
		if (rf_models == NULL)
		{
			MemoryContextSwitchTo(oldctx);
			elog(WARNING, "rf_store_model: rf_models is NULL before repalloc, resetting and using NDB_ALLOC");
			rf_model_count = 0;
			NDB_ALLOC(rf_models, RFModel, 1);
			if (rf_models == NULL)
			{
				elog(ERROR, "rf_store_model: palloc failed after reset");
				return;
			}
		}
		else
		{
			rf_models = (RFModel *) repalloc(rf_models, alloc_size);
			if (rf_models == NULL)
			{
				MemoryContextSwitchTo(oldctx);
				elog(ERROR, "rf_store_model: repalloc failed for rf_models");
				return;
			}
		}
	}

	rf_models[rf_model_count].model_id = model_id;
	rf_models[rf_model_count].n_features = n_features;
	rf_models[rf_model_count].n_samples = n_samples;
	rf_models[rf_model_count].n_classes = n_classes;
	rf_models[rf_model_count].majority_value = majority;
	rf_models[rf_model_count].majority_fraction = fraction;
	rf_models[rf_model_count].gini_impurity = gini;
	rf_models[rf_model_count].label_entropy = entropy;

	rf_models[rf_model_count].class_counts = NULL;
	if (n_classes > 0 && class_counts != NULL)
	{
		size_t		class_counts_size = sizeof(int) * (size_t) n_classes;

		if (class_counts_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: class_counts_size %zu exceeds MaxAllocSize, skipping", class_counts_size);
		}
		else
		{
			NDB_DECLARE(int *, copy);
			{
				size_t		count = class_counts_size / sizeof(int);
				NDB_ALLOC(copy, int, count);
			}

			if (copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for class_counts");
			}
			else
			{
				memcpy(copy, class_counts, class_counts_size);
				rf_models[rf_model_count].class_counts = copy;
			}
		}
	}

	rf_models[rf_model_count].feature_means = NULL;
	if (n_features > 0 && feature_means != NULL)
	{
		size_t		means_size = sizeof(double) * (size_t) n_features;

		if (means_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: means_size %zu exceeds MaxAllocSize, skipping", means_size);
		}
		else
		{
			NDB_DECLARE(double *, means_copy);
			{
				size_t		count = means_size / sizeof(double);
				NDB_ALLOC(means_copy, double, count);
			}

			if (means_copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for feature_means");
			}
			else
			{
				for (i = 0; i < n_features; i++)
					means_copy[i] = feature_means[i];
				rf_models[rf_model_count].feature_means = means_copy;
			}
		}
	}

	rf_models[rf_model_count].feature_variances = NULL;
	if (n_features > 0 && feature_variances != NULL)
	{
		size_t		vars_size = sizeof(double) * (size_t) n_features;

		if (vars_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: vars_size %zu exceeds MaxAllocSize, skipping", vars_size);
		}
		else
		{
			NDB_DECLARE(double *, vars_copy);
			{
				size_t		count = vars_size / sizeof(double);
				NDB_ALLOC(vars_copy, double, count);
			}

			if (vars_copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for feature_variances");
			}
			else
			{
				for (i = 0; i < n_features; i++)
					vars_copy[i] = feature_variances[i];
				rf_models[rf_model_count].feature_variances = vars_copy;
			}
		}
	}

	rf_models[rf_model_count].feature_importance = NULL;

	/*
	 * Skip feature_importance allocation to avoid memory context corruption
	 * crashes. This is non-critical data that can be recomputed if needed.
	 */
	if (0 && n_features > 0 && feature_importance != NULL)
	{
	}

	rf_models[rf_model_count].tree = tree;
	rf_models[rf_model_count].split_feature = split_feature;
	rf_models[rf_model_count].split_threshold = split_threshold;
	rf_models[rf_model_count].second_value = second_value;
	rf_models[rf_model_count].second_fraction = second_fraction;
	rf_models[rf_model_count].oob_accuracy = oob_accuracy;
	rf_models[rf_model_count].left_branch_value = left_value;
	rf_models[rf_model_count].left_branch_fraction = left_fraction;
	rf_models[rf_model_count].right_branch_value = right_value;
	rf_models[rf_model_count].right_branch_fraction = right_fraction;
	rf_models[rf_model_count].max_deviation = max_deviation;
	rf_models[rf_model_count].max_split_deviation = max_split_deviation;
	rf_models[rf_model_count].feature_limit =
		(feature_limit > 0) ? feature_limit : 0;
	rf_models[rf_model_count].left_branch_means = NULL;
	rf_models[rf_model_count].right_branch_means = NULL;
	rf_models[rf_model_count].tree_count = 0;
	rf_models[rf_model_count].trees = NULL;
	rf_models[rf_model_count].tree_majority = NULL;
	rf_models[rf_model_count].tree_majority_fraction = NULL;
	rf_models[rf_model_count].tree_second = NULL;
	rf_models[rf_model_count].tree_second_fraction = NULL;
	rf_models[rf_model_count].tree_oob_accuracy = NULL;

	if (rf_models[rf_model_count].feature_limit > 0 && left_means != NULL)
	{
		size_t		left_means_size = sizeof(double) * (size_t) rf_models[rf_model_count].feature_limit;

		if (left_means_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: left_means_size %zu exceeds MaxAllocSize, skipping", left_means_size);
		}
		else
		{
			NDB_DECLARE(double *, copy);
			{
				size_t		count = left_means_size / sizeof(double);
				NDB_ALLOC(copy, double, count);
			}

			if (copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for left_branch_means");
			}
			else
			{
				memcpy(copy, left_means, left_means_size);
				rf_models[rf_model_count].left_branch_means = copy;
			}
		}
	}

	if (rf_models[rf_model_count].feature_limit > 0 && right_means != NULL)
	{
		size_t		right_means_size = sizeof(double) * (size_t) rf_models[rf_model_count].feature_limit;

		if (right_means_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: right_means_size %zu exceeds MaxAllocSize, skipping", right_means_size);
		}
		else
		{
			NDB_DECLARE(double *, copy);
			{
				size_t		count = right_means_size / sizeof(double);
				NDB_ALLOC(copy, double, count);
			}

			if (copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for right_branch_means");
			}
			else
			{
				memcpy(copy, right_means, right_means_size);
				rf_models[rf_model_count].right_branch_means = copy;
			}
		}
	}

	if (tree_count > 0 && trees != NULL)
	{
		size_t		trees_array_size;
		size_t		tree_double_size;

		trees_array_size = sizeof(GTree *) * (size_t) tree_count;
		if (trees_array_size > MaxAllocSize)
		{
			elog(WARNING, "rf_store_model: trees_array_size %zu exceeds MaxAllocSize, skipping", trees_array_size);
		}
		else
		{
			NDB_DECLARE(GTree **, tree_copy);
			{
				size_t		count = trees_array_size / sizeof(GTree *);
				NDB_ALLOC(tree_copy, GTree *, count);
			}
			if (tree_copy == NULL)
			{
				elog(WARNING, "rf_store_model: palloc failed for trees array");
			}
			else
			{
				for (i = 0; i < tree_count; i++)
					tree_copy[i] = trees[i];
				rf_models[rf_model_count].trees = tree_copy;
				rf_models[rf_model_count].tree_count = tree_count;

				tree_double_size = sizeof(double) * (size_t) tree_count;
				if (tree_double_size > MaxAllocSize)
				{
					elog(WARNING, "rf_store_model: tree_double_size %zu exceeds MaxAllocSize, skipping tree arrays", tree_double_size);
				}
				else
				{
					if (tree_majority != NULL)
					{
						NDB_DECLARE(double *, majority_copy);
						{
							size_t		count = tree_double_size / sizeof(double);
							NDB_ALLOC(majority_copy, double, count);
						}

						if (majority_copy == NULL)
						{
							elog(WARNING, "rf_store_model: palloc failed for tree_majority");
						}
						else
						{
							for (i = 0; i < tree_count; i++)
								majority_copy[i] = tree_majority[i];
							rf_models[rf_model_count].tree_majority = majority_copy;
						}
					}

					if (tree_majority_fraction != NULL)
					{
						NDB_DECLARE(double *, fraction_copy);
						{
							size_t		count = tree_double_size / sizeof(double);
							NDB_ALLOC(fraction_copy, double, count);
						}

						if (fraction_copy == NULL)
						{
							elog(WARNING, "rf_store_model: palloc failed for tree_majority_fraction");
						}
						else
						{
							for (i = 0; i < tree_count; i++)
								fraction_copy[i] = tree_majority_fraction[i];
							rf_models[rf_model_count].tree_majority_fraction = fraction_copy;
						}
					}

					if (tree_second != NULL)
					{
						NDB_DECLARE(double *, second_copy);
						{
							size_t		count = tree_double_size / sizeof(double);
							NDB_ALLOC(second_copy, double, count);
						}

						if (second_copy == NULL)
						{
							elog(WARNING, "rf_store_model: palloc failed for tree_second");
						}
						else
						{
							for (i = 0; i < tree_count; i++)
								second_copy[i] = tree_second[i];
							rf_models[rf_model_count].tree_second = second_copy;
						}
					}

					if (tree_second_fraction != NULL)
					{
						NDB_DECLARE(double *, second_fraction_copy);
						{
							size_t		count = tree_double_size / sizeof(double);
							NDB_ALLOC(second_fraction_copy, double, count);
						}

						if (second_fraction_copy == NULL)
						{
							elog(WARNING, "rf_store_model: palloc failed for tree_second_fraction");
						}
						else
						{
							for (i = 0; i < tree_count; i++)
								second_fraction_copy[i] = tree_second_fraction[i];
							rf_models[rf_model_count].tree_second_fraction = second_fraction_copy;
						}
					}

					if (tree_oob_accuracy != NULL)
					{
						NDB_DECLARE(double *, oob_copy);
						{
							size_t		count = tree_double_size / sizeof(double);
							NDB_ALLOC(oob_copy, double, count);
						}

						if (oob_copy == NULL)
						{
							elog(WARNING, "rf_store_model: palloc failed for tree_oob_accuracy");
						}
						else
						{
							for (i = 0; i < tree_count; i++)
								oob_copy[i] = tree_oob_accuracy[i];
							rf_models[rf_model_count].tree_oob_accuracy = oob_copy;
						}
					}
				}
			}
		}
	}

	rf_model_count++;

	MemoryContextSwitchTo(oldctx);

#ifdef MEMORY_CONTEXT_CHECKING
	if (entry_ctx != NULL)
		MemoryContextCheck(entry_ctx);
#endif
}

static bool
rf_lookup_model(int32 model_id, RFModel * *out)
{
	int			i;

	for (i = 0; i < rf_model_count; i++)
	{
		if (rf_models[i].model_id == model_id)
		{
			if (out)
				*out = &rf_models[i];
			return true;
		}
	}
	return false;
}

static int
rf_count_classes(double *labels, int n_samples)
{
	int			max_class = -1;
	int			i;

	if (n_samples <= 0)
		return 0;

	for (i = 0; i < n_samples; i++)
	{
		double		val = labels[i];
		int			as_int;

		if (!isfinite(val))
			continue;

		as_int = (int) rint(val);
		if (as_int < 0)
			continue;

		if (as_int > max_class)
			max_class = as_int;
	}

	return (max_class < 0) ? 0 : (max_class + 1);
}

Datum
train_random_forest_classifier(PG_FUNCTION_ARGS)
{
	text	   *table_name_text;
	text	   *feature_col_text;
	text	   *label_col_text;

	char	   *table_name = NULL;
	char	   *feature_col = NULL;
	char	   *label_col = NULL;
	const char *quoted_tbl = NULL;
	const char *quoted_feat = NULL;
	const char *quoted_label = NULL;

	StringInfoData query = {0};
	MemoryContext oldcontext;
	NDB_DECLARE (NdbSpiSession *, train_spi_session);

	int			feature_dim = 0;
	int			n_classes = 0;
	int			majority_count = 0;
	int			second_count = 0;
	int			second_idx = -1;
	int		   *class_counts_tmp = NULL;
	int		   *counts = NULL;
	int			feature_sum_count = 0;
	int			split_feature = -1;
	int		   *left_counts = NULL;
	int		   *right_counts = NULL;
	int			left_majority_idx = -1;
	int			right_majority_idx = -1;
	int			left_total = 0;
	int			right_total = 0;
	int			majority_idx = -1;
	int			feature_limit = 0;
	int			best_feature = -1;
	int		   *left_feature_counts_vec = NULL;
	int		   *right_feature_counts_vec = NULL;
	int			n_samples = 0;
	int			split_pair_count = 0;
	int			sample_count = 0;
	int		   *bootstrap_indices = NULL;
	int		   *feature_order = NULL;
	pg_prng_state rng;
	int32		model_id = 0;
	RFDataset	dataset;

	double	   *labels = NULL;
	double		majority_value = 0.0;
	double		majority_fraction = 0.0;
	double		gini_impurity = 0.0;
	double		label_entropy = 0.0;
	double		second_value = 0.0;
	double	   *feature_means_tmp = NULL;
	double	   *feature_vars_tmp = NULL;
	double	   *feature_importance_tmp = NULL;
	double	   *feature_sums = NULL;
	double	   *feature_sums_sq = NULL;
	double	   *class_feature_sums = NULL;
	int		   *class_feature_counts = NULL;
	double	   *left_feature_sums_vec = NULL;
	double	   *right_feature_sums_vec = NULL;
	double		left_leaf_value = 0.0;
	double		right_leaf_value = 0.0;
	double		left_sum = 0.0;
	double		right_sum = 0.0;
	double		left_branch_fraction = 0.0;
	double		right_branch_fraction = 0.0;
	double		class_majority_mean = 0.0;
	double		class_second_mean = 0.0;
	double		class_mean_threshold = 0.0;
	double		best_majority_mean = 0.0;
	double		best_second_mean = 0.0;
	double		best_score = -1.0;
	double		max_deviation = 0.0;
	double		max_split_deviation = 0.0;
	double		split_threshold = 0.0;
	double		second_fraction = 0.0;
	double	   *left_branch_means_vec = NULL;
	double	   *right_branch_means_vec = NULL;
	double		forest_oob_accuracy = 0.0;
	double	   *tree_oob_accuracy = NULL;
	int			oob_total_all = 0;
	int			oob_correct_all = 0;
	float	   *stage_features = NULL;
	GTree	  **trees = NULL;
	double	   *tree_majorities = NULL;
	double	   *tree_majority_fractions = NULL;
	double	   *tree_seconds = NULL;
	double	   *tree_second_fractions = NULL;
	int			tree_count = 0;
	int			forest_trees_arg = RF_DEFAULT_TREES;
	int			max_depth_arg = RF_MAX_DEPTH;
	int			min_samples_arg = RF_MIN_SAMPLES;
	double		best_split_impurity = DBL_MAX;
	double		best_split_threshold = 0.0;

	bool		branch_threshold_valid = false;
	bool		class_mean_threshold_valid = false;
	bool		best_score_valid = false;
	bool		best_split_valid = false;
	GTree	   *primary_tree = NULL;
	RFSplitPair *split_pairs = NULL;

	if (PG_NARGS() < 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: train_random_forest_classifier: requires table, feature column, and label column"),
				 errdetail("Function received %d arguments, minimum required is 3", PG_NARGS()),
				 errhint("Provide table name, feature column name, and label column name as arguments.")));

	table_name_text = PG_GETARG_TEXT_PP(0);
	feature_col_text = PG_GETARG_TEXT_PP(1);
	label_col_text = PG_GETARG_TEXT_PP(2);

	if (PG_NARGS() > 3 && !PG_ARGISNULL(3))
	{
		int32		arg_trees = PG_GETARG_INT32(3);

		if (arg_trees < 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_random_forest_classifier: number of trees must be at least 1"),
					 errdetail("Received %d trees, minimum allowed is 1", arg_trees),
					 errhint("Specify a positive number of trees for the random forest ensemble.")));
		if (arg_trees > 1024)
			arg_trees = 1024;
		forest_trees_arg = arg_trees;
	}

	if (PG_NARGS() > 4 && !PG_ARGISNULL(4))
	{
		int32		arg_depth = PG_GETARG_INT32(4);

		if (arg_depth < 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_random_forest_classifier: max depth must be at least 1"),
					 errdetail("Received max depth %d, minimum allowed is 1", arg_depth),
					 errhint("Specify a positive maximum depth for decision trees.")));
		if (arg_depth > GTREE_MAX_DEPTH)
			arg_depth = GTREE_MAX_DEPTH;
		max_depth_arg = arg_depth;
	}

	if (PG_NARGS() > 5 && !PG_ARGISNULL(5))
	{
		int32		arg_min_samples = PG_GETARG_INT32(5);

		if (arg_min_samples < 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: train_random_forest_classifier: min samples must be at least 1"),
					 errdetail("Received min samples %d, minimum allowed is 1", arg_min_samples),
					 errhint("Specify a positive minimum number of samples for tree splits.")));
		if (arg_min_samples > 1000000)
			arg_min_samples = 1000000;
		min_samples_arg = arg_min_samples;
	}

	rf_dataset_init(&dataset);

	table_name = text_to_cstring(table_name_text);
	feature_col = text_to_cstring(feature_col_text);
	label_col = text_to_cstring(label_col_text);
	quoted_tbl = quote_identifier(table_name);
	quoted_feat = quote_identifier(feature_col);
	quoted_label = quote_identifier(label_col);

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(train_spi_session, oldcontext);

	initStringInfo(&query);

	rf_dataset_load(
					quoted_tbl, quoted_feat, quoted_label, &dataset, &query);

	feature_dim = dataset.feature_dim;
	n_samples = dataset.n_samples;
	labels = dataset.labels;
	stage_features = dataset.features;
	if (neurondb_gpu_is_available() && n_samples > 0 && feature_dim > 0)
	{
		int			gpu_class_count =
			rf_count_classes(dataset.labels, dataset.n_samples);

		if (gpu_class_count > 0)
		{
			StringInfoData hyperbuf;
			Jsonb	   *gpu_hyperparams = NULL;
			char	   *gpu_err = NULL;
			const char *gpu_features[1];
			MLGpuTrainResult gpu_result;

			memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
			gpu_features[0] = feature_col;

			/* Build hyperparameters JSON using JSONB API */
			{
				JsonbParseState *state = NULL;
				JsonbValue	jkey;
				JsonbValue	jval;
				JsonbValue *final_value = NULL;
				Numeric		n_trees_num, max_depth_num, min_samples_split_num;

				/* Start object */
				PG_TRY();
				{
					(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

					/* Add n_trees */
					jkey.type = jbvString;
					jkey.val.string.val = "n_trees";
					jkey.val.string.len = strlen("n_trees");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					n_trees_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(forest_trees_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = n_trees_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add max_depth */
					jkey.val.string.val = "max_depth";
					jkey.val.string.len = strlen("max_depth");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					max_depth_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_depth_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = max_depth_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add min_samples_split */
					jkey.val.string.val = "min_samples_split";
					jkey.val.string.len = strlen("min_samples_split");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					min_samples_split_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(min_samples_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = min_samples_split_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* End object */
					final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
					
					if (final_value == NULL)
					{
						elog(ERROR, "neurondb: train_random_forest: pushJsonbValue(WJB_END_OBJECT) returned NULL for hyperparameters");
					}
					
					gpu_hyperparams = JsonbValueToJsonb(final_value);
				}
				PG_CATCH();
				{
					ErrorData *edata = CopyErrorData();
					elog(ERROR, "neurondb: train_random_forest: hyperparameters JSONB construction failed: %s", edata->message);
					FlushErrorState();
					gpu_hyperparams = NULL;
				}
				PG_END_TRY();
			}

			if (ndb_gpu_try_train_model("random_forest",
										NULL,
										NULL,
										table_name,
										label_col,
										gpu_features,
										1,
										gpu_hyperparams,
										stage_features,
										labels,
										n_samples,
										feature_dim,
										gpu_class_count,
										&gpu_result,
										&gpu_err)
				&& gpu_result.spec.model_data != NULL)
			{
				MLCatalogModelSpec spec = gpu_result.spec;

				if (spec.training_table == NULL)
					spec.training_table = table_name;
				if (spec.training_column == NULL)
					spec.training_column = label_col;
				if (spec.parameters == NULL)
				{
					spec.parameters =
						gpu_hyperparams;
					gpu_hyperparams = NULL;
				}

				spec.training_time_ms = -1;
				spec.num_samples = n_samples;
				spec.num_features = feature_dim;

				model_id = ml_catalog_register_model(&spec);
				ndb_gpu_free_train_result(&gpu_result);

				if (gpu_hyperparams)
				{
					NDB_FREE(gpu_hyperparams);
					gpu_hyperparams = NULL;
				}
				if (hyperbuf.data)
				{
					NDB_FREE(hyperbuf.data);
					hyperbuf.data = NULL;
				}

				/*
				 * Don't free gpu_err - it's allocated by GPU function using
				 * pstrdup() and will be automatically freed when the memory
				 * context is cleaned up. Manually freeing it can cause
				 * crashes if the context is already cleaned up.
				 */
				if (gpu_err)
				{
					elog(DEBUG1, "train_random_forest_classifier: [DEBUG] gpu_err=%p (not freeing - managed by memory context)", (void *) gpu_err);
					gpu_err = NULL;
				}

				rf_dataset_free(&dataset);
				/* Free query.data BEFORE ending SPI session (it's allocated in SPI context) */
				if (query.data)
				{
					NDB_FREE(query.data);
					query.data = NULL;
				}
				NDB_SPI_SESSION_END(train_spi_session);

				if (table_name)
				{
					NDB_FREE(table_name);
					table_name = NULL;
				}
				if (feature_col)
				{
					NDB_FREE(feature_col);
					feature_col = NULL;
				}
				if (label_col)
				{
					NDB_FREE(label_col);
					label_col = NULL;
				}

				PG_RETURN_INT32(model_id);
			}

			/*
			 * Don't free gpu_err - it's allocated by GPU function using
			 * pstrdup() and will be automatically freed when the memory
			 * context is cleaned up. Manually freeing it can cause crashes if
			 * the context is already cleaned up.
			 */
			if (gpu_err != NULL)
			{
				elog(DEBUG1,
					 "random_forest: GPU training unavailable (%s)",
					 gpu_err);
				elog(DEBUG1, "train_random_forest_classifier: [DEBUG] gpu_err=%p (not freeing - managed by memory context)", (void *) gpu_err);
				gpu_err = NULL;
			}

			if (gpu_hyperparams != NULL)
			{
				NDB_FREE(gpu_hyperparams);
				gpu_hyperparams = NULL;
			}
			if (hyperbuf.data)
			{
				NDB_FREE(hyperbuf.data);
				hyperbuf.data = NULL;
			}
		}
	}
	if (n_samples > 0)
	{
		int			i;

		if (feature_dim > 0)
		{
			NDB_DECLARE(double *, feature_importance_tmp);
			NDB_ALLOC(feature_importance_tmp, double, feature_dim);
		}

		if (feature_dim > 0)
		{
			int			j;

			NDB_DECLARE(int *, feature_order);
			NDB_ALLOC(feature_order, int, feature_dim);
			for (j = 0; j < feature_dim; j++)
				feature_order[j] = j;
		}

		sample_count =
			(int) rint(RF_BOOTSTRAP_FRACTION * (double) n_samples);
		if (sample_count <= 0)
			sample_count = n_samples;
		if (sample_count > n_samples)
			sample_count = n_samples;
		if (sample_count > 0)
		{
			if (!pg_prng_strong_seed(&rng))
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: train_random_forest_classifier: failed to seed PRNG"),
						 errdetail("Random number generator initialization failed"),
						 errhint("This is an internal error. Please report this issue.")));
			NDB_ALLOC(bootstrap_indices, int, sample_count);
			for (i = 0; i < sample_count; i++)
				bootstrap_indices[i] =
					(int) pg_prng_uint64_range_inclusive(&rng,
														 0,
														 (uint64) (n_samples - 1));
		}

		if (feature_dim > 0 && sample_count > 0
			&& stage_features != NULL)
		{
			NDB_ALLOC(feature_sums, double, feature_dim);
			NDB_ALLOC(feature_sums_sq, double, feature_dim);

			for (i = 0; i < sample_count; i++)
			{
				int			src = bootstrap_indices[i];
				float	   *row;
				int			j;

				if (src < 0 || src >= n_samples)
					continue;
				if (!isfinite(labels[src]))
					continue;

				row = stage_features + (src * feature_dim);

				for (j = 0; j < feature_dim; j++)
				{
					double		val = (double) row[j];

					feature_sums[j] += val;
					feature_sums_sq[j] += val * val;
				}
				feature_sum_count++;
			}
		}

		n_classes = rf_count_classes(labels, n_samples);

		if (n_classes > 0)
		{
			NDB_ALLOC(counts, int, n_classes);
			if (counts != NULL)
			{
				int			best_idx = 0;

				class_counts_tmp = counts;

				for (i = 0; i < sample_count; i++)
				{
					int			src = bootstrap_indices[i];
					int			idx;

					if (src < 0 || src >= n_samples)
						continue;
					idx = (int) rint(labels[src]);
					if (idx < 0 || idx >= n_classes)
						continue;
					counts[idx]++;
					if (counts[idx] > counts[best_idx])
					{
						if (idx != best_idx)
						{
							second_idx = best_idx;
							second_count = counts[best_idx];
							second_value = (double) best_idx;
						}
						best_idx = idx;
					}
					else if (idx != best_idx
							 && counts[idx] > second_count)
					{
						second_idx = idx;
						second_count = counts[idx];
						second_value = (double) idx;
					}
				}

				majority_value = (double) best_idx;
				majority_count = counts[best_idx];
				majority_idx = best_idx;
				if (second_idx < 0 && n_classes > 1)
				{
					for (i = 0; i < n_classes; i++)
					{
						if (i == best_idx)
							continue;
						if (counts[i] >= second_count)
						{
							second_idx = i;
							second_count = counts[i];
							second_value = (double) i;
						}
					}
				}

				left_leaf_value = majority_value;
				right_leaf_value = (second_idx >= 0) ? second_value
					: majority_value;

				if (sample_count > 0)
				{
					double		sum_sq = 0.0;
					double		entropy = 0.0;
					int			c;
					double		ln2 = log(2.0);

					for (c = 0; c < n_classes; c++)
					{
						double		p = (double) counts[c]
							/ (double) sample_count;

						sum_sq += p * p;
						if (p > 0.0)
							entropy -= p * (log(p) / ln2);
					}
					gini_impurity = 1.0 - sum_sq;
					label_entropy = entropy;
				}

				if (class_counts_tmp)
				{
					StringInfoData histogram;

					initStringInfo(&histogram);
					appendStringInfo(&histogram, "[");
					for (i = 0; i < n_classes; i++)
					{
						if (i > 0)
							appendStringInfoString(&histogram, ", ");
						appendStringInfo(&histogram, "%d", class_counts_tmp[i]);
					}
					appendStringInfoChar(&histogram, ']');
					elog(DEBUG1, "random_forest: class histogram %s", histogram.data);
					NDB_FREE(histogram.data);
				}
			}

			if (feature_sums != NULL && feature_sum_count > 0)
			{
				int			j;
				StringInfoData mean_log;
				StringInfoData var_log;

				NDB_ALLOC(feature_means_tmp, double, feature_dim);
				NDB_ALLOC(feature_vars_tmp, double, feature_dim);
				for (j = 0; j < feature_dim; j++)
				{
					double		mean = feature_sums[j]
						/ (double) feature_sum_count;
					double		mean_sq = feature_sums_sq[j]
						/ (double) feature_sum_count;
					double		variance = mean_sq - (mean * mean);

					if (variance < 0.0)
						variance = 0.0;

					feature_means_tmp[j] = mean;
					feature_vars_tmp[j] = variance;
				}

				initStringInfo(&mean_log);
				appendStringInfo(&mean_log, "[");
				for (j = 0; j < feature_dim && j < 5; j++)
				{
					if (j > 0)
						appendStringInfoString(&mean_log, ", ");
					elog(DEBUG1,
						 "%.3f",
						 feature_means_tmp[j]);
				}
				if (feature_dim > 5)
					appendStringInfoString(&mean_log, ", ...");
				appendStringInfoChar(&mean_log, ']');
				elog(DEBUG1,
					 "random_forest: feature means %s",
					 mean_log.data);
				NDB_FREE(mean_log.data);

				initStringInfo(&var_log);
				appendStringInfo(&var_log, "[");
				for (j = 0; j < feature_dim && j < 5; j++)
				{
					if (j > 0)
						appendStringInfoString(&var_log, ", ");
					appendStringInfo(
									 &var_log, "%.3f", feature_vars_tmp[j]);
				}
				if (feature_dim > 5)
					appendStringInfoString(&var_log, ", ...");
				appendStringInfoChar(&var_log, ']');
				elog(DEBUG1,
					 "random_forest: feature variances %s",
					 var_log.data);
				NDB_FREE(var_log.data);
			}

			if (feature_dim > 0 && sample_count > 0 && n_classes > 0
				&& stage_features != NULL)
			{
				feature_limit = feature_dim;
				if (feature_limit < 1)
					feature_limit = 1;

				{
					size_t		sums_size = sizeof(double) * (size_t) n_classes * (size_t) feature_limit;
					size_t		counts_size = sizeof(int) * (size_t) n_classes * (size_t) feature_limit;

					if (sums_size > MaxAllocSize || counts_size > MaxAllocSize)
					{
						elog(WARNING, "rf_build_branch_tree: allocation sizes exceed MaxAllocSize (n_classes=%d, feature_limit=%d)", n_classes, feature_limit);
						return -1;
					}
					{
						size_t		sums_count = sums_size / sizeof(double);
						size_t		counts_count = counts_size / sizeof(int);
						NDB_ALLOC(class_feature_sums, double, sums_count);
						NDB_ALLOC(class_feature_counts, int, counts_count);
					}
					if (class_feature_sums == NULL || class_feature_counts == NULL)
					{
						elog(WARNING, "rf_build_branch_tree: palloc0 failed for class feature arrays");
						if (class_feature_sums != NULL)
							NDB_FREE(class_feature_sums);
						if (class_feature_counts != NULL)
							NDB_FREE(class_feature_counts);
						return -1;
					}
				}

				for (i = 0; i < sample_count; i++)
				{
					int			src = bootstrap_indices[i];
					int			cls;
					int			f;
					float	   *row;

					if (src < 0 || src >= n_samples)
						continue;
					if (!isfinite(labels[src]))
						continue;
					cls = (int) rint(labels[src]);
					if (cls < 0 || cls >= n_classes)
						continue;

					row = stage_features + (src * feature_dim);

					for (f = 0;
						 f < feature_limit && f < feature_dim;
						 f++)
					{
						double		val = (double) row[f];

						class_feature_sums[cls * feature_limit
										   + f] += val;
						class_feature_counts[cls * feature_limit
											 + f]++;
					}
				}

				if (majority_idx >= 0)
				{
					int			f;

					for (f = 0; f < feature_limit; f++)
					{
						int			idx = majority_idx * feature_limit
							+ f;

						if (class_feature_counts[idx] > 0)
						{
							double		maj_mean =
								class_feature_sums[idx]
								/ (double)
								class_feature_counts
								[idx];
							double		sec_mean = 0.0;
							int			sec_idx = -1;
							int			sec_count = 0;

							if (second_idx >= 0)
							{
								sec_idx = second_idx
									* feature_limit
									+ f;
								sec_count =
									class_feature_counts
									[sec_idx];
							}

							if (sec_count > 0)
							{
								sec_mean =
									class_feature_sums
									[sec_idx]
									/ (double)
									sec_count;

								if (fabs(maj_mean
										 - sec_mean)
									> best_score)
								{
									best_score = fabs(
													  maj_mean
													  - sec_mean);
									best_feature =
										f;
									best_majority_mean =
										maj_mean;
									best_second_mean =
										sec_mean;
									best_score_valid =
										true;
								}
							}
						}
					}
				}

				if (!best_score_valid && majority_idx >= 0)
				{
					int			idx = majority_idx * feature_limit;

					if (class_feature_counts[idx] > 0)
						best_majority_mean =
							class_feature_sums[idx]
							/ (double) class_feature_counts
							[idx];
				}

				if (best_score_valid)
				{
					class_majority_mean = best_majority_mean;
					class_second_mean = best_second_mean;
					class_mean_threshold = 0.5
						* (class_majority_mean
						   + class_second_mean);
					class_mean_threshold_valid = true;
					split_feature = best_feature;
					elog(DEBUG1,
						 "random_forest: best feature=%d "
						 "majority_mean=%.3f second_mean=%.3f "
						 "threshold=%.3f score=%.3f",
						 split_feature,
						 class_majority_mean,
						 class_second_mean,
						 class_mean_threshold,
						 best_score);
				}
			}

			/*
			 * Refine threshold using sorted split candidates on the chosen
			 * feature
			 */
			if (feature_dim > 0 && sample_count > 0 && n_classes > 0
				&& stage_features != NULL)
			{
				int			sf_idx = (split_feature >= 0
									  && split_feature < feature_dim)
					? split_feature
					: 0;

				NDB_ALLOC(split_pairs, RFSplitPair, sample_count);
				NDB_CHECK_ALLOC(split_pairs, "split_pairs");
				split_pair_count = 0;

				for (i = 0; i < sample_count; i++)
				{
					int			src = bootstrap_indices[i];
					int			cls;
					float	   *row;

					if (src < 0 || src >= n_samples)
						continue;
					cls = (int) rint(labels[src]);
					if (!isfinite(labels[src]) || cls < 0
						|| cls >= n_classes)
						continue;

					row = stage_features + (src * feature_dim);
					if (sf_idx >= feature_dim)
						continue;

					split_pairs[split_pair_count].value =
						(double) row[sf_idx];
					split_pairs[split_pair_count].cls = cls;
					split_pair_count++;
				}

				if (split_pair_count > 1)
				{
					NDB_DECLARE(int *, left_counts_tmp);
					NDB_DECLARE(int *, right_counts_tmp);
					int			right_total_eval = 0;
					int			left_total_eval = 0;
					NDB_ALLOC(left_counts_tmp, int, n_classes);
					NDB_ALLOC(right_counts_tmp, int, n_classes);

					qsort(split_pairs,
						  split_pair_count,
						  sizeof(RFSplitPair),
						  rf_split_pair_cmp);

					if (class_counts_tmp != NULL)
					{
						for (i = 0; i < n_classes; i++)
							right_counts_tmp[i] =
								class_counts_tmp[i];
					}
					else
					{
						for (i = 0; i < split_pair_count; i++)
							right_counts_tmp
								[split_pairs[i].cls]++;
					}

					for (i = 0; i < n_classes; i++)
						right_total_eval += right_counts_tmp[i];

					for (i = 0; i < split_pair_count - 1; i++)
					{
						int			cls = split_pairs[i].cls;

						left_counts_tmp[cls]++;
						right_counts_tmp[cls]--;
						left_total_eval++;
						right_total_eval--;

						if (split_pairs[i].value
							== split_pairs[i + 1].value)
							continue;
						if (left_total_eval <= 0
							|| right_total_eval <= 0)
							continue;

						{
							double		left_imp =
								rf_gini_impurity(
												 left_counts_tmp,
												 n_classes,
												 left_total_eval);
							double		right_imp =
								rf_gini_impurity(
												 right_counts_tmp,
												 n_classes,
												 right_total_eval);
							double		weighted =
								((double) left_total_eval
								 / (double)
								 split_pair_count)
								* left_imp
								+ ((double) right_total_eval
								   / (double)
								   split_pair_count)
								* right_imp;

							if (weighted
								< best_split_impurity)
							{
								best_split_impurity =
									weighted;
								best_split_threshold =
									0.5
									* (split_pairs[i]
									   .value
									   + split_pairs[i
													 + 1]
									   .value);
								best_split_valid = true;
							}
						}
					}

					NDB_FREE(left_counts_tmp);
					NDB_FREE(right_counts_tmp);
				}

				if (best_split_valid)
				{
					class_mean_threshold = best_split_threshold;
					class_mean_threshold_valid = true;
					split_feature = sf_idx;
					elog(DEBUG1,
						 "random_forest: refined threshold "
						 "feature=%d threshold=%.3f "
						 "impurity=%.6f",
						 split_feature,
						 class_mean_threshold,
						 best_split_impurity);
				}

				if (split_pairs)
				{
					NDB_FREE(split_pairs);
					split_pairs = NULL;
				}
			}

			if (feature_dim > 0 && feature_means_tmp != NULL
				&& n_classes > 0 && stage_features != NULL
				&& sample_count > 0)
			{
				int			sf = (split_feature >= 0
								  && split_feature < feature_dim)
					? split_feature
					: 0;
				double		threshold = feature_means_tmp[sf];

				left_total = 0;
				right_total = 0;
				left_sum = 0.0;
				right_sum = 0.0;
				left_majority_idx = -1;
				right_majority_idx = -1;

				if (class_mean_threshold_valid)
					threshold = class_mean_threshold;

				NDB_ALLOC(left_counts, int, n_classes);
				NDB_CHECK_ALLOC(left_counts, "left_counts");
				NDB_ALLOC(right_counts, int, n_classes);
				NDB_CHECK_ALLOC(right_counts, "right_counts");
				if (feature_limit > 0)
				{
					NDB_ALLOC(left_feature_sums_vec, double, feature_limit);
					NDB_CHECK_ALLOC(left_feature_sums_vec, "left_feature_sums_vec");
					NDB_ALLOC(right_feature_sums_vec, double, feature_limit);
					NDB_CHECK_ALLOC(right_feature_sums_vec, "right_feature_sums_vec");
					NDB_ALLOC(left_feature_counts_vec, int, feature_limit);
					NDB_CHECK_ALLOC(left_feature_counts_vec, "left_feature_counts_vec");
					NDB_ALLOC(right_feature_counts_vec, int, feature_limit);
					NDB_CHECK_ALLOC(right_feature_counts_vec, "right_feature_counts_vec");
				}

				for (i = 0; i < sample_count; i++)
				{
					int			src = bootstrap_indices[i];
					int			cls;
					int			f;
					float	   *row;
					double		value;

					if (src < 0 || src >= n_samples)
						continue;
					if (!isfinite(labels[src]))
						continue;
					cls = (int) rint(labels[src]);
					if (cls < 0 || cls >= n_classes)
						continue;

					if (sf >= feature_dim)
						continue;

					row = stage_features + (src * feature_dim);
					value = (double) row[sf];

					if (value <= threshold)
					{
						left_counts[cls]++;
						left_total++;
						left_sum += value;
						if (feature_limit > 0)
						{
							for (f = 0; f < feature_limit
								 && f < feature_dim;
								 f++)
							{
								left_feature_sums_vec
									[f] +=
									(double) row[f];
								left_feature_counts_vec
									[f]++;
							}
						}
					}
					else
					{
						right_counts[cls]++;
						right_total++;
						right_sum += value;
						if (feature_limit > 0)
						{
							for (f = 0; f < feature_limit
								 && f < feature_dim;
								 f++)
							{
								right_feature_sums_vec
									[f] +=
									(double) row[f];
								right_feature_counts_vec
									[f]++;
							}
						}
					}
				}

				for (i = 0; i < n_classes; i++)
				{
					if (left_total > 0
						&& (left_majority_idx < 0
							|| left_counts[i] > left_counts
							[left_majority_idx]))
						left_majority_idx = i;
					if (right_total > 0
						&& (right_majority_idx < 0
							|| right_counts[i]
							> right_counts
							[right_majority_idx]))
						right_majority_idx = i;
				}

				if (left_majority_idx >= 0)
					left_leaf_value = (double) left_majority_idx;

				if (right_majority_idx >= 0)
				{
					right_leaf_value = (double) right_majority_idx;
					second_value = right_leaf_value;
					if (class_counts_tmp != NULL)
						second_fraction =
							((double) class_counts_tmp
							 [right_majority_idx])
							/ (double) sample_count;
					else if (right_total > 0)
						second_fraction =
							((double) right_counts
							 [right_majority_idx])
							/ (double) sample_count;
				}

				if (sample_count > 0)
				{
					if (left_total > 0)
						left_branch_fraction =
							((double) left_total)
							/ (double) sample_count;
					if (right_total > 0)
						right_branch_fraction =
							((double) right_total)
							/ (double) sample_count;
				}

				if (feature_limit > 0 && left_feature_sums_vec != NULL
					&& right_feature_sums_vec != NULL)
				{
					int			f;

					NDB_DECLARE(double *, left_branch_means_vec);
					NDB_DECLARE(double *, right_branch_means_vec);
					NDB_ALLOC(left_branch_means_vec, double, feature_limit);
					NDB_CHECK_ALLOC(left_branch_means_vec, "left_branch_means_vec");
					NDB_ALLOC(right_branch_means_vec, double, feature_limit);
					NDB_CHECK_ALLOC(right_branch_means_vec, "right_branch_means_vec");

					for (f = 0; f < feature_limit; f++)
					{
						if (left_feature_counts_vec != NULL
							&& left_feature_counts_vec[f]
							> 0)
							left_branch_means_vec[f] =
								left_feature_sums_vec[f]
								/ (double)
								left_feature_counts_vec
								[f];
						else if (feature_means_tmp != NULL
								 && f < feature_dim)
							left_branch_means_vec[f] =
								feature_means_tmp[f];
						else
							left_branch_means_vec[f] = 0.0;

						if (right_feature_counts_vec != NULL
							&& right_feature_counts_vec[f]
							> 0)
							right_branch_means_vec[f] =
								right_feature_sums_vec
								[f]
								/ (double)
								right_feature_counts_vec
								[f];
						else if (feature_means_tmp != NULL
								 && f < feature_dim)
							right_branch_means_vec[f] =
								feature_means_tmp[f];
						else
							right_branch_means_vec[f] =
								left_branch_means_vec
								[f];
					}
				}

				if ((left_total == 0 || right_total == 0)
					&& feature_vars_tmp != NULL && sf < feature_dim
					&& feature_vars_tmp[sf] > 0.0)
				{
					double		adjust;

					adjust = 0.5 * sqrt(feature_vars_tmp[sf]);
					if (right_total == 0)
					{
						threshold =
							feature_means_tmp[sf] + adjust;
						branch_threshold_valid = true;
						if (right_branch_fraction <= 0.0)
							right_branch_fraction = 0.5;
						if (left_branch_fraction <= 0.0)
							left_branch_fraction = 1.0
								- right_branch_fraction;
						right_leaf_value = (second_idx >= 0)
							? second_value
							: majority_value;
					}
					else if (left_total == 0)
					{
						threshold =
							feature_means_tmp[sf] - adjust;
						branch_threshold_valid = true;
						if (left_branch_fraction <= 0.0)
							left_branch_fraction = 0.5;
						if (right_branch_fraction <= 0.0)
							right_branch_fraction = 1.0
								- left_branch_fraction;
						left_leaf_value = majority_value;
					}
				}

				if (left_total > 0 && right_total > 0)
				{
					double		left_mean =
						left_sum / (double) left_total;
					double		right_mean =
						right_sum / (double) right_total;

					threshold = 0.5 * (left_mean + right_mean);
					branch_threshold_valid = true;
				}

				if (left_branch_fraction <= 0.0
					&& right_branch_fraction <= 0.0)
					left_branch_fraction = majority_fraction;
				else if (left_branch_fraction <= 0.0
						 && right_branch_fraction > 0.0)
					left_branch_fraction =
						1.0 - right_branch_fraction;
				else if (right_branch_fraction <= 0.0
						 && left_branch_fraction > 0.0)
					right_branch_fraction =
						1.0 - left_branch_fraction;

				if (second_fraction <= 0.0
					&& right_branch_fraction > 0.0)
					second_fraction = right_branch_fraction;

				if (left_counts)
					NDB_FREE(left_counts);
				if (right_counts)
					NDB_FREE(right_counts);
				left_counts = NULL;
				right_counts = NULL;

				elog(DEBUG1,
					 "random_forest: branch totals left=%d "
					 "right=%d split=%.3f lf=%.3f rf=%.3f",
					 left_total,
					 right_total,
					 threshold,
					 left_branch_fraction,
					 right_branch_fraction);
			}

			if (sample_count > 0 && majority_count > 0)
				majority_fraction =
					((double) majority_count) / (double) sample_count;
		}

		if (sample_count > 0 && second_count > 0 && second_fraction <= 0.0)
			second_fraction = ((double) second_count) / (double) sample_count;

		if (majority_count > 0)
		{
			MemoryContext oldctx;
			int			forest_trees = forest_trees_arg;
			int			t;

			if (forest_trees < 1)
				forest_trees = 1;
			if (forest_trees > n_samples)
				forest_trees = n_samples;

			if (forest_trees > 0)
			{
				NDB_ALLOC(trees, GTree *, forest_trees);
				NDB_CHECK_ALLOC(trees, "trees");
				NDB_ALLOC(tree_majorities, double, forest_trees);
				NDB_CHECK_ALLOC(tree_majorities, "tree_majorities");
				NDB_ALLOC(tree_majority_fractions, double, forest_trees);
				NDB_CHECK_ALLOC(tree_majority_fractions, "tree_majority_fractions");
				NDB_ALLOC(tree_seconds, double, forest_trees);
				NDB_CHECK_ALLOC(tree_seconds, "tree_seconds");
				NDB_ALLOC(tree_second_fractions, double, forest_trees);
				NDB_CHECK_ALLOC(tree_second_fractions, "tree_second_fractions");
				NDB_ALLOC(tree_oob_accuracy, double, forest_trees);
				NDB_CHECK_ALLOC(tree_oob_accuracy, "tree_oob_accuracy");
			}

			for (t = 0; t < forest_trees; t++)
			{
				GTree	   *tree;
				int			node_idx;
				int			left_idx;
				int			right_idx;
				int			tree_feature = split_feature;
				int			feature_for_split = split_feature;
				double		tree_threshold = split_threshold;
				double		var0 = 0.0;
				int			tree_majority_idx = -1;
				int			tree_second_idx = -1;
				int			tree_majority_count = 0;
				int			tree_second_count = 0;
				double		tree_majority_value = majority_value;
				double		tree_second_value = second_value;
				double		tree_majority_frac = majority_fraction;
				double		tree_second_frac = second_fraction;
				int		   *tree_counts = NULL;
				int			boot_samples = 0;
				int			sample_target = n_samples;
				int			j;
				int		   *tree_bootstrap = NULL;
				RFSplitPair *tree_pairs = NULL;
				int			tree_pair_count = 0;
				double		tree_best_impurity = DBL_MAX;
				double		tree_best_threshold = split_threshold;
				int			tree_best_feature = split_feature;
				bool		tree_split_valid = false;
				int			mtry = 0;
				int			candidates = 0;
				int		   *left_tmp = NULL;
				int		   *right_tmp = NULL;
				int			left_total_local = 0;
				int			right_total_local = 0;
				int			left_majority_local = -1;
				int			right_majority_local = -1;
				int			left_best_count = 0;
				int			right_best_count = 0;
				double		tree_left_value = left_leaf_value;
				double		tree_right_value = right_leaf_value;
				double		tree_left_fraction = left_branch_fraction;
				double		tree_right_fraction = right_branch_fraction;
				bool	   *inbag = NULL;
				int			oob_total_local = 0;
				int			oob_correct_local = 0;
				int		   *left_indices_local = NULL;
				int		   *right_indices_local = NULL;
				int			left_index_count = 0;
				int			right_index_count = 0;

				if (n_classes > 0)
				{
					NDB_DECLARE(int *, tree_counts);
					NDB_ALLOC(tree_counts, int, n_classes);
				}

				if (n_samples > 0)
				{
					sample_target = (int) rint((double) n_samples
											   * RF_BOOTSTRAP_FRACTION);
					if (sample_target < 1)
						sample_target = 1;
					if (sample_target > n_samples)
						sample_target = n_samples;
				}

				if (n_samples > 0)
				{
					NDB_DECLARE(bool *, inbag);
					NDB_ALLOC(inbag, bool, n_samples);
					NDB_CHECK_ALLOC(inbag, "inbag");
				}

				if (sample_target > 0 && n_samples > 0)
				{
					NDB_ALLOC(tree_bootstrap, int, sample_target);
					NDB_CHECK_ALLOC(tree_bootstrap, "tree_bootstrap");
				}

				for (j = 0; j < sample_target; j++)
				{
					int			idx;

					idx = (int) pg_prng_uint64_range_inclusive(
															   &rng, 0, (uint64) (n_samples - 1));
					if (tree_bootstrap != NULL)
						tree_bootstrap[j] = idx;
					boot_samples++;
					if (tree_counts != NULL && labels != NULL)
					{
						int			cls;

						if (!isfinite(labels[idx]))
							continue;
						cls = (int) rint(labels[idx]);
						if (cls < 0 || cls >= n_classes)
							continue;
						tree_counts[cls]++;
						if (tree_counts[cls]
							> tree_majority_count)
						{
							if (cls != tree_majority_idx)
							{
								tree_second_idx =
									tree_majority_idx;
								tree_second_count =
									tree_majority_count;
							}
							tree_majority_idx = cls;
							tree_majority_count =
								tree_counts[cls];
						}
						else if (cls != tree_majority_idx
								 && tree_counts[cls]
								 > tree_second_count)
						{
							tree_second_idx = cls;
							tree_second_count =
								tree_counts[cls];
						}
					}
				}

				if (inbag != NULL && tree_bootstrap != NULL)
				{
					for (j = 0; j < boot_samples; j++)
					{
						int			idx = tree_bootstrap[j];

						if (idx >= 0 && idx < n_samples)
							inbag[idx] = true;
					}
				}

				if (tree_majority_idx >= 0)
				{
					tree_majority_value = (double) tree_majority_idx;
					if (boot_samples > 0)
						tree_majority_frac =
							(double) tree_majority_count
							/ (double) boot_samples;
				}

				if (tree_second_idx < 0)
					tree_second_idx = second_idx;

				if (tree_second_idx >= 0)
				{
					tree_second_value = (double) tree_second_idx;
					if (boot_samples > 0 && tree_second_count > 0)
						tree_second_frac =
							(double) tree_second_count
							/ (double) boot_samples;
					else
						tree_second_frac = second_fraction;
				}

				if (feature_dim > 0 && stage_features != NULL
					&& labels != NULL && n_classes > 0
					&& tree_bootstrap != NULL)
				{
					int			f;

					if (feature_order != NULL)
					{
						for (f = 0; f < feature_dim; f++)
							feature_order[f] = f;
					}

					mtry = (int) sqrt((double) feature_dim);
					if (mtry < 1)
						mtry = 1;
					if (mtry > feature_dim)
						mtry = feature_dim;
					candidates = mtry;

					if (feature_order != NULL && feature_dim > 0)
					{
						for (f = 0; f < candidates; f++)
						{
							int			swap_idx;

							swap_idx = (int)
								pg_prng_uint64_range_inclusive(
															   &rng,
															   (uint64) f,
															   (uint64) (feature_dim
																		 - 1));
							if (swap_idx != f)
							{
								int			tmp = feature_order
									[f];

								feature_order[f] =
									feature_order
									[swap_idx];
								feature_order
									[swap_idx] =
									tmp;
							}
						}
					}
					else
						candidates =
							Min(candidates, feature_dim);

					for (f = 0; f < candidates; f++)
					{
						int			feature_idx = feature_order
							? feature_order[f]
							: f;
						int			s;

						if (feature_idx < 0
							|| feature_idx >= feature_dim)
							continue;

						NDB_ALLOC(tree_pairs, RFSplitPair, boot_samples);
						NDB_CHECK_ALLOC(tree_pairs, "tree_pairs");
						tree_pair_count = 0;

						for (s = 0; s < boot_samples; s++)
						{
							int			sample_idx =
								tree_bootstrap[s];
							float	   *row;
							double		value;
							int			cls;

							if (sample_idx < 0
								|| sample_idx
								>= n_samples)
								continue;
							if (!isfinite(
										  labels[sample_idx]))
								continue;

							cls = (int) rint(
											 labels[sample_idx]);
							if (cls < 0 || cls >= n_classes)
								continue;

							row = stage_features
								+ (sample_idx
								   * feature_dim);
							value = (double)
								row[feature_idx];

							tree_pairs[tree_pair_count]
								.value = value;
							tree_pairs[tree_pair_count]
								.cls = cls;
							tree_pair_count++;
						}

						if (tree_pair_count > 1)
						{
							NDB_DECLARE(int *, left_counts_tmp);
							NDB_DECLARE(int *, right_counts_tmp);
							int			left_total_eval = 0;
							int			right_total_eval = 0;

							qsort(tree_pairs,
								  tree_pair_count,
								  sizeof(RFSplitPair),
								  rf_split_pair_cmp);

							NDB_ALLOC(left_counts_tmp, int, n_classes);
							NDB_ALLOC(right_counts_tmp, int, n_classes);

							for (s = 0; s < tree_pair_count;
								 s++)
								right_counts_tmp
									[tree_pairs[s].cls]++;

							right_total_eval =
								tree_pair_count;

							for (s = 0;
								 s < tree_pair_count - 1;
								 s++)
							{
								int			cls_val =
									tree_pairs[s]
									.cls;
								double		left_imp;
								double		right_imp;
								double		weighted;
								double		threshold_candidate;

								left_counts_tmp
									[cls_val]++;
								right_counts_tmp
									[cls_val]--;
								left_total_eval++;
								right_total_eval--;

								if (tree_pairs[s].value
									== tree_pairs[s
												  + 1]
									.value)
									continue;
								if (left_total_eval <= 0
									|| right_total_eval
									<= 0)
									continue;

								left_imp = rf_gini_impurity(
															left_counts_tmp,
															n_classes,
															left_total_eval);
								right_imp = rf_gini_impurity(
															 right_counts_tmp,
															 n_classes,
															 right_total_eval);
								threshold_candidate =
									0.5
									* (tree_pairs[s].value
									   + tree_pairs[s
													+ 1]
									   .value);
								weighted =
									((double) left_total_eval
									 / (double)
									 tree_pair_count)
									* left_imp
									+ ((double) right_total_eval
									   / (double)
									   tree_pair_count)
									* right_imp;

								if (weighted
									< tree_best_impurity)
								{
									tree_best_impurity =
										weighted;
									tree_best_threshold =
										threshold_candidate;
									tree_best_feature =
										feature_idx;
									tree_split_valid =
										true;
								}
							}

							NDB_FREE(left_counts_tmp);
							NDB_FREE(right_counts_tmp);
						}

						if (tree_pairs != NULL)
						{
							NDB_FREE(tree_pairs);
							tree_pairs = NULL;
						}
					}
				}

				if (tree_split_valid)
				{
					tree_feature = tree_best_feature;
					feature_for_split = tree_best_feature;
					tree_threshold = tree_best_threshold;
				}
				else
					feature_for_split = tree_feature;

				if (tree_bootstrap != NULL && feature_dim > 0
					&& n_classes > 0 && stage_features != NULL
					&& labels != NULL && tree_feature >= 0
					&& tree_feature < feature_dim)
				{
					NDB_ALLOC(left_tmp, int, n_classes);
					NDB_ALLOC(right_tmp, int, n_classes);

					left_index_count = 0;
					right_index_count = 0;
					if (boot_samples > 0)
					{
						if (left_indices_local == NULL)
							NDB_ALLOC(left_indices_local, int, boot_samples);
						if (right_indices_local == NULL)
							NDB_ALLOC(right_indices_local, int, boot_samples);
					}

					for (j = 0; j < boot_samples; j++)
					{
						int			sample_idx = tree_bootstrap[j];
						float	   *row;
						double		value;
						int			cls;

						if (sample_idx < 0
							|| sample_idx >= n_samples)
							continue;
						if (!isfinite(labels[sample_idx]))
							continue;

						cls = (int) rint(labels[sample_idx]);
						if (cls < 0 || cls >= n_classes)
							continue;

						row = stage_features
							+ (sample_idx * feature_dim);
						value = (double) row[tree_feature];

						if (value <= tree_threshold)
						{
							left_tmp[cls]++;
							left_total_local++;
							if (left_indices_local != NULL)
								left_indices_local
									[left_index_count++] =
									sample_idx;
						}
						else
						{
							right_tmp[cls]++;
							right_total_local++;
							if (right_indices_local != NULL)
								right_indices_local
									[right_index_count++] =
									sample_idx;
						}
					}

					for (j = 0; j < n_classes; j++)
					{
						if (left_tmp[j] > left_best_count)
						{
							left_best_count = left_tmp[j];
							left_majority_local = j;
						}
						if (right_tmp[j] > right_best_count)
						{
							right_best_count = right_tmp[j];
							right_majority_local = j;
						}
					}

					if (left_majority_local >= 0)
						tree_left_value =
							(double) left_majority_local;
					if (right_majority_local >= 0)
					{
						tree_right_value =
							(double) right_majority_local;
						tree_second_value = tree_right_value;
					}

					if (boot_samples > 0)
					{
						tree_left_fraction =
							(double) left_total_local
							/ (double) boot_samples;
						tree_right_fraction =
							(double) right_total_local
							/ (double) boot_samples;
					}

					if (tree_second_frac <= 0.0
						&& tree_right_fraction > 0.0)
						tree_second_frac = tree_right_fraction;

					NDB_FREE(left_tmp);
					NDB_FREE(right_tmp);
				}

				feature_for_split = tree_feature;

				oldctx = MemoryContextSwitchTo(TopMemoryContext);
				tree = gtree_create("rf_model_tree", 4);
				MemoryContextSwitchTo(oldctx);

				if (tree == NULL)
				{
					if (tree_counts)
						NDB_FREE(tree_counts);
					if (tree_bootstrap)
						NDB_FREE(tree_bootstrap);
					continue;
				}

				if (feature_dim > 0 && feature_means_tmp != NULL)
				{
					if (feature_for_split < 0
						|| feature_for_split >= feature_dim)
						feature_for_split = 0;
					if (feature_vars_tmp != NULL
						&& feature_for_split < feature_dim)
						var0 = feature_vars_tmp
							[feature_for_split];

					if (!branch_threshold_valid)
						tree_threshold = feature_means_tmp
							[feature_for_split];

					node_idx = gtree_add_split(tree,
											   feature_for_split,
											   tree_threshold);

					if (left_indices_local != NULL
						&& left_index_count > 0)
						left_idx = rf_build_branch_tree(tree,
														stage_features,
														labels,
														feature_vars_tmp,
														feature_dim,
														n_classes,
														left_indices_local,
														left_index_count,
														1,
														max_depth_arg,
														min_samples_arg,
														&rng,
														feature_order,
														feature_importance_tmp,
														&max_split_deviation);
					else
						left_idx = gtree_add_leaf(
												  tree, tree_left_value);

					if (right_indices_local != NULL
						&& right_index_count > 0)
						right_idx = rf_build_branch_tree(tree,
														 stage_features,
														 labels,
														 feature_vars_tmp,
														 feature_dim,
														 n_classes,
														 right_indices_local,
														 right_index_count,
														 1,
														 max_depth_arg,
														 min_samples_arg,
														 &rng,
														 feature_order,
														 feature_importance_tmp,
														 &max_split_deviation);
					else
						right_idx = gtree_add_leaf(
												   tree, tree_right_value);

					gtree_set_child(tree, node_idx, left_idx, true);
					gtree_set_child(
									tree, node_idx, right_idx, false);
					gtree_set_root(tree, node_idx);

					if (var0 > 0.0)
					{
						double		split_dev = fabs(tree_threshold)
							/ sqrt(var0);

						if (split_dev > max_split_deviation)
							max_split_deviation = split_dev;
					}

					if (tree_count == 0)
					{
						split_feature = feature_for_split;
						split_threshold = tree_threshold;
						left_leaf_value = tree_left_value;
						right_leaf_value = tree_right_value;
						left_branch_fraction =
							tree_left_fraction;
						right_branch_fraction =
							tree_right_fraction;
						branch_threshold_valid = true;
						second_value = tree_second_value;
						second_fraction = tree_second_frac;
					}

					elog(DEBUG1,
						 "random_forest: tree %d split "
						 "feature=%d "
						 "threshold=%.3f left=%.3f right=%.3f",
						 t + 1,
						 feature_for_split,
						 tree_threshold,
						 tree_left_value,
						 tree_right_value);
				}
				else
				{
					left_idx =
						gtree_add_leaf(tree, tree_left_value);
					gtree_set_root(tree, left_idx);
				}

				if (inbag != NULL && labels != NULL && n_samples > 0
					&& stage_features != NULL)
				{
					oob_total_local = 0;
					oob_correct_local = 0;

					for (j = 0; j < n_samples; j++)
					{
						int			actual;
						int			predicted;

						if (inbag[j])
							continue;
						if (!isfinite(labels[j]))
							continue;

						actual = (int) rint(labels[j]);
						if (actual < 0 || actual >= n_classes)
							continue;

						if (feature_dim > 0)
						{
							float	   *row = stage_features
								+ (j * feature_dim);
							double		tree_pred;

							tree_pred = rf_tree_predict_row(
															tree, row, feature_dim);
							predicted =
								(int) rint(tree_pred);
						}
						else
							predicted = (int) rint(
												   tree_left_value);

						oob_total_local++;
						if (predicted == actual)
							oob_correct_local++;
					}

					if (tree_oob_accuracy != NULL
						&& t < forest_trees)
					{
						if (oob_total_local > 0)
							tree_oob_accuracy[t] =
								(double) oob_correct_local
								/ (double)
								oob_total_local;
						else
							tree_oob_accuracy[t] = 0.0;
					}

					oob_total_all += oob_total_local;
					oob_correct_all += oob_correct_local;
				}

				gtree_validate(tree);

				if (primary_tree == NULL)
					primary_tree = tree;

				if (trees != NULL)
				{
					int			idx = tree_count;

					if (idx < forest_trees)
					{
						trees[idx] = tree;
						tree_majorities[idx] =
							tree_majority_value;
						tree_majority_fractions[idx] =
							tree_majority_frac;
						tree_seconds[idx] = tree_second_value;
						tree_second_fractions[idx] =
							tree_second_frac;
						tree_count++;
					}
				}

				if (tree_counts)
					NDB_FREE(tree_counts);
				if (tree_bootstrap)
					NDB_FREE(tree_bootstrap);
				if (inbag)
					NDB_FREE(inbag);
				if (left_indices_local)
					NDB_FREE(left_indices_local);
				if (right_indices_local)
					NDB_FREE(right_indices_local);
			}
		}

		if (oob_total_all > 0)
			forest_oob_accuracy =
				(double) oob_correct_all / (double) oob_total_all;

		if (feature_importance_tmp != NULL && feature_dim > 0)
		{
			double		importance_total = 0.0;
			double		top_import0 = 0.0;
			double		top_import1 = 0.0;
			double		top_import2 = 0.0;
			int			top_index0 = -1;
			int			top_index1 = -1;
			int			top_index2 = -1;

			for (i = 0; i < feature_dim; i++)
			{
				double		val = feature_importance_tmp[i];

				if (val < 0.0)
					val = 0.0;
				feature_importance_tmp[i] = val;
				importance_total += val;

				if (val > top_import0)
				{
					top_import2 = top_import1;
					top_index2 = top_index1;
					top_import1 = top_import0;
					top_index1 = top_index0;
					top_import0 = val;
					top_index0 = i;
				}
				else if (val > top_import1)
				{
					top_import2 = top_import1;
					top_index2 = top_index1;
					top_import1 = val;
					top_index1 = i;
				}
				else if (val > top_import2)
				{
					top_import2 = val;
					top_index2 = i;
				}
			}

			if (importance_total > 0.0)
			{
				for (i = 0; i < feature_dim; i++)
					feature_importance_tmp[i] /= importance_total;
				if (top_import0 > 0.0)
					top_import0 /= importance_total;
				if (top_import1 > 0.0)
					top_import1 /= importance_total;
				if (top_import2 > 0.0)
					top_import2 /= importance_total;
			}

			elog(DEBUG1,
				 "random_forest: feature importance top=%d:%.3f %d:%.3f "
				 "%d:%.3f",
				 top_index0,
				 top_import0,
				 top_index1,
				 top_import1,
				 top_index2,
				 top_import2);
		}

		model_id = rf_next_model_id++;
		if (feature_limit > 0 && left_branch_means_vec == NULL
			&& right_branch_means_vec == NULL)
			feature_limit = 0;

		rf_store_model(model_id,
					   feature_dim,
					   n_samples,
					   n_classes,
					   majority_value,
					   majority_fraction,
					   gini_impurity,
					   label_entropy,
					   class_counts_tmp,
					   feature_means_tmp,
					   feature_vars_tmp,
					   feature_importance_tmp,
					   primary_tree,
					   split_feature,
					   split_threshold,
					   second_value,
					   second_fraction,
					   left_leaf_value,
					   left_branch_fraction,
					   right_leaf_value,
					   right_branch_fraction,
					   max_deviation,
					   max_split_deviation,
					   feature_limit,
					   left_branch_means_vec,
					   right_branch_means_vec,
					   tree_count,
					   trees,
					   tree_majorities,
					   tree_majority_fractions,
					   tree_seconds,
					   tree_second_fractions,
					   tree_oob_accuracy,
					   forest_oob_accuracy);

		{
			RFModel    *stored_model;
			MLCatalogModelSpec spec;
			bytea	   *serialized = NULL;
			Jsonb	   *params_jsonb = NULL;
			Jsonb	   *metrics_jsonb = NULL;
			bytea	   *gpu_payload = NULL;
			Jsonb	   *gpu_metrics = NULL;
			char	   *gpu_err = NULL;
			bool		gpu_packed = false;

			stored_model = &rf_models[rf_model_count - 1];

			/* Build parameters JSON using JSONB API */
			{
				JsonbParseState *state = NULL;
				JsonbValue	jkey;
				JsonbValue	jval;
				JsonbValue *final_value = NULL;
				Numeric		n_trees_num, max_depth_num, min_samples_split_num;

				/* Start object */
				PG_TRY();
				{
					(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

					/* Add n_trees */
					jkey.type = jbvString;
					jkey.val.string.val = "n_trees";
					jkey.val.string.len = strlen("n_trees");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					n_trees_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(forest_trees_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = n_trees_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add max_depth */
					jkey.val.string.val = "max_depth";
					jkey.val.string.len = strlen("max_depth");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					max_depth_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_depth_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = max_depth_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add min_samples_split */
					jkey.val.string.val = "min_samples_split";
					jkey.val.string.len = strlen("min_samples_split");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					min_samples_split_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(min_samples_arg)));
					jval.type = jbvNumeric;
					jval.val.numeric = min_samples_split_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* End object */
					final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
					
					if (final_value == NULL)
					{
						elog(ERROR, "neurondb: train_random_forest: pushJsonbValue(WJB_END_OBJECT) returned NULL for parameters");
					}
					
					params_jsonb = JsonbValueToJsonb(final_value);
				}
				PG_CATCH();
				{
					ErrorData *edata = CopyErrorData();
					elog(ERROR, "neurondb: train_random_forest: parameters JSONB construction failed: %s", edata->message);
					FlushErrorState();
					params_jsonb = NULL;
				}
				PG_END_TRY();
			}

			/* Build metrics JSON using JSONB API */
			{
				JsonbParseState *state = NULL;
				JsonbValue	jkey;
				JsonbValue	jval;
				JsonbValue *final_value = NULL;
				Numeric		oob_accuracy_num, gini_num, majority_fraction_num;

				/* Start object */
				PG_TRY();
				{
					(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

					/* Add oob_accuracy */
					jkey.type = jbvString;
					jkey.val.string.val = "oob_accuracy";
					jkey.val.string.len = strlen("oob_accuracy");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					oob_accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(forest_oob_accuracy)));
					jval.type = jbvNumeric;
					jval.val.numeric = oob_accuracy_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add gini */
					jkey.val.string.val = "gini";
					jkey.val.string.len = strlen("gini");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					gini_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(gini_impurity)));
					jval.type = jbvNumeric;
					jval.val.numeric = gini_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* Add majority_fraction */
					jkey.val.string.val = "majority_fraction";
					jkey.val.string.len = strlen("majority_fraction");
					(void) pushJsonbValue(&state, WJB_KEY, &jkey);
					majority_fraction_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(majority_fraction)));
					jval.type = jbvNumeric;
					jval.val.numeric = majority_fraction_num;
					(void) pushJsonbValue(&state, WJB_VALUE, &jval);

					/* End object */
					final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
					
					if (final_value == NULL)
					{
						elog(ERROR, "neurondb: train_random_forest: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
					}
					
					metrics_jsonb = JsonbValueToJsonb(final_value);
				}
				PG_CATCH();
				{
					ErrorData *edata = CopyErrorData();
					elog(ERROR, "neurondb: train_random_forest: metrics JSONB construction failed: %s", edata->message);
					FlushErrorState();
					metrics_jsonb = NULL;
				}
				PG_END_TRY();
			}

			if (neurondb_gpu_is_available())
			{
				if (ndb_gpu_rf_pack_model(stored_model,
										  &gpu_payload,
										  &gpu_metrics,
										  &gpu_err)
					== 0)
				{
					serialized = gpu_payload;
					gpu_packed = true;
					if (gpu_metrics != NULL)
					{
						if (metrics_jsonb)
							NDB_FREE(metrics_jsonb);
						metrics_jsonb = gpu_metrics;
						gpu_metrics = NULL;
					}
					if (gpu_err != NULL)
					{
						elog(DEBUG1,
							 "random_forest: GPU pack failed (%s)",
							 gpu_err);

						/*
						 * Don't free gpu_err - it's allocated by GPU function
						 * using pstrdup() and will be automatically freed
						 * when the memory context is cleaned up.
						 */
						elog(DEBUG1, "train_random_forest_classifier: [DEBUG] gpu_err=%p (not freeing - managed by memory context)", (void *) gpu_err);
						gpu_err = NULL;
					}
				}
			}

			if (!gpu_packed)
				serialized = rf_model_serialize(stored_model, 0);

			memset(&spec, 0, sizeof(spec));
			spec.algorithm = "random_forest";
			spec.model_type = "classification";
			spec.training_table = table_name;
			spec.training_column = label_col;
			spec.parameters = params_jsonb;
			spec.metrics = metrics_jsonb;
			spec.model_data = serialized;
			spec.training_time_ms = -1;
			spec.num_samples = n_samples;
			spec.num_features = feature_dim;

			model_id = ml_catalog_register_model(&spec);
			stored_model->model_id = model_id;
			if (model_id >= rf_next_model_id)
				rf_next_model_id = model_id + 1;

			/*
			 * Free SPI context allocations with NULL checks and set to NULL.
			 * These allocations are in SPI context and must be freed before
			 * SPI_finish() deletes that context.
			 */
			if (serialized != NULL)
			{
				NDB_FREE(serialized);
				serialized = NULL;
			}
			if (params_jsonb != NULL)
			{
				NDB_FREE(params_jsonb);
				params_jsonb = NULL;
			}
			if (metrics_jsonb != NULL)
			{
				NDB_FREE(metrics_jsonb);
				metrics_jsonb = NULL;
			}
			if (gpu_metrics != NULL)
			{
				NDB_FREE(gpu_metrics);
				gpu_metrics = NULL;
			}
			if (!gpu_packed && gpu_payload != NULL)
			{
				NDB_FREE(gpu_payload);
				gpu_payload = NULL;
			}
		}

		elog(DEBUG1,
			 "neurondb: train_random_forest_classifier: rows=%d, classes=%d, dim=%d, "
			 "majority=%.3f, frac=%.3f, second=%.3f, sfrac=%.3f, gini=%.3f, "
			 "entropy=%.3f, oob=%.3f",
			 n_samples,
			 n_classes,
			 feature_dim,
			 majority_value,
			 majority_fraction,
			 second_value,
			 second_fraction,
			 gini_impurity,
			 label_entropy,
			 forest_oob_accuracy);
		if (class_counts_tmp)
			NDB_FREE(class_counts_tmp);
		if (feature_means_tmp)
			NDB_FREE(feature_means_tmp);
		if (feature_vars_tmp)
			NDB_FREE(feature_vars_tmp);
		if (feature_importance_tmp)
			NDB_FREE(feature_importance_tmp);
		if (feature_sums)
			NDB_FREE(feature_sums);
		if (feature_sums_sq)
			NDB_FREE(feature_sums_sq);
		if (class_feature_sums)
			NDB_FREE(class_feature_sums);
		if (class_feature_counts)
			NDB_FREE(class_feature_counts);
		if (left_feature_sums_vec)
			NDB_FREE(left_feature_sums_vec);
		if (right_feature_sums_vec)
			NDB_FREE(right_feature_sums_vec);
		if (left_feature_counts_vec)
			NDB_FREE(left_feature_counts_vec);
		if (right_feature_counts_vec)
			NDB_FREE(right_feature_counts_vec);
		if (left_branch_means_vec)
			NDB_FREE(left_branch_means_vec);
		if (right_branch_means_vec)
			NDB_FREE(right_branch_means_vec);
		if (feature_order)
			NDB_FREE(feature_order);
		if (trees)
			NDB_FREE(trees);
		if (tree_majorities)
			NDB_FREE(tree_majorities);
		if (tree_majority_fractions)
			NDB_FREE(tree_majority_fractions);
		if (tree_seconds)
			NDB_FREE(tree_seconds);
		if (tree_second_fractions)
			NDB_FREE(tree_second_fractions);
		if (tree_oob_accuracy)
			NDB_FREE(tree_oob_accuracy);
		if (bootstrap_indices)
			NDB_FREE(bootstrap_indices);
		rf_dataset_free(&dataset);
	}

	/* Free query.data BEFORE ending SPI session (it's allocated in SPI context) */
	if (query.data)
		NDB_FREE(query.data);
	NDB_SPI_SESSION_END(train_spi_session);

	if (table_name)
		NDB_FREE(table_name);
	if (feature_col)
		NDB_FREE(feature_col);
	if (label_col)
		NDB_FREE(label_col);
	PG_RETURN_INT32(model_id);
}

static double
rf_tree_predict_single(const GTree * tree,
					   const RFModel * model,
					   const Vector *vec,
					   double *left_dist,
					   double *right_dist,
					   int *leaf_out)
{
	const		GTreeNode *nodes;
	int			idx;
	int			steps = 0;
	int			path_nodes[GTREE_MAX_DEPTH + 1];
	char		path_dir[GTREE_MAX_DEPTH];
	int			path_len = 0;
	int			leaf_idx = -1;
	double		result = 0.0;
	int			i;

	if (leaf_out)
		*leaf_out = -1;

	if (left_dist)
		*left_dist = -1.0;
	if (right_dist)
		*right_dist = -1.0;

	if (tree == NULL)
		return (model != NULL) ? model->majority_value : 0.0;

	if (tree->root < 0 || tree->count <= 0)
		return (model != NULL) ? model->majority_value : 0.0;

	nodes = gtree_nodes(tree);
	idx = tree->root;

	while (idx >= 0 && idx < tree->count)
	{
		const		GTreeNode *node = &nodes[idx];

		if (path_len <= GTREE_MAX_DEPTH)
			path_nodes[path_len] = idx;

		if (node->is_leaf)
		{
			leaf_idx = idx;
			break;
		}

		if (vec == NULL || node->feature_idx < 0
			|| node->feature_idx >= vec->dim)
		{
			elog(DEBUG1,
				 "random_forest: path aborted at node %d "
				 "(feature %d)",
				 idx,
				 node->feature_idx);
			if (model)
				return model->majority_value;
			return 0.0;
		}

		if (vec->data[node->feature_idx] <= node->threshold)
		{
			if (path_len < GTREE_MAX_DEPTH)
				path_dir[path_len] = 'L';
			idx = node->left;
		}
		else
		{
			if (path_len < GTREE_MAX_DEPTH)
				path_dir[path_len] = 'R';
			idx = node->right;
		}

		path_len++;

		if (++steps > GTREE_MAX_DEPTH)
			break;
	}

	if (leaf_idx >= 0 && leaf_idx < tree->count && nodes[leaf_idx].is_leaf)
		result = nodes[leaf_idx].value;
	else
		result = (model != NULL) ? model->majority_value : 0.0;

	if (path_len > GTREE_MAX_DEPTH)
		path_len = GTREE_MAX_DEPTH;

	if (leaf_idx >= 0 && path_len <= GTREE_MAX_DEPTH)
		path_nodes[path_len] = leaf_idx;

	{
		StringInfoData path_log;
		int			edge_count = (leaf_idx >= 0) ? path_len : path_len - 1;

		initStringInfo(&path_log);
		appendStringInfo(&path_log, "[");
		for (i = 0; i <= edge_count && i <= GTREE_MAX_DEPTH; i++)
		{
			if (i > 0)
				appendStringInfoString(&path_log, ", ");
			appendStringInfo(&path_log, "%d", path_nodes[i]);
			if (i < edge_count && i < GTREE_MAX_DEPTH)
				appendStringInfo(&path_log, "%c", path_dir[i]);
		}
		appendStringInfoChar(&path_log, ']');
		elog(DEBUG1,
			 "random_forest: gtree path len=%d leaf_idx=%d path=%s",
			 (edge_count < 0) ? 0 : (edge_count + 1),
			 leaf_idx,
			 path_log.data);
		NDB_FREE(path_log.data);
	}

	if (left_dist != NULL && right_dist != NULL && vec != NULL
		&& model != NULL && model->feature_limit > 0
		&& model->left_branch_means != NULL
		&& model->right_branch_means != NULL)
	{
		int			limit = model->feature_limit;
		int			f;
		const float *vec_data;
		double		lsum = 0.0;
		double		rsum = 0.0;

		if (vec->dim < limit)
			limit = vec->dim;
		if (model->n_features < limit)
			limit = model->n_features;

		vec_data = vec->data;
		for (f = 0; f < limit; f++)
		{
			double		val = (double) vec_data[f];
			double		ldiff = val - model->left_branch_means[f];
			double		rdiff = val - model->right_branch_means[f];

			lsum += ldiff * ldiff;
			rsum += rdiff * rdiff;
		}

		*left_dist = sqrt(lsum);
		*right_dist = sqrt(rsum);
	}

	if (leaf_out)
		*leaf_out = leaf_idx;

	return result;
}

Datum
predict_random_forest(PG_FUNCTION_ARGS)
{
	int32		model_id;
	RFModel    *model;
	Vector	   *feature_vec = NULL;
	double		result;
	double		split_z = 0.0;
	bool		split_z_valid = false;
	const char *branch_name = "majority";
	double		branch_fraction = 0.0;
	double		branch_value = 0.0;
	double		left_mean_dist = -1.0;
	double		right_mean_dist = -1.0;
	int			mean_limit = 0;
	double		vote_majority = 0.0;
	double		best_vote_fraction = 0.0;
	double		second_vote_value = 0.0;
	double		second_vote_fraction = 0.0;
	double		vote_total_weight = 0.0;
	int			i;
	double	   *vote_histogram = NULL;
	int			vote_classes = 0;
	double		fallback_value = 0.0;
	double		fallback_fraction = 0.0;
	int			top_feature_idx = -1;
	double		top_feature_importance = 0.0;

	if (!PG_ARGISNULL(0))
		model_id = PG_GETARG_INT32(0);
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_random_forest: model ID is required"),
				 errdetail("First argument (model_id) cannot be NULL"),
				 errhint("Provide a valid model ID from the training function.")));

	if (!PG_ARGISNULL(1))
	{
		feature_vec = PG_GETARG_VECTOR_P(1);
		NDB_CHECK_VECTOR_VALID(feature_vec);
	}
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_random_forest: feature vector is required"),
				 errdetail("Second argument (feature vector) cannot be NULL"),
				 errhint("Provide a valid feature vector for prediction.")));

	if (!rf_lookup_model(model_id, &model))
	{
		if (rf_try_gpu_predict_catalog(model_id, feature_vec, &result))
			PG_RETURN_FLOAT8(result);
		if (!rf_load_model_from_catalog(model_id, &model))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: predict_random_forest: model %d not found", model_id),
					 errdetail("Model with ID %d does not exist in the catalog", model_id),
					 errhint("Verify the model ID or train a new model.")));
	}

	fallback_value = model->second_value;
	fallback_fraction = model->second_fraction;
	branch_fraction = model->majority_fraction;
	branch_value = model->majority_value;

	if (model->n_features > 0 && feature_vec != NULL
		&& feature_vec->dim != model->n_features)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict_random_forest: feature dimension mismatch"),
				 errdetail("Model expects %d features but received %d", model->n_features, feature_vec->dim),
				 errhint("Ensure the feature vector has the same dimension as the training data.")));

	if (model->feature_means != NULL && feature_vec != NULL)
	{
		float	   *vec_data = feature_vec->data;
		double		dist = 0.0;
		int			j;
		double		max_z = 0.0;

		for (j = 0; j < model->n_features && j < feature_vec->dim; j++)
		{
			double		diff =
				(double) vec_data[j] - model->feature_means[j];

			dist += diff * diff;
			if (model->feature_variances != NULL)
			{
				double		var = model->feature_variances[j];
				double		z;

				if (var <= 0.0)
					continue;
				z = fabs(diff) / sqrt(var);
				if (z > max_z)
					max_z = z;
			}
		}
		dist = sqrt(dist);
		if (model->feature_variances != NULL)
			elog(DEBUG1,
				 "random_forest: feature L2 distance %.3f max-z %.3f",
				 dist,
				 max_z);
		else
			elog(DEBUG1,
				 "random_forest: feature L2 distance to mean %.3f",
				 dist);
		if (model->feature_variances != NULL
			&& model->second_fraction > 0.0 && max_z > 1.5)
			elog(DEBUG1,
				 "random_forest: deviation %.3f exceeds "
				 "threshold, considering second class %.3f",
				 max_z,
				 model->second_value);
		model->max_deviation = max_z;
	}
	else if (!PG_ARGISNULL(1))
		model->max_deviation = 0.0;

	if (feature_vec != NULL && model->split_feature >= 0)
	{
		int			sf = model->split_feature;

		if (sf < feature_vec->dim)
		{
			double		value = (double) feature_vec->data[sf];

			if (value <= model->split_threshold)
			{
				branch_name = "left";
				branch_fraction = model->left_branch_fraction;
				branch_value = model->left_branch_value;
			}
			else
			{
				branch_name = "right";
				branch_fraction = model->right_branch_fraction;
				branch_value = model->right_branch_value;
			}
			if (branch_fraction <= 0.0)
				branch_fraction = model->majority_fraction;

			if (model->feature_variances != NULL
				&& sf < model->n_features)
			{
				double		var = model->feature_variances[sf];

				if (var > 0.0)
				{
					split_z =
						(value - model->split_threshold)
						/ sqrt(var);
					if (fabs(split_z)
						> model->max_split_deviation)
						model->max_split_deviation =
							fabs(split_z);
					split_z_valid = true;
				}
			}
		}
	}

	if (model->n_classes > 0)
	{
		vote_classes = model->n_classes;
		NDB_ALLOC(vote_histogram, double, vote_classes);
	}

	if (model->tree_count > 0 && model->trees != NULL)
	{
		for (i = 0; i < model->tree_count; i++)
		{
			const		GTree *tree = model->trees[i];
			double		tree_left = -1.0;
			double		tree_right = -1.0;
			int			leaf_idx = -1;
			double		tree_result;
			double		vote_weight = 1.0;

			tree_result = rf_tree_predict_single(tree,
												 model,
												 feature_vec,
												 &tree_left,
												 &tree_right,
												 &leaf_idx);

			if (model->tree_oob_accuracy != NULL
				&& i < model->tree_count)
			{
				vote_weight = model->tree_oob_accuracy[i];
				if (vote_weight <= 0.0)
					vote_weight = 1.0;
			}

			if (vote_histogram != NULL)
			{
				int			cls = (int) rint(tree_result);

				if (cls >= 0 && cls < vote_classes)
				{
					vote_histogram[cls] += vote_weight;
					vote_total_weight += vote_weight;
				}
			}

			if (tree_left >= 0.0 && tree_right >= 0.0)
			{
				if (mean_limit <= 0)
				{
					left_mean_dist = tree_left;
					right_mean_dist = tree_right;
					mean_limit = model->feature_limit;
				}
			}
		}
	}
	else
	{
		result = rf_tree_predict_single(model->tree,
										model,
										feature_vec,
										&left_mean_dist,
										&right_mean_dist,
										NULL);
		if (vote_histogram != NULL)
		{
			int			cls = (int) rint(result);

			if (cls >= 0 && cls < vote_classes)
			{
				vote_histogram[cls] += 1.0;
				vote_total_weight += 1.0;
			}
		}
	}

	if (vote_histogram != NULL && vote_total_weight > 0.0)
	{
		int			best_idx = -1;
		int			second_idx = -1;
		double		best_weight = -1.0;
		double		second_weight = -1.0;

		for (i = 0; i < vote_classes; i++)
		{
			double		weight = vote_histogram[i];

			if (weight > best_weight)
			{
				if (best_idx >= 0)
				{
					second_idx = best_idx;
					second_weight = best_weight;
				}
				best_idx = i;
				best_weight = weight;
			}
			else if (weight > second_weight)
			{
				second_idx = i;
				second_weight = weight;
			}
		}

		if (best_idx >= 0 && best_weight > 0.0)
		{
			vote_majority = (double) best_idx;
			best_vote_fraction = best_weight / vote_total_weight;
			result = vote_majority;
			branch_name = "forest";
			branch_fraction = best_vote_fraction;
			branch_value = vote_majority;
		}

		if (second_idx >= 0 && second_weight > 0.0)
		{
			second_vote_value = (double) second_idx;
			second_vote_fraction =
				second_weight / vote_total_weight;
		}

		NDB_FREE(vote_histogram);
		vote_histogram = NULL;
	}

	if (best_vote_fraction <= 0.0)
		result = model->majority_value;

	if (second_vote_fraction > 0.0)
	{
		fallback_value = second_vote_value;
		fallback_fraction = second_vote_fraction;
	}

	if (mean_limit > 0 && left_mean_dist >= 0.0 && right_mean_dist >= 0.0)
	{
		elog(DEBUG1,
			 "random_forest: branch mean distances left=%.3f "
			 "right=%.3f limit=%d",
			 left_mean_dist,
			 right_mean_dist,
			 mean_limit);

		if (right_mean_dist + 0.10 < left_mean_dist
			&& model->right_branch_fraction > 0.0)
		{
			result = model->right_branch_value;
			branch_name = "right-mean";
			branch_fraction = model->right_branch_fraction;
			branch_value = model->right_branch_value;
		}
		else if (left_mean_dist + 0.10 < right_mean_dist
				 && model->left_branch_fraction > 0.0)
		{
			result = model->left_branch_value;
			branch_name = "left-mean";
			branch_fraction = model->left_branch_fraction;
			branch_value = model->left_branch_value;
		}
	}

	if (model->feature_variances != NULL && fallback_fraction > 0.0
		&& !PG_ARGISNULL(1) && model->max_deviation > 2.0
		&& model->label_entropy > 0.1)
	{
		elog(DEBUG1,
			 "random_forest: high deviation %.3f -> returning "
			 "second class %.3f",
			 model->max_deviation,
			 fallback_value);
		result = fallback_value;
		branch_name = "fallback";
		branch_fraction = fallback_fraction;
		branch_value = fallback_value;
	}

	if (result != model->majority_value)
		elog(DEBUG1,
			 "random_forest: majority=%.3f frac=%.3f, fallback=%.3f "
			 "frac=%.3f branch=%s bfrac=%.3f entropy=%.3f "
			 "split_dev=%.3f",
			 model->majority_value,
			 model->majority_fraction,
			 result,
			 fallback_fraction,
			 branch_name,
			 branch_fraction,
			 model->label_entropy,
			 model->max_split_deviation);

	if (model->split_feature >= 0)
	{
		if (split_z_valid)
			elog(DEBUG1,
				 "random_forest: split f=%d thr=%.3f left=%.3f "
				 "lf=%.3f right=%.3f rf=%.3f z=%.3f",
				 model->split_feature,
				 model->split_threshold,
				 model->left_branch_value,
				 model->left_branch_fraction,
				 model->right_branch_value,
				 model->right_branch_fraction,
				 split_z);
		else
			elog(DEBUG1,
				 "random_forest: split f=%d thr=%.3f left=%.3f "
				 "lf=%.3f right=%.3f rf=%.3f",
				 model->split_feature,
				 model->split_threshold,
				 model->left_branch_value,
				 model->left_branch_fraction,
				 model->right_branch_value,
				 model->right_branch_fraction);
	}

	if (model->feature_importance != NULL && model->n_features > 0)
	{
		for (i = 0; i < model->n_features; i++)
		{
			double		val = model->feature_importance[i];

			if (val > top_feature_importance)
			{
				top_feature_importance = val;
				top_feature_idx = i;
			}
		}
	}

	elog(DEBUG1,
		 "predict_random_forest: returning %.3f (branch %s leaf %.3f "
		 "frac "
		 "%.3f majority %.3f frac %.3f gini %.3f ldist %.3f rdist %.3f "
		 "topfeat=%d topimp=%.3f)",
		 result,
		 branch_name,
		 branch_value,
		 branch_fraction,
		 model->majority_value,
		 model->majority_fraction,
		 model->gini_impurity,
		 left_mean_dist,
		 right_mean_dist,
		 top_feature_idx,
		 top_feature_importance);
	PG_RETURN_FLOAT8(result);
}

Datum
evaluate_random_forest(PG_FUNCTION_ARGS)
{
	Datum		result_datums[4];
	ArrayType  *result_array;
	int32		model_id;
	RFModel    *model = NULL;
	double		accuracy = 0.0;
	double		error_rate = 0.0;
	double		gini = 0.0;
	int			n_classes = 0;
	bytea	   *payload = NULL;
	Jsonb	   *metrics = NULL;

	if (PG_NARGS() < 1 || PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest: model_id is required"),
				 errdetail("First argument (model_id) cannot be NULL"),
				 errhint("Provide a valid model ID from the training function.")));

	model_id = PG_GETARG_INT32(0);

	if (!rf_lookup_model(model_id, &model))
	{
		elog(DEBUG1,
			 "evaluate_random_forest: model_id %d not in cache, trying catalog",
			 model_id);
		if (!ml_catalog_fetch_model_payload(
											model_id, &payload, NULL, &metrics))
		{
			elog(WARNING,
				 "evaluate_random_forest: ml_catalog_fetch_model_payload returned false for model_id %d",
				 model_id);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("random_forest: model %d not found",
							model_id)));
		}

		elog(DEBUG1,
			 "evaluate_random_forest: metrics=%p, rf_metadata_is_gpu=%d",
			 (void *) metrics,
			 metrics != NULL ? rf_metadata_is_gpu(metrics) : 0);
		if (rf_metadata_is_gpu(metrics) && metrics != NULL)
		{
			Datum		acc_datum;
			Datum		gini_datum;
			Datum		acc_numeric;
			Datum		gini_numeric;
			Numeric		acc_num;
			Numeric		gini_num;

			acc_datum = DirectFunctionCall2(
											jsonb_object_field,
											JsonbPGetDatum(metrics),
											CStringGetTextDatum("majority_fraction"));
			if (DatumGetPointer(acc_datum) == NULL)
			{
				acc_datum = DirectFunctionCall2(
												jsonb_object_field,
												JsonbPGetDatum(metrics),
												CStringGetTextDatum("oob_accuracy"));
			}
			if (DatumGetPointer(acc_datum) != NULL)
			{
				PG_TRY();
				{
					acc_numeric = DirectFunctionCall1(
													  jsonb_numeric,
													  acc_datum);
					if (DatumGetPointer(acc_numeric) != NULL)
					{
						acc_num = DatumGetNumeric(acc_numeric);
						accuracy = DatumGetFloat8(
												  DirectFunctionCall1(numeric_float8,
																	  NumericGetDatum(acc_num)));
					}
				}
				PG_CATCH();
				{
					elog(DEBUG1,
						 "neurondb: evaluate_random_forest: jsonb_numeric failed, trying text extraction");
					{
						Datum		acc_text;

						acc_text = DirectFunctionCall1(
													   jsonb_extract_path_text,
													   acc_datum);
						if (DatumGetPointer(acc_text) != NULL)
						{
							char	   *acc_str;

							acc_str = TextDatumGetCString(acc_text);

							if (acc_str != NULL && strlen(acc_str) > 0)
							{
								accuracy = strtod(acc_str, NULL);
								elog(DEBUG1,
									 "evaluate_random_forest: extracted accuracy=%.6f from text",
									 accuracy);
							}
							NDB_FREE(acc_str);
						}
					}
				}
				PG_END_TRY();
			}

			gini_datum = DirectFunctionCall2(
											 jsonb_object_field,
											 JsonbPGetDatum(metrics),
											 CStringGetTextDatum("gini"));
			if (DatumGetPointer(gini_datum) != NULL)
			{
				PG_TRY();
				{
					gini_numeric = DirectFunctionCall1(
													   jsonb_numeric,
													   gini_datum);
					if (DatumGetPointer(gini_numeric) != NULL)
					{
						gini_num = DatumGetNumeric(gini_numeric);
						gini = DatumGetFloat8(
											  DirectFunctionCall1(numeric_float8,
																  NumericGetDatum(gini_num)));
					}
				}
				PG_CATCH();
				{
					Datum		gini_text = DirectFunctionCall1(
																jsonb_extract_path_text,
																gini_datum);

					if (DatumGetPointer(gini_text) != NULL)
					{
						char	   *gini_str = TextDatumGetCString(gini_text);

						if (gini_str != NULL && strlen(gini_str) > 0)
							gini = strtod(gini_str, NULL);
						NDB_FREE(gini_str);
					}
				}
				PG_END_TRY();
			}

			/* n_classes is not in metrics, use default */
			n_classes = 2;

			if (metrics)
				NDB_FREE(metrics);
			if (payload)
				NDB_FREE(payload);

			error_rate = (accuracy > 1.0) ? 0.0 : (1.0 - accuracy);

			result_datums[0] = Float8GetDatum(accuracy);
			result_datums[1] = Float8GetDatum(error_rate);
			result_datums[2] = Float8GetDatum(gini);
			result_datums[3] = Float8GetDatum((double) n_classes);

			result_array = construct_array(
										   result_datums, 4, FLOAT8OID, sizeof(float8), true, 'd');

			PG_RETURN_ARRAYTYPE_P(result_array);
		}

		elog(DEBUG1,
			 "evaluate_random_forest: not a GPU model, trying CPU load for model_id %d",
			 model_id);

		PG_TRY();
		{
			if (!rf_load_model_from_catalog(model_id, &model))
			{
				elog(DEBUG1,
					 "evaluate_random_forest: rf_load_model_from_catalog failed for model_id %d",
					 model_id);
				if (payload)
					NDB_FREE(payload);
				if (metrics)
					NDB_FREE(metrics);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("random_forest: model %d not found or failed to load",
								model_id)));
			}
		}
		PG_CATCH();
		{
			elog(WARNING, "evaluate_random_forest: exception during CPU model load for model_id %d", model_id);
			if (payload)
				NDB_FREE(payload);
			if (metrics)
				NDB_FREE(metrics);
			FlushErrorState();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("random_forest: failed to load model %d (deserialization error)",
							model_id)));
		}
		PG_END_TRY();
	}

	if (model == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("random_forest: model %d not found",
						model_id)));

	if (model->oob_accuracy > 0.0)
		accuracy = model->oob_accuracy;
	else
		accuracy = model->majority_fraction;
	error_rate = (accuracy > 1.0) ? 0.0 : (1.0 - accuracy);
	gini = model->gini_impurity;
	n_classes = model->n_classes;

	result_datums[0] = Float8GetDatum(accuracy);
	result_datums[1] = Float8GetDatum(error_rate);
	result_datums[2] = Float8GetDatum(gini);
	result_datums[3] = Float8GetDatum((double) n_classes);

	result_array = construct_array(
								   result_datums, 4, FLOAT8OID, sizeof(float8), true, 'd');

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/*
 * rf_predict_batch
 *
 * Helper function to predict a batch of samples using Random Forest model.
 * Returns predictions array (double*) and updates confusion matrix.
 */
static void
rf_predict_batch(const RFModel * model,
				 const float *features,
				 const double *labels,
				 int n_samples,
				 int feature_dim,
				 int *tp_out,
				 int *tn_out,
				 int *fp_out,
				 int *fn_out)
{
	int			i;
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;
	int			n_classes = model->n_classes;
	double	   *vote_histogram = NULL;
	int			vote_classes = n_classes;

	if (model == NULL || features == NULL || labels == NULL || n_samples <= 0)
	{
		if (tp_out)
			*tp_out = 0;
		if (tn_out)
			*tn_out = 0;
		if (fp_out)
			*fp_out = 0;
		if (fn_out)
			*fn_out = 0;
		return;
	}

	/* Ensure n_classes is at least 2 for binary classification */
	if (vote_classes <= 0)
	{
		elog(DEBUG1, "rf_predict_batch: model->n_classes=%d is invalid, defaulting to 2", model->n_classes);
		vote_classes = 2;
		n_classes = 2;
	}

	if (vote_classes > 0)
		NDB_ALLOC(vote_histogram, double, vote_classes);
	NDB_CHECK_ALLOC(vote_histogram, "vote_histogram");

	for (i = 0; i < n_samples; i++)
	{
		const float *row = features + (i * feature_dim);
		double		y_true = labels[i];
		int			true_class;
		double		prediction = 0.0;
		int			pred_class;
		int			j;
		double		vote_total_weight = 0.0;

		if (!isfinite(y_true))
		{
			elog(DEBUG1, "rf_predict_batch: row %d: y_true is not finite (%.6f), skipping", i, y_true);
			continue;
		}

		true_class = (int) rint(y_true);
		if (true_class < 0 || true_class >= n_classes)
		{
			elog(DEBUG1, "rf_predict_batch: row %d: true_class=%d is out of range [0, %d), skipping", i, true_class, n_classes);
			continue;
		}

		if (vote_histogram != NULL)
		{
			for (j = 0; j < vote_classes; j++)
				vote_histogram[j] = 0.0;
		}

		if (model->tree_count > 0 && model->trees != NULL)
		{
			int			t;

			for (t = 0; t < model->tree_count; t++)
			{
				const		GTree *tree = model->trees[t];
				double		tree_result;
				double		vote_weight = 1.0;

				tree_result = rf_tree_predict_row(tree, row, feature_dim);

				if (model->tree_oob_accuracy != NULL && t < model->tree_count)
				{
					vote_weight = model->tree_oob_accuracy[t];
					if (vote_weight <= 0.0)
						vote_weight = 1.0;
				}

				if (vote_histogram != NULL)
				{
					int			cls = (int) rint(tree_result);

					if (cls >= 0 && cls < vote_classes)
					{
						vote_histogram[cls] += vote_weight;
						vote_total_weight += vote_weight;
					}
				}
			}
		}
		else if (model->tree != NULL)
		{
			double		tree_result;

			tree_result = rf_tree_predict_row(model->tree, row, feature_dim);
			if (vote_histogram != NULL)
			{
				int			cls = (int) rint(tree_result);

				if (cls >= 0 && cls < vote_classes)
				{
					vote_histogram[cls] += 1.0;
					vote_total_weight += 1.0;
				}
			}
		}
		else
		{
			prediction = model->majority_value;
			pred_class = (int) rint(prediction);
		}

		if (vote_histogram != NULL && vote_total_weight > 0.0)
		{
			int			best_idx = -1;
			double		best_weight = -1.0;

			for (j = 0; j < vote_classes; j++)
			{
				if (vote_histogram[j] > best_weight)
				{
					best_idx = j;
					best_weight = vote_histogram[j];
				}
			}

			if (best_idx >= 0)
			{
				prediction = (double) best_idx;
				pred_class = best_idx;
			}
			else
			{
				prediction = model->majority_value;
				pred_class = (int) rint(prediction);
			}
		}
		else
		{
			prediction = model->majority_value;
			pred_class = (int) rint(prediction);
		}

		/* Ensure pred_class is in valid range for binary classification */
		if (n_classes == 2)
		{
			if (pred_class < 0)
				pred_class = 0;
			else if (pred_class > 1)
				pred_class = 1;
			
			if (true_class == 1 && pred_class == 1)
				tp++;
			else if (true_class == 0 && pred_class == 0)
				tn++;
			else if (true_class == 0 && pred_class == 1)
				fp++;
			else if (true_class == 1 && pred_class == 0)
				fn++;
			else
			{
				elog(DEBUG1, "rf_predict_batch: row %d: unexpected class combination (true_class=%d, pred_class=%d, n_classes=%d, prediction=%.6f)", 
					 i, true_class, pred_class, n_classes, prediction);
			}
		}
		else
		{
			/* For multi-class, ensure pred_class is in valid range */
			if (pred_class < 0)
				pred_class = 0;
			else if (pred_class >= n_classes)
				pred_class = n_classes - 1;
			
			if (true_class == pred_class)
				tp++;
			else
				fn++;
		}
	}

	if (vote_histogram != NULL)
		NDB_FREE(vote_histogram);

	elog(DEBUG1, "rf_predict_batch: final confusion matrix - tp=%d, tn=%d, fp=%d, fn=%d (n_samples=%d, n_classes=%d)", 
		 tp, tn, fp, fn, n_samples, n_classes);

	if (tp_out)
		*tp_out = tp;
	if (tn_out)
		*tn_out = tn;
	if (fp_out)
		*fp_out = fp;
	if (fn_out)
		*fn_out = fn;
}

/*
 * evaluate_random_forest_by_model_id
 *
 * Evaluates Random Forest model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: accuracy, precision, recall, f1_score, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_random_forest_by_model_id);

Datum
evaluate_random_forest_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	char	   *tbl_str = NULL;
	char	   *feat_str = NULL;
	char	   *targ_str = NULL;
	int			ret;
	int			nvec = 0;
	int			i;
	int			j;
	Oid			feat_type_oid = InvalidOid;
	bool		feat_is_array = false;
	double		accuracy = 0.0;
	double		precision = 0.0;
	double		recall = 0.0;
	double		f1_score = 0.0;
	int			tp = 0;
	int			tn = 0;
	int			fp = 0;
	int			fn = 0;
	MemoryContext oldcontext;
	NDB_DECLARE (NdbSpiSession *, eval_spi_session);
	StringInfoData query = {0};
	RFModel    *model = NULL;
	Jsonb	   *result_jsonb = NULL;
	bytea	   *gpu_payload = NULL;
	Jsonb	   *gpu_metrics = NULL;
	bool		is_gpu_model = false;

	elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] Function entry, PG_NARGS=%d", PG_NARGS());

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);
	elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] model_id=%d", model_id);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] table=%s, feature_col=%s, label_col=%s", tbl_str, feat_str, targ_str);

	oldcontext = CurrentMemoryContext;
	elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] saved oldcontext=%p", (void *) oldcontext);

	if (!rf_lookup_model(model_id, &model))
	{
		if (!rf_load_model_from_catalog(model_id, &model))
		{
			if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
			{
				if (gpu_payload == NULL)
				{
					if (gpu_metrics)
						NDB_FREE(gpu_metrics);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: evaluate_random_forest_by_model_id: model %d has no model_data (model may not have been trained or stored correctly)",
									model_id),
							 errhint("The model exists in the catalog but has no training data. Please retrain the model.")));
				}

				is_gpu_model = (gpu_metrics != NULL) ? rf_metadata_is_gpu(gpu_metrics) : false;
				if (!is_gpu_model && gpu_payload != NULL)
				{
					elog(DEBUG1, "evaluate_random_forest_by_model_id: model %d not marked as GPU, trying CPU load", model_id);
					if (gpu_payload)
						NDB_FREE(gpu_payload);
					if (gpu_metrics)
						NDB_FREE(gpu_metrics);
					gpu_payload = NULL;
					gpu_metrics = NULL;
					if (!rf_load_model_from_catalog(model_id, &model))
					{
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neurondb: evaluate_random_forest_by_model_id: model %d not found",
										model_id)));
					}
				}
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_random_forest_by_model_id: model %d not found",
								model_id)));
			}
		}
	}

	if (model == NULL && !is_gpu_model && gpu_payload == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: model %d not found",
						model_id)));

	oldcontext = CurrentMemoryContext;
	Assert(oldcontext != NULL);
	NDB_SPI_SESSION_BEGIN(eval_spi_session, oldcontext);

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(targ_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(targ_str));
	elog(DEBUG1, "evaluate_random_forest_by_model_id: executing query: %s", query.data);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(query.data);
		if (model != NULL)
			rf_free_deserialized_model(model);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		NDB_SPI_SESSION_END(eval_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and label columns.")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		NDB_FREE(query.data);
		if (model != NULL)
			rf_free_deserialized_model(model);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		NDB_SPI_SESSION_END(eval_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_random_forest_by_model_id: no valid rows found"),
				 errdetail("Dataset contains %d rows, minimum required is 10", nvec),
				 errhint("Add more data rows to the evaluation table.")));
	}

	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/*
	 * GPU batch evaluation path - try GPU first if available and we have
	 * payload. Even if metadata doesn't indicate GPU, try GPU evaluation
	 * first.
	 */
	if (neurondb_gpu_is_available() && gpu_payload != NULL)
	{
#ifdef NDB_GPU_CUDA
		const		NdbCudaRfModelHeader *gpu_hdr;
		int		   *h_labels = NULL;
		float	   *h_features = NULL;
		int			feat_dim = 0;
		int			valid_rows = 0;
		size_t		payload_size;

		payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
		if (payload_size < sizeof(NdbCudaRfModelHeader))
		{
			elog(WARNING,
				 "evaluate_random_forest_by_model_id: GPU payload too small (%zu bytes), falling back to CPU",
				 payload_size);
			goto cpu_evaluation_path;
		}

		gpu_hdr = (const NdbCudaRfModelHeader *) VARDATA(gpu_payload);
		if (gpu_hdr == NULL)
		{
			elog(WARNING,
				 "evaluate_random_forest_by_model_id: NULL GPU header, falling back to CPU");
			goto cpu_evaluation_path;
		}

		feat_dim = gpu_hdr->feature_dim;
		if (feat_dim <= 0 || feat_dim > 100000)
		{
			elog(WARNING,
				 "evaluate_random_forest_by_model_id: invalid feature_dim (%d), falling back to CPU",
				 feat_dim);
			goto cpu_evaluation_path;
		}

		{
			size_t		features_size = sizeof(float) * (size_t) nvec * (size_t) feat_dim;
			size_t		labels_size = sizeof(int) * (size_t) nvec;

			if (features_size > MaxAllocSize || labels_size > MaxAllocSize)
			{
				elog(WARNING,
					 "evaluate_random_forest_by_model_id: allocation size too large (features=%zu, labels=%zu), falling back to CPU",
					 features_size, labels_size);
				goto cpu_evaluation_path;
			}

			NDB_ALLOC(h_features, float, features_size / sizeof(float));
			NDB_ALLOC(h_labels, int, labels_size / sizeof(int));

			if (h_features == NULL || h_labels == NULL)
			{
				elog(WARNING,
					 "evaluate_random_forest_by_model_id: memory allocation failed, falling back to CPU");
				if (h_features != NULL)
				{
					NDB_FREE(h_features);
					h_features = NULL;
				}
				if (h_labels != NULL)
				{
					NDB_FREE(h_labels);
					h_labels = NULL;
				}
				goto cpu_evaluation_path;
			}
		}

		/*
		 * Extract features and labels from SPI results - optimized batch
		 * extraction
		 */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			if (tupdesc == NULL)
			{
				elog(WARNING,
					 "evaluate_random_forest_by_model_id: NULL TupleDesc, falling back to CPU");
				NDB_FREE(h_features);
				NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple;
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				Vector	   *vec;
				ArrayType  *arr;
				float	   *feat_row;

				/* Safe access to SPI_tuptable - validate before access */
				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || i >= SPI_processed)
					break;

				tuple = SPI_tuptable->vals[i];
				if (tuple == NULL || tupdesc == NULL)
					continue;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				/* Safe access for target - validate tupdesc has at least 2 columns */
				if (tupdesc->natts < 2)
				{
					continue;
				}
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				if (valid_rows >= nvec)
				{
					elog(WARNING,
						 "evaluate_random_forest_by_model_id: valid_rows overflow, breaking");
					break;
				}

				feat_row = h_features + (valid_rows * feat_dim);
				if (feat_row == NULL || feat_row < h_features || feat_row >= h_features + (nvec * feat_dim))
				{
					elog(WARNING,
						 "evaluate_random_forest_by_model_id: feat_row out of bounds, skipping row");
					continue;
				}

				h_labels[valid_rows] = (int) rint(DatumGetFloat8(targ_datum));

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);
						int			j_remain = feat_dim % 4;
						int			j_end = feat_dim - j_remain;

						/*
						 * Process 4 elements at a time for better cache
						 * locality
						 */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float) data[j];
							feat_row[j + 1] = (float) data[j + 1];
							feat_row[j + 2] = (float) data[j + 2];
							feat_row[j + 3] = (float) data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float) data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);

						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			if (h_features != NULL)
			{
				NDB_FREE(h_features);
				h_features = NULL;
			}
			if (h_labels != NULL)
			{
				NDB_FREE(h_labels);
				h_labels = NULL;
			}
			if (gpu_payload != NULL)
			{
				NDB_FREE(gpu_payload);
				gpu_payload = NULL;
			}
			if (gpu_metrics != NULL)
			{
				NDB_FREE(gpu_metrics);
				gpu_metrics = NULL;
			}
			if (query.data != NULL)
			{
				NDB_FREE(query.data);
				query.data = NULL;
			}
			if (tbl_str != NULL)
			{
				NDB_FREE(tbl_str);
				tbl_str = NULL;
			}
			if (feat_str != NULL)
			{
				NDB_FREE(feat_str);
				feat_str = NULL;
			}
			if (targ_str != NULL)
			{
				NDB_FREE(targ_str);
				targ_str = NULL;
			}
			NDB_SPI_SESSION_END(eval_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_random_forest_by_model_id: no valid rows found"),
					 errdetail("Dataset contains %d rows, minimum required is 10", nvec),
					 errhint("Add more data rows to the evaluation table.")));
		}

		/* Use optimized GPU batch evaluation */
		{
			int			rc;
			char	   *gpu_errstr = NULL;
			bool		cleanup_done = false;
			bool		h_features_freed = false;
			bool		h_labels_freed = false;

			/*
			 * Note: gpu_errstr is allocated by GPU function using pstrdup() -
			 * never free it manually
			 */

			elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] Starting GPU evaluation path, valid_rows=%d, feat_dim=%d", valid_rows, feat_dim);

			/* Defensive checks before GPU call */
			if (h_features == NULL || h_labels == NULL || valid_rows <= 0 || feat_dim <= 0)
			{
				elog(WARNING,
					 "evaluate_random_forest_by_model_id: [DEBUG] invalid inputs for GPU evaluation (features=%p, labels=%p, rows=%d, dim=%d), falling back to CPU",
					 (void *) h_features, (void *) h_labels, valid_rows, feat_dim);
				if (h_features != NULL && !h_features_freed)
				{
					elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] freeing h_features (early exit)");
					NDB_FREE(h_features);
					h_features = NULL;
					h_features_freed = true;
				}
				if (h_labels != NULL && !h_labels_freed)
				{
					elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] freeing h_labels (early exit)");
					NDB_FREE(h_labels);
					h_labels = NULL;
					h_labels_freed = true;
				}
				goto cpu_evaluation_path;
			}

			elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] calling ndb_cuda_rf_evaluate_batch, h_features=%p, h_labels=%p", (void *) h_features, (void *) h_labels);

			PG_TRY();
			{
				rc = ndb_cuda_rf_evaluate_batch(gpu_payload,
												h_features,
												h_labels,
												valid_rows,
												feat_dim,
												&accuracy,
												&precision,
												&recall,
												&f1_score,
												&gpu_errstr);

				elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] GPU evaluation returned rc=%d, gpu_errstr=%p", rc, (void *) gpu_errstr);

				if (rc == 0)
				{
					elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] GPU evaluation succeeded, building JSONB result");
					/* End SPI session BEFORE creating JSONB to avoid context conflicts */
					ndb_spi_stringinfo_free(eval_spi_session, &query);
					NDB_SPI_SESSION_END(eval_spi_session);

					/* Switch to old context and build JSONB directly using JSONB API */
					MemoryContextSwitchTo(oldcontext);
					{
						JsonbParseState *state = NULL;
						JsonbValue	jkey;
						JsonbValue	jval;
						JsonbValue *final_value = NULL;
						Numeric		accuracy_num, precision_num, recall_num, f1_score_num, n_samples_num;

						/* Start object */
						PG_TRY();
						{
							(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

							/* Add accuracy */
							jkey.type = jbvString;
							jkey.val.string.val = "accuracy";
							jkey.val.string.len = strlen("accuracy");
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy)));
							jval.type = jbvNumeric;
							jval.val.numeric = accuracy_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							/* Add precision */
							jkey.val.string.val = "precision";
							jkey.val.string.len = strlen("precision");
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							precision_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(precision)));
							jval.type = jbvNumeric;
							jval.val.numeric = precision_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							/* Add recall */
							jkey.val.string.val = "recall";
							jkey.val.string.len = strlen("recall");
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							recall_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(recall)));
							jval.type = jbvNumeric;
							jval.val.numeric = recall_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							/* Add f1_score */
							jkey.val.string.val = "f1_score";
							jkey.val.string.len = strlen("f1_score");
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							f1_score_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(f1_score)));
							jval.type = jbvNumeric;
							jval.val.numeric = f1_score_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							/* Add n_samples */
							jkey.val.string.val = "n_samples";
							jkey.val.string.len = strlen("n_samples");
							(void) pushJsonbValue(&state, WJB_KEY, &jkey);
							n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(valid_rows)));
							jval.type = jbvNumeric;
							jval.val.numeric = n_samples_num;
							(void) pushJsonbValue(&state, WJB_VALUE, &jval);

							/* End object */
							final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
							
							if (final_value == NULL)
							{
								elog(ERROR, "neurondb: evaluate_random_forest: pushJsonbValue(WJB_END_OBJECT) returned NULL");
							}
							
							result_jsonb = JsonbValueToJsonb(final_value);
						}
						PG_CATCH();
						{
							ErrorData *edata = CopyErrorData();
							elog(ERROR, "neurondb: evaluate_random_forest: JSONB construction failed: %s", edata->message);
							FlushErrorState();
							result_jsonb = NULL;
						}
						PG_END_TRY();
					}
					{
						Jsonb	   *temp_jsonb = result_jsonb;

						/*
						 * Copy JSONB to caller's context before SPI_finish().
						 * The JSONB is allocated in SPI context and will be
						 * invalid after SPI_finish() deletes that context.
						 */
						MemoryContextSwitchTo(oldcontext);
						result_jsonb = (Jsonb *) PG_DETOAST_DATUM_COPY((Datum) temp_jsonb);

						/*
						 * Don't free temp_jsonb - it's in SPI context and
						 * will be cleaned up by SPI_finish()
						 */
					}

					elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] cleaning up GPU evaluation resources");
					if (h_features != NULL && !h_features_freed)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] freeing h_features (success path)");
						NDB_FREE(h_features);
						h_features = NULL;
						h_features_freed = true;
					}
					if (h_labels != NULL && !h_labels_freed)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] freeing h_labels (success path)");
						NDB_FREE(h_labels);
						h_labels = NULL;
						h_labels_freed = true;
					}
					if (gpu_payload != NULL)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] freeing gpu_payload");
						NDB_FREE(gpu_payload);
						gpu_payload = NULL;
					}
					if (gpu_metrics != NULL)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] freeing gpu_metrics");
						NDB_FREE(gpu_metrics);
						gpu_metrics = NULL;
					}

					/*
					 * Don't free gpu_errstr - it's allocated by GPU function
					 * using pstrdup() and will be automatically freed when
					 * the memory context is cleaned up. Manually freeing it
					 * can cause crashes if the context is already cleaned up.
					 */
					if (gpu_errstr)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] gpu_errstr=%p (not freeing - managed by memory context)", (void *) gpu_errstr);
						gpu_errstr = NULL;	/* Clear pointer but don't free */
					}
					cleanup_done = true;

					/*
					 * Free SPI context allocations before SPI_finish().
					 * StringInfo data and other allocations in SPI context
					 * must be freed before SPI_finish() deletes the context.
					 */
					elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] freeing SPI context allocations before SPI_finish()");
					if (query.data)
					{
						NDB_FREE(query.data);
						query.data = NULL;
					}
					if (tbl_str)
					{
						NDB_FREE(tbl_str);
						tbl_str = NULL;
					}
					if (feat_str)
					{
						NDB_FREE(feat_str);
						feat_str = NULL;
					}
					if (targ_str)
					{
						NDB_FREE(targ_str);
						targ_str = NULL;
					}
					NDB_SPI_SESSION_END(eval_spi_session);
					elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] returning JSONB result");
					PG_RETURN_JSONB_P(result_jsonb);
				}
				else
				{
					/* GPU evaluation failed, fall back to CPU */
					elog(WARNING,
						 "neurondb: evaluate_random_forest_by_model_id: [DEBUG] GPU batch evaluation failed: %s, falling back to CPU",
						 gpu_errstr ? gpu_errstr : "unknown error");
					elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] cleaning up after GPU failure");
					if (h_features != NULL && !h_features_freed)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] freeing h_features (failure path)");
						NDB_FREE(h_features);
						h_features = NULL;
						h_features_freed = true;
					}
					if (h_labels != NULL && !h_labels_freed)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] freeing h_labels (failure path)");
						NDB_FREE(h_labels);
						h_labels = NULL;
						h_labels_freed = true;
					}

					/*
					 * Don't free gpu_errstr - it's allocated by GPU function
					 * using pstrdup() and will be automatically freed when
					 * the memory context is cleaned up.
					 */
					if (gpu_errstr)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] gpu_errstr=%p (not freeing in failure path - managed by memory context)", (void *) gpu_errstr);
						gpu_errstr = NULL;	/* Clear pointer but don't free */
					}
					cleanup_done = true;
				}
			}
			PG_CATCH();
			{
				/*
				 * Flush error state and safely clean up. When an exception
				 * occurs, we need to clear the error state before attempting
				 * cleanup operations, otherwise subsequent operations may
				 * fail.
				 */
				FlushErrorState();
				elog(WARNING,
					 "evaluate_random_forest_by_model_id: [DEBUG] exception during GPU evaluation, falling back to CPU");

				/* Only free if not already freed and pointers are valid */
				if (!cleanup_done)
				{
					elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] exception cleanup: cleanup_done=false");
					if (h_features != NULL && !h_features_freed)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] exception cleanup: freeing h_features, ptr=%p", (void *) h_features);
						if (h_features != NULL)
						{
							NDB_FREE(h_features);
							h_features = NULL;
						}
						h_features_freed = true;
					}
					if (h_labels != NULL && !h_labels_freed)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] exception cleanup: freeing h_labels, ptr=%p", (void *) h_labels);
						if (h_labels != NULL)
						{
							NDB_FREE(h_labels);
							h_labels = NULL;
						}
						h_labels_freed = true;
					}

					/*
					 * Never free gpu_errstr in exception handler - it's
					 * allocated by GPU function using pstrdup() and will be
					 * automatically freed when the memory context is cleaned
					 * up. This pointer was causing crashes when freed
					 * manually.
					 */
					if (gpu_errstr != NULL)
					{
						elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] exception cleanup: gpu_errstr=%p (NOT freeing - this was the problematic pointer, now handled safely)", (void *) gpu_errstr);
						gpu_errstr = NULL;	/* Clear pointer but don't free -
											 * let memory context handle it */
					}
				}
				else
				{
					elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] exception cleanup: cleanup_done=true, skipping cleanup");
				}
			}
			PG_END_TRY();
		}
#endif							/* NDB_GPU_CUDA */
	}

	/* CPU evaluation path (fallback if GPU not available or failed) */
	if (!neurondb_gpu_is_available() || gpu_payload == NULL || model != NULL)
	{
		/* CPU path - model should already be loaded */
		if (model == NULL)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_random_forest_by_model_id: model %d not found",
							model_id)));
		}
	}
#ifndef NDB_GPU_CUDA
	/* When CUDA is not available, always use CPU path */
	if (false)
	{
	}
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-label"
cpu_evaluation_path:
#pragma GCC diagnostic pop
	/* CPU evaluation path (also used as fallback for GPU models) */
	elog(DEBUG1, "evaluate_random_forest_by_model_id: [DEBUG] Entering CPU evaluation path");
	/* Use optimized batch prediction */
	{
		float	   *h_features = NULL;
		double	   *h_labels = NULL;
		int			feat_dim = 0;
		int			valid_rows = 0;

		/* Determine feature dimension from model */
		if (model != NULL)
			feat_dim = model->n_features;
		else if (is_gpu_model && gpu_payload != NULL)
		{
			const		NdbCudaRfModelHeader *gpu_hdr;

			gpu_hdr = (const NdbCudaRfModelHeader *) VARDATA(gpu_payload);
			feat_dim = gpu_hdr->feature_dim;
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_random_forest_by_model_id: could not determine feature dimension")));
		}

		if (feat_dim <= 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_random_forest_by_model_id: invalid feature dimension %d",
							feat_dim)));
		}

		/* Allocate host buffers for features and labels */
		NDB_ALLOC(h_features, float, (size_t) nvec * (size_t) feat_dim);
		NDB_CHECK_ALLOC(h_features, "h_features");
		NDB_ALLOC(h_labels, double, (size_t) nvec);
		NDB_CHECK_ALLOC(h_labels, "h_labels");

		/*
		 * Extract features and labels from SPI results - optimized batch
		 * extraction
		 */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple;
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				Vector	   *vec;
				ArrayType  *arr;
				float	   *feat_row;

				/* Safe access to SPI_tuptable - validate before access */
				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
					i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
				{
					continue;
				}
				tuple = SPI_tuptable->vals[i];
				if (tupdesc == NULL)
				{
					continue;
				}

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				/* Safe access for target - validate tupdesc has at least 2 columns */
				if (tupdesc->natts < 2)
				{
					continue;
				}
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				feat_row = h_features + (valid_rows * feat_dim);
				h_labels[valid_rows] = DatumGetFloat8(targ_datum);

				/* Extract feature vector - optimized paths */
				if (feat_is_array)
				{
					arr = DatumGetArrayTypeP(feat_datum);
					if (ARR_NDIM(arr) != 1 || ARR_DIMS(arr)[0] != feat_dim)
						continue;
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						/* Optimized: bulk conversion with loop unrolling hint */
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);
						int			j_remain = feat_dim % 4;
						int			j_end = feat_dim - j_remain;

						/*
						 * Process 4 elements at a time for better cache
						 * locality
						 */
						for (j = 0; j < j_end; j += 4)
						{
							feat_row[j] = (float) data[j];
							feat_row[j + 1] = (float) data[j + 1];
							feat_row[j + 2] = (float) data[j + 2];
							feat_row[j + 3] = (float) data[j + 3];
						}
						/* Handle remaining elements */
						for (j = j_end; j < feat_dim; j++)
							feat_row[j] = (float) data[j];
					}
					else
					{
						/* FLOAT4ARRAYOID: direct memcpy (already optimal) */
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);

						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					/* Vector type: direct memcpy (already optimal) */
					vec = DatumGetVector(feat_datum);
					if (vec->dim != feat_dim)
						continue;
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				valid_rows++;
			}
		}

		if (valid_rows == 0)
		{
			if (h_features != NULL)
			{
				NDB_FREE(h_features);
				h_features = NULL;
			}
			if (h_labels != NULL)
			{
				NDB_FREE(h_labels);
				h_labels = NULL;
			}
			if (model != NULL)
				rf_free_deserialized_model(model);
			if (gpu_payload != NULL)
			{
				NDB_FREE(gpu_payload);
				gpu_payload = NULL;
			}
			if (gpu_metrics != NULL)
			{
				NDB_FREE(gpu_metrics);
				gpu_metrics = NULL;
			}
			if (query.data != NULL)
			{
				NDB_FREE(query.data);
				query.data = NULL;
			}
			if (tbl_str != NULL)
			{
				NDB_FREE(tbl_str);
				tbl_str = NULL;
			}
			if (feat_str != NULL)
			{
				NDB_FREE(feat_str);
				feat_str = NULL;
			}
			if (targ_str != NULL)
			{
				NDB_FREE(targ_str);
				targ_str = NULL;
			}
			NDB_SPI_SESSION_END(eval_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_random_forest_by_model_id: no valid rows found"),
					 errdetail("Dataset contains %d rows, minimum required is 10", nvec),
					 errhint("Add more data rows to the evaluation table.")));
		}

		elog(DEBUG1, "evaluate_random_forest_by_model_id: before GPU model check - is_gpu_model=%d, model=%p, valid_rows=%d", 
			 is_gpu_model, (void *)model, valid_rows);

		/* For GPU models, we need to load CPU model for evaluation */
		if (is_gpu_model && model == NULL)
		{
			/* GPU model to CPU model conversion for evaluation */
			/* Extract model structure from GPU payload */
			if (gpu_payload != NULL)
			{
				const		NdbCudaRfModelHeader *gpu_hdr;

				gpu_hdr = (const NdbCudaRfModelHeader *) VARDATA(gpu_payload);
				if (gpu_hdr != NULL)
				{
					/* Try to deserialize GPU model as CPU model */
					/* GPU models have compatible structure for CPU evaluation */
					model = rf_model_deserialize(gpu_payload, NULL);
					if (model != NULL)
					{
						elog(DEBUG1,
							 "evaluate_random_forest_by_model_id: "
							 "Successfully converted GPU model to CPU for evaluation");
					}
					else
					{
						elog(WARNING,
							 "evaluate_random_forest_by_model_id: "
							 "Failed to convert GPU model to CPU, "
							 "GPU evaluation required");
						/* Continue with GPU evaluation path if available */
					}
				}
			}

			/* If conversion failed, cannot evaluate GPU models on CPU */
			if (model == NULL)
			{
				if (h_features != NULL)
				{
					NDB_FREE(h_features);
					h_features = NULL;
				}
				if (h_labels != NULL)
				{
					NDB_FREE(h_labels);
					h_labels = NULL;
				}
				if (gpu_payload != NULL)
				{
					NDB_FREE(gpu_payload);
					gpu_payload = NULL;
				}
				if (gpu_metrics != NULL)
				{
					NDB_FREE(gpu_metrics);
					gpu_metrics = NULL;
				}
				if (query.data != NULL)
				{
					NDB_FREE(query.data);
					query.data = NULL;
				}
				if (tbl_str != NULL)
				{
					NDB_FREE(tbl_str);
					tbl_str = NULL;
				}
				if (feat_str != NULL)
				{
					NDB_FREE(feat_str);
					feat_str = NULL;
				}
				if (targ_str != NULL)
				{
					NDB_FREE(targ_str);
					targ_str = NULL;
				}
				NDB_SPI_SESSION_END(eval_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("neurondb: evaluate_random_forest_by_model_id: unable to evaluate GPU model"),
						 errdetail("GPU model format is incompatible with CPU evaluation"),
						 errhint("Use GPU-enabled evaluation or retrain the model for CPU evaluation.")));
			}
		}

		elog(DEBUG1, "evaluate_random_forest_by_model_id: after GPU model check - is_gpu_model=%d, model=%p, valid_rows=%d", 
			 is_gpu_model, (void *)model, valid_rows);

		/* Use batch prediction helper - this should run for both GPU and CPU models */
		if (model == NULL)
		{
			elog(ERROR, "evaluate_random_forest_by_model_id: model is NULL before rf_predict_batch");
		}
		if (model->n_classes <= 0)
		{
			elog(WARNING, "evaluate_random_forest_by_model_id: model->n_classes=%d is invalid, predictions may be incorrect", model->n_classes);
		}
		if (model->tree_count <= 0 || model->trees == NULL)
		{
			elog(WARNING, "evaluate_random_forest_by_model_id: model has no trees (tree_count=%d, trees=%p), predictions will use majority_value", 
				 model->tree_count, (void *)model->trees);
		}
		if (valid_rows <= 0)
		{
			elog(ERROR, "evaluate_random_forest_by_model_id: valid_rows=%d is invalid", valid_rows);
		}
		elog(DEBUG1, "evaluate_random_forest_by_model_id: calling rf_predict_batch with valid_rows=%d, feat_dim=%d, model->n_classes=%d, model->tree_count=%d", 
			 valid_rows, feat_dim, model ? model->n_classes : -1, model ? model->tree_count : -1);
		rf_predict_batch(model,
						 h_features,
						 h_labels,
						 valid_rows,
						 feat_dim,
						 &tp,
						 &tn,
						 &fp,
						 &fn);
		elog(DEBUG1, "evaluate_random_forest_by_model_id: rf_predict_batch returned tp=%d, tn=%d, fp=%d, fn=%d", tp, tn, fp, fn);

		/* Compute metrics */
		if (valid_rows > 0)
		{
			accuracy = (double) (tp + tn) / (double) valid_rows;

			if ((tp + fp) > 0)
				precision = (double) tp / (double) (tp + fp);
			else
				precision = 0.0;

			if ((tp + fn) > 0)
				recall = (double) tp / (double) (tp + fn);
			else
				recall = 0.0;

			if ((precision + recall) > 0.0)
				f1_score = 2.0 * (precision * recall) / (precision + recall);
			else
				f1_score = 0.0;
		}

		/* Cleanup */
		if (h_features != NULL)
		{
			NDB_FREE(h_features);
			h_features = NULL;
		}
		if (h_labels != NULL)
		{
			NDB_FREE(h_labels);
			h_labels = NULL;
		}
		if (model != NULL)
			rf_free_deserialized_model(model);
		if (gpu_payload != NULL)
		{
			NDB_FREE(gpu_payload);
			gpu_payload = NULL;
		}
		if (gpu_metrics != NULL)
		{
			NDB_FREE(gpu_metrics);
			gpu_metrics = NULL;
		}
	}

		/* End SPI session BEFORE creating JSONB to avoid context conflicts */
		ndb_spi_stringinfo_free(eval_spi_session, &query);
		NDB_SPI_SESSION_END(eval_spi_session);

		/* Switch to old context and build JSONB directly using JSONB API */
		MemoryContextSwitchTo(oldcontext);
		{
			JsonbParseState *state = NULL;
			JsonbValue	jkey;
			JsonbValue	jval;
			JsonbValue *final_value = NULL;
			Numeric		accuracy_num, precision_num, recall_num, f1_score_num, n_samples_num;

			/* Start object */
			PG_TRY();
			{
				(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

				/* Add accuracy */
				jkey.type = jbvString;
				jkey.val.string.val = "accuracy";
				jkey.val.string.len = strlen("accuracy");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy)));
				jval.type = jbvNumeric;
				jval.val.numeric = accuracy_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add precision */
				jkey.val.string.val = "precision";
				jkey.val.string.len = strlen("precision");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				precision_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(precision)));
				jval.type = jbvNumeric;
				jval.val.numeric = precision_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add recall */
				jkey.val.string.val = "recall";
				jkey.val.string.len = strlen("recall");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				recall_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(recall)));
				jval.type = jbvNumeric;
				jval.val.numeric = recall_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add f1_score */
				jkey.val.string.val = "f1_score";
				jkey.val.string.len = strlen("f1_score");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				f1_score_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(f1_score)));
				jval.type = jbvNumeric;
				jval.val.numeric = f1_score_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* Add n_samples */
				jkey.val.string.val = "n_samples";
				jkey.val.string.len = strlen("n_samples");
				(void) pushJsonbValue(&state, WJB_KEY, &jkey);
				n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(nvec)));
				jval.type = jbvNumeric;
				jval.val.numeric = n_samples_num;
				(void) pushJsonbValue(&state, WJB_VALUE, &jval);

				/* End object */
				final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
				
				if (final_value == NULL)
				{
					elog(ERROR, "neurondb: evaluate_random_forest: pushJsonbValue(WJB_END_OBJECT) returned NULL");
				}
				
				result_jsonb = JsonbValueToJsonb(final_value);
			}
			PG_CATCH();
			{
				ErrorData *edata = CopyErrorData();
				elog(ERROR, "neurondb: evaluate_random_forest: JSONB construction failed: %s", edata->message);
				FlushErrorState();
				result_jsonb = NULL;
			}
			PG_END_TRY();
		}

		if (result_jsonb == NULL)
		{
			if (tbl_str)
			{
				NDB_FREE(tbl_str);
				tbl_str = NULL;
			}
			if (feat_str)
			{
				NDB_FREE(feat_str);
				feat_str = NULL;
			}
			if (targ_str)
			{
				NDB_FREE(targ_str);
				targ_str = NULL;
			}
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: evaluate_random_forest_by_model_id: JSONB result is NULL")));
		}

		/* Cleanup */
		if (tbl_str)
		{
			NDB_FREE(tbl_str);
			tbl_str = NULL;
		}
		if (feat_str)
		{
			NDB_FREE(feat_str);
			feat_str = NULL;
		}
		if (targ_str)
		{
			NDB_FREE(targ_str);
			targ_str = NULL;
		}

		PG_RETURN_JSONB_P(result_jsonb);
	}

static void
rf_serialize_tree(StringInfo buf, const GTree * tree)
{
	const		GTreeNode *nodes;
	int			i;

	if (tree == NULL)
	{
		pq_sendbyte(buf, 0);
		return;
	}

	pq_sendbyte(buf, 1);
	pq_sendint32(buf, tree->root);
	pq_sendint32(buf, tree->max_depth);
	pq_sendint32(buf, tree->count);

	nodes = gtree_nodes(tree);
	for (i = 0; i < tree->count; i++)
	{
		pq_sendint32(buf, nodes[i].feature_idx);
		pq_sendfloat8(buf, nodes[i].threshold);
		pq_sendint32(buf, nodes[i].left);
		pq_sendint32(buf, nodes[i].right);
		pq_sendbyte(buf, nodes[i].is_leaf ? 1 : 0);
		pq_sendfloat8(buf, nodes[i].value);
	}
}

static GTree *
rf_deserialize_tree(StringInfo buf)
{
	int			flag = pq_getmsgbyte(buf);
	int			count;
	int			i;
	int			root;
	int			max_depth;
	GTree	   *tree;
	MemoryContext oldctx;

	if (flag == 0)
		return NULL;

	root = pq_getmsgint(buf, 4);
	max_depth = pq_getmsgint(buf, 4);
	count = pq_getmsgint(buf, 4);

	tree = gtree_create("rf_model_tree", Max(count, 4));
	oldctx = MemoryContextSwitchTo(tree->ctx);

	if (tree->nodes != NULL)
		NDB_FREE(tree->nodes);

	if (count > 0)
	{
		NDB_ALLOC(tree->nodes, GTreeNode, count);
		NDB_CHECK_ALLOC(tree, "tree");
		for (i = 0; i < count; i++)
		{
			tree->nodes[i].feature_idx = pq_getmsgint(buf, 4);
			tree->nodes[i].threshold = pq_getmsgfloat8(buf);
			tree->nodes[i].left = pq_getmsgint(buf, 4);
			tree->nodes[i].right = pq_getmsgint(buf, 4);
			tree->nodes[i].is_leaf = pq_getmsgbyte(buf);
			tree->nodes[i].value = pq_getmsgfloat8(buf);
		}
	}
	else
		tree->nodes = NULL;

	tree->capacity = count;
	tree->count = count;
	tree->root = root;
	tree->max_depth = max_depth;

	MemoryContextSwitchTo(oldctx);
	return tree;
}

static void
rf_write_int_array(StringInfo buf, const int *values, int count)
{
	int			i;

	if (values == NULL || count <= 0)
	{
		pq_sendbyte(buf, 0);
		return;
	}

	pq_sendbyte(buf, 1);
	pq_sendint32(buf, count);
	for (i = 0; i < count; i++)
		pq_sendint32(buf, values[i]);
}

static void
rf_write_double_array(StringInfo buf, const double *values, int count)
{
	int			i;

	if (values == NULL || count <= 0)
	{
		pq_sendbyte(buf, 0);
		return;
	}

	pq_sendbyte(buf, 1);
	pq_sendint32(buf, count);
	for (i = 0; i < count; i++)
		pq_sendfloat8(buf, values[i]);
}

static int *
rf_read_int_array(StringInfo buf, int expected_count)
{
	int			flag = pq_getmsgbyte(buf);
	int			len;
	int		   *result;
	int			i;
	size_t		alloc_size;

	if (flag == 0)
		return NULL;

	len = pq_getmsgint(buf, 4);

	/* Validate length before allocation */
	if (len < 0 || len > 1000000)
	{
		elog(ERROR,
			 "random_forest: invalid int array length %d",
			 len);
		return NULL;
	}

	if (expected_count >= 0 && len != expected_count)
	{
		elog(ERROR,
			 "random_forest: unexpected int array length %d (expected %d)",
			 len,
			 expected_count);
		return NULL;
	}

	/* Check buffer bounds */
	if (buf->cursor + (len * sizeof(int)) > buf->len)
	{
		elog(ERROR,
			 "random_forest: buffer overrun in rf_read_int_array: cursor=%d, len=%d, needed=%zu",
			 buf->cursor, buf->len, (size_t) len * sizeof(int));
		return NULL;
	}

	alloc_size = sizeof(int) * (size_t) len;
	if (alloc_size > MaxAllocSize)
	{
		elog(ERROR,
			 "random_forest: int array allocation size %zu exceeds MaxAllocSize",
			 alloc_size);
		return NULL;
	}

	NDB_ALLOC(result, int, len);
	if (result == NULL)
	{
		elog(ERROR, "random_forest: palloc failed for int array of size %zu", alloc_size);
		return NULL;
	}

	for (i = 0; i < len; i++)
	{
		/* Check buffer bounds before each read */
		if (buf->cursor + sizeof(int) > buf->len)
		{
			elog(ERROR,
				 "random_forest: buffer overrun reading int array element %d",
				 i);
			NDB_FREE(result);
			return NULL;
		}
		result[i] = pq_getmsgint(buf, 4);
	}

	return result;
}

static double *
rf_read_double_array(StringInfo buf, int expected_count)
{
	int			flag;
	int			len;
	double	   *result;
	int			i;
	size_t		alloc_size;

	/* Check buffer has enough space for flag (1 byte) */
	if (buf->cursor + 1 > buf->len)
	{
		elog(WARNING,
			 "random_forest: buffer overrun reading double array flag: cursor=%d, len=%d, need 1 byte",
			 buf->cursor, buf->len);
		return NULL;
	}

	flag = pq_getmsgbyte(buf);

	if (flag == 0)
	{
		/* Flag is 0, array not present - cursor was advanced by 1 byte */
		return NULL;
	}

	/* Flag is non-zero, array should be present - read length */
	/* Check buffer has enough space for length (4 bytes) */
	if (buf->cursor + 4 > buf->len)
	{
		elog(WARNING,
			 "random_forest: buffer overrun reading double array length: cursor=%d, len=%d, need 4 bytes (flag was %d)",
			 buf->cursor, buf->len, flag);
		/* Cursor was already advanced by flag (1 byte), buffer is corrupted */
		return NULL;
	}

	len = pq_getmsgint(buf, 4);

	/* Validate length before allocation */
	if (len < 0 || len > 1000000)
	{
		elog(WARNING,
			 "random_forest: invalid double array length %d (cursor=%d/%d)",
			 len, buf->cursor, buf->len);

		/*
		 * Cursor was already advanced by flag (1 byte) + length (4 bytes) = 5
		 * bytes
		 */
		/* Cannot rollback, so buffer is corrupted */
		return NULL;
	}

	if (expected_count >= 0 && len != expected_count)
	{
		elog(ERROR,
			 "random_forest: unexpected double array length %d "
			 "(expected %d, cursor=%d/%d)",
			 len,
			 expected_count,
			 buf->cursor, buf->len);
		/* Cursor was already advanced, buffer is corrupted */
		return NULL;
	}

	/* Check buffer bounds - ensure we have enough data */
	if (buf->cursor + (len * sizeof(double)) > buf->len)
	{
		elog(ERROR,
			 "random_forest: buffer overrun in rf_read_double_array: cursor=%d, len=%d, needed=%zu, available=%d",
			 buf->cursor, buf->len, (size_t) len * sizeof(double), buf->len - buf->cursor);
		/* Cursor was already advanced, buffer is corrupted */
		return NULL;
	}

	alloc_size = sizeof(double) * (size_t) len;
	if (alloc_size > MaxAllocSize)
	{
		elog(ERROR,
			 "random_forest: double array allocation size %zu exceeds MaxAllocSize",
			 alloc_size);
		return NULL;
	}

	NDB_ALLOC(result, double, len);
	if (result == NULL)
	{
		elog(ERROR, "random_forest: palloc failed for double array of size %zu", alloc_size);
		return NULL;
	}

	for (i = 0; i < len; i++)
	{
		/* Check buffer bounds before each read */
		if (buf->cursor + sizeof(double) > buf->len)
		{
			elog(ERROR,
				 "random_forest: buffer overrun reading double array element %d",
				 i);
			NDB_FREE(result);
			return NULL;
		}
		result[i] = pq_getmsgfloat8(buf);
	}

	return result;
}

static bytea *
rf_model_serialize(const RFModel * model, uint8 training_backend)
{
	StringInfoData buf;
	int			i;

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: rf_model_serialize: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	pq_begintypsend(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) */
	pq_sendbyte(&buf, training_backend);

	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendint32(&buf, model->n_classes);

	pq_sendfloat8(&buf, model->majority_value);
	pq_sendfloat8(&buf, model->majority_fraction);
	pq_sendfloat8(&buf, model->gini_impurity);
	pq_sendfloat8(&buf, model->label_entropy);
	pq_sendfloat8(&buf, model->max_deviation);
	pq_sendfloat8(&buf, model->max_split_deviation);
	pq_sendint32(&buf, model->split_feature);
	pq_sendfloat8(&buf, model->split_threshold);
	pq_sendfloat8(&buf, model->second_value);
	pq_sendfloat8(&buf, model->second_fraction);
	pq_sendfloat8(&buf, model->left_branch_value);
	pq_sendfloat8(&buf, model->left_branch_fraction);
	pq_sendfloat8(&buf, model->right_branch_value);
	pq_sendfloat8(&buf, model->right_branch_fraction);
	pq_sendint32(&buf, model->feature_limit);
	pq_sendfloat8(&buf, model->oob_accuracy);

	rf_write_int_array(&buf, model->class_counts, model->n_classes);
	rf_write_double_array(&buf, model->feature_means, model->n_features);
	rf_write_double_array(
						  &buf, model->feature_variances, model->n_features);
	rf_write_double_array(
						  &buf, model->feature_importance, model->n_features);
	rf_write_double_array(
						  &buf, model->left_branch_means, model->feature_limit);
	rf_write_double_array(
						  &buf, model->right_branch_means, model->feature_limit);

	pq_sendint32(&buf, model->tree_count);
	for (i = 0; i < model->tree_count; i++)
		rf_serialize_tree(&buf, model->trees[i]);

	rf_serialize_tree(&buf, model->tree);

	rf_write_double_array(&buf, model->tree_majority, model->tree_count);
	rf_write_double_array(
						  &buf, model->tree_majority_fraction, model->tree_count);
	rf_write_double_array(&buf, model->tree_second, model->tree_count);
	rf_write_double_array(
						  &buf, model->tree_second_fraction, model->tree_count);
	rf_write_double_array(
						  &buf, model->tree_oob_accuracy, model->tree_count);

	return pq_endtypsend(&buf);
}

static RFModel *
rf_model_deserialize(const bytea * data, uint8 *training_backend_out)
{
	StringInfoData buf;
	RFModel    *model = NULL;
	int			i;
	uint8		training_backend = 0;

	elog(DEBUG1, "neurondb: rf_model_deserialize() called, data=%p, CurrentMemoryContext=%p",
		 (void *) data, (void *) CurrentMemoryContext);

	if (data == NULL)
	{
		elog(WARNING, "rf_model_deserialize: data is NULL");
		return NULL;
	}

	if (VARSIZE(data) < VARHDRSZ)
	{
		elog(WARNING, "rf_model_deserialize: invalid data size %d", VARSIZE(data));
		return NULL;
	}

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.cursor = 0;
	buf.maxlen = buf.len;

	if (buf.len < 100)
	{
		elog(WARNING, "rf_model_deserialize: data too small (%d bytes)", buf.len);
		return NULL;
	}

	elog(DEBUG1, "neurondb: rf_model_deserialize() entering PG_TRY block");
	PG_TRY();
	{
		elog(DEBUG1, "neurondb: rf_model_deserialize() allocating RFModel structure");
		NDB_ALLOC(model, RFModel, 1);
		if (model == NULL)
		{
			elog(ERROR, "rf_model_deserialize: palloc0 failed");
			/* Will be caught by PG_CATCH */
		}
		else
		{
			elog(DEBUG1, "neurondb: rf_model_deserialize() allocated model at %p", (void *) model);
			
			/* Read training_backend first */
			training_backend = (uint8) pq_getmsgbyte(&buf);
			if (training_backend_out != NULL)
				*training_backend_out = training_backend;

			model->n_features = pq_getmsgint(&buf, 4);
			model->n_samples = pq_getmsgint(&buf, 4);
			model->n_classes = pq_getmsgint(&buf, 4);

			/* Validate basic parameters */
			if (model->n_features <= 0 || model->n_features > 1000000 ||
				model->n_classes <= 0 || model->n_classes > 10000 ||
				model->n_samples < 0 || model->n_samples > 1000000000)
			{
				elog(WARNING, "rf_model_deserialize: invalid model parameters (n_features=%d, n_classes=%d, n_samples=%d)",
					 model->n_features, model->n_classes, model->n_samples);
				NDB_FREE(model);
				model = NULL;
			}
		}

		if (model == NULL)
		{
			/* Error already logged, will be caught by PG_CATCH */
		}
		else
		{
			model->majority_value = pq_getmsgfloat8(&buf);
			model->majority_fraction = pq_getmsgfloat8(&buf);
			model->gini_impurity = pq_getmsgfloat8(&buf);
			model->label_entropy = pq_getmsgfloat8(&buf);
			model->max_deviation = pq_getmsgfloat8(&buf);
			model->max_split_deviation = pq_getmsgfloat8(&buf);
			model->split_feature = pq_getmsgint(&buf, 4);
			model->split_threshold = pq_getmsgfloat8(&buf);
			model->second_value = pq_getmsgfloat8(&buf);
			model->second_fraction = pq_getmsgfloat8(&buf);
			model->left_branch_value = pq_getmsgfloat8(&buf);
			model->left_branch_fraction = pq_getmsgfloat8(&buf);
			model->right_branch_value = pq_getmsgfloat8(&buf);
			model->right_branch_fraction = pq_getmsgfloat8(&buf);
			model->feature_limit = pq_getmsgint(&buf, 4);
			model->oob_accuracy = pq_getmsgfloat8(&buf);

			/* Validate feature_limit */
			if (model->feature_limit < 0 || model->feature_limit > model->n_features)
			{
				elog(WARNING, "rf_model_deserialize: invalid feature_limit %d (n_features=%d)", model->feature_limit, model->n_features);
				NDB_FREE(model);
				model = NULL;
			}
			else
			{
				/* Read arrays with validation */
				model->class_counts = rf_read_int_array(&buf, model->n_classes);
				if (model->class_counts == NULL && model->n_classes > 0)
				{
					elog(WARNING, "rf_model_deserialize: failed to read class_counts");
					NDB_FREE(model);
					model = NULL;
				}
			}
		}

		if (model == NULL)
		{
			/* Error already logged, will be caught by PG_CATCH */
		}
		else
		{
			model->feature_means = rf_read_double_array(&buf, model->n_features);
			if (model->feature_means == NULL && model->n_features > 0)
			{
				elog(WARNING, "rf_model_deserialize: failed to read feature_means");
				if (model->class_counts)
					NDB_FREE(model->class_counts);
				NDB_FREE(model);
				model = NULL;
			}
			else
			{
				model->feature_variances = rf_read_double_array(&buf, model->n_features);
				if (model->feature_variances == NULL && model->n_features > 0)
				{
					elog(WARNING, "rf_model_deserialize: failed to read feature_variances");
					if (model->class_counts)
						NDB_FREE(model->class_counts);
					if (model->feature_means)
						NDB_FREE(model->feature_means);
					NDB_FREE(model);
					model = NULL;
				}
				else
				{
					/*
					 * feature_importance is optional - may be NULL if not
					 * stored
					 */
					/*
					 * Save buffer cursor before reading to detect if read
					 * failed after advancing cursor
					 */
					int			saved_cursor_pos = buf.cursor;

					elog(DEBUG1, "rf_model_deserialize: reading feature_importance, cursor=%d/%d, n_features=%d", buf.cursor, buf.len, model->n_features);

					/* Read feature_importance array */
					model->feature_importance = rf_read_double_array(&buf, model->n_features);

					elog(DEBUG1, "rf_model_deserialize: after reading feature_importance, cursor=%d/%d, result=%p, saved_cursor_pos=%d", buf.cursor, buf.len, (void *) model->feature_importance, saved_cursor_pos);

					if (model != NULL && model->feature_importance == NULL && model->n_features > 0)
					{
						int			cursor_advance = buf.cursor - saved_cursor_pos;

						elog(DEBUG1, "rf_model_deserialize: checking cursor advancement: saved_cursor_pos=%d, cursor_after=%d, advance=%d", saved_cursor_pos, buf.cursor, cursor_advance);

						/*
						 * Check cursor advancement: - If cursor advanced by
						 * exactly 1 byte, flag was 0 (OK, feature_importance
						 * not stored) - If cursor advanced by more than 1
						 * byte but result is NULL, flag was non-zero but read
						 * failed (ERROR) - If cursor did not advance,
						 * something is wrong (ERROR)
						 */
						if (cursor_advance == 1)
						{
							/*
							 * Flag was 0, feature_importance not stored -
							 * this is OK
							 */
							elog(DEBUG1, "rf_model_deserialize: feature_importance not stored (flag=0, cursor advanced by 1 byte), continuing");
							model->feature_importance = NULL;
						}
						else if (cursor_advance > 1)
						{
							/*
							 * Flag was non-zero, cursor advanced by more than
							 * 1 byte, but result is NULL - buffer corrupted
							 */
							elog(ERROR, "rf_model_deserialize: failed to read feature_importance after advancing buffer cursor (buffer corrupted: cursor advanced from %d to %d by %d bytes, buffer len=%d, flag was non-zero but read failed)",
								 saved_cursor_pos, buf.cursor, cursor_advance, buf.len);
							/* Cleanup and abort */
							if (model->class_counts)
								NDB_FREE(model->class_counts);
							if (model->feature_means)
								NDB_FREE(model->feature_means);
							if (model->feature_variances)
								NDB_FREE(model->feature_variances);
							NDB_FREE(model);
							model = NULL;
						}
						else if (cursor_advance < 0)
						{
							/* Cursor went backwards - impossible */
							elog(ERROR, "rf_model_deserialize: cursor went backwards when reading feature_importance (saved_cursor_pos=%d, cursor_after=%d, buffer len=%d)",
								 saved_cursor_pos, buf.cursor, buf.len);
							if (model->class_counts)
								NDB_FREE(model->class_counts);
							if (model->feature_means)
								NDB_FREE(model->feature_means);
							if (model->feature_variances)
								NDB_FREE(model->feature_variances);
							NDB_FREE(model);
							model = NULL;
						}
						else
						{
							/*
							 * cursor_advance == 0 - cursor did not advance,
							 * which should not happen
							 */
							elog(ERROR, "rf_model_deserialize: cursor did not advance when reading feature_importance (cursor=%d, buffer len=%d)",
								 buf.cursor, buf.len);
							if (model->class_counts)
								NDB_FREE(model->class_counts);
							if (model->feature_means)
								NDB_FREE(model->feature_means);
							if (model->feature_variances)
								NDB_FREE(model->feature_variances);
							NDB_FREE(model);
							model = NULL;
						}
					}

					/* Only continue if model is still valid */
					if (model != NULL)
					{
						/* Continue with deserialization */
						elog(DEBUG1, "rf_model_deserialize: reading left_branch_means, cursor=%d/%d, feature_limit=%d", buf.cursor, buf.len, model->feature_limit);
						model->left_branch_means = rf_read_double_array(&buf, model->feature_limit);

						if (model != NULL && model->left_branch_means == NULL && model->feature_limit > 0)
						{
							elog(ERROR, "rf_model_deserialize: failed to read left_branch_means (required field, cursor=%d/%d, feature_limit=%d)",
								 buf.cursor, buf.len, model->feature_limit);
							if (model->class_counts)
								NDB_FREE(model->class_counts);
							if (model->feature_means)
								NDB_FREE(model->feature_means);
							if (model->feature_variances)
								NDB_FREE(model->feature_variances);
							if (model->feature_importance)
								NDB_FREE(model->feature_importance);
							NDB_FREE(model);
							model = NULL;
						}
						else if (model != NULL)
						{
							elog(DEBUG1, "rf_model_deserialize: reading right_branch_means, cursor=%d/%d, feature_limit=%d", buf.cursor, buf.len, model->feature_limit);
							model->right_branch_means = rf_read_double_array(&buf, model->feature_limit);

							if (model != NULL && model->right_branch_means == NULL && model->feature_limit > 0)
							{
								elog(ERROR, "rf_model_deserialize: failed to read right_branch_means (required field, cursor=%d/%d, feature_limit=%d)",
									 buf.cursor, buf.len, model->feature_limit);
								if (model->class_counts)
									NDB_FREE(model->class_counts);
								if (model->feature_means)
									NDB_FREE(model->feature_means);
								if (model->feature_variances)
									NDB_FREE(model->feature_variances);
								if (model->feature_importance)
									NDB_FREE(model->feature_importance);
								if (model->left_branch_means)
									NDB_FREE(model->left_branch_means);
								NDB_FREE(model);
								model = NULL;
							}
							else if (model != NULL)
							{
								model->tree_count = pq_getmsgint(&buf, 4);
								if (model->tree_count < 0 || model->tree_count > 10000)
								{
									elog(WARNING, "rf_model_deserialize: invalid tree_count %d", model->tree_count);
									if (model->class_counts)
										NDB_FREE(model->class_counts);
									if (model->feature_means)
										NDB_FREE(model->feature_means);
									if (model->feature_variances)
										NDB_FREE(model->feature_variances);
									if (model->feature_importance)
										NDB_FREE(model->feature_importance);
									if (model->left_branch_means)
										NDB_FREE(model->left_branch_means);
									if (model->right_branch_means)
										NDB_FREE(model->right_branch_means);
									NDB_FREE(model);
									model = NULL;
								}
								else if (model->tree_count > 0)
								{
									NDB_ALLOC(model->trees, GTree *, model->tree_count);
									if (model->trees == NULL)
									{
										elog(WARNING, "rf_model_deserialize: palloc failed for trees array");
										if (model->class_counts)
											NDB_FREE(model->class_counts);
										if (model->feature_means)
											NDB_FREE(model->feature_means);
										if (model->feature_variances)
											NDB_FREE(model->feature_variances);
										if (model->feature_importance)
											NDB_FREE(model->feature_importance);
										if (model->left_branch_means)
											NDB_FREE(model->left_branch_means);
										if (model->right_branch_means)
											NDB_FREE(model->right_branch_means);
										NDB_FREE(model);
										model = NULL;
									}
									else
									{
										for (i = 0; i < model->tree_count && model != NULL; i++)
										{
											model->trees[i] = rf_deserialize_tree(&buf);
											if (model->trees[i] == NULL)
											{
												elog(WARNING, "rf_model_deserialize: failed to deserialize tree %d", i);

												/*
												 * Free already deserialized
												 * trees
												 */
												for (i--; i >= 0; i--)
												{
													if (model->trees[i] != NULL)
														gtree_free(model->trees[i]);
												}
												NDB_FREE(model->trees);
												if (model->class_counts)
													NDB_FREE(model->class_counts);
												if (model->feature_means)
													NDB_FREE(model->feature_means);
												if (model->feature_variances)
													NDB_FREE(model->feature_variances);
												if (model->feature_importance)
													NDB_FREE(model->feature_importance);
												if (model->left_branch_means)
													NDB_FREE(model->left_branch_means);
												if (model->right_branch_means)
													NDB_FREE(model->right_branch_means);
												NDB_FREE(model);
												model = NULL;
											}
										}
									}
								}
								else
								{
									model->trees = NULL;
								}

								if (model != NULL)
								{
									model->tree = rf_deserialize_tree(&buf);
									if (model->tree == NULL)
									{
										elog(WARNING, "rf_model_deserialize: failed to deserialize main tree");
										if (model->trees != NULL)
										{
											for (i = 0; i < model->tree_count; i++)
											{
												if (model->trees[i] != NULL)
													gtree_free(model->trees[i]);
											}
											NDB_FREE(model->trees);
										}
										if (model->class_counts)
											NDB_FREE(model->class_counts);
										if (model->feature_means)
											NDB_FREE(model->feature_means);
										if (model->feature_variances)
											NDB_FREE(model->feature_variances);
										if (model->feature_importance)
											NDB_FREE(model->feature_importance);
										if (model->left_branch_means)
											NDB_FREE(model->left_branch_means);
										if (model->right_branch_means)
											NDB_FREE(model->right_branch_means);
										NDB_FREE(model);
										model = NULL;
									}
									else
									{
										model->tree_majority = rf_read_double_array(&buf, model->tree_count);
										model->tree_majority_fraction = rf_read_double_array(&buf, model->tree_count);
										model->tree_second = rf_read_double_array(&buf, model->tree_count);
										model->tree_second_fraction = rf_read_double_array(&buf, model->tree_count);
										model->tree_oob_accuracy = rf_read_double_array(&buf, model->tree_count);

										/* Final validation */
										if (buf.cursor > buf.len)
										{
											elog(WARNING, "rf_model_deserialize: buffer overrun (cursor=%d, len=%d)", buf.cursor, buf.len);
											rf_free_deserialized_model(model);
											model = NULL;
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	PG_CATCH();
	{
		elog(WARNING, "rf_model_deserialize: exception during deserialization");
		if (model != NULL)
		{
			rf_free_deserialized_model(model);
			model = NULL;
		}
		FlushErrorState();
	}
	PG_END_TRY();

	return model;
}

static void
rf_free_deserialized_model(RFModel * model)
{
	if (model == NULL)
		return;

	if (model->class_counts)
		NDB_FREE(model->class_counts);
	if (model->feature_means)
		NDB_FREE(model->feature_means);
	if (model->feature_variances)
		NDB_FREE(model->feature_variances);
	if (model->feature_importance)
		NDB_FREE(model->feature_importance);
	if (model->left_branch_means)
		NDB_FREE(model->left_branch_means);
	if (model->right_branch_means)
		NDB_FREE(model->right_branch_means);
	if (model->tree_majority)
		NDB_FREE(model->tree_majority);
	if (model->tree_majority_fraction)
		NDB_FREE(model->tree_majority_fraction);
	if (model->tree_second)
		NDB_FREE(model->tree_second);
	if (model->tree_second_fraction)
		NDB_FREE(model->tree_second_fraction);
	if (model->tree_oob_accuracy)
		NDB_FREE(model->tree_oob_accuracy);
	if (model->trees)
		NDB_FREE(model->trees);

	NDB_FREE(model);
}

static void
rf_dataset_init(RFDataset * dataset)
{
	if (dataset == NULL)
		return;
	dataset->features = NULL;
	dataset->labels = NULL;
	dataset->n_samples = 0;
	dataset->feature_dim = 0;
}

static void
rf_dataset_free(RFDataset * dataset)
{
	if (dataset == NULL)
		return;
	if (dataset->features != NULL)
		NDB_FREE(dataset->features);
	if (dataset->labels != NULL)
		NDB_FREE(dataset->labels);
	rf_dataset_init(dataset);
}

static void
rf_dataset_load(const char *quoted_tbl,
				const char *quoted_feat,
				const char *quoted_label,
				RFDataset * dataset,
				StringInfo query)
{
	int			feature_dim = 0;
	int			n_samples = 0;
	int			i;

	if (dataset == NULL || query == NULL)
		elog(ERROR, "random_forest: invalid dataset load arguments");

	rf_dataset_free(dataset);

	/* Try to get feature dimension - handle both vector and array types */
	/* Use safe free/reinit to handle potential memory context changes */
	NDB_FREE(query->data);
	initStringInfo(query);
	appendStringInfo(query,
					 "SELECT %s FROM %s WHERE %s IS NOT NULL LIMIT 1",
					 quoted_feat,
					 quoted_tbl,
					 quoted_feat);

	if (ndb_spi_execute_safe(query->data, true, 1) == SPI_OK_SELECT
		&& SPI_processed > 0)
	{
		HeapTuple	tup;
		TupleDesc	tupdesc;

		Datum		feat_datum;
		bool		feat_null;
		Oid			feat_type;

		NDB_CHECK_SPI_TUPTABLE();
		/* Safe access for complex types - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
			SPI_tuptable->vals[0] == NULL || SPI_tuptable->tupdesc == NULL)
		{
			/* Cannot determine feature dimension */
			feat_null = true;
		}
		else
		{
			tup = SPI_tuptable->vals[0];
			tupdesc = SPI_tuptable->tupdesc;
			feat_datum = SPI_getbinval(tup, tupdesc, 1, &feat_null);
		}
		if (!feat_null)
		{
			feat_type = SPI_gettypeid(tupdesc, 1);

			/* Check if it's an array type (double precision[]) */
			if (feat_type == FLOAT8ARRAYOID)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

				if (arr != NULL)
					feature_dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
			}
			/* Check if it's a float4 array */
			else if (feat_type == FLOAT4ARRAYOID)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

				if (arr != NULL)
					feature_dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
			}
			/* Try Vector type (check by attempting to cast) */
			else
			{
				Vector	   *vec = DatumGetVector(feat_datum);

				if (vec != NULL && vec->dim > 0)
					feature_dim = vec->dim;
			}
		}
	}

	dataset->feature_dim = feature_dim;

	/* Use safe free/reinit to handle potential memory context changes */
	NDB_FREE(query->data);
	initStringInfo(query);
	appendStringInfo(query,
					 "SELECT %s, (%s)::float8 FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quoted_feat,
					 quoted_label,
					 quoted_tbl,
					 quoted_feat,
					 quoted_label);

	{
		int			ret = ndb_spi_execute_safe(query->data, true, 0);
		
		if (ret != SPI_OK_SELECT)
		{
			NDB_CHECK_SPI_TUPTABLE();
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("random_forest: failed to fetch training data"),
					 errdetail("SPI execution returned code %d (expected %d), query: %s", ret, SPI_OK_SELECT, query->data),
					 errhint("Verify the table '%s' exists and contains valid feature and label columns.", quoted_tbl)));
		}
	}

	n_samples = SPI_processed;
	dataset->n_samples = n_samples;

	if (n_samples <= 0)
		return;

	/* Defensive check: prevent excessive memory allocation */
	if (feature_dim > 100000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("random_forest: feature dimension too large: feature_dim=%d",
						feature_dim),
				 errhint("Consider reducing feature dimension")));
	}

	/* Check for potential integer overflow in allocation size */
	if ((size_t) n_samples * (size_t) feature_dim > (SIZE_MAX / sizeof(float)))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("random_forest: allocation size would overflow: n_samples=%d, feature_dim=%d",
						n_samples, feature_dim)));
	}

	/* Check for PostgreSQL allocation limit (typically ~1GB) */
	{
		size_t		alloc_size = sizeof(float) * (size_t) feature_dim * (size_t) n_samples;
		size_t		max_alloc = 1024UL * 1024UL * 1024UL;	/* 1GB limit */

		if (alloc_size > max_alloc)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("random_forest: dataset too large for single allocation: n_samples=%d, feature_dim=%d, size=%.2f GB",
							n_samples, feature_dim, (double) alloc_size / (1024.0 * 1024.0 * 1024.0)),
					 errhint("Consider using a smaller dataset, increasing work_mem, or using batch training")));
		}
	}

	NDB_ALLOC(dataset->labels, double, (size_t) n_samples);
	NDB_CHECK_ALLOC(dataset->labels, "dataset->labels");
	if (feature_dim > 0)
	{
		NDB_ALLOC(dataset->features, float, (size_t) feature_dim * (size_t) n_samples);
		NDB_CHECK_ALLOC(dataset->features, "dataset->features");
	}

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tup = SPI_tuptable->vals[i];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		Datum		label_datum;
		bool		feat_null;
		bool		label_null;

		feat_datum = SPI_getbinval(tup, tupdesc, 1, &feat_null);
		label_datum = SPI_getbinval(tup, tupdesc, 2, &label_null);

		if (feat_null || label_null)
		{
			dataset->labels[i] = NAN;
			continue;
		}

		dataset->labels[i] = DatumGetFloat8(label_datum);

		if (dataset->features != NULL)
		{
			Oid			feat_type = SPI_gettypeid(tupdesc, 1);

			/* Handle array types (double precision[] or float[]) */
			if (feat_type == FLOAT8ARRAYOID || feat_type == FLOAT4ARRAYOID)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);
				float	   *dest_row = dataset->features + (i * feature_dim);
				int			arr_len;
				int			j;

				if (arr == NULL || ARR_NDIM(arr) != 1)
				{
					dataset->labels[i] = NAN;
					continue;
				}

				arr_len = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
				if (arr_len != feature_dim)
				{
					dataset->labels[i] = NAN;
					continue;
				}

				if (feat_type == FLOAT8ARRAYOID)
				{
					float8	   *fdat = (float8 *) ARR_DATA_PTR(arr);

					for (j = 0; j < feature_dim; j++)
						dest_row[j] = (float) fdat[j];
				}
				else
				{
					float4	   *fdat = (float4 *) ARR_DATA_PTR(arr);

					for (j = 0; j < feature_dim; j++)
						dest_row[j] = fdat[j];
				}
			}
			/* Handle Vector type */
			else
			{
				Vector	   *vec = DatumGetVector(feat_datum);
				float	   *vec_data;
				float	   *dest_row;
				int			j;

				if (vec == NULL || vec->dim != feature_dim)
				{
					dataset->labels[i] = NAN;
					continue;
				}

				vec_data = vec->data;
				dest_row = dataset->features + (i * feature_dim);
				for (j = 0; j < feature_dim; j++)
					dest_row[j] = vec_data[j];
			}
		}
	}
}

static bool
rf_load_model_from_catalog(int32 model_id, RFModel * *out)
{
	bytea	   *payload = NULL;
	Jsonb	   *metrics = NULL;
	RFModel    *decoded = NULL;
	bool		result = false;

	elog(DEBUG1, "neurondb: rf_load_model_from_catalog() called for model_id=%d, CurrentMemoryContext=%p, TopMemoryContext=%p",
		 model_id, (void *) CurrentMemoryContext, (void *) TopMemoryContext);

	if (model_id <= 0)
	{
		elog(WARNING, "rf_load_model_from_catalog: invalid model_id %d", model_id);
		return false;
	}

	if (out == NULL)
	{
		elog(WARNING, "rf_load_model_from_catalog: out parameter is NULL");
		return false;
	}

	*out = NULL;

#ifdef MEMORY_CONTEXT_CHECKING
	/* Check memory context at entry */
	if (CurrentMemoryContext != NULL)
		MemoryContextCheck(CurrentMemoryContext);
#endif

	elog(DEBUG1, "neurondb: rf_load_model_from_catalog() entering PG_TRY block");
	PG_TRY();
	{
		elog(DEBUG1, "neurondb: rf_load_model_from_catalog() calling ml_catalog_fetch_model_payload");
		if (!ml_catalog_fetch_model_payload(
											model_id, &payload, NULL, &metrics))
		{
			elog(DEBUG1, "rf_load_model_from_catalog: ml_catalog_fetch_model_payload failed for model_id %d", model_id);
			result = false;
		}
		else if (payload == NULL)
		{
			elog(DEBUG1, "rf_load_model_from_catalog: payload is NULL for model_id %d", model_id);
			if (metrics != NULL)
				NDB_FREE(metrics);
			result = false;
		}
		else if (VARSIZE(payload) < VARHDRSZ)
		{
			elog(WARNING, "rf_load_model_from_catalog: invalid payload size %d for model_id %d", VARSIZE(payload), model_id);
			NDB_FREE(payload);
			if (metrics != NULL)
				NDB_FREE(metrics);
			result = false;
		}
		else if (rf_metadata_is_gpu(metrics))
		{
			elog(DEBUG1, "rf_load_model_from_catalog: model_id %d is GPU model, skipping CPU load", model_id);
			if (payload != NULL)
				NDB_FREE(payload);
			if (metrics != NULL)
				NDB_FREE(metrics);
			result = false;
		}
		else
		{
			elog(DEBUG1, "neurondb: rf_load_model_from_catalog() payload size=%d bytes, calling rf_model_deserialize",
				 VARSIZE(payload));

			/* Deserialize with error handling */
			{
				uint8 training_backend = 0;
				decoded = rf_model_deserialize(payload, &training_backend);
			}

			if (decoded == NULL)
			{
				elog(WARNING, "rf_load_model_from_catalog: rf_model_deserialize returned NULL for model_id %d", model_id);
				NDB_FREE(payload);
				if (metrics != NULL)
					NDB_FREE(metrics);
				result = false;
			}
			else
			{
				/* Validate decoded model */
				if (decoded->n_features <= 0 || decoded->n_features > 1000000 ||
					decoded->n_classes <= 0 || decoded->n_classes > 10000 ||
					decoded->tree_count < 0 || decoded->tree_count > 10000)
				{
					elog(WARNING, "rf_load_model_from_catalog: invalid model parameters for model_id %d (n_features=%d, n_classes=%d, tree_count=%d)",
						 model_id, decoded->n_features, decoded->n_classes, decoded->tree_count);
					rf_free_deserialized_model(decoded);
					NDB_FREE(payload);
					if (metrics != NULL)
						NDB_FREE(metrics);
					result = false;
				}
				else
				{
					bool		store_succeeded = false;

					/* Validate memory context before attempting to store */
					if (TopMemoryContext == NULL)
					{
						elog(WARNING, "rf_load_model_from_catalog: TopMemoryContext is NULL, cannot store model %d", model_id);
						rf_free_deserialized_model(decoded);
						NDB_FREE(payload);
						if (metrics != NULL)
							NDB_FREE(metrics);
						result = false;
					}
					else
					{
						/*
						 * rf_store_model - errors will be caught by outer
						 * PG_TRY
						 */
						rf_store_model(model_id,
									   decoded->n_features,
									   decoded->n_samples,
									   decoded->n_classes,
									   decoded->majority_value,
									   decoded->majority_fraction,
									   decoded->gini_impurity,
									   decoded->label_entropy,
									   decoded->class_counts,
									   decoded->feature_means,
									   decoded->feature_variances,
									   decoded->feature_importance,
									   decoded->tree,
									   decoded->split_feature,
									   decoded->split_threshold,
									   decoded->second_value,
									   decoded->second_fraction,
									   decoded->left_branch_value,
									   decoded->left_branch_fraction,
									   decoded->right_branch_value,
									   decoded->right_branch_fraction,
									   decoded->max_deviation,
									   decoded->max_split_deviation,
									   decoded->feature_limit,
									   decoded->left_branch_means,
									   decoded->right_branch_means,
									   decoded->tree_count,
									   decoded->trees,
									   decoded->tree_majority,
									   decoded->tree_majority_fraction,
									   decoded->tree_second,
									   decoded->tree_second_fraction,
									   decoded->tree_oob_accuracy,
									   decoded->oob_accuracy);
						store_succeeded = true;
					}

					/* Only continue if rf_store_model succeeded */
					if (store_succeeded)
					{
						rf_free_deserialized_model(decoded);
						decoded = NULL;

						NDB_FREE(payload);
						payload = NULL;
						if (metrics != NULL)
						{
							NDB_FREE(metrics);
							metrics = NULL;
						}

						if (out != NULL)
							result = rf_lookup_model(model_id, out);
						else
							result = true;
					}
				}
			}
		}
	}
	PG_CATCH();
	{
		elog(WARNING, "rf_load_model_from_catalog: exception during model load for model_id %d", model_id);

		/* Cleanup on error */
		if (decoded != NULL)
			rf_free_deserialized_model(decoded);
		if (payload != NULL)
			NDB_FREE(payload);
		if (metrics != NULL)
			NDB_FREE(metrics);

		FlushErrorState();
		result = false;
	}
	PG_END_TRY();

	return result;
}

static bool
rf_metadata_is_gpu(Jsonb * metadata)
{
	bool		is_gpu = false;
	JsonbIterator *it;
	JsonbValue	v;
	JsonbIteratorToken r;

	if (metadata == NULL)
		return false;

	/* Check for training_backend integer in metrics */
	it = JsonbIteratorInit((JsonbContainer *) &metadata->root);
	while ((r = JsonbIteratorNext(&it, &v, true)) != WJB_DONE)
	{
		if (r == WJB_KEY && v.type == jbvString)
		{
			char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

			if (strcmp(key, "training_backend") == 0)
			{
				r = JsonbIteratorNext(&it, &v, true);
				if (r == WJB_VALUE && v.type == jbvNumeric)
				{
					int			backend = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					is_gpu = (backend == 1);
				}
			}
			NDB_FREE(key);
		}
	}

	return is_gpu;
}

static bool
rf_try_gpu_predict_catalog(int32 model_id,
						   const Vector *feature_vec,
						   double *result_out)
{
	bytea	   *payload = NULL;
	Jsonb	   *metrics = NULL;
	char	   *gpu_err = NULL;
	int			class_out = -1;
	bool		success = false;

	if (!neurondb_gpu_is_available())
		return false;
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;

	if (!ml_catalog_fetch_model_payload(
										model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
		goto cleanup;

	if (!rf_metadata_is_gpu(metrics))
		goto cleanup;

	if (ndb_gpu_rf_predict(payload,
						   feature_vec->data,
						   feature_vec->dim,
						   &class_out,
						   &gpu_err)
		== 0)
	{
		if (result_out != NULL)
			*result_out = (double) class_out;
		elog(DEBUG1,
			 "random_forest: GPU prediction used for model %d class=%d",
			 model_id,
			 class_out);
		success = true;
	}
	else if (gpu_err != NULL)
	{
		elog(WARNING,
			 "random_forest: GPU prediction failed for model %d (%s)",
			 model_id,
			 gpu_err);
	}

cleanup:

	/*
	 * Don't free gpu_err - it's allocated by GPU function using pstrdup() and
	 * will be automatically freed when the memory context is cleaned up.
	 */
	if (gpu_err != NULL)
	{
		elog(DEBUG1, "rf_try_gpu_predict_catalog: [DEBUG] gpu_err=%p (not freeing - managed by memory context)", (void *) gpu_err);
		gpu_err = NULL;			/* Clear pointer but don't free */
	}
	if (payload != NULL)
	{
		NDB_FREE(payload);
		payload = NULL;
	}
	if (metrics != NULL)
	{
		NDB_FREE(metrics);
		metrics = NULL;
	}

	return success;
}

typedef struct RFGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			feature_dim;
	int			class_count;
	int			sample_count;
}			RFGpuModelState;

static void
rf_gpu_release_state(RFGpuModelState * state)
{
	if (state == NULL)
		return;
	if (state->model_blob != NULL)
		NDB_FREE(state->model_blob);
	if (state->metrics != NULL)
		NDB_FREE(state->metrics);
	NDB_FREE(state);
}

static bool
rf_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	RFGpuModelState *state;
	bytea	   *payload;
	Jsonb	   *metrics;
	int			rc;
	
	ereport(DEBUG1,
			(errmsg("rf_gpu_train: function entry"),
			 errdetail("spec->sample_count=%d, spec->feature_dim=%d, spec->class_count=%d",
					  spec ? spec->sample_count : 0,
					  spec ? spec->feature_dim : 0,
					  spec ? spec->class_count : 0)));

	if (errstr != NULL)
		*errstr = NULL;
	
	ereport(DEBUG1,
			(errmsg("rf_gpu_train: validating parameters")));
	
	if (model == NULL || spec == NULL)
	{
		elog(DEBUG1, "neurondb: rf_gpu_train: model or spec is NULL");
		return false;
	}
	if (!neurondb_gpu_is_available())
	{
		elog(DEBUG1, "neurondb: rf_gpu_train: GPU not available");
		return false;
	}
	if (spec->feature_matrix == NULL || spec->label_vector == NULL)
	{
		elog(DEBUG1, "neurondb: rf_gpu_train: feature_matrix or label_vector is NULL");
		return false;
	}
	if (spec->sample_count <= 0 || spec->feature_dim <= 0)
	{
		elog(DEBUG1, "neurondb: rf_gpu_train: invalid sample_count or feature_dim (sample_count=%d, feature_dim=%d)",
			 spec->sample_count, spec->feature_dim);
		return false;
	}
	if (spec->class_count <= 0)
	{
		elog(DEBUG1, "neurondb: rf_gpu_train: invalid class_count (class_count=%d)", spec->class_count);
		return false;
	}

	ereport(DEBUG1,
			(errmsg("rf_gpu_train: parameters validated, initializing payload and metrics")));

	payload = NULL;
	metrics = NULL;

	ereport(DEBUG2,
			(errmsg("rf_gpu_train: about to call ndb_gpu_rf_train"),
			 errdetail("feature_matrix=%p, label_vector=%p, sample_count=%d, feature_dim=%d, class_count=%d",
					  (void *)spec->feature_matrix, (void *)spec->label_vector,
					  spec->sample_count, spec->feature_dim, spec->class_count)));

	rc = ndb_gpu_rf_train(spec->feature_matrix,
						  spec->label_vector,
						  spec->sample_count,
						  spec->feature_dim,
						  spec->class_count,
						  spec->hyperparameters,
						  &payload,
						  &metrics,
						  errstr);
	
	ereport(DEBUG2,
			(errmsg("rf_gpu_train: ndb_gpu_rf_train returned"),
			 errdetail("rc=%d, payload=%p, metrics=%p", rc, (void *)payload, (void *)metrics)));
	if (rc != 0 || payload == NULL)
	{
		if (payload != NULL)
			NDB_FREE(payload);
		if (metrics != NULL)
			NDB_FREE(metrics);
		return false;
	}

	if (model->backend_state != NULL)
	{
		rf_gpu_release_state((RFGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	NDB_ALLOC(state, RFGpuModelState, 1);
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload;
	state->metrics = metrics;
	state->feature_dim = spec->feature_dim;
	state->class_count = spec->class_count;
	state->sample_count = spec->sample_count;

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
rf_gpu_predict(const MLGpuModel * model,
			   const float *input,
			   int input_dim,
			   float *output,
			   int output_dim,
			   char **errstr)
{
	const		RFGpuModelState *state;
	int			rc;
	int			class_id;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = -1.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const RFGpuModelState *) model->backend_state;
	class_id = -1;

	rc = ndb_gpu_rf_predict(state->model_blob,
							input,
							state->feature_dim > 0 ? state->feature_dim : input_dim,
							&class_id,
							errstr);
	if (rc != 0)
		return false;

	output[0] = (float) class_id;
	return true;
}

static bool
rf_gpu_evaluate(const MLGpuModel * model,
				const MLGpuEvalSpec * spec,
				MLGpuMetrics * out,
				char **errstr)
{
	const		RFGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("rf_gpu_evaluate: invalid model or state");
		return false;
	}

	state = (const RFGpuModelState *) model->backend_state;

	/* Create metrics JSON with model information */
	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"random_forest\",\"storage\":\"gpu\","
					 "\"feature_dim\":%d,\"class_count\":%d,\"n_samples\":%d}",
					 state->feature_dim > 0 ? state->feature_dim : 0,
					 state->class_count > 0 ? state->class_count : 0,
					 state->sample_count > 0 ? state->sample_count : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetTextDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	elog(DEBUG1,
		 "rf_gpu_evaluate: GPU evaluation completed (feature_dim=%d, "
		 "class_count=%d)",
		 state->feature_dim,
		 state->class_count);

	return true;
}

static bool
rf_gpu_serialize(const MLGpuModel * model,
				 bytea * *payload_out,
				 Jsonb * *metadata_out,
				 char **errstr)
{
	const		RFGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const RFGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	payload_size = VARSIZE(state->model_blob);
	{
		NDB_DECLARE(char *, payload_buf);
		NDB_ALLOC(payload_buf, char, payload_size);
		payload_copy = (bytea *) payload_buf;
	}
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_FREE(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
	{
		int			metadata_size;
		Jsonb	   *metadata_copy;

		metadata_size = VARSIZE(state->metrics);
		{
			NDB_DECLARE(char *, metadata_buf);
			NDB_ALLOC(metadata_buf, char, metadata_size);
			metadata_copy = (Jsonb *) metadata_buf;
		}
		NDB_CHECK_ALLOC(metadata_copy, "metadata_copy");
		memcpy(metadata_copy, state->metrics, metadata_size);
		*metadata_out = metadata_copy;
	}

	return true;
}

static bool
rf_gpu_deserialize(MLGpuModel * model,
				   const bytea * payload,
				   const Jsonb * metadata,
				   char **errstr)
{
	RFGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	{
		NDB_DECLARE(char *, payload_buf);
		NDB_ALLOC(payload_buf, char, payload_size);
		payload_copy = (bytea *) payload_buf;
	}
	NDB_CHECK_ALLOC(payload_copy, "payload_copy");
	memcpy(payload_copy, payload, payload_size);

	NDB_ALLOC(state, RFGpuModelState, 1);
	NDB_CHECK_ALLOC(state, "state");
	state->model_blob = payload_copy;
	state->feature_dim = -1;
	state->class_count = -1;
	state->sample_count = -1;

	if (metadata != NULL)
	{
		int			metadata_size;
		Jsonb	   *metadata_copy;

		metadata_size = VARSIZE(metadata);
		{
			NDB_DECLARE(char *, metadata_buf);
			NDB_ALLOC(metadata_buf, char, metadata_size);
			metadata_copy = (Jsonb *) metadata_buf;
		}
		NDB_CHECK_ALLOC(metadata_copy, "metadata_copy");
		memcpy(metadata_copy, metadata, metadata_size);
		state->metrics = metadata_copy;
	}

	if (model->backend_state != NULL)
		rf_gpu_release_state((RFGpuModelState *) model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static void
rf_gpu_destroy(MLGpuModel * model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		rf_gpu_release_state((RFGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps rf_gpu_model_ops = {
	.algorithm = "random_forest",
	.train = rf_gpu_train,
	.predict = rf_gpu_predict,
	.evaluate = rf_gpu_evaluate,
	.serialize = rf_gpu_serialize,
	.deserialize = rf_gpu_deserialize,
	.destroy = rf_gpu_destroy,
};

void
neurondb_gpu_register_rf_model(void)
{
	static bool registered = false;

	if (registered)
		return;

	ndb_gpu_register_model_ops(&rf_gpu_model_ops);
	registered = true;
}
