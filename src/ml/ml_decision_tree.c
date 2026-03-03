/*-------------------------------------------------------------------------
 *
 * ml_decision_tree.c
 *    Decision tree implementation.
 *
 * This module implements CART decision trees for classification and regression,
 * with model serialization and catalog storage.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_decision_tree.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "utils/memutils.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_decision_tree_internal.h"
#include "ml_catalog.h"
#include "neurondb_gpu_bridge.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_model.h"
#include "neurondb_gpu_backend.h"
#include "ml_gpu_registry.h"
#include "ml_gpu_decision_tree.h"
#include "neurondb_cuda_dt.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_macros.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_json.h"
#include "neurondb_constants.h"
#include "neurondb_guc.h"

#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
#endif
#endif

#include <math.h>
#include <float.h>
#include <limits.h>

#define TreeNode DTNode

/*
 * DTDataset
 * Internal structure to hold dataset for training.
 */
typedef struct DTDataset
{
	float	   *features;
	double	   *labels;
	int			n_samples;
	int			feature_dim;
}			DTDataset;

static void dt_dataset_init(DTDataset * dataset);
static void dt_dataset_free(DTDataset * dataset);
static void dt_dataset_load(const char *quoted_tbl, const char *quoted_feat, const char *quoted_label, DTDataset * dataset);
static bytea * dt_model_serialize(const DTModel *model, uint8 training_backend);
static DTModel *dt_model_deserialize(const bytea * data, uint8 * training_backend_out);
static bool dt_metadata_is_gpu(Jsonb * metadata);
static bool dt_try_gpu_predict_catalog(int32 model_id, const Vector *feature_vec, double *result_out);
static bool dt_load_model_from_catalog(int32 model_id, DTModel **out);
static void dt_free_tree(DTNode *node);
static double dt_tree_predict(const DTNode *node, const float *x, int dim);

/*
 * dt_dataset_init - Initialize a DTDataset structure
 *
 * Initializes all fields of a DTDataset structure to zero. Safe to call
 * with NULL pointer.
 *
 * Parameters:
 *   dataset - DTDataset structure to initialize
 *
 * Notes:
 *   This function sets all fields to zero, preparing the structure for use.
 *   It does not allocate any memory.
 */
static void
dt_dataset_init(DTDataset * dataset)
{
	if (dataset == NULL)
		return;

	memset(dataset, 0, sizeof(DTDataset));
}

/*
 * dt_dataset_free - Free memory allocated in a DTDataset structure
 *
 * Frees all dynamically allocated memory in a DTDataset structure,
 * including feature and label arrays. Safe to call with NULL pointer.
 *
 * Parameters:
 *   dataset - DTDataset structure to free
 *
 * Notes:
 *   After calling this function, the dataset structure is reinitialized
 *   to zero state. The structure itself is not freed, only its contents.
 */
static void
dt_dataset_free(DTDataset * dataset)
{
	if (dataset == NULL)
		return;

	if (dataset->features)
		nfree(dataset->features);
	if (dataset->labels)
		nfree(dataset->labels);

	dt_dataset_init(dataset);
}

/*
 * dt_dataset_load - Load feature and label data from a table
 *
 * Loads training data from a PostgreSQL table into a DTDataset structure
 * using SPI. Filters out rows with NULL features or labels.
 *
 * Parameters:
 *   quoted_tbl - Quoted table name
 *   quoted_feat - Quoted feature column name
 *   quoted_label - Quoted label column name
 *   dataset - Output DTDataset structure to populate
 *
 * Notes:
 *   The function allocates memory for features and labels arrays in
 *   CurrentMemoryContext. Memory must be freed using dt_dataset_free().
 *   Uses SPI to execute a SELECT query and extract data.
 */
static void
dt_dataset_load(const char *quoted_tbl,
				const char *quoted_feat,
				const char *quoted_label,
				DTDataset * dataset)
{
	int			feature_dim = 0;
	int			i = 0;
	int			n_samples = 0;
	int			ret;
	MemoryContext oldcontext;
	NdbSpiSession *dt_load_spi_session = NULL;
	StringInfoData query;

	if (!dataset)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: dt_dataset_load: dataset is NULL")));

	oldcontext = CurrentMemoryContext;

	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quoted_feat,
					 quoted_label,
					 quoted_tbl,
					 quoted_feat,
					 quoted_label);

	NDB_SPI_SESSION_BEGIN(dt_load_spi_session, oldcontext);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		nfree(query.data);
		NDB_SPI_SESSION_END(dt_load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: dt_dataset_load: query failed"),
				 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
				 errhint("Verify the table exists and contains valid feature and label columns.")));
	}

	n_samples = SPI_processed;
	if (n_samples < 10)
	{
		nfree(query.data);
		NDB_SPI_SESSION_END(dt_load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				 errmsg("neurondb: dt_dataset_load: need at least 10 samples"),
				 errdetail("Found %d samples but need at least 10", n_samples),
				 errhint("Add more data to the table.")));
	}

	/* Safe access for complex types - validate before access */
	if (n_samples > 0 && SPI_tuptable != NULL && SPI_tuptable->vals != NULL &&
		SPI_tuptable->vals[0] != NULL && SPI_tuptable->tupdesc != NULL)
	{
		HeapTuple	first_tuple = SPI_tuptable->vals[0];
		TupleDesc	tupdesc = SPI_tuptable->tupdesc;
		Datum		feat_datum;
		bool		feat_null = false;
		Oid			feat_type;

		feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &feat_null);
		if (!feat_null)
		{
			feat_type = SPI_gettypeid(tupdesc, 1);

			if (feat_type == FLOAT8ARRAYOID)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

				if (arr != NULL)
					feature_dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
			}
			else if (feat_type == FLOAT4ARRAYOID)
			{
				ArrayType  *arr = DatumGetArrayTypeP(feat_datum);

				if (arr != NULL)
					feature_dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
			}
			else
			{
				Vector	   *vec = DatumGetVector(feat_datum);

				if (vec != NULL && vec->dim > 0)
					feature_dim = vec->dim;
			}
		}
	}

	if (feature_dim <= 0)
	{
		nfree(query.data);
		NDB_SPI_SESSION_END(dt_load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: dt_dataset_load: could not determine feature dimension"),
				 errdetail("No valid feature vectors found in the first row"),
				 errhint("Ensure the feature column contains valid array or vector data.")));
	}

	MemoryContextSwitchTo(oldcontext);

	{
		float *features_tmp = NULL;
		double *labels_tmp = NULL;
		nalloc(features_tmp, float, ((Size) n_samples) * ((Size) feature_dim));
		nalloc(labels_tmp, double, (Size) n_samples);
		dataset->features = features_tmp;
		dataset->labels = labels_tmp;
	}

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple;
		TupleDesc	tupdesc;
		Datum		feat_datum;
		Datum		label_datum;
		bool		feat_null = false;
		bool		label_null = false;
		Oid			feat_type;
		float *row = NULL;

		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			continue;
		}
		tuple = SPI_tuptable->vals[i];
		tupdesc = SPI_tuptable->tupdesc;
		if (tupdesc == NULL)
		{
			continue;
		}

		feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
		if (feat_null)
			continue;

		feat_type = SPI_gettypeid(tupdesc, 1);
		row = dataset->features + (i * feature_dim);

		if (feat_type == FLOAT8ARRAYOID)
		{
			ArrayType  *arr = DatumGetArrayTypeP(feat_datum);
			int			arr_dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
			float8 *data = NULL;

			if (arr_dim != feature_dim)
			{
				nfree(query.data);
				NDB_SPI_SESSION_END(dt_load_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: dt_dataset_load: inconsistent array dimensions"),
						 errdetail("Row %d has %d dimensions but expected %d", i + 1, arr_dim, feature_dim),
						 errhint("Ensure all feature arrays have the same dimension.")));
			}
			data = (float8 *) ARR_DATA_PTR(arr);
			for (int j = 0; j < feature_dim; j++)
				row[j] = (float) data[j];
		}
		else if (feat_type == FLOAT4ARRAYOID)
		{
			ArrayType  *arr = DatumGetArrayTypeP(feat_datum);
			int			arr_dim = ArrayGetNItems(ARR_NDIM(arr), ARR_DIMS(arr));
			float4 *data = NULL;

			if (arr_dim != feature_dim)
			{
				nfree(query.data);
				NDB_SPI_SESSION_END(dt_load_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: dt_dataset_load: inconsistent array dimensions"),
						 errdetail("Row %d has %d dimensions but expected %d", i + 1, arr_dim, feature_dim),
						 errhint("Ensure all feature arrays have the same dimension.")));
			}
			data = (float4 *) ARR_DATA_PTR(arr);
			memcpy(row, data, sizeof(float) * feature_dim);
		}
		else
		{
			Vector	   *vec = DatumGetVector(feat_datum);

			if (vec->dim != feature_dim)
			{
				nfree(query.data);
				NDB_SPI_SESSION_END(dt_load_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: dt_dataset_load: inconsistent vector dimensions"),
						 errdetail("Row %d has %d dimensions but expected %d", i + 1, vec->dim, feature_dim),
						 errhint("Ensure all feature vectors have the same dimension.")));
			}
			memcpy(row, vec->data, sizeof(float) * feature_dim);
		}

		/* Safe access for label - validate tupdesc has at least 2 columns */
		if (tupdesc->natts < 2)
		{
			continue;
		}
		label_datum = SPI_getbinval(tuple, tupdesc, 2, &label_null);
		if (label_null)
			continue;

		{
			Oid			label_type = SPI_gettypeid(tupdesc, 2);

			if (label_type == INT2OID)
				dataset->labels[i] = (double) DatumGetInt16(label_datum);
			else if (label_type == INT4OID)
				dataset->labels[i] = (double) DatumGetInt32(label_datum);
			else if (label_type == INT8OID)
				dataset->labels[i] = (double) DatumGetInt64(label_datum);
			else
				dataset->labels[i] = DatumGetFloat8(label_datum);
		}
	}

	dataset->n_samples = n_samples;
	dataset->feature_dim = feature_dim;

	nfree(query.data);
	NDB_SPI_SESSION_END(dt_load_spi_session);
}

/*
 * compute_gini
 *
 * Compute the Gini impurity for a vector of binary labels.
 */
static double
compute_gini(const double *labels, int n)
{
	int			i,
				class0 = 0,
				class1 = 0;
	double		p0,
				p1;

	if (n == 0)
		return 0.0;

	for (i = 0; i < n; i++)
	{
		if ((int) labels[i] == 0)
			class0++;
		else
			class1++;
	}

	p0 = (double) class0 / (double) n;
	p1 = (double) class1 / (double) n;
	return 1.0 - (p0 * p0 + p1 * p1);
}

/*
 * compute_variance
 *
 * Compute variance (mean squared deviation from the mean).
 */
static double
compute_variance(const double *values, int n)
{
	int			i;
	double		mean = 0.0;
	double		var = 0.0;

	if (n == 0)
		return 0.0;

	for (i = 0; i < n; i++)
		mean += values[i];
	mean /= n;

	for (i = 0; i < n; i++)
		var += (values[i] - mean) * (values[i] - mean);
	var /= n;

	return var;
}

/*
 * find_best_split_1d
 *
 * Find the best split on any feature using brute-force greedy split.
 */
static void
find_best_split_1d(const float *features,
				   const double *labels,
				   const int *indices,
				   int n_samples,
				   int dim,
				   int *best_feature,
				   float *best_threshold,
				   double *best_gain,
				   bool is_classification)
{
	int			feat;

	*best_gain = -DBL_MAX;
	*best_feature = -1;
	*best_threshold = 0.0;

	for (feat = 0; feat < dim; feat++)
	{
		float		min_val = FLT_MAX,
					max_val = -FLT_MAX;
		int			ii;

		for (ii = 0; ii < n_samples; ii++)
		{
			float		val = features[indices[ii] * dim + feat];

			if (val < min_val)
				min_val = val;
			if (val > max_val)
				max_val = val;
		}

		if (min_val == max_val)
			continue;

		for (ii = 1; ii < 10; ii++)
		{
			int			left_count = 0,
						right_count = 0,
						j;
			int			l_idx = 0,
						r_idx = 0;

			double *left_y = NULL;
			double *right_y = NULL;

			float		threshold = min_val + (max_val - min_val) * ii / 10.0f;
			double		left_imp,
						right_imp,
						gain;

			for (j = 0; j < n_samples; j++)
			{
				if (features[indices[j] * dim + feat] <= threshold)
					left_count++;
				else
					right_count++;
			}

			if (left_count == 0 || right_count == 0)
				continue;

			nalloc(left_y, double, left_count);
			nalloc(right_y, double, right_count);

			l_idx = r_idx = 0;
			for (j = 0; j < n_samples; j++)
			{
				if (features[indices[j] * dim + feat] <= threshold)
					left_y[l_idx++] = labels[indices[j]];
				else
					right_y[r_idx++] = labels[indices[j]];
			}

			Assert(l_idx == left_count);
			Assert(r_idx == right_count);

			if (is_classification)
			{
				left_imp = compute_gini(left_y, left_count);
				right_imp = compute_gini(right_y, right_count);
			}
			else
			{
				left_imp = compute_variance(left_y, left_count);
				right_imp = compute_variance(right_y, right_count);
			}

			gain = -(((double) left_count / (double) n_samples) * left_imp +
					 ((double) right_count / (double) n_samples) * right_imp);

			if (gain > *best_gain)
			{
				*best_gain = gain;
				*best_feature = feat;
				*best_threshold = threshold;
			}

			nfree(left_y);
			nfree(right_y);
		}
	}
}

/*
 * build_tree_1d
 *
 * Recursively build a decision tree using 1D flat arrays.
 */
static DTNode *
build_tree_1d(const float *features,
			  const double *labels,
			  const int *indices,
			  int n_samples,
			  int dim,
			  int max_depth,
			  int min_samples_split,
			  bool is_classification)
{
	DTNode	   *node = NULL;
	int			i,
				best_feature;
	float		best_threshold;
	double		best_gain;

	int *left_indices = NULL;
	int *right_indices = NULL;
	int			left_count = 0,
				right_count = 0;

	node = (DTNode *) palloc0(sizeof(DTNode));

	if (max_depth == 0 || n_samples < min_samples_split)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int			class0 = 0,
						class1 = 0;

			for (i = 0; i < n_samples; i++)
			{
				int			l = (int) labels[indices[i]];

				if (l == 0)
					class0++;
				else
					class1++;
			}
			node->leaf_value = (class1 > class0) ? 1.0 : 0.0;
		}
		else
		{
			if (n_samples > 0)
			{
				double		sum = 0.0;

				for (i = 0; i < n_samples; i++)
					sum += labels[indices[i]];
				node->leaf_value = sum / n_samples;
			}
			else
			{
				node->leaf_value = 0.0;
			}
		}
		return node;
	}

	find_best_split_1d(features, labels, indices, n_samples, dim,
					   &best_feature, &best_threshold, &best_gain, is_classification);

	if (best_feature == -1)
	{
		node->is_leaf = true;
		if (is_classification)
		{
			int			class0 = 0,
						class1 = 0;

			for (i = 0; i < n_samples; i++)
			{
				int			l = (int) labels[indices[i]];

				if (l == 0)
					class0++;
				else
					class1++;
			}
			node->leaf_value = (class1 > class0) ? 1.0 : 0.0;
		}
		else
		{
			if (n_samples > 0)
			{
				double		sum = 0.0;

				for (i = 0; i < n_samples; i++)
					sum += labels[indices[i]];
				node->leaf_value = sum / n_samples;
			}
			else
			{
				node->leaf_value = 0.0;
			}
		}
		return node;
	}

	node->is_leaf = false;
	node->feature_idx = best_feature;
	node->threshold = best_threshold;

	nalloc(left_indices, int, n_samples);
	nalloc(right_indices, int, n_samples);

	for (i = 0; i < n_samples; i++)
	{
		if (features[indices[i] * dim + best_feature] <= best_threshold)
			left_indices[left_count++] = indices[i];
		else
			right_indices[right_count++] = indices[i];
	}
	Assert(left_count + right_count == n_samples);

	node->left = build_tree_1d(features, labels, left_indices, left_count, dim,
							   max_depth - 1, min_samples_split, is_classification);
	node->right = build_tree_1d(features, labels, right_indices, right_count, dim,
								max_depth - 1, min_samples_split, is_classification);

	nfree(left_indices);
	nfree(right_indices);

	return node;
}

/*
 * dt_tree_predict
 *
 * Predict by walking the tree for an input vector.
 */
static double
dt_tree_predict(const DTNode *node, const float *x, int dim)
{
	if (node == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: dt_tree_predict: NULL node")));
	if (node->is_leaf)
		return node->leaf_value;
	if (node->feature_idx < 0 || node->feature_idx >= dim)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: dt_tree_predict: invalid feature_idx %d (dim=%d)",
						node->feature_idx, dim)));

	if (x[node->feature_idx] <= node->threshold)
		return dt_tree_predict(node->left, x, dim);
	else
		return dt_tree_predict(node->right, x, dim);
}

/*
 * dt_free_tree
 *
 * Free the memory allocated for the tree recursively.
 */
static void
dt_free_tree(DTNode *node)
{
	if (node == NULL)
		return;

	if (!node->is_leaf)
	{
		dt_free_tree(node->left);
		dt_free_tree(node->right);
	}
	nfree(node);
}

/*
 * dt_serialize_node
 *
 * Recursively write a tree node into a StringInfo buffer.
 */
static void
dt_serialize_node(StringInfo buf, const DTNode *node)
{
	if (!node)
	{
		pq_sendint8(buf, 0);
		return;
	}

	pq_sendint8(buf, 1);
	pq_sendint8(buf, node->is_leaf ? 1 : 0);
	pq_sendfloat8(buf, node->leaf_value);
	pq_sendint32(buf, node->feature_idx);
	pq_sendfloat4(buf, node->threshold);

	if (!node->is_leaf)
	{
		dt_serialize_node(buf, node->left);
		dt_serialize_node(buf, node->right);
	}
}

/*
 * dt_deserialize_node
 *
 * Recursively parse a tree node from a StringInfo buffer.
 */
static DTNode *
dt_deserialize_node(StringInfo buf)
{
	DTNode *node = NULL;
	int8		marker,
				is_leaf;

	marker = pq_getmsgint(buf, 1);
	if (marker == 0)
		return NULL;

	node = (DTNode *) palloc0(sizeof(DTNode));
	is_leaf = pq_getmsgint(buf, 1);
	node->is_leaf = (is_leaf != 0);
	node->leaf_value = pq_getmsgfloat8(buf);
	node->feature_idx = pq_getmsgint(buf, 4);
	node->threshold = pq_getmsgfloat4(buf);

	if (!node->is_leaf)
	{
		node->left = dt_deserialize_node(buf);
		node->right = dt_deserialize_node(buf);
	}

	return node;
}

/*
 * dt_model_serialize
 *
 * Serialize a DTModel structure into a binary blob.
 */
static bytea *
dt_model_serialize(const DTModel *model, uint8 training_backend)
{
	StringInfoData buf;
	bytea *result = NULL;
	char *result_raw = NULL;

	if (model == NULL)
		return NULL;

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: dt_model_serialize: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	initStringInfo(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) */
	pq_sendbyte(&buf, training_backend);

	pq_sendint32(&buf, model->model_id);
	pq_sendint32(&buf, model->n_features);
	pq_sendint32(&buf, model->n_samples);
	pq_sendint32(&buf, model->max_depth);
	pq_sendint32(&buf, model->min_samples_split);

	dt_serialize_node(&buf, model->root);

	nalloc(result_raw, char, VARHDRSZ + buf.len);
	result = (bytea *) result_raw;
	SET_VARSIZE(result, VARHDRSZ + buf.len);
	memcpy(VARDATA(result), buf.data, buf.len);
	nfree(buf.data);

	return result;
}

/*
 * dt_model_deserialize
 *
 * Deserialize a binary blob into a DTModel struct.
 */
static DTModel *
dt_model_deserialize(const bytea * data, uint8 * training_backend_out)
{
	StringInfoData buf;
	DTModel *model = NULL;
	uint8		training_backend = 0;

	if (data == NULL)
		return NULL;

	/* Deserialization with defensive error handling */
	model = (DTModel *) palloc0(sizeof(DTModel));

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	/* Read training_backend first */
	training_backend = (uint8) pq_getmsgbyte(&buf);

	/* Check if we have enough data for the header */
	if (buf.len < (training_backend == 0 && buf.cursor == 0 ? 20 : 21))
	{
		elog(WARNING, "dt: model data too small (%d bytes) for header, expected at least %d bytes", buf.len, training_backend == 0 && buf.cursor == 0 ? 20 : 21);
		nfree(model);
		return NULL;
	}

	/* Read header fields with bounds checking */
	if (buf.cursor + 4 > buf.len)
	{
		elog(WARNING, "dt: buffer overflow reading model_id, cursor=%d, len=%d", buf.cursor, buf.len);
		nfree(model);
		return NULL;
	}
	model->model_id = (int32) pq_getmsgint(&buf, 4);

	if (buf.cursor + 4 > buf.len)
	{
		elog(WARNING, "dt: buffer overflow reading n_features, cursor=%d, len=%d", buf.cursor, buf.len);
		nfree(model);
		return NULL;
	}
	{
		unsigned int raw_value = pq_getmsgint(&buf, 4);
		model->n_features = (int32) raw_value;
	}

	if (buf.cursor + 4 > buf.len)
	{
		elog(WARNING, "dt: buffer overflow reading n_samples, cursor=%d, len=%d", buf.cursor, buf.len);
		nfree(model);
		return NULL;
	}
	model->n_samples = (int32) pq_getmsgint(&buf, 4);

	if (buf.cursor + 4 > buf.len)
	{
		elog(WARNING, "dt: buffer overflow reading max_depth, cursor=%d, len=%d", buf.cursor, buf.len);
		nfree(model);
		return NULL;
	}
	model->max_depth = (int32) pq_getmsgint(&buf, 4);

	if (buf.cursor + 4 > buf.len)
	{
		elog(WARNING, "dt: buffer overflow reading min_samples_split, cursor=%d, len=%d", buf.cursor, buf.len);
		nfree(model);
		return NULL;
	}
	model->min_samples_split = (int32) pq_getmsgint(&buf, 4);


	if (model->n_features <= 0 || model->n_features > 10000)
	{
		elog(WARNING, "dt: invalid n_features %d in deserialized model, treating as corrupted", model->n_features);
		nfree(model);
		return NULL;
	}
	if (model->n_samples < 0 || model->n_samples > 100000000)
	{
		elog(WARNING, "dt: invalid n_samples %d in deserialized model, treating as corrupted", model->n_samples);
		nfree(model);
		return NULL;
	}

	/* Deserialize the tree structure */
	model->root = dt_deserialize_node(&buf);
	if (model->root == NULL)
	{
		nfree(model);
		return NULL;
	}

	/* Return training_backend if output parameter provided */
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;

	return model;
}

/*
 * dt_metadata_is_gpu
 *
 * Return true if model metadata indicates GPU storage.
 */
static bool
dt_metadata_is_gpu(Jsonb * metadata)
{
	bool		is_gpu = false;
	JsonbIterator *it = NULL;
	JsonbIteratorToken r;
	JsonbValue	v;

	if (metadata == NULL)
		return false;

	/* Check for training_backend integer in metrics */
	it = JsonbIteratorInit((JsonbContainer *) & metadata->root);
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
			nfree(key);
		}
	}

	return is_gpu;
}

/*
 * dt_try_gpu_predict_catalog
 *
 * Attempt to run prediction on GPU for given model and feature vector.
 */
static bool
dt_try_gpu_predict_catalog(int32 model_id, const Vector *feature_vec, double *result_out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	char *gpu_err = NULL;
	double		prediction = 0.0;
	bool		success = false;

	/* Check compute mode - only try GPU if compute mode allows it */
	if (!NDB_SHOULD_TRY_GPU())
		return false;

	if (!neurondb_gpu_is_available())
		return false;
	if (feature_vec == NULL)
		return false;
	if (feature_vec->dim <= 0)
		return false;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
		goto cleanup;

	/* Check if this is a GPU model - either by metrics or by payload format */
	{
		bool		is_gpu_model = false;
		uint32		payload_size;

		/* First check metrics for training_backend */
		if (dt_metadata_is_gpu(metrics))
		{
			is_gpu_model = true;
		}
		else
		{
			/* If metrics check didn't find GPU indicator, check payload format */
			/* GPU models start with NdbCudaDtModelHeader, CPU models start with uint8 training_backend */
			payload_size = VARSIZE(payload) - VARHDRSZ;
			
			/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
			/* GPU format: first field is feature_dim (int32) */
			/* Check if payload looks like GPU format (starts with int32, not uint8) */
			if (payload_size >= sizeof(int32))
			{
				const int32 *first_int = (const int32 *) VARDATA(payload);
				int32		first_value = *first_int;
				
				/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
				if (first_value > 0 && first_value <= 100000)
				{
					/* Check if payload size matches GPU format */
					if (payload_size >= sizeof(NdbCudaDtModelHeader))
					{
						const NdbCudaDtModelHeader *hdr = (const NdbCudaDtModelHeader *) VARDATA(payload);
						
						/* Validate header fields match the first int32 */
						if (hdr->feature_dim == first_value &&
							hdr->n_samples >= 0 && hdr->n_samples <= 1000000000 &&
							hdr->node_count > 0 && hdr->node_count <= 1000000)
						{
							size_t		expected_gpu_size = sizeof(NdbCudaDtModelHeader) +
								sizeof(NdbCudaDtNode) * (size_t) hdr->node_count;
							
							/* Size matches GPU format - likely a GPU model */
							if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
							{
								is_gpu_model = true;
							}
						}
					}
				}
			}
		}

		if (!is_gpu_model)
			goto cleanup;
	}

	if (ndb_gpu_dt_predict(payload,
						   feature_vec->data,
						   feature_vec->dim,
						   &prediction,
						   &gpu_err) == 0)
	{
		if (result_out)
			*result_out = prediction;
		success = true;
	}

cleanup:
	if (payload)
		nfree(payload);
	if (metrics)
		nfree(metrics);
	if (gpu_err)
		nfree(gpu_err);

	return success;
}

/*
 * dt_load_model_from_catalog
 *
 * Load a model from catalog into a DTModel structure.
 */
static bool
dt_load_model_from_catalog(int32 model_id, DTModel **out)
{
	bytea *payload = NULL;
	Jsonb *metrics = NULL;

	if (out == NULL)
		return false;
	*out = NULL;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
	{
		if (metrics)
			nfree(metrics);
		return false;
	}

	/* Skip GPU models - they should be handled by GPU prediction */
	/* Check both metrics and payload format to determine if this is a GPU model */
	{
		bool		is_gpu_model = false;
		uint32		payload_size;

		/* First check metrics for training_backend */
		if (metrics != NULL)
		{
			JsonbIterator *it = NULL;
			JsonbValue	v;
			JsonbIteratorToken r;

			it = JsonbIteratorInit((JsonbContainer *) & metrics->root);
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

							is_gpu_model = (backend == 1);
						}
					}
					nfree(key);
				}
			}
		}

		/* If metrics check didn't find GPU indicator, check payload format */
		/* GPU models start with NdbCudaDtModelHeader, CPU models start with uint8 training_backend */
		if (!is_gpu_model)
		{
			payload_size = VARSIZE(payload) - VARHDRSZ;
			
			/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
			/* GPU format: first field is feature_dim (int32) */
			/* Check if payload looks like GPU format (starts with int32, not uint8) */
			if (payload_size >= sizeof(int32))
			{
				const int32 *first_int = (const int32 *) VARDATA(payload);
				int32		first_value = *first_int;
				
				/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
				if (first_value > 0 && first_value <= 100000)
				{
					/* Check if payload size matches GPU format */
					if (payload_size >= sizeof(NdbCudaDtModelHeader))
					{
						const NdbCudaDtModelHeader *hdr = (const NdbCudaDtModelHeader *) VARDATA(payload);
						
						/* Validate header fields match the first int32 */
						if (hdr->feature_dim == first_value &&
							hdr->n_samples >= 0 && hdr->n_samples <= 1000000000 &&
							hdr->node_count > 0 && hdr->node_count <= 1000000)
						{
							size_t		expected_gpu_size = sizeof(NdbCudaDtModelHeader) +
								sizeof(NdbCudaDtNode) * (size_t) hdr->node_count;
							
							/* Size matches GPU format - likely a GPU model */
							if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
							{
								is_gpu_model = true;
							}
						}
					}
				}
			}
		}

		if (is_gpu_model)
		{
			if (payload)
				nfree(payload);
			if (metrics)
				nfree(metrics);
			return false;
		}
	}
	
	/* Unified format supports both CPU and GPU - deserialize if training_backend is 0 or 1 */
	*out = dt_model_deserialize(payload, NULL);
	if (*out == NULL)
	{
		if (metrics)
			nfree(metrics);
		return false;
	}

	if (metrics)
		nfree(metrics);

	return true;
}

/*
 * train_decision_tree_classifier - Train a decision tree classifier
 *
 * User-facing function that trains a decision tree classifier on data from
 * a table and saves the model to the catalog. Supports both CPU and GPU
 * training backends.
 *
 * Parameters:
 *   table_name - Name of table containing training data (text)
 *   feature_col - Name of feature column (text)
 *   label_col - Name of label column (text)
 *   hyperparams - JSONB hyperparameters (optional)
 *
 * Returns:
 *   Model ID (int32) of the trained model stored in catalog
 *
 * Notes:
 *   The function automatically selects CPU or GPU backend based on GUC
 *   settings and algorithm support. The trained model is serialized and
 *   stored in the ML catalog for later prediction.
 */
PG_FUNCTION_INFO_V1(train_decision_tree_classifier);

Datum
train_decision_tree_classifier(PG_FUNCTION_ARGS)
{
	text *table_name = NULL;
	text *feature_col = NULL;
	text *label_col = NULL;
	int32		max_depth;
	int32		min_samples_split;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *label_str = NULL;
	DTDataset	dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_label;
	MLGpuTrainResult gpu_result;

	char *gpu_err = NULL;
	Jsonb *gpu_hyperparams = NULL;
	StringInfoData hyperbuf;
	int32		model_id = 0;

	int *indices = NULL;
	DTModel *model = NULL;
	bytea *model_blob = NULL;
	MLCatalogModelSpec spec;

	Jsonb *params_jsonb = NULL;
	Jsonb *metrics_jsonb = NULL;
	int			i;

	table_name = PG_GETARG_TEXT_PP(0);
	feature_col = PG_GETARG_TEXT_PP(1);
	label_col = PG_GETARG_TEXT_PP(2);
	max_depth = PG_GETARG_INT32(3);
	min_samples_split = PG_GETARG_INT32(4);

	if (max_depth <= 0)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("dt: max_depth must be positive, got %d", max_depth)));
	if (min_samples_split < 2)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("dt: min_samples_split must be at least 2, got %d", min_samples_split)));

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	label_str = text_to_cstring(label_col);

	dt_dataset_init(&dataset);

	quoted_tbl = quote_identifier(tbl_str);
	quoted_feat = quote_identifier(feat_str);
	quoted_label = quote_identifier(label_str);

	dt_dataset_load(quoted_tbl, quoted_feat, quoted_label, &dataset);

	if (dataset.n_samples < 10)
	{
		dt_dataset_free(&dataset);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(label_str);
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				 errmsg("neurondb: dt: need at least 10 samples, got %d", dataset.n_samples)));
	}

	/* Initialize GPU if enabled */
	ndb_gpu_init_if_needed();

	if (neurondb_gpu_is_available() &&
		dataset.n_samples > 0 &&
		dataset.feature_dim > 0)
	{
		initStringInfo(&hyperbuf);
		appendStringInfo(&hyperbuf, "{\"max_depth\":%d,\"min_samples_split\":%d}",
						 max_depth, min_samples_split);
		/* Use ndb_jsonb_in_cstring (consistent with other ML algorithms) */
		gpu_hyperparams = ndb_jsonb_in_cstring(hyperbuf.data);
		if (gpu_hyperparams == NULL)
		{
			nfree(hyperbuf.data);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("neurondb: train_decision_tree_classifier: failed to parse GPU hyperparameters JSON")));
		}
		nfree(hyperbuf.data);
		hyperbuf.data = NULL;

		if (ndb_gpu_try_train_model("decision_tree",
									NULL,
									NULL,
									tbl_str,
									label_str,
									NULL,
									0,
									gpu_hyperparams,
									dataset.features,
									dataset.labels,
									dataset.n_samples,
									dataset.feature_dim,
									0,
									&gpu_result,
									&gpu_err) &&
			gpu_result.spec.model_data != NULL)
		{
			spec = gpu_result.spec;

			if (spec.training_table == NULL)
				spec.training_table = tbl_str;
			if (spec.training_column == NULL)
				spec.training_column = label_str;
			if (spec.parameters == NULL)
			{
				spec.parameters = gpu_hyperparams;
				gpu_hyperparams = NULL;
			}

			spec.algorithm = "decision_tree";
			spec.model_type = "classification";

			model_id = ml_catalog_register_model(&spec);

			if (gpu_err)
				nfree(gpu_err);
			if (gpu_hyperparams)
				nfree(gpu_hyperparams);
			ndb_gpu_free_train_result(&gpu_result);
			if (hyperbuf.data != NULL)
			{
				nfree(hyperbuf.data);
				hyperbuf.data = NULL;
			}
			dt_dataset_free(&dataset);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(label_str);

			PG_RETURN_INT32(model_id);
		}
		if (gpu_err)
		{
			nfree(gpu_err);
		}
		else
		{
		}
		if (gpu_hyperparams)
			nfree(gpu_hyperparams);

		ndb_gpu_free_train_result(&gpu_result);
		if (hyperbuf.data != NULL)
		{
			nfree(hyperbuf.data);
			hyperbuf.data = NULL;
		}

	}

	nalloc(indices, int, dataset.n_samples);

	for (i = 0; i < dataset.n_samples; i++)
		indices[i] = i;

	model = (DTModel *) palloc0(sizeof(DTModel));
	model->n_features = dataset.feature_dim;
	model->n_samples = dataset.n_samples;
	model->max_depth = max_depth;
	model->min_samples_split = min_samples_split;
	model->root = build_tree_1d(dataset.features, dataset.labels, indices,
								dataset.n_samples, dataset.feature_dim, max_depth,
								min_samples_split, true);

	model_blob = dt_model_serialize(model, 0);
	if (model_blob == NULL)
	{
		dt_free_tree(model->root);
		nfree(model);
		nfree(indices);
		dt_dataset_free(&dataset);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(label_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: dt: failed to serialize model")));
	}

	/* Build parameters JSON using JSONB API */
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;

		JsonbValue *final_value = NULL;
		Numeric		max_depth_num,
					min_samples_split_num;

		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			/* Add max_depth */
			jkey.type = jbvString;
			jkey.val.string.val = "max_depth";
			jkey.val.string.len = strlen("max_depth");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			max_depth_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_depth)));
			jval.type = jbvNumeric;
			jval.val.numeric = max_depth_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add min_samples_split */
			jkey.val.string.val = "min_samples_split";
			jkey.val.string.len = strlen("min_samples_split");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			min_samples_split_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(min_samples_split)));
			jval.type = jbvNumeric;
			jval.val.numeric = min_samples_split_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: train_decision_tree_classifier: pushJsonbValue(WJB_END_OBJECT) returned NULL for parameters");
			}

			params_jsonb = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData  *edata = CopyErrorData();

			elog(ERROR, "neurondb: train_decision_tree_classifier: parameters JSONB construction failed: %s", edata->message);
			FlushErrorState();
			params_jsonb = NULL;
		}
		PG_END_TRY();
	}

	if (params_jsonb == NULL)
	{
		dt_free_tree(model->root);
		nfree(model);
		nfree(indices);
		dt_dataset_free(&dataset);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(label_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: train_decision_tree_classifier: failed to create parameters JSONB")));
	}

	/* Build metrics JSON using JSONB API */
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;

		JsonbValue *final_value = NULL;
		Numeric		n_features_num,
					n_samples_num,
					max_depth_num,
					min_samples_split_num;

		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			/* Add algorithm */
			jkey.type = jbvString;
			jkey.val.string.val = "algorithm";
			jkey.val.string.len = strlen("algorithm");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvString;
			jval.val.string.val = "decision_tree";
			jval.val.string.len = strlen("decision_tree");
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add storage */
			jkey.val.string.val = "storage";
			jkey.val.string.len = strlen("storage");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvString;
			jval.val.string.val = "cpu";
			jval.val.string.len = strlen("cpu");
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add n_features */
			jkey.val.string.val = "n_features";
			jkey.val.string.len = strlen("n_features");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_features_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(dataset.feature_dim)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_features_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add n_samples */
			jkey.val.string.val = "n_samples";
			jkey.val.string.len = strlen("n_samples");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(dataset.n_samples)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_samples_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add max_depth */
			jkey.val.string.val = "max_depth";
			jkey.val.string.len = strlen("max_depth");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			max_depth_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(max_depth)));
			jval.type = jbvNumeric;
			jval.val.numeric = max_depth_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			/* Add min_samples_split */
			jkey.val.string.val = "min_samples_split";
			jkey.val.string.len = strlen("min_samples_split");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			min_samples_split_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(min_samples_split)));
			jval.type = jbvNumeric;
			jval.val.numeric = min_samples_split_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: train_decision_tree_classifier: pushJsonbValue(WJB_END_OBJECT) returned NULL for metrics");
			}

			metrics_jsonb = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData  *edata = CopyErrorData();

			elog(ERROR, "neurondb: train_decision_tree_classifier: metrics JSONB construction failed: %s", edata->message);
			FlushErrorState();
			metrics_jsonb = NULL;
		}
		PG_END_TRY();
	}

	if (metrics_jsonb == NULL)
	{
		if (params_jsonb != NULL)
		{
			nfree(params_jsonb);
			params_jsonb = NULL;
		}
		dt_free_tree(model->root);
		nfree(model);
		nfree(indices);
		dt_dataset_free(&dataset);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(label_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: train_decision_tree_classifier: failed to create metrics JSONB")));
	}

	memset(&spec, 0, sizeof(spec));
	spec.algorithm = "decision_tree";
	spec.model_type = "classification";
	spec.training_table = tbl_str;
	spec.training_column = label_str;
	spec.model_data = model_blob;
	spec.parameters = params_jsonb;
	spec.metrics = metrics_jsonb;

	model_id = ml_catalog_register_model(&spec);
	model->model_id = model_id;

	dt_free_tree(model->root);
	nfree(model);
	nfree(indices);
	nfree(model_blob);

	dt_dataset_free(&dataset);
	nfree(tbl_str);
	nfree(feat_str);
	nfree(label_str);

	PG_RETURN_INT32(model_id);
}

PG_FUNCTION_INFO_V1(predict_decision_tree_model_id);

/*
 * predict_decision_tree_model_id
 *
 * SQL-callable UDF for prediction using a trained decision tree model.
 */
Datum
predict_decision_tree_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	Vector *feature_vec = NULL;
	DTModel *model = NULL;

	double		result;
	bool		found_gpu;

	model_id = PG_GETARG_INT32(0);
	feature_vec = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(feature_vec);
	model = NULL;
	result = 0.0;
	found_gpu = false;

	if (feature_vec == NULL)
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						errmsg("dt: feature vector cannot be NULL")));

	if (dt_try_gpu_predict_catalog(model_id, feature_vec, &result))
	{
		found_gpu = true;
	}

	if (!found_gpu)
	{
		if (!dt_load_model_from_catalog(model_id, &model))
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("dt: model %d exists but has corrupted data", model_id)));

		if (model->root == NULL)
		{
			nfree(model);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("dt: model %d has NULL root (corrupted?)", model_id)));
		}

		if (feature_vec->dim != model->n_features)
		{
			nfree(model);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("dt: feature dimension mismatch: expected %d, got %d",
								   model->n_features, feature_vec->dim)));
		}

		result = dt_tree_predict(model->root, feature_vec->data, model->n_features);

		dt_free_tree(model->root);
		nfree(model);
	}

	PG_RETURN_FLOAT8(result);
}

/*
 * (removed unused function dt_predict_batch)
 */
#if 0
static void
dt_predict_batch_removed(const DTModel *model,
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

	if (model == NULL || model->root == NULL || features == NULL || labels == NULL || n_samples <= 0)
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


	for (i = 0; i < n_samples; i++)
	{
		const float *row = features + (i * feature_dim);
		double		y_true = labels[i];
		int			true_class;
		double		prediction;
		int			pred_class;

		if (!isfinite(y_true))
		{
			continue;
		}

		/*
		 * Convert label to binary class: values <= 0.5 -> class 0, > 0.5 ->
		 * class 1
		 */
		true_class = (y_true > 0.5) ? 1 : 0;

		prediction = dt_tree_predict(model->root, row, feature_dim);
		if (!isfinite(prediction))
		{
			continue;
		}

		/*
		 * Convert prediction to binary class: values <= 0.5 -> class 0, > 0.5
		 * -> class 1
		 */
		pred_class = (prediction > 0.5) ? 1 : 0;

		if (i < 10 || (i % 100 == 0))
		{
		}

		if (true_class == 1 && pred_class == 1)
			tp++;
		else if (true_class == 0 && pred_class == 0)
			tn++;
		else if (true_class == 0 && pred_class == 1)
			fp++;
		else if (true_class == 1 && pred_class == 0)
			fn++;
	}


	if (tp_out)
		*tp_out = tp;
	if (tn_out)
		*tn_out = tn;
	if (fp_out)
		*fp_out = fp;
	if (fn_out)
		*fn_out = fn;
}
#endif

/*
 * evaluate_decision_tree_by_model_id
 *
 * Evaluates Decision Tree model by model_id using optimized batch evaluation.
 * Supports both GPU and CPU models with GPU-accelerated batch evaluation when available.
 *
 * Returns jsonb with metrics: accuracy, precision, recall, f1_score, n_samples
 */
PG_FUNCTION_INFO_V1(evaluate_decision_tree_by_model_id);

Datum
evaluate_decision_tree_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *table_name = NULL;
	text *feature_col = NULL;
	text *label_col = NULL;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	int			ret;
	int			nvec = 0;
	int			i;
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
	StringInfoData query;

	DTModel *model = NULL;

	Jsonb *result_jsonb = NULL;
	bytea *gpu_payload = NULL;
	Jsonb *gpu_metrics = NULL;
	bool		is_gpu_model = false;

	NdbSpiSession *spi_session = NULL;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_decision_tree_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	/* Validate model_id before attempting to load */
	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_decision_tree_by_model_id: model_id must be positive, got %d", model_id),
				 errdetail("Invalid model_id: %d", model_id),
				 errhint("Provide a valid model_id from neurondb.ml_models catalog.")));

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_decision_tree_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	if (!dt_load_model_from_catalog(model_id, &model))
	{
		if (ml_catalog_fetch_model_payload(model_id, &gpu_payload, NULL, &gpu_metrics))
		{
			/* Validate GPU payload */
			if (gpu_payload == NULL)
			{
				if (gpu_metrics)
					nfree(gpu_metrics);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_decision_tree_by_model_id: model %d has NULL payload",
								model_id)));
			}
			
			/* Check if this is a GPU model - either by metrics or by payload format */
			{
				uint32		payload_size;

				/* First check metrics for training_backend */
				if (dt_metadata_is_gpu(gpu_metrics))
				{
					is_gpu_model = true;
				}
				else
				{
					/* If metrics check didn't find GPU indicator, check payload format */
					/* GPU models start with NdbCudaDtModelHeader, CPU models start with uint8 training_backend */
					payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
					
					/* CPU format: first byte is training_backend (uint8), then model_id (int32) */
					/* GPU format: first field is feature_dim (int32) */
					/* Check if payload looks like GPU format (starts with int32, not uint8) */
					if (payload_size >= sizeof(int32))
					{
						const int32 *first_int = (const int32 *) VARDATA(gpu_payload);
						int32		first_value = *first_int;
						
						/* If first 4 bytes look like a reasonable feature_dim, check for GPU format */
						if (first_value > 0 && first_value <= 100000)
						{
							/* Check if payload size matches GPU format */
							if (payload_size >= sizeof(NdbCudaDtModelHeader))
							{
								const NdbCudaDtModelHeader *hdr = (const NdbCudaDtModelHeader *) VARDATA(gpu_payload);
								
								/* Validate header fields match the first int32 */
								if (hdr->feature_dim == first_value &&
									hdr->n_samples >= 0 && hdr->n_samples <= 1000000000 &&
									hdr->node_count > 0 && hdr->node_count <= 1000000)
								{
									size_t		expected_gpu_size = sizeof(NdbCudaDtModelHeader) +
										sizeof(NdbCudaDtNode) * (size_t) hdr->node_count;
									
									/* Size matches GPU format - likely a GPU model */
									if (payload_size >= expected_gpu_size && payload_size < expected_gpu_size + 1000)
									{
										is_gpu_model = true;
									}
								}
							}
						}
					}
				}
			}
			
			if (!is_gpu_model)
			{
				if (gpu_payload)
					nfree(gpu_payload);
				if (gpu_metrics)
					nfree(gpu_metrics);
				nfree(tbl_str);
				nfree(feat_str);
				nfree(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_decision_tree_by_model_id: model %d exists but has corrupted data and no GPU fallback available",
								model_id)));
			}
		}
		else
		{
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: model %d not found",
							model_id)));
		}
	}

	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 quote_identifier(feat_str),
					 quote_identifier(targ_str),
					 quote_identifier(tbl_str),
					 quote_identifier(feat_str),
					 quote_identifier(targ_str));

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		if (model != NULL)
		{
			if (model->root != NULL)
				dt_free_tree(model->root);
			nfree(model);
		}
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_decision_tree_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 1)
	{
		StringInfoData check_query;
		int			check_ret;
		int			total_rows = 0;

		ndb_spi_stringinfo_init(spi_session, &check_query);
		appendStringInfo(&check_query,
						 "SELECT COUNT(*) FROM %s",
						 quote_identifier(tbl_str));
		check_ret = ndb_spi_execute(spi_session, check_query.data, true, 0);
		if (check_ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			/* Use safe function to get int32 count (will be cast to int64) */
			int32		count_val_int32;

			if (ndb_spi_get_int32(spi_session, 0, 1, &count_val_int32))
			{
				int64		count_val = (int64) count_val_int32;

				total_rows = count_val;
			}
		}
		ndb_spi_stringinfo_free(spi_session, &check_query);

		if (model != NULL)
		{
			if (model->root != NULL)
				dt_free_tree(model->root);
			nfree(model);
		}
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);

		/* Report error using tbl_str/feat_str/targ_str before freeing them */
		if (total_rows == 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: table/view '%s' has no rows",
							tbl_str),
					 errhint("Ensure the table/view exists and contains data")));
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: no valid rows found in '%s' (table has %d total rows, but all have NULL values in '%s' or '%s')",
							tbl_str, total_rows, feat_str, targ_str),
					 errhint("Ensure columns '%s' and '%s' are not NULL for evaluation rows",
							 feat_str, targ_str)));
		}
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
	}

	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	/* Unified evaluation: Determine predict function based on compute mode */
	/* All metrics calculation is the same - only difference is predict function */
	{
		bool		use_gpu_predict = false;
		int			processed_count = 0;
		int			feat_dim = 0;
		const NdbCudaDtModelHeader *gpu_hdr = NULL;

		/* Determine if we should use GPU predict or CPU predict based on compute mode */
		if (is_gpu_model && neurondb_gpu_is_available() && !NDB_COMPUTE_MODE_IS_CPU())
		{
			/* GPU model and GPU mode: use GPU predict */
			if (gpu_payload != NULL && VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaDtModelHeader))
			{
				gpu_hdr = (const NdbCudaDtModelHeader *) VARDATA(gpu_payload);
				feat_dim = gpu_hdr->feature_dim;
				use_gpu_predict = true;
			}
		}
		else if (model != NULL)
		{
			/* CPU model or CPU mode: use CPU predict */
			feat_dim = model->n_features;
			use_gpu_predict = false;
		}
		else if (is_gpu_model && gpu_payload != NULL)
		{
			/* GPU model but CPU mode: convert to CPU format for CPU predict */
			if (VARSIZE(gpu_payload) - VARHDRSZ >= sizeof(NdbCudaDtModelHeader))
			{
				gpu_hdr = (const NdbCudaDtModelHeader *) VARDATA(gpu_payload);
				feat_dim = gpu_hdr->feature_dim;

				/* Try to deserialize GPU model as CPU model */
				model = dt_model_deserialize(gpu_payload, NULL);
				if (model == NULL)
				{
					NDB_SPI_SESSION_END(spi_session);
					nfree(gpu_payload);
					nfree(gpu_metrics);
					nfree(tbl_str);
					nfree(feat_str);
					nfree(targ_str);
					ndb_spi_stringinfo_free(spi_session, &query);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: evaluate_decision_tree_by_model_id: failed to convert GPU model to CPU format"),
							 errdetail("GPU model conversion failed for model %d", model_id),
							 errhint("GPU model cannot be evaluated in CPU mode. Use GPU mode or retrain the model.")));
				}
				use_gpu_predict = false;
			}
		}

		/* Ensure we have a valid model or GPU payload */
		if (model == NULL && !use_gpu_predict)
		{
			NDB_SPI_SESSION_END(spi_session);
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: no valid model found"),
					 errdetail("Neither CPU model nor GPU payload is available"),
					 errhint("Verify the model exists in the catalog and is in the correct format.")));
		}

		if (feat_dim <= 0)
		{
			NDB_SPI_SESSION_END(spi_session);
			if (model != NULL)
			{
				if (model->root != NULL)
					dt_free_tree(model->root);
				nfree(model);
			}
			nfree(gpu_payload);
			nfree(gpu_metrics);
			nfree(tbl_str);
			nfree(feat_str);
			nfree(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: invalid feature dimension %d",
							feat_dim)));
		}

		/* Unified evaluation loop - prediction based on compute mode */
		for (i = 0; i < nvec; i++)
		{
			HeapTuple	tuple;
			TupleDesc	tupdesc;
			Datum		feat_datum;
			Datum		targ_datum;
			bool		feat_null;
			bool		targ_null;
			int			y_true;
			int			y_pred = -1;
			int			actual_dim;
			float	   *feat_row = NULL;

			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
				i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
				continue;

			tuple = SPI_tuptable->vals[i];
			tupdesc = SPI_tuptable->tupdesc;
			if (tupdesc == NULL)
				continue;

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
			if (tupdesc->natts < 2)
				continue;
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

			if (feat_null || targ_null)
				continue;

			/* Handle both integer and float label types */
			{
				Oid			label_type = SPI_gettypeid(tupdesc, 2);
				
				if (label_type == INT4OID || label_type == INT2OID || label_type == INT8OID)
					y_true = DatumGetInt32(targ_datum);
				else
					y_true = (int) rint(DatumGetFloat8(targ_datum));
			}

			/* Extract features and determine dimension */
			if (feat_is_array)
			{
				ArrayType *arr = DatumGetArrayTypeP(feat_datum);
				if (arr == NULL || ARR_NDIM(arr) != 1)
					continue;
				actual_dim = ARR_DIMS(arr)[0];
			}
			else
			{
				Vector *vec = DatumGetVector(feat_datum);
				if (vec == NULL)
					continue;
				actual_dim = vec->dim;
			}

			/* Validate feature dimension matches model */
			if (actual_dim != feat_dim)
				continue;

			/* Extract features to float array for prediction */
			{
				int			j;
				double		prediction = 0.0;

				nalloc(feat_row, float, feat_dim);
				if (feat_is_array)
				{
					ArrayType *arr = DatumGetArrayTypeP(feat_datum);
					if (feat_type_oid == FLOAT8ARRAYOID)
					{
						float8	   *data = (float8 *) ARR_DATA_PTR(arr);
						for (j = 0; j < feat_dim; j++)
							feat_row[j] = (float) data[j];
					}
					else
					{
						float4	   *data = (float4 *) ARR_DATA_PTR(arr);
						memcpy(feat_row, data, sizeof(float) * feat_dim);
					}
				}
				else
				{
					Vector *vec = DatumGetVector(feat_datum);
					memcpy(feat_row, vec->data, sizeof(float) * feat_dim);
				}

				/* Call appropriate predict function based on compute mode */
				if (use_gpu_predict)
				{
					/* GPU predict path - prediction based on compute mode */
#ifdef NDB_GPU_CUDA
					int			predict_rc;
					char	   *gpu_err = NULL;

					predict_rc = ndb_gpu_dt_predict(gpu_payload,
													 feat_row,
													 feat_dim,
													 &prediction,
													 &gpu_err);
					if (predict_rc != 0)
					{
						/* GPU predict failed - check compute mode */
						if (NDB_REQUIRE_GPU())
						{
							/* Strict GPU mode: error out */
							if (gpu_err)
								nfree(gpu_err);
							nfree(feat_row);
							NDB_SPI_SESSION_END(spi_session);
							if (model != NULL)
							{
								if (model->root != NULL)
									dt_free_tree(model->root);
								nfree(model);
							}
							nfree(gpu_payload);
							nfree(gpu_metrics);
							nfree(tbl_str);
							nfree(feat_str);
							nfree(targ_str);
							ndb_spi_stringinfo_free(spi_session, &query);
							ereport(ERROR,
									(errcode(ERRCODE_INTERNAL_ERROR),
									 errmsg("neurondb: evaluate_decision_tree_by_model_id: GPU prediction failed in GPU mode"),
									 errdetail("GPU prediction failed for row %d: %s", i, gpu_err ? gpu_err : "unknown error"),
									 errhint("GPU mode requires GPU prediction to succeed. Check GPU availability and model compatibility.")));
						}
						else
						{
							/* AUTO mode: fall back to CPU if available */
							if (gpu_err)
								nfree(gpu_err);
							if (model != NULL)
							{
								/* Use CPU model for prediction */
								prediction = dt_tree_predict(model->root, feat_row, feat_dim);
							}
							else
							{
								/* No CPU model available - cannot fall back */
								if (gpu_err)
									nfree(gpu_err);
								nfree(feat_row);
								continue;
							}
						}
					}
					if (gpu_err)
						nfree(gpu_err);
#endif
				}
				else
				{
					/* CPU predict path - prediction based on compute mode */
					if (model == NULL || model->root == NULL)
					{
						nfree(feat_row);
						continue;
					}

					/* Use CPU model for prediction */
					prediction = dt_tree_predict(model->root, feat_row, feat_dim);
				}

				y_pred = (int) rint(prediction);
			}

			/* Clamp y_pred to valid binary classification range [0, 1] */
			if (y_pred < 0)
				y_pred = 0;
			else if (y_pred > 1)
				y_pred = 1;

			/* Compute confusion matrix (same for both CPU and GPU) */
			if (y_true == 1)
			{
				if (y_pred == 1)
					tp++;
				else
					fn++;
			}
			else
			{
				if (y_pred == 1)
					fp++;
				else
					tn++;
			}

			processed_count++;
			nfree(feat_row);
		}

		/* Calculate metrics from confusion matrix (same for both CPU and GPU) */
		if (processed_count > 0)
		{
			accuracy = (double) (tp + tn) / (tp + tn + fp + fn);
			precision = (tp + fp > 0) ? (double) tp / (tp + fp) : 0.0;
			recall = (tp + fn > 0) ? (double) tp / (tp + fn) : 0.0;
			f1_score = (precision + recall > 0)
				? 2.0 * precision * recall / (precision + recall)
				: 0.0;
		}
		else
		{
			accuracy = 0.0;
			precision = 0.0;
			recall = 0.0;
			f1_score = 0.0;
		}

		/* Cleanup */
		if (model != NULL)
		{
			if (model->root != NULL)
				dt_free_tree(model->root);
			nfree(model);
		}
		if (gpu_payload != NULL)
			nfree(gpu_payload);
		if (gpu_metrics != NULL)
			nfree(gpu_metrics);
	}

	/* Build JSONB result */
	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	/* Switch to old context and build JSONB directly using JSONB API */
	MemoryContextSwitchTo(oldcontext);
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;
		JsonbValue *final_value = NULL;
		Numeric		accuracy_num,
					precision_num,
					recall_num,
					f1_score_num,
					n_samples_num;

		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);

			jkey.type = jbvString;
			jkey.val.string.val = "accuracy";
			jkey.val.string.len = strlen("accuracy");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			accuracy_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(accuracy)));
			jval.type = jbvNumeric;
			jval.val.numeric = accuracy_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "precision";
			jkey.val.string.len = strlen("precision");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			precision_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(precision)));
			jval.type = jbvNumeric;
			jval.val.numeric = precision_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "recall";
			jkey.val.string.len = strlen("recall");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			recall_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(recall)));
			jval.type = jbvNumeric;
			jval.val.numeric = recall_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "f1_score";
			jkey.val.string.len = strlen("f1_score");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			f1_score_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum(f1_score)));
			jval.type = jbvNumeric;
			jval.val.numeric = f1_score_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			jkey.val.string.val = "n_samples";
			jkey.val.string.len = strlen("n_samples");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(tp + tn + fp + fn)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_samples_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);

			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);

			if (final_value == NULL)
			{
				elog(ERROR, "neurondb: evaluate_decision_tree_by_model_id: pushJsonbValue(WJB_END_OBJECT) returned NULL");
			}

			result_jsonb = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			ErrorData  *edata = CopyErrorData();

			elog(ERROR, "neurondb: evaluate_decision_tree_by_model_id: JSONB construction failed: %s", edata->message);
			FlushErrorState();
			result_jsonb = NULL;
		}
		PG_END_TRY();
	}

	if (result_jsonb == NULL)
	{
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_decision_tree_by_model_id: failed to create JSONB result")));
	}

	nfree(tbl_str);
	nfree(feat_str);
	nfree(targ_str);
	PG_RETURN_JSONB_P(result_jsonb);
}

/* Old GPU evaluation kernel code removed - replaced with unified evaluation pattern above */

#include "neurondb_gpu_model.h"

typedef struct DTGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	int			feature_dim;
	int			n_samples;
}			DTGpuModelState;

static void
dt_gpu_release_state(DTGpuModelState * state)
{
	if (state == NULL)
		return;
	if (state->model_blob != NULL)
		nfree(state->model_blob);
	if (state->metrics != NULL)
		nfree(state->metrics);
	nfree(state);
}

static bool
dt_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	DTGpuModelState *state = NULL;
	bytea *payload = NULL;
	Jsonb *metrics = NULL;
	int			rc;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
		return false;
	if (!neurondb_gpu_is_available())
		return false;
	if (spec->feature_matrix == NULL || spec->label_vector == NULL)
		return false;
	if (spec->sample_count <= 0 || spec->feature_dim <= 0)
		return false;

	/* Initialize output parameters to NULL */
	payload = NULL;
	metrics = NULL;

	rc = ndb_gpu_dt_train(spec->feature_matrix,
						  spec->label_vector,
						  spec->sample_count,
						  spec->feature_dim,
						  spec->hyperparameters,
						  &payload,
						  &metrics,
						  errstr);

	if (rc != 0 || payload == NULL)
	{
		/* On error, GPU backend should have cleaned up - don't free here */
		return false;
	}

	if (model->backend_state != NULL)
	{
		dt_gpu_release_state((DTGpuModelState *) model->backend_state);
		model->backend_state = NULL;
	}

	state = (DTGpuModelState *) palloc0(sizeof(DTGpuModelState));
	state->model_blob = payload;
	state->feature_dim = spec->feature_dim;
	state->n_samples = spec->sample_count;

	if (metrics != NULL)
	{
		state->metrics = metrics;
	}
	else
	{
		state->metrics = NULL;
	}

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;

	return true;
}

static bool
dt_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
			   float *output, int output_dim, char **errstr)
{
	const		DTGpuModelState *state;
	double		prediction;
	int			rc;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
		return false;
	if (output_dim <= 0)
		return false;
	if (!model->gpu_ready || model->backend_state == NULL)
		return false;

	state = (const DTGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	rc = ndb_gpu_dt_predict(state->model_blob, input,
							state->feature_dim > 0 ? state->feature_dim : input_dim,
							&prediction, errstr);
	if (rc != 0)
		return false;

	output[0] = (float) prediction;
	return true;
}

static bool
dt_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
				MLGpuMetrics *out, char **errstr)
{
	const		DTGpuModelState *state;
	Jsonb	   *metrics_json = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || out == NULL)
		return false;
	if (model->backend_state == NULL)
		return false;

	state = (const DTGpuModelState *) model->backend_state;
	{
		StringInfoData buf;

		initStringInfo(&buf);
		appendStringInfo(&buf,
						 "{\"algorithm\":\"decision_tree\",\"storage\":\"gpu\",\"n_features\":%d,\"n_samples\":%d}",
						 state->feature_dim > 0 ? state->feature_dim : 0,
						 state->n_samples > 0 ? state->n_samples : 0);
		/* Use ndb_jsonb_in_cstring (consistent with other ML algorithms) */
		metrics_json = ndb_jsonb_in_cstring(buf.data);
		if (metrics_json == NULL)
		{
			nfree(buf.data);
			if (errstr != NULL)
				*errstr = pstrdup("failed to parse metrics JSON");
			return false;
		}
		nfree(buf.data);
	}
	if (out != NULL)
		out->payload = metrics_json;
	return true;
}

static bool
dt_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
				 Jsonb * *metadata_out, char **errstr)
{
	const		DTGpuModelState *state;
	bytea *payload_copy = NULL;
	char *payload_copy_raw = NULL;
	int			payload_size;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
		return false;

	state = (const DTGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
		return false;

	payload_size = VARSIZE(state->model_blob);
	nalloc(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		nfree(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
	{
		/* Copy Jsonb directly without using DirectFunctionCall (unsafe in GPU context) */
		/* Use PG_DETOAST_DATUM_COPY to create a copy in current memory context */
		Jsonb  *metrics_copy = (Jsonb *) PG_DETOAST_DATUM_COPY(PointerGetDatum(state->metrics));
		*metadata_out = metrics_copy;
	}
	else if (metadata_out != NULL)
	{
		*metadata_out = NULL;
	}
	return true;
}

static bool
dt_gpu_deserialize(MLGpuModel *model, const bytea * payload,
				   const Jsonb * metadata, char **errstr)
{
	DTGpuModelState *state = NULL;
	bytea *payload_copy = NULL;
	char *payload_copy_raw = NULL;
	int			payload_size;
	int			feature_dim = -1;
	int			n_samples = -1;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	nalloc(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
	memcpy(payload_copy, payload, payload_size);

	/* Extract feature_dim and n_samples from metadata if available */
	if (metadata != NULL)
	{
		JsonbIterator *it = NULL;
		JsonbValue	v;
		int			r;

		PG_TRY();
		{
			it = JsonbIteratorInit((JsonbContainer *) & metadata->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_KEY && v.type == jbvString)
				{
					char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

					r = JsonbIteratorNext(&it, &v, false);
					if (strcmp(key, "n_features") == 0 && v.type == jbvNumeric)
					{
						feature_dim = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																		NumericGetDatum(v.val.numeric)));
					}
					else if (strcmp(key, "n_samples") == 0 && v.type == jbvNumeric)
					{
						n_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																	  NumericGetDatum(v.val.numeric)));
					}
					nfree(key);
				}
			}
		}
		PG_CATCH();
		{
			/* If metadata parsing fails, use defaults */
			feature_dim = -1;
			n_samples = -1;
		}
		PG_END_TRY();
	}

	state = (DTGpuModelState *) palloc0(sizeof(DTGpuModelState));
	state->model_blob = payload_copy;
	state->feature_dim = feature_dim;
	state->n_samples = n_samples;

	if (metadata != NULL)
	{
		text   *metadata_text = DatumGetTextP(DirectFunctionCall1(jsonb_out,
											 PointerGetDatum(metadata)));
		char   *metadata_cstr = text_to_cstring(metadata_text);

		state->metrics = ndb_jsonb_in_cstring(metadata_cstr);
		pfree(metadata_text);
		nfree(metadata_cstr);
	}
	else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
		dt_gpu_release_state((DTGpuModelState *) model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = true;
	return true;
}

static void
dt_gpu_destroy(MLGpuModel *model)
{
	if (model == NULL)
		return;
	if (model->backend_state != NULL)
		dt_gpu_release_state((DTGpuModelState *) model->backend_state);
	model->backend_state = NULL;
	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps dt_gpu_model_ops = {
	.algorithm = "decision_tree",
	.train = dt_gpu_train,
	.predict = dt_gpu_predict,
	.evaluate = dt_gpu_evaluate,
	.serialize = dt_gpu_serialize,
	.deserialize = dt_gpu_deserialize,
	.destroy = dt_gpu_destroy,
};

void
neurondb_gpu_register_dt_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&dt_gpu_model_ops);
	registered = true;
}
