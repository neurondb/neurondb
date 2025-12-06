/*-------------------------------------------------------------------------
 *
 * ml_decision_tree.c
 *    Decision tree implementation.
 *
 * This module implements CART decision trees for classification and regression,
 * with model serialization and catalog storage.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
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

#ifdef NDB_GPU_CUDA
#include "neurondb_cuda_runtime.h"
#include <cublas_v2.h>
extern cublasHandle_t ndb_cuda_get_cublas_handle(void);
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
		NDB_FREE(dataset->features);
	if (dataset->labels)
		NDB_FREE(dataset->labels);

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
	StringInfoData query;
	MemoryContext oldcontext;
	int			ret;
	int			n_samples = 0;
	int			feature_dim = 0;
	int			i = 0;

	NDB_DECLARE(NdbSpiSession *, dt_load_spi_session);

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
	elog(DEBUG1, "dt_dataset_load: executing query: %s", query.data);

	NDB_SPI_SESSION_BEGIN(dt_load_spi_session, oldcontext);

	ret = ndb_spi_execute_safe(query.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT)
	{
		NDB_FREE(query.data);
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
		NDB_FREE(query.data);
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
		NDB_FREE(query.data);
		NDB_SPI_SESSION_END(dt_load_spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: dt_dataset_load: could not determine feature dimension"),
				 errdetail("No valid feature vectors found in the first row"),
				 errhint("Ensure the feature column contains valid array or vector data.")));
	}

	MemoryContextSwitchTo(oldcontext);

	NDB_ALLOC(dataset->features, float, ((Size) n_samples) * ((Size) feature_dim));
	NDB_ALLOC(dataset->labels, double, (Size) n_samples);

	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	tuple;
		TupleDesc	tupdesc;
		Datum		feat_datum;
		Datum		label_datum;
		bool		feat_null = false;
		bool		label_null = false;
		Oid			feat_type;
		float	   *row;

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
			float8	   *data;

			if (arr_dim != feature_dim)
			{
				NDB_FREE(query.data);
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
			float4	   *data;

			if (arr_dim != feature_dim)
			{
				NDB_FREE(query.data);
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
				NDB_FREE(query.data);
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

	NDB_FREE(query.data);
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

			NDB_DECLARE(double *, left_y);
			NDB_DECLARE(double *, right_y);

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

			NDB_ALLOC(left_y, double, left_count);
			NDB_ALLOC(right_y, double, right_count);

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

			NDB_FREE(left_y);
			NDB_FREE(right_y);
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
	DTNode	   *node;
	int			i,
				best_feature;
	float		best_threshold;
	double		best_gain;

	NDB_DECLARE(int *, left_indices);
	NDB_DECLARE(int *, right_indices);
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

	NDB_ALLOC(left_indices, int, n_samples);
	NDB_ALLOC(right_indices, int, n_samples);

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

	NDB_FREE(left_indices);
	NDB_FREE(right_indices);

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
	NDB_FREE(node);
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
	DTNode	   *node;
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
	NDB_DECLARE(bytea *, result);
	NDB_DECLARE(char *, result_raw);

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

	NDB_ALLOC(result_raw, char, VARHDRSZ + buf.len);
	result = (bytea *) result_raw;
	SET_VARSIZE(result, VARHDRSZ + buf.len);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

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
	DTModel    *model;
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
		NDB_FREE(model);
		return NULL;
	}

	/* Read header fields with bounds checking */
	if (buf.cursor + 4 > buf.len)
	{
		elog(WARNING, "dt: buffer overflow reading model_id, cursor=%d, len=%d", buf.cursor, buf.len);
		NDB_FREE(model);
		return NULL;
	}
	model->model_id = pq_getmsgint(&buf, 4);

	if (buf.cursor + 4 > buf.len)
	{
		elog(WARNING, "dt: buffer overflow reading n_features, cursor=%d, len=%d", buf.cursor, buf.len);
		NDB_FREE(model);
		return NULL;
	}
	model->n_features = pq_getmsgint(&buf, 4);

	if (buf.cursor + 4 > buf.len)
	{
		elog(WARNING, "dt: buffer overflow reading n_samples, cursor=%d, len=%d", buf.cursor, buf.len);
		NDB_FREE(model);
		return NULL;
	}
	model->n_samples = pq_getmsgint(&buf, 4);

	if (buf.cursor + 4 > buf.len)
	{
		elog(WARNING, "dt: buffer overflow reading max_depth, cursor=%d, len=%d", buf.cursor, buf.len);
		NDB_FREE(model);
		return NULL;
	}
	model->max_depth = pq_getmsgint(&buf, 4);

	if (buf.cursor + 4 > buf.len)
	{
		elog(WARNING, "dt: buffer overflow reading min_samples_split, cursor=%d, len=%d", buf.cursor, buf.len);
		NDB_FREE(model);
		return NULL;
	}
	model->min_samples_split = pq_getmsgint(&buf, 4);

	elog(DEBUG1, "dt_model_deserialize: model_id=%d, n_features=%d, n_samples=%d, max_depth=%d, min_samples_split=%d, data_size=%d",
		 model->model_id, model->n_features, model->n_samples, model->max_depth, model->min_samples_split, buf.len);

	if (model->n_features <= 0 || model->n_features > 10000)
	{
		elog(WARNING, "dt: invalid n_features %d in deserialized model, treating as corrupted", model->n_features);
		NDB_FREE(model);
		return NULL;
	}
	if (model->n_samples < 0 || model->n_samples > 100000000)
	{
		elog(WARNING, "dt: invalid n_samples %d in deserialized model, treating as corrupted", model->n_samples);
		NDB_FREE(model);
		return NULL;
	}

	/* Deserialize the tree structure */
	model->root = dt_deserialize_node(&buf);
	if (model->root == NULL)
	{
		NDB_FREE(model);
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
	JsonbIterator *it;
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
			NDB_FREE(key);
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
	NDB_DECLARE(bytea *, payload);
	NDB_DECLARE(Jsonb *, metrics);
	NDB_DECLARE(char *, gpu_err);
	double		prediction = 0.0;
	bool		success = false;

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

	if (!dt_metadata_is_gpu(metrics))
		goto cleanup;

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
		NDB_FREE(payload);
	if (metrics)
		NDB_FREE(metrics);
	if (gpu_err)
		NDB_FREE(gpu_err);

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
	NDB_DECLARE(bytea *, payload);
	NDB_DECLARE(Jsonb *, metrics);

	if (out == NULL)
		return false;
	*out = NULL;

	if (!ml_catalog_fetch_model_payload(model_id, &payload, NULL, &metrics))
		return false;

	if (payload == NULL)
	{
		if (metrics)
			NDB_FREE(metrics);
		return false;
	}

	*out = dt_model_deserialize(payload, NULL);
	if (*out == NULL)
	{
		if (metrics)
			NDB_FREE(metrics);
		return false;
	}

	if (metrics)
		NDB_FREE(metrics);

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
 */
Datum
train_decision_tree_classifier(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	int32		max_depth;
	int32		min_samples_split;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *label_str;
	DTDataset	dataset;
	const char *quoted_tbl;
	const char *quoted_feat;
	const char *quoted_label;
	MLGpuTrainResult gpu_result;

	NDB_DECLARE(char *, gpu_err);
	NDB_DECLARE(Jsonb *, gpu_hyperparams);
	StringInfoData hyperbuf;
	int32		model_id = 0;

	NDB_DECLARE(int *, indices);
	NDB_DECLARE(DTModel *, model);
	NDB_DECLARE(bytea *, model_blob);
	MLCatalogModelSpec spec;
	StringInfoData paramsbuf;
	StringInfoData metricsbuf;

	NDB_DECLARE(Jsonb *, params_jsonb);
	NDB_DECLARE(Jsonb *, metrics_jsonb);
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
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(label_str);
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
		/* Use ndb_jsonb_in_cstring like other ML algorithms fix */
		gpu_hyperparams = ndb_jsonb_in_cstring(hyperbuf.data);
		if (gpu_hyperparams == NULL)
		{
			NDB_FREE(hyperbuf.data);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("neurondb: train_decision_tree_classifier: failed to parse GPU hyperparameters JSON")));
		}
		NDB_FREE(hyperbuf.data);
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
			elog(DEBUG1, "neurondb: dt: GPU training succeeded");
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
				NDB_FREE(gpu_err);
			if (gpu_hyperparams)
				NDB_FREE(gpu_hyperparams);
			ndb_gpu_free_train_result(&gpu_result);
			if (hyperbuf.data != NULL)
			{
				NDB_FREE(hyperbuf.data);
				hyperbuf.data = NULL;
			}
			dt_dataset_free(&dataset);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(label_str);

			PG_RETURN_INT32(model_id);
		}
		if (gpu_err)
		{
			elog(DEBUG1, "neurondb: dt: GPU training failed: %s", gpu_err);
			NDB_FREE(gpu_err);
		}
		else
		{
			elog(DEBUG1, "neurondb: dt: GPU training returned false (no error string)");
		}
		if (gpu_hyperparams)
			NDB_FREE(gpu_hyperparams);

		ndb_gpu_free_train_result(&gpu_result);
		if (hyperbuf.data != NULL)
		{
			NDB_FREE(hyperbuf.data);
			hyperbuf.data = NULL;
		}

	}

	NDB_ALLOC(indices, int, dataset.n_samples);

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
		NDB_FREE(model);
		NDB_FREE(indices);
		dt_dataset_free(&dataset);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(label_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: dt: failed to serialize model")));
	}

	initStringInfo(&paramsbuf);
	appendStringInfo(&paramsbuf, "{\"max_depth\":%d,\"min_samples_split\":%d}",
					 max_depth, min_samples_split);
	/* Use ndb_jsonb_in_cstring like other ML algorithms fix */
	params_jsonb = ndb_jsonb_in_cstring(paramsbuf.data);
	if (params_jsonb == NULL)
	{
		NDB_FREE(paramsbuf.data);
		dt_free_tree(model->root);
		NDB_FREE(model);
		NDB_FREE(indices);
		dt_dataset_free(&dataset);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(label_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("neurondb: train_decision_tree_classifier: failed to parse parameters JSON")));
	}
	NDB_FREE(paramsbuf.data);
	paramsbuf.data = NULL;

	initStringInfo(&metricsbuf);
	appendStringInfo(&metricsbuf,
					 "{\"algorithm\":\"decision_tree\","
					 "\"storage\":\"cpu\","
					 "\"n_features\":%d,"
					 "\"n_samples\":%d,"
					 "\"max_depth\":%d,"
					 "\"min_samples_split\":%d}",
					 dataset.feature_dim, dataset.n_samples, max_depth, min_samples_split);
	/* Use ndb_jsonb_in_cstring like other ML algorithms fix */
	metrics_jsonb = ndb_jsonb_in_cstring(metricsbuf.data);
	if (metrics_jsonb == NULL)
	{
		if (paramsbuf.data != NULL)
			NDB_FREE(paramsbuf.data);
		NDB_FREE(metricsbuf.data);
		dt_free_tree(model->root);
		NDB_FREE(model);
		NDB_FREE(indices);
		dt_dataset_free(&dataset);
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(label_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("neurondb: train_decision_tree_classifier: failed to parse metrics JSON")));
	}
	NDB_FREE(metricsbuf.data);
	metricsbuf.data = NULL;

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
	NDB_FREE(model);
	NDB_FREE(indices);
	NDB_FREE(model_blob);

	NDB_FREE(paramsbuf.data);
	NDB_FREE(metricsbuf.data);

	dt_dataset_free(&dataset);
	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);
	NDB_FREE(label_str);

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
	Vector	   *feature_vec;
	DTModel    *model;

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
			NDB_FREE(model);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("dt: model %d has NULL root (corrupted?)", model_id)));
		}

		if (feature_vec->dim != model->n_features)
		{
			NDB_FREE(model);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							errmsg("dt: feature dimension mismatch: expected %d, got %d",
								   model->n_features, feature_vec->dim)));
		}

		result = dt_tree_predict(model->root, feature_vec->data, model->n_features);

		dt_free_tree(model->root);
		NDB_FREE(model);
	}

	PG_RETURN_FLOAT8(result);
}

/*
 * dt_predict_batch
 *
 * Helper function to predict a batch of samples using Decision Tree model.
 * Updates confusion matrix.
 */
static void
dt_predict_batch(const DTModel *model,
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
		elog(DEBUG1,
			 "neurondb: dt_predict_batch: early return - model=%p, model->root=%p, features=%p, labels=%p, n_samples=%d",
			 (void *) model, model ? (void *) model->root : NULL, (void *) features, (void *) labels, n_samples);
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

	elog(DEBUG1,
		 "neurondb: dt_predict_batch: starting - n_samples=%d, feature_dim=%d, model->n_features=%d",
		 n_samples, feature_dim, model->n_features);

	for (i = 0; i < n_samples; i++)
	{
		const float *row = features + (i * feature_dim);
		double		y_true = labels[i];
		int			true_class;
		double		prediction;
		int			pred_class;

		if (!isfinite(y_true))
		{
			elog(DEBUG1, "neurondb: dt_predict_batch: sample %d - skipping non-finite y_true=%.6f", i, y_true);
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
			elog(DEBUG1, "neurondb: dt_predict_batch: sample %d - skipping non-finite prediction=%.6f", i, prediction);
			continue;
		}

		/*
		 * Convert prediction to binary class: values <= 0.5 -> class 0, > 0.5
		 * -> class 1
		 */
		pred_class = (prediction > 0.5) ? 1 : 0;

		if (i < 10 || (i % 100 == 0))
		{
			elog(DEBUG1,
				 "neurondb: dt_predict_batch: sample %d - y_true=%.6f (class=%d), prediction=%.6f (class=%d)",
				 i, y_true, true_class, prediction, pred_class);
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

	/* Count actual class distribution for debugging */
	{
		int			class0_count = 0,
					class1_count = 0;

		for (int j = 0; j < i; j++)
		{
			double		y = labels[j];

			if (isfinite(y))
			{
				int			c = (y > 0.5) ? 1 : 0;

				if (c == 0)
					class0_count++;
				else
					class1_count++;
			}
		}
		elog(DEBUG1,
			 "neurondb: dt_predict_batch: summary - processed=%d, class0=%d, class1=%d, tp=%d, tn=%d, fp=%d, fn=%d",
			 i, class0_count, class1_count, tp, tn, fp, fn);
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
	text	   *table_name;
	text	   *feature_col;
	text	   *label_col;
	char	   *tbl_str;
	char	   *feat_str;
	char	   *targ_str;
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
	StringInfoData query;

	NDB_DECLARE(DTModel *, model);
	StringInfoData jsonbuf;

	NDB_DECLARE(Jsonb *, result_jsonb);
	NDB_DECLARE(bytea *, gpu_payload);
	NDB_DECLARE(Jsonb *, gpu_metrics);
	bool		is_gpu_model = false;

	NDB_DECLARE(NdbSpiSession *, spi_session);

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_decision_tree_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

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
			is_gpu_model = dt_metadata_is_gpu(gpu_metrics);
			/* Validate GPU payload */
			if (gpu_payload == NULL)
			{
				NDB_FREE(gpu_metrics);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_decision_tree_by_model_id: model %d has NULL payload",
								model_id)));
			}
			is_gpu_model = dt_metadata_is_gpu(gpu_metrics);
			if (!is_gpu_model)
			{
				NDB_FREE(gpu_payload);
				NDB_FREE(gpu_metrics);
				NDB_FREE(tbl_str);
				NDB_FREE(feat_str);
				NDB_FREE(targ_str);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: evaluate_decision_tree_by_model_id: model %d exists but has corrupted data and no GPU fallback available",
								model_id)));
			}
		}
		else
		{
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: model %d exists but has corrupted data and no GPU fallback available",
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
	elog(DEBUG1, "evaluate_decision_tree_by_model_id: executing query: %s", query.data);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		if (model != NULL)
		{
			if (model->root != NULL)
				dt_free_tree(model->root);
			NDB_FREE(model);
		}
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
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
			NDB_FREE(model);
		}
		NDB_FREE(tbl_str);
		NDB_FREE(feat_str);
		NDB_FREE(targ_str);
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);

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
	}

	if (SPI_tuptable != NULL && SPI_tuptable->tupdesc != NULL)
		feat_type_oid = SPI_gettypeid(SPI_tuptable->tupdesc, 1);
	if (feat_type_oid == FLOAT8ARRAYOID || feat_type_oid == FLOAT4ARRAYOID)
		feat_is_array = true;

	if (is_gpu_model && neurondb_gpu_is_available())
	{
#ifdef NDB_GPU_CUDA
		const NdbCudaDtModelHeader *gpu_hdr;

		NDB_DECLARE(int *, h_labels);
		NDB_DECLARE(float *, h_features);
		int			feat_dim = 0;
		int			valid_rows = 0;
		size_t		payload_size;

		payload_size = VARSIZE(gpu_payload) - VARHDRSZ;
		if (payload_size < sizeof(NdbCudaDtModelHeader))
		{
			elog(DEBUG1,
				 "neurondb: evaluate_decision_tree_by_model_id: GPU payload too small (%zu bytes), falling back to CPU",
				 payload_size);
			goto cpu_evaluation_path;
		}

		gpu_hdr = (const NdbCudaDtModelHeader *) VARDATA(gpu_payload);
		if (gpu_hdr == NULL)
		{
			elog(DEBUG1,
				 "neurondb: evaluate_decision_tree_by_model_id: NULL GPU header, falling back to CPU");
			goto cpu_evaluation_path;
		}

		feat_dim = gpu_hdr->feature_dim;
		if (feat_dim <= 0 || feat_dim > 100000)
		{
			elog(DEBUG1,
				 "neurondb: evaluate_decision_tree_by_model_id: invalid feature_dim (%d), falling back to CPU",
				 feat_dim);
			goto cpu_evaluation_path;
		}

		{
			size_t		features_size = sizeof(float) * (size_t) nvec * (size_t) feat_dim;
			size_t		labels_size = sizeof(int) * (size_t) nvec;

			if (features_size > MaxAllocSize || labels_size > MaxAllocSize)
			{
				elog(DEBUG1,
					 "neurondb: evaluate_decision_tree_by_model_id: allocation size too large (features=%zu, labels=%zu), falling back to CPU",
					 features_size, labels_size);
				goto cpu_evaluation_path;
			}

			NDB_ALLOC(h_features, float, nvec * feat_dim);
			NDB_ALLOC(h_labels, int, nvec);

			if (h_features == NULL || h_labels == NULL)
			{
				elog(DEBUG1,
					 "neurondb: evaluate_decision_tree_by_model_id: memory allocation failed, falling back to CPU");
				NDB_FREE(h_features);
				NDB_FREE(h_labels);
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
				elog(DEBUG1,
					 "neurondb: evaluate_decision_tree_by_model_id: NULL TupleDesc, falling back to CPU");
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

				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || i >= SPI_processed)
					break;

				tuple = SPI_tuptable->vals[i];
				if (tuple == NULL)
					continue;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
				targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

				if (feat_null || targ_null)
					continue;

				if (valid_rows >= nvec)
				{
					elog(DEBUG1,
						 "neurondb: evaluate_decision_tree_by_model_id: valid_rows overflow, breaking");
					break;
				}

				feat_row = h_features + (valid_rows * feat_dim);
				if (feat_row == NULL || feat_row < h_features || feat_row >= h_features + (nvec * feat_dim))
				{
					elog(DEBUG1,
						 "neurondb: evaluate_decision_tree_by_model_id: feat_row out of bounds, skipping row");
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
			NDB_FREE(h_features);
			NDB_FREE(h_labels);
			NDB_FREE(gpu_payload);
			NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			NDB_SPI_SESSION_END(spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: no valid rows found")));
		}

		/* Use optimized GPU batch evaluation */
		{
			int			rc;

			NDB_DECLARE(char *, gpu_errstr);

			/* Defensive checks before GPU call */
			if (h_features == NULL || h_labels == NULL || valid_rows <= 0 || feat_dim <= 0)
			{
				elog(DEBUG1,
					 "neurondb: evaluate_decision_tree_by_model_id: invalid inputs for GPU evaluation (features=%p, labels=%p, rows=%d, dim=%d), falling back to CPU",
					 (void *) h_features, (void *) h_labels, valid_rows, feat_dim);
				NDB_FREE(h_features);
				NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}

			PG_TRY();
			{
				rc = ndb_cuda_dt_evaluate_batch(gpu_payload,
												h_features,
												h_labels,
												valid_rows,
												feat_dim,
												&accuracy,
												&precision,
												&recall,
												&f1_score,
												&gpu_errstr);

				if (rc == 0)
				{
					/* Success - cleanup and build result */
					/* Free GPU-allocated memory before ending SPI */
					NDB_FREE(h_features);
					NDB_FREE(h_labels);
					if (gpu_payload)
						NDB_FREE(gpu_payload);
					if (gpu_metrics)
						NDB_FREE(gpu_metrics);
					if (gpu_errstr)
						NDB_FREE(gpu_errstr);
					NDB_FREE(tbl_str);
					NDB_FREE(feat_str);
					NDB_FREE(targ_str);
					ndb_spi_stringinfo_free(spi_session, &query);

					/* End SPI session BEFORE creating JSONB */
					NDB_SPI_SESSION_END(spi_session);

					/* Switch to oldcontext to create JSONB */
					MemoryContextSwitchTo(oldcontext);

					/* Build result JSON string */
					initStringInfo(&jsonbuf);
					appendStringInfo(&jsonbuf,
									 "{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"n_samples\":%d}",
									 accuracy,
									 precision,
									 recall,
									 f1_score,
									 valid_rows);

					/* Create JSONB in oldcontext using ndb_jsonb_in_cstring */
					result_jsonb = ndb_jsonb_in_cstring(jsonbuf.data);
					NDB_FREE(jsonbuf.data);

					if (result_jsonb == NULL)
					{
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
								 errmsg("neurondb: evaluate_decision_tree_by_model_id: failed to parse metrics JSON")));
					}

					PG_RETURN_JSONB_P(result_jsonb);
				}
				else
				{
					/* GPU evaluation failed - fall back to CPU */
					elog(DEBUG1,
						 "neurondb: evaluate_decision_tree_by_model_id: GPU batch evaluation failed: %s, falling back to CPU",
						 gpu_errstr ? gpu_errstr : "unknown error");
					if (gpu_errstr)
						NDB_FREE(gpu_errstr);
					NDB_FREE(h_features);
					NDB_FREE(h_labels);
					goto cpu_evaluation_path;
				}
			}
			PG_CATCH();
			{
				elog(DEBUG1,
					 "neurondb: evaluate_decision_tree_by_model_id: exception during GPU evaluation, falling back to CPU");
				NDB_FREE(h_features);
				NDB_FREE(h_labels);
				goto cpu_evaluation_path;
			}
			PG_END_TRY();
		}
#endif							/* NDB_GPU_CUDA */
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

	/* CPU evaluation path */
	/* Use optimized batch prediction */
	{
		NDB_DECLARE(float *, h_features);
		NDB_DECLARE(double *, h_labels);
		int			feat_dim = 0;
		int			valid_rows = 0;

		/* Determine feature dimension from model */
		if (model != NULL)
			feat_dim = model->n_features;
		else if (is_gpu_model && gpu_payload != NULL)
		{
			const NdbCudaDtModelHeader *gpu_hdr;

			gpu_hdr = (const NdbCudaDtModelHeader *) VARDATA(gpu_payload);
			feat_dim = gpu_hdr->feature_dim;
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: could not determine feature dimension")));
		}

		if (feat_dim <= 0)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: invalid feature dimension %d",
							feat_dim)));
		}

		/* Allocate host buffers for features and labels */
		NDB_ALLOC(h_features, float, nvec * feat_dim);
		NDB_ALLOC(h_labels, double, nvec);

		/*
		 * Extract features and labels from SPI results - optimized batch
		 * extraction
		 */
		/* Cache TupleDesc to avoid repeated lookups */
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;

			for (i = 0; i < nvec; i++)
			{
				HeapTuple	tuple = SPI_tuptable->vals[i];
				Datum		feat_datum;
				Datum		targ_datum;
				bool		feat_null;
				bool		targ_null;
				Vector	   *vec;
				ArrayType  *arr;
				float	   *feat_row;

				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);
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
			NDB_FREE(h_features);
			NDB_FREE(h_labels);
			if (model != NULL)
			{
				if (model->root != NULL)
					dt_free_tree(model->root);
				NDB_FREE(model);
			}
			NDB_FREE(gpu_payload);
			NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			NDB_SPI_SESSION_END(spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: no valid rows found")));
		}

		/* For GPU models, we cannot evaluate on CPU without model conversion */
		if (is_gpu_model && model == NULL)
		{
			NDB_FREE(h_features);
			NDB_FREE(h_labels);
			NDB_FREE(gpu_payload);
			NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			NDB_SPI_SESSION_END(spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: GPU model evaluation requires GPU evaluation kernel")));
		}

		/* Ensure model is not NULL before prediction */
		if (model == NULL)
		{
			NDB_FREE(h_features);
			NDB_FREE(h_labels);
			NDB_FREE(gpu_payload);
			NDB_FREE(gpu_metrics);
			NDB_FREE(tbl_str);
			NDB_FREE(feat_str);
			NDB_FREE(targ_str);
			ndb_spi_stringinfo_free(spi_session, &query);
			NDB_SPI_SESSION_END(spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: evaluate_decision_tree_by_model_id: model is NULL")));
		}

		/* Use batch prediction helper */
		/* Add debug logging before prediction */
		if (model != NULL)
		{
			elog(DEBUG1,
				 "neurondb: evaluate_decision_tree_by_model_id: CPU batch prediction: model->n_features=%d, model->root=%p, valid_rows=%d, feat_dim=%d",
				 model->n_features, (void *) model->root, valid_rows, feat_dim);
		}
		else
		{
			elog(DEBUG1,
				 "neurondb: evaluate_decision_tree_by_model_id: model is NULL before CPU batch prediction");
		}

		dt_predict_batch(model,
						 h_features,
						 h_labels,
						 valid_rows,
						 feat_dim,
						 &tp,
						 &tn,
						 &fp,
						 &fn);

		elog(DEBUG1,
			 "neurondb: evaluate_decision_tree_by_model_id: after batch prediction: tp=%d, tn=%d, fp=%d, fn=%d, valid_rows=%d",
			 tp, tn, fp, fn, valid_rows);

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

		NDB_FREE(h_features);
		NDB_FREE(h_labels);
		if (model != NULL)
		{
			if (model->root != NULL)
				dt_free_tree(model->root);
			NDB_FREE(model);
		}
		NDB_FREE(gpu_payload);
		NDB_FREE(gpu_metrics);
	}

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);
	NDB_FREE(tbl_str);
	NDB_FREE(feat_str);
	NDB_FREE(targ_str);

	/* Build jsonb result */
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"accuracy\":%.6f,\"precision\":%.6f,\"recall\":%.6f,\"f1_score\":%.6f,\"n_samples\":%d}",
					 accuracy,
					 precision,
					 recall,
					 f1_score,
					 nvec);

	/* Use ndb_jsonb_in_cstring like other ML algorithms fix */
	result_jsonb = ndb_jsonb_in_cstring(jsonbuf.data);
	if (result_jsonb == NULL)
	{
		NDB_FREE(jsonbuf.data);

		/*
		 * Note: model, h_features, h_labels, gpu_payload, gpu_metrics,
		 * tbl_str, feat_str, targ_str, and query have already been freed
		 * above
		 */
		NDB_SPI_SESSION_END(spi_session);
		MemoryContextSwitchTo(oldcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("neurondb: evaluate_decision_tree_by_model_id: failed to parse metrics JSON")));
	}
	NDB_FREE(jsonbuf.data);
	MemoryContextSwitchTo(oldcontext);
	PG_RETURN_JSONB_P(result_jsonb);
}

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
		NDB_FREE(state->model_blob);
	if (state->metrics != NULL)
		NDB_FREE(state->metrics);
	NDB_FREE(state);
}

static bool
dt_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	DTGpuModelState *state;
	bytea	   *payload;
	Jsonb	   *metrics;
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
		if (payload != NULL)
			NDB_FREE(payload);
		if (metrics != NULL)
			NDB_FREE(metrics);
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
		state->metrics = (Jsonb *) PG_DETOAST_DATUM_COPY(PointerGetDatum(metrics));
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
	Jsonb	   *metrics_json;

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
		/* Use ndb_jsonb_in_cstring like other ML algorithms fix */
		metrics_json = ndb_jsonb_in_cstring(buf.data);
		if (metrics_json == NULL)
		{
			NDB_FREE(buf.data);
			if (errstr != NULL)
				*errstr = pstrdup("failed to parse metrics JSON");
			return false;
		}
		NDB_FREE(buf.data);
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
	NDB_DECLARE(bytea *, payload_copy);
	NDB_DECLARE(char *, payload_copy_raw);
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
	NDB_ALLOC(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_FREE(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
	{
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(PointerGetDatum(state->metrics));
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
	DTGpuModelState *state;
	NDB_DECLARE(bytea *, payload_copy);
	NDB_DECLARE(char *, payload_copy_raw);
	int			payload_size;
	int			feature_dim = -1;
	int			n_samples = -1;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
		return false;

	payload_size = VARSIZE(payload);
	NDB_ALLOC(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
	memcpy(payload_copy, payload, payload_size);

	/* Extract feature_dim and n_samples from metadata if available */
	if (metadata != NULL)
	{
		NDB_DECLARE(JsonbIterator *, it);
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
					NDB_FREE(key);
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
		state->metrics = (Jsonb *) PG_DETOAST_DATUM_COPY(PointerGetDatum(metadata));
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
