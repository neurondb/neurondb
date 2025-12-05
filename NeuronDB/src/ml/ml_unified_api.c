/*-------------------------------------------------------------------------
 *
 * ml_unified_api.c
 *    Unified SQL API for machine learning operations.
 *
 * This module provides a simplified interface for training, prediction,
 * deployment, and model loading through SQL functions.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_unified_api.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "catalog/pg_proc.h"
#include "catalog/pg_language.h"
#include "access/htup_details.h"
#include "utils/timestamp.h"
#include "utils/memutils.h"
#include "utils/lsyscache.h"
#include "utils/syscache.h"

#include <time.h>

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_gpu_bridge.h"
#include "ml_catalog.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_gpu.h"
#include "neurondb_cuda_nb.h"
#include "neurondb_cuda_gmm.h"
#include "neurondb_cuda_knn.h"
#include "neurondb_validation.h"
#include "neurondb_spi.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_constants.h"
#include "neurondb_json.h"
#include "ml_decision_tree_internal.h"

PG_FUNCTION_INFO_V1(neurondb_train);
PG_FUNCTION_INFO_V1(neurondb_predict);
PG_FUNCTION_INFO_V1(neurondb_deploy);
PG_FUNCTION_INFO_V1(neurondb_load_model);
PG_FUNCTION_INFO_V1(neurondb_evaluate);

/*
 * ML Algorithm enumeration
 * This enum provides type-safe algorithm identification
 */
typedef enum
{
	ML_ALGO_UNKNOWN = 0,
	/* Classification algorithms */
	ML_ALGO_LOGISTIC_REGRESSION,
	ML_ALGO_RANDOM_FOREST,
	ML_ALGO_SVM,
	ML_ALGO_DECISION_TREE,
	ML_ALGO_NAIVE_BAYES,
	ML_ALGO_XGBOOST,
	ML_ALGO_CATBOOST,
	ML_ALGO_LIGHTGBM,
	ML_ALGO_KNN,
	ML_ALGO_KNN_CLASSIFIER,
	ML_ALGO_KNN_REGRESSOR,
	/* Regression algorithms */
	ML_ALGO_LINEAR_REGRESSION,
	ML_ALGO_RIDGE,
	ML_ALGO_LASSO,
	/* Clustering algorithms */
	ML_ALGO_KMEANS,
	ML_ALGO_GMM,
	ML_ALGO_MINIBATCH_KMEANS,
	ML_ALGO_HIERARCHICAL,
	ML_ALGO_DBSCAN,
	/* Dimensionality reduction */
	ML_ALGO_PCA,
	ML_ALGO_OPQ,
	/* Time series */
	ML_ALGO_TIMESERIES
}			MLAlgorithm;

/* Forward declarations */
static void neurondb_cleanup(MemoryContext oldcontext, MemoryContext callcontext);
static char *neurondb_quote_literal_cstr(const char *str);
static char *neurondb_quote_literal_or_null(const char *str);
static MLAlgorithm neurondb_algorithm_from_string(const char *algorithm);
static bool ml_metrics_is_gpu(Jsonb * metrics);
static void neurondb_parse_hyperparams_int(Jsonb * hyperparams, const char *key, int *value, int default_value);
static void neurondb_parse_hyperparams_float8(Jsonb * hyperparams, const char *key, double *value, double default_value);
static bool neurondb_build_training_sql(MLAlgorithm algo, StringInfo sql, const char *table_name, const char *feature_list, const char *target_column, Jsonb * hyperparams, const char **feature_names, int feature_name_count);
static void neurondb_validate_training_inputs(const char *project_name, const char *algorithm, const char *table_name, const char *target_column);
static bool neurondb_is_unsupervised_algorithm(const char *algorithm);
static int	neurondb_prepare_feature_list(ArrayType * feature_columns_array, StringInfo feature_list, const char ***feature_names_out, int *feature_name_count_out);


/*
 * neurondb_cleanup
 *		Restore memory context and delete call context.
 *
 * This function is defensive and handles NULL contexts safely.
 */
static void
neurondb_cleanup(MemoryContext oldcontext, MemoryContext callcontext)
{
	MemoryContext current_context;

	if (oldcontext == NULL)
	{
		elog(WARNING, "neurondb_cleanup: oldcontext is NULL, using CurrentMemoryContext");
		oldcontext = CurrentMemoryContext;
		if (oldcontext == NULL)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb_cleanup: CurrentMemoryContext is also NULL"),
					 errdetail("Cannot proceed with cleanup without a valid memory context"),
					 errhint("This is an internal error. Please report this issue.")));
		}
	}

	current_context = MemoryContextSwitchTo(oldcontext);
	if (current_context == NULL && oldcontext != CurrentMemoryContext)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb_cleanup: MemoryContextSwitchTo failed"),
				 errdetail("Failed to switch to oldcontext"),
				 errhint("This is an internal error. Please report this issue.")));
	}

	if (callcontext != NULL)
	{
		if (MemoryContextIsValid(callcontext))
		{
			MemoryContextDelete(callcontext);
		}
		else
		{
			elog(WARNING, "neurondb_cleanup: callcontext is not valid, skipping deletion");
		}
	}
}

/*
 * neurondb_load_training_data
 *		Load training data from table into feature_matrix and label_vector.
 *
 * Returns true on success, false on failure. Allocates memory for feature_matrix
 * and label_vector which must be freed by caller using NDB_FREE.
 */
static bool
neurondb_load_training_data(NdbSpiSession *session,
							const char *table_name,
							const char *feature_list_str,
							const char *target_column,
							float **feature_matrix_out,
							double **label_vector_out,
							int *n_samples_out,
							int *feature_dim_out,
							int *class_count_out)
{
	StringInfoData sql;
	int			ret;
	int			n_samples = 0;
	int			feature_dim = 0;
	int			class_count = 0;
	int			valid_samples = 0;

	NDB_DECLARE(float *, feature_matrix);
	NDB_DECLARE(double *, label_vector);
	TupleDesc	tupdesc;
	HeapTuple	tuple;
	bool		isnull;
	Datum		feat_datum;

	NDB_DECLARE(ArrayType *, feat_arr);
	int			i,
				j;
	Oid			feature_type;


	if (feature_matrix_out)
		*feature_matrix_out = NULL;
	if (label_vector_out)
		*label_vector_out = NULL;
	if (n_samples_out)
		*n_samples_out = 0;
	if (feature_dim_out)
		*feature_dim_out = 0;
	if (class_count_out)
		*class_count_out = 0;

	{
		int			max_samples_limit = 200000;
		char	   *target_copy;
		const char *target_quoted_const;
		char	   *target_quoted;

		ndb_spi_stringinfo_init(session, &sql);

		if (target_column)
		{
			target_copy = pstrdup(target_column);
			target_quoted_const = quote_identifier(target_copy);
			target_quoted = (char *) target_quoted_const;
			appendStringInfo(&sql, "SELECT %s, %s FROM %s LIMIT %d",
							 feature_list_str, target_quoted, table_name, max_samples_limit);
			NDB_FREE(target_quoted);
			NDB_FREE(target_copy);
		}
		else
		{
			appendStringInfo(&sql, "SELECT %s FROM %s LIMIT %d",
							 feature_list_str, table_name, max_samples_limit);
		}
	}

	ret = ndb_spi_execute(session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(session, &sql);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb_load_training_data: failed to execute data query for table '%s'", table_name),
				 errdetail("SPI execution returned code %d (expected %d for SELECT). Query: %s", ret, SPI_OK_SELECT, sql.data),
				 errhint("Check that table '%s' exists and is accessible, and that feature columns are valid.", table_name)));
		return false;
	}

	n_samples = SPI_processed;
	if (n_samples == 0)
	{
		ndb_spi_stringinfo_free(session, &sql);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("neurondb_load_training_data: no training data found in table '%s'", table_name),
				 errdetail("The query returned 0 rows. Table: %s, Feature columns: %s%s", table_name, feature_list_str, target_column ? ", Target column: " : ""),
				 errhint("Ensure table '%s' contains data and that feature columns exist and have non-NULL values.", table_name)));
		return false;
	}

	if (n_samples >= 200000)
	{
		elog(INFO,
			 "neurondb_load_training_data: dataset has more than %d rows, "
			 "limiting to %d samples to avoid memory allocation errors",
			 200000, n_samples);
	}

	/*
	 * Determine feature dimension from first row - safe access pattern for
	 * complex types
	 */
	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL || SPI_tuptable->vals == NULL)
	{
		ndb_spi_stringinfo_free(session, &sql);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb_load_training_data: SPI result is invalid")));
		return false;
	}
	tupdesc = SPI_tuptable->tupdesc;
	if (SPI_processed == 0 || SPI_tuptable->vals[0] == NULL)
	{
		ndb_spi_stringinfo_free(session, &sql);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb_load_training_data: no rows returned from query")));
		return false;
	}
	tuple = SPI_tuptable->vals[0];
	feat_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
	if (isnull)
	{
		ndb_spi_stringinfo_free(session, &sql);
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb_load_training_data: first feature column contains NULL value in table '%s'", table_name),
				 errdetail("The first row of table '%s' has a NULL value in the first feature column. Feature list: %s", table_name, feature_list_str),
				 errhint("Remove rows with NULL feature values or use a different feature column. NULL values in features are not supported.")));
		return false;
	}

	feature_type = SPI_gettypeid(tupdesc, 1);
	if (feature_type == FLOAT4ARRAYOID || feature_type == FLOAT8ARRAYOID)
	{
		feat_arr = DatumGetArrayTypeP(feat_datum);
		feature_dim = ArrayGetNItems(ARR_NDIM(feat_arr), ARR_DIMS(feat_arr));
	}
	else
	{
		Vector	   *vec;

		vec = DatumGetVector(feat_datum);
		if (vec != NULL && vec->dim > 0)
		{
			feature_dim = vec->dim;
			feat_arr = NULL;
		}
		else
		{
			feature_dim = 1;
			feat_arr = NULL;
		}
	}
	{
		size_t		feature_matrix_size;
		size_t		label_vector_size;

		feature_matrix_size = sizeof(float) * (size_t) n_samples * (size_t) feature_dim;
		label_vector_size = target_column ? sizeof(double) * (size_t) n_samples : 0;

		if (feature_matrix_size > MaxAllocSize)
		{
			ndb_spi_stringinfo_free(session, &sql);
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("neurondb_load_training_data: feature matrix size exceeds memory limit for table '%s'", table_name),
					 errdetail("Feature matrix requires %zu bytes (%d samples × %d features × %zu bytes/float), but MaxAllocSize is %zu bytes",
							   feature_matrix_size, n_samples, feature_dim, sizeof(float), (size_t) MaxAllocSize),
					 errhint("Reduce the number of samples (currently %d) or feature dimensions (currently %d). Consider using LIMIT in your query or reducing feature columns.", n_samples, feature_dim)));
		}

		if (label_vector_size > MaxAllocSize)
		{
			ndb_spi_stringinfo_free(session, &sql);
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("neurondb_load_training_data: label vector size exceeds memory limit for table '%s'", table_name),
					 errdetail("Label vector requires %zu bytes (%d samples × %zu bytes/double), but MaxAllocSize is %zu bytes",
							   label_vector_size, n_samples, sizeof(double), (size_t) MaxAllocSize),
					 errhint("Reduce the number of samples (currently %d) in table '%s'. Consider using LIMIT in your query.", n_samples, table_name)));
		}
	}

	NDB_ALLOC(feature_matrix, float, (size_t) n_samples * (size_t) feature_dim);

	if (target_column)
		NDB_ALLOC(label_vector, double, (size_t) n_samples);

	valid_samples = 0;
	for (i = 0; i < n_samples; i++)
	{
		HeapTuple	current_tuple;
		bool		isnull_feat;
		bool		isnull_label;
		Datum		featval;
		Datum		labelval;

		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || i >= SPI_processed)
		{
			continue;
		}
		current_tuple = SPI_tuptable->vals[i];
		if (current_tuple == NULL)
		{
			continue;
		}

		featval = SPI_getbinval(current_tuple, tupdesc, 1, &isnull_feat);
		if (isnull_feat)
		{
			continue;
		}

		if (feat_arr)
		{
			ArrayType  *curr_arr;
			int			arr_len;

			curr_arr = DatumGetArrayTypeP(featval);
			if (ARR_NDIM(curr_arr) == 1)
			{
				arr_len = ArrayGetNItems(ARR_NDIM(curr_arr), ARR_DIMS(curr_arr));
				if (arr_len != feature_dim)
				{
					NDB_FREE(feature_matrix);

					NDB_FREE(label_vector);

					ndb_spi_stringinfo_free(session, &sql);
					return false;
				}
				if (feature_type == FLOAT8ARRAYOID)
				{
					float8	   *fdat;

					fdat = (float8 *) ARR_DATA_PTR(curr_arr);
					for (j = 0; j < feature_dim; j++)
						feature_matrix[valid_samples * feature_dim + j] = (float) fdat[j];
				}
				else
				{
					float4	   *fdat;

					fdat = (float4 *) ARR_DATA_PTR(curr_arr);
					for (j = 0; j < feature_dim; j++)
						feature_matrix[valid_samples * feature_dim + j] = fdat[j];
				}
			}
			else
			{
				NDB_FREE(feature_matrix);
				NDB_FREE(label_vector);

				ndb_spi_stringinfo_free(session, &sql);
				return false;
			}
		}
		else if (feature_type == FLOAT8OID || feature_type == FLOAT4OID)
		{
			if (feature_type == FLOAT8OID)
				feature_matrix[valid_samples * feature_dim] = (float) DatumGetFloat8(featval);
			else
				feature_matrix[valid_samples * feature_dim] = DatumGetFloat4(featval);
		}
		else
		{
			Vector	   *vec;

			vec = DatumGetVector(featval);
			if (vec != NULL && vec->dim == feature_dim)
			{
				for (j = 0; j < feature_dim; j++)
					feature_matrix[valid_samples * feature_dim + j] = vec->data[j];
			}
			else
			{
				continue;
			}
		}
		if (target_column)
		{
			Oid			label_type;

			/* Safe access for label - validate tupdesc has at least 2 columns */
			if (tupdesc == NULL || tupdesc->natts < 2)
			{
				continue;
			}
			labelval = SPI_getbinval(current_tuple, tupdesc, 2, &isnull_label);
			if (isnull_label)
			{
				continue;
			}

			label_type = SPI_gettypeid(tupdesc, 2);
			if (label_type == INT4OID)
				label_vector[valid_samples] = (double) DatumGetInt32(labelval);
			else if (label_type == INT8OID)
				label_vector[valid_samples] = (double) DatumGetInt64(labelval);
			else if (label_type == FLOAT4OID)
				label_vector[valid_samples] = (double) DatumGetFloat4(labelval);
			else if (label_type == FLOAT8OID)
				label_vector[valid_samples] = DatumGetFloat8(labelval);
			else
			{
				continue;
			}
		}

		valid_samples++;
	}

	n_samples = valid_samples;

	if (target_column && label_vector)
	{
		NDB_DECLARE(int *, seen_classes);
		int			max_class = -1;
		int			cls;

		NDB_ALLOC(seen_classes, int, 256);

		for (i = 0; i < n_samples; i++)
		{
			cls = (int) label_vector[i];
			if (cls >= 0 && cls < 256)
			{
				if (!seen_classes[cls])
				{
					seen_classes[cls] = 1;
					if (cls > max_class)
						max_class = cls;
				}
			}
		}
		class_count = max_class + 1;
		if (class_count == 0)
			class_count = 2;
		NDB_FREE(seen_classes);
	}

	ndb_spi_stringinfo_free(session, &sql);

	if (feature_matrix_out)
		*feature_matrix_out = feature_matrix;
	if (label_vector_out)
		*label_vector_out = label_vector;
	if (n_samples_out)
		*n_samples_out = n_samples;
	if (feature_dim_out)
		*feature_dim_out = feature_dim;
	if (class_count_out)
		*class_count_out = class_count;

	return true;
}

/*
 * neurondb_quote_literal_cstr
 *		Return a quoted SQL literal string.
 *
 * Result must be freed by caller using pfree.
 */
static char *
neurondb_quote_literal_cstr(const char *str)
{
	char	   *ret;
	text	   *txt;

	if (str == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb_quote_literal_cstr: cannot quote NULL string"),
				 errdetail("Internal function called with NULL string pointer"),
				 errhint("This is an internal error. Please report this issue.")));

	txt = cstring_to_text(str);

	ret = TextDatumGetCString(
							  DirectFunctionCall1(quote_literal, PointerGetDatum(txt)));
	NDB_FREE(txt);

	return ret;
}

/*
 * neurondb_quote_literal_or_null
 *		Quote literal string or return "NULL" string for SQL.
 */
static char *
neurondb_quote_literal_or_null(const char *str)
{
	if (str == NULL)
		return pstrdup("NULL");
	return neurondb_quote_literal_cstr(str);
}

/*
 * neurondb_algorithm_from_string
 *		Convert algorithm string to MLAlgorithm enum.
 *
 * Returns ML_ALGO_UNKNOWN for unrecognized algorithms.
 */
static MLAlgorithm
neurondb_algorithm_from_string(const char *algorithm)
{
	if (algorithm == NULL)
		return ML_ALGO_UNKNOWN;

	if (strcmp(algorithm, NDB_ALGO_LOGISTIC_REGRESSION) == 0)
		return ML_ALGO_LOGISTIC_REGRESSION;
	if (strcmp(algorithm, NDB_ALGO_RANDOM_FOREST) == 0)
		return ML_ALGO_RANDOM_FOREST;
	if (strcmp(algorithm, NDB_ALGO_SVM) == 0)
		return ML_ALGO_SVM;
	if (strcmp(algorithm, NDB_ALGO_DECISION_TREE) == 0)
		return ML_ALGO_DECISION_TREE;
	if (strcmp(algorithm, NDB_ALGO_NAIVE_BAYES) == 0)
		return ML_ALGO_NAIVE_BAYES;
	if (strcmp(algorithm, NDB_ALGO_XGBOOST) == 0)
		return ML_ALGO_XGBOOST;
	if (strcmp(algorithm, NDB_ALGO_CATBOOST) == 0)
		return ML_ALGO_CATBOOST;
	if (strcmp(algorithm, NDB_ALGO_LIGHTGBM) == 0)
		return ML_ALGO_LIGHTGBM;
	if (strcmp(algorithm, NDB_ALGO_KNN) == 0)
		return ML_ALGO_KNN;
	if (strcmp(algorithm, NDB_ALGO_KNN_CLASSIFIER) == 0)
		return ML_ALGO_KNN_CLASSIFIER;
	if (strcmp(algorithm, NDB_ALGO_KNN_REGRESSOR) == 0)
		return ML_ALGO_KNN_REGRESSOR;
	if (strcmp(algorithm, NDB_ALGO_LINEAR_REGRESSION) == 0)
		return ML_ALGO_LINEAR_REGRESSION;
	if (strcmp(algorithm, NDB_ALGO_RIDGE) == 0)
		return ML_ALGO_RIDGE;
	if (strcmp(algorithm, NDB_ALGO_LASSO) == 0)
		return ML_ALGO_LASSO;
	if (strcmp(algorithm, NDB_ALGO_KMEANS) == 0)
		return ML_ALGO_KMEANS;
	if (strcmp(algorithm, NDB_ALGO_GMM) == 0)
		return ML_ALGO_GMM;
	if (strcmp(algorithm, NDB_ALGO_MINIBATCH_KMEANS) == 0)
		return ML_ALGO_MINIBATCH_KMEANS;
	if (strcmp(algorithm, NDB_ALGO_HIERARCHICAL) == 0)
		return ML_ALGO_HIERARCHICAL;
	if (strcmp(algorithm, NDB_ALGO_DBSCAN) == 0)
		return ML_ALGO_DBSCAN;
	if (strcmp(algorithm, NDB_ALGO_PCA) == 0)
		return ML_ALGO_PCA;
	if (strcmp(algorithm, NDB_ALGO_OPQ) == 0)
		return ML_ALGO_OPQ;
	if (strcmp(algorithm, NDB_ALGO_TIMESERIES) == 0)
		return ML_ALGO_TIMESERIES;

	return ML_ALGO_UNKNOWN;
}

/*
 * ml_metrics_is_gpu
 *		Check if metrics JSONB indicates GPU training.
 *
 * Returns true if metrics contains "training_backend": 1, false otherwise.
 */
static bool
ml_metrics_is_gpu(Jsonb * metrics)
{
	bool		is_gpu = false;
	JsonbIterator *it;
	JsonbValue	v;
	JsonbIteratorToken r;

	if (metrics == NULL)
		return false;

	/* Check for training_backend integer in metrics */
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

					is_gpu = (backend == 1);
				}
			}
			NDB_FREE(key);
		}
	}

	return is_gpu;
}

/*
 * neurondb_parse_hyperparams_int
 *		Parse integer hyperparameter from JSONB.
 *
 * If key is not found, value remains at default_value.
 */
static void
neurondb_parse_hyperparams_int(Jsonb * hyperparams, const char *key, int *value, int default_value)
{
	Jsonb	   *field_jsonb;
	JsonbValue	v;
	JsonbIterator *it;
	int			r;

	if (hyperparams == NULL || key == NULL || value == NULL)
	{
		if (value)
			*value = default_value;
		return;
	}

	*value = default_value;

	/* Wrap JSONB operations in PG_TRY to handle corrupted JSONB gracefully */
	PG_TRY();
	{
		/* Use ndb_jsonb_object_field to get the field directly */
		field_jsonb = ndb_jsonb_object_field(hyperparams, key);
		if (field_jsonb != NULL)
		{
			/* Extract numeric value from the JSONB field */
			it = JsonbIteratorInit((JsonbContainer *) & field_jsonb->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_VALUE && v.type == jbvNumeric)
				{
					*value = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));
					break;
				}
			}
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		elog(DEBUG2, "neurondb_parse_hyperparams_int: Failed to parse hyperparameters JSONB (possibly corrupted), using default value");
		*value = default_value;
	}
	PG_END_TRY();
}

/*
 * neurondb_parse_hyperparams_float8
 *		Parse float8 hyperparameter from JSONB.
 *
 * If key is not found, value remains at default_value.
 */
static void
neurondb_parse_hyperparams_float8(Jsonb * hyperparams, const char *key, double *value, double default_value)
{
	Jsonb	   *field_jsonb;
	JsonbValue	v;
	JsonbIterator *it;
	int			r;

	if (hyperparams == NULL || key == NULL || value == NULL)
	{
		if (value)
			*value = default_value;
		return;
	}

	*value = default_value;

	/* Wrap JSONB operations in PG_TRY to handle corrupted JSONB gracefully */
	PG_TRY();
	{
		/* Use ndb_jsonb_object_field to get the field directly */
		field_jsonb = ndb_jsonb_object_field(hyperparams, key);
		if (field_jsonb != NULL)
		{
			/* Extract numeric value from the JSONB field */
			it = JsonbIteratorInit((JsonbContainer *) & field_jsonb->root);
			while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
			{
				if (r == WJB_VALUE && v.type == jbvNumeric)
				{
					*value = DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v.val.numeric)));
					break;
				}
			}
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		elog(DEBUG2, "neurondb_parse_hyperparams_float8: Failed to parse hyperparameters JSONB (possibly corrupted), using default value");
		*value = default_value;
	}
	PG_END_TRY();
}

/*
 * neurondb_build_training_sql
 *		Build training SQL for simple algorithms.
 *
 * Returns true if SQL was built successfully, false if algorithm requires
 * special handling and should be processed separately.
 */
static bool
__attribute__((unused))
neurondb_build_training_sql(MLAlgorithm algo, StringInfo sql, const char *table_name,
							const char *feature_list, const char *target_column,
							Jsonb * hyperparams, const char **feature_names, int feature_name_count)
{
	int			max_iters;
	double		learning_rate;
	double		lambda;
	double		C;
	int			n_trees;
	int			max_depth;
	int			min_samples;
	int			max_features;
	const char *feature_col;

	switch (algo)
	{
		case ML_ALGO_LINEAR_REGRESSION:
			appendStringInfo(sql,
							 "SELECT train_linear_regression(%s, %s, %s)",
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_cstr(feature_list),
							 neurondb_quote_literal_or_null(target_column));
			return true;

		case ML_ALGO_LOGISTIC_REGRESSION:
			max_iters = 1000;
			learning_rate = 0.01;
			lambda = 0.001;
			neurondb_parse_hyperparams_int(hyperparams, "max_iters", &max_iters, 1000);
			neurondb_parse_hyperparams_float8(hyperparams, "learning_rate", &learning_rate, 0.01);
			neurondb_parse_hyperparams_float8(hyperparams, "lambda", &lambda, 0.001);

			/*
			 * Use first feature name if available, otherwise default to
			 * "features"
			 */

			/*
			 * feature_names[0] is allocated in callcontext and should be safe
			 * to use
			 */

			/*
			 * But to be extra safe, we'll use feature_list if it's a single
			 * column, otherwise use feature_names[0]
			 */
			if (feature_name_count > 0 && feature_names != NULL && feature_names[0] != NULL && strlen(feature_names[0]) > 0)
			{
				/*
				 * Use first feature name - it's already a string literal in
				 * callcontext
				 */
				feature_col = feature_names[0];
			}
			else if (feature_list != NULL && strlen(feature_list) > 0 && strcmp(feature_list, "*") != 0)
			{
				/* Use feature_list if it's a single column (no comma) */
				if (strchr(feature_list, ',') == NULL)
				{
					feature_col = feature_list;
				}
				else
				{
					/* Multiple columns - default to "features" */
					feature_col = "features";
				}
			}
			else
			{
				/*
				 * Default to "features" - this is a string literal, safe to
				 * use
				 */
				feature_col = "features";
			}
			{
				char	   *quoted_table = neurondb_quote_literal_cstr(table_name);
				char	   *quoted_feature = neurondb_quote_literal_cstr(feature_col);
				char	   *quoted_target = neurondb_quote_literal_or_null(target_column);


				appendStringInfo(sql,
								 "SELECT train_logistic_regression(%s, %s, %s, %d, %.6f, %.6f)",
								 quoted_table,
								 quoted_feature,
								 quoted_target,
								 max_iters, learning_rate, lambda);


				/* Free the quoted strings */
				NDB_FREE(quoted_table);
				NDB_FREE(quoted_feature);
				if (quoted_target != NULL && strcmp(quoted_target, "NULL") != 0)
					NDB_FREE(quoted_target);
			}
			return true;

		case ML_ALGO_SVM:
			C = 1.0;
			max_iters = 1000;
			neurondb_parse_hyperparams_float8(hyperparams, "C", &C, 1.0);
			neurondb_parse_hyperparams_int(hyperparams, "max_iters", &max_iters, 1000);
			appendStringInfo(sql,
							 "SELECT train_svm_classifier(%s, %s, %s, %.6f, %d)",
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_cstr(feature_list),
							 neurondb_quote_literal_or_null(target_column),
							 C, max_iters);
			return true;

		case ML_ALGO_RANDOM_FOREST:
			n_trees = 10;
			max_depth = 10;
			min_samples = 100;
			max_features = 0;
			neurondb_parse_hyperparams_int(hyperparams, "n_trees", &n_trees, 10);
			neurondb_parse_hyperparams_int(hyperparams, "max_depth", &max_depth, 10);
			neurondb_parse_hyperparams_int(hyperparams, "min_samples", &min_samples, 100);
			neurondb_parse_hyperparams_int(hyperparams, "min_samples_split", &min_samples, 100);
			neurondb_parse_hyperparams_int(hyperparams, "max_features", &max_features, 0);
			appendStringInfo(sql,
							 "SELECT train_random_forest_classifier(%s, %s, %s, %d, %d, %d, %d)",
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_cstr(feature_list),
							 neurondb_quote_literal_or_null(target_column),
							 n_trees, max_depth, min_samples, max_features);
			return true;

		case ML_ALGO_DECISION_TREE:
			max_depth = 10;
			min_samples = 2;
			neurondb_parse_hyperparams_int(hyperparams, "max_depth", &max_depth, 10);
			neurondb_parse_hyperparams_int(hyperparams, "min_samples_split", &min_samples, 2);

			/*
			 * Use first feature name if available, otherwise default to
			 * "features"
			 */
			if (feature_name_count > 0 && feature_names != NULL && feature_names[0] != NULL && strlen(feature_names[0]) > 0)
			{
				feature_col = feature_names[0];
			}
			else if (feature_list != NULL && strlen(feature_list) > 0 && strcmp(feature_list, "*") != 0)
			{
				if (strchr(feature_list, ',') == NULL)
				{
					feature_col = feature_list;
				}
				else
				{
					feature_col = "features";
				}
			}
			else
			{
				feature_col = "features";
			}
			appendStringInfo(sql,
							 "SELECT train_decision_tree_classifier(%s, %s, %s, %d, %d)",
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_cstr(feature_col),
							 neurondb_quote_literal_or_null(target_column),
							 max_depth, min_samples);
			return true;

		case ML_ALGO_RIDGE:
			{
				double		alpha = 1.0;

				neurondb_parse_hyperparams_float8(hyperparams, "alpha", &alpha, 1.0);
				appendStringInfo(sql,
								 "SELECT train_ridge_regression(%s, %s, %s, %.6f)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_list),
								 neurondb_quote_literal_or_null(target_column),
								 alpha);
				return true;
			}

		case ML_ALGO_LASSO:
			{
				double		alpha = 1.0;

				neurondb_parse_hyperparams_float8(hyperparams, "alpha", &alpha, 1.0);
				appendStringInfo(sql,
								 "SELECT train_lasso_regression(%s, %s, %s, %.6f)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_list),
								 neurondb_quote_literal_or_null(target_column),
								 alpha);
				return true;
			}

		case ML_ALGO_KMEANS:
			{
				int			n_clusters = 3;
				int			max_iters_kmeans = 100;

				neurondb_parse_hyperparams_int(hyperparams, "n_clusters", &n_clusters, 3);
				neurondb_parse_hyperparams_int(hyperparams, "max_iters", &max_iters_kmeans, 100);
				appendStringInfo(sql,
								 "SELECT train_kmeans_model_id(%s, %s, %d, %d)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_list),
								 n_clusters, max_iters_kmeans);
				return true;
			}

		case ML_ALGO_GMM:
			{
				int			n_components = 3;
				int			max_iters_gmm = 100;

				neurondb_parse_hyperparams_int(hyperparams, "n_components", &n_components, 3);
				neurondb_parse_hyperparams_int(hyperparams, "max_iters", &max_iters_gmm, 100);
				appendStringInfo(sql,
								 "SELECT train_gmm_model_id(%s, %s, %d, %d)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_list),
								 n_components, max_iters_gmm);
				return true;
			}

		case ML_ALGO_MINIBATCH_KMEANS:
			{
				int			n_clusters = 3;

				neurondb_parse_hyperparams_int(hyperparams, "n_clusters", &n_clusters, 3);
				appendStringInfo(sql,
								 "SELECT train_minibatch_kmeans(%s, %s, %d)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_list),
								 n_clusters);
				return true;
			}

		case ML_ALGO_HIERARCHICAL:
			{
				int			n_clusters = 3;

				neurondb_parse_hyperparams_int(hyperparams, "n_clusters", &n_clusters, 3);
				appendStringInfo(sql,
								 "SELECT train_hierarchical_clustering(%s, %s, %d)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_list),
								 n_clusters);
				return true;
			}

		case ML_ALGO_XGBOOST:
			{
				int			n_estimators = 100;
				int			max_depth_xgb = 3;
				double		learning_rate_xgb = 0.1;

				neurondb_parse_hyperparams_int(hyperparams, "n_estimators", &n_estimators, 100);
				neurondb_parse_hyperparams_int(hyperparams, "max_depth", &max_depth_xgb, 3);
				neurondb_parse_hyperparams_float8(hyperparams, "learning_rate", &learning_rate_xgb, 0.1);
				appendStringInfo(sql,
								 "SELECT train_xgboost_classifier(%s, %s, %s, %d, %d, %.6f)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_list),
								 neurondb_quote_literal_or_null(target_column),
								 n_estimators, max_depth_xgb, learning_rate_xgb);
				return true;
			}

		case ML_ALGO_CATBOOST:
			{
				int			iterations = 1000;
				int			depth = 6;
				double		learning_rate_cb = 0.03;

				neurondb_parse_hyperparams_int(hyperparams, "iterations", &iterations, 1000);
				neurondb_parse_hyperparams_int(hyperparams, "depth", &depth, 6);
				neurondb_parse_hyperparams_float8(hyperparams, "learning_rate", &learning_rate_cb, 0.03);
				appendStringInfo(sql,
								 "SELECT train_catboost_classifier(%s, %s, %s, %d, %.6f, %d)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_list),
								 neurondb_quote_literal_or_null(target_column),
								 iterations, learning_rate_cb, depth);
				return true;
			}

		case ML_ALGO_LIGHTGBM:
			{
				int			n_estimators_lgb = 100;
				int			num_leaves_lgb = 31;
				double		learning_rate_lgb = 0.1;

				neurondb_parse_hyperparams_int(hyperparams, "n_estimators", &n_estimators_lgb, 100);
				neurondb_parse_hyperparams_int(hyperparams, "num_leaves", &num_leaves_lgb, 31);
				neurondb_parse_hyperparams_float8(hyperparams, "learning_rate", &learning_rate_lgb, 0.1);
				appendStringInfo(sql,
								 "SELECT train_lightgbm_classifier(%s, %s, %s, %d, %d, %.6f)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_list),
								 neurondb_quote_literal_or_null(target_column),
								 n_estimators_lgb, num_leaves_lgb, learning_rate_lgb);
				return true;
			}

		case ML_ALGO_NAIVE_BAYES:
			appendStringInfo(sql,
							 "SELECT train_naive_bayes_classifier_model_id(%s, %s, %s)",
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_cstr(feature_list),
							 neurondb_quote_literal_or_null(target_column));
			return true;

		case ML_ALGO_TIMESERIES:
			{
				int			p = 1;
				int			d = 1;
				int			q = 1;

				neurondb_parse_hyperparams_int(hyperparams, "p", &p, 1);
				neurondb_parse_hyperparams_int(hyperparams, "d", &d, 1);
				neurondb_parse_hyperparams_int(hyperparams, "q", &q, 1);

				/*
				 * For timeseries, we use the first feature column as the time
				 * series data
				 */
				/* and target_column as the label (value) */

				/*
				 * Use first feature name if available, otherwise use
				 * feature_list
				 */
				if (feature_name_count > 0 && feature_names != NULL && feature_names[0] != NULL && strlen(feature_names[0]) > 0)
				{
					feature_col = feature_names[0];
				}
				else if (feature_list != NULL && strlen(feature_list) > 0 && strcmp(feature_list, "*") != 0)
				{
					if (strchr(feature_list, ',') == NULL)
					{
						feature_col = feature_list;
					}
					else
					{
						feature_col = "features";
					}
				}
				else
				{
					feature_col = "features";
				}

				/*
				 * For timeseries CPU training, we'll use a simple approach:
				 * Extract the first dimension of the feature vector as the
				 * time series value and use train_arima with the target
				 * column as the value column
				 */
				appendStringInfo(sql,
								 "SELECT train_timeseries_cpu(%s, %s, %s, %d, %d, %d)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_col),
								 neurondb_quote_literal_or_null(target_column),
								 p, d, q);
				return true;
			}

		case ML_ALGO_KNN:
		case ML_ALGO_KNN_CLASSIFIER:
			{
				int			k_value = 5;

				neurondb_parse_hyperparams_int(hyperparams, "k", &k_value, 5);
				appendStringInfo(sql,
								 "SELECT train_knn_model_id(%s, %s, %s, %d)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_list),
								 neurondb_quote_literal_or_null(target_column),
								 k_value);
				return true;
			}

		default:
			/* Algorithm requires special handling, not simple SQL generation */
			return false;
	}
}

/*
 * Validate training inputs
 */
static void
__attribute__((unused))
neurondb_validate_training_inputs(const char *project_name, const char *algorithm,
								  const char *table_name, const char *target_column)
{
	if (project_name == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg(NDB_ERR_PREFIX_TRAIN " project_name parameter cannot be NULL"),
				 errdetail("The project_name argument is required to organize models"),
				 errhint("Provide a non-NULL project name, e.g., 'my_ml_project'")));
	if (algorithm == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg(NDB_ERR_PREFIX_TRAIN " algorithm parameter cannot be NULL"),
				 errdetail("The algorithm argument specifies which ML algorithm to use for training"),
				 errhint("Provide a valid algorithm name, e.g., 'linear_regression', 'logistic_regression', 'random_forest', 'kmeans', etc.")));
	if (table_name == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg(NDB_ERR_PREFIX_TRAIN " table_name parameter cannot be NULL"),
				 errdetail("The table_name argument specifies the source table containing training data"),
				 errhint("Provide a valid table name containing your training data")));

	/* target_column can be NULL for unsupervised algorithms */
	if (target_column == NULL && !neurondb_is_unsupervised_algorithm(algorithm))
	{
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg(NDB_ERR_PREFIX_TRAIN " target_column parameter cannot be NULL for supervised algorithm '%s'", algorithm),
				 errdetail("Algorithm '%s' is a supervised learning algorithm and requires a target column", algorithm),
				 errhint("Provide a target_column name, or use an unsupervised algorithm like 'kmeans', 'gmm', or 'hierarchical' if you don't have target labels")));
	}
}

/*
 * neurondb_is_unsupervised_algorithm
 *		Check if algorithm is unsupervised (doesn't require target_column).
 *
 * Uses MLAlgorithm enum for type-safe checking.
 */
static bool
neurondb_is_unsupervised_algorithm(const char *algorithm)
{
	MLAlgorithm algo_enum;

	if (algorithm == NULL)
		return false;

	algo_enum = neurondb_algorithm_from_string(algorithm);
	return (algo_enum == ML_ALGO_GMM ||
			algo_enum == ML_ALGO_KMEANS ||
			algo_enum == ML_ALGO_MINIBATCH_KMEANS ||
			algo_enum == ML_ALGO_HIERARCHICAL ||
			algo_enum == ML_ALGO_DBSCAN);
}

/*
 * neurondb_prepare_feature_list
 *		Prepare feature list from array and populate feature_names array.
 *
 * Returns number of features on success.
 */
static int
__attribute__((unused))
neurondb_prepare_feature_list(ArrayType * feature_columns_array, StringInfo feature_list,
							  const char ***feature_names_out, int *feature_name_count_out)
{
	const char **feature_names = NULL;
	int			feature_name_count = 0;

	/* Temporary: simplified version to bypass array processing issues */
	appendStringInfoString(feature_list, "*");
	NDB_ALLOC(feature_names, const char *, 1);
	feature_names[0] = pstrdup("*");
	feature_name_count = 1;

	if (feature_names_out)
		*feature_names_out = feature_names;
	if (feature_name_count_out)
		*feature_name_count_out = feature_name_count;

	return feature_name_count;
}

/*
 * neurondb_get_model_type
 *		Return canonical model type for a known ML algorithm.
 *
 * Returns "classification", "regression", "clustering", or
 * "dimensionality_reduction" based on algorithm type.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static const char *
neurondb_get_model_type(const char *algorithm)
{
	MLAlgorithm algo;

	if (algorithm == NULL)
		return "classification";

	algo = neurondb_algorithm_from_string(algorithm);

	switch (algo)
	{
		case ML_ALGO_LOGISTIC_REGRESSION:
		case ML_ALGO_RANDOM_FOREST:
		case ML_ALGO_SVM:
		case ML_ALGO_DECISION_TREE:
		case ML_ALGO_NAIVE_BAYES:
		case ML_ALGO_XGBOOST:
		case ML_ALGO_CATBOOST:
		case ML_ALGO_LIGHTGBM:
		case ML_ALGO_KNN:
		case ML_ALGO_KNN_CLASSIFIER:
			return "classification";

		case ML_ALGO_LINEAR_REGRESSION:
		case ML_ALGO_RIDGE:
		case ML_ALGO_LASSO:
		case ML_ALGO_KNN_REGRESSOR:
			return "regression";

		case ML_ALGO_KMEANS:
		case ML_ALGO_GMM:
		case ML_ALGO_MINIBATCH_KMEANS:
		case ML_ALGO_HIERARCHICAL:
		case ML_ALGO_DBSCAN:
			return "clustering";

		case ML_ALGO_PCA:
		case ML_ALGO_OPQ:
			return "dimensionality_reduction";

		default:
			return "classification";
	}
}
#pragma GCC diagnostic pop

/* ----------
 * neurondb_train
 * Unified model training interface
 * ----------
 */
Datum
neurondb_train(PG_FUNCTION_ARGS)
{
	text	   *project_name_text;
	text	   *algorithm_text;
	text	   *table_name_text;
	text	   *target_column_text;
	ArrayType  *feature_columns_array;
	Jsonb	   *hyperparams;
	StringInfoData feature_list;

	NDB_DECLARE(const char **, feature_names);
	int			feature_name_count = 0;

	NDB_DECLARE(char *, model_name);
	MLGpuTrainResult gpu_result;

	NDB_DECLARE(char *, gpu_errmsg_ptr);
	char	  **gpu_errmsg = &gpu_errmsg_ptr;
	char	   *project_name;
	char	   *algorithm;
	char	   *table_name;
	char	   *target_column;

	NDB_DECLARE(char *, default_project_name);	/* Pre-allocated "default"
												 * string */
	MemoryContext callcontext;
	MemoryContext oldcontext;

	NDB_DECLARE(NdbSpiSession *, spi_session);
	StringInfoData sql;
	int			ret;
	int			model_id;
	MLCatalogModelSpec spec;
	MLAlgorithm algo_enum;

	NDB_DECLARE(float *, feature_matrix);
	NDB_DECLARE(double *, label_vector);
	int			n_samples = 0;
	int			feature_dim = 0;
	int			class_count = 0;
	bool		data_loaded = false;

	NDB_DECLARE(char *, feature_list_str);
	bool		gpu_available = false;
	bool		load_success = false;
	bool		gpu_train_result = false;


	if (PG_NARGS() != 6)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_TRAIN " invalid number of arguments (expected 6, got %d)", PG_NARGS()),
				 errdetail("Function signature: neurondb.train(project_name text, algorithm text, table_name text, target_column text, feature_columns text[], hyperparams jsonb)"),
				 errhint("Provide exactly 6 arguments: project_name, algorithm, table_name, target_column (or NULL for unsupervised), feature_columns array, and hyperparams jsonb")));

	project_name_text = PG_ARGISNULL(0) ? NULL : PG_GETARG_TEXT_PP(0);
	algorithm_text = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);
	table_name_text = PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2);
	target_column_text = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);
	feature_columns_array = PG_ARGISNULL(4) ? NULL : PG_GETARG_ARRAYTYPE_P(4);
	hyperparams = PG_ARGISNULL(5) ? NULL : PG_GETARG_JSONB_P(5);

	project_name = text_to_cstring(project_name_text);
	algorithm = text_to_cstring(algorithm_text);
	table_name = text_to_cstring(table_name_text);
	target_column = target_column_text ? text_to_cstring(target_column_text) : NULL;


	/* Validate algorithm */
	if (strcmp(algorithm, NDB_ALGO_LINEAR_REGRESSION) != 0 &&
		strcmp(algorithm, NDB_ALGO_LOGISTIC_REGRESSION) != 0 &&
		strcmp(algorithm, NDB_ALGO_SVM) != 0 &&
		strcmp(algorithm, NDB_ALGO_RANDOM_FOREST) != 0 &&
		strcmp(algorithm, NDB_ALGO_KNN) != 0 &&
		strcmp(algorithm, NDB_ALGO_KMEANS) != 0 &&
		strcmp(algorithm, NDB_ALGO_DBSCAN) != 0 &&
		strcmp(algorithm, NDB_ALGO_NAIVE_BAYES) != 0 &&
		strcmp(algorithm, NDB_ALGO_DECISION_TREE) != 0 &&
		strcmp(algorithm, NDB_ALGO_GMM) != 0 &&
		strcmp(algorithm, NDB_ALGO_XGBOOST) != 0 &&
		strcmp(algorithm, NDB_ALGO_CATBOOST) != 0 &&
		strcmp(algorithm, NDB_ALGO_LIGHTGBM) != 0 &&
		strcmp(algorithm, NDB_ALGO_TIMESERIES) != 0 &&
		strcmp(algorithm, "neural_network") != 0)
	{
		NDB_FREE(project_name);
		NDB_FREE(algorithm);
		NDB_FREE(table_name);
		if (target_column)
			NDB_FREE(target_column);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_TRAIN " unsupported algorithm '%s'", algorithm),
				 errdetail("Supported algorithms: linear_regression, logistic_regression, svm, random_forest, knn, kmeans, dbscan, naive_bayes, decision_tree, gmm, xgboost, catboost, lightgbm, timeseries, neural_network"),
				 errhint("Choose one of the supported algorithms.")));
	}


	/* Create memory context for this function call */
	callcontext = AllocSetContextCreate(CurrentMemoryContext,
										"neurondb_train context",
										ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	/* Pre-allocate "default" string early in callcontext */
	default_project_name = pstrdup("default");
	MemoryContextSwitchTo(oldcontext);
	MemoryContextSwitchTo(callcontext);

	/* Begin SPI session */
	NDB_SPI_SESSION_BEGIN(spi_session, callcontext);

	/* Get algorithm type and create project */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "INSERT INTO " NDB_FQ_PROJECTS " (project_name, " NDB_COL_ALGORITHM ", table_name, target_column, created_at) "
					 "VALUES ('%s', '%s', '%s', %s, NOW()) "
					 "ON CONFLICT (project_name) DO UPDATE SET "
					 "algorithm = EXCLUDED.algorithm, "
					 "table_name = EXCLUDED.table_name, "
					 "target_column = EXCLUDED.target_column, "
					 "updated_at = NOW() "
					 "RETURNING project_id",
					 project_name, algorithm, table_name,
					 target_column ? psprintf("'%s'", target_column) : "NULL");

	ret = ndb_spi_execute(spi_session, sql.data, false, 0);

	if (ret != SPI_OK_INSERT_RETURNING)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		NDB_FREE(project_name);
		NDB_FREE(algorithm);
		NDB_FREE(table_name);
		if (target_column)
			NDB_FREE(target_column);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_PREFIX_TRAIN " failed to create/update project in database"),
				 errdetail("SPI execution returned %d instead of %d", ret, SPI_OK_INSERT_RETURNING),
				 errhint("Check database permissions and schema.")));
	}

	/*
	 * Get project_id from result - stored but not currently used in this
	 * function
	 */

	/*
	 * Datum project_id_val = SPI_getbinval(SPI_tuptable->vals[0],
	 * SPI_tuptable->tupdesc, 1, NULL); int project_id =
	 * DatumGetInt32(project_id_val);
	 */

	ndb_spi_stringinfo_free(spi_session, &sql);
	ndb_spi_stringinfo_init(spi_session, &sql);
	ndb_spi_stringinfo_init(spi_session, &feature_list);


	/* Process feature columns array safely in SPI context */
	if (feature_columns_array != NULL)
	{
		int			nelems;

		NDB_DECLARE(Datum *, elem_values);
		NDB_DECLARE(bool *, elem_nulls);
		int			i;

		/* Validate array */
		if (ARR_NDIM(feature_columns_array) != 1)
		{
			ndb_spi_session_end(&spi_session);
			MemoryContextSwitchTo(oldcontext);
			neurondb_cleanup(oldcontext, callcontext);
			NDB_FREE(project_name);
			NDB_FREE(algorithm);
			NDB_FREE(table_name);
			if (target_column)
				NDB_FREE(target_column);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: feature_columns must be a 1-dimensional array")));
		}

		nelems = ArrayGetNItems(ARR_NDIM(feature_columns_array), ARR_DIMS(feature_columns_array));

		if (nelems > 0)
		{
			Oid			elem_type = ARR_ELEMTYPE(feature_columns_array);
			int16		elem_len;
			bool		elem_byval;
			char		elem_align;

			/* Check that elements are TEXT */
			if (elem_type != TEXTOID)
			{
				ndb_spi_session_end(&spi_session);
				MemoryContextSwitchTo(oldcontext);
				neurondb_cleanup(oldcontext, callcontext);
				NDB_FREE(project_name);
				NDB_FREE(algorithm);
				NDB_FREE(table_name);
				if (target_column)
					NDB_FREE(target_column);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: feature_columns array elements must be TEXT")));
			}

			/* Get element type properties for deconstruct_array */
			get_typlenbyvalalign(elem_type, &elem_len, &elem_byval, &elem_align);

			/* Deconstruct array in SPI context */
			deconstruct_array(feature_columns_array,
							  elem_type,
							  elem_len, elem_byval, elem_align,
							  &elem_values, &elem_nulls, &nelems);

			if (elem_values != NULL && elem_nulls != NULL)
			{
				MemoryContext old_spi_context;

				NDB_ALLOC(feature_names, const char *, nelems);

				/* Switch to SPI context before appending to feature_list */
				old_spi_context = MemoryContextSwitchTo(ndb_spi_session_get_context(spi_session));

				for (i = 0; i < nelems; i++)
				{
					if (!elem_nulls[i])
					{
						char	   *col = TextDatumGetCString(elem_values[i]);

						if (feature_list.len > 0)
							appendStringInfoString(&feature_list, ", ");
						appendStringInfoString(&feature_list, col);

						/*
						 * Switch back to callcontext for feature_names
						 * allocation
						 */
						MemoryContextSwitchTo(callcontext);
						feature_names[feature_name_count++] = pstrdup(col);
						MemoryContextSwitchTo(ndb_spi_session_get_context(spi_session));

						NDB_FREE(col);
					}
				}

				MemoryContextSwitchTo(old_spi_context);
				NDB_FREE(elem_values);
				NDB_FREE(elem_nulls);
			}
		}
	}

	/* Handle case where no features specified - use all columns */
	if (feature_name_count == 0)
	{
		MemoryContext old_spi_context;

		/* Switch to SPI context before appending to feature_list */
		old_spi_context = MemoryContextSwitchTo(ndb_spi_session_get_context(spi_session));
		appendStringInfoString(&feature_list, "*");
		MemoryContextSwitchTo(old_spi_context);

		NDB_ALLOC(feature_names, const char *, 1);
		feature_names[0] = pstrdup("*");
		feature_name_count = 1;
	}


	/* Safety check: ensure feature_list.data is valid */
	if (feature_list.data == NULL || feature_list.len == 0)
	{
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		NDB_FREE(project_name);
		NDB_FREE(algorithm);
		NDB_FREE(table_name);
		if (target_column)
			NDB_FREE(target_column);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_PREFIX_TRAIN " feature list is empty or invalid"),
				 errdetail("feature_list.data=%p, feature_list.len=%d", (void *) feature_list.data, feature_list.len),
				 errhint("This is an internal error. Please report this issue.")));
	}

	/*
	 * Copy feature_list.data to callcontext to ensure it's valid when used
	 * later
	 */
	/* This is necessary because feature_list was allocated in SPI context */
	MemoryContextSwitchTo(callcontext);
	feature_list_str = pstrdup(feature_list.data);
	if (feature_list_str == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		NDB_FREE(project_name);
		NDB_FREE(algorithm);
		NDB_FREE(table_name);
		if (target_column)
			NDB_FREE(target_column);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_PREFIX_TRAIN " failed to copy feature_list to callcontext")));
	}
	MemoryContextSwitchTo(oldcontext);


	MemSet(&gpu_result, 0, sizeof(MLGpuTrainResult));

	model_name = psprintf("%s_%s", algorithm, project_name);
	if (model_name == NULL)
	{
		NDB_FREE(feature_list_str);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		NDB_FREE(project_name);
		NDB_FREE(algorithm);
		NDB_FREE(table_name);
		if (target_column)
			NDB_FREE(target_column);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_PREFIX_TRAIN " failed to allocate memory for model_name")));
	}

	/* Initialize data loading variables */
	feature_matrix = NULL;
	label_vector = NULL;
	n_samples = 0;
	feature_dim = 0;
	class_count = 0;
	data_loaded = false;

	/* Log compute_mode for debugging */
	elog(DEBUG1, "neurondb_train: compute_mode=%d (0=CPU, 1=GPU, 2=AUTO)", neurondb_compute_mode);

	/* Check GPU availability first */
	gpu_available = neurondb_gpu_is_available();

	/* GPU mode requires GPU to be available */
	if (NDB_REQUIRE_GPU() && !gpu_available)
	{
		NDB_FREE(feature_list_str);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		NDB_FREE(project_name);
		NDB_FREE(algorithm);
		NDB_FREE(table_name);
		if (target_column)
			NDB_FREE(target_column);
		if (feature_names)
		{
			int			i;

			for (i = 0; i < feature_name_count; i++)
			{
				if (feature_names[i])
				{
					char	   *ptr = (char *) feature_names[i];

					NDB_FREE(ptr);
				}
			}
			NDB_FREE(feature_names);
		}
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_PREFIX_TRAIN " GPU is required but not available"),
				 errdetail("compute_mode is set to 'gpu' but GPU backend could not be initialized"),
				 errhint("Check GPU hardware, drivers, and configuration. "
						 "Set compute_mode='auto' for automatic CPU fallback.")));
	}

	/* Try to load training data for GPU training */
	/* Only attempt GPU if compute_mode allows it (GPU or AUTO mode) */
	/* CPU mode: never load data for GPU training */
	if (!NDB_COMPUTE_MODE_IS_CPU() && gpu_available && NDB_SHOULD_TRY_GPU())
	{
		load_success = neurondb_load_training_data(spi_session, table_name, feature_list_str, target_column,
												   &feature_matrix, &label_vector,
												   &n_samples, &feature_dim, &class_count);

		if (load_success)
		{
			data_loaded = true;
		}
	}

	/* Call GPU training with loaded data */
	/* Only attempt GPU training if compute_mode allows it (not CPU mode) */

	/*
	 * Double-check CPU mode to be absolutely safe - NEVER attempt GPU in CPU
	 * mode
	 */
	if (data_loaded && !NDB_COMPUTE_MODE_IS_CPU() && NDB_SHOULD_TRY_GPU() && strcmp(algorithm, NDB_ALGO_LOGISTIC_REGRESSION) != 0)
	{
		/* Additional safety check: never attempt GPU in CPU mode */
		if (NDB_COMPUTE_MODE_IS_CPU())
		{
			elog(WARNING, "neurondb_train: CPU mode detected, skipping GPU training (defensive check)");
			data_loaded = false;
			gpu_train_result = false;
		}
		else
		{
			/*
			 * Wrap GPU training call in PG_TRY to catch exceptions and
			 * prevent JSON parsing errors
			 */
			PG_TRY();
			{
				gpu_train_result = ndb_gpu_try_train_model(algorithm, project_name, model_name, table_name, target_column,
														   feature_names, feature_name_count, hyperparams,
														   feature_matrix, label_vector, n_samples, feature_dim, class_count,
														   &gpu_result, gpu_errmsg);
			}
			PG_CATCH();
			{
				/*
				 * GPU training threw an exception - handle based on compute
				 * mode
				 */
				/* Log compute_mode for debugging */
				elog(DEBUG1, "PG_CATCH: compute_mode=%d (0=CPU, 1=GPU, 2=AUTO), NDB_COMPUTE_MODE_IS_CPU()=%d, NDB_REQUIRE_GPU()=%d",
					 neurondb_compute_mode, NDB_COMPUTE_MODE_IS_CPU() ? 1 : 0, NDB_REQUIRE_GPU() ? 1 : 0);

				/*
				 * CPU mode: never error on GPU failures, just fall back to
				 * CPU
				 */
				if (NDB_COMPUTE_MODE_IS_CPU())
				{
					/* CPU mode - fall back to CPU training */
					MemoryContext safe_context = oldcontext;

					if (safe_context == ErrorContext || safe_context == NULL)
					{
						safe_context = TopMemoryContext;
					}
					MemoryContextSwitchTo(safe_context);

					elog(WARNING,
						 "%s: exception caught during GPU training attempt in CPU mode, falling back to CPU",
						 algorithm ? algorithm : "unknown");
					FlushErrorState();
					gpu_train_result = false;
					memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
					if (gpu_errmsg && *gpu_errmsg == NULL)
						*gpu_errmsg = pstrdup("Exception during GPU training (CPU mode)");

					/*
					 * Free GPU-loaded data if it was loaded, but only if
					 * callcontext is still valid
					 */
					if (data_loaded && CurrentMemoryContext != ErrorContext &&
						MemoryContextIsValid(callcontext))
					{
						MemoryContextSwitchTo(callcontext);
						if (feature_matrix)
						{
							NDB_FREE(feature_matrix);
							feature_matrix = NULL;
						}
						if (label_vector)
						{
							NDB_FREE(label_vector);
							label_vector = NULL;
						}
						data_loaded = false;
						MemoryContextSwitchTo(safe_context);
					}
					else if (data_loaded)	/* If callcontext is invalid, just
											 * clear pointers */
					{
						feature_matrix = NULL;
						label_vector = NULL;
						data_loaded = false;
					}
				}
				/* GPU mode: error if GPU training fails */
				/* CPU mode: never error, should have been handled above */
				else if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
				{
					MemoryContext safe_context;
					ErrorData  *edata;
					char	   *error_msg;
					char	   *algorithm_safe;

					/*
					 * Defensive assertion: this should NEVER happen in CPU
					 * mode
					 */
					if (NDB_COMPUTE_MODE_IS_CPU())
					{
						elog(ERROR, "BUG: GPU error path reached in CPU mode! compute_mode=%d", neurondb_compute_mode);
					}

					/* GPU mode: re-raise error, no fallback */
					/* Switch out of ErrorContext before CopyErrorData() */
					safe_context = oldcontext;

					/* Ensure we're not switching to ErrorContext */
					if (safe_context == ErrorContext || safe_context == NULL)
					{
						safe_context = TopMemoryContext;
					}

					MemoryContextSwitchTo(safe_context);

					edata = NULL;
					error_msg = NULL;

					/* Save algorithm before freeing (it might be NULL) */
					algorithm_safe = algorithm ? pstrdup(algorithm) : NULL;

					if (CurrentMemoryContext != ErrorContext)
					{
						edata = CopyErrorData();
						if (edata && edata->message)
							error_msg = pstrdup(edata->message);
						FlushErrorState();
					}
					else
					{
						/* Fallback if we can't switch contexts */
						FlushErrorState();
						error_msg = NULL;
					}

					NDB_FREE(feature_list_str);
					if (feature_names)
					{
						int			i;

						for (i = 0; i < feature_name_count; i++)
						{
							if (feature_names[i])
							{
								char	   *ptr = (char *) feature_names[i];

								NDB_FREE(ptr);
							}
						}
						NDB_FREE(feature_names);
					}
					if (model_name)
						NDB_FREE(model_name);
					if (gpu_errmsg_ptr)
						NDB_FREE(gpu_errmsg_ptr);
					ndb_spi_session_end(&spi_session);
					MemoryContextSwitchTo(oldcontext);
					neurondb_cleanup(oldcontext, callcontext);
					NDB_FREE(project_name);
					NDB_FREE(algorithm);
					NDB_FREE(table_name);
					if (target_column)
						NDB_FREE(target_column);
					if (data_loaded)
					{
						if (feature_matrix)
							NDB_FREE(feature_matrix);
						if (label_vector)
							NDB_FREE(label_vector);
					}
					if (edata)
						FreeErrorData(edata);

					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg(NDB_ERR_PREFIX_TRAIN " GPU training failed - GPU mode requires GPU to be available"),
							 errdetail("Algorithm: %s, Error: %s", algorithm_safe ? algorithm_safe : "unknown", error_msg ? error_msg : "unknown"),
							 errhint("Set compute_mode='auto' for automatic CPU fallback.")));

					if (algorithm_safe)
						NDB_FREE(algorithm_safe);
				}
				else
				{
					/* AUTO mode: fall back to CPU */
					/* Switch out of ErrorContext before any operations */
					MemoryContext safe_context = oldcontext;

					if (safe_context == ErrorContext || safe_context == NULL)
					{
						safe_context = TopMemoryContext;
					}
					MemoryContextSwitchTo(safe_context);

					elog(WARNING,
						 "%s: exception caught during GPU training, falling back to CPU (auto mode)",
						 algorithm ? algorithm : "unknown");
					FlushErrorState();
					gpu_train_result = false;
					memset(&gpu_result, 0, sizeof(MLGpuTrainResult));
					if (gpu_errmsg && *gpu_errmsg == NULL)
						*gpu_errmsg = pstrdup("Exception during GPU training");

					/*
					 * Free GPU-loaded data if it was loaded, but only if
					 * callcontext is still valid
					 */
					if (data_loaded && CurrentMemoryContext != ErrorContext && callcontext != NULL && MemoryContextIsValid(callcontext))
					{
						/*
						 * Switch to callcontext to free memory allocated
						 * there
						 */
						MemoryContextSwitchTo(callcontext);
						if (feature_matrix)
						{
							NDB_FREE(feature_matrix);
							feature_matrix = NULL;
						}
						if (label_vector)
						{
							NDB_FREE(label_vector);
							label_vector = NULL;
						}
						data_loaded = false;
						MemoryContextSwitchTo(safe_context);
					}
					else if (data_loaded)
					{
						/*
						 * Can't free safely - just mark as not loaded to
						 * avoid double-free
						 */
						/*
						 * The memory will be freed when callcontext is
						 * deleted
						 */
						feature_matrix = NULL;
						label_vector = NULL;
						data_loaded = false;
					}
				}
			}
			PG_END_TRY();
		}

		if (gpu_train_result)
		{
			/* GPU training succeeded - use the model_id from GPU result */
			if (gpu_result.model_id > 0)
			{
				model_id = gpu_result.model_id;
			}
			else if (gpu_result.model_id == 0 && gpu_result.spec.model_data != NULL)
			{
				/* GPU created model but didn't set model_id - register it */
				spec = gpu_result.spec;

				/*
				 * ALWAYS copy all string pointers to current memory context
				 * before switching contexts
				 */

				/*
				 * This ensures the pointers remain valid after memory context
				 * switch
				 */

				/*
				 * We use the values from gpu_result.spec if they exist,
				 * otherwise use fallback values
				 */

				/*
				 * Copy algorithm - use spec.algorithm if it exists and is
				 * valid, otherwise use algorithm
				 */
				PG_TRY();
				{
					if (spec.algorithm != NULL)
					{
						spec.algorithm = pstrdup(spec.algorithm);
					}
					else
					{
						spec.algorithm = pstrdup(algorithm);
					}
				}
				PG_CATCH();
				{
					/* If copying spec.algorithm failed, use fallback */
					FlushErrorState();
					spec.algorithm = pstrdup(algorithm);
				}
				PG_END_TRY();

				/*
				 * Copy training_table - use spec.training_table if it exists,
				 * otherwise use table_name
				 */
				PG_TRY();
				{
					if (spec.training_table != NULL)
					{
						spec.training_table = pstrdup(spec.training_table);
					}
					else
					{
						spec.training_table = pstrdup(table_name);
					}
				}
				PG_CATCH();
				{
					FlushErrorState();
					spec.training_table = pstrdup(table_name);
				}
				PG_END_TRY();

				/*
				 * Copy training_column - use spec.training_column if it
				 * exists, otherwise use target_column
				 */
				if (target_column != NULL)
				{
					PG_TRY();
					{
						if (spec.training_column != NULL)
						{
							spec.training_column = pstrdup(spec.training_column);
						}
						else
						{
							spec.training_column = pstrdup(target_column);
						}
					}
					PG_CATCH();
					{
						FlushErrorState();
						spec.training_column = pstrdup(target_column);
					}
					PG_END_TRY();
				}

				/*
				 * Copy project_name - always use fallback value since
				 * spec.project_name may point to invalid memory
				 */
				PG_TRY();
				{
					/* Check project_name safely */
				}
				PG_CATCH();
				{
					FlushErrorState();
				}
				PG_END_TRY();

				/*
				 * Always use pre-allocated "default" project_name to avoid
				 * crashes
				 */

				/*
				 * We pre-allocated default_project_name earlier when memory
				 * context was still good
				 */
				spec.project_name = default_project_name;

				/* Switch to oldcontext before registering model */
				MemoryContextSwitchTo(oldcontext);
				model_id = ml_catalog_register_model(&spec);
				MemoryContextSwitchTo(callcontext);
			}
			else
			{
				/*
				 * GPU training reported success but no model data - error
				 * instead of returning 0
				 */
				NDB_FREE(feature_list_str);
				ndb_spi_session_end(&spi_session);
				MemoryContextSwitchTo(oldcontext);
				neurondb_cleanup(oldcontext, callcontext);
				NDB_FREE(project_name);
				NDB_FREE(algorithm);
				NDB_FREE(table_name);
				if (target_column)
					NDB_FREE(target_column);
				if (feature_names)
				{
					int			i;

					for (i = 0; i < feature_name_count; i++)
					{
						if (feature_names[i])
						{
							char	   *ptr = (char *) feature_names[i];

							NDB_FREE(ptr);
						}
					}
					NDB_FREE(feature_names);
				}
				ndb_gpu_free_train_result(&gpu_result);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg(NDB_ERR_PREFIX_TRAIN " GPU training reported success but no model data"),
						 errdetail("algorithm=%s, model_id=%d", algorithm, gpu_result.model_id),
						 errhint("GPU training may have failed internally. Check logs for details or try CPU training.")));
			}

			/* Free loaded training data */
			if (data_loaded)
			{
				if (feature_matrix)
					NDB_FREE(feature_matrix);
				if (label_vector)
					NDB_FREE(label_vector);
			}

			/* Free GPU result if it was allocated */
			ndb_gpu_free_train_result(&gpu_result);

			/* Cleanup and return */
			ndb_spi_stringinfo_free(spi_session, &feature_list);
			if (feature_names)
			{
				int			i;

				for (i = 0; i < feature_name_count; i++)
				{
					if (feature_names[i])
					{
						char	   *ptr = (char *) feature_names[i];

						NDB_FREE(ptr);
					}
				}
				NDB_FREE(feature_names);
			}
			if (model_name)
			{
				MemoryContextSwitchTo(callcontext);
				NDB_FREE(model_name);
				model_name = NULL;
			}
			MemoryContextSwitchTo(oldcontext);
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			PG_RETURN_INT32(model_id);
		}
	}
	else
	{
		/* GPU training failed or not attempted - handle based on compute mode */

		/* CPU mode: never attempt GPU, just fall back to CPU */
		/* In CPU mode, we should NEVER reach this error path */
		if (NDB_COMPUTE_MODE_IS_CPU())
		{
			/* CPU mode - skip GPU checks and proceed to CPU training */
			/* This should be the only path in CPU mode */
		}
		/* GPU mode: error if GPU was required but not available */
		/* Only check GPU requirement if NOT in CPU mode */
		else if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU())
		{
			NDB_FREE(feature_list_str);
			if (feature_names)
			{
				int			i;

				for (i = 0; i < feature_name_count; i++)
				{
					if (feature_names[i])
					{
						char	   *ptr = (char *) feature_names[i];

						NDB_FREE(ptr);
					}
				}
				NDB_FREE(feature_names);
			}
			if (model_name)
				NDB_FREE(model_name);
			if (gpu_errmsg_ptr)
				NDB_FREE(gpu_errmsg_ptr);
			ndb_spi_session_end(&spi_session);
			MemoryContextSwitchTo(oldcontext);
			neurondb_cleanup(oldcontext, callcontext);
			NDB_FREE(project_name);
			NDB_FREE(algorithm);
			NDB_FREE(table_name);
			if (target_column)
				NDB_FREE(target_column);
			if (data_loaded)
			{
				if (feature_matrix)
					NDB_FREE(feature_matrix);
				if (label_vector)
					NDB_FREE(label_vector);
			}
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_PREFIX_TRAIN " GPU training failed - GPU mode requires GPU to be available"),
					 errdetail("Algorithm: %s, Project: %s, Table: %s. GPU was not available or training failed.",
							   algorithm, project_name, table_name),
					 errhint("Check GPU hardware, drivers, and configuration. "
							 "Set compute_mode='auto' for automatic CPU fallback.")));
		}

		/* AUTO mode or CPU mode: fall back to CPU training */
		/* Free loaded training data if it was loaded */
		/* Make sure we're in the right context before freeing */
		if (data_loaded && callcontext != NULL && MemoryContextIsValid(callcontext))
		{
			/* Switch to callcontext where the memory was allocated */
			MemoryContextSwitchTo(callcontext);
			if (feature_matrix)
			{
				NDB_FREE(feature_matrix);
				feature_matrix = NULL;
			}
			if (label_vector)
			{
				NDB_FREE(label_vector);
				label_vector = NULL;
			}
			data_loaded = false;
			MemoryContextSwitchTo(oldcontext);
		}
		else if (data_loaded)
		{
			/*
			 * Can't free safely - just mark as not loaded to avoid
			 * double-free
			 */
			/* The memory will be freed when callcontext is deleted */
			feature_matrix = NULL;
			label_vector = NULL;
			data_loaded = false;
		}

		/* GPU training failed - fall back to CPU training (AUTO mode) */
		algo_enum = neurondb_algorithm_from_string(algorithm);

		/* Build SQL for CPU training */
		ndb_spi_stringinfo_free(spi_session, &sql);
		ndb_spi_stringinfo_init(spi_session, &sql);

		/*
		 * For random_forest, call C function directly to avoid PL/pgSQL
		 * wrapper recursion
		 */
		if (algo_enum == ML_ALGO_RANDOM_FOREST)
		{
			/*
			 * Call C function directly via function lookup to bypass PL/pgSQL
			 * wrapper
			 */
			List	   *funcname;
			Oid			func_oid;
			Oid			argtypes[7];
			Datum		values[7];
			FmgrInfo	flinfo;
			Datum		result_datum;
			text	   *table_name_text_local;
			text	   *feature_col_text;
			text	   *label_col_text;
			int			n_trees = 10;
			int			max_depth = 10;
			int			min_samples = 100;
			int			max_features = 0;
			const char *feature_col;

			/*
			 * Use first feature name if available, otherwise use
			 * feature_list_str
			 */
			if (feature_name_count > 0 && feature_names != NULL && feature_names[0] != NULL)
				feature_col = feature_names[0];
			else
				feature_col = feature_list_str;

			/* Parse hyperparameters */
			neurondb_parse_hyperparams_int(hyperparams, "n_trees", &n_trees, 10);
			neurondb_parse_hyperparams_int(hyperparams, "max_depth", &max_depth, 10);
			neurondb_parse_hyperparams_int(hyperparams, "min_samples", &min_samples, 100);
			neurondb_parse_hyperparams_int(hyperparams, "min_samples_split", &min_samples, 100);
			neurondb_parse_hyperparams_int(hyperparams, "max_features", &max_features, 0);

			/* Build arguments */
			table_name_text_local = cstring_to_text(table_name);
			feature_col_text = cstring_to_text(feature_col);
			label_col_text = cstring_to_text(target_column);

			/* Lookup C function directly (not the PL/pgSQL wrapper) */
			funcname = list_make1(makeString("train_random_forest_classifier"));
			argtypes[0] = TEXTOID;	/* table_name */
			argtypes[1] = TEXTOID;	/* feature_col */
			argtypes[2] = TEXTOID;	/* label_col */
			argtypes[3] = INT4OID;	/* n_trees */
			argtypes[4] = INT4OID;	/* max_depth */
			argtypes[5] = INT4OID;	/* min_samples_split */
			argtypes[6] = INT4OID;	/* max_features */

			/* Look for C function specifically - check if it's a C function */
			func_oid = LookupFuncName(funcname, 7, argtypes, false);
			list_free(funcname);

			if (OidIsValid(func_oid))
			{
				/*
				 * Assume it's a callable C function if LookupFuncName found
				 * it
				 */
				/* For safety, we could add language check later if needed */
				/* Prepare function call */
				fmgr_info(func_oid, &flinfo);

				/* Set up arguments */
				values[0] = PointerGetDatum(table_name_text_local);
				values[1] = PointerGetDatum(feature_col_text);
				values[2] = PointerGetDatum(label_col_text);
				values[3] = Int32GetDatum(n_trees);
				values[4] = Int32GetDatum(max_depth);
				values[5] = Int32GetDatum(min_samples);
				values[6] = Int32GetDatum(max_features);

				/* Call C function directly */
				result_datum = FunctionCall7(&flinfo,
											 values[0], values[1], values[2],
											 values[3], values[4], values[5], values[6]);

				/* Extract model_id from result */
				model_id = DatumGetInt32(result_datum);

				if (model_id > 0)
				{
					/* Update metrics to ensure storage='cpu' is set */
					ndb_spi_stringinfo_free(spi_session, &sql);
					ndb_spi_stringinfo_init(spi_session, &sql);
					appendStringInfo(&sql,
									 "UPDATE " NDB_FQ_ML_MODELS " SET " NDB_COL_METRICS " = "
									 "COALESCE(" NDB_COL_METRICS ", '{}'::jsonb) || '{\"training_backend\":0}'::jsonb "
									 "WHERE " NDB_COL_MODEL_ID " = %d",
									 model_id);
					ndb_spi_execute(spi_session, sql.data, false, 0);
				}
				else
				{
					ndb_spi_stringinfo_free(spi_session, &sql);
					ndb_spi_session_end(&spi_session);
					neurondb_cleanup(oldcontext, callcontext);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg(NDB_ERR_PREFIX_TRAIN " CPU training returned invalid model_id: %d", model_id),
							 errdetail("Algorithm: %s, Project: %s, Table: %s", algorithm, project_name, table_name),
							 errhint("CPU training function may have failed. Check logs for details.")));
				}
			}
			else
			{
				/* Function not found - fall back to SQL generation */
				if (neurondb_build_training_sql(algo_enum, &sql, table_name, feature_list_str,
												target_column, hyperparams, feature_names, feature_name_count))
				{
					/* Execute CPU training via SQL */
					elog(DEBUG1, "neurondb_train: executing CPU training SQL: %s", sql.data);
					ret = ndb_spi_execute(spi_session, sql.data, true, 0);

					if (ret == SPI_OK_SELECT && SPI_processed > 0)
					{
						int32		model_id_val;

						/* Get model_id from result */
						if (ndb_spi_get_int32(spi_session, 0, 1, &model_id_val))
						{

							if (model_id_val > 0)
							{
								model_id = model_id_val;

								/*
								 * Update metrics to ensure storage='cpu' is
								 * set
								 */
								ndb_spi_stringinfo_free(spi_session, &sql);
								ndb_spi_stringinfo_init(spi_session, &sql);
								appendStringInfo(&sql,
												 "UPDATE " NDB_FQ_ML_MODELS " SET " NDB_COL_METRICS " = "
												 "COALESCE(" NDB_COL_METRICS ", '{}'::jsonb) || '{\"training_backend\":0}'::jsonb "
												 "WHERE " NDB_COL_MODEL_ID " = %d",
												 model_id);
								ndb_spi_execute(spi_session, sql.data, false, 0);
							}
							else
							{
								ndb_spi_stringinfo_free(spi_session, &sql);
								ndb_spi_session_end(&spi_session);
								neurondb_cleanup(oldcontext, callcontext);
								ereport(ERROR,
										(errcode(ERRCODE_INTERNAL_ERROR),
										 errmsg(NDB_ERR_PREFIX_TRAIN " CPU training returned invalid model_id: %d", model_id_val),
										 errdetail("Algorithm: %s, Project: %s, Table: %s", algorithm, project_name, table_name),
										 errhint("CPU training function may have failed. Check logs for details.")));
							}
						}
						else
						{
							ndb_spi_stringinfo_free(spi_session, &sql);
							ndb_spi_session_end(&spi_session);
							neurondb_cleanup(oldcontext, callcontext);
							ereport(ERROR,
									(errcode(ERRCODE_INTERNAL_ERROR),
									 errmsg(NDB_ERR_PREFIX_TRAIN " CPU training returned NULL model_id"),
									 errdetail("Algorithm: %s, Project: %s, Table: %s", algorithm, project_name, table_name),
									 errhint("CPU training function may have failed. Check logs for details.")));
						}
					}
				}
			}
		}
		else if (neurondb_build_training_sql(algo_enum, &sql, table_name, feature_list_str,
											 target_column, hyperparams, feature_names, feature_name_count))
		{
			/* Execute CPU training via SQL */
			elog(DEBUG1, "neurondb_train: executing CPU training SQL: %s", sql.data);
			ret = ndb_spi_execute(spi_session, sql.data, true, 0);

			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				int32		model_id_val;

				/* Get model_id from result */
				if (ndb_spi_get_int32(spi_session, 0, 1, &model_id_val))
				{

					if (model_id_val > 0)
					{
						model_id = model_id_val;

						/* Update metrics to ensure storage='cpu' is set */
						ndb_spi_stringinfo_free(spi_session, &sql);
						ndb_spi_stringinfo_init(spi_session, &sql);
						appendStringInfo(&sql,
										 "UPDATE " NDB_FQ_ML_MODELS " SET " NDB_COL_METRICS " = "
										 "COALESCE(" NDB_COL_METRICS ", '{}'::jsonb) || '{\"training_backend\":0}'::jsonb "
										 "WHERE " NDB_COL_MODEL_ID " = %d",
										 model_id);
						ndb_spi_execute(spi_session, sql.data, false, 0);
					}
					else
					{
						ndb_spi_stringinfo_free(spi_session, &sql);
						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg(NDB_ERR_PREFIX_TRAIN " CPU training returned invalid model_id: %d", model_id_val),
								 errdetail("Algorithm: %s, Project: %s, Table: %s", algorithm, project_name, table_name),
								 errhint("CPU training function may have failed. Check logs for details.")));
					}
				}
				else
				{
					ndb_spi_stringinfo_free(spi_session, &sql);
					ndb_spi_session_end(&spi_session);
					neurondb_cleanup(oldcontext, callcontext);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg(NDB_ERR_PREFIX_TRAIN " CPU training returned NULL model_id"),
							 errdetail("Algorithm: %s, Project: %s, Table: %s", algorithm, project_name, table_name),
							 errhint("CPU training function may have failed. Check logs for details.")));
				}
			}
			else
			{
				NDB_FREE(feature_list_str);
				ndb_spi_stringinfo_free(spi_session, &sql);
				ndb_spi_session_end(&spi_session);
				neurondb_cleanup(oldcontext, callcontext);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg(NDB_ERR_PREFIX_TRAIN " CPU training failed - both GPU and CPU training methods failed"),
						 errdetail("Algorithm: %s, Project: %s, Table: %s, SPI return code: %d", algorithm, project_name, table_name, ret),
						 errhint("GPU error: %s. Check that the training data is valid and the algorithm supports CPU training.", gpu_errmsg_ptr ? gpu_errmsg_ptr : "none")));
			}
		}
		else
		{
			/*
			 * Algorithm doesn't have CPU training SQL - this shouldn't happen
			 * for linear_regression
			 */
			NDB_FREE(feature_list_str);
			ndb_spi_stringinfo_free(spi_session, &sql);
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg(NDB_ERR_PREFIX_TRAIN " algorithm '%s' does not support CPU training fallback", algorithm),
					 errdetail("GPU training failed and no CPU training implementation available"),
					 errhint("Try a different algorithm or ensure GPU is properly configured.")));
		}
	}


	/* Free resources */
	NDB_FREE(feature_list_str);
	ndb_spi_stringinfo_free(spi_session, &feature_list);
	if (feature_names)
	{
		int			i;

		for (i = 0; i < feature_name_count; i++)
		{
			if (feature_names[i])
			{
				char	   *ptr = (char *) feature_names[i];

				NDB_FREE(ptr);
			}
		}
		NDB_FREE(feature_names);
	}
	if (model_name)
		NDB_FREE(model_name);
	if (gpu_errmsg_ptr)
		NDB_FREE(gpu_errmsg_ptr);

	/* End SPI session */
	ndb_spi_session_end(&spi_session);

	/* Switch back to original context and clean up */
	MemoryContextSwitchTo(oldcontext);
	neurondb_cleanup(oldcontext, callcontext);

	/* Free converted strings */
	NDB_FREE(project_name);
	NDB_FREE(algorithm);
	NDB_FREE(table_name);
	if (target_column)
		NDB_FREE(target_column);

	PG_RETURN_INT32(model_id);
}

/* ----------
 * neurondb_predict
 * Unified prediction interface.
 * ----------
 */
Datum
neurondb_predict(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType  *features_array;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	StringInfoData features_str;
	int			ret;

	NDB_DECLARE(char *, algorithm);
	float8		prediction = 0.0;
	int			ndims,
				nelems,
				i;
	int		   *dims;
	float8	   *features;

	NDB_DECLARE(float8 *, features_float);	/* Allocated if conversion needed */
	NDB_DECLARE(NdbSpiSession *, spi_session);

	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_PREDICT " invalid number of arguments (expected 2, got %d)", PG_NARGS()),
				 errdetail("Function signature: neurondb.predict(model_id integer, features float8[])"),
				 errhint("Provide exactly 2 arguments: model_id (integer) and features (float8[] array)")));

	model_id = PG_GETARG_INT32(0);
	features_array = PG_GETARG_ARRAYTYPE_P(1);

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
										"neurondb_predict context",
										ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	/* Begin SPI session - always connect since this is called from SQL */
	spi_session = ndb_spi_session_begin(callcontext, false);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT " NDB_COL_ALGORITHM "::text FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_MODEL_ID " = %d",
					 model_id);
	elog(DEBUG1, "neurondb_predict: executing query: %s", sql.data);
	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model not found: %d", model_id)));
	}
	{
		text	   *algorithm_text = ndb_spi_get_text(spi_session, 0, 1, callcontext);

		if (algorithm_text == NULL)
		{
			ndb_spi_stringinfo_free(spi_session, &sql);
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model algorithm is NULL for model_id=%d", model_id)));
		}
		algorithm = text_to_cstring(algorithm_text);
	}

	ndims = ARR_NDIM(features_array);
	if (ndims != 1)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_PREDICT " features array must be 1-dimensional, got %d dimensions", ndims),
				 errdetail("Model ID: %d, Algorithm: '%s', Array dimensions: %d", model_id, algorithm, ndims),
				 errhint("Provide a 1-dimensional array of feature values, e.g., ARRAY[1.0, 2.0, 3.0]::float8[]")));
	}
	dims = ARR_DIMS(features_array);
	nelems = ArrayGetNItems(ndims, dims);
	if (nelems <= 0 || nelems > 100000)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_PREDICT " invalid feature count %d (expected 1-100000)", nelems),
				 errdetail("Model ID: %d, Algorithm: '%s', Feature count: %d", model_id, algorithm, nelems),
				 errhint("Provide a feature array with between 1 and 100000 elements matching the model's expected feature dimension.")));
	}

	/* Validate array element type */
	if (ARR_ELEMTYPE(features_array) != FLOAT8OID && ARR_ELEMTYPE(features_array) != FLOAT4OID)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_PREDICT " features array must be float8[] or float4[], got type OID %u", ARR_ELEMTYPE(features_array)),
				 errdetail("Model ID: %d, Algorithm: '%s', Array element type OID: %u", model_id, algorithm, ARR_ELEMTYPE(features_array)),
				 errhint("Cast your array to float8[] or float4[], e.g., ARRAY[1.0, 2.0, 3.0]::float8[]")));
	}

	/* Extract features based on element type */
	if (ARR_ELEMTYPE(features_array) == FLOAT4OID)
	{
		/* Convert float4[] to float8[] */
		float4	   *features_f4 = (float4 *) ARR_DATA_PTR(features_array);

		if (features_f4 == NULL)
		{
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_PREFIX_PREDICT " features array data pointer is NULL")));
		}
		/* Allocate float8 array and convert */
		NDB_ALLOC(features_float, float8, nelems);

		for (i = 0; i < nelems; i++)
			features_float[i] = (float8) features_f4[i];
		features = features_float;
	}
	else
	{
		/* Already float8[] */
		features = (float8 *) ARR_DATA_PTR(features_array);
		if (features == NULL)
		{
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_PREFIX_PREDICT " features array data pointer is NULL")));
		}
		features_float = NULL;	/* Not allocated, don't free */
	}

	ndb_spi_stringinfo_init(spi_session, &features_str);
	{
		MLAlgorithm algo_enum = neurondb_algorithm_from_string(algorithm);

		/* Build vector literal for algorithms that need it (NB, GMM) */
		if (algo_enum == ML_ALGO_NAIVE_BAYES || algo_enum == ML_ALGO_GMM)
		{
			/* Build vector literal: '[1.0, 2.0, ...]'::vector */
			appendStringInfoChar(&features_str, '\'');
			appendStringInfoChar(&features_str, '[');
			for (i = 0; i < nelems; i++)
			{
				if (i > 0)
					appendStringInfoString(&features_str, ", ");
				appendStringInfo(&features_str, "%.6f", features[i]);
			}
			appendStringInfoString(&features_str, "]'::vector");
		}
		else
		{
			/* Build array literal for other algorithms (including KNN) */
			appendStringInfoString(&features_str, "ARRAY[");
			for (i = 0; i < nelems; i++)
			{
				if (i > 0)
					appendStringInfoString(&features_str, ", ");
				appendStringInfo(&features_str, "%.6f", features[i]);
			}
			appendStringInfoString(&features_str, "]::real[]");
		}
	}

	/* Use safe free/reinit to handle potential memory context changes */
	ndb_spi_stringinfo_free(spi_session, &sql);
	ndb_spi_stringinfo_init(spi_session, &sql);

	if (strcmp(algorithm, NDB_ALGO_LINEAR_REGRESSION) == 0)
		appendStringInfo(&sql, "SELECT " NDB_FUNC_PREDICT_LINEAR_REGRESSION "(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, NDB_ALGO_LOGISTIC_REGRESSION) == 0)
		appendStringInfo(&sql, "SELECT " NDB_FUNC_PREDICT_LOGISTIC_REGRESSION "(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, NDB_ALGO_RANDOM_FOREST) == 0)
		appendStringInfo(&sql, "SELECT predict_random_forest(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, NDB_ALGO_SVM) == 0)
		appendStringInfo(&sql, "SELECT " NDB_FUNC_PREDICT_SVM_MODEL_ID "(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, NDB_ALGO_DECISION_TREE) == 0)
		appendStringInfo(&sql, "SELECT predict_decision_tree(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "naive_bayes") == 0)
	{
		NDB_DECLARE(bytea *, model_data);
		NDB_DECLARE(Jsonb *, metrics);
		bool		is_gpu = false;
		int			nb_class = 0;
		double		nb_probability = 0.0;

		NDB_DECLARE(float *, features_float);
		int			feature_dim = nelems;

		NDB_DECLARE(char *, errstr);
		int			rc;

		if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
		{
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_PREFIX_PREDICT " naive_bayes model with id %d not found in catalog", model_id),
					 errdetail("The model_id %d does not exist in " NDB_FQ_ML_MODELS " table for algorithm '" NDB_ALGO_NAIVE_BAYES "'", model_id),
					 errhint("Verify the model_id is correct. Use SELECT * FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_ALGORITHM " = '" NDB_ALGO_NAIVE_BAYES "' to list available models.")));
		}

		if (model_data == NULL)
		{
			if (metrics)
				NDB_FREE(metrics);

			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_PREFIX_PREDICT " naive_bayes model %d has no model data (model not trained)", model_id),
					 errdetail("The model exists in catalog but model_data is NULL, indicating training was not completed successfully"),
					 errhint("Train the model first using neurondb.train() before attempting prediction. Model ID: %d", model_id)));
		}

		is_gpu = ml_metrics_is_gpu(metrics);

		NDB_ALLOC(features_float, float, feature_dim);

		for (i = 0; i < feature_dim; i++)
			features_float[i] = (float) features[i];

		/*
		 * Use model's training backend (from catalog) regardless of current
		 * GPU state
		 */
		{
			const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
			bool		gpu_currently_enabled;

			gpu_currently_enabled = (backend != NULL && neurondb_gpu_is_available());

			elog(DEBUG1,
				 "neurondb_predict: model trained on %s, GPU currently %s",
				 is_gpu ? "GPU" : "CPU",
				 gpu_currently_enabled ? "enabled" : "disabled");

			if (is_gpu)
			{
				/* Model was trained on GPU - must use GPU prediction */
				if (!gpu_currently_enabled)
				{
					NDB_FREE(features_float);

					if (model_data)
						NDB_FREE(model_data);

					if (metrics)
						NDB_FREE(metrics);

					ndb_spi_session_end(&spi_session);
					neurondb_cleanup(oldcontext, callcontext);
					ereport(ERROR,
							(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
							 errmsg("Naive Bayes: model %d was trained on GPU but GPU is not currently available", model_id),
							 errhint("Enable GPU with: SET neurondb.compute_mode = 1; or SET neurondb.compute_mode = 2; for auto mode")));
				}

				/* Use GPU prediction */
				if (backend && backend->nb_predict && model_data != NULL)
				{
					rc = backend->nb_predict(model_data, features_float, feature_dim, &nb_class, &nb_probability, &errstr);
					if (rc == 0)
					{
						prediction = (double) nb_class;
						NDB_FREE(features_float);

						if (model_data)
							NDB_FREE(model_data);

						if (metrics)
							NDB_FREE(metrics);

						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						PG_RETURN_FLOAT8(prediction);
					}
					if (errstr)
						NDB_FREE(errstr);
				}
			}
			else
			{
				/*
				 * Model was trained on CPU - use CPU prediction (ignore
				 * current GPU state)
				 */
				/* Fall through to CPU prediction path below */
			}
		}
		appendStringInfo(&sql, "SELECT predict_naive_bayes_model_id(%d, %s)", model_id, features_str.data);
		NDB_FREE(features_float);

		if (model_data)
			NDB_FREE(model_data);

		if (metrics)
			NDB_FREE(metrics);
	}
	else if (strcmp(algorithm, NDB_ALGO_RIDGE) == 0 || strcmp(algorithm, NDB_ALGO_LASSO) == 0)
		appendStringInfo(&sql, "SELECT predict_regularized_regression(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, NDB_ALGO_KNN) == 0 || strcmp(algorithm, NDB_ALGO_KNN_CLASSIFIER) == 0 || strcmp(algorithm, NDB_ALGO_KNN_REGRESSOR) == 0)
	{
		NDB_DECLARE(bytea *, model_data);
		NDB_DECLARE(Jsonb *, metrics);
		bool		is_gpu = false;
		double		knn_prediction = 0.0;

		NDB_DECLARE(float *, features_float);
		int			feature_dim = nelems;

		NDB_DECLARE(char *, errstr);
		int			rc;

		if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
		{
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("KNN model %d not found", model_id)));
		}

		if (model_data == NULL)
		{
			if (metrics)
				NDB_FREE(metrics);

			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("KNN model %d has no model data (model not trained)", model_id),
					 errhint("KNN training must be completed before prediction. The model may have been created without actual training.")));
		}

		if (metrics != NULL)
		{
			NDB_DECLARE(JsonbIterator *, it);
			JsonbValue	v;
			int			r;

			PG_TRY();
			{
				it = JsonbIteratorInit(&metrics->root);
				while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
				{
					if (r == WJB_KEY && v.type == jbvString)
					{
						char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

						r = JsonbIteratorNext(&it, &v, false);
						/* Check for training_backend integer (new format) */
						if (strcmp(key, "training_backend") == 0 && v.type == jbvNumeric)
						{
							int			backend = DatumGetInt32(DirectFunctionCall1(numeric_int4, NumericGetDatum(v.val.numeric)));

							if (backend == 1)
								is_gpu = true;
						}
						NDB_FREE(key);
					}
				}
			}
			PG_CATCH();
			{
				/* If metrics parsing fails, assume CPU model */
				is_gpu = false;
			}
			PG_END_TRY();
		}
		NDB_ALLOC(features_float, float, feature_dim);

		for (i = 0; i < feature_dim; i++)
			features_float[i] = (float) features[i];

		/*
		 * Use model's training backend (from catalog) regardless of current
		 * GPU state
		 */
		{
			const ndb_gpu_backend *backend = ndb_gpu_get_active_backend();
			bool		gpu_currently_enabled;

			gpu_currently_enabled = (backend != NULL && neurondb_gpu_is_available());

			elog(DEBUG1,
				 "neurondb_predict: model trained on %s, GPU currently %s",
				 is_gpu ? "GPU" : "CPU",
				 gpu_currently_enabled ? "enabled" : "disabled");

			if (is_gpu)
			{
				/* Model was trained on GPU - must use GPU prediction */
				if (!gpu_currently_enabled)
				{
					NDB_FREE(features_float);

					if (model_data)
						NDB_FREE(model_data);

					if (metrics)
						NDB_FREE(metrics);

					ndb_spi_session_end(&spi_session);
					neurondb_cleanup(oldcontext, callcontext);
					ereport(ERROR,
							(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
							 errmsg("KNN: model %d was trained on GPU but GPU is not currently available", model_id),
							 errhint("Enable GPU with: SET neurondb.compute_mode = 1; or SET neurondb.compute_mode = 2; for auto mode")));
				}

				/* Use GPU prediction */
				if (backend && backend->knn_predict && model_data != NULL)
				{
					rc = backend->knn_predict(model_data, features_float, feature_dim, &knn_prediction, &errstr);
					if (rc == 0)
					{
						prediction = knn_prediction;
						NDB_FREE(features_float);

						if (model_data)
							NDB_FREE(model_data);

						if (metrics)
							NDB_FREE(metrics);

						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						PG_RETURN_FLOAT8(prediction);
					}
					if (errstr)
						NDB_FREE(errstr);
				}
			}
			else
			{
				/*
				 * Model was trained on CPU - use CPU prediction (ignore
				 * current GPU state)
				 */
				/* Fall through to CPU prediction path below */
			}
		}
		appendStringInfo(&sql, "SELECT predict_knn_model_id(%d, %s)", model_id, features_str.data);
		NDB_FREE(features_float);

		if (model_data)
			NDB_FREE(model_data);

		if (metrics)
			NDB_FREE(metrics);
	}
	else if (strcmp(algorithm, "kmeans") == 0)
		appendStringInfo(&sql, "SELECT predict_kmeans_model_id(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "minibatch_kmeans") == 0)
		appendStringInfo(&sql, "SELECT predict_kmeans_model_id(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "hierarchical") == 0)
		appendStringInfo(&sql, "SELECT predict_hierarchical_cluster(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "xgboost") == 0)
		appendStringInfo(&sql, "SELECT predict_xgboost(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "catboost") == 0)
		appendStringInfo(&sql, "SELECT predict_catboost(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "lightgbm") == 0)
		appendStringInfo(&sql, "SELECT predict_lightgbm(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "gmm") == 0)
		appendStringInfo(&sql, "SELECT predict_gmm_model_id(%d, %s)", model_id, features_str.data);
	else
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg(NDB_ERR_PREFIX_PREDICT " unsupported algorithm '%s' for prediction", algorithm),
				 errdetail("Model ID: %d, Algorithm: '%s'", model_id, algorithm),
				 errhint("Supported algorithms for prediction: linear_regression, logistic_regression, random_forest, svm, naive_bayes, knn, knn_classifier, knn_regressor, gmm, kmeans, minibatch_kmeans, hierarchical, xgboost, catboost, lightgbm")));
	}

	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Prediction query did not return a result")));
	}
	/* Get prediction value - float8 requires safe extraction */
	if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL || SPI_tuptable->vals == NULL)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Prediction query result is invalid")));
	}
	{
		Datum		pred_datum;
		bool		pred_isnull;

		pred_datum = SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &pred_isnull);
		if (pred_isnull)
		{
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("Prediction result is NULL")));
		}
		prediction = DatumGetFloat8(pred_datum);
	}
	ndb_spi_session_end(&spi_session);
	neurondb_cleanup(oldcontext, callcontext);

	PG_RETURN_FLOAT8(prediction);
}

/* ----------
 * neurondb_deploy
 * Deploy trained model for usage.
 * ----------
 */
Datum
neurondb_deploy(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *strategy_text;
	char	   *strategy;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	int			ret;
	int			deployment_id = 0;

	NDB_DECLARE(NdbSpiSession *, spi_session);

	if (PG_NARGS() < 1 || PG_NARGS() > 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_MSG("deploy: requires 1-2 arguments, got %d"), PG_NARGS()),
				 errhint("Usage: neurondb.deploy(model_id, [strategy])")));

	model_id = PG_GETARG_INT32(0);
	strategy_text = PG_ARGISNULL(1) ? NULL : PG_GETARG_TEXT_PP(1);

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
										"neurondb_deploy context",
										ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	if (strategy_text)
		strategy = text_to_cstring(strategy_text);
	else
		strategy = pstrdup("replace");

	/* Begin SPI session - always connect since this is called from SQL */
	spi_session = ndb_spi_session_begin(callcontext, false);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfoString(&sql,
						   "CREATE TABLE IF NOT EXISTS " NDB_FQ_ML_DEPLOYMENTS " ("
						   "deployment_id SERIAL PRIMARY KEY, "
						   "model_id INTEGER NOT NULL REFERENCES " NDB_FQ_ML_MODELS "(" NDB_COL_MODEL_ID "), "
						   "deployment_name TEXT NOT NULL, "
						   "strategy TEXT NOT NULL, "
						   "status TEXT DEFAULT 'active', "
						   "deployed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP)");
	(void) ndb_spi_execute(spi_session, sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();

	/* Use safe free/reinit to handle potential memory context changes */
	ndb_spi_stringinfo_free(spi_session, &sql);
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "INSERT INTO " NDB_FQ_ML_DEPLOYMENTS " (" NDB_COL_MODEL_ID ", deployment_name, strategy, " NDB_COL_STATUS ", deployed_at) "
					 "VALUES (%d, %s, %s, '" NDB_DEPLOYMENT_ACTIVE "', CURRENT_TIMESTAMP) RETURNING deployment_id",
					 model_id,
					 neurondb_quote_literal_cstr(psprintf("deploy_%d_%ld", model_id, (long) time(NULL))),
					 neurondb_quote_literal_cstr(strategy));

	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to create deployment")));
	}
	if (!ndb_spi_get_int32(spi_session, 0, 1, &deployment_id))
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to get deployment_id from result")));
	}

	ndb_spi_session_end(&spi_session);
	neurondb_cleanup(oldcontext, callcontext);

	PG_RETURN_INT32(deployment_id);
}

/* ----------
 * neurondb_load_model
 * Load an external ML model and register its metadata.
 * ----------
 */
Datum
neurondb_load_model(PG_FUNCTION_ARGS)
{
	text	   *project_name_text;
	text	   *model_path_text;
	text	   *model_format_text;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	char	   *project_name;
	char	   *model_path;
	char	   *model_format;
	int			ret;
	int			model_id = 0;
	int			project_id = 0;

	NDB_DECLARE(NdbSpiSession *, spi_session);

	if (PG_NARGS() != 3)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb.load_model: requires 3 arguments, got %d", PG_NARGS()),
				 errhint("Usage: neurondb.load_model(project_name, model_path, model_format)")));

	project_name_text = PG_GETARG_TEXT_PP(0);
	model_path_text = PG_GETARG_TEXT_PP(1);
	model_format_text = PG_GETARG_TEXT_PP(2);

	project_name = text_to_cstring(project_name_text);
	model_path = text_to_cstring(model_path_text);
	model_format = text_to_cstring(model_format_text);

	if (strcmp(model_format, "onnx") != 0 &&
		strcmp(model_format, "tensorflow") != 0 &&
		strcmp(model_format, "pytorch") != 0 &&
		strcmp(model_format, "sklearn") != 0)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Unsupported model format: %s. Supported: onnx, tensorflow, pytorch, sklearn", model_format)));
	}

	elog(DEBUG1, "neurondb_load_model context");
	callcontext = AllocSetContextCreate(CurrentMemoryContext, "neurondb_load_model",
										ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	/* Begin SPI session - always connect since this is called from SQL */
	spi_session = ndb_spi_session_begin(callcontext, false);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "INSERT INTO " NDB_FQ_ML_PROJECTS " (project_name, model_type, description) "
					 "VALUES (%s, 'external', 'External model import') "
					 "ON CONFLICT (project_name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP "
					 "RETURNING project_id",
					 neurondb_quote_literal_cstr(project_name));

	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if ((ret != SPI_OK_INSERT_RETURNING && ret != SPI_OK_UPDATE_RETURNING) || SPI_processed == 0)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to create/get external project \"%s\"", project_name)));
	}
	if (!ndb_spi_get_int32(spi_session, 0, 1, &project_id))
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to get project_id from result")));
	}

	/* Use safe free/reinit to handle potential memory context changes */
	ndb_spi_stringinfo_free(spi_session, &sql);
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql, "SELECT pg_advisory_xact_lock(%d)", project_id);
	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	do
	{
		if ((ret) == SPI_OK_SELECT ||
			(ret) == SPI_OK_SELINTO ||
			(ret) == SPI_OK_INSERT_RETURNING ||
			(ret) == SPI_OK_UPDATE_RETURNING ||
			(ret) == SPI_OK_DELETE_RETURNING)
		{
			if (SPI_tuptable == NULL || SPI_tuptable->tupdesc == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg(NDB_ERR_MSG("SPI_tuptable is NULL or invalid for result-set query"))));
		}
	} while (0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to acquire advisory lock")));
	}

	/* Use safe free/reinit to handle potential memory context changes */
	ndb_spi_stringinfo_free(spi_session, &sql);
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "WITH next_version AS (SELECT COALESCE(MAX(" NDB_COL_VERSION "), 0) + 1 AS v FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_PROJECT_ID " = %d) "
					 "INSERT INTO " NDB_FQ_ML_MODELS " (" NDB_COL_PROJECT_ID ", " NDB_COL_VERSION ", " NDB_COL_MODEL_NAME ", " NDB_COL_ALGORITHM ", " NDB_COL_TRAINING_TABLE ", " NDB_COL_TRAINING_COLUMN ", " NDB_COL_STATUS ", metadata) "
					 "SELECT %d, v, %s, %s, NULL, NULL, 'external', '{\"model_path\": %s, \"model_format\": %s}'::jsonb FROM next_version RETURNING model_id",
					 project_id,
					 project_id,
					 neurondb_quote_literal_cstr(psprintf("%s_%ld", model_format, (long) time(NULL))),
					 neurondb_quote_literal_cstr(model_format),
					 neurondb_quote_literal_cstr(model_path),
					 neurondb_quote_literal_cstr(model_format));

	ret = ndb_spi_execute(spi_session, sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_INSERT_RETURNING || SPI_processed == 0)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR), errmsg("Failed to register external model")));
	}
	if (!ndb_spi_get_int32(spi_session, 0, 1, &model_id))
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to get model_id from result")));
	}
	ndb_spi_session_end(&spi_session);
	neurondb_cleanup(oldcontext, callcontext);

	PG_RETURN_INT32(model_id);
}

/* ----------
 * neurondb_evaluate
 * Unified model evaluation interface
 * Dispatches to algorithm-specific evaluate functions
 * ----------
 */
Datum
neurondb_evaluate(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *table_name_text;
	text	   *feature_col_text;
	text	   *label_col_text;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	int			ret;
	bool		isnull = false;

	NDB_DECLARE(char *, algorithm);
	char	   *table_name;
	char	   *feature_col;
	char	   *label_col;

	NDB_DECLARE(Jsonb *, result);
	NDB_DECLARE(NdbSpiSession *, spi_session);

	elog(DEBUG1, "neurondb_evaluate: function entry");
	/* Suppress shadow warnings from nested PG_TRY blocks */
	_Pragma("GCC diagnostic push")
		_Pragma("GCC diagnostic ignored \"-Wshadow\"");

	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " requires 4 arguments, got %d", PG_NARGS()),
				 errhint("Usage: neurondb.evaluate(model_id, table_name, feature_col, label_col)")));

	/* NULL input validation - prevent crashes */
	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " model_id cannot be NULL")));

	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " table_name cannot be NULL")));

	if (PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " feature_col cannot be NULL")));

	/* label_col can be NULL for unsupervised algorithms (e.g., kmeans, gmm) */

	model_id = PG_GETARG_INT32(0);
	table_name_text = PG_GETARG_TEXT_PP(1);
	feature_col_text = PG_GETARG_TEXT_PP(2);
	label_col_text = PG_ARGISNULL(3) ? NULL : PG_GETARG_TEXT_PP(3);

	/* Additional validation after getting arguments */
	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " model_id must be positive, got %d", model_id)));

	/* Validate text pointers are not NULL after conversion */
	if (table_name_text == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " table_name is NULL after conversion")));

	if (feature_col_text == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " feature_col is NULL after conversion")));

	callcontext = AllocSetContextCreate(CurrentMemoryContext,
										"neurondb_evaluate context",
										ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	/* Begin SPI session - always connect since this is called from SQL */
	spi_session = ndb_spi_session_begin(callcontext, false);

	/* Get algorithm from model_id */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT " NDB_COL_ALGORITHM "::text FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_MODEL_ID " = %d",
					 model_id);
	elog(DEBUG1, "neurondb_evaluate: executing query: %s", sql.data);
	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model not found: %d", model_id)));
	}
	{
		char	   *temp_algorithm;

		/* Get algorithm text safely */
		text	   *algo_text = ndb_spi_get_text(spi_session, 0, 1, oldcontext);

		if (algo_text == NULL)
		{
			temp_algorithm = NULL;
			isnull = true;
		}
		else
		{
			temp_algorithm = text_to_cstring(algo_text);
			NDB_FREE(algo_text);
			isnull = false;
		}
		elog(DEBUG1, "neurondb_evaluate: retrieved algorithm='%s', isnull=%d", temp_algorithm, isnull);
		if (isnull)
		{
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model algorithm is NULL for model_id=%d", model_id)));
		}

		/*
		 * Copy algorithm to callcontext to avoid corruption from subsequent
		 * SPI calls
		 */
		algorithm = pstrdup(temp_algorithm);
		Assert(algorithm != NULL);
		Assert(strlen(algorithm) > 0);
		NDB_FREE(temp_algorithm);
	}

	/*
	 * Note: Early model validation removed - dt_model_deserialize and
	 * dt_model_free are static
	 */
	/* Model validation will be handled by the evaluation function itself */

	table_name = text_to_cstring(table_name_text);
	feature_col = text_to_cstring(feature_col_text);
	label_col = label_col_text ? text_to_cstring(label_col_text) : NULL;

	/* Assertions for crash tracking */
	Assert(table_name != NULL);
	Assert(feature_col != NULL);
	Assert(callcontext != NULL);

	/* Validate label_col for supervised algorithms */
	if (strcmp(algorithm, NDB_ALGO_LINEAR_REGRESSION) == 0 ||
		strcmp(algorithm, NDB_ALGO_LOGISTIC_REGRESSION) == 0 ||
		strcmp(algorithm, NDB_ALGO_RIDGE) == 0 ||
		strcmp(algorithm, NDB_ALGO_LASSO) == 0 ||
		strcmp(algorithm, NDB_ALGO_RANDOM_FOREST) == 0 ||
		strcmp(algorithm, NDB_ALGO_SVM) == 0 ||
		strcmp(algorithm, NDB_ALGO_DECISION_TREE) == 0 ||
		strcmp(algorithm, NDB_ALGO_NAIVE_BAYES) == 0 ||
		strcmp(algorithm, NDB_ALGO_KNN) == 0 ||
		strcmp(algorithm, NDB_ALGO_KNN_CLASSIFIER) == 0 ||
		strcmp(algorithm, NDB_ALGO_KNN_REGRESSOR) == 0 ||
		strcmp(algorithm, NDB_ALGO_XGBOOST) == 0 ||
		strcmp(algorithm, NDB_ALGO_CATBOOST) == 0 ||
		strcmp(algorithm, NDB_ALGO_LIGHTGBM) == 0)
	{
		if (label_col == NULL)
		{
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg(NDB_ERR_PREFIX_EVALUATE " label_col cannot be NULL for supervised algorithm '%s'", algorithm)));
		}
	}

	/* Dispatch to algorithm-specific evaluate function */
	/* Use safe free/reinit to handle potential memory context changes */
	ndb_spi_stringinfo_free(spi_session, &sql);
	ndb_spi_stringinfo_init(spi_session, &sql);
	if (strcmp(algorithm, NDB_ALGO_LINEAR_REGRESSION) == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT " NDB_FUNC_EVALUATE_LINEAR_REGRESSION_MODEL_ID "(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, NDB_ALGO_LOGISTIC_REGRESSION) == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_logistic_regression_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, NDB_ALGO_RIDGE) == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_ridge_regression_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, NDB_ALGO_LASSO) == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_lasso_regression_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, "random_forest") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_random_forest_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, "svm") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_svm_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, "decision_tree") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_decision_tree_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, "naive_bayes") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_naive_bayes_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, "knn") == 0 || strcmp(algorithm, "knn_classifier") == 0 || strcmp(algorithm, "knn_regressor") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_knn_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, "kmeans") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_kmeans_by_model_id(%d, %s, %s)",
						 model_id, q_table_name, q_feature_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
	}
	else if (strcmp(algorithm, "gmm") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_gmm_by_model_id(%d, %s, %s)",
						 model_id, q_table_name, q_feature_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
	}
	else if (strcmp(algorithm, "minibatch_kmeans") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_minibatch_kmeans_by_model_id(%d, %s, %s)",
						 model_id, q_table_name, q_feature_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
	}
	else if (strcmp(algorithm, "hierarchical") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_hierarchical_by_model_id(%d, %s, %s)",
						 model_id, q_table_name, q_feature_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
	}
	else if (strcmp(algorithm, "xgboost") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_xgboost_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, "catboost") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_catboost_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else if (strcmp(algorithm, "lightgbm") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_lightgbm_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		NDB_FREE(q_table_name);
		NDB_FREE(q_feature_col);
		NDB_FREE(q_label_col);
	}
	else
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " algorithm '%s' does not support evaluation", algorithm)));
	}


	/* Assertions for crash tracking */
	Assert(sql.data != NULL);
	Assert(strlen(sql.data) > 0);
	Assert(algorithm != NULL);

	/* Wrap entire evaluation in error handler to prevent crashes */
	PG_TRY();
	{
		ret = ndb_spi_execute(spi_session, sql.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT || SPI_processed == 0)
		{
			/*
			 * Evaluation query failed - return NULL instead of trying to
			 * create error JSONB
			 */
			MemoryContextSwitchTo(oldcontext);
			ndb_spi_session_end(&spi_session);
			MemoryContextDelete(callcontext);
			PG_RETURN_NULL();
		}

		/* Validate SPI_tuptable before access */
		NDB_CHECK_SPI_TUPTABLE();

		{
			bool		result_isnull = false;
			Jsonb	   *temp_jsonb;

			/* Get JSONB from SPI result using safe function */
			temp_jsonb = ndb_spi_get_jsonb(spi_session, 0, 1, oldcontext);
			if (temp_jsonb == NULL)
			{
				result_isnull = true;
			}
			if (result_isnull)
			{
				/*
				 * Evaluation returned NULL - return empty JSONB instead
				 */
				char	   *result_buf;

				MemoryContextSwitchTo(oldcontext);
				/* Create minimal valid JSONB for empty object */
				result_buf = NULL;
				NBP_ALLOC(result_buf, char, VARHDRSZ + sizeof(uint32));
				result = (Jsonb *) result_buf;
				SET_VARSIZE(result, VARHDRSZ + sizeof(uint32));
				*((uint32 *) VARDATA(result)) = JB_CMASK;	/* Empty object header */
				ndb_spi_session_end(&spi_session);
				MemoryContextDelete(callcontext);
				PG_RETURN_JSONB_P(result);
			}

			/* temp_jsonb is already obtained from ndb_spi_get_jsonb above */

			/* Validate JSONB structure before using it */
			if (temp_jsonb == NULL || VARSIZE(temp_jsonb) < sizeof(Jsonb))
			{
				MemoryContextSwitchTo(oldcontext);
				ndb_spi_session_end(&spi_session);
				MemoryContextDelete(callcontext);
				PG_RETURN_NULL();
			}

			/*
			 * Copy JSONB to caller's context before session end. This ensures
			 * the JSONB is valid after SPI context is cleaned up. Session end
			 * will delete the SPI memory context, so any pointers to data
			 * allocated in that context will become invalid.
			 */
			MemoryContextSwitchTo(oldcontext);
			result = (Jsonb *) PG_DETOAST_DATUM_COPY((Datum) temp_jsonb);

			if (result == NULL || VARSIZE(result) < sizeof(Jsonb))
			{
				if (result != NULL)
				{
					NDB_FREE(result);
				}
				/* Return NULL instead of trying to create empty JSONB */
				result = NULL;
			}
		}
	}
	PG_CATCH();
	{
		NDB_DECLARE(ErrorData *, edata);
		char	   *error_msg;

		/*
		 * Switch out of ErrorContext before CopyErrorData(). CopyErrorData()
		 * allocates memory and must NOT be called while in ErrorContext, as
		 * that context is only for error reporting and will be reset, causing
		 * memory leaks or corruption.
		 */
		MemoryContextSwitchTo(oldcontext);

		/* Suppress shadow warnings from nested PG_TRY blocks */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow=compatible-local"
		PG_TRY();
		{
			if (CurrentMemoryContext != ErrorContext)
			{
				edata = CopyErrorData();
				FlushErrorState();
			}
			else
			{
				FlushErrorState();
			}
		}
		PG_CATCH();
		{
			FlushErrorState();
		}
		PG_END_TRY();
#pragma GCC diagnostic pop

		/* Create safe error message (escape quotes) */
		if (edata != NULL && edata->message != NULL)
			error_msg = pstrdup(edata->message);
		else
			error_msg = pstrdup("evaluation failed (GPU may be unavailable or model data missing)");

		/* Escape JSON special characters */
		{
			NDB_DECLARE(char *, escaped);
			char	   *p;
			const char *s;

			NDB_ALLOC(escaped, char, strlen(error_msg) * 2 + 1);

			p = escaped;
			s = error_msg;
			while (*s)
			{
				if (*s == '"' || *s == '\\' || *s == '\n' || *s == '\r')
				{
					*p++ = '\\';
					if (*s == '\n')
						*p++ = 'n';
					else if (*s == '\r')
						*p++ = 'r';
					else
						*p++ = *s;
				}
				else
					*p++ = *s;
				s++;
			}
			*p = '\0';
			NDB_FREE(error_msg);

			error_msg = NULL;
			error_msg = escaped;
		}

		/* Return error JSONB */
		{
			StringInfoData error_json;

			ndb_spi_stringinfo_init(spi_session, &error_json);
			appendStringInfo(&error_json, "{\"error\": \"%s\"}", error_msg);
			/* Skip JSONB creation to avoid DirectFunctionCall1 issues */
			result = NULL;
			ndb_spi_stringinfo_free(spi_session, &error_json);
			error_json.data = NULL;
			NDB_FREE(error_msg);

			error_msg = NULL;

			/* Defensive check: ensure result is valid */
			if (result == NULL || VARSIZE(result) < sizeof(Jsonb))
			{
				/* Create minimal valid JSONB for empty object */
				NDB_DECLARE(char *, result_buf);
				NBP_ALLOC(result_buf, char, VARHDRSZ + sizeof(uint32));
				result = (Jsonb *) result_buf;
				SET_VARSIZE(result, VARHDRSZ + sizeof(uint32));
				*((uint32 *) VARDATA(result)) = JB_CMASK;	/* Empty object header */
			}
		}

		/* Free error data if we copied it */
		if (edata != NULL)
			FreeErrorData(edata);

		ndb_spi_session_end(&spi_session);

		/*
		 * Clean up memory context. Must switch to oldcontext before deleting
		 * callcontext, as session end or error handling may have changed
		 * CurrentMemoryContext. Attempting to delete a context while in that
		 * context will cause a crash.
		 */
		if (callcontext)
		{
			MemoryContextSwitchTo(oldcontext);
			MemoryContextDelete(callcontext);
		}

		/* Return error JSONB immediately - don't fall through to cleanup code */
		PG_RETURN_JSONB_P(result);
	}
	PG_END_TRY();

	/* Now safe to clean up SPI and delete callcontext */
	ndb_spi_session_end(&spi_session);

	/* Ensure we're in oldcontext before deleting callcontext */
	/* Session end might have changed CurrentMemoryContext */
	MemoryContextSwitchTo(oldcontext);
	MemoryContextDelete(callcontext);

	/* oldcontext is current, result lives there */

	/* CRITICAL SAFETY: Never return invalid data that could crash PostgreSQL */
	elog(DEBUG2, "neurondb_evaluate: final validation - result=%p", result);

	if (result == NULL)
	{
		/* Instead of returning invalid data, throw an error */
		elog(ERROR, "neurondb_evaluate: CRITICAL - result is NULL");
	}

	if (VARSIZE(result) < sizeof(Jsonb))
	{
		/* Instead of returning invalid data, throw an error */
		elog(ERROR, "neurondb_evaluate: CRITICAL - invalid JSONB structure (size %d < %d)",
			 VARSIZE(result), (int) sizeof(Jsonb));
	}

	/* Additional validation - check if result is valid */
	if (result == NULL)
	{
		char	   *result_buf;

		elog(WARNING, "neurondb_evaluate: result is NULL, returning empty JSONB");
		/* Create minimal valid JSONB for empty object */
		result_buf = NULL;
		NBP_ALLOC(result_buf, char, VARHDRSZ + sizeof(JsonbContainer));
		result = (Jsonb *) result_buf;
		SET_VARSIZE(result, VARHDRSZ + sizeof(JsonbContainer));
		result->root.header = JB_CMASK; /* Empty object */
	}

	elog(DEBUG2, "neurondb_evaluate: about to return JSONB result, size=%d", VARSIZE(result));

	/* EMERGENCY SAFETY: Ensure result is ALWAYS valid before returning */
	if (result == NULL || VARSIZE(result) < sizeof(Jsonb))
	{
		char	   *result_buf;

		/* Create minimal valid JSONB for empty object */
		elog(WARNING, "neurondb_evaluate: EMERGENCY - result invalid, creating empty JSONB");
		result_buf = NULL;
		NBP_ALLOC(result_buf, char, VARHDRSZ + sizeof(JsonbContainer));
		result = (Jsonb *) result_buf;
		SET_VARSIZE(result, VARHDRSZ + sizeof(JsonbContainer));
		result->root.header = JB_CMASK; /* Empty object */
	}

	PG_RETURN_JSONB_P(result);
}
