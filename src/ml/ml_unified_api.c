/*-------------------------------------------------------------------------
 *
 * ml_unified_api.c
 *    Unified SQL API for machine learning operations.
 *
 * This module provides a simplified, unified interface for machine learning
 * operations in NeuronDB. It abstracts the complexity of algorithm-specific
 * implementations behind a consistent SQL API, supporting both GPU and CPU
 * execution paths with automatic fallback.
 *
 * Main Functions:
 *    - neurondb_train: Train ML models from SQL tables
 *    - neurondb_predict: Generate predictions using trained models
 *    - neurondb_deploy: Deploy models for production use
 *    - neurondb_load_model: Load external models (ONNX, TensorFlow, etc.)
 *    - neurondb_evaluate: Evaluate model performance on test data
 *
 * Key Features:
 *    - Automatic GPU/CPU fallback based on availability and configuration
 *    - Support for supervised and unsupervised algorithms
 *    - Hyperparameter management via JSONB
 *    - Model versioning and catalog management
 *    - Memory-safe operations with proper context management
 *
 * Memory Management:
 *    All functions use dedicated memory contexts to prevent leaks. Memory
 *    allocated in call contexts must be freed before context deletion.
 *    SPI sessions are managed separately and must be properly ended.
 *
 * Thread Safety:
 *    Functions are called from PostgreSQL backend processes. Each function
 *    call operates in its own memory context. SPI operations are session-local.
 *
 * CHANGE NOTES:
 *    - CAN MODIFY: Algorithm support, hyperparameter defaults, error messages
 *    - CANNOT MODIFY: Memory context lifecycle, SPI session management order
 *    - BREAKS IF: Memory contexts deleted while in use, SPI sessions used after end
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
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
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
#include "neurondb_cuda_lr.h"
#include "ml_logistic_regression_internal.h"
#include "libpq/pqformat.h"
#endif

PG_FUNCTION_INFO_V1(neurondb_train);
PG_FUNCTION_INFO_V1(neurondb_predict);
PG_FUNCTION_INFO_V1(neurondb_deploy);
PG_FUNCTION_INFO_V1(neurondb_load_model);
PG_FUNCTION_INFO_V1(neurondb_evaluate);

/*
 * MLAlgorithm
 *		Type-safe enumeration of supported machine learning algorithms.
 *
 * This enum provides compile-time type safety for algorithm identification
 * throughout the unified API. Algorithms are grouped by learning type:
 *
 * Classification Algorithms:
 *		- ML_ALGO_LOGISTIC_REGRESSION: Binary/multi-class classification
 *		- ML_ALGO_RANDOM_FOREST: Ensemble tree-based classifier
 *		- ML_ALGO_SVM: Support Vector Machine classifier
 *		- ML_ALGO_DECISION_TREE: Single decision tree classifier
 *		- ML_ALGO_NAIVE_BAYES: Probabilistic classifier
 *		- ML_ALGO_XGBOOST: Gradient boosting classifier
 *		- ML_ALGO_CATBOOST: Categorical boosting classifier
 *		- ML_ALGO_LIGHTGBM: Light gradient boosting classifier
 *		- ML_ALGO_KNN: K-nearest neighbors (classification mode)
 *		- ML_ALGO_KNN_CLASSIFIER: Explicit KNN classifier
 *
 * Regression Algorithms:
 *		- ML_ALGO_LINEAR_REGRESSION: Linear regression
 *		- ML_ALGO_RIDGE: Ridge regression (L2 regularization)
 *		- ML_ALGO_LASSO: Lasso regression (L1 regularization)
 *		- ML_ALGO_KNN_REGRESSOR: K-nearest neighbors (regression mode)
 *
 * Clustering Algorithms:
 *		- ML_ALGO_KMEANS: K-means clustering
 *		- ML_ALGO_GMM: Gaussian Mixture Model clustering
 *		- ML_ALGO_MINIBATCH_KMEANS: Mini-batch K-means (scalable)
 *		- ML_ALGO_HIERARCHICAL: Hierarchical clustering
 *		- ML_ALGO_DBSCAN: Density-based clustering
 *
 * Dimensionality Reduction:
 *		- ML_ALGO_PCA: Principal Component Analysis
 *		- ML_ALGO_OPQ: Optimized Product Quantization
 *
 * Time Series:
 *		- ML_ALGO_TIMESERIES: Time series forecasting (ARIMA)
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Add new algorithm types (must update string mapping)
 *		- CANNOT MODIFY: ML_ALGO_UNKNOWN = 0 (used as sentinel value)
 *		- BREAKS IF: Enum values reordered without updating string mappings
 */
typedef enum
{
	ML_ALGO_UNKNOWN = 0,		/* Sentinel value for invalid/unknown algorithms */
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
	ML_ALGO_TIMESERIES,
	/* Language Models */
	ML_ALGO_TRANSFORMER_LLM,
	ML_ALGO_CUSTOM_LLM,
	ML_ALGO_TITANS_LLM
}			MLAlgorithm;

/* Forward declarations */
/*
 * Memory and resource management
 */
static void neurondb_cleanup(MemoryContext oldcontext, MemoryContext callcontext);

/*
 * SQL string utilities
 */
static char *neurondb_quote_literal_cstr(const char *str);
static char *neurondb_quote_literal_or_null(const char *str);

/*
 * Algorithm identification and validation
 */
static MLAlgorithm neurondb_algorithm_from_string(const char *algorithm);
static bool neurondb_is_unsupervised_algorithm(const char *algorithm);
/* Removed unused function neurondb_get_model_type */
static ErrorData *neurondb_safe_copy_error_data(void);

/*
 * neurondb_safe_copy_error_data
 *		Safely copy current error data in PostgreSQL error handling context.
 */
static ErrorData *
neurondb_safe_copy_error_data(void)
{
	return CopyErrorData();
}


/*
 * Training data and feature management
 */
static bool neurondb_load_training_data(NdbSpiSession *session,
										const char *table_name,
										const char *feature_list_str,
										const char *target_column,
										float **feature_matrix_out,
										double **label_vector_out,
										int *n_samples_out,
										int *feature_dim_out,
										int *class_count_out);
static int	neurondb_prepare_feature_list(ArrayType * feature_columns_array, StringInfo feature_list, const char ***feature_names_out, int *feature_name_count_out);

/*
 * Hyperparameter parsing
 */
static void neurondb_parse_hyperparams_int(Jsonb * hyperparams, const char *key, int *value, int default_value);
static void neurondb_parse_hyperparams_float8(Jsonb * hyperparams, const char *key, double *value, double default_value);

/*
 * Training SQL generation
 */
static bool neurondb_build_training_sql(MLAlgorithm algo, StringInfo sql, const char *table_name, const char *feature_list, const char *target_column, Jsonb * hyperparams, const char **feature_names, int feature_name_count);

/*
 * Input validation
 */
static void neurondb_validate_training_inputs(const char *project_name, const char *algorithm, const char *table_name, const char *target_column);

/*
 * Model metadata utilities
 */
static bool ml_metrics_is_gpu(Jsonb * metrics);


/*
 * neurondb_cleanup
 *		Restore memory context and delete call context.
 *
 * This function performs safe cleanup of memory contexts used during function
 * execution. It switches back to the original memory context and deletes the
 * call-specific context, freeing all memory allocated within it.
 *
 * Parameters:
 *		oldcontext - The memory context to restore (typically the caller's context)
 *		callcontext - The call-specific context to delete (may be NULL)
 *
 * Side Effects:
 *		- Switches CurrentMemoryContext back to oldcontext
 *		- Deletes callcontext and all memory allocated within it
 *		- May raise ERROR if context operations fail
 *
 * Error Handling:
 *		- Raises ERROR if both oldcontext and CurrentMemoryContext are NULL
 *		- Raises ERROR if MemoryContextSwitchTo fails unexpectedly
 *		- Logs WARNING if callcontext is invalid (but continues)
 *
 * Thread Safety:
 *		Memory context operations are thread-local. This function is safe to call
 *		from any PostgreSQL backend process.
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Warning/error message text, defensive NULL checks
 *		- CANNOT MODIFY: Order of context switch before deletion (must switch first)
 *		- BREAKS IF: Deletes callcontext before switching away from it
 *		- BREAKS IF: Attempts to use callcontext after deletion
 */
static void
neurondb_cleanup(MemoryContext oldcontext, MemoryContext callcontext)
{
	MemoryContext current_context;

	ereport(DEBUG2,
			(errmsg("neurondb_cleanup: starting cleanup"),
			 errdetail("oldcontext=%p, callcontext=%p", (void *) oldcontext, (void *) callcontext)));

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

	/* CRITICAL: Must switch away from callcontext before deleting it */
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
			ereport(DEBUG2,
					(errmsg("neurondb_cleanup: deleting callcontext")));
			MemoryContextDelete(callcontext);
		}
		else
		{
			elog(WARNING, "neurondb_cleanup: callcontext is not valid, skipping deletion");
		}
	}

	ereport(DEBUG2,
			(errmsg("neurondb_cleanup: cleanup completed successfully")));
}

/*
 * neurondb_load_training_data
 *		Load training data from SQL table into feature matrix and label vector.
 *
 * This function executes a SELECT query to retrieve training data from a table,
 * converts it into dense float arrays suitable for ML algorithms, and determines
 * the number of classes for classification tasks.
 *
 * Parameters:
 *		session - Active SPI session for executing queries (must be valid)
 *		table_name - Name of the table containing training data
 *		feature_list_str - SQL expression for feature columns (e.g., "col1, col2" or "*")
 *		target_column - Name of target/label column (NULL for unsupervised algorithms)
 *		feature_matrix_out - Output: allocated feature matrix (n_samples × feature_dim)
 *		label_vector_out - Output: allocated label vector (n_samples) or NULL
 *		n_samples_out - Output: number of valid training samples
 *		feature_dim_out - Output: number of features per sample
 *		class_count_out - Output: number of unique classes (for classification)
 *
 * Returns:
 *		true on success, false on failure. On failure, output pointers are set to NULL.
 *
 * Memory Management:
 *		- Allocates feature_matrix using nalloc (caller must free with nfree)
 *		- Allocates label_vector using nalloc if target_column is provided
 *		- Memory is allocated in the current memory context
 *		- On failure, allocated memory is freed before returning
 *
 * Data Format Support:
 *		- PostgreSQL arrays (float4[], float8[])
 *		- Vector type (neurondb vector extension)
 *		- Scalar float4/float8 (treated as 1D feature)
 *
 * Limitations:
 *		- Maximum 200,000 samples per call (hardcoded limit)
 *		- NULL values in features are skipped (row excluded)
 *		- All feature vectors must have same dimension
 *		- Memory allocation checked against MaxAllocSize
 *
 * Side Effects:
 *		- Executes SPI query (must be in valid SPI context)
 *		- Allocates memory for feature_matrix and label_vector
 *		- May raise ERROR on invalid input or memory limits
 *
 * Error Handling:
 *		- Raises ERROR if table doesn't exist or is inaccessible
 *		- Raises ERROR if no data found in table
 *		- Raises ERROR if memory allocation exceeds MaxAllocSize
 *		- Returns false if feature dimension mismatch detected
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: max_samples_limit (200000), supported data types
 *		- CANNOT MODIFY: Memory allocation order (must allocate after dimension check)
 *		- CANNOT MODIFY: SPI session usage (must use provided session)
 *		- BREAKS IF: Memory freed before caller uses it
 *		- BREAKS IF: SPI session used after it's been ended
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

	float *feature_matrix = NULL;
	double *label_vector = NULL;
	TupleDesc	tupdesc;
	HeapTuple	tuple;
	bool		isnull;
	Datum		feat_datum;

	ArrayType *feat_arr = NULL;
	int			i,
				j;
	Oid			feature_type;


	ereport(DEBUG1,
			(errmsg("neurondb_load_training_data: starting data load"),
			 errdetail("table_name=%s, feature_list=%s, target_column=%s",
					   table_name ? table_name : "NULL",
					   feature_list_str ? feature_list_str : "NULL",
					   target_column ? target_column : "NULL")));

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
		int			max_samples_limit = neurondb_ml_max_samples;
		char	   *target_copy = NULL;
		const char *target_quoted_const;
		const char *table_quoted;
		StringInfoData quoted_features;
		char	   *fcopy = NULL;
		char	   *start = NULL;
		char	   *p = NULL;

		ndb_spi_stringinfo_init(session, &sql);

		/* Quote each comma-separated identifier in feature_list_str */
		initStringInfo(&quoted_features);
		fcopy = pstrdup(feature_list_str);
		for (start = fcopy; *start; start = p ? p + 1 : start + strlen(start))
		{
			p = strchr(start, ',');
			if (p)
				*p = '\0';
			while (*start == ' ' || *start == '\t')
				start++;
			if (*start)
			{
				if (quoted_features.len > 0)
					appendStringInfoChar(&quoted_features, ',');
				appendStringInfoString(&quoted_features, quote_identifier(start));
			}
			if (!p)
				break;
			*p = ',';
		}
		pfree(fcopy);

		/* Quote table name to prevent SQL injection (after feature list so buffer not overwritten) */
		table_quoted = quote_identifier(table_name);

		if (target_column)
		{
			target_copy = pstrdup(target_column);
			target_quoted_const = quote_identifier(target_copy);
			appendStringInfo(&sql, "SELECT %s, %s FROM %s LIMIT %d",
							 quoted_features.data,
							 target_quoted_const,
							 table_quoted,
							 max_samples_limit);
			nfree(target_copy);
		}
		else
		{
			appendStringInfo(&sql, "SELECT %s FROM %s LIMIT %d",
							 quoted_features.data,
							 table_quoted,
							 max_samples_limit);
		}
		pfree(quoted_features.data);
	}

	ereport(DEBUG2,
			(errmsg("neurondb_load_training_data: executing query"),
			 errdetail("query=%s", sql.data)));

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
	ereport(DEBUG2,
			(errmsg("neurondb_load_training_data: query returned rows"),
			 errdetail("n_samples=%d", n_samples)));
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

	if (n_samples > neurondb_ml_max_samples)
	{
		ndb_spi_stringinfo_free(session, &sql);
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("neurondb_load_training_data: sample count %d exceeds neurondb.ml_max_samples (%d)", n_samples, neurondb_ml_max_samples),
				 errhint("Increase neurondb.ml_max_samples or reduce the training table size.")));
		return false;
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
	ereport(DEBUG2,
			(errmsg("neurondb_load_training_data: determining feature dimension"),
			 errdetail("feature_type=%u", feature_type)));

	if (feature_type == FLOAT4ARRAYOID || feature_type == FLOAT8ARRAYOID)
	{
		feat_arr = DatumGetArrayTypeP(feat_datum);
		feature_dim = ArrayGetNItems(ARR_NDIM(feat_arr), ARR_DIMS(feat_arr));
		ereport(DEBUG2,
				(errmsg("neurondb_load_training_data: feature type is array"),
				 errdetail("feature_dim=%d", feature_dim)));
	}
	else
	{
		Vector *vec = NULL;

		vec = DatumGetVector(feat_datum);
		if (vec != NULL && vec->dim > 0)
		{
			feature_dim = vec->dim;
			feat_arr = NULL;
			ereport(DEBUG2,
					(errmsg("neurondb_load_training_data: feature type is vector"),
					 errdetail("feature_dim=%d", feature_dim)));
		}
		else
		{
			feature_dim = 1;
			feat_arr = NULL;
			ereport(DEBUG2,
					(errmsg("neurondb_load_training_data: feature type is scalar"),
					 errdetail("feature_dim=1")));
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

	ereport(DEBUG2,
			(errmsg("neurondb_load_training_data: allocating memory"),
			 errdetail("n_samples=%d, feature_dim=%d, matrix_size=%zu bytes",
					   n_samples, feature_dim,
					   (size_t) n_samples * (size_t) feature_dim * sizeof(float))));

	nalloc(feature_matrix, float, (size_t) n_samples * (size_t) feature_dim);

	if (target_column)
	{
		ereport(DEBUG2,
				(errmsg("neurondb_load_training_data: allocating label vector"),
				 errdetail("label_vector_size=%zu bytes",
						   (size_t) n_samples * sizeof(double))));
		nalloc(label_vector, double, (size_t) n_samples);
	}

	valid_samples = 0;
	ereport(DEBUG2,
			(errmsg("neurondb_load_training_data: starting data extraction loop")));
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

		/* Bounds check: never write past allocated feature_matrix (n_samples * feature_dim) */
		if (valid_samples >= n_samples)
		{
			nfree(feature_matrix);
			nfree(label_vector);
			ndb_spi_stringinfo_free(session, &sql);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb_load_training_data: valid_samples exceeded n_samples")));
			return false;
		}

		if (feat_arr)
		{
			ArrayType *curr_arr = NULL;
			int			arr_len;

			curr_arr = DatumGetArrayTypeP(featval);
			if (ARR_NDIM(curr_arr) == 1)
			{
				arr_len = ArrayGetNItems(ARR_NDIM(curr_arr), ARR_DIMS(curr_arr));
				if (arr_len != feature_dim)
				{
					nfree(feature_matrix);

					nfree(label_vector);

					ndb_spi_stringinfo_free(session, &sql);
					return false;
				}
				if (feature_type == FLOAT8ARRAYOID)
				{
					float8 *fdat = NULL;

					fdat = (float8 *) ARR_DATA_PTR(curr_arr);
					for (j = 0; j < feature_dim; j++)
						feature_matrix[valid_samples * feature_dim + j] = (float) fdat[j];
				}
				else
				{
					float4 *fdat = NULL;

					fdat = (float4 *) ARR_DATA_PTR(curr_arr);
					for (j = 0; j < feature_dim; j++)
						feature_matrix[valid_samples * feature_dim + j] = fdat[j];
				}
			}
			else
			{
				nfree(feature_matrix);
				nfree(label_vector);

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
			Vector *vec = NULL;

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
		int *seen_classes = NULL;
		int			max_class = -1;
		int			cls;

		nalloc(seen_classes, int, 256);

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
		nfree(seen_classes);
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

	ereport(DEBUG1,
			(errmsg("neurondb_load_training_data: data load completed successfully"),
			 errdetail("n_samples=%d, feature_dim=%d, class_count=%d, valid_samples=%d",
					   n_samples, feature_dim, class_count, valid_samples)));

	return true;
}

/*
 * neurondb_quote_literal_cstr
 *		Quote a C string for safe use in SQL queries.
 *
 * This function wraps PostgreSQL's quote_literal() function to safely quote
 * strings for use in dynamically generated SQL. The result is properly
 * escaped and quoted according to SQL string literal rules.
 *
 * Parameters:
 *		str - C string to quote (must not be NULL)
 *
 * Returns:
 *		Quoted SQL string literal (e.g., 'value' becomes '''value''')
 *		Memory allocated in CurrentMemoryContext, caller must free with pfree/nfree
 *
 * Side Effects:
 *		- Allocates memory for quoted string
 *		- Allocates temporary text object (freed before return)
 *		- Raises ERROR if str is NULL
 *
 * Memory Management:
 *		- Returned string must be freed by caller using pfree() or nfree()
 *		- Temporary text object is freed internally
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Error message text
 *		- CANNOT MODIFY: Use of PostgreSQL quote_literal() function (SQL safety)
 *		- BREAKS IF: Returned string not freed (memory leak)
 *		- BREAKS IF: NULL string passed (will raise ERROR)
 */
static char *
neurondb_quote_literal_cstr(const char *str)
{
	char *ret = NULL;
	text *txt = NULL;

	if (str == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb_quote_literal_cstr: cannot quote NULL string"),
				 errdetail("Internal function called with NULL string pointer"),
				 errhint("This is an internal error. Please report this issue.")));

	ereport(DEBUG2,
			(errmsg("neurondb_quote_literal_cstr: quoting string"),
			 errdetail("str_length=%zu", strlen(str))));

	txt = cstring_to_text(str);

	ret = TextDatumGetCString(
							  DirectFunctionCall1(quote_literal, PointerGetDatum(txt)));
	nfree(txt);

	return ret;
}

/*
 * neurondb_quote_literal_or_null
 *		Quote a C string for SQL, or return "NULL" if string is NULL.
 *
 * This is a convenience wrapper around neurondb_quote_literal_cstr() that
 * handles NULL strings by returning the SQL literal "NULL" instead of
 * raising an error. Useful for optional parameters in SQL generation.
 *
 * Parameters:
 *		str - C string to quote, or NULL
 *
 * Returns:
 *		- If str is NULL: returns pstrdup("NULL") (caller must free)
 *		- If str is not NULL: returns quoted string from neurondb_quote_literal_cstr()
 *
 * Memory Management:
 *		- Returned string must be freed by caller using pfree() or nfree()
 *		- Always allocates new memory (never returns input pointer)
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: NULL handling (currently returns "NULL" string)
 *		- CANNOT MODIFY: Delegation to neurondb_quote_literal_cstr() for non-NULL
 *		- BREAKS IF: Returned string not freed (memory leak)
 */
static char *
neurondb_quote_literal_or_null(const char *str)
{
	if (str == NULL)
	{
		ereport(DEBUG2,
				(errmsg("neurondb_quote_literal_or_null: input is NULL, returning 'NULL'")));
		return pstrdup("NULL");
	}
	return neurondb_quote_literal_cstr(str);
}

/*
 * neurondb_algorithm_from_string
 *		Convert algorithm name string to MLAlgorithm enum value.
 *
 * This function performs case-sensitive string matching to convert algorithm
 * names (as provided by users in SQL) to the corresponding enum value. This
 * provides type safety and allows compile-time checking of algorithm support.
 *
 * Parameters:
 *		algorithm - Algorithm name string (e.g., "logistic_regression", "kmeans")
 *
 * Returns:
 *		MLAlgorithm enum value corresponding to the algorithm name
 *		Returns ML_ALGO_UNKNOWN if algorithm is NULL or unrecognized
 *
 * Algorithm Name Mapping:
 *		The function matches against constants defined in neurondb_constants.h:
 *		- NDB_ALGO_LOGISTIC_REGRESSION, NDB_ALGO_RANDOM_FOREST, etc.
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Add new algorithm mappings (must add to enum first)
 *		- CANNOT MODIFY: ML_ALGO_UNKNOWN return for NULL/invalid (used as sentinel)
 *		- BREAKS IF: Algorithm name constants changed without updating this function
 *		- BREAKS IF: Enum values reordered without updating string comparisons
 */
static MLAlgorithm
neurondb_algorithm_from_string(const char *algorithm)
{
	if (algorithm == NULL)
	{
		ereport(DEBUG2,
				(errmsg("neurondb_algorithm_from_string: NULL algorithm, returning UNKNOWN")));
		return ML_ALGO_UNKNOWN;
	}

	ereport(DEBUG2,
			(errmsg("neurondb_algorithm_from_string: converting algorithm string"),
			 errdetail("algorithm=%s", algorithm)));

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
	{
		ereport(DEBUG2,
				(errmsg("neurondb_algorithm_from_string: matched TIMESERIES")));
		return ML_ALGO_TIMESERIES;
	}
	if (strcmp(algorithm, NDB_ALGO_TRANSFORMER_LLM) == 0)
	{
		ereport(DEBUG2,
				(errmsg("neurondb_algorithm_from_string: matched TRANSFORMER_LLM")));
		return ML_ALGO_TRANSFORMER_LLM;
	}
	if (strcmp(algorithm, NDB_ALGO_CUSTOM_LLM) == 0)
	{
		ereport(DEBUG2,
				(errmsg("neurondb_algorithm_from_string: matched CUSTOM_LLM")));
		return ML_ALGO_TRANSFORMER_LLM;  /* Use same implementation */
	}
	if (strcmp(algorithm, NDB_ALGO_TITANS_LLM) == 0)
	{
		ereport(DEBUG2,
				(errmsg("neurondb_algorithm_from_string: matched TITANS_LLM")));
		return ML_ALGO_TITANS_LLM;
	}

	ereport(DEBUG2,
			(errmsg("neurondb_algorithm_from_string: unrecognized algorithm"),
			 errdetail("algorithm=%s, returning UNKNOWN", algorithm)));
	return ML_ALGO_UNKNOWN;
}

/*
 * ml_metrics_is_gpu
 *		Check if metrics JSONB indicates GPU training was used.
 *
 * This function examines the metrics JSONB object to determine if the model
 * was trained on GPU by checking for the "training_backend" field. A value
 * of 1 indicates GPU training, 0 indicates CPU training.
 *
 * Parameters:
 *		metrics - JSONB object containing training metrics (may be NULL)
 *
 * Returns:
 *		true if metrics contains "training_backend": 1 (GPU training)
 *		false if metrics is NULL, field missing, or value is not 1
 *
 * JSONB Structure Expected:
 *		{"training_backend": 1, ...}  - GPU training
 *		{"training_backend": 0, ...}  - CPU training
 *		{} or missing field           - Unknown (returns false)
 *
 * Side Effects:
 *		- Iterates through JSONB structure
 *		- Allocates temporary string for key comparison (freed internally)
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Field name checked ("training_backend"), value checked (1)
 *		- CANNOT MODIFY: JSONB iteration pattern (PostgreSQL API)
 *		- BREAKS IF: JSONB structure changes without updating field name
 */
static bool
ml_metrics_is_gpu(Jsonb * metrics)
{
	bool		is_gpu = false;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	JsonbIteratorToken r;

	if (metrics == NULL)
	{
		ereport(DEBUG2,
				(errmsg("ml_metrics_is_gpu: metrics is NULL, returning false")));
		return false;
	}

	ereport(DEBUG2,
			(errmsg("ml_metrics_is_gpu: checking metrics for GPU training")));

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
					ereport(DEBUG2,
							(errmsg("ml_metrics_is_gpu: found training_backend value"),
							 errdetail("backend=%d, is_gpu=%s", backend, is_gpu ? "true" : "false")));
				}
			}
			nfree(key);
		}
	}

	ereport(DEBUG2,
			(errmsg("ml_metrics_is_gpu: final result"),
			 errdetail("is_gpu=%s", is_gpu ? "true" : "false")));
	return is_gpu;
}

/*
 * neurondb_parse_hyperparams_int
 *		Parse integer hyperparameter from JSONB hyperparameters object.
 *
 * This function safely extracts an integer hyperparameter value from a JSONB
 * object. If the key is not found or the JSONB is corrupted, the default value
 * is used. The function handles JSONB parsing errors gracefully.
 *
 * Parameters:
 *		hyperparams - JSONB object containing hyperparameters (may be NULL)
 *		key - Key name to look up in hyperparams object
 *		value - Output parameter: parsed integer value (set to default if not found)
 *		default_value - Default value to use if key not found or parsing fails
 *
 * Side Effects:
 *		- Modifies *value parameter
 *		- May allocate temporary memory for JSONB iteration (freed internally)
 *		- Catches and handles JSONB parsing errors
 *
 * Error Handling:
 *		- If hyperparams, key, or value is NULL: sets *value to default and returns
 *		- If JSONB is corrupted: catches error, uses default, logs DEBUG2 message
 *		- If key not found: *value remains at default_value
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Default value handling, error message text
 *		- CANNOT MODIFY: PG_TRY/PG_CATCH structure (required for error handling)
 *		- BREAKS IF: JSONB structure changes without updating field access
 */
static void
neurondb_parse_hyperparams_int(Jsonb * hyperparams, const char *key, int *value, int default_value)
{
	Jsonb *field_jsonb = NULL;
	JsonbValue	v;
	JsonbIterator *it = NULL;
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
					ereport(DEBUG2,
							(errmsg("neurondb_parse_hyperparams_int: parsed value"),
							 errdetail("key=%s, value=%d", key, *value)));
					break;
				}
			}
		}
		else
		{
			ereport(DEBUG2,
					(errmsg("neurondb_parse_hyperparams_int: key not found, using default"),
					 errdetail("key=%s, default_value=%d", key, default_value)));
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		ereport(DEBUG2,
				(errmsg("neurondb_parse_hyperparams_int: JSONB parsing error, using default"),
				 errdetail("key=%s, default_value=%d", key, default_value)));
		*value = default_value;
	}
	PG_END_TRY();
}

/*
 * neurondb_parse_hyperparams_float8
 *		Parse float8 (double precision) hyperparameter from JSONB hyperparameters object.
 *
 * This function safely extracts a floating-point hyperparameter value from a JSONB
 * object. If the key is not found or the JSONB is corrupted, the default value
 * is used. The function handles JSONB parsing errors gracefully.
 *
 * Parameters:
 *		hyperparams - JSONB object containing hyperparameters (may be NULL)
 *		key - Key name to look up in hyperparams object
 *		value - Output parameter: parsed double value (set to default if not found)
 *		default_value - Default value to use if key not found or parsing fails
 *
 * Side Effects:
 *		- Modifies *value parameter
 *		- May allocate temporary memory for JSONB iteration (freed internally)
 *		- Catches and handles JSONB parsing errors
 *
 * Error Handling:
 *		- If hyperparams, key, or value is NULL: sets *value to default and returns
 *		- If JSONB is corrupted: catches error, uses default, logs DEBUG2 message
 *		- If key not found: *value remains at default_value
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Default value handling, error message text
 *		- CANNOT MODIFY: PG_TRY/PG_CATCH structure (required for error handling)
 *		- BREAKS IF: JSONB structure changes without updating field access
 */
static void
neurondb_parse_hyperparams_float8(Jsonb * hyperparams, const char *key, double *value, double default_value)
{
	Jsonb *field_jsonb = NULL;
	JsonbValue	v;
	JsonbIterator *it = NULL;
	int			r;

	if (hyperparams == NULL || key == NULL || value == NULL)
	{
		if (value)
			*value = default_value;
		ereport(DEBUG2,
				(errmsg("neurondb_parse_hyperparams_float8: NULL parameter, using default"),
				 errdetail("default_value=%.6f", default_value)));
		return;
	}

	*value = default_value;
	ereport(DEBUG2,
			(errmsg("neurondb_parse_hyperparams_float8: parsing hyperparameter"),
			 errdetail("key=%s, default_value=%.6f", key, default_value)));

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
					ereport(DEBUG2,
							(errmsg("neurondb_parse_hyperparams_float8: parsed value"),
							 errdetail("key=%s, value=%.6f", key, *value)));
					break;
				}
			}
		}
		else
		{
			ereport(DEBUG2,
					(errmsg("neurondb_parse_hyperparams_float8: key not found, using default"),
					 errdetail("key=%s, default_value=%.6f", key, default_value)));
		}
	}
	PG_CATCH();
	{
		FlushErrorState();
		ereport(DEBUG2,
				(errmsg("neurondb_parse_hyperparams_float8: JSONB parsing error, using default"),
				 errdetail("key=%s, default_value=%.6f", key, default_value)));
		*value = default_value;
	}
	PG_END_TRY();
}

/*
 * neurondb_build_training_sql
 *		Generate SQL statement for algorithm-specific training functions.
 *
 * This function constructs SQL statements that call algorithm-specific training
 * functions (e.g., train_linear_regression, train_logistic_regression). It handles
 * hyperparameter extraction and SQL string generation for algorithms that support
 * direct SQL-based training.
 *
 * Parameters:
 *		algo - MLAlgorithm enum value specifying which algorithm to train
 *		sql - StringInfo buffer to append generated SQL to (must be initialized)
 *		table_name - Name of table containing training data
 *		feature_list - SQL expression for feature columns (e.g., "col1, col2" or "*")
 *		target_column - Name of target/label column (NULL for unsupervised algorithms)
 *		hyperparams - JSONB object containing hyperparameters (may be NULL)
 *		feature_names - Array of feature column names (may be NULL)
 *		feature_name_count - Number of elements in feature_names array
 *
 * Returns:
 *		true if SQL was successfully generated and appended to sql buffer
 *		false if algorithm requires special handling (e.g., hierarchical clustering)
 *
 * Supported Algorithms:
 *		- Linear/Logistic Regression, Ridge, Lasso
 *		- Random Forest, Decision Tree, SVM
 *		- K-means, GMM, Mini-batch K-means
 *		- XGBoost, CatBoost, LightGBM
 *		- Naive Bayes, KNN variants
 *		- Time series (ARIMA)
 *
 * SQL Generation:
 *		- Appends SELECT statement calling algorithm-specific training function
 *		- Parameters are properly quoted using neurondb_quote_literal_cstr()
 *		- Hyperparameters are extracted from JSONB and included in function call
 *
 * Side Effects:
 *		- Modifies sql StringInfo buffer (appends SQL)
 *		- May allocate memory for quoted strings (freed by caller)
 *		- Calls hyperparameter parsing functions
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Default hyperparameter values, SQL function names
 *		- CAN MODIFY: Add new algorithm cases (must have corresponding SQL function)
 *		- CANNOT MODIFY: StringInfo append order (must match function signatures)
 *		- BREAKS IF: SQL function signatures change without updating SQL generation
 *		- BREAKS IF: Feature column selection logic changes incorrectly
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

	ereport(DEBUG1,
			(errmsg("neurondb_build_training_sql: building SQL for algorithm"),
			 errdetail("algo=%d, table_name=%s", (int) algo, table_name ? table_name : "NULL")));

	switch (algo)
	{
		case ML_ALGO_LINEAR_REGRESSION:
			ereport(DEBUG2,
					(errmsg("neurondb_build_training_sql: generating SQL for linear regression")));
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


				nfree(quoted_table);
				nfree(quoted_feature);
				if (quoted_target != NULL && strcmp(quoted_target, "NULL") != 0)
					nfree(quoted_target);
			}
			return true;

		case ML_ALGO_SVM:
			C = 1.0;
			max_iters = 1000;
			neurondb_parse_hyperparams_float8(hyperparams, "C", &C, 1.0);
			neurondb_parse_hyperparams_int(hyperparams, "max_iters", &max_iters, 1000);

			/*
			 * Use first feature name if available and not "*", otherwise default to
			 * "features". SVM requires a single column name, not "*" or comma-separated list.
			 */
			if (feature_name_count > 0 && feature_names != NULL && feature_names[0] != NULL && 
				strlen(feature_names[0]) > 0 && strcmp(feature_names[0], "*") != 0)
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
							 "SELECT train_svm_classifier(%s, %s, %s, %.6f, %d)",
							 neurondb_quote_literal_cstr(table_name),
							 neurondb_quote_literal_cstr(feature_col),
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
				/* Hierarchical clustering uses direct function cluster_hierarchical()
				 * which doesn't return a model_id. It's not supported through the
				 * unified API SQL path. Return false to indicate special handling needed.
				 */
				return false;
			}

		case ML_ALGO_XGBOOST:
			{
				int			n_estimators = 100;
				int			max_depth_xgb = 3;
				double		learning_rate_xgb = 0.1;
				const char *feature_col_xgb;

				neurondb_parse_hyperparams_int(hyperparams, "n_estimators", &n_estimators, 100);
				neurondb_parse_hyperparams_int(hyperparams, "max_depth", &max_depth_xgb, 3);
				neurondb_parse_hyperparams_float8(hyperparams, "learning_rate", &learning_rate_xgb, 0.1);
				
				/* Use first feature name if available, otherwise use feature_list */
				if (feature_name_count > 0 && feature_names != NULL && feature_names[0] != NULL)
					feature_col_xgb = feature_names[0];
				else
					feature_col_xgb = feature_list;
				
				appendStringInfo(sql,
								 "SELECT train_xgboost_classifier(%s::text, %s::text, %s::text, %d, %d, %.6f::double precision)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_col_xgb),
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
								 "SELECT train_catboost_classifier(%s::text, %s::text, %s::text, %d, %.6f::double precision, %d)",
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
								 "SELECT train_lightgbm_classifier(%s::text, %s::text, %s::text, %d, %d, %.6f::double precision)",
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
				const char *feature_col_knn;

				neurondb_parse_hyperparams_int(hyperparams, "k", &k_value, 5);
				
				/* Use first feature name if available, otherwise use feature_list */
				if (feature_name_count > 0 && feature_names != NULL && feature_names[0] != NULL)
					feature_col_knn = feature_names[0];
				else
					feature_col_knn = feature_list;
				
				appendStringInfo(sql,
								 "SELECT train_knn_model_id(%s, %s, %s, %d)",
								 neurondb_quote_literal_cstr(table_name),
								 neurondb_quote_literal_cstr(feature_col_knn),
								 neurondb_quote_literal_or_null(target_column),
								 k_value);
				return true;
			}

		case ML_ALGO_TRANSFORMER_LLM:
		case ML_ALGO_TITANS_LLM:
			{
				/* Transformer/Titans LLM requires special handling - call C function directly */
				ereport(DEBUG2,
						(errmsg("neurondb_build_training_sql: transformer_llm/titans_llm requires special handling"),
						 errdetail("algo=%d, returning false", (int) algo)));
				return false;
			}

		default:
			/* Algorithm requires special handling, not simple SQL generation */
			ereport(DEBUG2,
					(errmsg("neurondb_build_training_sql: algorithm requires special handling"),
					 errdetail("algo=%d, returning false", (int) algo)));
			return false;
	}

	ereport(DEBUG1,
			(errmsg("neurondb_build_training_sql: SQL generation completed"),
			 errdetail("sql_length=%d", sql->len)));
}

/*
 * neurondb_validate_training_inputs
 *		Validate input parameters for training operations.
 *
 * This function performs comprehensive validation of all input parameters
 * required for model training. It checks for NULL values and ensures that
 * supervised algorithms have target columns specified.
 *
 * Parameters:
 *		project_name - Name of the ML project (must not be NULL)
 *		algorithm - Algorithm name string (must not be NULL)
 *		table_name - Name of training data table (must not be NULL)
 *		target_column - Name of target/label column (may be NULL for unsupervised)
 *
 * Side Effects:
 *		- Raises ERROR if any required parameter is NULL
 *		- Raises ERROR if supervised algorithm lacks target_column
 *
 * Error Handling:
 *		- All validation errors raise ERROR with detailed messages
 *		- Error messages include hints for users
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Error message text, validation rules
 *		- CANNOT MODIFY: NULL checks for project_name, algorithm, table_name
 *		- BREAKS IF: Validation logic removed (will allow invalid inputs)
 */
static void
__attribute__((unused))
neurondb_validate_training_inputs(const char *project_name, const char *algorithm,
								  const char *table_name, const char *target_column)
{
	ereport(DEBUG1,
			(errmsg("neurondb_validate_training_inputs: validating inputs"),
			 errdetail("project_name=%s, algorithm=%s, table_name=%s",
					   project_name ? project_name : "NULL",
					   algorithm ? algorithm : "NULL",
					   table_name ? table_name : "NULL")));

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

	ereport(DEBUG1,
			(errmsg("neurondb_validate_training_inputs: validation passed")));
}

/*
 * neurondb_is_unsupervised_algorithm
 *		Determine if an algorithm is unsupervised (doesn't require target_column).
 *
 * Unsupervised algorithms learn patterns from data without labeled examples.
 * They include clustering algorithms (K-means, GMM, hierarchical) and
 * dimensionality reduction techniques.
 *
 * Parameters:
 *		algorithm - Algorithm name string (may be NULL)
 *
 * Returns:
 *		true if algorithm is unsupervised (doesn't need target_column)
 *		false if algorithm is supervised or NULL/invalid
 *
 * Unsupervised Algorithms:
 *		- ML_ALGO_KMEANS: K-means clustering
 *		- ML_ALGO_GMM: Gaussian Mixture Model clustering
 *		- ML_ALGO_MINIBATCH_KMEANS: Mini-batch K-means
 *		- ML_ALGO_HIERARCHICAL: Hierarchical clustering
 *		- ML_ALGO_DBSCAN: Density-based clustering
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Add new unsupervised algorithms to the check list
 *		- CANNOT MODIFY: Use of neurondb_algorithm_from_string() (type safety)
 *		- BREAKS IF: Algorithm enum values change without updating comparisons
 */
static bool
neurondb_is_unsupervised_algorithm(const char *algorithm)
{
	MLAlgorithm algo_enum;
	bool		is_unsupervised;

	if (algorithm == NULL)
	{
		ereport(DEBUG2,
				(errmsg("neurondb_is_unsupervised_algorithm: NULL algorithm, returning false")));
		return false;
	}

	algo_enum = neurondb_algorithm_from_string(algorithm);
	is_unsupervised = (algo_enum == ML_ALGO_GMM ||
						algo_enum == ML_ALGO_KMEANS ||
						algo_enum == ML_ALGO_MINIBATCH_KMEANS ||
						algo_enum == ML_ALGO_HIERARCHICAL ||
						algo_enum == ML_ALGO_DBSCAN);

	ereport(DEBUG2,
			(errmsg("neurondb_is_unsupervised_algorithm: algorithm check"),
			 errdetail("algorithm=%s, is_unsupervised=%s", algorithm, is_unsupervised ? "true" : "false")));

	return is_unsupervised;
}

/*
 * neurondb_prepare_feature_list
 *		Prepare feature list string and feature names array from input array.
 *
 * This function processes a PostgreSQL array of feature column names and
 * generates both a SQL-compatible feature list string and an array of
 * individual feature names. Currently implements a simplified version that
 * uses "*" to select all columns.
 *
 * Parameters:
 *		feature_columns_array - PostgreSQL array of feature column names (currently unused)
 *		feature_list - StringInfo buffer to append feature list to (must be initialized)
 *		feature_names_out - Output: array of feature name strings (caller must free)
 *		feature_name_count_out - Output: number of feature names
 *
 * Returns:
 *		Number of features prepared (currently always 1 for "*")
 *
 * Memory Management:
 *		- Allocates feature_names array using nalloc (caller must free with nfree)
 *		- Allocates individual feature name strings (freed when array freed)
 *		- Modifies feature_list StringInfo buffer
 *
 * Current Implementation:
 *		- Simplified version that always uses "*" (all columns)
 *		- Future: should parse feature_columns_array to extract individual names
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Implementation to parse feature_columns_array properly
 *		- CANNOT MODIFY: Output parameter initialization (must set before return)
 *		- BREAKS IF: feature_names not freed by caller (memory leak)
 */
static int
__attribute__((unused))
neurondb_prepare_feature_list(ArrayType * feature_columns_array, StringInfo feature_list,
							  const char ***feature_names_out, int *feature_name_count_out)
{
	const char **feature_names = NULL;
	int			feature_name_count = 0;

	ereport(DEBUG2,
			(errmsg("neurondb_prepare_feature_list: preparing feature list")));

	/* Use simplified version to avoid array processing complexity */
	appendStringInfoString(feature_list, "*");
	nalloc(feature_names, const char *, 1);
	feature_names[0] = pstrdup("*");
	feature_name_count = 1;

	if (feature_names_out)
		*feature_names_out = feature_names;
	if (feature_name_count_out)
		*feature_name_count_out = feature_name_count;

	ereport(DEBUG2,
			(errmsg("neurondb_prepare_feature_list: feature list prepared"),
			 errdetail("feature_name_count=%d, feature_list=%s", feature_name_count, feature_list->data)));

	return feature_name_count;
}

/* Removed unused function neurondb_get_model_type */

/* ----------
 * neurondb_train
 *		Unified SQL interface for training machine learning models.
 *
 * This is the main entry point for model training in NeuronDB. It provides a
 * unified API that abstracts GPU/CPU execution, algorithm-specific details, and
 * model catalog management. The function supports automatic fallback from GPU
 * to CPU when GPU is unavailable or when algorithms are unsupported on GPU.
 *
 * SQL Function Signature:
 *		neurondb.train(project_name TEXT, algorithm TEXT, table_name TEXT,
 *					   target_column TEXT, feature_columns TEXT[],
 *					   hyperparams JSONB) RETURNS INTEGER
 *
 * Parameters:
 *		project_name - Name of ML project (organizes models)
 *		algorithm - Algorithm name (e.g., 'logistic_regression', 'kmeans')
 *		table_name - Name of table containing training data
 *		target_column - Name of target/label column (NULL for unsupervised)
 *		feature_columns - Array of feature column names (currently simplified)
 *		hyperparams - JSONB object with algorithm-specific hyperparameters
 *
 * Returns:
 *		INTEGER: model_id of the trained model (positive integer)
 *		Raises ERROR on failure
 *
 * Execution Flow:
 *		1. Validate and parse input parameters
 *		2. Create/get project in catalog
 *		3. Attempt GPU training (if GPU available and algorithm supported)
 *		4. Fall back to CPU training if GPU fails or unavailable
 *		5. Register model in catalog with metadata
 *		6. Return model_id
 *
 * GPU/CPU Fallback Logic:
 *		- GPU mode (compute_mode='gpu'): Requires GPU, errors if unavailable
 *		- AUTO mode (compute_mode='auto'): Tries GPU first, falls back to CPU
 *		- CPU mode (compute_mode='cpu'): Uses CPU only
 *		- Metal backend: Some algorithms unsupported, auto-fallback to CPU
 *
 * Memory Management:
 *		- Creates dedicated memory context for function execution
 *		- Allocates training data arrays (feature_matrix, label_vector)
 *		- Manages SPI session lifecycle
 *		- All memory freed via neurondb_cleanup() on exit
 *
 * Model Catalog:
 *		- Creates/updates project in neurondb.ml_projects
 *		- Registers model in neurondb.ml_models with full metadata
 *		- Stores training metrics, hyperparameters, and model data
 *		- Uses advisory locks for concurrent safety
 *
 * Error Handling:
 *		- Comprehensive error messages with hints
 *		- Proper cleanup on all error paths
 *		- GPU errors include detailed diagnostics
 *		- CPU fallback errors include GPU error context
 *
 * Thread Safety:
 *		- Uses advisory locks for project-level concurrency control
 *		- Each function call operates in its own memory context
 *		- SPI sessions are session-local
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Default hyperparameters, error message text, algorithm support
 *		- CANNOT MODIFY: Memory context lifecycle (must create before use, delete after)
 *		- CANNOT MODIFY: SPI session management order (begin before use, end after)
 *		- CANNOT MODIFY: GPU/CPU fallback decision logic (critical for correctness)
 *		- BREAKS IF: Memory contexts deleted while in use
 *		- BREAKS IF: SPI sessions used after end
 *		- BREAKS IF: Model catalog schema changes without updating SQL
 * ----------
 */
Datum
neurondb_train(PG_FUNCTION_ARGS)
{
	text *project_name_text = NULL;
	text *algorithm_text = NULL;
	text *table_name_text = NULL;
	text *target_column_text = NULL;
	ArrayType *feature_columns_array = NULL;
	Jsonb *hyperparams = NULL;
	StringInfoData feature_list;

	const char **feature_names = NULL;
	int			feature_name_count = 0;

	char *model_name = NULL;
	MLGpuTrainResult gpu_result;

	char *gpu_errmsg_ptr = NULL;
	char	  **gpu_errmsg = &gpu_errmsg_ptr;
	char *project_name = NULL;
	char *algorithm = NULL;
	char *table_name = NULL;
	char *target_column = NULL;
	char *tmp_target_lit = NULL;

	char *default_project_name = NULL;	/* Pre-allocated "default"
												 * string */
	MemoryContext callcontext;
	MemoryContext oldcontext;

	NdbSpiSession *spi_session = NULL;
	StringInfoData sql;
	int			ret;
	int			model_id = 0;	/* Initialize to 0 to avoid returning garbage if training fails */
	MLCatalogModelSpec spec;
	MLAlgorithm algo_enum;

	float *feature_matrix = NULL;
	double *label_vector = NULL;
	int			n_samples = 0;
	int			feature_dim = 0;
	int			class_count = 0;
	bool		data_loaded = false;
	char *feature_list_str = NULL;
	bool		gpu_available = false;
	bool		load_success = false;
	bool		gpu_train_result = false;
	bool		metal_requested_fallback = false;  /* Track if Metal backend requested CPU fallback */
	int			saved_compute_mode = -1;  /* Save compute_mode when temporarily changing for Metal unsupported algorithms */
	char *safe_algorithm = NULL;
	char *safe_table_name = NULL;
	char *safe_target_column = NULL;
	char *safe_project_name = NULL;

	ereport(DEBUG1,
			(errmsg("neurondb_train: starting model training"),
			 errdetail("nargs=%d", PG_NARGS())));


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
		strcmp(algorithm, NDB_ALGO_HIERARCHICAL) != 0 &&
		strcmp(algorithm, NDB_ALGO_XGBOOST) != 0 &&
		strcmp(algorithm, NDB_ALGO_CATBOOST) != 0 &&
		strcmp(algorithm, NDB_ALGO_LIGHTGBM) != 0 &&
		strcmp(algorithm, NDB_ALGO_TIMESERIES) != 0 &&
		strcmp(algorithm, "neural_network") != 0 &&
		strcmp(algorithm, NDB_ALGO_TRANSFORMER_LLM) != 0 &&
		strcmp(algorithm, NDB_ALGO_CUSTOM_LLM) != 0 &&
		strcmp(algorithm, NDB_ALGO_TITANS_LLM) != 0)
	{
		nfree(project_name);
		nfree(algorithm);
		nfree(table_name);
		if (target_column)
			nfree(target_column);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_TRAIN " unsupported algorithm '%s'", algorithm),
				 errdetail("Supported algorithms: linear_regression, logistic_regression, svm, random_forest, knn, kmeans, dbscan, naive_bayes, decision_tree, gmm, xgboost, catboost, lightgbm, timeseries, neural_network"),
				 errhint("Choose one of the supported algorithms: linear_regression, logistic_regression, svm, random_forest, knn, kmeans, dbscan, naive_bayes, decision_tree, gmm, xgboost, catboost, lightgbm, timeseries, neural_network, transformer_llm, custom_llm, titans_llm")));
	}


	/* Create memory context for this function call */
	callcontext = AllocSetContextCreate(CurrentMemoryContext,
										"neurondb_train context",
										ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

	default_project_name = pstrdup("default");
	
	if (hyperparams != NULL)
	{
		hyperparams = (Jsonb *) PG_DETOAST_DATUM_COPY(PointerGetDatum(hyperparams));
	}
	
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
					 target_column ? (tmp_target_lit = psprintf("'%s'", target_column)) : "NULL");

	ret = ndb_spi_execute(spi_session, sql.data, false, 0);

	if (tmp_target_lit != NULL)
	{
		pfree(tmp_target_lit);
		tmp_target_lit = NULL;
	}

	if (ret != SPI_OK_INSERT_RETURNING)
	{
		ndb_spi_stringinfo_free(spi_session, &sql);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		nfree(project_name);
		nfree(algorithm);
		nfree(table_name);
		if (target_column)
			nfree(target_column);
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

		Datum *elem_values = NULL;
		bool *elem_nulls = NULL;
		int			i;

		/* Validate array */
		if (ARR_NDIM(feature_columns_array) != 1)
		{
			ndb_spi_session_end(&spi_session);
			MemoryContextSwitchTo(oldcontext);
			neurondb_cleanup(oldcontext, callcontext);
			nfree(project_name);
			nfree(algorithm);
			nfree(table_name);
			if (target_column)
				nfree(target_column);
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

			if (elem_type != TEXTOID)
			{
				ndb_spi_session_end(&spi_session);
				MemoryContextSwitchTo(oldcontext);
				neurondb_cleanup(oldcontext, callcontext);
				nfree(project_name);
				nfree(algorithm);
				nfree(table_name);
				if (target_column)
					nfree(target_column);
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

				nalloc(feature_names, const char *, nelems);
				MemSet(feature_names, 0, nelems * sizeof(const char *));

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
						 * Copy col to callcontext for feature_names.
						 * We don't free col here - it's in SPI context and will be
						 * cleaned up when the SPI session ends.
						 */
						MemoryContextSwitchTo(callcontext);
						feature_names[feature_name_count++] = pstrdup(col);
						MemoryContextSwitchTo(ndb_spi_session_get_context(spi_session));
					}
				}

				/* 
				 * Don't manually free elem_values and elem_nulls - they were allocated
				 * by deconstruct_array in callcontext and will be automatically cleaned
				 * up when callcontext is destroyed at function end.
				 */
				
				MemoryContextSwitchTo(old_spi_context);
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

		nalloc(feature_names, const char *, 1);
		MemSet(feature_names, 0, sizeof(const char *));
		feature_names[0] = pstrdup("*");
		feature_name_count = 1;
	}


	if (feature_list.data == NULL || feature_list.len == 0)
	{
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		nfree(project_name);
		nfree(algorithm);
		nfree(table_name);
		if (target_column)
			nfree(target_column);
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
	MemoryContextSwitchTo(callcontext);
	feature_list_str = pstrdup(feature_list.data);
	if (feature_list_str == NULL)
	{
		MemoryContextSwitchTo(oldcontext);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		nfree(project_name);
		nfree(algorithm);
		nfree(table_name);
		if (target_column)
			nfree(target_column);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_PREFIX_TRAIN " failed to copy feature_list to callcontext")));
	}
	MemoryContextSwitchTo(oldcontext);


	MemSet(&gpu_result, 0, sizeof(MLGpuTrainResult));

	model_name = psprintf("%s_%s", algorithm, project_name);
	if (model_name == NULL)
	{
		nfree(feature_list_str);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		nfree(project_name);
		nfree(algorithm);
		nfree(table_name);
		if (target_column)
			nfree(target_column);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_PREFIX_TRAIN " failed to allocate memory for model_name")));
	}

	feature_matrix = NULL;
	label_vector = NULL;
	n_samples = 0;
	feature_dim = 0;
	class_count = 0;
	data_loaded = false;

	if (NDB_SHOULD_TRY_GPU())
	{
		ndb_gpu_init_if_needed();
	}

	gpu_available = neurondb_gpu_is_available();

	/* GPU mode requires GPU to be available */
	if (NDB_REQUIRE_GPU() && !gpu_available)
	{
		if (feature_names)
		{
			int			i;

			MemoryContextSwitchTo(callcontext);
			for (i = 0; i < feature_name_count; i++)
			{
				if (feature_names[i] != NULL)
				{
					char	   *ptr = (char *) feature_names[i];

					nfree(ptr);
				}
			}
			nfree(feature_names);
			feature_names = NULL;
		}
		
		nfree(feature_list_str);
		ndb_spi_session_end(&spi_session);
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		nfree(project_name);
		nfree(algorithm);
		nfree(table_name);
		if (target_column)
			nfree(target_column);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg(NDB_ERR_PREFIX_TRAIN " GPU is required but not available"),
				 errdetail("compute_mode is set to 'gpu' but GPU backend could not be initialized"),
				 errhint("Check GPU hardware, drivers, and configuration. "
						 "Set compute_mode='auto' for automatic CPU fallback.")));
	}

	/* Try to load training data for GPU training */
	/* GPU mode: require GPU, no fallback. AUTO mode: try GPU with fallback. CPU mode: never load GPU data */
	if (!NDB_COMPUTE_MODE_IS_CPU() && gpu_available && 
		(NDB_REQUIRE_GPU() || NDB_COMPUTE_MODE_IS_AUTO()))
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
	 * CRITICAL: Save copies of all string arguments in TopMemoryContext
	 * BEFORE GPU training, as GPU training may destroy the current context
	 */
	if (data_loaded && !NDB_COMPUTE_MODE_IS_CPU())
	{
		MemoryContext prev_context = MemoryContextSwitchTo(TopMemoryContext);
		safe_algorithm = algorithm ? pstrdup(algorithm) : NULL;
		safe_table_name = table_name ? pstrdup(table_name) : NULL;
		safe_target_column = target_column ? pstrdup(target_column) : NULL;
		safe_project_name = project_name ? pstrdup(project_name) : NULL;
		MemoryContextSwitchTo(prev_context);
	}

	/*
	 * Double-check CPU mode to be absolutely safe - NEVER attempt GPU in CPU
	 * mode
	 * GPU mode: require GPU, no fallback. AUTO mode: try GPU with fallback.
	 */
	if (data_loaded && !NDB_COMPUTE_MODE_IS_CPU() &&
		(NDB_REQUIRE_GPU() || NDB_COMPUTE_MODE_IS_AUTO()))
	{
		/*
		 * Ensure hyperparams is always a valid Jsonb (even if empty) to
		 * prevent JSON parsing errors in GPU code. This matches the
		 * behavior in CPU training code (ml_linear_regression.c).
		 */
		Jsonb	   *gpu_hyperparams = NULL;
		MemoryContext prev_gpu_context = MemoryContextSwitchTo(callcontext);

		if (hyperparams != NULL)
		{
			gpu_hyperparams = (Jsonb *) PG_DETOAST_DATUM_COPY(PointerGetDatum(hyperparams));
		}
		else
		{
			/* Create empty JSONB object like CPU training does */
			PG_TRY();
			{
				gpu_hyperparams = ndb_jsonb_in_cstring("{}");
				if (gpu_hyperparams == NULL)
				{
					MemoryContextSwitchTo(prev_gpu_context);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
							 errmsg(NDB_ERR_PREFIX_TRAIN " failed to create empty hyperparams JSONB")));
				}
			}
			PG_CATCH();
			{
				FlushErrorState();
				MemoryContextSwitchTo(prev_gpu_context);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
						 errmsg(NDB_ERR_PREFIX_TRAIN " failed to parse empty hyperparams JSON: %m")));
			}
			PG_END_TRY();
		}

		MemoryContextSwitchTo(prev_gpu_context);

		/*
		 * Wrap GPU training call in PG_TRY to catch exceptions and
		 * prevent JSON parsing errors
			 */
			PG_TRY();
			{
				gpu_train_result = ndb_gpu_try_train_model(algorithm, project_name, model_name, table_name, target_column,
														   feature_names, feature_name_count, gpu_hyperparams,
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
							nfree(feature_matrix);
							feature_matrix = NULL;
						}
						if (label_vector)
						{
							nfree(label_vector);
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
					ErrorData *edata = NULL;
					char *error_msg = NULL;
					char *algorithm_safe = NULL;

					/*
					 * Defensive assertion: this should NEVER happen in CPU
					 * mode
					 */
					if (NDB_COMPUTE_MODE_IS_CPU())
					{
						elog(ERROR, "Internal error: GPU error path reached in CPU mode (compute_mode=%d)", neurondb_compute_mode);
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

				algorithm_safe = algorithm ? pstrdup(algorithm) : NULL;

				if (CurrentMemoryContext != ErrorContext)
				{
					edata = CopyErrorData();
					if (edata)
					{
						/* Prefer detail message over main message for more specific errors */
						if (edata->detail && strlen(edata->detail) > 0)
						{
							error_msg = pstrdup(edata->detail);
						}
						else if (edata->message && strlen(edata->message) > 0)
						{
							error_msg = pstrdup(edata->message);
						}
								if (gpu_errmsg_ptr == NULL && error_msg != NULL)
						{
							gpu_errmsg_ptr = pstrdup(error_msg);
						}
					}
					FlushErrorState();
				}
				else
				{
					/* Fallback if we can't switch contexts */
					FlushErrorState();
					error_msg = NULL;
				}

					nfree(feature_list_str);
					if (feature_names)
					{
						int			i;

						for (i = 0; i < feature_name_count; i++)
						{
							if (feature_names[i] != NULL)
							{
								char	   *ptr = (char *) feature_names[i];

								nfree(ptr);
							}
						}
						nfree(feature_names);
					}
					if (model_name)
						nfree(model_name);
					if (gpu_errmsg_ptr)
						nfree(gpu_errmsg_ptr);
					ndb_spi_session_end(&spi_session);
					MemoryContextSwitchTo(oldcontext);
					neurondb_cleanup(oldcontext, callcontext);
					nfree(project_name);
					nfree(algorithm);
					nfree(table_name);
					if (target_column)
						nfree(target_column);
					if (data_loaded)
					{
						if (feature_matrix)
							nfree(feature_matrix);
						if (label_vector)
							nfree(label_vector);
					}
					if (edata)
						FreeErrorData(edata);

					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg(NDB_ERR_PREFIX_TRAIN " GPU training failed - GPU mode requires GPU to be available"),
							 errdetail("Algorithm: %s, Error: %s", algorithm_safe ? algorithm_safe : "unknown", error_msg ? error_msg : "unknown"),
							 errhint("Set compute_mode='auto' for automatic CPU fallback.")));

					if (algorithm_safe)
						nfree(algorithm_safe);
				}
				else
				{
					if (NDB_REQUIRE_GPU())
					{
						/* GPU mode: re-raise error, no fallback */
						ErrorData *edata = NULL;
						char *error_msg = NULL;
						char *algorithm_safe = NULL;
						MemoryContext safe_context;

						/* Switch out of ErrorContext before CopyErrorData() */
						safe_context = oldcontext;

						/* Ensure we're not switching to ErrorContext */
						if (safe_context == ErrorContext || safe_context == NULL)
						{
							safe_context = TopMemoryContext;
						}

						MemoryContextSwitchTo(safe_context);

						algorithm_safe = algorithm ? pstrdup(algorithm) : NULL;

						if (CurrentMemoryContext != ErrorContext)
						{
							edata = CopyErrorData();
							if (edata)
							{
								/* Prefer detail message over main message for more specific errors */
								if (edata->detail && strlen(edata->detail) > 0)
								{
									error_msg = pstrdup(edata->detail);
								}
								else if (edata->message && strlen(edata->message) > 0)
								{
									error_msg = pstrdup(edata->message);
								}
								if (gpu_errmsg_ptr == NULL && error_msg != NULL)
								{
									gpu_errmsg_ptr = pstrdup(error_msg);
								}
							}
							FlushErrorState();
						}

						if (algorithm_safe)
							nfree(algorithm_safe);

						if (data_loaded && callcontext != NULL && MemoryContextIsValid(callcontext))
						{
							MemoryContextSwitchTo(callcontext);
							if (feature_matrix)
							{
								nfree(feature_matrix);
								feature_matrix = NULL;
							}
							if (label_vector)
							{
								nfree(label_vector);
								label_vector = NULL;
							}
							data_loaded = false;
							MemoryContextSwitchTo(safe_context);
						}
						else if (data_loaded)
						{
							feature_matrix = NULL;
							label_vector = NULL;
							data_loaded = false;
						}

						nfree(feature_list_str);
						if (feature_names)
						{
							int			i;

							MemoryContextSwitchTo(callcontext);
							for (i = 0; i < feature_name_count; i++)
							{
								if (feature_names[i])
								{
									char	   *ptr = (char *) feature_names[i];

									nfree(ptr);
								}
							}
							nfree(feature_names);
						}
						if (model_name)
							nfree(model_name);
						ndb_spi_session_end(&spi_session);
						MemoryContextSwitchTo(oldcontext);
						neurondb_cleanup(oldcontext, callcontext);

						/* Report error */
						{
							char *gpu_error_msg = NULL;
							if (gpu_errmsg_ptr && strlen(gpu_errmsg_ptr) > 0)
							{
								gpu_error_msg = pstrdup(gpu_errmsg_ptr);
							}
							else if (error_msg)
							{
								gpu_error_msg = pstrdup(error_msg);
							}
							else
							{
								gpu_error_msg = psprintf("Exception during GPU training for algorithm '%s'", algorithm ? algorithm : "unknown");
							}
							if (gpu_errmsg_ptr)
								nfree(gpu_errmsg_ptr);
							if (error_msg)
								pfree(error_msg);
							if (edata)
								FreeErrorData(edata);
							
							ereport(ERROR,
									(errcode(ERRCODE_INTERNAL_ERROR),
									 errmsg(NDB_ERR_PREFIX_TRAIN " GPU training failed - GPU mode requires GPU to be available"),
									 errdetail("Algorithm: %s, Project: %s, Table: %s. Error: %s",
											   algorithm ? algorithm : "unknown",
											   project_name ? project_name : "unknown",
											   table_name ? table_name : "unknown",
											   gpu_error_msg),
									 errhint("GPU training encountered an exception. Check GPU hardware, drivers, and configuration. "
											 "Set compute_mode='auto' for automatic CPU fallback.")));
							if (gpu_error_msg)
								pfree(gpu_error_msg);
						}
					}
					
					/* AUTO mode: fall back to CPU */
					/* Switch out of ErrorContext before any operations */
					{
						MemoryContext safe_context;

						safe_context = oldcontext;

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
							nfree(feature_matrix);
							feature_matrix = NULL;
						}
						if (label_vector)
						{
							nfree(label_vector);
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
			}
		PG_END_TRY();
	}

	if (gpu_errmsg_ptr && 
		(strstr(gpu_errmsg_ptr, "using CPU fallback") != NULL ||
		 strstr(gpu_errmsg_ptr, "Metal backend:") != NULL ||
		 (strstr(gpu_errmsg_ptr, "Metal") != NULL && strstr(gpu_errmsg_ptr, "unsupported") != NULL)))
	{
		metal_requested_fallback = true;
	}

	/* In AUTO mode, or if Metal backend requested fallback, skip GPU result processing and go to CPU training */
	if (!gpu_train_result && (NDB_COMPUTE_MODE_IS_AUTO() || NDB_COMPUTE_MODE_IS_CPU() || metal_requested_fallback))
	{
		/* Skip GPU result processing, jump directly to CPU training fallback below */
		goto cpu_fallback_training;
	}

	if (gpu_train_result && (gpu_result.model_id > 0 || gpu_result.spec.model_data != NULL))
	{
		ereport(DEBUG1,
				(errmsg("neurondb_train: GPU training succeeded"),
				 errdetail("algorithm=%s, model_id=%d", algorithm ? algorithm : "unknown", gpu_result.model_id)));

		/* GPU training succeeded - use the model_id from GPU result */
		if (gpu_result.model_id > 0)
			{
				model_id = gpu_result.model_id;
				ereport(DEBUG2,
						(errmsg("neurondb_train: using GPU training model_id"),
						 errdetail("model_id=%d", model_id)));
			}
			else if (gpu_result.model_id == 0 && gpu_result.spec.model_data != NULL)
			{
				spec = gpu_result.spec;
				
				/* Convert GPU format to unified format for logistic_regression */
				if (algorithm != NULL && strcmp(algorithm, "logistic_regression") == 0)
				{
#if defined(NDB_GPU_CUDA) || defined(NDB_GPU_METAL)
					bytea *unified_model_data = NULL;
					bytea *gpu_model_data = spec.model_data;
					
					if (gpu_model_data != NULL && VARSIZE(gpu_model_data) - VARHDRSZ >= sizeof(int32))
					{
						const int32 *first_int = (const int32 *) VARDATA(gpu_model_data);
						int32		first_value = *first_int;
						
						/* GPU format: first field is feature_dim (int32), typically 1-10000 */
						/* Unified format: first byte is training_backend (uint8), typically 0 or 1 */
						if (first_value > 0 && first_value <= 10000)
						{
							size_t		gpu_header_size = sizeof(NdbCudaLrModelHeader);
							size_t		expected_gpu_size = gpu_header_size + sizeof(float) * (size_t) first_value;
							size_t		actual_size = VARSIZE(gpu_model_data) - VARHDRSZ;
							
							if (actual_size >= gpu_header_size && 
								actual_size >= expected_gpu_size - 10 && 
								actual_size <= expected_gpu_size + 10)
							{
								/* This is GPU format - convert to unified format */
								LRModel		lr_model;
								char	   *base = NULL;
								NdbCudaLrModelHeader *hdr = NULL;
								float	   *weights_src = NULL;
								int			i;
								double		accuracy = 0.0;
								double		final_loss = 0.0;
								
								
								/* Convert GPU format to unified format */
								base = VARDATA(gpu_model_data);
								hdr = (NdbCudaLrModelHeader *) base;
								weights_src = (float *) (base + sizeof(NdbCudaLrModelHeader));
								
								/* Extract final_loss and accuracy from metrics if available */
								if (spec.metrics != NULL)
								{
									text	   *metrics_text = DatumGetTextP(DirectFunctionCall1(jsonb_out, PointerGetDatum(spec.metrics)));
									char	   *metrics_str = text_to_cstring(metrics_text);
									char	   *loss_ptr = strstr(metrics_str, "\"final_loss\":");
									char	   *acc_ptr = strstr(metrics_str, "\"accuracy\":");
									
									if (loss_ptr != NULL)
										final_loss = strtod(loss_ptr + 13, NULL);
									if (acc_ptr != NULL)
										accuracy = strtod(acc_ptr + 12, NULL);
									
									nfree(metrics_str);
								}
								
								/* Build LRModel structure */
								memset(&lr_model, 0, sizeof(LRModel));
								lr_model.n_features = hdr->feature_dim;
								lr_model.n_samples = hdr->n_samples;
								lr_model.bias = hdr->bias;
								lr_model.learning_rate = hdr->learning_rate;
								lr_model.lambda = hdr->lambda;
								lr_model.max_iters = hdr->max_iters;
								lr_model.final_loss = final_loss;
								lr_model.accuracy = accuracy;
								
								/* Convert float weights to double */
								if (lr_model.n_features > 0)
								{
									double *weights_tmp = NULL;
									nalloc(weights_tmp, double, lr_model.n_features);
									for (i = 0; i < lr_model.n_features; i++)
										weights_tmp[i] = (double) weights_src[i];
									lr_model.weights = weights_tmp;
								}
								
								/* Serialize using unified format with training_backend=1 (GPU) */
								/* Declare the function - it's static in ml_logistic_regression.c */
								{
									StringInfoData buf;
									int			j;
									
									pq_begintypsend(&buf);
									
									/* Write training_backend first (0=CPU, 1=GPU) */
									pq_sendbyte(&buf, 1);  /* GPU */
									
									pq_sendint32(&buf, lr_model.n_features);
									pq_sendint32(&buf, lr_model.n_samples);
									pq_sendfloat8(&buf, lr_model.bias);
									pq_sendfloat8(&buf, lr_model.learning_rate);
									pq_sendfloat8(&buf, lr_model.lambda);
									pq_sendint32(&buf, lr_model.max_iters);
									pq_sendfloat8(&buf, lr_model.final_loss);
									pq_sendfloat8(&buf, lr_model.accuracy);
									
									if (lr_model.weights != NULL && lr_model.n_features > 0)
									{
										for (j = 0; j < lr_model.n_features; j++)
											pq_sendfloat8(&buf, lr_model.weights[j]);
									}
									
									unified_model_data = pq_endtypsend(&buf);
									
									/* Cleanup LRModel weights */
									if (lr_model.weights != NULL)
									{
										nfree(lr_model.weights);
										lr_model.weights = NULL;
									}
								}
								
								if (unified_model_data != NULL)
								{
									spec.model_data = unified_model_data;
									
									{
										Jsonb	   *gpu_metrics_update = NULL;
										Jsonb	   *updated_metrics = NULL;
										
										PG_TRY();
										{
											/* Create JSONB with storage="gpu" and training_backend=1 */
											JsonbParseState *state = NULL;
											JsonbValue	jkey;
											JsonbValue	jval;
											JsonbValue *final_value = NULL;
											
											(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);
											
											/* Add storage="gpu" */
											jkey.type = jbvString;
											jkey.val.string.val = "storage";
											jkey.val.string.len = strlen("storage");
											(void) pushJsonbValue(&state, WJB_KEY, &jkey);
											jval.type = jbvString;
											jval.val.string.val = "gpu";
											jval.val.string.len = strlen("gpu");
											(void) pushJsonbValue(&state, WJB_VALUE, &jval);
											
											/* Add training_backend=1 */
											jkey.val.string.val = "training_backend";
											jkey.val.string.len = strlen("training_backend");
											(void) pushJsonbValue(&state, WJB_KEY, &jkey);
											jval.type = jbvNumeric;
											jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(1)));
											(void) pushJsonbValue(&state, WJB_VALUE, &jval);
											
											final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
											
											if (final_value != NULL)
											{
												gpu_metrics_update = JsonbValueToJsonb(final_value);
												
												/* Merge with existing metrics if any */
												if (spec.metrics != NULL && gpu_metrics_update != NULL)
												{
													/* Use JSONB merge: existing || update (update overwrites) */
													updated_metrics = DatumGetJsonbP(DirectFunctionCall2(jsonb_concat,
																										  PointerGetDatum(spec.metrics),
																										  PointerGetDatum(gpu_metrics_update)));
												}
												else if (gpu_metrics_update != NULL)
												{
													updated_metrics = gpu_metrics_update;
												}
												else if (spec.metrics != NULL)
												{
													updated_metrics = spec.metrics;
												}
												
												if (updated_metrics != NULL)
												{
													spec.metrics = updated_metrics;
												}
											}
										}
										PG_CATCH();
										{
											elog(WARNING, "neurondb_train: failed to update metrics JSONB, using original metrics");
											FlushErrorState();
										}
										PG_END_TRY();
									}
								}
								else
								{
									elog(WARNING,
										 "neurondb_train: failed to convert logistic_regression model to unified format, using GPU format");
								}
							}
						}
					}
#endif
				}

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

				/* Defensive: NULL-safe fallback - use safe copies from TopMemoryContext */
				spec.algorithm = safe_algorithm ? MemoryContextStrdup(TopMemoryContext, safe_algorithm) : 
								 (algorithm ? MemoryContextStrdup(TopMemoryContext, algorithm) : NULL);
				spec.training_table = safe_table_name ? MemoryContextStrdup(TopMemoryContext, safe_table_name) : 
									 (table_name ? MemoryContextStrdup(TopMemoryContext, table_name) : NULL);
				spec.training_column = safe_target_column ? MemoryContextStrdup(TopMemoryContext, safe_target_column) : 
									  (target_column ? MemoryContextStrdup(TopMemoryContext, target_column) : NULL);

				/*
				 * Copy project_name - always use fallback value since
				 * spec.project_name may point to invalid memory
				 */
				PG_TRY();
				{
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
				
				/* Verify model was registered and is visible */
				if (model_id > 0)
				{
					/* 
					 * Note: ml_catalog_register_model already validates the INSERT succeeded.
					 * The verification query below may fail due to SPI session isolation even though
					 * the model was successfully registered. We trust ml_catalog_register_model's
					 * return value since it validates the INSERT internally.
					 */
					ndb_spi_stringinfo_free(spi_session, &sql);
					ndb_spi_stringinfo_init(spi_session, &sql);
					appendStringInfo(&sql,
									 "SELECT COUNT(*) FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_MODEL_ID " = %d",
									 model_id);
					ret = ndb_spi_execute(spi_session, sql.data, true, 0);
					
					if (ret == SPI_OK_SELECT && SPI_processed > 0)
					{
						int32 count = 0;
						if (ndb_spi_get_int32(spi_session, 0, 1, &count) && count == 0)
						{
							/* 
							 * Model not immediately visible - this can happen due to SPI session isolation.
							 * Since ml_catalog_register_model already validated the INSERT succeeded,
							 * we log a warning but don't error out. The model should be visible after
							 * the transaction commits.
							 */
							elog(WARNING,
								 "neurondb:train: GPU training registered model_id %d but model not immediately visible in catalog (this may be due to SPI session isolation)",
								 model_id);
						}
					}
				}
			}
			else
			{
				/*
				 * GPU training reported success but no model data - error
				 * instead of returning 0
				 */
				nfree(feature_list_str);
				ndb_spi_session_end(&spi_session);
				MemoryContextSwitchTo(oldcontext);
				neurondb_cleanup(oldcontext, callcontext);
				nfree(project_name);
				nfree(algorithm);
				nfree(table_name);
				if (target_column)
					nfree(target_column);
				if (feature_names)
				{
					int			i;

					for (i = 0; i < feature_name_count; i++)
					{
						if (feature_names[i])
						{
							char	   *ptr = (char *) feature_names[i];

							nfree(ptr);
						}
					}
					nfree(feature_names);
				}
				ndb_gpu_free_train_result(&gpu_result);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg(NDB_ERR_PREFIX_TRAIN " GPU training reported success but no model data"),
						 errdetail("algorithm=%s, model_id=%d", algorithm, gpu_result.model_id),
						 errhint("GPU training may have failed internally. Check logs for details or try CPU training.")));
			}

		if (data_loaded)
		{
			if (feature_matrix)
				nfree(feature_matrix);
			if (label_vector)
				nfree(label_vector);
		}

		ndb_gpu_free_train_result(&gpu_result);
		if (gpu_errmsg_ptr)
			nfree(gpu_errmsg_ptr);
		if (safe_algorithm)
			pfree(safe_algorithm);
		if (safe_table_name)
			pfree(safe_table_name);
		if (safe_target_column)
			pfree(safe_target_column);
		if (safe_project_name)
			pfree(safe_project_name);

		/* Cleanup and return */
		if (feature_list_str)
			nfree(feature_list_str);
		if (feature_names)
		{
			int			i;

			for (i = 0; i < feature_name_count; i++)
			{
				if (feature_names[i])
				{
					char	   *ptr = (char *) feature_names[i];

					nfree(ptr);
				}
			}
			nfree(feature_names);
		}
		if (model_name)
		{
			MemoryContextSwitchTo(callcontext);
			nfree(model_name);
			model_name = NULL;
		}
		MemoryContextSwitchTo(oldcontext);
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);

		ereport(DEBUG1,
				(errmsg("neurondb_train: training completed successfully"),
				 errdetail("model_id=%d, algorithm=%s, project=%s", model_id, algorithm ? algorithm : "unknown", project_name ? project_name : "unknown")));

		PG_RETURN_INT32(model_id);
	}
	else
	{
cpu_fallback_training:
		ereport(DEBUG1,
				(errmsg("neurondb_train: attempting CPU fallback training"),
				 errdetail("algorithm=%s, gpu_train_result=%s", algorithm ? algorithm : "unknown", gpu_train_result ? "true" : "false")));

		/* GPU training failed or not attempted - handle based on compute mode */
		/* GPU mode: error if GPU was required but not available */
		if (!NDB_COMPUTE_MODE_IS_CPU() && NDB_REQUIRE_GPU() && !metal_requested_fallback)
		{
			nfree(feature_list_str);
			if (feature_names)
			{
				int			i;

				for (i = 0; i < feature_name_count; i++)
				{
					if (feature_names[i])
					{
						char	   *ptr = (char *) feature_names[i];

						nfree(ptr);
					}
				}
				nfree(feature_names);
			}
			if (model_name)
				nfree(model_name);
			ndb_spi_session_end(&spi_session);
		{
			char *gpu_error_msg = NULL;
			if (gpu_errmsg_ptr && strlen(gpu_errmsg_ptr) > 0)
			{
				gpu_error_msg = pstrdup(gpu_errmsg_ptr);
			}
			else
			{
				/* If no error message, provide a more specific fallback */
				gpu_error_msg = psprintf("ndb_gpu_try_train_model returned false for algorithm '%s' - check GPU availability and backend registration",
										 algorithm ? algorithm : "unknown");
			}
			if (gpu_errmsg_ptr)
				nfree(gpu_errmsg_ptr);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_PREFIX_TRAIN " GPU training failed - GPU mode requires GPU to be available"),
					 errdetail("Algorithm: %s, Project: %s, Table: %s. Error: %s",
							   algorithm ? algorithm : "unknown",
							   project_name ? project_name : "unknown",
							   table_name ? table_name : "unknown",
							   gpu_error_msg),
					 errhint("Check GPU hardware, drivers, and configuration. "
							 "Set compute_mode='auto' for automatic CPU fallback.")));
			if (gpu_error_msg)
				pfree(gpu_error_msg);
		}

		/* Cleanup after error (should not reach here, but included for safety) */
		MemoryContextSwitchTo(oldcontext);
		neurondb_cleanup(oldcontext, callcontext);
		nfree(project_name);
		nfree(algorithm);
		nfree(table_name);
		if (target_column)
			nfree(target_column);
		if (data_loaded)
		{
			if (feature_matrix)
				nfree(feature_matrix);
			if (label_vector)
				nfree(label_vector);
		}
		}

		if (data_loaded && callcontext != NULL && MemoryContextIsValid(callcontext))
		{
			MemoryContextSwitchTo(callcontext);
			if (feature_matrix)
			{
				nfree(feature_matrix);
				feature_matrix = NULL;
			}
			if (label_vector)
			{
				nfree(label_vector);
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
			feature_matrix = NULL;
			label_vector = NULL;
			data_loaded = false;
		}

	/* GPU training failed - only fall back to CPU training in AUTO mode or if Metal requested fallback */
	/* In GPU mode, error out unless Metal explicitly requested CPU fallback. In AUTO mode, continue to CPU. */
	if (NDB_COMPUTE_MODE_IS_AUTO() || NDB_COMPUTE_MODE_IS_CPU() || metal_requested_fallback)
	{
		/* AUTO/CPU mode, or Metal requested fallback - fall back to/perform CPU training */
		/* Explicitly reset model_id to 0 to ensure we don't return garbage from GPU result */
		model_id = 0;
		algo_enum = neurondb_algorithm_from_string(algorithm);

		/* For Metal unsupported algorithms in GPU mode, temporarily set compute_mode to AUTO */
		/* so that CPU training SQL functions will use CPU directly without trying GPU first */
		saved_compute_mode = neurondb_compute_mode;
		if (metal_requested_fallback && NDB_COMPUTE_MODE_IS_GPU())
		{
			neurondb_compute_mode = NDB_COMPUTE_MODE_AUTO;
		}

		/* Build SQL for CPU training */
		ndb_spi_stringinfo_free(spi_session, &sql);
		ndb_spi_stringinfo_init(spi_session, &sql);
	}
	else if (NDB_COMPUTE_MODE_IS_GPU())
		{
			/* GPU mode - error out if GPU training failed */
			char *gpu_error_msg = NULL;
			if (gpu_errmsg_ptr && strlen(gpu_errmsg_ptr) > 0)
			{
				gpu_error_msg = pstrdup(gpu_errmsg_ptr);
			}
			else
			{
				gpu_error_msg = psprintf("GPU training failed for algorithm '%s' - check GPU availability and error logs",
										 algorithm ? algorithm : "unknown");
			}
			
			/* Clean up before error */
			if (gpu_errmsg_ptr)
				nfree(gpu_errmsg_ptr);
			nfree(feature_list_str);
			if (feature_names)
			{
				int			i;
				for (i = 0; i < feature_name_count; i++)
				{
					if (feature_names[i])
					{
						char	   *ptr = (char *) feature_names[i];
						nfree(ptr);
					}
				}
				nfree(feature_names);
			}
			if (model_name)
				nfree(model_name);
			if (data_loaded)
			{
				if (feature_matrix)
					nfree(feature_matrix);
				if (label_vector)
					nfree(label_vector);
			}
			MemoryContextSwitchTo(oldcontext);
			neurondb_cleanup(oldcontext, callcontext);
			ndb_spi_session_end(&spi_session);
			nfree(project_name);
			nfree(algorithm);
			nfree(table_name);
			if (target_column)
				nfree(target_column);
			
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_PREFIX_TRAIN " GPU training failed - GPU mode does not allow CPU fallback"),
					 errdetail("Algorithm: %s, Project: %s, Table: %s. GPU Error: %s",
							   algorithm ? algorithm : "unknown",
							   project_name ? project_name : "unknown",
							   table_name ? table_name : "unknown",
							   gpu_error_msg),
					 errhint("Check GPU hardware, drivers, and configuration. "
							 "Set compute_mode='auto' for automatic CPU fallback, or 'cpu' for CPU-only training.")));
			if (gpu_error_msg)
				pfree(gpu_error_msg);
		}
	else
	{
		/* CPU mode - should not reach GPU training failure path */
		/* But it's OK - just proceed to CPU training below without cleanup */
		/* Don't free anything here - we need these variables for CPU training */
	}

	/* AUTO mode - execute CPU fallback code block */
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
			List *funcname = NULL;
			Oid			func_oid;
			Oid			argtypes[7];
			Datum		values[7];
			FmgrInfo	flinfo;
			Datum		result_datum;
			text *table_name_text_local = NULL;
			text *feature_col_text = NULL;
			text *label_col_text = NULL;
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
									 "COALESCE(" NDB_COL_METRICS ", '{}'::jsonb) || '{\"storage\":\"cpu\",\"training_backend\":0}'::jsonb "
									 "WHERE " NDB_COL_MODEL_ID " = %d",
									 model_id);
					ret = ndb_spi_execute(spi_session, sql.data, false, 0);
					if (ret != SPI_OK_UPDATE && ret != SPI_OK_UPDATE_RETURNING)
					{
						ndb_spi_stringinfo_free(spi_session, &sql);
						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg(NDB_ERR_PREFIX_TRAIN " failed to update model metrics (SPI returned %d)", ret)));
					}
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
												 "COALESCE(" NDB_COL_METRICS ", '{}'::jsonb) || '{\"storage\":\"cpu\",\"training_backend\":0}'::jsonb "
												 "WHERE " NDB_COL_MODEL_ID " = %d",
												 model_id);
								ret = ndb_spi_execute(spi_session, sql.data, false, 0);
								if (ret != SPI_OK_UPDATE && ret != SPI_OK_UPDATE_RETURNING)
								{
									ndb_spi_stringinfo_free(spi_session, &sql);
									ndb_spi_session_end(&spi_session);
									neurondb_cleanup(oldcontext, callcontext);
									ereport(ERROR,
											(errcode(ERRCODE_INTERNAL_ERROR),
											 errmsg(NDB_ERR_PREFIX_TRAIN " failed to update model metrics (SPI returned %d)", ret)));
								}
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
				else
				{
					/* SQL execution failed or returned no rows - allow fallback to next attempt */
				}
			}
		}
	else if (neurondb_build_training_sql(algo_enum, &sql, table_name, feature_list_str,
										 target_column, hyperparams, feature_names, feature_name_count))
	{
		/* Execute CPU training via SQL */
		ret = ndb_spi_execute(spi_session, sql.data, true, 0);


		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			int32		model_id_val = 0;	/* Initialize to 0 */

		/* Get model_id from result */
		if (ndb_spi_get_int32(spi_session, 0, 1, &model_id_val))
		{

			/* Validate model_id is positive - 0 or negative indicates failure */
				if (model_id_val <= 0)
				{
						char *sql_copy = sql.data ? pstrdup(sql.data) : NULL;
						ndb_spi_stringinfo_free(spi_session, &sql);
						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg(NDB_ERR_PREFIX_TRAIN " CPU training returned invalid model_id: %d", model_id_val),
								 errdetail("Algorithm: %s, Project: %s, Table: %s. The training function returned a non-positive model_id, which indicates the training or model registration failed.", algorithm, project_name, table_name),
								 errhint("CPU training function may have failed. Check logs for details. SQL executed: %s", sql_copy ? sql_copy : "(unavailable)")));
						if (sql_copy)
							pfree(sql_copy);
					}

		if (model_id_val > 0)
		{
			model_id = model_id_val;

		/* Verify model exists in catalog before updating */
		/* Don't free or reset sql - just truncate and reuse the buffer */
		sql.len = 0;
		sql.data[0] = '\0';
		appendStringInfo(&sql,
						 "SELECT COUNT(*) FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_MODEL_ID " = %d",
						 model_id);
		ret = ndb_spi_execute(spi_session, sql.data, true, 0);
				
				if (ret == SPI_OK_SELECT && SPI_processed > 0)
					{
						int32 count = 0;
						if (ndb_spi_get_int32(spi_session, 0, 1, &count) && count > 0)
						{
							/* Model exists - update metrics to ensure storage='cpu' is set */
							/* Don't free or reset sql - just truncate and reuse the buffer */
							sql.len = 0;
							sql.data[0] = '\0';
							appendStringInfo(&sql,
											 "UPDATE " NDB_FQ_ML_MODELS " SET " NDB_COL_METRICS " = "
											 "COALESCE(" NDB_COL_METRICS ", '{}'::jsonb) || '{\"storage\":\"cpu\",\"training_backend\":0}'::jsonb "
											 "WHERE " NDB_COL_MODEL_ID " = %d",
											 model_id);
							ret = ndb_spi_execute(spi_session, sql.data, false, 0);
							if (ret != SPI_OK_UPDATE && ret != SPI_OK_UPDATE_RETURNING)
								elog(WARNING, "neurondb:train: update model metrics returned %d", ret);
						}
						else
						{
							/* 
							 * Model not immediately visible - this can happen due to SPI session isolation.
							 * Since the CPU training function returned a valid model_id, we trust it.
							 * The model should be visible after the transaction commits.
							 * Log a warning but continue.
							 */
							elog(WARNING,
								 "neurondb:train: CPU training registered model_id %d but model not immediately visible in catalog (this may be due to SPI session isolation)",
								 model_id_val);
							/* Still try to update metrics in case the model becomes visible */
							sql.len = 0;
							sql.data[0] = '\0';
							appendStringInfo(&sql,
											 "UPDATE " NDB_FQ_ML_MODELS " SET " NDB_COL_METRICS " = "
											 "COALESCE(" NDB_COL_METRICS ", '{}'::jsonb) || '{\"storage\":\"cpu\",\"training_backend\":0}'::jsonb "
											 "WHERE " NDB_COL_MODEL_ID " = %d",
											 model_id);
							ret = ndb_spi_execute(spi_session, sql.data, false, 0);
							if (ret != SPI_OK_UPDATE && ret != SPI_OK_UPDATE_RETURNING)
								elog(WARNING, "neurondb:train: update model metrics (fallback) returned %d", ret);
						}
					}
						else
						{
							/* Could not verify model existence - treat as error */
							ndb_spi_stringinfo_free(spi_session, &sql);
							ndb_spi_session_end(&spi_session);
							neurondb_cleanup(oldcontext, callcontext);
							ereport(ERROR,
									(errcode(ERRCODE_INTERNAL_ERROR),
									 errmsg(NDB_ERR_PREFIX_TRAIN " CPU training returned model_id %d but could not verify model in catalog", model_id_val),
									 errdetail("Algorithm: %s, Project: %s, Table: %s. SPI return code: %d", algorithm, project_name, table_name, ret),
									 errhint("This may indicate a transaction issue. Check logs for details.")));
						}
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
				nfree(feature_list_str);
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
		else if (algo_enum == ML_ALGO_TITANS_LLM)
		{
			/* Titans LLM - call special C function */
			ereport(DEBUG1,
					(errmsg("neurondb_train: calling titans_llm training function")));

			/* Call neurondb_train_titans_llm C function */
			{
				List *funcname = NULL;
				Oid			func_oid;
				Oid			argtypes[5];
				Datum		values[5];
				FmgrInfo	flinfo;
				Datum		result_datum;
				text *inner_project_name_text = NULL;
				text *inner_table_name_text = NULL;
				text *target_col_text = NULL;
				ArrayType *feature_cols_array = NULL;
				Jsonb *hyperparams_copy = NULL;

				/* Build arguments */
				inner_project_name_text = cstring_to_text(project_name);
				inner_table_name_text = cstring_to_text(table_name);
				target_col_text = target_column ? cstring_to_text(target_column) : NULL;

				/* Convert feature_names array to PostgreSQL array */
				if (feature_names != NULL && feature_name_count > 0)
				{
					Datum *elems = NULL;
					bool *nulls = NULL;
					int i;

					nalloc(elems, Datum, feature_name_count);
					nalloc(nulls, bool, feature_name_count);

					for (i = 0; i < feature_name_count; i++)
					{
						if (feature_names[i] != NULL)
						{
							elems[i] = PointerGetDatum(cstring_to_text(feature_names[i]));
							nulls[i] = false;
						}
						else
						{
							elems[i] = (Datum) 0;
							nulls[i] = true;
						}
					}

					feature_cols_array = construct_array(elems, feature_name_count, TEXTOID, -1, false, 'i');

					for (i = 0; i < feature_name_count; i++)
					{
						if (!nulls[i] && elems[i] != 0)
							pfree(DatumGetPointer(elems[i]));
					}
					nfree(elems);
					nfree(nulls);
				}
				else
					feature_cols_array = construct_empty_array(TEXTOID);

				if (hyperparams != NULL)
					hyperparams_copy = (Jsonb *) PG_DETOAST_DATUM_COPY(PointerGetDatum(hyperparams));
				else
					hyperparams_copy = ndb_jsonb_in_cstring("{}");

				funcname = list_make1(makeString("neurondb_train_titans_llm"));
				argtypes[0] = TEXTOID;
				argtypes[1] = TEXTOID;
				argtypes[2] = TEXTOID;
				argtypes[3] = get_array_type(TEXTOID);
				argtypes[4] = JSONBOID;

				func_oid = LookupFuncName(funcname, 5, argtypes, false);
				list_free(funcname);

				if (OidIsValid(func_oid))
				{
					fmgr_info(func_oid, &flinfo);
					values[0] = PointerGetDatum(inner_project_name_text);
					values[1] = PointerGetDatum(inner_table_name_text);
					values[2] = target_col_text ? PointerGetDatum(target_col_text) : (Datum) 0;
					values[3] = PointerGetDatum(feature_cols_array);
					values[4] = PointerGetDatum(hyperparams_copy);

					result_datum = FunctionCall5(&flinfo,
											   values[0], values[1], values[2],
											   values[3], values[4]);
					model_id = DatumGetInt32(result_datum);

					pfree(inner_project_name_text);
					pfree(inner_table_name_text);
					if (target_col_text)
						pfree(target_col_text);
					if (feature_cols_array)
						pfree(feature_cols_array);
					if (hyperparams_copy)
						pfree(hyperparams_copy);

					if (model_id > 0)
					{
						ndb_spi_stringinfo_free(spi_session, &sql);
						ndb_spi_stringinfo_init(spi_session, &sql);
						appendStringInfo(&sql,
										 "UPDATE " NDB_FQ_ML_MODELS " SET " NDB_COL_METRICS " = "
										 "COALESCE(" NDB_COL_METRICS ", '{}'::jsonb) || '{\"storage\":\"cpu\",\"training_backend\":0}'::jsonb "
										 "WHERE " NDB_COL_MODEL_ID " = %d",
										 model_id);
						ret = ndb_spi_execute(spi_session, sql.data, false, 0);
						if (ret != SPI_OK_UPDATE && ret != SPI_OK_UPDATE_RETURNING)
						{
							ndb_spi_stringinfo_free(spi_session, &sql);
							ndb_spi_session_end(&spi_session);
							neurondb_cleanup(oldcontext, callcontext);
							ereport(ERROR,
									(errcode(ERRCODE_INTERNAL_ERROR),
									 errmsg(NDB_ERR_PREFIX_TRAIN " failed to update titans_llm model metrics (SPI returned %d)", ret)));
						}
					}
					else
					{
						ndb_spi_stringinfo_free(spi_session, &sql);
						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg(NDB_ERR_PREFIX_TRAIN " titans_llm training returned invalid model_id: %d", model_id),
								 errdetail("Algorithm: %s, Project: %s, Table: %s", algorithm, project_name, table_name),
								 errhint("Check training logs for details.")));
					}
				}
				else
				{
					pfree(inner_project_name_text);
					pfree(inner_table_name_text);
					if (target_col_text)
						pfree(target_col_text);
					if (feature_cols_array)
						pfree(feature_cols_array);
					if (hyperparams_copy)
						pfree(hyperparams_copy);
					nfree(feature_list_str);
					ndb_spi_stringinfo_free(spi_session, &sql);
					ndb_spi_session_end(&spi_session);
					neurondb_cleanup(oldcontext, callcontext);
					ereport(ERROR,
							(errcode(ERRCODE_UNDEFINED_FUNCTION),
							 errmsg(NDB_ERR_PREFIX_TRAIN " titans_llm training function not found"),
							 errdetail("Algorithm: %s, Project: %s, Table: %s", algorithm, project_name, table_name),
							 errhint("The neurondb_train_titans_llm function must be implemented.")));
				}
			}
		}
		else if (algo_enum == ML_ALGO_TRANSFORMER_LLM)
		{
			/* Transformer LLM - call special C function */
			ereport(DEBUG1,
					(errmsg("neurondb_train: calling transformer_llm training function")));
			
			/* Call neurondb_train_transformer_llm C function */
			{
				List *funcname = NULL;
				Oid			func_oid;
				Oid			argtypes[5];
				Datum		values[5];
				FmgrInfo	flinfo;
				Datum		result_datum;
				text *inner_project_name_text = NULL;
				text *inner_table_name_text = NULL;
				text *target_col_text = NULL;
				ArrayType *feature_cols_array = NULL;
				Jsonb *hyperparams_copy = NULL;
				
				/* Build arguments */
				inner_project_name_text = cstring_to_text(project_name);
				inner_table_name_text = cstring_to_text(table_name);
				target_col_text = target_column ? cstring_to_text(target_column) : NULL;
				
				/* Convert feature_names array to PostgreSQL array */
				if (feature_names != NULL && feature_name_count > 0)
				{
					Datum *elems = NULL;
					bool *nulls = NULL;
					int i;
					
					nalloc(elems, Datum, feature_name_count);
					nalloc(nulls, bool, feature_name_count);
					
					for (i = 0; i < feature_name_count; i++)
					{
						if (feature_names[i] != NULL)
						{
							elems[i] = PointerGetDatum(cstring_to_text(feature_names[i]));
							nulls[i] = false;
						}
						else
						{
							elems[i] = (Datum) 0;
							nulls[i] = true;
						}
					}
					
					feature_cols_array = construct_array(elems, feature_name_count, TEXTOID, -1, false, 'i');
					
					/* Free temporary elements */
					for (i = 0; i < feature_name_count; i++)
					{
						if (!nulls[i] && elems[i] != 0)
							pfree(DatumGetPointer(elems[i]));
					}
					nfree(elems);
					nfree(nulls);
				}
				else
				{
					/* Empty array */
					feature_cols_array = construct_empty_array(TEXTOID);
				}
				
				/* Copy hyperparams */
				if (hyperparams != NULL)
				{
					hyperparams_copy = (Jsonb *) PG_DETOAST_DATUM_COPY(PointerGetDatum(hyperparams));
				}
				else
				{
					hyperparams_copy = ndb_jsonb_in_cstring("{}");
				}
				
				/* Lookup C function */
				funcname = list_make1(makeString("neurondb_train_transformer_llm"));
				argtypes[0] = TEXTOID;	/* project_name */
				argtypes[1] = TEXTOID;	/* table_name */
				argtypes[2] = TEXTOID;	/* target_column */
				/* Get text array type OID */
				argtypes[3] = get_array_type(TEXTOID); /* feature_columns text[] */
				argtypes[4] = JSONBOID;	/* hyperparams */
				
				func_oid = LookupFuncName(funcname, 5, argtypes, false);
				list_free(funcname);
				
				if (OidIsValid(func_oid))
				{
					fmgr_info(func_oid, &flinfo);
					
					values[0] = PointerGetDatum(inner_project_name_text);
					values[1] = PointerGetDatum(inner_table_name_text);
					values[2] = target_col_text ? PointerGetDatum(target_col_text) : (Datum) 0;
					values[3] = PointerGetDatum(feature_cols_array);
					values[4] = PointerGetDatum(hyperparams_copy);
					
					result_datum = FunctionCall5(&flinfo,
												  values[0], values[1], values[2],
												  values[3], values[4]);
					
					model_id = DatumGetInt32(result_datum);
					
					/* Cleanup */
					pfree(project_name_text);
					pfree(table_name_text);
					if (target_col_text)
						pfree(target_col_text);
					if (feature_cols_array)
						pfree(feature_cols_array);
					if (hyperparams_copy)
						pfree(hyperparams_copy);
					
					if (model_id > 0)
					{
						/* Update metrics */
						ndb_spi_stringinfo_free(spi_session, &sql);
						ndb_spi_stringinfo_init(spi_session, &sql);
						appendStringInfo(&sql,
										 "UPDATE " NDB_FQ_ML_MODELS " SET " NDB_COL_METRICS " = "
										 "COALESCE(" NDB_COL_METRICS ", '{}'::jsonb) || '{\"storage\":\"cpu\",\"training_backend\":0}'::jsonb "
										 "WHERE " NDB_COL_MODEL_ID " = %d",
										 model_id);
						ret = ndb_spi_execute(spi_session, sql.data, false, 0);
						if (ret != SPI_OK_UPDATE && ret != SPI_OK_UPDATE_RETURNING)
						{
							ndb_spi_stringinfo_free(spi_session, &sql);
							ndb_spi_session_end(&spi_session);
							neurondb_cleanup(oldcontext, callcontext);
							ereport(ERROR,
									(errcode(ERRCODE_INTERNAL_ERROR),
									 errmsg(NDB_ERR_PREFIX_TRAIN " failed to update transformer_llm model metrics (SPI returned %d)", ret)));
						}
					}
					else
					{
						ndb_spi_stringinfo_free(spi_session, &sql);
						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg(NDB_ERR_PREFIX_TRAIN " transformer_llm training returned invalid model_id: %d", model_id),
								 errdetail("Algorithm: %s, Project: %s, Table: %s", algorithm, project_name, table_name),
								 errhint("Check training logs for details.")));
					}
				}
				else
				{
					/* Function not found */
					pfree(project_name_text);
					pfree(table_name_text);
					if (target_col_text)
						pfree(target_col_text);
					if (feature_cols_array)
						pfree(feature_cols_array);
					if (hyperparams_copy)
						pfree(hyperparams_copy);
					
					nfree(feature_list_str);
					ndb_spi_stringinfo_free(spi_session, &sql);
					ndb_spi_session_end(&spi_session);
					neurondb_cleanup(oldcontext, callcontext);
					ereport(ERROR,
							(errcode(ERRCODE_UNDEFINED_FUNCTION),
							 errmsg(NDB_ERR_PREFIX_TRAIN " transformer_llm training function not found"),
							 errdetail("Algorithm: %s, Project: %s, Table: %s", algorithm, project_name, table_name),
							 errhint("The neurondb_train_transformer_llm function must be implemented.")));
				}
			}
		}
		else
		{
			/*
			 * Algorithm doesn't have CPU training SQL - this shouldn't happen
			 * for most algorithms
			 */
			nfree(feature_list_str);
			ndb_spi_stringinfo_free(spi_session, &sql);
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg(NDB_ERR_PREFIX_TRAIN " algorithm '%s' does not support CPU training fallback", algorithm),
					 errdetail("GPU training failed and no CPU training implementation available. Algorithm: %s, Project: %s, Table: %s", algorithm, project_name, table_name),
					 errhint("Try a different algorithm or ensure GPU is properly configured.")));
		}
		
		/* Restore compute_mode if we temporarily changed it for Metal unsupported algorithms */
		if (saved_compute_mode >= 0 && saved_compute_mode != neurondb_compute_mode)
		{
			neurondb_compute_mode = saved_compute_mode;
		}
		
		/* Ensure model_id was set by CPU training - if not, error out */
		if (model_id <= 0)
		{
			nfree(feature_list_str);
			ndb_spi_stringinfo_free(spi_session, &sql);
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg(NDB_ERR_PREFIX_TRAIN " CPU training fallback failed to return valid model_id"),
					 errdetail("Algorithm: %s, Project: %s, Table: %s, model_id: %d", algorithm, project_name, table_name, model_id),
					 errhint("CPU training may have failed. Check logs for details.")));
		}
		/* End of AUTO mode CPU fallback block */
		PG_RETURN_INT32(model_id);
	}
	/* GPU-only mode and GPU failed: should have errored above; satisfy compiler */
	PG_RETURN_INT32(0);
}

/* ----------
 * neurondb_predict
 *		Generate predictions using a trained ML model.
 *
 * This function provides a unified interface for making predictions with
 * trained models. It loads model metadata, dispatches to algorithm-specific
 * prediction functions, and returns the prediction result.
 *
 * SQL Function Signature:
 *		neurondb.predict(model_id INTEGER, features FLOAT8[]) RETURNS FLOAT8
 *
 * Parameters:
 *		model_id - ID of the trained model (from neurondb.train)
 *		features - Array of feature values (must match training feature count)
 *
 * Returns:
 *		FLOAT8: Prediction value (class probability, regression value, etc.)
 *		Raises ERROR if model not found or features invalid
 *
 * Algorithm Support:
 *		Supports all algorithms that have prediction functions registered.
 *		The function looks up the algorithm from the model catalog and
 *		dispatches to the appropriate prediction function.
 *
 * Memory Management:
 *		- Creates dedicated memory context for function execution
 *		- Manages SPI session for catalog queries
 *		- All memory freed via neurondb_cleanup() on exit
 *
 * Error Handling:
 *		- Validates model_id exists in catalog
 *		- Validates features array is 1-dimensional
 *		- Validates feature count matches model expectations
 *		- Comprehensive error messages with hints
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Feature validation rules, error messages
 *		- CANNOT MODIFY: Memory context lifecycle, SPI session management
 *		- BREAKS IF: Model catalog schema changes without updating queries
 * ----------
 */
Datum
neurondb_predict(PG_FUNCTION_ARGS)
{
	int32		model_id;
	ArrayType *features_array = NULL;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	StringInfoData features_str;
	int			ret;

	char *algorithm = NULL;
	float8		prediction = 0.0;
	int			ndims,
				nelems,
				i;
	int *dims = NULL;
	float8 *features = NULL;

	float8 *features_float = NULL;	/* Allocated if conversion needed */
	NdbSpiSession *spi_session = NULL;

	ereport(DEBUG1,
			(errmsg("neurondb_predict: starting prediction"),
			 errdetail("nargs=%d", PG_NARGS())));

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

	spi_session = ndb_spi_session_begin(callcontext, false);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT " NDB_COL_ALGORITHM "::text FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_MODEL_ID " = %d",
					 model_id);
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
	if (nelems <= 0 || nelems > neurondb_ml_max_feature_elements)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_PREDICT " invalid feature count %d (expected 1-%d)", nelems, neurondb_ml_max_feature_elements),
				 errdetail("Model ID: %d, Algorithm: '%s', Feature count: %d", model_id, algorithm, nelems),
				 errhint("Provide a feature array with between 1 and %d elements matching the model's expected feature dimension.", neurondb_ml_max_feature_elements)));
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
		nalloc(features_float, float8, nelems);

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
		bytea *model_data = NULL;
		Jsonb *metrics = NULL;
		bool		is_gpu = false;
		int			nb_class = 0;
		double		nb_probability = 0.0;

		float *features_float = NULL;
		int			feature_dim = nelems;

		char *errstr = NULL;
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
				nfree(metrics);

			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg(NDB_ERR_PREFIX_PREDICT " naive_bayes model %d has no model data (model not trained)", model_id),
					 errdetail("The model exists in catalog but model_data is NULL, indicating training was not completed successfully"),
					 errhint("Train the model first using neurondb.train() before attempting prediction. Model ID: %d", model_id)));
		}

		is_gpu = ml_metrics_is_gpu(metrics);

		nalloc(features_float, float, feature_dim);

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


			if (is_gpu)
			{
				/* Model was trained on GPU - must use GPU prediction */
				if (!gpu_currently_enabled)
				{
					nfree(features_float);

					if (model_data)
						nfree(model_data);

					if (metrics)
						nfree(metrics);

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
						nfree(features_float);

						if (model_data)
							nfree(model_data);

						if (metrics)
							nfree(metrics);

						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						PG_RETURN_FLOAT8(prediction);
					}
					if (errstr)
						nfree(errstr);
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
		nfree(features_float);

		if (model_data)
			nfree(model_data);

		if (metrics)
			nfree(metrics);
	}
	else if (strcmp(algorithm, NDB_ALGO_RIDGE) == 0 || strcmp(algorithm, NDB_ALGO_LASSO) == 0)
		appendStringInfo(&sql, "SELECT predict_regularized_regression(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, NDB_ALGO_KNN) == 0 || strcmp(algorithm, NDB_ALGO_KNN_CLASSIFIER) == 0 || strcmp(algorithm, NDB_ALGO_KNN_REGRESSOR) == 0)
	{
		bytea *model_data = NULL;
		Jsonb *metrics = NULL;
		bool		is_gpu = false;
		double		knn_prediction = 0.0;

		float *features_float = NULL;
		int			feature_dim = nelems;

		char *errstr = NULL;
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
				nfree(metrics);

			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("KNN model %d has no model data (model not trained)", model_id),
					 errhint("KNN training must be completed before prediction. The model may have been created without actual training.")));
		}

		if (metrics != NULL)
		{
			JsonbIterator *it = NULL;
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
						nfree(key);
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
		nalloc(features_float, float, feature_dim);

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


			if (is_gpu)
			{
				/* Model was trained on GPU - must use GPU prediction */
				if (!gpu_currently_enabled)
				{
					nfree(features_float);

					if (model_data)
						nfree(model_data);

					if (metrics)
						nfree(metrics);

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
						nfree(features_float);

						if (model_data)
							nfree(model_data);

						if (metrics)
							nfree(metrics);

						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						PG_RETURN_FLOAT8(prediction);
					}
					if (errstr)
						nfree(errstr);
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
		nfree(features_float);

		if (model_data)
			nfree(model_data);

		if (metrics)
			nfree(metrics);
	}
	else if (strcmp(algorithm, "kmeans") == 0)
		appendStringInfo(&sql, "SELECT predict_kmeans_model_id(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "minibatch_kmeans") == 0)
		appendStringInfo(&sql, "SELECT predict_kmeans_model_id(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "hierarchical") == 0)
		appendStringInfo(&sql, "SELECT predict_hierarchical_cluster(%d, %s)", model_id, features_str.data);
	else if (strcmp(algorithm, "xgboost") == 0)
	{
		bytea *model_data = NULL;
		Jsonb *metrics = NULL;
		bool		is_gpu = false;
		double		xgboost_prediction = 0.0;

		float *features_float = NULL;
		int			feature_dim = nelems;

		char *errstr = NULL;
		int			rc;

		if (!ml_catalog_fetch_model_payload(model_id, &model_data, NULL, &metrics))
		{
			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("XGBoost model %d not found", model_id)));
		}

		if (model_data == NULL)
		{
			if (metrics)
				nfree(metrics);

			ndb_spi_session_end(&spi_session);
			neurondb_cleanup(oldcontext, callcontext);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("XGBoost model %d has no model data (model not trained)", model_id),
					 errhint("XGBoost training must be completed before prediction. The model may have been created without actual training.")));
		}

		if (metrics != NULL)
		{
			JsonbIterator *it = NULL;
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
						/* Also check storage field as fallback for XGBoost */
						else if (strcmp(key, "storage") == 0 && v.type == jbvString)
						{
							char	   *storage_val = pnstrdup(v.val.string.val, v.val.string.len);

							if (strcmp(storage_val, "gpu") == 0)
								is_gpu = true;
							nfree(storage_val);
						}
						nfree(key);
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
		nalloc(features_float, float, feature_dim);

		for (i = 0; i < feature_dim; i++)
			features_float[i] = (float) features[i];

		/*
		 * Use model's training backend (from catalog) regardless of current
		 * GPU state
		 */
		{
			const ndb_gpu_backend *backend;
			bool		gpu_currently_enabled;

			/* Initialize GPU if needed (lazy initialization) before checking availability */
			if (NDB_SHOULD_TRY_GPU())
			{
				ndb_gpu_init_if_needed();
			}

			backend = ndb_gpu_get_active_backend();
			gpu_currently_enabled = (backend != NULL && neurondb_gpu_is_available());


			if (is_gpu)
			{
				/* Model was trained on GPU - must use GPU prediction */
				if (!gpu_currently_enabled)
				{
					nfree(features_float);

					if (model_data)
						nfree(model_data);

					if (metrics)
						nfree(metrics);

					ndb_spi_session_end(&spi_session);
					neurondb_cleanup(oldcontext, callcontext);
					ereport(ERROR,
							(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
							 errmsg("XGBoost: model %d was trained on GPU but GPU is not currently available", model_id),
							 errhint("Enable GPU with: SET neurondb.compute_mode = 1; or SET neurondb.compute_mode = 2; for auto mode")));
				}

				/* Use GPU prediction */
				if (backend && backend->xgboost_predict && model_data != NULL)
				{
					rc = backend->xgboost_predict(model_data, features_float, feature_dim, &xgboost_prediction, &errstr);
					if (rc == 0)
					{
						prediction = xgboost_prediction;
						nfree(features_float);

						if (model_data)
							nfree(model_data);

						if (metrics)
							nfree(metrics);

						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						PG_RETURN_FLOAT8(prediction);
					}
					/* GPU prediction failed - error out since model was GPU-trained */
					{
						char	   *error_msg = NULL;

						if (errstr && errstr[0] != '\0')
							error_msg = pstrdup(errstr);
						else
							error_msg = pstrdup("GPU prediction failed with unknown error");

						nfree(features_float);
						if (model_data)
							nfree(model_data);
						if (metrics)
							nfree(metrics);
						if (errstr)
							nfree(errstr);
						ndb_spi_session_end(&spi_session);
						neurondb_cleanup(oldcontext, callcontext);
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("XGBoost: GPU prediction failed for GPU-trained model %d", model_id),
								 errdetail("Error: %s", error_msg),
								 errhint("GPU backend may not be properly initialized or model data may be corrupted.")));
						if (error_msg)
							pfree(error_msg);
					}
				}
				else
				{
					/* GPU backend or predict function not available */
					nfree(features_float);
					if (model_data)
						nfree(model_data);
					if (metrics)
						nfree(metrics);
					ndb_spi_session_end(&spi_session);
					neurondb_cleanup(oldcontext, callcontext);
					ereport(ERROR,
							(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
							 errmsg("XGBoost: GPU prediction not available for GPU-trained model %d", model_id),
							 errdetail("Backend: %p, predict function: %p, model_data: %p", 
									   (void *) backend, 
									   backend ? (void *) backend->xgboost_predict : NULL,
									   (void *) model_data),
							 errhint("GPU backend may not be properly initialized.")));
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
		/* Fall through to CPU prediction */
		appendStringInfo(&sql, "SELECT predict_xgboost(%d, %s)", model_id, features_str.data);
		nfree(features_float);

		if (model_data)
			nfree(model_data);

		if (metrics)
			nfree(metrics);
	}
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

	ereport(DEBUG1,
			(errmsg("neurondb_predict: prediction completed"),
			 errdetail("model_id=%d, algorithm=%s, prediction=%.6f", model_id, algorithm ? algorithm : "unknown", prediction)));

	PG_RETURN_FLOAT8(prediction);
}

/* ----------
 * neurondb_deploy
 *		Deploy a trained model for production use.
 *
 * This function creates a deployment record for a trained model, making it
 * available for production predictions. Deployments track model versions,
 * deployment strategies, and status.
 *
 * SQL Function Signature:
 *		neurondb.deploy(model_id INTEGER, [strategy TEXT]) RETURNS INTEGER
 *
 * Parameters:
 *		model_id - ID of the trained model to deploy
 *		strategy - Deployment strategy (optional, defaults to 'replace')
 *
 * Returns:
 *		INTEGER: deployment_id of the created deployment
 *		Raises ERROR if model not found or deployment creation fails
 *
 * Deployment Strategies:
 *		- 'replace': Replace existing active deployment (default)
 *		- Other strategies may be supported in future versions
 *
 * Side Effects:
 *		- Creates deployment table if it doesn't exist
 *		- Inserts deployment record in neurondb.ml_deployments
 *		- Generates unique deployment name with timestamp
 *
 * Memory Management:
 *		- Creates dedicated memory context for function execution
 *		- Manages SPI session for catalog operations
 *		- All memory freed via neurondb_cleanup() on exit
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Default strategy, deployment name format
 *		- CANNOT MODIFY: Memory context lifecycle, SPI session management
 *		- BREAKS IF: Deployment table schema changes without updating SQL
 * ----------
 */
Datum
neurondb_deploy(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *strategy_text = NULL;
	char *strategy = NULL;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;

	int			ret;
	int			deployment_id = 0;
	NdbSpiSession *spi_session = NULL;

	ereport(DEBUG1,
			(errmsg("neurondb_deploy: starting deployment"),
			 errdetail("nargs=%d", PG_NARGS())));

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

	spi_session = ndb_spi_session_begin(callcontext, false);

	/* PostgreSQL 18 B-tree deduplication bug workaround: create sequence separately */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfoString(&sql, "CREATE SEQUENCE IF NOT EXISTS neurondb_ml_deployments_id_seq");
	(void) ndb_spi_execute(spi_session, sql.data, false, 0);
	NDB_CHECK_SPI_TUPTABLE();
	ndb_spi_stringinfo_free(spi_session, &sql);

	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfoString(&sql,
						   "CREATE TABLE IF NOT EXISTS " NDB_FQ_ML_DEPLOYMENTS " ("
						   "deployment_id INTEGER DEFAULT nextval('neurondb_ml_deployments_id_seq') PRIMARY KEY, "
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
	{
		char	   *deploy_name = psprintf("deploy_%d_%ld", model_id, (long) time(NULL));
		char	   *deploy_name_quoted = neurondb_quote_literal_cstr(deploy_name);

		appendStringInfo(&sql,
						 "INSERT INTO " NDB_FQ_ML_DEPLOYMENTS " (" NDB_COL_MODEL_ID ", deployment_name, strategy, " NDB_COL_STATUS ", deployed_at) "
						 "VALUES (%d, %s, %s, '" NDB_DEPLOYMENT_ACTIVE "', CURRENT_TIMESTAMP) RETURNING deployment_id",
						 model_id,
						 deploy_name_quoted,
						 neurondb_quote_literal_cstr(strategy));
		pfree(deploy_name);
		pfree(deploy_name_quoted);
	}

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

	ereport(DEBUG1,
			(errmsg("neurondb_deploy: deployment completed successfully"),
			 errdetail("deployment_id=%d, model_id=%d, strategy=%s", deployment_id, model_id, strategy ? strategy : "NULL")));

	PG_RETURN_INT32(deployment_id);
}

/* ----------
 * neurondb_load_model
 *		Load an external ML model and register its metadata in the catalog.
 *
 * This function allows importing models trained outside NeuronDB (e.g., with
 * scikit-learn, TensorFlow, PyTorch, ONNX) and registering them in the
 * NeuronDB catalog for use with the unified prediction API.
 *
 * SQL Function Signature:
 *		neurondb.load_model(project_name TEXT, model_path TEXT, model_format TEXT)
 *		RETURNS INTEGER
 *
 * Parameters:
 *		project_name - Name of ML project to register model under
 *		model_path - File system path to the model file
 *		model_format - Model format: 'onnx', 'tensorflow', 'pytorch', or 'sklearn'
 *
 * Returns:
 *		INTEGER: model_id of the registered model
 *		Raises ERROR if format unsupported or registration fails
 *
 * Supported Formats:
 *		- 'onnx': ONNX format models
 *		- 'tensorflow': TensorFlow SavedModel or H5 format
 *		- 'pytorch': PyTorch model files
 *		- 'sklearn': scikit-learn pickle files
 *
 * Side Effects:
 *		- Creates/updates project in neurondb.ml_projects
 *		- Registers model in neurondb.ml_models with external model metadata
 *		- Uses advisory locks for concurrent safety
 *
 * Memory Management:
 *		- Creates dedicated memory context for function execution
 *		- Manages SPI session for catalog operations
 *		- All memory freed via neurondb_cleanup() on exit
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Supported formats, validation rules
 *		- CANNOT MODIFY: Memory context lifecycle, SPI session management
 *		- BREAKS IF: Model catalog schema changes without updating SQL
 * ----------
 */
Datum
neurondb_load_model(PG_FUNCTION_ARGS)
{
	text *project_name_text = NULL;
	text *model_path_text = NULL;
	text *model_format_text = NULL;
	MemoryContext callcontext;
	MemoryContext oldcontext;

	StringInfoData sql;
	char *project_name = NULL;
	char *model_path = NULL;
	char *model_format = NULL;
	int			ret;
	int			model_id = 0;
	int			project_id = 0;
	NdbSpiSession *spi_session = NULL;

	ereport(DEBUG1,
			(errmsg("neurondb_load_model: starting model load"),
			 errdetail("nargs=%d", PG_NARGS())));

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

	callcontext = AllocSetContextCreate(CurrentMemoryContext, "neurondb_load_model",
										ALLOCSET_DEFAULT_SIZES);
	oldcontext = MemoryContextSwitchTo(callcontext);

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
	{
		char	   *ext_name = psprintf("%s_%ld", model_format, (long) time(NULL));
		char	   *ext_name_quoted = neurondb_quote_literal_cstr(ext_name);

		appendStringInfo(&sql,
						 "WITH next_version AS (SELECT COALESCE(MAX(" NDB_COL_VERSION "), 0) + 1 AS v FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_PROJECT_ID " = %d) "
						 "INSERT INTO " NDB_FQ_ML_MODELS " (" NDB_COL_PROJECT_ID ", " NDB_COL_VERSION ", " NDB_COL_MODEL_NAME ", " NDB_COL_ALGORITHM ", " NDB_COL_TRAINING_TABLE ", " NDB_COL_TRAINING_COLUMN ", " NDB_COL_STATUS ", metadata) "
						 "SELECT %d, v, %s, %s, NULL, NULL, 'external', '{\"model_path\": %s, \"model_format\": %s}'::jsonb FROM next_version RETURNING model_id",
						 project_id,
						 project_id,
						 ext_name_quoted,
						 neurondb_quote_literal_cstr(model_format),
						 neurondb_quote_literal_cstr(model_path),
						 neurondb_quote_literal_cstr(model_format));
		pfree(ext_name);
		pfree(ext_name_quoted);
	}

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

	ereport(DEBUG1,
			(errmsg("neurondb_load_model: model load completed successfully"),
			 errdetail("model_id=%d, project_name=%s, model_format=%s", model_id, project_name ? project_name : "NULL", model_format ? model_format : "NULL")));

	PG_RETURN_INT32(model_id);
}

/* ----------
 * neurondb_evaluate
 *		Evaluate a trained model's performance on test data.
 *
 * This function provides a unified interface for model evaluation. It loads
 * model metadata, validates inputs, and dispatches to algorithm-specific
 * evaluation functions that compute metrics (accuracy, precision, recall,
 * F1-score, MSE, R², etc.).
 *
 * SQL Function Signature:
 *		neurondb.evaluate(model_id INTEGER, table_name TEXT, feature_col TEXT,
 *						label_col TEXT) RETURNS JSONB
 *
 * Parameters:
 *		model_id - ID of the trained model to evaluate
 *		table_name - Name of table containing test data
 *		feature_col - Name of feature column in test table
 *		label_col - Name of label column (NULL for unsupervised algorithms)
 *
 * Returns:
 *		JSONB: Evaluation metrics (algorithm-specific)
 *		Returns NULL if evaluation fails (instead of raising error)
 *		Raises ERROR if model not found or parameters invalid
 *
 * Evaluation Metrics:
 *		Classification: accuracy, precision, recall, F1-score, confusion matrix
 *		Regression: MSE, RMSE, MAE, R²
 *		Clustering: silhouette score, inertia, cluster assignments
 *
 * Algorithm Support:
 *		Supports all algorithms that have evaluation functions registered.
 *		The function looks up the algorithm from the model catalog and
 *		dispatches to the appropriate evaluation function.
 *
 * Error Handling:
 *		- Comprehensive input validation
 *		- Validates model exists in catalog
 *		- Validates label_col required for supervised algorithms
 *		- Returns NULL (not error) if evaluation query fails
 *		- Wraps evaluation in PG_TRY/PG_CATCH for safety
 *
 * Memory Management:
 *		- Creates dedicated memory context for function execution
 *		- Manages SPI session for catalog queries and evaluation
 *		- All memory freed via neurondb_cleanup() on exit
 *		- JSONB result copied to caller's context before cleanup
 *
 * CHANGE NOTES:
 *		- CAN MODIFY: Evaluation metrics returned, error handling strategy
 *		- CANNOT MODIFY: Memory context lifecycle, SPI session management
 *		- CANNOT MODIFY: PG_TRY/PG_CATCH structure (required for safety)
 *		- BREAKS IF: Model catalog schema changes without updating queries
 *		- BREAKS IF: Algorithm-specific evaluation function signatures change
 * ----------
 */
Datum
neurondb_evaluate(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *table_name_text = NULL;
	text *feature_col_text = NULL;
	text *label_col_text = NULL;
	MemoryContext callcontext;
	MemoryContext oldcontext;
	StringInfoData sql;
	int			ret;
	bool		isnull = false;
	char *algorithm = NULL;
	char *table_name = NULL;
	char *feature_col = NULL;
	char *label_col = NULL;
	Jsonb *result = NULL;
	NdbSpiSession *spi_session = NULL;

	ereport(DEBUG1,
			(errmsg("neurondb_evaluate: starting model evaluation"),
			 errdetail("nargs=%d", PG_NARGS())));

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

	spi_session = ndb_spi_session_begin(callcontext, false);

	/* Get algorithm from model_id */
	ndb_spi_stringinfo_init(spi_session, &sql);
	appendStringInfo(&sql,
					 "SELECT " NDB_COL_ALGORITHM "::text FROM " NDB_FQ_ML_MODELS " WHERE " NDB_COL_MODEL_ID " = %d",
					 model_id);
	ret = ndb_spi_execute(spi_session, sql.data, true, 0);
	NDB_CHECK_SPI_TUPTABLE();
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		ndb_spi_session_end(&spi_session);
		neurondb_cleanup(oldcontext, callcontext);
		ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE), errmsg("Model not found: %d", model_id)));
	}
	{
		char *temp_algorithm = NULL;

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
			nfree(algo_text);
			isnull = false;
		}
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
		nfree(temp_algorithm);
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
	
	/* Validate strings are not empty */
	if (table_name == NULL || strlen(table_name) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " table_name cannot be empty or NULL")));
	if (feature_col == NULL || strlen(feature_col) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " feature_col cannot be empty or NULL")));
	if (label_col != NULL && strlen(label_col) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg(NDB_ERR_PREFIX_EVALUATE " label_col cannot be empty")));

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

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, NDB_ALGO_LOGISTIC_REGRESSION) == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_logistic_regression_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, NDB_ALGO_RIDGE) == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_ridge_regression_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, NDB_ALGO_LASSO) == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_lasso_regression_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, "random_forest") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_random_forest_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, "svm") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_svm_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, "decision_tree") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_decision_tree_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, "naive_bayes") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_naive_bayes_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, "knn") == 0 || strcmp(algorithm, "knn_classifier") == 0 || strcmp(algorithm, "knn_regressor") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_knn_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, "kmeans") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_kmeans_by_model_id(%d, %s, %s)",
						 model_id, q_table_name, q_feature_col);

		nfree(q_table_name);
		nfree(q_feature_col);
	}
	else if (strcmp(algorithm, "gmm") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_gmm_by_model_id(%d, %s, %s)",
						 model_id, q_table_name, q_feature_col);

		nfree(q_table_name);
		nfree(q_feature_col);
	}
	else if (strcmp(algorithm, "minibatch_kmeans") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_minibatch_kmeans_by_model_id(%d, %s, %s)",
						 model_id, q_table_name, q_feature_col);

		nfree(q_table_name);
		nfree(q_feature_col);
	}
	else if (strcmp(algorithm, "hierarchical") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);

		appendStringInfo(&sql, "SELECT evaluate_hierarchical_by_model_id(%d, %s, %s)",
						 model_id, q_table_name, q_feature_col);

		nfree(q_table_name);
		nfree(q_feature_col);
	}
	else if (strcmp(algorithm, "xgboost") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_xgboost_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, "catboost") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_catboost_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
	}
	else if (strcmp(algorithm, "lightgbm") == 0)
	{
		char	   *q_table_name = neurondb_quote_literal_cstr(table_name);
		char	   *q_feature_col = neurondb_quote_literal_cstr(feature_col);
		char	   *q_label_col = neurondb_quote_literal_cstr(label_col);

		appendStringInfo(&sql, "SELECT evaluate_lightgbm_by_model_id(%d, %s, %s, %s)",
						 model_id, q_table_name, q_feature_col, q_label_col);

		nfree(q_table_name);
		nfree(q_feature_col);
		nfree(q_label_col);
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
			Jsonb *temp_jsonb = NULL;

			/* Get JSONB from SPI result using safe function */
			temp_jsonb = ndb_spi_get_jsonb(spi_session, 0, 1, oldcontext);
			if (temp_jsonb == NULL)
			{
				result_isnull = true;
			}
			if (result_isnull)
			{
				/*
				 * Evaluation returned NULL - return NULL instead of empty JSONB
				 */
				MemoryContextSwitchTo(oldcontext);
				ndb_spi_session_end(&spi_session);
				MemoryContextDelete(callcontext);
				PG_RETURN_NULL();
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
					nfree(result);
				}
				/* Return NULL instead of trying to create empty JSONB */
				result = NULL;
			}
		}
	}
	PG_CATCH();
	{
		ErrorData *edata = NULL;
		char *error_msg = NULL;

		/*
		 * Switch out of ErrorContext before CopyErrorData(). CopyErrorData()
		 * allocates memory and must NOT be called while in ErrorContext, as
		 * that context is only for error reporting and will be reset, causing
		 * memory leaks or corruption.
		 */
		MemoryContextSwitchTo(oldcontext);

		/* Safely copy error data, handling errors in error handling */
		/* Use helper function to avoid nested PG_TRY shadow warnings */
		edata = neurondb_safe_copy_error_data();

		/* Create safe error message (escape quotes) */
		if (edata != NULL && edata->message != NULL)
			error_msg = pstrdup(edata->message);
		else
			error_msg = pstrdup("evaluation failed (GPU may be unavailable or model data missing)");

		/* Escape JSON special characters */
		{
			char *escaped = NULL;
			char *p = NULL;
			const char *s;

			nalloc(escaped, char, strlen(error_msg) * 2 + 1);

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
			nfree(error_msg);

			error_msg = NULL;
			error_msg = escaped;
		}

		/* Return error JSONB using proper PostgreSQL API */
		{
			StringInfoData error_json;

			ndb_spi_stringinfo_init(spi_session, &error_json);
			appendStringInfo(&error_json, "{\"error\": \"%s\"}", error_msg);
			
			/* Use PostgreSQL's jsonb_in function to create proper JSONB */
			result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, 
				PointerGetDatum(error_json.data)));
			
			ndb_spi_stringinfo_free(spi_session, &error_json);
			error_json.data = NULL;
			nfree(error_msg);
			error_msg = NULL;
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

	ereport(DEBUG2,
			(errmsg("neurondb_evaluate: final validation"),
			 errdetail("result=%p", (void *) result)));

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
		elog(WARNING, "neurondb_evaluate: result is NULL, returning NULL");
		PG_RETURN_NULL();
	}

	ereport(DEBUG2,
			(errmsg("neurondb_evaluate: about to return JSONB result"),
			 errdetail("result_size=%d", VARSIZE(result))));

	if (result == NULL || VARSIZE(result) < sizeof(Jsonb))
	{
		/* Return NULL instead of invalid JSONB */
		elog(WARNING, "neurondb_evaluate: EMERGENCY - result invalid, returning NULL");
		PG_RETURN_NULL();
	}

	ereport(DEBUG1,
			(errmsg("neurondb_evaluate: evaluation completed successfully"),
			 errdetail("model_id=%d, algorithm=%s, result_size=%d", model_id, algorithm ? algorithm : "unknown", VARSIZE(result))));

	PG_RETURN_JSONB_P(result);
}
