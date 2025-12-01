/*-------------------------------------------------------------------------
 *
 * neurondb_constants.h
 *    Centralized constants, schema names, table names, and configuration
 *    for NeuronDB
 *
 * All hardcoded strings, table names, schema names, status values,
 * algorithm names, function names, JSON/JSONB keys, and other constants
 * should be defined here. This enables single-point-of-change for all
 * constants across 100+ source files.
 *
 * Catalog tables use 'nb_' prefix (like PostgreSQL uses 'pg_')
 *
 * Copyright (c) 2024-2025, neurondb, Inc. <admin@neurondb.com>
 *
 * IDENTIFICATION
 *    include/neurondb_constants.h
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CONSTANTS_H
#define NEURONDB_CONSTANTS_H

#include "postgres.h"
#include "utils/jsonb.h"

/* ----------
 * Schema
 * ----------
 */
#define NDB_SCHEMA_NAME                     "neurondb"

/* ----------
 * Catalog Table Names (clean names without prefix)
 * ----------
 * All catalog tables use clean names since they're already in neurondb schema
 */
/* ML Core Tables */
#define NDB_TABLE_ML_PROJECTS               "ml_projects"
#define NDB_TABLE_ML_MODELS                 "ml_models"
#define NDB_TABLE_ML_EXPERIMENTS            "ml_experiments"
#define NDB_TABLE_ML_DEPLOYMENTS            "ml_deployments"
#define NDB_TABLE_ML_PREDICTIONS            "ml_predictions"
#define NDB_TABLE_ML_TRAINED_MODELS         "ml_trained_models"
#define NDB_TABLE_ML_PIPELINES              "ml_pipelines"

/* Feature Store Tables */
#define NDB_TABLE_FEATURE_STORES            "feature_stores"
#define NDB_TABLE_FEATURE_DEFINITIONS       "feature_definitions"
#define NDB_TABLE_FEATURES                  "features"

/* MLOps Tables */
#define NDB_TABLE_AB_TESTS                  "ab_tests"
#define NDB_TABLE_MODEL_MONITORING          "model_monitoring"
#define NDB_TABLE_MODEL_VERSIONS            "model_versions"
#define NDB_TABLE_DRIFT_DETECTION           "drift_detection"
#define NDB_TABLE_FEATURE_FLAGS             "feature_flags"
#define NDB_TABLE_EXPERIMENT_METRICS        "experiment_metrics"
#define NDB_TABLE_MODEL_AUDIT_LOG           "model_audit_log"

/* Recommender System Tables */
#define NDB_TABLE_COLLABORATIVE_FILTER_MODELS "collaborative_filter_models"
#define NDB_TABLE_RECOMMENDATIONS_CACHE     "recommendations_cache"
#define NDB_TABLE_CF_USER_FACTORS           "cf_user_factors"
#define NDB_TABLE_CF_ITEM_FACTORS           "cf_item_factors"

/* Deep Learning & Specialized Tables */
#define NDB_TABLE_DL_MODELS                 "dl_models"
#define NDB_TABLE_TIMESERIES_MODELS         "timeseries_models"
#define NDB_TABLE_TEXT_MODELS               "text_models"

/* RAG & LLM Tables */
#define NDB_TABLE_RAG_PIPELINES             "rag_pipelines"
#define NDB_TABLE_LLM_CACHE                 "llm_cache"

/* Unified API Tables */
#define NDB_TABLE_PROJECTS                  "nb_catalog"  /* Unified API catalog */
#define NDB_TABLE_HYPERPARAMETER_RESULTS    "hyperparameter_results"

/* Index & System Tables */
#define NDB_TABLE_INDEX_REBUILD_HISTORY     "index_rebuild_history"

/* Worker & Queue Tables */
#define NDB_TABLE_JOB_QUEUE                 "job_queue"
#define NDB_TABLE_QUERY_METRICS             "query_metrics"
#define NDB_TABLE_EMBEDDING_CACHE           "embedding_cache"
#define NDB_TABLE_EMBEDDING_MODEL_CONFIG    "embedding_model_config"
#define NDB_TABLE_INDEX_MAINTENANCE         "index_maintenance"
#define NDB_TABLE_HISTOGRAMS                "histograms"
#define NDB_TABLE_PROMETHEUS_METRICS        "prometheus_metrics"
#define NDB_TABLE_LLM_JOBS                  "llm_jobs"
#define NDB_TABLE_LLM_STATS                 "llm_stats"
#define NDB_TABLE_LLM_ERRORS                "llm_errors"

/* ARIMA Tables */
#define NDB_TABLE_ARIMA_MODELS              "arima_models"
#define NDB_TABLE_ARIMA_HISTORY             "arima_history"

/* ----------
 * Fully Qualified Table Names (schema.table)
 * ----------
 */
#define NDB_FQ_TABLE(name)                  NDB_SCHEMA_NAME "." name
#define NDB_FQ_ML_MODELS                    NDB_FQ_TABLE(NDB_TABLE_ML_MODELS)                /* neurondb.ml_models */
#define NDB_FQ_ML_PROJECTS                  NDB_FQ_TABLE(NDB_TABLE_ML_PROJECTS)              /* neurondb.ml_projects */
#define NDB_FQ_ML_EXPERIMENTS               NDB_FQ_TABLE(NDB_TABLE_ML_EXPERIMENTS)           /* neurondb.ml_experiments */
#define NDB_FQ_ML_DEPLOYMENTS               NDB_FQ_TABLE(NDB_TABLE_ML_DEPLOYMENTS)           /* neurondb.ml_deployments */
#define NDB_FQ_ML_PREDICTIONS               NDB_FQ_TABLE(NDB_TABLE_ML_PREDICTIONS)           /* neurondb.ml_predictions */
#define NDB_FQ_FEATURE_STORES               NDB_FQ_TABLE(NDB_TABLE_FEATURE_STORES)           /* neurondb.feature_stores */
#define NDB_FQ_FEATURE_DEFINITIONS          NDB_FQ_TABLE(NDB_TABLE_FEATURE_DEFINITIONS)      /* neurondb.feature_definitions */
#define NDB_FQ_FEATURES                     NDB_FQ_TABLE(NDB_TABLE_FEATURES)                 /* neurondb.features */
#define NDB_FQ_LLM_CACHE                    NDB_FQ_TABLE(NDB_TABLE_LLM_CACHE)                /* neurondb.llm_cache */
#define NDB_FQ_INDEX_REBUILD_HISTORY        NDB_FQ_TABLE(NDB_TABLE_INDEX_REBUILD_HISTORY)    /* neurondb.index_rebuild_history */
#define NDB_FQ_PROJECTS                     NDB_FQ_TABLE(NDB_TABLE_PROJECTS)                 /* neurondb.nb_catalog */
#define NDB_FQ_HYPERPARAMETER_RESULTS       NDB_FQ_TABLE(NDB_TABLE_HYPERPARAMETER_RESULTS)   /* neurondb.hyperparameter_results */

/* ----------
 * Column Names
 * ----------
 */
#define NDB_COL_MODEL_ID                    "model_id"
#define NDB_COL_PROJECT_ID                  "project_id"
#define NDB_COL_EXPERIMENT_ID               "experiment_id"
#define NDB_COL_DEPLOYMENT_ID               "deployment_id"
#define NDB_COL_ALGORITHM                   "algorithm"
#define NDB_COL_STATUS                      "status"
#define NDB_COL_VERSION                     "version"
#define NDB_COL_MODEL_NAME                  "model_name"
#define NDB_COL_PROJECT_NAME                "project_name"
#define NDB_COL_TRAINING_TABLE              "training_table"
#define NDB_COL_TRAINING_COLUMN             "training_column"
#define NDB_COL_FEATURE_COLUMNS             "feature_columns"
#define NDB_COL_TARGET_COLUMN               "target_column"
#define NDB_COL_HYPERPARAMETERS             "hyperparameters"
#define NDB_COL_METRICS                     "metrics"
#define NDB_COL_MODEL_DATA                  "model_data"
#define NDB_COL_PARAMETERS                  "parameters"
#define NDB_COL_CREATED_AT                  "created_at"
#define NDB_COL_UPDATED_AT                  "updated_at"
#define NDB_COL_DEPLOYED_AT                 "deployed_at"

/* ----------
 * Status Values
 * ----------
 */
/* Model Status */
#define NDB_STATUS_TRAINING                 "training"
#define NDB_STATUS_COMPLETED                "completed"
#define NDB_STATUS_FAILED                   "failed"
#define NDB_STATUS_DEPLOYED                 "deployed"

/* Project Status */
#define NDB_PROJECT_ACTIVE                  "active"
#define NDB_PROJECT_ARCHIVED                "archived"
#define NDB_PROJECT_DELETED                 "deleted"

/* Deployment Status */
#define NDB_DEPLOYMENT_ACTIVE               "active"
#define NDB_DEPLOYMENT_INACTIVE             "inactive"
#define NDB_DEPLOYMENT_ROLLBACK             "rollback"

/* Experiment Status */
#define NDB_EXPERIMENT_RUNNING              "running"
#define NDB_EXPERIMENT_COMPLETED            "completed"
#define NDB_EXPERIMENT_FAILED               "failed"

/* AB Test Status */
#define NDB_AB_TEST_RUNNING                 "running"
#define NDB_AB_TEST_COMPLETED               "completed"
#define NDB_AB_TEST_PAUSED                  "paused"

/* Job/Worker Status */
#define NDB_JOB_DONE                        "done"
#define NDB_JOB_FAILED                      "failed"

/* ----------
 * Algorithm Names
 * ----------
 */
#define NDB_ALGO_LINEAR_REGRESSION          "linear_regression"
#define NDB_ALGO_LOGISTIC_REGRESSION        "logistic_regression"
#define NDB_ALGO_RIDGE                      "ridge"
#define NDB_ALGO_LASSO                      "lasso"
#define NDB_ALGO_RANDOM_FOREST              "random_forest"
#define NDB_ALGO_SVM                        "svm"
#define NDB_ALGO_KNN                        "knn"
#define NDB_ALGO_KNN_CLASSIFIER             "knn_classifier"
#define NDB_ALGO_KNN_REGRESSOR              "knn_regressor"
#define NDB_ALGO_DECISION_TREE              "decision_tree"
#define NDB_ALGO_NAIVE_BAYES                "naive_bayes"
#define NDB_ALGO_XGBOOST                    "xgboost"
#define NDB_ALGO_KMEANS                     "kmeans"
#define NDB_ALGO_GMM                        "gmm"
#define NDB_ALGO_MINIBATCH_KMEANS           "minibatch_kmeans"
#define NDB_ALGO_HIERARCHICAL               "hierarchical"
#define NDB_ALGO_DBSCAN                     "dbscan"
#define NDB_ALGO_PCA                        "pca"
#define NDB_ALGO_OPQ                        "opq"

/* ----------
 * SQL Function Names
 * ----------
 */
/* Core ML Functions */
#define NDB_FUNC_TRAIN                      "neurondb.train"
#define NDB_FUNC_PREDICT                    "neurondb.predict"
#define NDB_FUNC_EVALUATE                   "neurondb.evaluate"
#define NDB_FUNC_DEPLOY                     "neurondb.deploy"
#define NDB_FUNC_LOAD_MODEL                 "neurondb.load_model"

/* Linear Regression Functions */
#define NDB_FUNC_TRAIN_LINEAR_REGRESSION                "train_linear_regression"
#define NDB_FUNC_PREDICT_LINEAR_REGRESSION_MODEL_ID     "predict_linear_regression_model_id"
#define NDB_FUNC_PREDICT_LINEAR_REGRESSION              "predict_linear_regression"
#define NDB_FUNC_EVALUATE_LINEAR_REGRESSION             "evaluate_linear_regression"
#define NDB_FUNC_EVALUATE_LINEAR_REGRESSION_MODEL_ID    "evaluate_linear_regression_by_model_id"

/* Logistic Regression Functions */
#define NDB_FUNC_TRAIN_LOGISTIC_REGRESSION              "train_logistic_regression"
#define NDB_FUNC_PREDICT_LOGISTIC_REGRESSION            "predict_logistic_regression"
#define NDB_FUNC_PREDICT_LOGISTIC_REGRESSION_MODEL_ID   "predict_logistic_regression_model_id"
#define NDB_FUNC_EVALUATE_LOGISTIC_REGRESSION           "evaluate_logistic_regression"
#define NDB_FUNC_EVALUATE_LOGISTIC_REGRESSION_MODEL_ID  "evaluate_logistic_regression_by_model_id"

/* KNN Functions */
#define NDB_FUNC_TRAIN_KNN_MODEL_ID                     "train_knn_model_id"
#define NDB_FUNC_PREDICT_KNN_MODEL_ID                   "predict_knn_model_id"

/* SVM Functions */
#define NDB_FUNC_TRAIN_SVM_CLASSIFIER                   "train_svm_classifier"
#define NDB_FUNC_PREDICT_SVM_MODEL_ID                   "predict_svm_model_id"

/* Naive Bayes Functions */
#define NDB_FUNC_TRAIN_NAIVE_BAYES_CLASSIFIER           "train_naive_bayes_classifier"
#define NDB_FUNC_TRAIN_NAIVE_BAYES_CLASSIFIER_MODEL_ID  "train_naive_bayes_classifier_model_id"
#define NDB_FUNC_PREDICT_NAIVE_BAYES                   "predict_naive_bayes"
#define NDB_FUNC_PREDICT_NAIVE_BAYES_MODEL_ID          "predict_naive_bayes_model_id"

/* Recommender Functions */
#define NDB_FUNC_TRAIN_COLLABORATIVE_FILTER             "train_collaborative_filter"
#define NDB_FUNC_PREDICT_COLLABORATIVE_FILTER           "predict_collaborative_filter"

/* GPU Vector Functions */
#define NDB_FUNC_HNSW_KNN_SEARCH_GPU                    "hnsw_knn_search_gpu"
#define NDB_FUNC_IVF_KNN_SEARCH_GPU                     "ivf_knn_search_gpu"

/* RAG Functions */
#define NDB_FUNC_RERANK_ENSEMBLE_WEIGHTED               "neurondb.rerank_ensemble_weighted"
#define NDB_FUNC_RERANK_ENSEMBLE_BORDA                  "neurondb.rerank_ensemble_borda"

/* ----------
 * PostgreSQL Type Names
 * ----------
 */
#define NDB_TYPE_ML_ALGORITHM               "ml_algorithm_type"
#define NDB_TYPE_ML_MODEL                   "ml_model_type"
#define NDB_TYPE_VECTOR                     "vector"
#define NDB_TYPE_SPARSE_VECTOR              "sparse_vector"

/* Fully qualified type names */
#define NDB_FQ_TYPE_ML_ALGORITHM            NDB_SCHEMA_NAME "." NDB_TYPE_ML_ALGORITHM   /* neurondb.ml_algorithm_type */
#define NDB_FQ_TYPE_ML_MODEL                NDB_SCHEMA_NAME "." NDB_TYPE_ML_MODEL       /* neurondb.ml_model_type */

/* ----------
 * GUC Configuration Variable Names
 * ----------
 */
/* Index GUC Variables */
#define NDB_GUC_EF_CONSTRUCTION             "neurondb_ef_construction"
#define NDB_GUC_EF_SEARCH                   "neurondb_ef_search"
#define NDB_GUC_VECTOR_DIM_LIMIT            "neurondb_vector_dim_limit"
#define NDB_GUC_IVF_PROBES                  "neurondb.ivf_probes"

/* Performance GUC Variables */
#define NDB_GUC_MAX_CONNECTIONS             "neurondb_max_connections"
#define NDB_GUC_INDEX_PARALLELISM           "neurondb_index_parallelism"

/* Memory GUC Variables */
#define NDB_GUC_BUFFER_SIZE                 "neurondb_buffer_size"

/* GPU GUC Variables */
#define NDB_GUC_USE_GPU                     "neurondb_use_gpu"

/* Search GUC Variables */
#define NDB_GUC_HYBRID_THRESHOLD            "neurondb_hybrid_threshold"

/* Replication GUC Variables */
#define NDB_GUC_WAL_COMPRESSION             "neurondb_wal_compression"

/* Worker GUC Variables */
#define NDB_GUC_NEURANMON_NAPTIME           "neurondb.neuranmon_naptime"
#define NDB_GUC_NEURANMON_SAMPLE_SIZE       "neurondb.neuranmon_sample_size"
#define NDB_GUC_NEURANMON_TARGET_LATENCY    "neurondb.neuranmon_target_latency"
#define NDB_GUC_NEURANMON_TARGET_RECALL     "neurondb.neuranmon_target_recall"
#define NDB_GUC_NEURANMON_ENABLED           "neurondb.neuranmon_enabled"
#define NDB_GUC_NEURANQ_NAPTIME             "neurondb.neuranq_naptime"
#define NDB_GUC_NEURANQ_QUEUE_DEPTH         "neurondb.neuranq_queue_depth"
#define NDB_GUC_NEURANQ_BATCH_SIZE          "neurondb.neuranq_batch_size"
#define NDB_GUC_NEURANQ_TIMEOUT             "neurondb.neuranq_timeout"
#define NDB_GUC_NEURANQ_MAX_RETRIES         "neurondb.neuranq_max_retries"
#define NDB_GUC_NEURANQ_ENABLED             "neurondb.neuranq_enabled"

/* ----------
 * Error Message Patterns
 * ----------
 */
#define NDB_ERR_PREFIX                      "neurondb:"
#define NDB_ERR_MSG(fmt, ...)               NDB_ERR_PREFIX " " fmt, ##__VA_ARGS__

/* Function-specific error prefixes */
#define NDB_ERR_PREFIX_TRAIN                NDB_ERR_PREFIX "train:"
#define NDB_ERR_PREFIX_PREDICT              NDB_ERR_PREFIX "predict:"
#define NDB_ERR_PREFIX_EVALUATE             NDB_ERR_PREFIX "evaluate:"

/* ----------
 * Distance Metrics
 * ----------
 */
#define NDB_DIST_L2                         "l2"
#define NDB_DIST_COSINE                     "cosine"
#define NDB_DIST_INNER_PRODUCT              "inner_product"
#define NDB_DIST_HAMMING                    "hamming"
#define NDB_DIST_JACCARD                    "jaccard"

/* ----------
 * Storage/Backend Values
 * ----------
 */
#define NDB_STORAGE_GPU                     "gpu"
#define NDB_STORAGE_CPU                     "cpu"

/* ----------
 * Compute Mode Enum
 * ----------
 * Controls execution mode for ML operations
 */
typedef enum {
	NDB_COMPUTE_MODE_CPU = 0,
	NDB_COMPUTE_MODE_GPU = 1,
	NDB_COMPUTE_MODE_AUTO = 2
} NDBComputeMode;

/* ----------
 * GPU Backend Type Enum
 * ----------
 * Selects GPU backend implementation (only valid when compute_mode is GPU or AUTO)
 */
typedef enum {
	NDB_GPU_BACKEND_TYPE_CUDA = 0,
	NDB_GPU_BACKEND_TYPE_ROCM = 1,
	NDB_GPU_BACKEND_TYPE_METAL = 2
} NDBGpuBackendType;

/* ----------
 * Compute Mode Helper Macros
 * ----------
 * These macros check the current compute_mode GUC value
 */
#define NDB_COMPUTE_MODE_IS_CPU() \
	(neurondb_compute_mode == NDB_COMPUTE_MODE_CPU)

#define NDB_COMPUTE_MODE_IS_GPU() \
	(neurondb_compute_mode == NDB_COMPUTE_MODE_GPU)

#define NDB_COMPUTE_MODE_IS_AUTO() \
	(neurondb_compute_mode == NDB_COMPUTE_MODE_AUTO)

#define NDB_SHOULD_TRY_GPU() \
	(neurondb_compute_mode == NDB_COMPUTE_MODE_GPU || neurondb_compute_mode == NDB_COMPUTE_MODE_AUTO)

#define NDB_REQUIRE_GPU() \
	(neurondb_compute_mode == NDB_COMPUTE_MODE_GPU)

/* ----------
 * GPU Backend Type Helper Macros
 * ----------
 * These macros check the current gpu_backend_type GUC value
 */
#define NDB_GPU_BACKEND_TYPE_IS_CUDA() \
	(neurondb_gpu_backend_type == NDB_GPU_BACKEND_TYPE_CUDA)

#define NDB_GPU_BACKEND_TYPE_IS_ROCM() \
	(neurondb_gpu_backend_type == NDB_GPU_BACKEND_TYPE_ROCM)

#define NDB_GPU_BACKEND_TYPE_IS_METAL() \
	(neurondb_gpu_backend_type == NDB_GPU_BACKEND_TYPE_METAL)

/* Forward declarations for GUC variables (defined in neurondb_guc.c) */
extern int neurondb_compute_mode;
extern int neurondb_gpu_backend_type;

/* ----------
 * JSON/JSONB Field Names (COMPREHENSIVE)
 * ----------
 */
/* Core JSON Fields */
#define NDB_JSON_KEY_STORAGE                "storage"
#define NDB_JSON_KEY_METRICS                "metrics"
#define NDB_JSON_KEY_HYPERPARAMETERS        "hyperparameters"
#define NDB_JSON_KEY_PARAMETERS             "parameters"
#define NDB_JSON_KEY_CONFIG                 "config"
#define NDB_JSON_KEY_RESULTS                "results"
#define NDB_JSON_KEY_METADATA               "metadata"
#define NDB_JSON_KEY_TEXT                   "text"
#define NDB_JSON_KEY_VALUE                  "value"

/* Hyperparameter Keys */
#define NDB_JSON_KEY_C                      "C"
#define NDB_JSON_KEY_KERNEL                 "kernel"
#define NDB_JSON_KEY_MAX_DEPTH              "max_depth"
#define NDB_JSON_KEY_N_TREES                "n_trees"
#define NDB_JSON_KEY_EPOCHS                 "epochs"
#define NDB_JSON_KEY_LR                     "lr"
#define NDB_JSON_KEY_FIT_INTERCEPT          "fit_intercept"
#define NDB_JSON_KEY_N_ESTIMATORS           "n_estimators"
#define NDB_JSON_KEY_MIN_SAMPLES            "min_samples"
#define NDB_JSON_KEY_MIN_SAMPLES_SPLIT      "min_samples_split"
#define NDB_JSON_KEY_VAR_SMOOTHING          "var_smoothing"
#define NDB_JSON_KEY_K                      "k"
#define NDB_JSON_KEY_MAX_ITERS              "max_iters"
#define NDB_JSON_KEY_N_FACTORS              "n_factors"
#define NDB_JSON_KEY_LAMBDA                 "lambda"
#define NDB_JSON_KEY_ALPHA                  "alpha"
#define NDB_JSON_KEY_GAMMA                  "gamma"
#define NDB_JSON_KEY_NU                     "nu"
#define NDB_JSON_KEY_CONTAMINATION          "contamination"
#define NDB_JSON_KEY_N_CLUSTERS             "n_clusters"
#define NDB_JSON_KEY_N_COMPONENTS           "n_components"

/* Metrics Keys */
#define NDB_JSON_KEY_MSE                    "mse"
#define NDB_JSON_KEY_RMSE                   "rmse"
#define NDB_JSON_KEY_MAE                    "mae"
#define NDB_JSON_KEY_R_SQUARED              "r_squared"
#define NDB_JSON_KEY_ACCURACY               "accuracy"
#define NDB_JSON_KEY_PRECISION              "precision"
#define NDB_JSON_KEY_RECALL                 "recall"
#define NDB_JSON_KEY_F1_SCORE               "f1_score"
#define NDB_JSON_KEY_SILHOUETTE_SCORE       "silhouette_score"
#define NDB_JSON_KEY_INERTIA                "inertia"
#define NDB_JSON_KEY_TRAINING_TIME_MS       "training_time_ms"

/* LLM/Generation Keys */
#define NDB_JSON_KEY_TEMPERATURE            "temperature"
#define NDB_JSON_KEY_TOP_P                  "top_p"
#define NDB_JSON_KEY_TOP_K                  "top_k"
#define NDB_JSON_KEY_MAX_TOKENS             "max_tokens"
#define NDB_JSON_KEY_MIN_TOKENS             "min_tokens"
#define NDB_JSON_KEY_REPETITION_PENALTY     "repetition_penalty"
#define NDB_JSON_KEY_DO_SAMPLE              "do_sample"
#define NDB_JSON_KEY_STOP_SEQUENCES         "stop_sequences"
#define NDB_JSON_KEY_LOGIT_BIAS             "logit_bias"

/* OpenAI API Keys */
#define NDB_JSON_KEY_CHOICES                "choices"
#define NDB_JSON_KEY_MESSAGE                "message"
#define NDB_JSON_KEY_CONTENT                "content"
#define NDB_JSON_KEY_USAGE                  "usage"
#define NDB_JSON_KEY_PROMPT_TOKENS          "prompt_tokens"
#define NDB_JSON_KEY_COMPLETION_TOKENS      "completion_tokens"
#define NDB_JSON_KEY_TOTAL_TOKENS           "total_tokens"
#define NDB_JSON_KEY_EMBEDDING              "embedding"
#define NDB_JSON_KEY_DATA                   "data"

/* ----------
 * JSON Operation Helper Macros
 * ----------
 * These macros use functions from neurondb_json.h
 */
#ifndef NEURONDB_JSON_H
/* Forward declarations - will be available when neurondb_json.h is included */
extern Jsonb *ndb_jsonb_object_field(Jsonb *jsonb, const char *field_name);
extern char *ndb_jsonb_out_cstring(Jsonb *jsonb);
#endif

/* Safe JSON field extraction using constants */
#define NDB_JSON_GET_FIELD(jsonb, key_const) \
	ndb_jsonb_object_field((jsonb), (key_const))

/* Extract string value from JSON field */
#define NDB_JSON_GET_STRING(jsonb, key_const) \
	({ \
		Jsonb *_j = ndb_jsonb_object_field((jsonb), (key_const)); \
		_j ? ndb_jsonb_out_cstring(_j) : NULL; \
	})

/* Check if JSON field exists (requires jsonb_exists check - simplified version) */
#define NDB_JSON_HAS_FIELD(jsonb, key_const) \
	(ndb_jsonb_object_field((jsonb), (key_const)) != NULL)

/* ----------
 * Numeric Constants
 * ----------
 */
#define NDB_MIN_SAMPLES_FOR_TRAINING        10
#define NDB_DEFAULT_MAX_ITERATIONS          1000
#define NDB_DEFAULT_N_TREES                 10
#define NDB_DEFAULT_MAX_DEPTH               10
#define NDB_DEFAULT_MIN_SAMPLES             100
#define NDB_MAX_VECTOR_DIM                  16384
#define NDB_DEFAULT_VECTOR_DIM_LIMIT        4096
#define NDB_DEFAULT_EF_CONSTRUCTION         200
#define NDB_DEFAULT_EF_SEARCH               100
#define NDB_MAX_TRAINING_TIME_MS            3600000 /* 1 hour */
#define NDB_DEFAULT_RAG_CHUNK_SIZE          500
#define NDB_DEFAULT_RAG_OVERLAP             50
#define NDB_NUMERICAL_PRECISION             1e-10
#define NDB_MAX_CLASSES                     256
#define NDB_DEFAULT_IVF_PROBES              10

/* ----------
 * Index Naming Patterns
 * ----------
 */
#define NDB_INDEX_PREFIX_HNSW               "__hnsw_"
#define NDB_INDEX_NAME_FMT_HNSW             NDB_INDEX_PREFIX_HNSW "%s_%s_%08x"

/* ----------
 * Feature Types
 * ----------
 */
#define NDB_FEATURE_TYPE_NUMERIC            "numeric"
#define NDB_FEATURE_TYPE_CATEGORICAL        "categorical"
#define NDB_FEATURE_TYPE_VECTOR             "vector"
#define NDB_FEATURE_TYPE_TEXT               "text"

/* ----------
 * Framework Names
 * ----------
 */
#define NDB_FRAMEWORK_PYTORCH               "pytorch"
#define NDB_FRAMEWORK_TENSORFLOW            "tensorflow"
#define NDB_FRAMEWORK_ONNX                  "onnx"

/* ----------
 * Drift Types
 * ----------
 */
#define NDB_DRIFT_TYPE_DATA                 "data"
#define NDB_DRIFT_TYPE_CONCEPT              "concept"
#define NDB_DRIFT_TYPE_PREDICTION           "prediction"

#endif /* NEURONDB_CONSTANTS_H */

