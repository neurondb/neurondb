/*-------------------------------------------------------------------------
 *
 * ml_gpu_registry.c
 *    GPU-capable ML model registry.
 *
 * This module registers GPU-capable ML model implementations for
 * accelerated training and inference.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_gpu_registry.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "ml_gpu_registry.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

/* Declarations for per-algorithm registration routines */
/* Core Supervised */
extern void neurondb_gpu_register_rf_model(void);
extern void neurondb_gpu_register_lr_model(void);
extern void neurondb_gpu_register_linreg_model(void);
extern void neurondb_gpu_register_svm_model(void);
extern void neurondb_gpu_register_dt_model(void);
extern void neurondb_gpu_register_nb_model(void);
extern void neurondb_gpu_register_ridge_model(void);
extern void neurondb_gpu_register_lasso_model(void);

/* Clustering */
extern void neurondb_gpu_register_gmm_model(void);
extern void neurondb_gpu_register_kmeans_model(void);
extern void neurondb_gpu_register_minibatch_kmeans_model(void);
extern void neurondb_gpu_register_hierarchical_model(void);
extern void neurondb_gpu_register_dbscan_model(void);

/* Advanced ML */
extern void neurondb_gpu_register_xgboost_model(void);
extern void neurondb_gpu_register_catboost_model(void);
extern void neurondb_gpu_register_lightgbm_model(void);
extern void neurondb_gpu_register_neural_network_model(void);

/* Specialized */
extern void neurondb_gpu_register_knn_model(void);
extern void neurondb_gpu_register_opq_model(void);
extern void neurondb_gpu_register_automl_model(void);
extern void neurondb_gpu_register_recommender_model(void);
extern void neurondb_gpu_register_pca_whitening_model(void);
extern void neurondb_gpu_register_product_quantization_model(void);

/* Text & NLP */
extern void neurondb_gpu_register_text_model(void);
extern void neurondb_gpu_register_nlp_production_model(void);
extern void neurondb_gpu_register_topic_discovery_model(void);

/* Time Series & Metrics */
extern void neurondb_gpu_register_timeseries_model(void);
extern void neurondb_gpu_register_davies_bouldin_model(void);

void
neurondb_gpu_register_models(void)
{
	static bool registered = false;

	if (registered)
		return;

	/* Core Supervised Learning */
	neurondb_gpu_register_rf_model();
	neurondb_gpu_register_lr_model();
	neurondb_gpu_register_linreg_model();
	neurondb_gpu_register_svm_model();
	neurondb_gpu_register_dt_model();
	neurondb_gpu_register_nb_model();
	neurondb_gpu_register_ridge_model();
	neurondb_gpu_register_lasso_model();

	/* Clustering */
	neurondb_gpu_register_gmm_model();
	neurondb_gpu_register_kmeans_model();
	neurondb_gpu_register_minibatch_kmeans_model();
	neurondb_gpu_register_hierarchical_model();
	neurondb_gpu_register_dbscan_model();

	/* Advanced ML */
	neurondb_gpu_register_xgboost_model();
	neurondb_gpu_register_catboost_model();
	neurondb_gpu_register_lightgbm_model();
	neurondb_gpu_register_neural_network_model();

	/* Specialized */
	neurondb_gpu_register_knn_model();
	neurondb_gpu_register_opq_model();
	neurondb_gpu_register_automl_model();
	neurondb_gpu_register_recommender_model();
	neurondb_gpu_register_pca_whitening_model();
	neurondb_gpu_register_product_quantization_model();

	/* Text & NLP */
	neurondb_gpu_register_text_model();
	neurondb_gpu_register_nlp_production_model();
	neurondb_gpu_register_topic_discovery_model();

	/* Time Series & Metrics */
	neurondb_gpu_register_timeseries_model();
	neurondb_gpu_register_davies_bouldin_model();

	registered = true;
}
