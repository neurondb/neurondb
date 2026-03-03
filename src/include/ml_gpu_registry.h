/*-------------------------------------------------------------------------
 *
 * ml_gpu_registry.h
 *    Registration helper for GPU-capable ML model implementations.
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_ML_GPU_REGISTRY_H
#define NEURONDB_ML_GPU_REGISTRY_H

extern void neurondb_gpu_register_models(void);

/* Core Supervised Learning */
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

#endif							/* NEURONDB_ML_GPU_REGISTRY_H */
