/*
 * neurondb_ml_training.h
 *    Header file for ML Training and Inference API
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 */

#ifndef NEURONDB_ML_TRAINING_H
#define NEURONDB_ML_TRAINING_H

#include "postgres.h"
#include "fmgr.h"

/* Model Training Functions */
extern Datum neurondb_train_model(PG_FUNCTION_ARGS);
extern Datum neurondb_predict(PG_FUNCTION_ARGS);
extern Datum neurondb_predict_proba(PG_FUNCTION_ARGS);
extern Datum neurondb_predict_batch(PG_FUNCTION_ARGS);

/* Embedding Functions */
extern Datum neurondb_generate_embedding_c(PG_FUNCTION_ARGS);
extern Datum neurondb_batch_embed_c(PG_FUNCTION_ARGS);

/* RAG Pipeline Functions */
extern Datum neurondb_chunk_text_c(PG_FUNCTION_ARGS);
extern Datum neurondb_retrieve_context_c(PG_FUNCTION_ARGS);
extern Datum neurondb_rerank_results_c(PG_FUNCTION_ARGS);
extern Datum neurondb_generate_answer_c(PG_FUNCTION_ARGS);

#endif							/* NEURONDB_ML_TRAINING_H */
