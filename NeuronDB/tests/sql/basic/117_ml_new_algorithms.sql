-- ============================================================================
-- NeurondB: New ML Algorithms SQL Function Definitions
-- ============================================================================
-- This file defines SQL functions for:
-- - Anomaly Detection (Isolation Forest, LOF, One-Class SVM)
-- - Reinforcement Learning (Q-Learning, Multi-Armed Bandits)
-- - Graph Neural Networks (GCN, GraphSAGE)
-- - Explainable AI (SHAP, LIME, Feature Importance)
-- ============================================================================

-- ============================================================================
-- ANOMALY DETECTION FUNCTIONS
-- ============================================================================
-- Functions should already be created by extension, use CREATE OR REPLACE if needed

DO $$
BEGIN
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.detect_anomalies_isolation_forest(
      table_name text,
      vector_column text,
      n_trees integer DEFAULT 100,
      contamination double precision DEFAULT 0.1
    )
    RETURNS boolean[]
    AS 'MODULE_PATHNAME', 'detect_anomalies_isolation_forest'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'detect_anomalies_isolation_forest may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.detect_anomalies_lof(
      table_name text,
      vector_column text,
      k integer DEFAULT 20,
      threshold double precision DEFAULT 1.5
    )
    RETURNS boolean[]
    AS 'MODULE_PATHNAME', 'detect_anomalies_lof'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'detect_anomalies_lof may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.detect_anomalies_ocsvm(
      table_name text,
      vector_column text,
      nu double precision DEFAULT 0.1,
      gamma double precision DEFAULT 1.0
    )
    RETURNS boolean[]
    AS 'MODULE_PATHNAME', 'detect_anomalies_ocsvm'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'detect_anomalies_ocsvm may already exist or not be available: %', SQLERRM;
  END;
END$$;

-- Comment only when function was created (optional; skip if MODULE_PATHNAME failed)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_proc p JOIN pg_namespace n ON p.pronamespace = n.oid WHERE n.nspname = 'neurondb' AND p.proname = 'detect_anomalies_ocsvm') THEN
    EXECUTE 'COMMENT ON FUNCTION neurondb.detect_anomalies_ocsvm(text, text, double precision, double precision) IS ''One-Class SVM anomaly detection. Returns boolean array indicating anomalies.''';
  END IF;
END$$;

-- ============================================================================
-- REINFORCEMENT LEARNING FUNCTIONS
-- ============================================================================

DO $$
BEGIN
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.qlearning_train(
      table_name text,
      n_states integer,
      n_actions integer,
      learning_rate double precision DEFAULT 0.1,
      discount_factor double precision DEFAULT 0.95,
      epsilon double precision DEFAULT 0.1,
      iterations integer DEFAULT 1000
    )
    RETURNS integer
    AS 'MODULE_PATHNAME', 'qlearning_train'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'qlearning_train may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.qlearning_predict(
      model_id integer,
      state_id integer
    )
    RETURNS integer
    AS 'MODULE_PATHNAME', 'qlearning_predict'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'qlearning_predict may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.multi_armed_bandit(
      table_name text,
      algorithm text,
      n_arms integer,
      epsilon double precision DEFAULT 0.1,
      alpha double precision DEFAULT 1.0,
      beta double precision DEFAULT 1.0
    )
    RETURNS double precision[]
    AS 'MODULE_PATHNAME', 'multi_armed_bandit'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'multi_armed_bandit may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.gcn_train(
      graph_table text,
      features_table text,
      labels_table text,
      n_nodes integer,
      feature_dim integer,
      hidden_dim integer DEFAULT 64,
      output_dim integer DEFAULT 2,
      learning_rate double precision DEFAULT 0.01,
      epochs integer DEFAULT 100
    )
    RETURNS integer
    AS 'MODULE_PATHNAME', 'gcn_train'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'gcn_train may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.graphsage_aggregate(
      graph_table text,
      features_table text,
      node_id integer,
      n_samples integer DEFAULT 10,
      depth integer DEFAULT 2
    )
    RETURNS real[]
    AS 'MODULE_PATHNAME', 'graphsage_aggregate'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'graphsage_aggregate may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.calculate_shap_values(
      model_id integer,
      instance real[],
      n_samples integer DEFAULT 100
    )
    RETURNS double precision[]
    AS 'MODULE_PATHNAME', 'calculate_shap_values'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'calculate_shap_values may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.explain_with_lime(
      model_id integer,
      instance real[],
      n_samples integer DEFAULT 1000,
      n_features integer DEFAULT 10
    )
    RETURNS jsonb
    AS 'MODULE_PATHNAME', 'explain_with_lime'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'explain_with_lime may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.feature_importance(
      model_id integer,
      table_name text,
      feature_column text,
      target_column text,
      metric text DEFAULT 'mse'
    )
    RETURNS double precision[]
    AS 'MODULE_PATHNAME', 'feature_importance'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'feature_importance may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.reduce_tsne(
      table_name text,
      vector_column text,
      n_components integer DEFAULT 2,
      perplexity double precision DEFAULT 30.0,
      learning_rate double precision DEFAULT 200.0,
      iterations integer DEFAULT 1000
    )
    RETURNS SETOF real[]
    AS 'MODULE_PATHNAME', 'reduce_tsne'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'reduce_tsne may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.reduce_umap(
      table_name text,
      vector_column text,
      n_components integer DEFAULT 2,
      n_neighbors integer DEFAULT 15,
      min_dist double precision DEFAULT 0.1,
      iterations integer DEFAULT 200
    )
    RETURNS SETOF real[]
    AS 'MODULE_PATHNAME', 'reduce_umap'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'reduce_umap may already exist or not be available: %', SQLERRM;
  END;
  
  BEGIN
    CREATE OR REPLACE FUNCTION neurondb.train_autoencoder(
      table_name text,
      vector_column text,
      encoding_dim integer,
      hidden_dims integer[] DEFAULT ARRAY[64, 32],
      learning_rate double precision DEFAULT 0.001,
      epochs integer DEFAULT 100
    )
    RETURNS integer
    AS 'MODULE_PATHNAME', 'train_autoencoder'
    LANGUAGE C STRICT;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'train_autoencoder may already exist or not be available: %', SQLERRM;
  END;
END$$;

