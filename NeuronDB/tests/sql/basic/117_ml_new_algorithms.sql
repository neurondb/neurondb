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

CREATE FUNCTION neurondb.detect_anomalies_isolation_forest(
	table_name text,
	vector_column text,
	n_trees integer DEFAULT 100,
	contamination double precision DEFAULT 0.1
)
RETURNS boolean[]
AS 'MODULE_PATHNAME', 'detect_anomalies_isolation_forest'
LANGUAGE C STRICT;

COMMENT ON FUNCTION neurondb.detect_anomalies_isolation_forest(text, text, integer, double precision) IS
'Isolation Forest anomaly detection. Returns boolean array indicating anomalies.';

CREATE FUNCTION neurondb.detect_anomalies_lof(
	table_name text,
	vector_column text,
	k integer DEFAULT 20,
	threshold double precision DEFAULT 1.5
)
RETURNS boolean[]
AS 'MODULE_PATHNAME', 'detect_anomalies_lof'
LANGUAGE C STRICT;

COMMENT ON FUNCTION neurondb.detect_anomalies_lof(text, text, integer, double precision) IS
'Local Outlier Factor (LOF) anomaly detection. Returns boolean array indicating anomalies.';

CREATE FUNCTION neurondb.detect_anomalies_ocsvm(
	table_name text,
	vector_column text,
	nu double precision DEFAULT 0.1,
	gamma double precision DEFAULT 1.0
)
RETURNS boolean[]
AS 'MODULE_PATHNAME', 'detect_anomalies_ocsvm'
LANGUAGE C STRICT;

COMMENT ON FUNCTION neurondb.detect_anomalies_ocsvm(text, text, double precision, double precision) IS
'One-Class SVM anomaly detection. Returns boolean array indicating anomalies.';

-- ============================================================================
-- REINFORCEMENT LEARNING FUNCTIONS
-- ============================================================================

CREATE FUNCTION neurondb.qlearning_train(
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

COMMENT ON FUNCTION neurondb.qlearning_train(text, integer, integer, double precision, double precision, double precision, integer) IS
'Train Q-Learning agent. Table must have columns: state_id, action_id, reward, next_state_id. Returns model_id.';

CREATE FUNCTION neurondb.qlearning_predict(
	model_id integer,
	state_id integer
)
RETURNS integer
AS 'MODULE_PATHNAME', 'qlearning_predict'
LANGUAGE C STRICT;

COMMENT ON FUNCTION neurondb.qlearning_predict(integer, integer) IS
'Get best action for a state using trained Q-Learning model.';

CREATE FUNCTION neurondb.multi_armed_bandit(
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

COMMENT ON FUNCTION neurondb.multi_armed_bandit(text, text, integer, double precision, double precision, double precision) IS
'Multi-armed bandit algorithms (thompson, ucb, epsilon_greedy). Table must have columns: arm_id, reward. Returns selection probabilities.';

-- ============================================================================
-- GRAPH NEURAL NETWORK FUNCTIONS
-- ============================================================================

CREATE FUNCTION neurondb.gcn_train(
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

COMMENT ON FUNCTION neurondb.gcn_train(text, text, text, integer, integer, integer, integer, double precision, integer) IS
'Train Graph Convolutional Network. Returns model_id.';

CREATE FUNCTION neurondb.graphsage_aggregate(
	graph_table text,
	features_table text,
	node_id integer,
	n_samples integer DEFAULT 10,
	depth integer DEFAULT 2
)
RETURNS real[]
AS 'MODULE_PATHNAME', 'graphsage_aggregate'
LANGUAGE C STRICT;

COMMENT ON FUNCTION neurondb.graphsage_aggregate(text, text, integer, integer, integer) IS
'GraphSAGE neighbor sampling and aggregation. Returns aggregated feature vector.';

-- ============================================================================
-- EXPLAINABLE AI FUNCTIONS
-- ============================================================================

CREATE FUNCTION neurondb.calculate_shap_values(
	model_id integer,
	instance real[],
	n_samples integer DEFAULT 100
)
RETURNS double precision[]
AS 'MODULE_PATHNAME', 'calculate_shap_values'
LANGUAGE C STRICT;

COMMENT ON FUNCTION neurondb.calculate_shap_values(integer, real[], integer) IS
'Calculate SHAP values for a prediction. Returns array of feature contributions.';

CREATE FUNCTION neurondb.explain_with_lime(
	model_id integer,
	instance real[],
	n_samples integer DEFAULT 1000,
	n_features integer DEFAULT 10
)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'explain_with_lime'
LANGUAGE C STRICT;

COMMENT ON FUNCTION neurondb.explain_with_lime(integer, real[], integer, integer) IS
'Generate LIME explanation for a prediction. Returns JSONB with feature importance.';

CREATE FUNCTION neurondb.feature_importance(
	model_id integer,
	table_name text,
	feature_column text,
	target_column text,
	metric text DEFAULT 'mse'
)
RETURNS double precision[]
AS 'MODULE_PATHNAME', 'feature_importance'
LANGUAGE C STRICT;

COMMENT ON FUNCTION neurondb.feature_importance(integer, text, text, text, text) IS
'Calculate feature importance using permutation method. Returns array of importance scores.';

-- ============================================================================
-- DIMENSIONALITY REDUCTION FUNCTIONS
-- ============================================================================

CREATE FUNCTION neurondb.reduce_tsne(
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

COMMENT ON FUNCTION neurondb.reduce_tsne(text, text, integer, double precision, double precision, integer) IS
't-SNE dimensionality reduction. Returns reduced vectors.';

CREATE FUNCTION neurondb.reduce_umap(
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

COMMENT ON FUNCTION neurondb.reduce_umap(text, text, integer, integer, double precision, integer) IS
'UMAP dimensionality reduction. Returns reduced vectors.';

CREATE FUNCTION neurondb.train_autoencoder(
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

COMMENT ON FUNCTION neurondb.train_autoencoder(text, text, integer, integer[], double precision, integer) IS
'Train autoencoder for dimensionality reduction. Returns model_id.';

