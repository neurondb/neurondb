/*-------------------------------------------------------------------------
 *
 * neuronagent_self_improvement_schema.sql
 *    Database schema for self-improvement features
 *
 * Creates tables for execution results, performance feedback, and A/B testing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/sql/neuronagent_self_improvement_schema.sql
 *
 *-------------------------------------------------------------------------
 */

-- Execution results table: Stores execution results for learning
CREATE TABLE IF NOT EXISTS neurondb_agent.execution_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    session_id UUID REFERENCES neurondb_agent.sessions(id) ON DELETE SET NULL,
    user_message TEXT NOT NULL,
    final_answer TEXT,
    success BOOLEAN NOT NULL DEFAULT true,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    tokens_used INTEGER,
    execution_time INTERVAL,
    tool_calls JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_execution_results_agent_id ON neurondb_agent.execution_results(agent_id);
CREATE INDEX IF NOT EXISTS idx_execution_results_session_id ON neurondb_agent.execution_results(session_id);
CREATE INDEX IF NOT EXISTS idx_execution_results_created_at ON neurondb_agent.execution_results(created_at);
CREATE INDEX IF NOT EXISTS idx_execution_results_success ON neurondb_agent.execution_results(success);
CREATE INDEX IF NOT EXISTS idx_execution_results_quality_score ON neurondb_agent.execution_results(quality_score);

COMMENT ON TABLE neurondb_agent.execution_results IS 'Execution results for agent learning and improvement';

-- Performance feedback table: Stores performance feedback
CREATE TABLE IF NOT EXISTS neurondb_agent.performance_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    execution_id UUID REFERENCES neurondb_agent.execution_results(id) ON DELETE SET NULL,
    success BOOLEAN NOT NULL,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    feedback TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_performance_feedback_agent_id ON neurondb_agent.performance_feedback(agent_id);
CREATE INDEX IF NOT EXISTS idx_performance_feedback_execution_id ON neurondb_agent.performance_feedback(execution_id);
CREATE INDEX IF NOT EXISTS idx_performance_feedback_created_at ON neurondb_agent.performance_feedback(created_at);

COMMENT ON TABLE neurondb_agent.performance_feedback IS 'Performance feedback for agent improvement';

-- A/B tests table: Stores A/B test configurations
CREATE TABLE IF NOT EXISTS neurondb_agent.ab_tests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES neurondb_agent.agents(id) ON DELETE CASCADE,
    variant_a JSONB NOT NULL,
    variant_b JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'cancelled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_ab_tests_agent_id ON neurondb_agent.ab_tests(agent_id);
CREATE INDEX IF NOT EXISTS idx_ab_tests_status ON neurondb_agent.ab_tests(status);

COMMENT ON TABLE neurondb_agent.ab_tests IS 'A/B test configurations for agent optimization';

-- A/B test results table: Stores A/B test results
CREATE TABLE IF NOT EXISTS neurondb_agent.ab_test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_id UUID NOT NULL REFERENCES neurondb_agent.ab_tests(id) ON DELETE CASCADE,
    variant TEXT NOT NULL CHECK (variant IN ('A', 'B')),
    success BOOLEAN NOT NULL,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ab_test_results_test_id ON neurondb_agent.ab_test_results(test_id);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_variant ON neurondb_agent.ab_test_results(variant);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_created_at ON neurondb_agent.ab_test_results(created_at);

COMMENT ON TABLE neurondb_agent.ab_test_results IS 'A/B test results for statistical analysis';

