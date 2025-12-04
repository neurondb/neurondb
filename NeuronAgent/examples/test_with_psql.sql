-- Test NeuronAgent with psql
-- Run this in your psql session: \i test_with_psql.sql

-- 1. Check if NeuronAgent tables exist
\echo 'Checking NeuronAgent schema...'
SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables 
WHERE schemaname = 'neurondb_agent'
ORDER BY tablename;

-- 2. Check agents
\echo ''
\echo 'Listing agents...'
SELECT 
    id,
    name,
    model_name,
    enabled_tools,
    created_at
FROM neurondb_agent.agents
ORDER BY created_at DESC
LIMIT 5;

-- 3. Check sessions
\echo ''
\echo 'Listing recent sessions...'
SELECT 
    s.id,
    s.agent_id,
    a.name as agent_name,
    s.external_user_id,
    s.created_at,
    s.last_activity_at
FROM neurondb_agent.sessions s
LEFT JOIN neurondb_agent.agents a ON s.agent_id = a.id
ORDER BY s.created_at DESC
LIMIT 5;

-- 4. Check messages
\echo ''
\echo 'Listing recent messages...'
SELECT 
    m.id,
    m.session_id,
    m.role,
    LEFT(m.content, 50) as content_preview,
    m.token_count,
    m.created_at
FROM neurondb_agent.messages m
ORDER BY m.created_at DESC
LIMIT 10;

-- 5. Check API keys
\echo ''
\echo 'API Keys:'
SELECT 
    key_prefix,
    organization_id,
    user_id,
    rate_limit_per_minute,
    roles,
    created_at
FROM neurondb_agent.api_keys;

-- 6. Check memory chunks (if any)
\echo ''
\echo 'Memory chunks count:'
SELECT 
    COUNT(*) as total_chunks,
    COUNT(DISTINCT agent_id) as agents_with_memory,
    COUNT(DISTINCT session_id) as sessions_with_memory
FROM neurondb_agent.memory_chunks;

-- 7. Check tools
\echo ''
\echo 'Available tools:'
SELECT 
    name,
    description,
    handler_type,
    enabled
FROM neurondb_agent.tools
ORDER BY name;

-- 8. Statistics
\echo ''
\echo 'NeuronAgent Statistics:'
SELECT 
    (SELECT COUNT(*) FROM neurondb_agent.agents) as total_agents,
    (SELECT COUNT(*) FROM neurondb_agent.sessions) as total_sessions,
    (SELECT COUNT(*) FROM neurondb_agent.messages) as total_messages,
    (SELECT COUNT(*) FROM neurondb_agent.memory_chunks) as total_memory_chunks,
    (SELECT COUNT(*) FROM neurondb_agent.api_keys) as total_api_keys;

