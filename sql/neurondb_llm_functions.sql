-- -------------------------------------------------------------------------
-- NeuronDB LLM PL/Python Functions (3.1.0) — OPTIONAL, NOT run by CREATE EXTENSION
--
-- Install after the core extension and Python language exist:
--   CREATE EXTENSION plpython3u;
--   \i /path/to/share/extension/neurondb_llm_functions.sql
-- Or: psql -v ON_ERROR_STOP=1 -f "$(pg_config --sharedir)/extension/neurondb_llm_functions.sql"
--
-- Requires: Python package httpx (and any others referenced in functions below).
-- Config: GUCs neurondb.llm_base_url, neurondb.llm_api_key, neurondb.llm_timeout
--         or env LLM_SQL_BASE_URL, LLM_SQL_API_KEY, LLM_SQL_TIMEOUT
-- -------------------------------------------------------------------------

-- =========================================================================
-- 9.1 Internal Helpers
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb._llm_config()
RETURNS TABLE(base_url text, api_key text, timeout_seconds float)
LANGUAGE plpython3u
AS $$
import os
try:
    base = plpy.execute("SELECT current_setting('neurondb.llm_base_url', true) AS v")
    base_url = base[0]['v'] if base and base[0]['v'] else os.environ.get('LLM_SQL_BASE_URL', 'http://localhost:8080')
    api = plpy.execute("SELECT current_setting('neurondb.llm_api_key', true) AS v")
    api_key = api[0]['v'] if api and api[0]['v'] else os.environ.get('LLM_SQL_API_KEY') or ''
    t = plpy.execute("SELECT current_setting('neurondb.llm_timeout', true) AS v")
    timeout = float(t[0]['v']) if t and t[0]['v'] else float(os.environ.get('LLM_SQL_TIMEOUT', '30'))
    return [(base_url.rstrip('/'), api_key, timeout)]
except Exception:
    return [('http://localhost:8080', '', 30.0)]
$$;

CREATE OR REPLACE FUNCTION neurondb._llm_http(
    method text,
    url text,
    body jsonb DEFAULT NULL,
    headers jsonb DEFAULT NULL,
    timeout float DEFAULT 30.0
)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
try:
    import httpx
except ImportError:
    return json.dumps({'error': 'httpx not installed', 'code': 'dependency'})
h = json.loads(headers) if headers else {}
if body:
    try:
        payload = json.loads(body) if isinstance(body, str) else body
    except Exception:
        payload = None
else:
    payload = None
try:
    with httpx.Client(timeout=timeout) as client:
        r = client.request(method, url, headers=h, json=payload)
    if r.status_code >= 400:
        return json.dumps({'error': r.text or ('HTTP %s' % r.status_code), 'code': 'http', 'status_code': r.status_code})
    return r.json() if r.content else {}
except Exception as e:
    return json.dumps({'error': str(e), 'code': 'transport'})
$$;

CREATE OR REPLACE FUNCTION neurondb._llm_provider_config(provider_name text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
rv = plpy.execute("""
    SELECT api_base, api_key_encrypted, default_model, timeout_ms, retry_config, headers, capabilities
    FROM neurondb.llm_providers WHERE name = $1 AND is_active
""", [provider_name])
if not rv:
    return json.dumps({})
row = rv[0]
cfg = {
    'api_base': row['api_base'].rstrip('/') if row['api_base'] else '',
    'api_key': row['api_key_encrypted'] or '',
    'default_model': row['default_model'],
    'timeout_ms': row['timeout_ms'] or 30000,
    'retry_config': row['retry_config'],
    'headers': row['headers'] or {},
    'capabilities': list(row['capabilities']) if row['capabilities'] else []
}
return json.dumps(cfg)
$$;

CREATE OR REPLACE FUNCTION neurondb._llm_resolve_provider(model_name text, capability text DEFAULT 'completion')
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
# Prefer deployment that has this model; else fall back to provider with matching default_model or first active
rv = plpy.execute("""
    SELECT p.name, p.api_base, p.api_key_encrypted, p.timeout_ms, p.headers,
           d.endpoint_model_name
    FROM neurondb.llm_model_deployments d
    JOIN neurondb.llm_providers p ON p.provider_id = d.provider_id AND p.is_active
    JOIN neurondb.llm_models m ON m.model_id = d.model_id
    WHERE m.name = $1 AND d.is_active
    ORDER BY p.priority ASC
    LIMIT 1
""", [model_name])
if rv:
    row = rv[0]
    return json.dumps({
        'provider': row['name'],
        'api_base': row['api_base'].rstrip('/') if row['api_base'] else '',
        'api_key': row['api_key_encrypted'] or '',
        'timeout_ms': row['timeout_ms'] or 30000,
        'headers': dict(row['headers']) if row['headers'] else {},
        'endpoint_model_name': row['endpoint_model_name'] or model_name
    })
# Fallback: GUC / _llm_config
cfg = plpy.execute("SELECT * FROM neurondb._llm_config()")
if cfg:
    r = cfg[0]
    return json.dumps({
        'provider': 'default',
        'api_base': r['base_url'],
        'api_key': r['api_key'] or '',
        'timeout_ms': int(float(r['timeout_seconds']) * 1000),
        'headers': {},
        'endpoint_model_name': model_name
    })
return json.dumps({})
$$;

CREATE OR REPLACE FUNCTION neurondb._llm_track_usage(
    model_name text,
    provider_name text,
    operation text,
    tokens_in integer DEFAULT 0,
    tokens_out integer DEFAULT 0,
    latency_ms integer DEFAULT NULL,
    cost_usd numeric DEFAULT NULL,
    cached boolean DEFAULT false,
    conversation_id bigint DEFAULT NULL
)
RETURNS void
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_token_usage
    (model_name, provider_name, operation, tokens_input, tokens_output, latency_ms, cost_usd, cached, conversation_id)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
""", [model_name, provider_name or None, operation, tokens_in or 0, tokens_out or 0, latency_ms, cost_usd, cached, conversation_id])
$$;

CREATE OR REPLACE FUNCTION neurondb._llm_check_budget(model_name text, user_id text DEFAULT NULL)
RETURNS boolean
LANGUAGE plpython3u
AS $$
uid = user_id or plpy.execute("SELECT current_user")[0]['current_user']
rv = plpy.execute("""
    SELECT budget_id, name, max_cost_usd, current_cost_usd, max_tokens, current_tokens, action_on_exceed
    FROM neurondb.llm_budgets
    WHERE is_active AND (llm_budgets.user_id IS NULL OR llm_budgets.user_id = $1)
      AND (model_name IS NULL OR model_name = $2)
""", [uid, model_name])
for row in rv:
    if row['max_cost_usd'] and row['current_cost_usd'] >= row['max_cost_usd']:
        if row['action_on_exceed'] == 'block':
            return False
    if row['max_tokens'] and row['current_tokens'] >= row['max_tokens']:
        if row['action_on_exceed'] == 'block':
            return False
return True
$$;

CREATE OR REPLACE FUNCTION neurondb._llm_check_guardrails(
    input_text text,
    model_name text,
    direction text DEFAULT 'input'
)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
# direction: 'input' or 'output'
rv = plpy.execute("""
    SELECT guardrail_id, name, guardrail_type, config, action, applies_to
    FROM neurondb.llm_guardrails
    WHERE is_active AND (applies_to IS NULL OR $1 = ANY(applies_to))
""", [model_name or ''])
blocked = False
warnings = []
for row in rv:
    applies = row['applies_to'] is None or (model_name and model_name in (row['applies_to'] or []))
    if not applies:
        continue
    # Simple check: sql_validator could run EXPLAIN; topic_block could check keywords; here we just log
    triggered = False
    if row['guardrail_type'] == 'input_filter' and direction == 'input' and input_text:
        pass  # Could add keyword blocklist from config
    if triggered or (direction == 'input' and row['guardrail_type'] == 'input_filter'):
        pass
    if row['action'] == 'block':
        blocked = True
    warnings.append({'guardrail': row['name'], 'action': row['action']})
return json.dumps({'ok': not blocked, 'blocked': blocked, 'warnings': warnings})
$$;

CREATE OR REPLACE FUNCTION neurondb._llm_format_messages(messages_jsonb jsonb)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
# Normalize messages array for OpenAI-compatible API: [{"role":"user","content":"..."}]
msg = messages_jsonb if isinstance(messages_jsonb, list) else json.loads(messages_jsonb)
out = []
for m in msg:
    if isinstance(m, dict) and 'role' in m and 'content' in m:
        out.append({'role': m['role'], 'content': m.get('content') or ''})
    elif isinstance(m, dict):
        out.append(m)
return json.dumps(out)
$$;

-- =========================================================================
-- 9.2 Model Registry CRUD
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_register_model(
    name text,
    family text DEFAULT NULL,
    architecture jsonb DEFAULT '{}',
    base_model text DEFAULT NULL,
    model_type text DEFAULT 'causal_lm',
    task text DEFAULT NULL,
    description text DEFAULT NULL,
    parameter_count bigint DEFAULT NULL,
    context_length integer DEFAULT NULL,
    format text DEFAULT NULL,
    quantization text DEFAULT NULL,
    storage_mode text DEFAULT 'filesystem',
    filesystem_path text DEFAULT NULL,
    source_url text DEFAULT NULL
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
rv = plpy.execute("""
    INSERT INTO neurondb.llm_models
    (name, family, architecture, base_model, model_type, task, description,
     parameter_count, context_length, format, quantization, storage_mode, filesystem_path, source_url)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
    RETURNING model_id
""", [name, family, architecture, base_model, model_type, task, description,
      parameter_count, context_length, format, quantization, storage_mode, filesystem_path, source_url])
return rv[0]['model_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_update_model(
    model_id_or_name text,
    family text DEFAULT NULL,
    architecture jsonb DEFAULT NULL,
    description text DEFAULT NULL,
    tags text[] DEFAULT NULL,
    is_active boolean DEFAULT NULL
)
RETURNS boolean
LANGUAGE plpython3u
AS $$
# Resolve model_id
try:
    mid = int(model_id_or_name)
except (ValueError, TypeError):
    r = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_id_or_name])
    if not r:
        return False
    mid = r[0]['model_id']
updates = []
args = []
i = 1
if family is not None:
    updates.append("family = $%s" % i); args.append(family); i += 1
if architecture is not None:
    updates.append("architecture = $%s" % i); args.append(architecture); i += 1
if description is not None:
    updates.append("description = $%s" % i); args.append(description); i += 1
if tags is not None:
    updates.append("tags = $%s" % i); args.append(tags); i += 1
if is_active is not None:
    updates.append("is_active = $%s" % i); args.append(is_active); i += 1
if not updates:
    return True
updates.append("updated_at = now()")
args.append(mid)
plpy.execute("UPDATE neurondb.llm_models SET " + ", ".join(updates) + " WHERE model_id = $%s" % i, args)
return True
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_delete_model(model_id_or_name text, cascade boolean DEFAULT true)
RETURNS boolean
LANGUAGE plpython3u
AS $$
try:
    mid = int(model_id_or_name)
except (ValueError, TypeError):
    r = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_id_or_name])
    if not r:
        return False
    mid = r[0]['model_id']
plpy.execute("DELETE FROM neurondb.llm_models WHERE model_id = $1", [mid])
return True
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_list_models(
    family text DEFAULT NULL,
    task text DEFAULT NULL,
    tags text[] DEFAULT NULL,
    format text DEFAULT NULL,
    quantization text DEFAULT NULL
)
RETURNS SETOF neurondb.llm_models
LANGUAGE plpython3u
AS $$
q = "SELECT * FROM neurondb.llm_models WHERE is_active"
args = []
i = 1
if family:
    q += " AND family = $%s" % i; args.append(family); i += 1
if task:
    q += " AND task = $%s" % i; args.append(task); i += 1
if tags:
    q += " AND tags && $%s" % i; args.append(tags); i += 1
if format:
    q += " AND format = $%s" % i; args.append(format); i += 1
if quantization:
    q += " AND quantization = $%s" % i; args.append(quantization); i += 1
q += " ORDER BY name"
return plpy.execute(q, args)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_get_model(model_id_or_name text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
try:
    mid = int(model_id_or_name)
    rv = plpy.execute("SELECT * FROM neurondb.llm_models WHERE model_id = $1", [mid])
except (ValueError, TypeError):
    rv = plpy.execute("SELECT * FROM neurondb.llm_models WHERE name = $1", [model_id_or_name])
if not rv:
    return json.dumps({})
row = dict(rv[0])
for k, v in list(row.items()):
    if hasattr(v, 'isoformat'):
        row[k] = v.isoformat()
    elif hasattr(v, 'tolist'):
        row[k] = v.tolist()
return json.dumps(row)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_model_info(model_name text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
rv = plpy.execute("""
    SELECT m.model_id, m.name, m.family, m.model_type, m.parameter_count, m.context_length,
           m.format, m.quantization, m.storage_mode, m.created_at,
           (SELECT count(*) FROM neurondb.llm_model_files f WHERE f.model_id = m.model_id) AS file_count
    FROM neurondb.llm_models m WHERE m.name = $1
""", [model_name])
if not rv:
    return json.dumps({})
row = dict(rv[0])
for k, v in list(row.items()):
    if hasattr(v, 'isoformat'):
        row[k] = v.isoformat()
return json.dumps(row)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_search_models(query text, limit_count integer DEFAULT 20)
RETURNS SETOF neurondb.llm_models
LANGUAGE plpython3u
AS $$
pat = '%' + query + '%'
return plpy.execute("""
    SELECT * FROM neurondb.llm_models
    WHERE is_active AND (name ILIKE $1 OR description ILIKE $1 OR family ILIKE $1 OR $2 = ANY(tags))
    ORDER BY name LIMIT $3
""", [pat, query, limit_count])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_import_model(source_url text, name text, storage_mode text DEFAULT 'filesystem')
RETURNS bigint
LANGUAGE plpython3u
AS $$
# Metadata-only import; actual file fetch would be done externally
rv = plpy.execute("""
    INSERT INTO neurondb.llm_models (name, source_url, storage_mode)
    VALUES ($1, $2, $3) RETURNING model_id
""", [name, source_url, storage_mode])
return rv[0]['model_id']
$$;

-- =========================================================================
-- 9.3 Model File Management
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_upload_model_file(
    model_name text,
    filename text,
    file_data bytea,
    file_type text DEFAULT 'weight'
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
r = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_name])
if not r:
    plpy.error("Model not found: " + model_name)
mid = r[0]['model_id']
plpy.execute("""
    INSERT INTO neurondb.llm_model_files (model_id, filename, file_type, storage_mode, file_data)
    VALUES ($1, $2, $3, 'database', $4)
    ON CONFLICT (model_id, filename) DO UPDATE SET file_data = EXCLUDED.file_data, storage_mode = 'database'
    RETURNING file_id
""", [mid, filename, file_type, file_data])
return plpy.execute("SELECT file_id FROM neurondb.llm_model_files WHERE model_id = $1 AND filename = $2", [mid, filename])[0]['file_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_register_model_file(
    model_name text,
    filename text,
    filesystem_path text,
    file_type text DEFAULT 'weight',
    size_bytes bigint DEFAULT NULL,
    sha256 text DEFAULT NULL
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
r = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_name])
if not r:
    plpy.error("Model not found: " + model_name)
mid = r[0]['model_id']
plpy.execute("""
    INSERT INTO neurondb.llm_model_files (model_id, filename, file_type, storage_mode, filesystem_path, size_bytes, sha256)
    VALUES ($1, $2, $3, 'filesystem', $4, $5, $6)
    ON CONFLICT (model_id, filename) DO UPDATE SET filesystem_path = EXCLUDED.filesystem_path, size_bytes = EXCLUDED.size_bytes, sha256 = EXCLUDED.sha256
""", [mid, filename, file_type, filesystem_path, size_bytes, sha256])
return plpy.execute("SELECT file_id FROM neurondb.llm_model_files WHERE model_id = $1 AND filename = $2", [mid, filename])[0]['file_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_list_model_files(model_name text)
RETURNS TABLE(file_id bigint, filename text, file_type text, storage_mode text, size_bytes bigint)
LANGUAGE plpython3u
AS $$
r = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_name])
if not r:
    return
mid = r[0]['model_id']
return plpy.execute("SELECT file_id, filename, file_type, storage_mode, size_bytes FROM neurondb.llm_model_files WHERE model_id = $1", [mid])
$$;

-- =========================================================================
-- 9.4 Tokenizer Management (stubs; full impl would load tokenizer)
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_register_tokenizer(
    model_name text,
    tokenizer_type text,
    config jsonb DEFAULT '{}',
    vocab_data bytea DEFAULT NULL
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
r = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_name])
if not r:
    plpy.error("Model not found: " + model_name)
mid = r[0]['model_id']
plpy.execute("""
    INSERT INTO neurondb.llm_tokenizers (model_id, tokenizer_type, tokenizer_config, vocab_data)
    VALUES ($1, $2, $3, $4)
""", [mid, tokenizer_type, config, vocab_data])
return plpy.execute("SELECT tokenizer_id FROM neurondb.llm_tokenizers WHERE model_id = $1 ORDER BY tokenizer_id DESC LIMIT 1", [mid])[0]['tokenizer_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_get_tokenizer_config(model_name text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
r = plpy.execute("SELECT t.tokenizer_config FROM neurondb.llm_tokenizers t JOIN neurondb.llm_models m ON m.model_id = t.model_id WHERE m.name = $1 LIMIT 1", [model_name])
if not r:
    return json.dumps({})
return r[0]['tokenizer_config'] or json.dumps({})
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_count_tokens(text_input text, model_name text)
RETURNS integer
LANGUAGE plpython3u
AS $$
# Stub: real impl would load tokenizer; approximate by word count * 1.3
if not text_input:
    return 0
words = len(text_input.split())
return int(words * 1.3) + 1
$$;

-- =========================================================================
-- 9.5 Adapter Management
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_register_adapter(
    name text,
    model_name text,
    adapter_type text DEFAULT 'lora',
    rank integer DEFAULT NULL,
    alpha float DEFAULT NULL,
    target_modules text[] DEFAULT NULL,
    adapter_data bytea DEFAULT NULL
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
r = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_name])
if not r:
    plpy.error("Model not found: " + model_name)
mid = r[0]['model_id']
plpy.execute("""
    INSERT INTO neurondb.llm_adapters (name, model_id, adapter_type, rank, alpha, target_modules, adapter_data)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
""", [name, mid, adapter_type, rank, alpha, target_modules or [], adapter_data])
return plpy.execute("SELECT adapter_id FROM neurondb.llm_adapters WHERE name = $1", [name])[0]['adapter_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_list_adapters(model_name text DEFAULT NULL)
RETURNS SETOF neurondb.llm_adapters
LANGUAGE plpython3u
AS $$
if model_name:
    return plpy.execute("SELECT a.* FROM neurondb.llm_adapters a JOIN neurondb.llm_models m ON m.model_id = a.model_id WHERE m.name = $1 AND a.is_active", [model_name])
return plpy.execute("SELECT * FROM neurondb.llm_adapters WHERE is_active")
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_delete_adapter(adapter_name text)
RETURNS boolean
LANGUAGE plpython3u
AS $$
plpy.execute("DELETE FROM neurondb.llm_adapters WHERE name = $1", [adapter_name])
return True
$$;

-- =========================================================================
-- 9.6 Version Management
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_create_version(
    model_name text,
    version text,
    change_type text DEFAULT 'initial',
    changelog text DEFAULT NULL
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
r = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_name])
if not r:
    plpy.error("Model not found: " + model_name)
mid = r[0]['model_id']
plpy.execute("UPDATE neurondb.llm_model_versions SET is_current = false WHERE model_id = $1", [mid])
plpy.execute("""
    INSERT INTO neurondb.llm_model_versions (model_id, version, change_type, changelog)
    VALUES ($1, $2, $3, $4) RETURNING version_id
""", [mid, version, change_type, changelog])
return plpy.execute("SELECT version_id FROM neurondb.llm_model_versions WHERE model_id = $1 AND version = $2", [mid, version])[0]['version_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_list_versions(model_name text)
RETURNS TABLE(version_id bigint, version text, change_type text, is_current boolean, created_at timestamptz)
LANGUAGE plpython3u
AS $$
r = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_name])
if not r:
    return
mid = r[0]['model_id']
return plpy.execute("SELECT version_id, version, change_type, is_current, created_at FROM neurondb.llm_model_versions WHERE model_id = $1 ORDER BY created_at DESC", [mid])
$$;

-- =========================================================================
-- 9.7 Provider Management
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_register_provider(
    name text,
    provider_type text,
    api_base text,
    api_key text DEFAULT NULL,
    default_model text DEFAULT NULL,
    timeout_ms integer DEFAULT 30000,
    capabilities text[] DEFAULT ARRAY['completion','chat','embedding']
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_providers (name, provider_type, api_base, api_key_encrypted, default_model, timeout_ms, capabilities)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT (name) DO UPDATE SET api_base = EXCLUDED.api_base, api_key_encrypted = EXCLUDED.api_key_encrypted,
        default_model = EXCLUDED.default_model, timeout_ms = EXCLUDED.timeout_ms, capabilities = EXCLUDED.capabilities
""", [name, provider_type, api_base, api_key, default_model, timeout_ms, capabilities or ['completion','chat','embedding']])
return plpy.execute("SELECT provider_id FROM neurondb.llm_providers WHERE name = $1", [name])[0]['provider_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_list_providers()
RETURNS SETOF neurondb.llm_providers
LANGUAGE plpython3u
AS $$
return plpy.execute("SELECT * FROM neurondb.llm_providers ORDER BY priority, name")
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_update_provider(
    name text,
    api_base text DEFAULT NULL,
    api_key text DEFAULT NULL,
    default_model text DEFAULT NULL,
    timeout_ms integer DEFAULT NULL,
    is_active boolean DEFAULT NULL
)
RETURNS void
LANGUAGE plpython3u
AS $$
updates = []
args = []
i = 1
if api_base is not None:
    updates.append("api_base = $%s" % i); args.append(api_base); i += 1
if api_key is not None:
    updates.append("api_key_encrypted = $%s" % i); args.append(api_key); i += 1
if default_model is not None:
    updates.append("default_model = $%s" % i); args.append(default_model); i += 1
if timeout_ms is not None:
    updates.append("timeout_ms = $%s" % i); args.append(timeout_ms); i += 1
if is_active is not None:
    updates.append("is_active = $%s" % i); args.append(is_active); i += 1
if updates:
    args.append(name)
    plpy.execute("UPDATE neurondb.llm_providers SET " + ", ".join(updates) + " WHERE name = $%s" % i, args)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_delete_provider(provider_name text)
RETURNS boolean
LANGUAGE plpython3u
AS $$
plpy.execute("DELETE FROM neurondb.llm_providers WHERE name = $1", [provider_name])
return True
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_deploy_model(
    model_name text,
    provider_name text,
    endpoint_model_name text DEFAULT NULL
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
mr = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_name])
pr = plpy.execute("SELECT provider_id FROM neurondb.llm_providers WHERE name = $1", [provider_name])
if not mr or not pr:
    plpy.error("Model or provider not found")
mid, pid = mr[0]['model_id'], pr[0]['provider_id']
ename = endpoint_model_name or model_name
plpy.execute("""
    INSERT INTO neurondb.llm_model_deployments (model_id, provider_id, endpoint_model_name)
    VALUES ($1, $2, $3) RETURNING deployment_id
""", [mid, pid, ename])
return plpy.execute("SELECT deployment_id FROM neurondb.llm_model_deployments WHERE model_id = $1 AND provider_id = $2", [mid, pid])[0]['deployment_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_health_check_provider(provider_name text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
try:
    import httpx
except ImportError:
    return json.dumps({'ok': False, 'error': 'httpx not installed'})
cfg = plpy.execute("SELECT api_base, timeout_ms FROM neurondb.llm_providers WHERE name = $1", [provider_name])
if not cfg:
    return json.dumps({'ok': False, 'error': 'Provider not found'})
base = cfg[0]['api_base'].rstrip('/')
timeout = (cfg[0]['timeout_ms'] or 30000) / 1000.0
try:
    with httpx.Client(timeout=min(5.0, timeout)) as c:
        r = c.get(base + '/health' if not base.endswith('health') else base)
    return json.dumps({'ok': r.status_code < 500, 'status_code': r.status_code})
except Exception as e:
    return json.dumps({'ok': False, 'error': str(e)})
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_health_check_all()
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
rv = plpy.execute("SELECT name FROM neurondb.llm_providers WHERE is_active")
result = {}
for row in rv:
    name = row['name']
    r = plpy.execute("SELECT neurondb.llm_health_check_provider($1) AS j", [name])
    result[name] = json.loads(r[0]['j'])
return json.dumps(result)
$$;

-- =========================================================================
-- 9.8 Inference Functions (Core)
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_generate(
    prompt text,
    model_name text DEFAULT NULL,
    temperature float DEFAULT 0.7,
    max_tokens integer DEFAULT 1024,
    top_p float DEFAULT 1.0,
    stop_sequences text[] DEFAULT NULL,
    provider_name text DEFAULT NULL
)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
model = model_name or plpy.execute("SELECT current_setting('neurondb.llm_model', true)")[0].get('current_setting') or 'sql-llm-70b'
cfg = json.loads(plpy.execute("SELECT neurondb._llm_resolve_provider($1) AS j", [model])[0]['j'])
if not cfg or not cfg.get('api_base'):
    fallback = plpy.execute("SELECT * FROM neurondb._llm_config()")
    if fallback:
        cfg = {'api_base': fallback[0]['base_url'], 'api_key': fallback[0]['api_key'], 'endpoint_model_name': model, 'timeout_ms': 30000}
    else:
        return json.dumps({'text': '', 'error': 'No provider configured'})
url = cfg['api_base'] + '/v1/chat/completions'
body = {
    'model': cfg.get('endpoint_model_name') or model,
    'messages': [{'role': 'user', 'content': prompt}],
    'temperature': temperature,
    'max_tokens': max_tokens,
    'top_p': top_p
}
if stop_sequences:
    body['stop'] = stop_sequences
headers = dict(cfg.get('headers') or {})
headers['Content-Type'] = 'application/json'
if cfg.get('api_key'):
    headers['Authorization'] = 'Bearer ' + cfg['api_key']
try:
    import httpx
    with httpx.Client(timeout=cfg.get('timeout_ms', 30000) / 1000.0) as client:
        r = client.post(url, json=body, headers=headers)
    if r.status_code >= 400:
        return json.dumps({'text': '', 'error': r.text or str(r.status_code)})
    data = r.json()
    choice = (data.get('choices') or [{}])[0]
    msg = choice.get('message') or {}
    text = msg.get('content') or ''
    usage = data.get('usage') or {}
    plpy.execute("SELECT neurondb._llm_track_usage($1, $2, 'completion', $3, $4, NULL, NULL, false, NULL)",
                 [model, cfg.get('provider'), usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)])
    return json.dumps({'text': text, 'usage': usage, 'finish_reason': choice.get('finish_reason')})
except Exception as e:
    return json.dumps({'text': '', 'error': str(e)})
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_chat(
    messages_jsonb jsonb,
    model_name text DEFAULT NULL,
    temperature float DEFAULT 0.7,
    max_tokens integer DEFAULT 1024,
    tools jsonb DEFAULT NULL
)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
model = model_name or 'sql-llm-70b'
cfg = json.loads(plpy.execute("SELECT neurondb._llm_resolve_provider($1) AS j", [model])[0]['j'])
if not cfg or not cfg.get('api_base'):
    fallback = plpy.execute("SELECT * FROM neurondb._llm_config()")
    if fallback:
        cfg = {'api_base': fallback[0]['base_url'], 'api_key': fallback[0]['api_key'], 'endpoint_model_name': model, 'timeout_ms': 30000}
    else:
        return json.dumps({'content': '', 'error': 'No provider configured'})
url = cfg['api_base'] + '/v1/chat/completions'
msg = messages_jsonb if isinstance(messages_jsonb, list) else json.loads(messages_jsonb)
body = {'model': cfg.get('endpoint_model_name') or model, 'messages': msg, 'temperature': temperature, 'max_tokens': max_tokens}
if tools:
    body['tools'] = json.loads(tools) if isinstance(tools, str) else tools
headers = dict(cfg.get('headers') or {})
headers['Content-Type'] = 'application/json'
if cfg.get('api_key'):
    headers['Authorization'] = 'Bearer ' + cfg['api_key']
try:
    import httpx
    with httpx.Client(timeout=cfg.get('timeout_ms', 30000) / 1000.0) as client:
        r = client.post(url, json=body, headers=headers)
    if r.status_code >= 400:
        return json.dumps({'content': '', 'error': r.text or str(r.status_code)})
    data = r.json()
    choice = (data.get('choices') or [{}])[0]
    msg_out = choice.get('message') or {}
    content = msg_out.get('content') or ''
    usage = data.get('usage') or {}
    plpy.execute("SELECT neurondb._llm_track_usage($1, $2, 'chat', $3, $4, NULL, NULL, false, NULL)",
                 [model, cfg.get('provider'), usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)])
    return json.dumps({'content': content, 'role': 'assistant', 'usage': usage, 'finish_reason': choice.get('finish_reason'), 'tool_calls': msg_out.get('tool_calls')})
except Exception as e:
    return json.dumps({'content': '', 'error': str(e)})
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_embed(text_input text, model_name text DEFAULT NULL, provider_name text DEFAULT NULL)
RETURNS vector
LANGUAGE plpython3u
AS $$
# Delegates to NeuronDB C embed if available; else HTTP embedding endpoint
# Stub: return zero vector for now; real impl would call ndb_llm_embed or HTTP
try:
    r = plpy.execute("SELECT embed_text($1, $2) AS emb", [text_input, model_name or 'default'])
    if r and r[0].get('emb'):
        return r[0]['emb']
except Exception:
    pass
# Fallback: try OpenAI-compatible /embeddings
import json
model = model_name or 'default'
cfg = json.loads(plpy.execute("SELECT neurondb._llm_resolve_provider($1) AS j", [model])[0]['j'])
if not cfg or not cfg.get('api_base'):
    row = plpy.execute("SELECT * FROM neurondb._llm_config()")
    if row:
        r = row[0]
        cfg = {'api_base': r.get('base_url', ''), 'api_key': r.get('api_key', ''), 'timeout_ms': 30000}
    else:
        plpy.error('No LLM provider or config available')
url = cfg.get('api_base', '').rstrip('/') + '/v1/embeddings'
body = {'input': text_input, 'model': model}
headers = {'Content-Type': 'application/json'}
if cfg.get('api_key'):
    headers['Authorization'] = 'Bearer ' + cfg.get('api_key', '')
try:
    import httpx
    with httpx.Client(timeout=30) as client:
        r = client.post(url, json=body, headers=headers)
    if r.status_code >= 400:
        plpy.error(r.text or str(r.status_code))
    data = r.json()
    emb = (data.get('data') or [{}])[0].get('embedding') or []
    return '[' + ','.join(str(x) for x in emb) + ']'
except Exception as e:
    plpy.error(str(e))
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_rerank(
    query text,
    documents text[],
    model_name text DEFAULT NULL,
    top_k integer DEFAULT 5
)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
# Stub: return first top_k indices; real impl would call rerank API
n = min(top_k, len(documents))
indices = list(range(n))
scores = [1.0 - i * 0.1 for i in range(n)]
return json.dumps({'indices': indices, 'scores': scores})
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_summarize(text_input text, max_length integer DEFAULT 150, model_name text DEFAULT NULL)
RETURNS text
LANGUAGE plpython3u
AS $$
import json
result = plpy.execute("SELECT neurondb.llm_generate($1, $2, 0.3, $3) AS j",
                      ['Summarize the following in at most ' + str(max_length) + ' words:\n\n' + (text_input or ''), model_name, max_length * 2])
if not result:
    return ''
j = json.loads(result[0]['j'])
return j.get('text', '') or j.get('error', '')
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_classify(text_input text, labels text[], model_name text DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
prompt = 'Classify the following text into exactly one of these labels: ' + ', '.join(labels) + '.\n\nText: ' + (text_input or '') + '\n\nLabel:'
result = plpy.execute("SELECT neurondb.llm_generate($1, $2, 0.0, 10) AS j", [prompt, model_name])
if not result:
    return json.dumps({'label': None, 'confidence': 0})
j = json.loads(result[0]['j'])
text = j.get('text', '').strip().lower()
for L in labels:
    if L.lower() in text or text in L.lower():
        return json.dumps({'label': L, 'confidence': 0.9})
return json.dumps({'label': labels[0] if labels else None, 'confidence': 0.5})
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_extract(text_input text, schema_jsonb jsonb, model_name text DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
prompt = 'Extract structured data from the following text according to this schema. Return only valid JSON.\nSchema: ' + json.dumps(schema_jsonb) + '\n\nText: ' + (text_input or '')
return plpy.execute("SELECT neurondb.llm_json($1, $2, $3) AS j", [prompt, schema_jsonb, model_name])[0]['j']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_vision(image_url_or_bytea text, prompt text, model_name text DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
# Stub: real impl would call vision API with image URL or base64
return plpy.execute("SELECT neurondb.llm_generate($1, $2) AS j", [prompt, model_name])[0]['j']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_batch_generate(prompts text[], model_name text DEFAULT NULL, temperature float DEFAULT 0.7, max_tokens integer DEFAULT 1024)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
results = []
for p in (prompts or []):
    r = plpy.execute("SELECT neurondb.llm_generate($1, $2, $3, $4) AS j", [p, model_name, temperature, max_tokens])
    results.append(json.loads(r[0]['j']) if r else {})
return json.dumps(results)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_json(prompt text, output_schema jsonb DEFAULT NULL, model_name text DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
sys_msg = 'Respond with valid JSON only, no other text.'
if output_schema:
    sys_msg += ' Schema: ' + json.dumps(output_schema)
messages = [{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': prompt}]
result = plpy.execute("SELECT neurondb.llm_chat($1, $2, 0.2, 2048) AS j", [json.dumps(messages), model_name])
if not result:
    return json.dumps({})
j = json.loads(result[0]['j'])
content = j.get('content', '').strip()
try:
    return json.loads(content)
except Exception:
    return json.dumps({'raw': content})
$$;

-- =========================================================================
-- 9.9 SQL-Specific Functions (from neuron-llm)
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.nl_to_sql(
    prompt text,
    schema_name text DEFAULT NULL,
    options jsonb DEFAULT NULL
)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
cfg = plpy.execute("SELECT * FROM neurondb._llm_config()")
if not cfg:
    return json.dumps({'sql': '', 'explanation': '', 'confidence': 0.0, 'warnings': [], 'error': 'config unavailable'})
row = cfg[0]
base_url, api_key, timeout = row['base_url'], row['api_key'] or None, row['timeout_seconds']
path = '/api/v1/llm/sql/generate'
url = base_url + path
headers = {'Content-Type': 'application/json'}
if api_key:
    headers['Authorization'] = 'Bearer ' + api_key
body = {'prompt': prompt, 'dialect': 'postgresql', 'schema': schema_name}
if options:
    body['options'] = json.loads(options) if isinstance(options, str) else options
try:
    import httpx
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)
    if r.status_code >= 400:
        return json.dumps({'sql': '', 'explanation': '', 'confidence': 0.0, 'warnings': [], 'error': r.text or str(r.status_code)})
    data = r.json()
    return json.dumps({
        'sql': data.get('sql', ''),
        'explanation': data.get('explanation', ''),
        'confidence': data.get('confidence', 0.0),
        'warnings': data.get('warnings', [])
    })
except Exception as e:
    return json.dumps({'sql': '', 'explanation': '', 'confidence': 0.0, 'warnings': [], 'error': str(e)})
$$;

CREATE OR REPLACE FUNCTION neurondb.explain_sql(sql text, detail_level text DEFAULT 'detailed')
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
cfg = plpy.execute("SELECT * FROM neurondb._llm_config()")
if not cfg:
    return json.dumps({'explanation': '', 'error': 'config unavailable'})
row = cfg[0]
base_url, api_key, timeout = row['base_url'], row['api_key'] or None, row['timeout_seconds']
url = base_url + '/api/v1/llm/sql/explain'
headers = {'Content-Type': 'application/json'}
if api_key:
    headers['Authorization'] = 'Bearer ' + api_key
body = {'sql': sql, 'detail_level': detail_level}
try:
    import httpx
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)
    if r.status_code >= 400:
        return json.dumps({'explanation': '', 'error': r.text or str(r.status_code)})
    data = r.json()
    return json.dumps({'explanation': data.get('explanation', '')})
except Exception as e:
    return json.dumps({'explanation': '', 'error': str(e)})
$$;

CREATE OR REPLACE FUNCTION neurondb.optimize_sql(sql text, schema_context text DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
cfg = plpy.execute("SELECT * FROM neurondb._llm_config()")
if not cfg:
    return json.dumps({'optimized_sql': sql, 'suggestions': [], 'explanation': '', 'error': 'config unavailable'})
row = cfg[0]
base_url, api_key, timeout = row['base_url'], row['api_key'] or None, row['timeout_seconds']
url = base_url + '/api/v1/llm/sql/optimize'
headers = {'Content-Type': 'application/json'}
if api_key:
    headers['Authorization'] = 'Bearer ' + api_key
body = {'sql': sql, 'schema': schema_context}
try:
    import httpx
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)
    if r.status_code >= 400:
        return json.dumps({'optimized_sql': sql, 'suggestions': [], 'explanation': '', 'error': r.text or str(r.status_code)})
    data = r.json()
    return json.dumps({
        'optimized_sql': data.get('optimized_sql', sql),
        'suggestions': data.get('suggestions', []),
        'explanation': data.get('explanation', '')
    })
except Exception as e:
    return json.dumps({'optimized_sql': sql, 'suggestions': [], 'explanation': '', 'error': str(e)})
$$;

CREATE OR REPLACE FUNCTION neurondb.translate_sql(
    sql text,
    source_dialect text DEFAULT 'postgresql',
    target_dialect text DEFAULT 'mysql'
)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
cfg = plpy.execute("SELECT * FROM neurondb._llm_config()")
if not cfg:
    return json.dumps({'translated_sql': '', 'error': 'config unavailable'})
row = cfg[0]
base_url, api_key, timeout = row['base_url'], row['api_key'] or None, row['timeout_seconds']
url = base_url + '/api/v1/llm/sql/translate'
headers = {'Content-Type': 'application/json'}
if api_key:
    headers['Authorization'] = 'Bearer ' + api_key
body = {'sql': sql, 'source_dialect': source_dialect, 'target_dialect': target_dialect}
try:
    import httpx
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)
    if r.status_code >= 400:
        return json.dumps({'translated_sql': '', 'error': r.text or str(r.status_code)})
    data = r.json()
    return json.dumps({'translated_sql': data.get('translated_sql', '')})
except Exception as e:
    return json.dumps({'translated_sql': '', 'error': str(e)})
$$;

CREATE OR REPLACE FUNCTION neurondb.debug_sql(sql text, error_message text DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
cfg = plpy.execute("SELECT * FROM neurondb._llm_config()")
if not cfg:
    return json.dumps({'fixed_sql': sql, 'issues': [], 'explanation': '', 'error': 'config unavailable'})
row = cfg[0]
base_url, api_key, timeout = row['base_url'], row['api_key'] or None, row['timeout_seconds']
url = base_url + '/api/v1/llm/sql/debug'
headers = {'Content-Type': 'application/json'}
if api_key:
    headers['Authorization'] = 'Bearer ' + api_key
body = {'sql': sql, 'error_message': error_message}
try:
    import httpx
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, headers=headers, json=body)
    if r.status_code >= 400:
        return json.dumps({'fixed_sql': sql, 'issues': [], 'explanation': '', 'error': r.text or str(r.status_code)})
    data = r.json()
    return json.dumps({
        'fixed_sql': data.get('fixed_sql', sql),
        'issues': data.get('issues', []),
        'explanation': data.get('explanation', '')
    })
except Exception as e:
    return json.dumps({'fixed_sql': sql, 'issues': [], 'explanation': '', 'error': str(e)})
$$;

CREATE OR REPLACE FUNCTION neurondb.validate_sql(sql text, dialect text DEFAULT 'postgresql')
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
# Trusted single-statement SQL only; EXPLAIN does not execute the query
try:
    plpy.execute("EXPLAIN (FORMAT JSON) " + (sql or ''))
    return json.dumps({'valid': True, 'error': None})
except Exception as e:
    return json.dumps({'valid': False, 'error': str(e)})
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_sql_suggest_index(table_name text, query text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
prompt = 'Suggest a PostgreSQL index for table "' + table_name + '" given this query (return only CREATE INDEX statement):\n' + (query or '')
r = plpy.execute("SELECT neurondb.llm_generate($1, NULL, 0.2, 256) AS j", [prompt])
if not r:
    return json.dumps({'suggestion': None, 'explanation': ''})
j = json.loads(r[0]['j'])
return json.dumps({'suggestion': j.get('text'), 'explanation': ''})
$$;

-- =========================================================================
-- 9.10 Prompt Template and Conversation Functions
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_create_template(
    name text,
    template text,
    system_prompt text DEFAULT NULL,
    variables jsonb DEFAULT '{}',
    model_name text DEFAULT NULL,
    temperature float DEFAULT 0.7,
    max_tokens integer DEFAULT 1024
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_prompt_templates (name, template, system_prompt, variables, model_name, temperature, max_tokens)
    VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (name) DO UPDATE SET template = EXCLUDED.template, system_prompt = EXCLUDED.system_prompt, variables = EXCLUDED.variables, updated_at = now()
""", [name, template, system_prompt, variables, model_name, temperature, max_tokens])
return plpy.execute("SELECT template_id FROM neurondb.llm_prompt_templates WHERE name = $1", [name])[0]['template_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_render_template(template_name text, variables_jsonb jsonb)
RETURNS text
LANGUAGE plpython3u
AS $$
import re
t = plpy.execute("SELECT template FROM neurondb.llm_prompt_templates WHERE name = $1", [template_name])
if not t:
    return ''
template = t[0]['template']
vars = variables_jsonb if isinstance(variables_jsonb, dict) else {}
for k, v in (vars or {}).items():
    template = template.replace('{{' + k + '}}', str(v))
return re.sub(r'\{\{[^}]+\}\}', '', template)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_run_template(template_name text, variables_jsonb jsonb DEFAULT '{}', model_name text DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
rendered = plpy.execute("SELECT neurondb.llm_render_template($1, $2) AS t", [template_name, variables_jsonb])[0]['t']
return plpy.execute("SELECT neurondb.llm_generate($1, $2) AS j", [rendered, model_name])[0]['j']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_list_templates()
RETURNS SETOF neurondb.llm_prompt_templates
LANGUAGE plpython3u
AS $$
return plpy.execute("SELECT * FROM neurondb.llm_prompt_templates ORDER BY name")
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_delete_template(name text)
RETURNS void
LANGUAGE plpython3u
AS $$
plpy.execute("DELETE FROM neurondb.llm_prompt_templates WHERE name = $1", [name])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_create_chain(name text, steps_jsonb jsonb)
RETURNS bigint
LANGUAGE plpython3u
AS $$
plpy.execute("INSERT INTO neurondb.llm_prompt_chains (name, steps) VALUES ($1, $2) ON CONFLICT (name) DO UPDATE SET steps = EXCLUDED.steps", [name, steps_jsonb])
return plpy.execute("SELECT chain_id FROM neurondb.llm_prompt_chains WHERE name = $1", [name])[0]['chain_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_run_chain(chain_name text, initial_input text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
r = plpy.execute("SELECT steps FROM neurondb.llm_prompt_chains WHERE name = $1", [chain_name])
if not r:
    return json.dumps({'output': '', 'error': 'chain not found'})
steps = r[0]['steps'] or []
current = initial_input
for step in steps:
    template_id = step.get('template_id') or step.get('template_name')
    if template_id:
        t = plpy.execute("SELECT name FROM neurondb.llm_prompt_templates WHERE template_id = $1 OR name = $2", [template_id, template_id])
        if t:
            current = plpy.execute("SELECT neurondb.llm_render_template($1, $2) AS t", [t[0]['name'], json.dumps({'input': current})])[0]['t']
return plpy.execute("SELECT neurondb.llm_generate($1) AS j", [current])[0]['j']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_create_conversation(title text DEFAULT NULL, model_name text DEFAULT NULL, system_prompt text DEFAULT NULL)
RETURNS bigint
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_conversations (title, model_name, system_prompt) VALUES ($1, $2, $3)
""", [title, model_name, system_prompt])
return plpy.execute("SELECT conversation_id FROM neurondb.llm_conversations ORDER BY conversation_id DESC LIMIT 1")[0]['conversation_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_send_message(conversation_id bigint, content text, role text DEFAULT 'user')
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
plpy.execute("INSERT INTO neurondb.llm_messages (conversation_id, role, content) VALUES ($1, $2, $3)", [conversation_id, role, content])
conv = plpy.execute("SELECT model_name FROM neurondb.llm_conversations WHERE conversation_id = $1", [conversation_id])
model = conv[0]['model_name'] if conv else None
msgs = plpy.execute("SELECT role, content FROM neurondb.llm_messages WHERE conversation_id = $1 ORDER BY message_id", [conversation_id])
messages = [{'role': r['role'], 'content': r['content']} for r in msgs]
resp = plpy.execute("SELECT neurondb.llm_chat($1, $2) AS j", [json.dumps(messages), model])
if not resp:
    return json.dumps({'content': ''})
j = json.loads(resp[0]['j'])
assistant_content = j.get('content', '')
plpy.execute("INSERT INTO neurondb.llm_messages (conversation_id, role, content) VALUES ($1, 'assistant', $2)", [conversation_id, assistant_content])
return resp[0]['j']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_get_conversation(conversation_id bigint)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
msgs = plpy.execute("SELECT message_id, role, content, created_at FROM neurondb.llm_messages WHERE conversation_id = $1 ORDER BY message_id", [conversation_id])
out = []
for r in msgs:
    out.append({'role': r['role'], 'content': r['content'], 'created_at': r['created_at'].isoformat() if hasattr(r['created_at'], 'isoformat') else str(r['created_at'])})
return json.dumps(out)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_list_conversations(limit_count integer DEFAULT 50, off integer DEFAULT 0)
RETURNS SETOF neurondb.llm_conversations
LANGUAGE plpython3u
AS $$
return plpy.execute("SELECT * FROM neurondb.llm_conversations ORDER BY updated_at DESC LIMIT $1 OFFSET $2", [limit_count, off])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_delete_conversation(conversation_id bigint)
RETURNS void
LANGUAGE plpython3u
AS $$
plpy.execute("DELETE FROM neurondb.llm_conversations WHERE conversation_id = $1", [conversation_id])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_fork_conversation(conversation_id bigint, from_message_id bigint DEFAULT NULL)
RETURNS bigint
LANGUAGE plpython3u
AS $$
conv = plpy.execute("SELECT title, model_name, system_prompt FROM neurondb.llm_conversations WHERE conversation_id = $1", [conversation_id])
if not conv:
    return None
plpy.execute("INSERT INTO neurondb.llm_conversations (title, model_name, system_prompt) VALUES ($1, $2, $3)", [conv[0]['title'], conv[0]['model_name'], conv[0]['system_prompt']])
new_id = plpy.execute("SELECT conversation_id FROM neurondb.llm_conversations ORDER BY conversation_id DESC LIMIT 1")[0]['conversation_id']
plpy.execute("""
    INSERT INTO neurondb.llm_messages (conversation_id, role, content)
    SELECT $1, role, content FROM neurondb.llm_messages
    WHERE conversation_id = $2 AND ($3::bigint IS NULL OR message_id <= $3)
""", [new_id, conversation_id, from_message_id])
return new_id
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_summarize_conversation(conversation_id bigint)
RETURNS text
LANGUAGE plpython3u
AS $$
msgs = plpy.execute("SELECT role, content FROM neurondb.llm_messages WHERE conversation_id = $1 ORDER BY message_id", [conversation_id])
text = '\n'.join((m['role'] + ': ' + (m['content'] or '')) for m in msgs)
return plpy.execute("SELECT (neurondb.llm_summarize($1))::text AS t", [text])[0]['t']
$$;

-- =========================================================================
-- 9.12 RAG Functions
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_rag_query(
    query text,
    table_name text,
    embedding_column text DEFAULT 'embedding',
    content_column text DEFAULT 'content',
    model_name text DEFAULT NULL,
    top_k integer DEFAULT 5
)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
# Use quote_ident to safely build query
tq = plpy.execute("SELECT quote_ident($1) AS q", [table_name])[0]['q']
cq = plpy.execute("SELECT quote_ident($1) AS q", [content_column])[0]['q']
eq = plpy.execute("SELECT quote_ident($1) AS q", [embedding_column])[0]['q']
try:
    emb = plpy.execute("SELECT embed_text($1, $2) AS e", [query, model_name or 'default'])
    if not emb or not emb[0].get('e'):
        return json.dumps({'answer': '', 'sources': [], 'error': 'embedding failed'})
    vec = emb[0]['e']
    sql = "SELECT " + cq + " FROM " + tq + " ORDER BY " + eq + " <-> $1 LIMIT $2"
    rows = plpy.execute(sql, [vec, top_k])
except Exception as e:
    return json.dumps({'answer': '', 'sources': [], 'error': str(e)})
sources = [r[content_column] for r in rows] if rows else []
context = '\n\n'.join(sources)
prompt = 'Based on the following context, answer the question. Context:\n' + context + '\n\nQuestion: ' + query
result = plpy.execute("SELECT neurondb.llm_generate($1, $2) AS j", [prompt, model_name])
if not result:
    return json.dumps({'answer': '', 'sources': sources})
j = json.loads(result[0]['j'])
return json.dumps({'answer': j.get('text', ''), 'sources': sources})
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_create_rag_pipeline(name text, config_jsonb jsonb)
RETURNS void
LANGUAGE plpython3u
AS $$
# Store in a table if we had one; for now no-op
pass
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_run_rag_pipeline(pipeline_name text, query text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
return plpy.execute("SELECT neurondb.llm_rag_query($1, 'documents', 'embedding', 'content', NULL, 5) AS j", [query])[0]['j']
$$;

-- =========================================================================
-- 9.13 Guardrail Functions
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_create_guardrail(
    name text,
    guardrail_type text,
    config jsonb,
    action text DEFAULT 'block'
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_guardrails (name, guardrail_type, config, action)
    VALUES ($1, $2, $3, $4) ON CONFLICT (name) DO UPDATE SET config = EXCLUDED.config, action = EXCLUDED.action
""", [name, guardrail_type, config, action])
return plpy.execute("SELECT guardrail_id FROM neurondb.llm_guardrails WHERE name = $1", [name])[0]['guardrail_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_list_guardrails()
RETURNS SETOF neurondb.llm_guardrails
LANGUAGE plpython3u
AS $$
return plpy.execute("SELECT * FROM neurondb.llm_guardrails WHERE is_active")
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_validate_input(text_input text, model_name text DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
return plpy.execute("SELECT neurondb._llm_check_guardrails($1, $2, 'input') AS j", [text_input, model_name])[0]['j']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_validate_output(text_input text, model_name text DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
return plpy.execute("SELECT neurondb._llm_check_guardrails($1, $2, 'output') AS j", [text_input, model_name])[0]['j']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_delete_guardrail(name text)
RETURNS void
LANGUAGE plpython3u
AS $$
plpy.execute("DELETE FROM neurondb.llm_guardrails WHERE name = $1", [name])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_redact_pii(text_input text)
RETURNS text
LANGUAGE plpython3u
AS $$
import re
# Simple stub: redact email-like and 10+ digit numbers
text = text_input or ''
text = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w+\b', '[EMAIL]', text)
text = re.sub(r'\b\d{10,}\b', '[NUMBER]', text)
return text
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_detect_sql_injection(sql text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
danger = ['; DROP', '; DELETE', '1=1', 'UNION SELECT', 'pg_sleep', 'information_schema']
sql_upper = (sql or '').upper()
found = [p for p in danger if p.upper() in sql_upper or p in (sql or '')]
return json.dumps({'suspicious': len(found) > 0, 'patterns': found})
$$;

-- =========================================================================
-- 9.14 Training and Fine-tuning Tracking
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_start_training_run(
    model_name text,
    run_type text,
    config jsonb,
    dataset_name text DEFAULT NULL
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
mid = None
if model_name:
    r = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_name])
    if r:
        mid = r[0]['model_id']
plpy.execute("""
    INSERT INTO neurondb.llm_training_runs (model_id, run_type, config, dataset_name, status, started_at)
    VALUES ($1, $2, $3, $4, 'running', now())
""", [mid, run_type, config, dataset_name])
return plpy.execute("SELECT run_id FROM neurondb.llm_training_runs ORDER BY run_id DESC LIMIT 1")[0]['run_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_log_training_metric(run_id bigint, step_num bigint, metric_name text, metric_value float)
RETURNS void
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_training_metrics (run_id, step, metric_name, metric_value) VALUES ($1, $2, $3, $4)
""", [run_id, step_num, metric_name, metric_value])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_log_training_metrics_batch(run_id bigint, step_num bigint, metrics_jsonb jsonb)
RETURNS void
LANGUAGE plpython3u
AS $$
import json
metrics = metrics_jsonb if isinstance(metrics_jsonb, dict) else (json.loads(metrics_jsonb) if metrics_jsonb else {})
for k, v in (metrics or {}).items():
    if isinstance(v, (int, float)):
        plpy.execute("INSERT INTO neurondb.llm_training_metrics (run_id, step, metric_name, metric_value) VALUES ($1, $2, $3, $4)", [run_id, step_num, k, float(v)])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_training_curve(run_id bigint, metric_name text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
r = plpy.execute("SELECT step, metric_value FROM neurondb.llm_training_metrics WHERE run_id = $1 AND metric_name = $2 ORDER BY step", [run_id, metric_name])
return json.dumps([{'step': row['step'], 'value': row['metric_value']} for row in r])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_complete_training_run(run_id bigint, final_metrics jsonb DEFAULT NULL)
RETURNS void
LANGUAGE plpython3u
AS $$
plpy.execute("""
    UPDATE neurondb.llm_training_runs SET status = 'completed', completed_at = now(), final_metrics = $2 WHERE run_id = $1
""", [run_id, final_metrics])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_fail_training_run(run_id bigint, error_message text)
RETURNS void
LANGUAGE plpython3u
AS $$
plpy.execute("""
    UPDATE neurondb.llm_training_runs SET status = 'failed', error_message = $2 WHERE run_id = $1
""", [run_id, error_message])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_save_checkpoint(run_id bigint, step_num bigint, path text, metrics jsonb DEFAULT NULL)
RETURNS void
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_checkpoints (run_id, step, filesystem_path, metrics) VALUES ($1, $2, $3, $4)
""", [run_id, step_num, path, metrics or '{}'])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_get_training_run(run_id bigint)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
r = plpy.execute("SELECT * FROM neurondb.llm_training_runs WHERE run_id = $1", [run_id])
if not r:
    return json.dumps({})
row = dict(r[0])
metrics = plpy.execute("SELECT step, metric_name, metric_value FROM neurondb.llm_training_metrics WHERE run_id = $1 ORDER BY step", [run_id])
row['metrics'] = [dict(m) for m in metrics]
for k, v in list(row.items()):
    if hasattr(v, 'isoformat'):
        row[k] = v.isoformat()
return json.dumps(row)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_list_training_runs(model_name text DEFAULT NULL, status_filter text DEFAULT NULL)
RETURNS SETOF neurondb.llm_training_runs
LANGUAGE plpython3u
AS $$
if model_name:
    return plpy.execute("SELECT r.* FROM neurondb.llm_training_runs r JOIN neurondb.llm_models m ON m.model_id = r.model_id WHERE m.name = $1 AND ($2 IS NULL OR r.status = $2) ORDER BY r.run_id DESC", [model_name, status_filter])
return plpy.execute("SELECT * FROM neurondb.llm_training_runs WHERE $1 IS NULL OR status = $1 ORDER BY run_id DESC", [status_filter])
$$;

-- =========================================================================
-- 9.15 Dataset Management
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_register_dataset(
    name text,
    source text DEFAULT NULL,
    description text DEFAULT NULL,
    filesystem_path text DEFAULT NULL,
    num_examples bigint DEFAULT NULL
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_datasets (name, source, description, filesystem_path, num_examples)
    VALUES ($1, $2, $3, $4, $5) ON CONFLICT (name) DO NOTHING
""", [name, source, description, filesystem_path, num_examples])
r = plpy.execute("SELECT dataset_id FROM neurondb.llm_datasets WHERE name = $1", [name])
return r[0]['dataset_id'] if r else None
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_list_datasets()
RETURNS SETOF neurondb.llm_datasets
LANGUAGE plpython3u
AS $$
return plpy.execute("SELECT * FROM neurondb.llm_datasets ORDER BY name")
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_dataset_stats(name text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
r = plpy.execute("SELECT * FROM neurondb.llm_datasets WHERE name = $1", [name])
if not r:
    return json.dumps({})
row = dict(r[0])
for k, v in list(row.items()):
    if hasattr(v, 'isoformat'):
        row[k] = v.isoformat()
return json.dumps(row)
$$;

-- =========================================================================
-- 9.16 Evaluation Functions
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_create_benchmark(name text, dataset_id bigint DEFAULT NULL, metric_names text[] DEFAULT ARRAY['exact_match'])
RETURNS bigint
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_benchmarks (name, dataset_id, metric_names) VALUES ($1, $2, $3)
""", [name, dataset_id, metric_names or ['exact_match']])
return plpy.execute("SELECT benchmark_id FROM neurondb.llm_benchmarks WHERE name = $1", [name])[0]['benchmark_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_evaluate_model(model_name text, benchmark_name text, options jsonb DEFAULT NULL)
RETURNS bigint
LANGUAGE plpython3u
AS $$
import json
mr = plpy.execute("SELECT model_id FROM neurondb.llm_models WHERE name = $1", [model_name])
br = plpy.execute("SELECT benchmark_id FROM neurondb.llm_benchmarks WHERE name = $1", [benchmark_name])
if not mr or not br:
    return None
plpy.execute("""
    INSERT INTO neurondb.llm_eval_results (model_id, benchmark_id, metrics, num_total)
    VALUES ($1, $2, $3, 0)
""", [mr[0]['model_id'], br[0]['benchmark_id'], json.dumps({})])
return plpy.execute("SELECT result_id FROM neurondb.llm_eval_results ORDER BY result_id DESC LIMIT 1")[0]['result_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_compare_models(model_names text[], benchmark_name text DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
out = []
for name in (model_names or []):
    r = plpy.execute("SELECT * FROM neurondb.llm_eval_results($1, $2)", [name, benchmark_name])
    out.append({'model': name, 'results': [dict(x) for x in r] if r else []})
return json.dumps(out)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_leaderboard(benchmark_name text, metric_name text DEFAULT 'exact_match', limit_count integer DEFAULT 10)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
r = plpy.execute("""
    SELECT m.name, er.metrics FROM neurondb.llm_eval_results er
    JOIN neurondb.llm_models m ON m.model_id = er.model_id
    JOIN neurondb.llm_benchmarks b ON b.benchmark_id = er.benchmark_id
    WHERE b.name = $1 ORDER BY (er.metrics->>$2)::float DESC NULLS LAST LIMIT $3
""", [benchmark_name, metric_name, limit_count])
return json.dumps([{'model': row['name'], 'metrics': dict(row['metrics']) if row['metrics'] else {}} for row in r])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_eval_results(model_name text, benchmark_name text DEFAULT NULL)
RETURNS SETOF neurondb.llm_eval_results
LANGUAGE plpython3u
AS $$
if benchmark_name:
    return plpy.execute("""
        SELECT er.* FROM neurondb.llm_eval_results er
        JOIN neurondb.llm_models m ON m.model_id = er.model_id
        JOIN neurondb.llm_benchmarks b ON b.benchmark_id = er.benchmark_id
        WHERE m.name = $1 AND b.name = $2 ORDER BY er.evaluated_at DESC
    """, [model_name, benchmark_name])
return plpy.execute("""
    SELECT er.* FROM neurondb.llm_eval_results er
    JOIN neurondb.llm_models m ON m.model_id = er.model_id WHERE m.name = $1 ORDER BY er.evaluated_at DESC
""", [model_name])
$$;

-- =========================================================================
-- 9.17 Cost and Usage Functions
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_set_pricing(
    model_name text,
    provider_name text,
    input_cost_per_1k numeric DEFAULT 0,
    output_cost_per_1k numeric DEFAULT 0
)
RETURNS void
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_pricing (model_name, provider_name, input_cost_per_1k, output_cost_per_1k)
    VALUES ($1, $2, $3, $4)
""", [model_name, provider_name, input_cost_per_1k, output_cost_per_1k])
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_create_budget(
    name text,
    max_cost_usd numeric DEFAULT NULL,
    max_tokens bigint DEFAULT NULL,
    period text DEFAULT 'monthly',
    user_id text DEFAULT NULL
)
RETURNS bigint
LANGUAGE plpython3u
AS $$
plpy.execute("""
    INSERT INTO neurondb.llm_budgets (name, max_cost_usd, max_tokens, period, user_id)
    VALUES ($1, $2, $3, $4, $5)
""", [name, max_cost_usd, max_tokens, period, user_id])
return plpy.execute("SELECT budget_id FROM neurondb.llm_budgets WHERE name = $1", [name])[0]['budget_id']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_usage_summary(
    period text DEFAULT 'day',
    model_name text DEFAULT NULL,
    user_id text DEFAULT NULL
)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
interval_map = {'day': '1 day', 'week': '7 days', 'month': '30 days'}
iv = interval_map.get(period, '1 day')
q = "SELECT count(*) AS requests, sum(tokens_input + tokens_output) AS total_tokens, sum(coalesce(cost_usd, 0)) AS total_cost FROM neurondb.llm_token_usage WHERE created_at > now() - $1::interval"
args = [iv]
if model_name:
    q += " AND model_name = $2"; args.append(model_name)
if user_id:
    q += " AND user_id = $3"; args.append(user_id)
r = plpy.execute(q, args)
if not r:
    return json.dumps({'requests': 0, 'total_tokens': 0, 'total_cost': 0})
row = r[0]
return json.dumps({'requests': row['requests'], 'total_tokens': row['total_tokens'] or 0, 'total_cost': float(row['total_cost'] or 0)})
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_budget_status(budget_name text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
r = plpy.execute("SELECT * FROM neurondb.llm_budget_overview WHERE name = $1", [budget_name])
if not r:
    return json.dumps({})
row = dict(r[0])
for k, v in list(row.items()):
    if hasattr(v, 'isoformat'):
        row[k] = v.isoformat()
return json.dumps(row)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_cost_report(start_date date DEFAULT NULL, end_date date DEFAULT NULL)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
if not start_date:
    start_date = plpy.execute("SELECT (now() - interval '30 days')::date AS d")[0]['d']
if not end_date:
    end_date = plpy.execute("SELECT current_date AS d")[0]['d']
r = plpy.execute("""
    SELECT model_name, count(*) AS requests, sum(tokens_input + tokens_output) AS total_tokens, sum(coalesce(cost_usd, 0)) AS total_cost
    FROM neurondb.llm_token_usage WHERE created_at::date BETWEEN $1 AND $2 GROUP BY model_name
""", [start_date, end_date])
return json.dumps([dict(row) for row in r])
$$;

-- =========================================================================
-- 9.18 Utility and Diagnostic Functions
-- =========================================================================

CREATE OR REPLACE FUNCTION neurondb.llm_healthcheck()
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
result = {'ok': False, 'plpython': True, 'httpx': False, 'endpoint_reachable': False, 'base_url': None, 'error': None}
try:
    import httpx
    result['httpx'] = True
except ImportError:
    result['error'] = 'httpx not installed'
    return json.dumps(result)
cfg = plpy.execute("SELECT * FROM neurondb._llm_config()")
if not cfg:
    result['error'] = 'config unavailable'
    return json.dumps(result)
row = cfg[0]
base_url, api_key, timeout = row['base_url'], row['api_key'] or None, row['timeout_seconds']
result['base_url'] = base_url
path = base_url + '/api/v1/llm/sql/models' if '/api' not in base_url else base_url.rstrip('/') + '/models'
headers = {}
if api_key:
    headers['Authorization'] = 'Bearer ' + api_key
try:
    with httpx.Client(timeout=min(5.0, timeout)) as client:
        r = client.get(path, headers=headers)
    result['endpoint_reachable'] = r.status_code < 500
    result['status_code'] = r.status_code
    result['ok'] = result['httpx'] and result['endpoint_reachable']
except Exception as e:
    result['error'] = str(e)
return json.dumps(result)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_version()
RETURNS text
LANGUAGE plpython3u
AS $$
return '3.1.0'
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_capabilities()
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
r = plpy.execute("SELECT name, capabilities FROM neurondb.llm_providers WHERE is_active")
caps = {}
for row in r:
    caps[row['name']] = list(row['capabilities']) if row['capabilities'] else []
return json.dumps(caps)
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_cache_clear(model_name text DEFAULT NULL)
RETURNS integer
LANGUAGE plpython3u
AS $$
if model_name:
    r = plpy.execute("DELETE FROM neurondb.llm_cache WHERE key LIKE $1", ['%' + model_name + '%'])
else:
    plpy.execute("DELETE FROM neurondb.llm_cache")
return 1
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_cache_stats()
RETURNS jsonb
LANGUAGE plpython3u
AS $$
import json
r = plpy.execute("SELECT count(*) AS cnt FROM neurondb.llm_cache")
return json.dumps({'entries': r[0]['cnt']})
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_export_model_config(model_name text)
RETURNS jsonb
LANGUAGE plpython3u
AS $$
return plpy.execute("SELECT neurondb.llm_get_model($1) AS j", [model_name])[0]['j']
$$;

CREATE OR REPLACE FUNCTION neurondb.llm_import_from_huggingface(repo_id text, revision text DEFAULT 'main', storage_mode text DEFAULT 'filesystem')
RETURNS bigint
LANGUAGE plpython3u
AS $$
url = 'https://huggingface.co/' + repo_id.replace('/', '/') + '/resolve/' + revision
return plpy.execute("SELECT neurondb.llm_import_model($1, $2, $3) AS id", [url, repo_id.replace('/', '_'), storage_mode])[0]['id']
$$;
