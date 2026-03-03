-- =============================================================================
-- Optional: PL/Python functions for in-database NL-to-SQL
-- =============================================================================
-- Requires: CREATE EXTENSION plpython3u; and Python with urllib (standard library).
-- Set env NEURONDB_LLM_SQL_URL (default http://localhost:8000) for model server.
-- =============================================================================

-- Create extension if not present (may require superuser)
-- CREATE EXTENSION IF NOT EXISTS plpython3u;

CREATE OR REPLACE FUNCTION neurondb.nl_to_sql(
  prompt text,
  schema_name text DEFAULT 'public'
)
RETURNS TABLE(sql_text text, explanation text, confidence float)
LANGUAGE plpython3u
AS $$
  import json
  import os
  try:
    import urllib.request
    url = os.environ.get('NEURONDB_LLM_SQL_URL', 'http://localhost:8000').rstrip('/')
    body = json.dumps({
      'model': 'sql-llm-70b',
      'messages': [
        {'role': 'system', 'content': 'You are an expert SQL assistant. Respond with <sql>...</sql> then Explanation: ...'},
        {'role': 'user', 'content': prompt}
      ],
      'temperature': 0.2,
      'max_tokens': 2048,
      'stop': ['</sql>', '\n\nUser:']
    }).encode('utf-8')
    req = urllib.request.Request(url + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=30) as resp:
      data = json.loads(resp.read().decode())
    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
    sql_text = content
    explanation = ''
    confidence = 0.95
    if '<sql>' in content and '</sql>' in content:
      start = content.index('<sql>') + 5
      end = content.index('</sql>')
      sql_text = content[start:end].strip()
      if len(content) > end + 6:
        rest = content[end+6:].strip()
        if rest.lower().startswith('explanation:'):
          explanation = rest[12:].strip()
        else:
          explanation = rest
    return [(sql_text, explanation, confidence)]
  except Exception as e:
    return [('', 'Error: ' + str(e), 0.0)]
$$;

COMMENT ON FUNCTION neurondb.nl_to_sql(text, text) IS
  'Generate SQL from natural language via external model server. Requires plpython3u and NEURONDB_LLM_SQL_URL (default http://localhost:8000).';


CREATE OR REPLACE FUNCTION neurondb.explain_sql(
  sql text,
  detail_level text DEFAULT 'detailed'
)
RETURNS text
LANGUAGE plpython3u
AS $$
  import json
  import os
  try:
    import urllib.request
    url = os.environ.get('NEURONDB_LLM_SQL_URL', 'http://localhost:8000').rstrip('/')
    prompt = f'Explain this SQL query in {detail_level} detail:\n\n{sql}'
    body = json.dumps({
      'model': 'sql-llm-70b',
      'messages': [{'role': 'user', 'content': prompt}],
      'max_tokens': 1024
    }).encode('utf-8')
    req = urllib.request.Request(url + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=30) as resp:
      data = json.loads(resp.read().decode())
    return data.get('choices', [{}])[0].get('message', {}).get('content', '')
  except Exception as e:
    return 'Error: ' + str(e)
$$;

COMMENT ON FUNCTION neurondb.explain_sql(text, text) IS
  'Explain SQL via external model server. Requires plpython3u and NEURONDB_LLM_SQL_URL.';


CREATE OR REPLACE FUNCTION neurondb.optimize_sql(sql text)
RETURNS TABLE(optimized_sql text, explanation text)
LANGUAGE plpython3u
AS $$
  import json
  import os
  try:
    import urllib.request
    url = os.environ.get('NEURONDB_LLM_SQL_URL', 'http://localhost:8000').rstrip('/')
    prompt = 'Optimize this SQL for performance. Provide optimized query and brief explanation:\n\n' + sql
    body = json.dumps({
      'model': 'sql-llm-70b',
      'messages': [{'role': 'user', 'content': prompt}],
      'max_tokens': 2048,
      'stop': ['</sql>']
    }).encode('utf-8')
    req = urllib.request.Request(url + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=30) as resp:
      data = json.loads(resp.read().decode())
    content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
    opt_sql = content
    explanation = ''
    if '<sql>' in content and '</sql>' in content:
      start = content.index('<sql>') + 5
      end = content.index('</sql>')
      opt_sql = content[start:end].strip()
      if len(content) > end + 6:
        explanation = content[end+6:].strip()
    return [(opt_sql, explanation)]
  except Exception as e:
    return [('', 'Error: ' + str(e))]
$$;

COMMENT ON FUNCTION neurondb.optimize_sql(text) IS
  'Suggest optimized SQL via external model server. Requires plpython3u and NEURONDB_LLM_SQL_URL.';
