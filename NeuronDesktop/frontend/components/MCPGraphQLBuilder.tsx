'use client'

import { useState } from 'react'
import { PlayIcon } from '@heroicons/react/24/outline'
import JSONViewer from './JSONViewer'

export default function MCPGraphQLBuilder() {
  const [query, setQuery] = useState(`{
  tools {
    name
    description
    inputSchema
  }
}`)
  const [variables, setVariables] = useState('{}')
  const [response, setResponse] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  
  const handleQuery = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/v1/mcp/graphql', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          variables: JSON.parse(variables || '{}'),
        }),
      })
      const data = await res.json()
      setResponse(data)
    } catch (error) {
      setResponse({ error: String(error) })
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">GraphQL Query</label>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="input font-mono text-sm"
          rows={10}
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Variables (JSON)</label>
        <textarea
          value={variables}
          onChange={(e) => setVariables(e.target.value)}
          className="input font-mono text-sm"
          rows={5}
        />
      </div>
      
      <button
        onClick={handleQuery}
        disabled={loading}
        className="btn-primary flex items-center gap-2"
      >
        <PlayIcon className="w-4 h-4" />
        {loading ? 'Executing...' : 'Execute Query'}
      </button>
      
      {response && (
        <div>
          <label className="block text-sm font-medium mb-2">Response</label>
          <JSONViewer data={response} />
        </div>
      )}
    </div>
  )
}


