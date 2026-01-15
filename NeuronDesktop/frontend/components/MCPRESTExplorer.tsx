'use client'

import { useState } from 'react'
import { PlayIcon } from '@heroicons/react/24/outline'

export default function MCPRESTExplorer() {
  const [endpoint, setEndpoint] = useState('/api/v1/mcp/tools')
  const [method, setMethod] = useState<'GET' | 'POST'>('GET')
  const [requestBody, setRequestBody] = useState('')
  const [response, setResponse] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  
  const handleRequest = async () => {
    setLoading(true)
    try {
      const options: RequestInit = {
        method,
        headers: { 'Content-Type': 'application/json' },
      }
      if (method === 'POST' && requestBody) {
        options.body = requestBody
      }
      const res = await fetch(endpoint, options)
      const data = await res.json()
      setResponse({ status: res.status, data })
    } catch (error) {
      setResponse({ error: String(error) })
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value as 'GET' | 'POST')}
          className="input w-32"
        >
          <option value="GET">GET</option>
          <option value="POST">POST</option>
        </select>
        <input
          type="text"
          value={endpoint}
          onChange={(e) => setEndpoint(e.target.value)}
          className="input flex-1"
          placeholder="/api/v1/mcp/tools"
        />
        <button
          onClick={handleRequest}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          <PlayIcon className="w-4 h-4" />
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>
      
      {method === 'POST' && (
        <div>
          <label className="block text-sm font-medium mb-2">Request Body (JSON)</label>
          <textarea
            value={requestBody}
            onChange={(e) => setRequestBody(e.target.value)}
            className="input font-mono text-sm"
            rows={10}
          />
        </div>
      )}
      
      {response && (
        <div>
          <label className="block text-sm font-medium mb-2">Response</label>
          <pre className="input font-mono text-sm overflow-auto max-h-96">
            {JSON.stringify(response, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}


