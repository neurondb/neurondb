'use client'

import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import MCPSSEStream from '@/components/MCPSSEStream'
import SplitPanel from '@/components/SplitPanel'
import { useState } from 'react'
import { CodeBracketIcon, PlayIcon } from '@heroicons/react/24/outline'

export default function MCPHTTPPage() {
  const [endpoint, setEndpoint] = useState('http://localhost:8082/mcp')
  const [requestBody, setRequestBody] = useState('{\n  "method": "tools/list",\n  "params": {}\n}')
  const [response, setResponse] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  
  const handleRequest = async () => {
    setLoading(true)
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: requestBody,
      })
      const data = await res.json()
      setResponse(data)
    } catch (error) {
      setResponse({ error: String(error) })
    } finally {
      setLoading(false)
    }
  }
  
  const leftPanel = (
    <div className="h-full p-4 space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Endpoint</label>
        <input
          type="text"
          value={endpoint}
          onChange={(e) => setEndpoint(e.target.value)}
          className="input"
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Request Body (JSON-RPC 2.0)</label>
        <textarea
          value={requestBody}
          onChange={(e) => setRequestBody(e.target.value)}
          className="input font-mono text-sm"
          rows={15}
        />
      </div>
      
      <button
        onClick={handleRequest}
        disabled={loading}
        className="btn-primary flex items-center gap-2 w-full"
      >
        <PlayIcon className="w-4 h-4" />
        {loading ? 'Sending...' : 'Send Request'}
      </button>
    </div>
  )
  
  const rightPanel = (
    <div className="h-full p-4">
      {response ? (
        <div className="space-y-4">
          <h3 className="font-semibold">Response</h3>
          <pre className="input font-mono text-sm overflow-auto max-h-full">
            {JSON.stringify(response, null, 2)}
          </pre>
        </div>
      ) : (
        <div className="flex items-center justify-center h-full text-slate-500">
          Send a request to see the response
        </div>
      )}
    </div>
  )
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'MCP Console', href: '/mcp' },
            { label: 'HTTP Transport' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            HTTP Transport
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Test MCP protocol over HTTP with SSE streaming
          </p>
        </div>
        
        <div className="h-[calc(100vh-300px)] mb-6">
          <SplitPanel left={leftPanel} right={rightPanel} defaultLeftWidth={50} />
        </div>
        
        <div className="card">
          <h3 className="font-semibold mb-4">SSE Streaming</h3>
          <MCPSSEStream />
        </div>
      </div>
    </MainContent>
  )
}
