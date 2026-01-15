'use client'

import { useState } from 'react'
import { PlayIcon } from '@heroicons/react/24/outline'
import JSONViewer from './JSONViewer'

interface MCPToolTesterProps {
  toolName: string
  inputSchema: Record<string, any>
  onExecute: (params: Record<string, any>) => Promise<any>
}

export default function MCPToolTester({
  toolName,
  inputSchema,
  onExecute,
}: MCPToolTesterProps) {
  const [params, setParams] = useState<Record<string, any>>({})
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const handleExecute = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await onExecute(params)
      setResult(response)
    } catch (err: any) {
      setError(err.message || 'Execution failed')
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold mb-2">Test Tool: {toolName}</h3>
        <div className="space-y-2">
          {Object.entries(inputSchema.properties || {}).map(([key, schema]: [string, any]) => (
            <div key={key}>
              <label className="block text-sm font-medium mb-1">{key}</label>
              <input
                type="text"
                value={params[key] || ''}
                onChange={(e) => setParams({ ...params, [key]: e.target.value })}
                className="input"
                placeholder={schema.description || key}
              />
            </div>
          ))}
        </div>
        <button
          onClick={handleExecute}
          disabled={loading}
          className="btn-primary mt-4 flex items-center gap-2"
        >
          <PlayIcon className="w-4 h-4" />
          {loading ? 'Executing...' : 'Execute'}
        </button>
      </div>
      
      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-red-800 dark:text-red-200">{error}</p>
        </div>
      )}
      
      {result && (
        <div>
          <h4 className="font-semibold mb-2">Result</h4>
          <JSONViewer data={result} />
        </div>
      )}
    </div>
  )
}


