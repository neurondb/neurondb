'use client'

import { useState } from 'react'

interface KnowledgeRouterProps {
  profileId: string
  agentId: string
}

export default function KnowledgeRouter({ profileId, agentId }: KnowledgeRouterProps) {
  const [query, setQuery] = useState('')
  const [routing, setRouting] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const testRouting = async () => {
    if (!query.trim()) return

    setLoading(true)
    try {
      // This would call the knowledge routing tool via MCP or agent API
      // For now, showing a placeholder
      setTimeout(() => {
        setRouting({
          sources: ['vector_db', 'web'],
          confidence: 0.85,
          reasoning: 'Query requires both vector database search and web search for comprehensive results',
        })
        setLoading(false)
      }, 1000)
    } catch (err) {
      setLoading(false)
    }
  }

  return (
    <div className="card p-6">
      <h3 className="text-lg font-semibold mb-4">Knowledge Routing</h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        Test how queries are routed to different knowledge sources
      </p>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Query</label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
            rows={3}
            placeholder="Enter a query to test routing..."
          />
        </div>

        <button
          onClick={testRouting}
          disabled={loading || !query.trim()}
          className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Routing...' : 'Test Routing'}
        </button>

        {routing && (
          <div className="mt-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-md">
            <div className="mb-2">
              <span className="font-semibold">Recommended Sources: </span>
              <span className="text-purple-600 dark:text-purple-400">
                {routing.sources.join(', ')}
              </span>
            </div>
            <div className="mb-2">
              <span className="font-semibold">Confidence: </span>
              <span>{(routing.confidence * 100).toFixed(1)}%</span>
            </div>
            <div>
              <span className="font-semibold">Reasoning: </span>
              <span className="text-slate-600 dark:text-slate-400">{routing.reasoning}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
