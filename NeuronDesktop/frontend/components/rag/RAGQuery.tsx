'use client'

import { useState } from 'react'
import { ragAPI, type CollectionInfo, type RAGQueryRequest } from '@/lib/api'
import LoadingSpinner from '@/components/LoadingSpinner'
import ContextViewer from './ContextViewer'
import { MagnifyingGlassIcon } from '@/components/Icons'

interface RAGQueryProps {
  profileId: string
  collections: CollectionInfo[]
}

export default function RAGQuery({ profileId, collections }: RAGQueryProps) {
  const [query, setQuery] = useState('')
  const [selectedCollection, setSelectedCollection] = useState<string>('')
  const [vectorCol, setVectorCol] = useState('embedding')
  const [textCol, setTextCol] = useState('content')
  const [model, setModel] = useState('default')
  const [topK, setTopK] = useState(5)
  const [rerank, setRerank] = useState(false)
  const [hybrid, setHybrid] = useState(false)
  const [temporal, setTemporal] = useState(false)
  const [faceted, setFaceted] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string>('')

  const handleQuery = async () => {
    if (!profileId || !selectedCollection || !query.trim()) {
      setError('Please fill in all required fields')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const request: RAGQueryRequest = {
        query,
        table_name: selectedCollection,
        vector_col: vectorCol,
        text_col: textCol,
        model,
        top_k: topK,
        rerank,
        hybrid,
        temporal,
        faceted,
      }

      const response = await ragAPI.query(profileId, request)
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to execute RAG query')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-slate-900 dark:text-slate-100">
          RAG Query
        </h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Collection (Table)
            </label>
            <select
              value={selectedCollection}
              onChange={(e) => setSelectedCollection(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
            >
              <option value="">Select a collection</option>
              {collections.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.schema ? `${col.schema}.${col.name}` : col.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Query
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
              placeholder="Enter your question..."
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Vector Column
              </label>
              <input
                type="text"
                value={vectorCol}
                onChange={(e) => setVectorCol(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Text Column
              </label>
              <input
                type="text"
                value={textCol}
                onChange={(e) => setTextCol(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Model
              </label>
              <input
                type="text"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Top K
              </label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 5)}
                min={1}
                max={50}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
              />
            </div>
          </div>

          <div className="flex flex-wrap gap-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={rerank}
                onChange={(e) => setRerank(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm text-slate-700 dark:text-slate-300">Rerank Results</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={hybrid}
                onChange={(e) => setHybrid(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm text-slate-700 dark:text-slate-300">Hybrid Search</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={temporal}
                onChange={(e) => setTemporal(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm text-slate-700 dark:text-slate-300">Temporal</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={faceted}
                onChange={(e) => setFaceted(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm text-slate-700 dark:text-slate-300">Faceted</span>
            </label>
          </div>

          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
              <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
            </div>
          )}

          <button
            onClick={handleQuery}
            disabled={loading || !selectedCollection || !query.trim()}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <LoadingSpinner className="w-5 h-5" />
                Querying...
              </>
            ) : (
              <>
                <MagnifyingGlassIcon className="w-5 h-5" />
                Execute Query
              </>
            )}
          </button>
        </div>
      </div>

      {result && (
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4 text-slate-900 dark:text-slate-100">
            Results
          </h3>
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Answer</h4>
              <div className="bg-slate-50 dark:bg-slate-900 rounded-md p-4 text-slate-900 dark:text-slate-100">
                {result.answer}
              </div>
            </div>
            <ContextViewer documents={result.documents || []} method={result.method} />
          </div>
        </div>
      )}
    </div>
  )
}
