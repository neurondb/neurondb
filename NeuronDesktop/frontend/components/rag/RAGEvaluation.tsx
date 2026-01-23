'use client'

import { useState } from 'react'
import { ragAPI, type RAGEvaluateRequest } from '@/lib/api'
import LoadingSpinner from '@/components/LoadingSpinner'
import { ChartBarIcon } from '@/components/Icons'

interface RAGEvaluationProps {
  profileId: string
  query?: string
  answer?: string
  contextChunks?: string[]
}

export default function RAGEvaluation({ profileId, query: initialQuery, answer: initialAnswer, contextChunks: initialContextChunks }: RAGEvaluationProps) {
  const [query, setQuery] = useState(initialQuery || '')
  const [answer, setAnswer] = useState(initialAnswer || '')
  const [contextChunks, setContextChunks] = useState<string[]>(initialContextChunks || [])
  const [evaluationType, setEvaluationType] = useState('basic')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string>('')

  const handleEvaluate = async () => {
    if (!profileId || !query.trim() || !answer.trim() || contextChunks.length === 0) {
      setError('Please provide query, answer, and at least one context chunk')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const request: RAGEvaluateRequest = {
        query,
        answer,
        context_chunks: contextChunks,
        evaluation_type: evaluationType,
      }

      const response = await ragAPI.evaluate(profileId, request)
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to evaluate RAG')
    } finally {
      setLoading(false)
    }
  }

  const addContextChunk = () => {
    setContextChunks([...contextChunks, ''])
  }

  const updateContextChunk = (index: number, value: string) => {
    const updated = [...contextChunks]
    updated[index] = value
    setContextChunks(updated)
  }

  const removeContextChunk = (index: number) => {
    setContextChunks(contextChunks.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-slate-900 dark:text-slate-100">
          RAG Evaluation
        </h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Query
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={2}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
              placeholder="Enter the query..."
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Answer
            </label>
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
              placeholder="Enter the generated answer..."
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
                Context Chunks
              </label>
              <button
                onClick={addContextChunk}
                className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
              >
                + Add Chunk
              </button>
            </div>
            <div className="space-y-2">
              {contextChunks.map((chunk, index) => (
                <div key={index} className="flex gap-2">
                  <textarea
                    value={chunk}
                    onChange={(e) => updateContextChunk(index, e.target.value)}
                    rows={2}
                    className="flex-1 px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
                    placeholder={`Context chunk ${index + 1}...`}
                  />
                  <button
                    onClick={() => removeContextChunk(index)}
                    className="px-3 py-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Evaluation Type
            </label>
            <select
              value={evaluationType}
              onChange={(e) => setEvaluationType(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
            >
              <option value="basic">Basic</option>
              <option value="advanced">Advanced</option>
            </select>
          </div>

          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
              <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
            </div>
          )}

          <button
            onClick={handleEvaluate}
            disabled={loading || !query.trim() || !answer.trim() || contextChunks.length === 0}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <LoadingSpinner className="w-5 h-5" />
                Evaluating...
              </>
            ) : (
              <>
                <ChartBarIcon className="w-5 h-5" />
                Evaluate
              </>
            )}
          </button>
        </div>
      </div>

      {result && (
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4 text-slate-900 dark:text-slate-100">
            Evaluation Results
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-slate-50 dark:bg-slate-900 rounded-md p-4">
              <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Faithfulness</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                {(result.faithfulness * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-slate-50 dark:bg-slate-900 rounded-md p-4">
              <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Relevancy</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                {(result.relevancy * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-slate-50 dark:bg-slate-900 rounded-md p-4">
              <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Context Precision</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                {(result.context_precision * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-slate-50 dark:bg-slate-900 rounded-md p-4">
              <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Context Recall</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                {(result.context_recall * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-slate-50 dark:bg-slate-900 rounded-md p-4">
              <div className="text-sm text-slate-500 dark:text-slate-400 mb-1">Semantic Similarity</div>
              <div className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                {(result.semantic_similarity * 100).toFixed(1)}%
              </div>
            </div>
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-md p-4 border-2 border-blue-200 dark:border-blue-800">
              <div className="text-sm text-blue-600 dark:text-blue-400 mb-1">Overall Score</div>
              <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">
                {(result.overall_score * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
