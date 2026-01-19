'use client'

import { useState } from 'react'
import { factoryAPI } from '@/lib/api'

interface MemoryFeedbackFormProps {
  profileId: string
  memoryId: string
  agentId: string
  sessionId?: string
  onSuccess?: () => void
}

export default function MemoryFeedbackForm({
  profileId,
  memoryId,
  agentId,
  sessionId,
  onSuccess,
}: MemoryFeedbackFormProps) {
  const [feedbackType, setFeedbackType] = useState('positive')
  const [feedbackText, setFeedbackText] = useState('')
  const [relevanceScore, setRelevanceScore] = useState(0.8)
  const [query, setQuery] = useState('')
  const [memoryTier, setMemoryTier] = useState('lpm')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    setError(null)
    setSuccess(false)

    try {
      await factoryAPI.post(`/profiles/${profileId}/agent/memory/${memoryId}/feedback`, {
        agent_id: agentId,
        session_id: sessionId,
        memory_tier: memoryTier,
        feedback_type: feedbackType,
        feedback_text: feedbackText,
        query: query,
        relevance_score: relevanceScore,
      })
      setSuccess(true)
      if (onSuccess) {
        onSuccess()
      }
      // Reset form after a delay
      setTimeout(() => {
        setFeedbackText('')
        setQuery('')
        setSuccess(false)
      }, 2000)
    } catch (err: any) {
      setError(err.message || 'Failed to submit feedback')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="card p-6">
      <h3 className="text-lg font-semibold mb-4">Submit Memory Feedback</h3>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Feedback Type</label>
          <select
            value={feedbackType}
            onChange={(e) => setFeedbackType(e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
          >
            <option value="positive">Positive</option>
            <option value="negative">Negative</option>
            <option value="neutral">Neutral</option>
            <option value="correction">Correction</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Memory Tier</label>
          <select
            value={memoryTier}
            onChange={(e) => setMemoryTier(e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
          >
            <option value="chunk">Chunk</option>
            <option value="stm">Short-term Memory (STM)</option>
            <option value="mtm">Medium-term Memory (MTM)</option>
            <option value="lpm">Long-term Memory (LPM)</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Query (optional)</label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
            placeholder="Query that led to this memory retrieval"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Relevance Score</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={relevanceScore}
            onChange={(e) => setRelevanceScore(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="text-sm text-slate-600 dark:text-slate-400 mt-1">
            {(relevanceScore * 100).toFixed(0)}%
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Feedback Text (optional)</label>
          <textarea
            value={feedbackText}
            onChange={(e) => setFeedbackText(e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
            rows={3}
            placeholder="Additional feedback or corrections..."
          />
        </div>

        {error && (
          <div className="text-red-600 dark:text-red-400 text-sm">{error}</div>
        )}

        {success && (
          <div className="text-green-600 dark:text-green-400 text-sm">
            Feedback submitted successfully!
          </div>
        )}

        <button
          type="submit"
          disabled={submitting}
          className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {submitting ? 'Submitting...' : 'Submit Feedback'}
        </button>
      </form>
    </div>
  )
}
