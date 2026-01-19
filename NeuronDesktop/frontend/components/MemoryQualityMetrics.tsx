'use client'

import { useEffect, useState } from 'react'
import { LineChart, BarChart } from './Charts'
import { factoryAPI } from '@/lib/api'

interface MemoryQualityMetricsProps {
  profileId: string
  agentId: string
  memoryId?: string
  tier?: string
}

export default function MemoryQualityMetrics({
  profileId,
  agentId,
  memoryId,
  tier,
}: MemoryQualityMetricsProps) {
  const [metrics, setMetrics] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadMetrics()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileId, agentId, memoryId, tier])

  const loadMetrics = async () => {
    try {
      setLoading(true)
      setError(null)
      let url = `/profiles/${profileId}/agent/agents/${agentId}/memory/quality`
      if (memoryId && tier) {
        url += `?memory_id=${memoryId}&tier=${tier}`
      }
      const response = await factoryAPI.get(url)
      setMetrics(response.data.data || response.data)
    } catch (err: any) {
      setError(err.message || 'Failed to load memory quality metrics')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="card p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-1/4"></div>
          <div className="h-32 bg-slate-200 dark:bg-slate-700 rounded"></div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card p-6">
        <div className="text-red-600 dark:text-red-400">{error}</div>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="card p-6">
        <div className="text-slate-600 dark:text-slate-400">No quality metrics available</div>
      </div>
    )
  }

  return (
    <div className="card p-6">
      <h3 className="text-lg font-semibold mb-4">Memory Quality Metrics</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metrics.quality_score && (
          <div>
            <div className="text-sm text-slate-600 dark:text-slate-400">Quality Score</div>
            <div className="text-2xl font-bold">
              {typeof metrics.quality_score === 'number'
                ? (metrics.quality_score * 100).toFixed(1)
                : 'N/A'}
              %
            </div>
          </div>
        )}
        {metrics.retrieval_count !== undefined && (
          <div>
            <div className="text-sm text-slate-600 dark:text-slate-400">Retrieval Count</div>
            <div className="text-2xl font-bold">{metrics.retrieval_count}</div>
          </div>
        )}
        {metrics.positive_feedback !== undefined && (
          <div>
            <div className="text-sm text-slate-600 dark:text-slate-400">Positive Feedback</div>
            <div className="text-2xl font-bold">{metrics.positive_feedback}</div>
          </div>
        )}
        {metrics.negative_feedback !== undefined && (
          <div>
            <div className="text-sm text-slate-600 dark:text-slate-400">Negative Feedback</div>
            <div className="text-2xl font-bold">{metrics.negative_feedback}</div>
          </div>
        )}
      </div>
    </div>
  )
}
