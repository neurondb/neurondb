'use client'

import { useEffect, useState } from 'react'
import { BarChart, PieChart } from './Charts'
import { factoryAPI } from '@/lib/api'

interface RetrievalStats {
  agent_id: string
  days: number
  total_decisions: number
  avg_confidence: number
  source_usage: Record<string, number>
  avg_quality_score: number
  duration_ms: number
}

interface RetrievalStatsProps {
  profileId: string
  agentId: string
  days?: number
}

export default function RetrievalStats({ profileId, agentId, days = 30 }: RetrievalStatsProps) {
  const [stats, setStats] = useState<RetrievalStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadStats()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileId, agentId, days])

  const loadStats = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await factoryAPI.get(
        `/profiles/${profileId}/agent/agents/${agentId}/retrieval-stats?days=${days}`
      )
      setStats(response.data.data || response.data)
    } catch (err: any) {
      setError(err.message || 'Failed to load retrieval statistics')
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

  if (!stats) {
    return (
      <div className="card p-6">
        <div className="text-slate-600 dark:text-slate-400">No retrieval statistics available</div>
      </div>
    )
  }

  const sourceData = Object.entries(stats.source_usage || {}).map(([name, value]) => ({
    name,
    value,
  }))

  return (
    <div className="space-y-6">
      <div className="card p-6">
        <h3 className="text-lg font-semibold mb-4">Retrieval Statistics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div>
            <div className="text-sm text-slate-600 dark:text-slate-400">Total Decisions</div>
            <div className="text-2xl font-bold">{stats.total_decisions}</div>
          </div>
          <div>
            <div className="text-sm text-slate-600 dark:text-slate-400">Avg Confidence</div>
            <div className="text-2xl font-bold">{(stats.avg_confidence * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-sm text-slate-600 dark:text-slate-400">Avg Quality Score</div>
            <div className="text-2xl font-bold">{(stats.avg_quality_score * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-sm text-slate-600 dark:text-slate-400">Time Period</div>
            <div className="text-2xl font-bold">{stats.days} days</div>
          </div>
        </div>

        {sourceData.length > 0 && (
          <div className="space-y-4">
            <h4 className="font-semibold">Source Usage</h4>
            <div className="h-64">
              <BarChart
                data={sourceData}
                xAxisKey="name"
                bars={[{ key: 'value', name: 'Usage Count', color: 'rgb(139, 92, 246)' }]}
                height={256}
              />
            </div>
            <div className="h-64">
              <PieChart
                data={sourceData}
                height={256}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
