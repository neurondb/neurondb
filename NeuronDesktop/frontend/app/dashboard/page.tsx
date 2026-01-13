'use client'

import { useState, useEffect } from 'react'
import { profilesAPI, dashboardAPI, type Profile, type DashboardStats } from '@/lib/api'
import { LineChart, BarChart, PieChart } from '@/components/Charts'
import { DraggableGrid } from '@/components/GridLayout'
import { 
  DatabaseIcon, 
  CpuChipIcon, 
  ChatBubbleLeftRightIcon,
  ActivityIcon,
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon
} from '@/components/Icons'
import { formatDistanceToNow } from 'date-fns'

export default function DashboardPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [dashboardData, setDashboardData] = useState<DashboardStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadProfiles()
  }, [])

  useEffect(() => {
    if (selectedProfile) {
      loadDashboard()
      const interval = setInterval(loadDashboard, 30000) // Refresh every 30s
      return () => clearInterval(interval)
    }
  }, [selectedProfile])

  const loadProfiles = async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0) {
        const defaultProfile = response.data.find((p) => p.is_default) || response.data[0]
        setSelectedProfile(defaultProfile.id)
      }
    } catch (err: any) {
      setError('Failed to load profiles: ' + (err.response?.data?.error || err.message))
    }
  }

  const loadDashboard = async () => {
    if (!selectedProfile) return
    setLoading(true)
    setError(null)
    try {
      const response = await dashboardAPI.getDashboard(selectedProfile)
      setDashboardData(response.data)
    } catch (err: any) {
      setError('Failed to load dashboard: ' + (err.response?.data?.error || err.message))
    } finally {
      setLoading(false)
    }
  }

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 dark:text-green-400'
      case 'unhealthy':
        return 'text-red-600 dark:text-red-400'
      default:
        return 'text-yellow-600 dark:text-yellow-400'
    }
  }

  const getHealthIcon = (status: string) => {
    return status === 'healthy' ? (
      <CheckCircleIcon className="w-5 h-5 text-green-600" />
    ) : (
      <XCircleIcon className="w-5 h-5 text-red-600" />
    )
  }

  const dashboardItems = dashboardData ? [
    {
      id: 'system-metrics',
      content: (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 flex items-center">
              <ActivityIcon className="w-5 h-5 mr-2" />
              System Metrics
            </h3>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400">Total Requests</p>
              <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                {dashboardData.system_metrics?.total_requests?.toLocaleString() || 0}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-600 dark:text-slate-400">Success Rate</p>
              <p className="text-2xl font-bold text-green-600">
                {dashboardData.system_metrics?.total_requests 
                  ? ((dashboardData.system_metrics.successful_requests / dashboardData.system_metrics.total_requests) * 100).toFixed(1)
                  : 0}%
              </p>
            </div>
          </div>
        </div>
      ),
      defaultLayout: { i: 'system-metrics', x: 0, y: 0, w: 6, h: 3 },
    },
    {
      id: 'neurondb-stats',
      content: (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 flex items-center">
              <DatabaseIcon className="w-5 h-5 mr-2" />
              NeuronDB
            </h3>
            {dashboardData.health_status?.components?.neurondb && 
              getHealthIcon(dashboardData.health_status.components.neurondb)}
          </div>
          {dashboardData.neurondb_stats ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Collections</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.neurondb_stats.collections_count}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Total Vectors</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.neurondb_stats.total_vectors.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Indexes</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.neurondb_stats.indexes_count}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Avg Query Time</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.neurondb_stats.avg_query_time?.toFixed(2) || 0}ms
                </p>
              </div>
            </div>
          ) : (
            <p className="text-slate-600 dark:text-slate-400">No NeuronDB connection configured</p>
          )}
        </div>
      ),
      defaultLayout: { i: 'neurondb-stats', x: 6, y: 0, w: 6, h: 3 },
    },
    {
      id: 'agent-stats',
      content: (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 flex items-center">
              <CpuChipIcon className="w-5 h-5 mr-2" />
              NeuronAgent
            </h3>
            {dashboardData.health_status?.components?.neuronagent && 
              getHealthIcon(dashboardData.health_status.components.neuronagent)}
          </div>
          {dashboardData.neuronagent_stats ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Agents</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.neuronagent_stats.agents_count}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Sessions</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.neuronagent_stats.sessions_count}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Messages</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.neuronagent_stats.messages_count.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Avg Response</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.neuronagent_stats.avg_response_time?.toFixed(2) || 0}ms
                </p>
              </div>
            </div>
          ) : (
            <p className="text-slate-600 dark:text-slate-400">No NeuronAgent connection configured</p>
          )}
        </div>
      ),
      defaultLayout: { i: 'agent-stats', x: 0, y: 3, w: 6, h: 3 },
    },
    {
      id: 'mcp-stats',
      content: (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 flex items-center">
              <ChatBubbleLeftRightIcon className="w-5 h-5 mr-2" />
              NeuronMCP
            </h3>
            {dashboardData.health_status?.components?.mcp && 
              getHealthIcon(dashboardData.health_status.components.mcp)}
          </div>
          {dashboardData.mcp_stats ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Tools</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.mcp_stats.tools_count}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Tools Called</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.mcp_stats.tools_called.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-sm text-slate-600 dark:text-slate-400">Active Connections</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                  {dashboardData.mcp_stats.active_connections}
                </p>
              </div>
            </div>
          ) : (
            <p className="text-slate-600 dark:text-slate-400">No MCP connection configured</p>
          )}
        </div>
      ),
      defaultLayout: { i: 'mcp-stats', x: 6, y: 3, w: 6, h: 3 },
    },
    {
      id: 'recent-activity',
      content: (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 flex items-center">
              <ClockIcon className="w-5 h-5 mr-2" />
              Recent Activity
            </h3>
          </div>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {dashboardData.recent_activity && dashboardData.recent_activity.length > 0 ? (
              dashboardData.recent_activity.map((activity) => (
                <div key={activity.id} className="border-b border-slate-200 dark:border-slate-700 pb-3 last:border-0">
                  <p className="text-sm text-slate-900 dark:text-slate-100">{activity.description}</p>
                  <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                    {formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-slate-600 dark:text-slate-400">No recent activity</p>
            )}
          </div>
        </div>
      ),
      defaultLayout: { i: 'recent-activity', x: 0, y: 6, w: 12, h: 4 },
    },
  ] : []

  return (
    <div className="min-h-full bg-transparent p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">Dashboard</h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">Unified view of your NeuronDB ecosystem</p>
        </div>

        {/* Profile Selector */}
        {profiles.length > 0 && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Profile
            </label>
            <select
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value)}
              className="input"
            >
              {profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} {profile.is_default && '(Default)'}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {/* Loading State */}
        {loading && !dashboardData ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto"></div>
            <p className="text-slate-600 dark:text-slate-400 mt-4">Loading dashboard...</p>
          </div>
        ) : dashboardData ? (
          <DraggableGrid
            items={dashboardItems}
            cols={12}
            rowHeight={100}
            className="dashboard-grid"
          />
        ) : null}
      </div>
    </div>
  )
}





