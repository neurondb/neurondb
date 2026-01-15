'use client'

import { useState, useEffect, useCallback } from 'react'
import { profilesAPI } from '@/lib/api'
import { showErrorToast } from '@/lib/errors'
import { 
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  DatabaseIcon,
  TableCellsIcon,
  CpuChipIcon
} from '@/components/Icons'

interface DBHealth {
  status: string
  version?: string
  connections?: number
  uptime?: string
  metrics?: Record<string, any>
}

interface IndexHealth {
  table_name: string
  index_name: string
  index_type: string
  status: string
  size?: string
  build_progress?: number
}

interface WorkerStatus {
  worker_name: string
  status: string
  last_run?: string
  next_run?: string
  jobs_processed?: number
  errors?: number
}

interface UsageStats {
  total_requests: number
  errors: number
  avg_duration_ms: number
  total_tokens: number
}

export default function ObservabilityPage() {
  const [profiles, setProfiles] = useState<any[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [dbHealth, setDbHealth] = useState<DBHealth | null>(null)
  const [indexHealth, setIndexHealth] = useState<IndexHealth[]>([])
  const [workerStatus, setWorkerStatus] = useState<WorkerStatus[]>([])
  const [usageStats, setUsageStats] = useState<UsageStats | null>(null)
  const [loading, setLoading] = useState(false)

  const loadProfiles = useCallback(async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0) {
        const defaultProfile = response.data.find((p: any) => p.is_default) || response.data[0]
        setSelectedProfile(defaultProfile.id)
      }
    } catch (error: any) {
      showErrorToast('Failed to load profiles: ' + (error.response?.data?.error || error.message))
    }
  }, [])

  const loadObservabilityData = useCallback(async () => {
    if (!selectedProfile) return
    setLoading(true)

    try {
      // Use axios directly for observability endpoints
      const axios = (await import('axios')).default
      const baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081/api/v1'
      
      // Load all observability data in parallel
      const [dbRes, indexRes, workerRes, usageRes] = await Promise.allSettled([
        axios.get(`${baseURL}/profiles/${selectedProfile}/observability/db-health`, { withCredentials: true }),
        axios.get(`${baseURL}/profiles/${selectedProfile}/observability/indexes`, { withCredentials: true }),
        axios.get(`${baseURL}/profiles/${selectedProfile}/observability/workers`, { withCredentials: true }),
        axios.get(`${baseURL}/profiles/${selectedProfile}/observability/usage`, { withCredentials: true }),
      ])

      if (dbRes.status === 'fulfilled') {
        setDbHealth(dbRes.value.data)
      }
      if (indexRes.status === 'fulfilled') {
        setIndexHealth(indexRes.value.data)
      }
      if (workerRes.status === 'fulfilled') {
        setWorkerStatus(workerRes.value.data)
      }
      if (usageRes.status === 'fulfilled') {
        setUsageStats(usageRes.value.data)
      }
    } catch (error: any) {
      showErrorToast('Failed to load observability data: ' + (error.response?.data?.error || error.message))
    } finally {
      setLoading(false)
    }
  }, [selectedProfile])

  useEffect(() => {
    loadProfiles()
  }, [loadProfiles])

  useEffect(() => {
    if (selectedProfile) {
      loadObservabilityData()
      const interval = setInterval(loadObservabilityData, 30000) // Refresh every 30s
      return () => clearInterval(interval)
    }
  }, [selectedProfile, loadObservabilityData])

  return (
    <div className="min-h-full bg-transparent p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">Observability</h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">Monitor system health and performance</p>
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

        {loading && !dbHealth ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto"></div>
            <p className="text-slate-600 dark:text-slate-400 mt-4">Loading observability data...</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Database Health */}
            <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100 flex items-center">
                  <DatabaseIcon className="w-5 h-5 mr-2" />
                  Database Health
                </h2>
                {dbHealth && (
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    dbHealth.status === 'healthy' 
                      ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                      : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                  }`}>
                    {dbHealth.status}
                  </span>
                )}
              </div>
              {dbHealth && (
                <div className="space-y-2">
                  {dbHealth.version && (
                    <div className="flex justify-between">
                      <span className="text-slate-600 dark:text-slate-400">Version:</span>
                      <span className="text-slate-900 dark:text-slate-100">{dbHealth.version}</span>
                    </div>
                  )}
                  {dbHealth.connections !== undefined && (
                    <div className="flex justify-between">
                      <span className="text-slate-600 dark:text-slate-400">Connections:</span>
                      <span className="text-slate-900 dark:text-slate-100">{dbHealth.connections}</span>
                    </div>
                  )}
                  {dbHealth.metrics && (
                    <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700">
                      <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Metrics</h3>
                      {Object.entries(dbHealth.metrics).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span className="text-slate-600 dark:text-slate-400">{key}:</span>
                          <span className="text-slate-900 dark:text-slate-100">{String(value)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Index Health */}
            <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6">
              <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100 flex items-center mb-4">
                <TableCellsIcon className="w-5 h-5 mr-2" />
                Index Health ({indexHealth.length})
              </h2>
              {indexHealth.length === 0 ? (
                <p className="text-slate-600 dark:text-slate-400">No indexes found</p>
              ) : (
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {indexHealth.map((idx, i) => (
                    <div key={i} className="border-b border-slate-200 dark:border-slate-700 pb-3 last:border-0">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-slate-900 dark:text-slate-100">
                          {idx.table_name}.{idx.index_name}
                        </span>
                        {idx.status === 'healthy' ? (
                          <CheckCircleIcon className="w-5 h-5 text-green-600" />
                        ) : (
                          <XCircleIcon className="w-5 h-5 text-yellow-600" />
                        )}
                      </div>
                      <div className="text-sm text-slate-600 dark:text-slate-400">
                        {idx.index_type} • {idx.size || 'Unknown size'}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Worker Status */}
            <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6">
              <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100 flex items-center mb-4">
                <CpuChipIcon className="w-5 h-5 mr-2" />
                Background Workers ({workerStatus.length})
              </h2>
              {workerStatus.length === 0 ? (
                <p className="text-slate-600 dark:text-slate-400">No workers configured</p>
              ) : (
                <div className="space-y-3">
                  {workerStatus.map((worker, i) => (
                    <div key={i} className="border-b border-slate-200 dark:border-slate-700 pb-3 last:border-0">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium text-slate-900 dark:text-slate-100">
                          {worker.worker_name}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs ${
                          worker.status === 'running'
                            ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                        }`}>
                          {worker.status}
                        </span>
                      </div>
                      {worker.jobs_processed !== undefined && (
                        <div className="text-sm text-slate-600 dark:text-slate-400">
                          Jobs: {worker.jobs_processed} • Errors: {worker.errors || 0}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Usage Statistics */}
            <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6">
              <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">
                Usage Statistics (30 days)
              </h2>
              {usageStats ? (
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-600 dark:text-slate-400">Total Requests:</span>
                    <span className="text-slate-900 dark:text-slate-100 font-medium">
                      {usageStats.total_requests.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600 dark:text-slate-400">Errors:</span>
                    <span className="text-slate-900 dark:text-slate-100 font-medium text-red-600">
                      {usageStats.errors.toLocaleString()}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600 dark:text-slate-400">Avg Duration:</span>
                    <span className="text-slate-900 dark:text-slate-100 font-medium">
                      {usageStats.avg_duration_ms?.toFixed(2)}ms
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-600 dark:text-slate-400">Total Tokens:</span>
                    <span className="text-slate-900 dark:text-slate-100 font-medium">
                      {usageStats.total_tokens?.toLocaleString() || '0'}
                    </span>
                  </div>
                </div>
              ) : (
                <p className="text-slate-600 dark:text-slate-400">No usage data available</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

