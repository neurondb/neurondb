'use client'

import { useState, useEffect, useCallback } from 'react'
import { 
  DocumentTextIcon,
  FunnelIcon,
  CalendarIcon,
  CheckCircleIcon,
  XCircleIcon,
  ArrowDownTrayIcon,
  TrashIcon
} from '@/components/Icons'
import JSONViewer from '@/components/JSONViewer'
import { requestLogsAPI, profilesAPI, type Profile, type RequestLog } from '@/lib/api'
import { showSuccessToast, showErrorToast } from '@/lib/errors'
import ProfileSelector from '@/components/ProfileSelector'

export default function LogsPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [logs, setLogs] = useState<RequestLog[]>([])
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState({
    status: 'all',
    endpoint: '',
    dateRange: 'today',
  })

  const loadProfiles = useCallback(async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0 && !selectedProfile) {
        const activeProfileId = localStorage.getItem('active_profile_id')
        if (activeProfileId) {
          const active = response.data.find((p: Profile) => p.id === activeProfileId)
          if (active) {
            setSelectedProfile(activeProfileId)
            return
          }
        }
        const defaultProfile = response.data.find((p: Profile) => p.is_default)
        setSelectedProfile(defaultProfile ? defaultProfile.id : response.data[0].id)
      }
    } catch (error) {
      showErrorToast('Failed to load profiles')
    }
  }, [selectedProfile])

  const loadLogs = useCallback(async () => {
    if (!selectedProfile) return
    
    setLoading(true)
    try {
      const params: any = { limit: 100 }
      
      if (filter.status !== 'all') {
        if (filter.status === '200') {
          params.status_code = 200
        } else if (filter.status === '400') {
          params.status_code = 400
        } else if (filter.status === '500') {
          params.status_code = 500
        }
      }
      
      if (filter.endpoint) {
        params.endpoint = filter.endpoint
      }
      
      if (filter.dateRange !== 'all') {
        const now = new Date()
        let startDate: Date
        if (filter.dateRange === 'today') {
          startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate())
        } else if (filter.dateRange === 'week') {
          startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)
        } else {
          startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000)
        }
        params.start_date = startDate.toISOString()
        params.end_date = now.toISOString()
      }
      
      const response = await requestLogsAPI.listLogs(selectedProfile, params)
      setLogs(response.data)
    } catch (error: any) {
      showErrorToast('Failed to load logs: ' + (error.response?.data?.error || error.message))
      setLogs([])
    } finally {
      setLoading(false)
    }
  }, [selectedProfile, filter])

  useEffect(() => {
    loadProfiles()
  }, [loadProfiles])

  useEffect(() => {
    if (selectedProfile) {
      loadLogs()
    } else {
      setLogs([])
    }
  }, [selectedProfile, filter, loadLogs])

  const handleDeleteLog = async (logId: string) => {
    if (!selectedProfile) return
    if (!confirm('Are you sure you want to delete this log?')) return
    
    try {
      await requestLogsAPI.deleteLog(selectedProfile, logId)
      showSuccessToast('Log deleted successfully')
      loadLogs()
    } catch (error: any) {
      showErrorToast('Failed to delete log: ' + (error.response?.data?.error || error.message))
    }
  }

  const handleExport = async (format: 'json' | 'csv') => {
    if (!selectedProfile) return
    
    try {
      const response = await requestLogsAPI.exportLogs(selectedProfile, format)
      const blob = new Blob([response.data], { 
        type: format === 'csv' ? 'text/csv' : 'application/json' 
      })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `logs-${selectedProfile}-${new Date().toISOString().split('T')[0]}.${format}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
      showSuccessToast(`Logs exported as ${format.toUpperCase()}`)
    } catch (error: any) {
      showErrorToast('Failed to export logs: ' + (error.response?.data?.error || error.message))
    }
  }

  const filteredLogs = logs.filter(log => {
    if (filter.status === 'all') return true
    if (filter.status === '200') return log.status_code >= 200 && log.status_code < 300
    if (filter.status === '400') return log.status_code >= 400 && log.status_code < 500
    if (filter.status === '500') return log.status_code >= 500
    return true
  })

  return (
    <div className="h-full flex flex-col bg-slate-50 dark:bg-slate-800">
      {/* Header */}
      <div className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-slate-100">Request Logs</h1>
            <p className="text-sm text-gray-600 dark:text-slate-400 mt-1">View and inspect all API requests and responses</p>
          </div>
          <div className="flex items-center gap-4">
            <ProfileSelector
              profiles={profiles}
              selectedProfile={selectedProfile}
              onSelect={setSelectedProfile}
            />
            {selectedProfile && (
              <div className="flex gap-2">
                <button
                  onClick={() => handleExport('json')}
                  className="btn btn-secondary flex items-center gap-2"
                  title="Export as JSON"
                >
                  <ArrowDownTrayIcon className="w-4 h-4" />
                  JSON
                </button>
                <button
                  onClick={() => handleExport('csv')}
                  className="btn btn-secondary flex items-center gap-2"
                  title="Export as CSV"
                >
                  <ArrowDownTrayIcon className="w-4 h-4" />
                  CSV
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 px-6 py-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <FunnelIcon className="w-5 h-5 text-gray-600 dark:text-slate-500" />
            <span className="text-sm font-medium text-gray-900 dark:text-slate-200">Filters:</span>
          </div>
          
          <select
            value={filter.status}
            onChange={(e) => setFilter({ ...filter, status: e.target.value })}
            className="input w-40"
          >
            <option value="all">All Status</option>
            <option value="200">Success (200)</option>
            <option value="400">Client Error (4xx)</option>
            <option value="500">Server Error (5xx)</option>
          </select>
          
          <input
            type="text"
            value={filter.endpoint}
            onChange={(e) => setFilter({ ...filter, endpoint: e.target.value })}
            placeholder="Filter by endpoint..."
            className="input w-64"
          />
          
          <select
            value={filter.dateRange}
            onChange={(e) => setFilter({ ...filter, dateRange: e.target.value })}
            className="input w-40"
          >
            <option value="today">Today</option>
            <option value="week">This Week</option>
            <option value="month">This Month</option>
            <option value="all">All Time</option>
          </select>
        </div>
      </div>

      {/* Logs List */}
      <div className="flex-1 overflow-y-auto p-6">
        {loading ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-700 dark:text-slate-400">Loading logs...</p>
          </div>
        ) : !selectedProfile ? (
          <div className="text-center py-12">
            <DocumentTextIcon className="w-12 h-12 text-gray-500 dark:text-slate-500 mx-auto mb-4" />
            <p className="text-gray-700 dark:text-slate-400">Please select a profile to view logs.</p>
          </div>
        ) : filteredLogs.length === 0 ? (
          <div className="text-center py-12">
            <DocumentTextIcon className="w-12 h-12 text-gray-500 dark:text-slate-500 mx-auto mb-4" />
            <p className="text-gray-700 dark:text-slate-400">No logs available yet.</p>
            <p className="text-sm text-gray-600 dark:text-slate-500 mt-2">Logs will appear here as you make requests.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {filteredLogs.map((log) => (
              <div key={log.id} className="card">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3 flex-1">
                    {log.status_code >= 200 && log.status_code < 300 ? (
                      <CheckCircleIcon className="w-5 h-5 text-green-500" />
                    ) : (
                      <XCircleIcon className="w-5 h-5 text-red-500" />
                    )}
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          log.method === 'GET' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300' :
                          log.method === 'POST' ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300' :
                          log.method === 'PUT' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300' :
                          'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300'
                        }`}>
                          {log.method}
                        </span>
                        <span className="font-mono text-sm text-gray-900 dark:text-slate-100">{log.endpoint}</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          log.status_code >= 200 && log.status_code < 300 ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300' :
                          log.status_code >= 400 && log.status_code < 500 ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300' :
                          'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300'
                        }`}>
                          {log.status_code}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 mt-2 text-xs text-gray-600 dark:text-slate-400">
                        <div className="flex items-center gap-1">
                          <CalendarIcon className="w-3 h-3" />
                          {new Date(log.created_at).toLocaleString()}
                        </div>
                        <div>{log.duration_ms}ms</div>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteLog(log.id)}
                    className="text-red-500 hover:text-red-700 p-2"
                    title="Delete log"
                  >
                    <TrashIcon className="w-4 h-4" />
                  </button>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-slate-200 mb-2">Request</h4>
                    <JSONViewer data={log.request_body || {}} defaultExpanded={false} />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 dark:text-slate-200 mb-2">Response</h4>
                    <JSONViewer data={log.response_body || {}} defaultExpanded={false} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
