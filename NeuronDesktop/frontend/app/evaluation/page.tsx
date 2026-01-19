'use client'

import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import { BarChart, LineChart } from '@/components/Charts'
import DataTable, { type Column } from '@/components/DataTable'
import { factoryAPI } from '@/lib/api'
import { useRouter } from 'next/navigation'

interface EvaluationTask {
  id: string
  task_type: string
  input: string
  expected_output?: string
  metadata?: Record<string, any>
}

interface EvaluationRun {
  id: string
  agent_id: string
  agent_name?: string
  status: string
  score?: number
  metrics?: {
    accuracy: number
    response_time: number
    cost: number
  }
  created_at: string
}

export default function EvaluationPage() {
  const router = useRouter()
  const [profileId, setProfileId] = useState<string>('')
  const [tasks, setTasks] = useState<EvaluationTask[]>([])
  const [runs, setRuns] = useState<EvaluationRun[]>([])
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    loadProfile()
  }, [])

  useEffect(() => {
    if (profileId) {
      loadTasks()
      loadRuns()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileId])

  const loadProfile = async () => {
    try {
      const profilesResponse = await factoryAPI.get('/profiles')
      const profiles = profilesResponse.data.data || profilesResponse.data
      if (profiles && profiles.length > 0) {
        const activeProfile = profiles.find((p: any) => p.is_active) || profiles[0]
        setProfileId(activeProfile.id)
      }
    } catch (err) {
      console.error('Failed to load profile:', err)
    } finally {
      setLoading(false)
    }
  }

  const loadTasks = async () => {
    try {
      const response = await factoryAPI.get(`/profiles/${profileId}/agent/eval/tasks`)
      const data = response.data.data || response.data
      setTasks(Array.isArray(data) ? data : [])
    } catch (err) {
      console.error('Failed to load evaluation tasks:', err)
    }
  }

  const loadRuns = async () => {
    try {
      const response = await factoryAPI.get(`/profiles/${profileId}/agent/eval/runs`)
      const data = response.data.data || response.data
      const runsList = Array.isArray(data) ? data : []

      /* Load agent names */
      const agentsResponse = await factoryAPI.get(`/profiles/${profileId}/agent/agents`)
      const agents = agentsResponse.data.data || agentsResponse.data
      const agentMap = new Map(agents.map((a: any) => [a.id, a.name]))

      setRuns(
        runsList.map((run: any) => ({
          ...run,
          agent_name: agentMap.get(run.agent_id) || 'Unknown',
        }))
      )
    } catch (err) {
      console.error('Failed to load evaluation runs:', err)
    }
  }

  const handleCreateRun = async () => {
    try {
      const response = await factoryAPI.post(`/profiles/${profileId}/agent/eval/runs`, {
        task_ids: tasks.map((t) => t.id),
      })
      loadRuns()
    } catch (err) {
      console.error('Failed to create evaluation run:', err)
    }
  }

  const columns: Column<EvaluationRun>[] = [
    { key: 'agent_name', label: 'Agent', sortable: true },
    {
      key: 'score',
      label: 'Quality Score',
      sortable: true,
      render: (value) => (value ? `${(Number(value) * 100).toFixed(1)}%` : 'N/A'),
    },
    {
      key: 'metrics.accuracy',
      label: 'Accuracy',
      sortable: true,
      render: (value, row) =>
        row.metrics?.accuracy ? `${(row.metrics.accuracy * 100).toFixed(1)}%` : 'N/A',
    },
    {
      key: 'metrics.response_time',
      label: 'Response Time',
      sortable: true,
      render: (value, row) =>
        row.metrics?.response_time ? `${row.metrics.response_time}ms` : 'N/A',
    },
    {
      key: 'metrics.cost',
      label: 'Cost',
      sortable: true,
      render: (value, row) =>
        row.metrics?.cost ? `$${row.metrics.cost.toFixed(4)}` : 'N/A',
    },
    { key: 'status', label: 'Status', sortable: true },
    { key: 'created_at', label: 'Created', sortable: true },
  ]
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'Agents', href: '/agents' },
            { label: 'Evaluation Framework' },
          ]}
          className="mb-6"
        />
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              Evaluation Framework
            </h1>
            <p className="text-slate-600 dark:text-slate-400 mt-1">
              Agent performance evaluation and quality scoring
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => router.push('/evaluation/tasks')}
              className="px-4 py-2 bg-slate-600 text-white rounded-md hover:bg-slate-700"
            >
              Manage Tasks
            </button>
            <button
              onClick={handleCreateRun}
              disabled={tasks.length === 0}
              className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Create Run
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4">Quality Scores</h3>
            {runs.length > 0 ? (
              <BarChart
                data={runs.filter((r) => r.score !== undefined)}
                bars={[{ key: 'score', name: 'Quality Score', color: 'rgb(139, 92, 246)' }]}
                xAxisKey="agent_name"
                height={200}
              />
            ) : (
              <div className="text-slate-600 dark:text-slate-400">No evaluation data</div>
            )}
          </div>
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-slate-600 dark:text-slate-400">Total Tasks</span>
                <span className="text-sm font-semibold">{tasks.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-slate-600 dark:text-slate-400">Total Runs</span>
                <span className="text-sm font-semibold">{runs.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-slate-600 dark:text-slate-400">Completed Runs</span>
                <span className="text-sm font-semibold">
                  {runs.filter((r) => r.status === 'completed').length}
                </span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="card p-6">
          <h2 className="text-xl font-semibold mb-4">Evaluation Runs</h2>
          {runs.length > 0 ? (
            <DataTable data={runs} columns={columns} searchable />
          ) : (
            <div className="text-center text-slate-600 dark:text-slate-400 py-8">
              No evaluation runs yet. Create a run to evaluate agent performance.
            </div>
          )}
        </div>
      </div>
    </MainContent>
  )
}




