'use client'

import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'
import EmptyState from '@/components/EmptyState'
import WorkflowDAG from '@/components/WorkflowDAG'
import ApprovalGate from '@/components/ApprovalGate'
import { factoryAPI } from '@/lib/api'
import { useRouter } from 'next/navigation'

interface Workflow {
  id: string
  name: string
  status: 'active' | 'paused' | 'error'
  lastRun: string
  runs: number
  description?: string
}

interface WorkflowExecution {
  id: string
  workflow_id: string
  status: string
  current_step?: string
  requires_approval?: boolean
}

export default function WorkflowsPage() {
  const router = useRouter()
  const [profileId, setProfileId] = useState<string>('')
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null)
  const [executions, setExecutions] = useState<WorkflowExecution[]>([])
  const [pendingApprovals, setPendingApprovals] = useState<WorkflowExecution[]>([])
  const [loading, setLoading] = useState(true)
  const [view, setView] = useState<'list' | 'dag' | 'executions'>('list')
  
  useEffect(() => {
    loadProfile()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (profileId) {
      loadWorkflows()
      loadPendingApprovals()
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

  const loadWorkflows = async () => {
    try {
      const response = await factoryAPI.get(`/profiles/${profileId}/agent/workflows`)
      const data = response.data.data || response.data
      setWorkflows(Array.isArray(data) ? data : [])
    } catch (err) {
      console.error('Failed to load workflows:', err)
    }
  }

  const loadPendingApprovals = async () => {
    try {
      /* Load executions that require approval */
      const response = await factoryAPI.get(`/profiles/${profileId}/agent/workflow-executions`)
      const data = response.data.data || response.data
      const execs = Array.isArray(data) ? data : []
      setPendingApprovals(execs.filter((e: any) => e.requires_approval && e.status === 'waiting_approval'))
    } catch (err) {
      console.error('Failed to load pending approvals:', err)
    }
  }

  const loadExecutions = async (workflowId: string) => {
    try {
      const response = await factoryAPI.get(
        `/profiles/${profileId}/agent/workflows/${workflowId}/executions`
      )
      const data = response.data.data || response.data
      setExecutions(Array.isArray(data) ? data : [])
    } catch (err) {
      console.error('Failed to load executions:', err)
    }
  }

  const handleApprove = async (executionId: string, stepId: string) => {
    try {
      await factoryAPI.post(
        `/profiles/${profileId}/agent/workflow-executions/${executionId}/approve`,
        { step_id: stepId }
      )
      loadPendingApprovals()
    } catch (err) {
      console.error('Failed to approve:', err)
    }
  }

  const handleReject = async (executionId: string, stepId: string, reason?: string) => {
    try {
      await factoryAPI.post(
        `/profiles/${profileId}/agent/workflow-executions/${executionId}/reject`,
        { step_id: stepId, reason }
      )
      loadPendingApprovals()
    } catch (err) {
      console.error('Failed to reject:', err)
    }
  }

  const columns: Column<Workflow>[] = [
    { key: 'name', label: 'Name', sortable: true },
    {
      key: 'status',
      label: 'Status',
      render: (value) => (
        <span
          className={`
            px-2 py-1 rounded text-xs font-medium
            ${
              value === 'active'
                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                : value === 'paused'
                ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400'
                : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
            }
          `}
        >
          {value}
        </span>
      ),
    },
    { key: 'lastRun', label: 'Last Run', sortable: true },
    { key: 'runs', label: 'Runs', sortable: true },
  ]
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'Agents', href: '/agents' },
            { label: 'Workflows' },
          ]}
          className="mb-6"
        />
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              Workflows
            </h1>
            <p className="text-slate-600 dark:text-slate-400 mt-1">
              DAG-based workflow execution with HITL approval gates
            </p>
          </div>
          <button
            className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
            onClick={() => router.push('/workflows/create')}
          >
            Create Workflow
          </button>
        </div>

        {pendingApprovals.length > 0 && (
          <div className="mb-6 space-y-4">
            <h2 className="text-xl font-semibold">Pending Approvals</h2>
            {pendingApprovals.map((exec) => (
              <ApprovalGate
                key={exec.id}
                executionId={exec.id}
                workflowId={exec.workflow_id}
                stepId={exec.current_step || ''}
                message={`Workflow execution requires approval at step: ${exec.current_step}`}
                onApprove={handleApprove}
                onReject={handleReject}
              />
            ))}
          </div>
        )}

        <div className="mb-4 flex gap-2">
          <button
            onClick={() => setView('list')}
            className={`px-4 py-2 rounded-md ${
              view === 'list'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            List
          </button>
          {selectedWorkflow && (
            <>
              <button
                onClick={() => {
                  setView('dag')
                  loadExecutions(selectedWorkflow.id)
                }}
                className={`px-4 py-2 rounded-md ${
                  view === 'dag'
                    ? 'bg-purple-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                }`}
              >
                DAG View
              </button>
              <button
                onClick={() => {
                  setView('executions')
                  loadExecutions(selectedWorkflow.id)
                }}
                className={`px-4 py-2 rounded-md ${
                  view === 'executions'
                    ? 'bg-purple-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                }`}
              >
                Executions
              </button>
            </>
          )}
        </div>

        {view === 'list' && (
          <>
            {workflows.length > 0 ? (
              <DataTable
                data={workflows}
                columns={columns}
                onRowClick={(workflow) => {
                  setSelectedWorkflow(workflow)
                  setView('dag')
                  loadExecutions(workflow.id)
                }}
              />
            ) : (
              <EmptyState
                icon="🔄"
                title="No workflows"
                description="Create a workflow to automate agent tasks"
                action={{
                  label: 'Create Workflow',
                  onClick: () => router.push('/workflows/create'),
                }}
              />
            )}
          </>
        )}

        {view === 'dag' && selectedWorkflow && (
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4">{selectedWorkflow.name} - DAG View</h2>
            <WorkflowDAG
              nodes={[
                { id: '1', type: 'agent', label: 'Start', status: 'completed' },
                { id: '2', type: 'tool', label: 'Process', status: 'running' },
                { id: '3', type: 'approval', label: 'Approve', status: 'pending' },
                { id: '4', type: 'agent', label: 'Complete', status: 'pending' },
              ]}
              edges={[
                { from: '1', to: '2' },
                { from: '2', to: '3' },
                { from: '3', to: '4' },
              ]}
            />
          </div>
        )}

        {view === 'executions' && selectedWorkflow && (
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4">Executions</h2>
            <DataTable
              data={executions}
              columns={[
                { key: 'id', label: 'Execution ID', sortable: true },
                { key: 'status', label: 'Status', sortable: true },
                { key: 'current_step', label: 'Current Step', sortable: true },
              ]}
            />
          </div>
        )}
      </div>
    </MainContent>
  )
}




