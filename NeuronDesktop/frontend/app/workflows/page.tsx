'use client'

import { useState } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'
import EmptyState from '@/components/EmptyState'

interface Workflow {
  id: string
  name: string
  status: 'active' | 'paused' | 'error'
  lastRun: string
  runs: number
}

export default function WorkflowsPage() {
  const [workflows] = useState<Workflow[]>([])
  
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
          <button className="btn-primary">Create Workflow</button>
        </div>
        
        {workflows.length > 0 ? (
          <DataTable data={workflows} columns={columns} />
        ) : (
          <EmptyState
            icon="🔄"
            title="No workflows"
            description="Create a workflow to automate agent tasks"
            action={{
              label: 'Create Workflow',
              onClick: () => {},
            }}
          />
        )}
      </div>
    </MainContent>
  )
}

