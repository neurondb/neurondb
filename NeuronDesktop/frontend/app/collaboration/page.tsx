'use client'

import { useState } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'
import EmptyState from '@/components/EmptyState'

interface Collaboration {
  id: string
  agent1: string
  agent2: string
  task: string
  status: 'active' | 'completed' | 'failed'
  createdAt: string
}

export default function CollaborationPage() {
  const [collaborations] = useState<Collaboration[]>([])
  
  const columns: Column<Collaboration>[] = [
    { key: 'agent1', label: 'Agent 1', sortable: true },
    { key: 'agent2', label: 'Agent 2', sortable: true },
    { key: 'task', label: 'Task', sortable: true },
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
                : value === 'completed'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
            }
          `}
        >
          {value}
        </span>
      ),
    },
    { key: 'createdAt', label: 'Created', sortable: true },
  ]
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'Agents', href: '/agents' },
            { label: 'Multi-Agent Collaboration' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Multi-Agent Collaboration
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Agent-to-agent communication and task delegation
          </p>
        </div>
        
        {collaborations.length > 0 ? (
          <DataTable data={collaborations} columns={columns} />
        ) : (
          <EmptyState
            icon="🤝"
            title="No collaborations"
            description="Agents will appear here when they collaborate on tasks"
          />
        )}
      </div>
    </MainContent>
  )
}

