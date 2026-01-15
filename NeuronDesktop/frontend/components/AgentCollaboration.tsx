'use client'

import { useState } from 'react'
import DataTable, { type Column } from './DataTable'

interface Collaboration {
  id: string
  agent1: string
  agent2: string
  task: string
  status: 'active' | 'completed' | 'failed'
  createdAt: string
}

export default function AgentCollaboration({ collaborations }: { collaborations: Collaboration[] }) {
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
            px-2 py-1 rounded text-xs
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
  
  return <DataTable data={collaborations} columns={columns} searchable />
}

