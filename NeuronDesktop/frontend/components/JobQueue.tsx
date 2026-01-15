'use client'

import DataTable, { type Column } from './DataTable'

interface Job {
  id: string
  type: 'memory_promotion' | 'background_task'
  status: 'pending' | 'running' | 'completed' | 'failed'
  createdAt: string
  completedAt?: string
  error?: string
}

export default function JobQueue({ jobs }: { jobs: Job[] }) {
  const columns: Column<Job>[] = [
    { key: 'id', label: 'Job ID', sortable: true },
    { key: 'type', label: 'Type', sortable: true },
    {
      key: 'status',
      label: 'Status',
      render: (value) => (
        <span
          className={`
            px-2 py-1 rounded text-xs
            ${
              value === 'completed'
                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                : value === 'running'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'
                : value === 'failed'
                ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
                : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400'
            }
          `}
        >
          {value}
        </span>
      ),
    },
    { key: 'createdAt', label: 'Created', sortable: true },
    { key: 'completedAt', label: 'Completed', sortable: true },
  ]
  
  return <DataTable data={jobs} columns={columns} searchable />
}


