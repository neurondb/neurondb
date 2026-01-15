'use client'

import { useState } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'

interface Job {
  id: string
  type: 'memory_promotion' | 'background_task'
  status: 'pending' | 'running' | 'completed' | 'failed'
  createdAt: string
  completedAt?: string
  error?: string
}

export default function JobsPage() {
  const [jobs] = useState<Job[]>([])
  
  const columns: Column<Job>[] = [
    { key: 'id', label: 'Job ID', sortable: true },
    { key: 'type', label: 'Type', sortable: true },
    {
      key: 'status',
      label: 'Status',
      render: (value) => (
        <span
          className={`
            px-2 py-1 rounded text-xs font-medium
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
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'Agents', href: '/agents' },
            { label: 'Background Jobs' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Background Jobs
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Monitor background job queue and worker pool
          </p>
        </div>
        
        <DataTable data={jobs} columns={columns} searchable />
      </div>
    </MainContent>
  )
}

