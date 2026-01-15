'use client'

import { useState } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'
import EmptyState from '@/components/EmptyState'

interface Index {
  name: string
  type: 'HNSW' | 'IVF'
  table: string
  column: string
  dimensions: number
  status: 'active' | 'building' | 'error'
}

export default function IndexManagerPage() {
  const [indexes, setIndexes] = useState<Index[]>([])
  
  const columns: Column<Index>[] = [
    { key: 'name', label: 'Index Name', sortable: true },
    { key: 'type', label: 'Type', sortable: true },
    { key: 'table', label: 'Table', sortable: true },
    { key: 'column', label: 'Column', sortable: true },
    { key: 'dimensions', label: 'Dimensions', sortable: true },
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
                : value === 'building'
                ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400'
                : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
            }
          `}
        >
          {value}
        </span>
      ),
    },
  ]
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'NeuronDB', href: '/neurondb' },
            { label: 'Index Management' },
          ]}
          className="mb-6"
        />
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              Index Management
            </h1>
            <p className="text-slate-600 dark:text-slate-400 mt-1">
              Manage HNSW and IVF indexes with quantization
            </p>
          </div>
          <button className="btn-primary">Create Index</button>
        </div>
        
        {indexes.length > 0 ? (
          <DataTable data={indexes} columns={columns} />
        ) : (
          <EmptyState
            icon="📊"
            title="No indexes found"
            description="Create an index to improve vector search performance"
            action={{
              label: 'Create Index',
              onClick: () => {},
            }}
          />
        )}
      </div>
    </MainContent>
  )
}

