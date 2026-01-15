'use client'

import { useState } from 'react'
import DataTable, { type Column } from './DataTable'
import EmptyState from './EmptyState'

interface Index {
  name: string
  type: 'HNSW' | 'IVF'
  table: string
  column: string
  dimensions: number
  status: 'active' | 'building' | 'error'
}

export default function IndexManager() {
  const [indexes] = useState<Index[]>([])
  
  const columns: Column<Index>[] = [
    { key: 'name', label: 'Name', sortable: true },
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
            px-2 py-1 rounded text-xs
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
    <div>
      {indexes.length > 0 ? (
        <DataTable data={indexes} columns={columns} />
      ) : (
        <EmptyState
          icon="📊"
          title="No indexes"
          description="Create an index to improve search performance"
          action={{
            label: 'Create Index',
            onClick: () => {},
          }}
        />
      )}
    </div>
  )
}

