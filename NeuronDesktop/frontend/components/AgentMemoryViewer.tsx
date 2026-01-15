'use client'

import { useState } from 'react'
import { LineChart } from './Charts'
import DataTable, { type Column } from './DataTable'

interface MemoryChunk {
  id: string
  content: string
  importance: number
  createdAt: string
  accessedCount: number
}

export default function AgentMemoryViewer({ chunks }: { chunks: MemoryChunk[] }) {
  const columns: Column<MemoryChunk>[] = [
    {
      key: 'content',
      label: 'Content',
      render: (value) => (
        <div className="max-w-md truncate" title={String(value)}>
          {String(value)}
        </div>
      ),
    },
    { key: 'importance', label: 'Importance', sortable: true },
    { key: 'createdAt', label: 'Created', sortable: true },
    { key: 'accessedCount', label: 'Accesses', sortable: true },
  ]
  
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Memory Access Over Time</h3>
          <LineChart
            data={[]}
            dataKey="time"
            lines={[{ key: 'accesses', name: 'Accesses', color: 'rgb(139, 92, 246)' }]}
            height={200}
          />
        </div>
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Statistics</h3>
          <div className="space-y-4">
            <div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Total Chunks</div>
              <div className="text-2xl font-bold">{chunks.length}</div>
            </div>
            <div>
              <div className="text-sm text-slate-600 dark:text-slate-400">Avg Importance</div>
              <div className="text-2xl font-bold">
                {chunks.length > 0
                  ? (chunks.reduce((sum, c) => sum + c.importance, 0) / chunks.length).toFixed(2)
                  : '0.00'}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <DataTable data={chunks} columns={columns} searchable />
    </div>
  )
}

