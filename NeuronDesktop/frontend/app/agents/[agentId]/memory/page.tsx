'use client'

import { useState } from 'react'
import { useParams } from 'next/navigation'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'
import { LineChart } from '@/components/Charts'

interface MemoryChunk {
  id: string
  content: string
  importance: number
  createdAt: string
  accessedCount: number
}

export default function AgentMemoryPage() {
  const params = useParams()
  const agentId = params.agentId as string
  const [chunks] = useState<MemoryChunk[]>([])
  
  const columns: Column<MemoryChunk>[] = [
    { key: 'content', label: 'Content', sortable: true },
    { key: 'importance', label: 'Importance', sortable: true },
    { key: 'createdAt', label: 'Created', sortable: true },
    { key: 'accessedCount', label: 'Accesses', sortable: true },
  ]
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'Agents', href: '/agents' },
            { label: 'Agent Details', href: `/agents/${agentId}` },
            { label: 'Memory' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Hierarchical Memory
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Memory chunks with HNSW-based vector search
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
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
            <h3 className="text-lg font-semibold mb-4">Memory Statistics</h3>
            <div className="space-y-4">
              <div>
                <div className="text-sm text-slate-600 dark:text-slate-400">Total Chunks</div>
                <div className="text-2xl font-bold">0</div>
              </div>
              <div>
                <div className="text-sm text-slate-600 dark:text-slate-400">Avg Importance</div>
                <div className="text-2xl font-bold">0.0</div>
              </div>
            </div>
          </div>
        </div>
        
        <DataTable data={chunks} columns={columns} searchable />
      </div>
    </MainContent>
  )
}

