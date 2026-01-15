'use client'

import { useState } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import { BarChart, LineChart } from '@/components/Charts'
import DataTable, { type Column } from '@/components/DataTable'

interface Evaluation {
  agentId: string
  agentName: string
  score: number
  metrics: {
    accuracy: number
    responseTime: number
    cost: number
  }
}

export default function EvaluationPage() {
  const [evaluations] = useState<Evaluation[]>([])
  
  const columns: Column<Evaluation>[] = [
    { key: 'agentName', label: 'Agent', sortable: true },
    { key: 'score', label: 'Quality Score', sortable: true },
    { key: 'metrics.accuracy', label: 'Accuracy', sortable: true },
    { key: 'metrics.responseTime', label: 'Response Time', sortable: true },
    { key: 'metrics.cost', label: 'Cost', sortable: true },
  ]
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'Agents', href: '/agents' },
            { label: 'Evaluation Framework' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Evaluation Framework
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Agent performance evaluation and quality scoring
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">Quality Scores</h3>
            <BarChart
              data={[]}
              bars={[{ key: 'score', name: 'Quality Score', color: 'rgb(139, 92, 246)' }]}
              xAxisKey="agentName"
              height={200}
            />
          </div>
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">Performance Over Time</h3>
            <LineChart
              data={[]}
              dataKey="time"
              lines={[{ key: 'score', name: 'Score', color: 'rgb(139, 92, 246)' }]}
              height={200}
            />
          </div>
        </div>
        
        <DataTable data={evaluations} columns={columns} />
      </div>
    </MainContent>
  )
}


