'use client'

import { BarChart, LineChart } from './Charts'
import DataTable, { type Column } from './DataTable'

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

export default function EvaluationDashboard({ evaluations }: { evaluations: Evaluation[] }) {
  const columns: Column<Evaluation>[] = [
    { key: 'agentName', label: 'Agent', sortable: true },
    { key: 'score', label: 'Quality Score', sortable: true },
    { key: 'metrics.accuracy', label: 'Accuracy', sortable: true },
    { key: 'metrics.responseTime', label: 'Response Time', sortable: true },
    { key: 'metrics.cost', label: 'Cost', sortable: true },
  ]
  
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Quality Scores</h3>
          <BarChart
            data={evaluations.map(e => ({ agentName: e.agentName, score: e.score }))}
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
  )
}


