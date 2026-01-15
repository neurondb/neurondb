'use client'

import { LineChart, BarChart } from './Charts'
import DataTable, { type Column } from './DataTable'

interface Budget {
  agentId: string
  agentName: string
  budget: number
  spent: number
  remaining: number
  status: 'ok' | 'warning' | 'exceeded'
}

export default function BudgetDashboard({ budgets }: { budgets: Budget[] }) {
  const columns: Column<Budget>[] = [
    { key: 'agentName', label: 'Agent', sortable: true },
    { key: 'budget', label: 'Budget', sortable: true },
    { key: 'spent', label: 'Spent', sortable: true },
    { key: 'remaining', label: 'Remaining', sortable: true },
    {
      key: 'status',
      label: 'Status',
      render: (value) => (
        <span
          className={`
            px-2 py-1 rounded text-xs
            ${
              value === 'ok'
                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                : value === 'warning'
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
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Cost Over Time</h3>
          <LineChart
            data={[]}
            dataKey="time"
            lines={[{ key: 'cost', name: 'Cost', color: 'rgb(139, 92, 246)' }]}
            height={200}
          />
        </div>
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Budget Summary</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-600 dark:text-slate-400">Total Budget</span>
                <span className="text-sm font-semibold">
                  ${budgets.reduce((sum, b) => sum + b.budget, 0).toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between mb-1">
                <span className="text-sm text-slate-600 dark:text-slate-400">Total Spent</span>
                <span className="text-sm font-semibold">
                  ${budgets.reduce((sum, b) => sum + b.spent, 0).toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-slate-600 dark:text-slate-400">Remaining</span>
                <span className="text-sm font-semibold">
                  ${budgets.reduce((sum, b) => sum + b.remaining, 0).toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <DataTable data={budgets} columns={columns} />
    </div>
  )
}

