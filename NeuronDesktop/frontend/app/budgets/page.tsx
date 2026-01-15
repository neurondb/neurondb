'use client'

import { useState } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import { LineChart } from '@/components/Charts'
import DataTable, { type Column } from '@/components/DataTable'

interface Budget {
  agentId: string
  agentName: string
  budget: number
  spent: number
  remaining: number
  status: 'ok' | 'warning' | 'exceeded'
}

export default function BudgetsPage() {
  const [budgets] = useState<Budget[]>([])
  
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
            px-2 py-1 rounded text-xs font-medium
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
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'Agents', href: '/agents' },
            { label: 'Budget & Cost Management' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Budget & Cost Management
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Real-time cost tracking and budget controls
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
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
                  <span className="text-sm font-semibold">$0.00</span>
                </div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Total Spent</span>
                  <span className="text-sm font-semibold">$0.00</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Remaining</span>
                  <span className="text-sm font-semibold">$0.00</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <DataTable data={budgets} columns={columns} />
      </div>
    </MainContent>
  )
}


