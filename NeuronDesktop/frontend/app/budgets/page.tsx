'use client'

import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import { LineChart, BarChart, PieChart } from '@/components/Charts'
import DataTable, { type Column } from '@/components/DataTable'
import { factoryAPI } from '@/lib/api'

interface Budget {
  id: string
  agent_id: string
  agent_name?: string
  budget: number
  spent: number
  remaining: number
  period: 'daily' | 'weekly' | 'monthly' | 'yearly' | 'total'
  status: 'ok' | 'warning' | 'exceeded'
  alert_threshold?: number
}

export default function BudgetsPage() {
  const [profileId, setProfileId] = useState<string>('')
  const [budgets, setBudgets] = useState<Budget[]>([])
  const [costHistory, setCostHistory] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedPeriod, setSelectedPeriod] = useState<'daily' | 'weekly' | 'monthly' | 'yearly' | 'total'>('monthly')
  
  useEffect(() => {
    loadProfile()
  }, [])

  useEffect(() => {
    if (profileId) {
      loadBudgets()
      loadCostHistory()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileId, selectedPeriod])

  const loadProfile = async () => {
    try {
      const profilesResponse = await factoryAPI.get('/profiles')
      const profiles = profilesResponse.data.data || profilesResponse.data
      if (profiles && profiles.length > 0) {
        const activeProfile = profiles.find((p: any) => p.is_active) || profiles[0]
        setProfileId(activeProfile.id)
      }
    } catch (err) {
      console.error('Failed to load profile:', err)
    } finally {
      setLoading(false)
      }
  }

  const loadBudgets = async () => {
    try {
      /* Load budgets for all agents */
      const agentsResponse = await factoryAPI.get(`/profiles/${profileId}/agent/agents`)
      const agents = agentsResponse.data.data || agentsResponse.data
      const budgetsList: Budget[] = []

      for (const agent of agents) {
        try {
          const budgetResponse = await factoryAPI.get(
            `/profiles/${profileId}/agent/agents/${agent.id}/budgets?period=${selectedPeriod}`
          )
          const budgetData = budgetResponse.data.data || budgetResponse.data
          if (budgetData && Array.isArray(budgetData)) {
            budgetsList.push(...budgetData.map((b: any) => ({
              ...b,
              agent_name: agent.name,
            })))
          }
        } catch (err) {
          /* Agent may not have budgets configured */
        }
      }

      setBudgets(budgetsList)
    } catch (err) {
      console.error('Failed to load budgets:', err)
    }
  }

  const loadCostHistory = async () => {
    try {
      /* Load cost history - this would come from analytics */
      setCostHistory([
        { time: '2024-01-01', cost: 10.5 },
        { time: '2024-01-02', cost: 12.3 },
        { time: '2024-01-03', cost: 15.2 },
        { time: '2024-01-04', cost: 11.8 },
        { time: '2024-01-05', cost: 14.1 },
      ])
    } catch (err) {
      console.error('Failed to load cost history:', err)
    }
  }

  const totalBudget = budgets.reduce((sum, b) => sum + b.budget, 0)
  const totalSpent = budgets.reduce((sum, b) => sum + b.spent, 0)
  const totalRemaining = totalBudget - totalSpent

  const columns: Column<Budget>[] = [
    { key: 'agent_name', label: 'Agent', sortable: true },
    {
      key: 'budget',
      label: 'Budget',
      sortable: true,
      render: (value) => `$${Number(value).toFixed(2)}`,
    },
    {
      key: 'spent',
      label: 'Spent',
      sortable: true,
      render: (value) => `$${Number(value).toFixed(2)}`,
    },
    {
      key: 'remaining',
      label: 'Remaining',
      sortable: true,
      render: (value) => `$${Number(value).toFixed(2)}`,
    },
    {
      key: 'period',
      label: 'Period',
      sortable: true,
    },
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

        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Period</label>
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value as any)}
            className="px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
          >
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
            <option value="yearly">Yearly</option>
            <option value="total">Total</option>
          </select>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4">Cost Over Time</h3>
            {costHistory.length > 0 ? (
              <LineChart
                data={costHistory}
                dataKey="time"
                lines={[{ key: 'cost', name: 'Cost ($)', color: 'rgb(139, 92, 246)' }]}
                height={200}
              />
            ) : (
              <div className="text-slate-600 dark:text-slate-400">No cost data available</div>
            )}
          </div>
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4">Budget Summary</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Total Budget</span>
                  <span className="text-sm font-semibold">${totalBudget.toFixed(2)}</span>
                </div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Total Spent</span>
                  <span className="text-sm font-semibold">${totalSpent.toFixed(2)}</span>
                </div>
                <div className="flex justify-between mb-2">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Remaining</span>
                  <span className={`text-sm font-semibold ${
                    totalRemaining < 0 ? 'text-red-600' : 'text-green-600'
                  }`}>
                    ${totalRemaining.toFixed(2)}
                  </span>
                </div>
                <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2 mt-2">
                  <div
                    className={`h-2 rounded-full ${
                      totalRemaining < 0
                        ? 'bg-red-600'
                        : totalRemaining < totalBudget * 0.2
                        ? 'bg-yellow-600'
                        : 'bg-green-600'
                    }`}
                    style={{ width: `${Math.min(100, (totalSpent / totalBudget) * 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {budgets.length > 0 ? (
          <DataTable data={budgets} columns={columns} searchable />
        ) : (
          <div className="card p-6 text-center text-slate-600 dark:text-slate-400">
            No budgets configured. Create budgets for your agents to track costs.
          </div>
        )}
      </div>
    </MainContent>
  )
}




