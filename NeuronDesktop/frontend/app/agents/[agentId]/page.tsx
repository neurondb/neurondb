'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import SplitPanel from '@/components/SplitPanel'
import { LineChart, BarChart } from '@/components/Charts'
import DataTable, { type Column } from '@/components/DataTable'
import ToolRegistry from '@/components/ToolRegistry'

interface Agent {
  id: string
  name: string
  description: string
  model: string
  enabledTools: string[]
}

interface Session {
  id: string
  createdAt: string
  lastActivity: string
  messageCount: number
}

export default function AgentDetailPage() {
  const params = useParams()
  const agentId = params.agentId as string
  const [agent, setAgent] = useState<Agent | null>(null)
  const [sessions] = useState<Session[]>([])
  
  const sessionColumns: Column<Session>[] = [
    { key: 'id', label: 'Session ID', sortable: true },
    { key: 'createdAt', label: 'Created', sortable: true },
    { key: 'lastActivity', label: 'Last Activity', sortable: true },
    { key: 'messageCount', label: 'Messages', sortable: true },
  ]
  
  const leftPanel = (
    <div className="h-full p-4 space-y-6">
      {agent && (
        <>
          <div>
            <h2 className="text-xl font-bold mb-2">{agent.name}</h2>
            <p className="text-slate-600 dark:text-slate-400">{agent.description}</p>
          </div>
          
          <div>
            <h3 className="font-semibold mb-2">Model</h3>
            <p className="text-slate-600 dark:text-slate-400">{agent.model}</p>
          </div>
          
          <div>
            <h3 className="font-semibold mb-2">Enabled Tools</h3>
            <div className="flex flex-wrap gap-2">
              {agent.enabledTools.map((tool) => (
                <span
                  key={tool}
                  className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded text-xs"
                >
                  {tool}
                </span>
              ))}
            </div>
          </div>
          
          <div>
            <h3 className="font-semibold mb-2">Sessions</h3>
            <DataTable data={sessions} columns={sessionColumns} pageSize={5} />
          </div>
        </>
      )}
    </div>
  )
  
  const rightPanel = (
    <div className="h-full p-4">
      <div className="space-y-6">
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
          <LineChart
            data={[]}
            dataKey="time"
            lines={[
              { key: 'responseTime', name: 'Response Time', color: 'rgb(139, 92, 246)' },
            ]}
            height={200}
          />
        </div>
        
        <div>
          <h3 className="font-semibold mb-2">Tool Registry</h3>
          <ToolRegistry />
        </div>
      </div>
    </div>
  )
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'Agents', href: '/agents' },
            { label: agent?.name || 'Agent Details' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Agent Details
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Comprehensive agent interface with all features
          </p>
        </div>
        
        <div className="h-[calc(100vh-300px)]">
          <SplitPanel left={leftPanel} right={rightPanel} defaultLeftWidth={40} />
        </div>
      </div>
    </MainContent>
  )
}


