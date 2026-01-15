'use client'

import { useState } from 'react'
import { useParams } from 'next/navigation'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'

interface Session {
  id: string
  createdAt: string
  lastActivity: string
  messageCount: number
}

export default function AgentSessionsPage() {
  const params = useParams()
  const agentId = params.agentId as string
  const [sessions] = useState<Session[]>([])
  
  const columns: Column<Session>[] = [
    { key: 'id', label: 'Session ID', sortable: true },
    { key: 'createdAt', label: 'Created', sortable: true },
    { key: 'lastActivity', label: 'Last Activity', sortable: true },
    { key: 'messageCount', label: 'Messages', sortable: true },
  ]
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'Agents', href: '/agents' },
            { label: 'Agent Details', href: `/agents/${agentId}` },
            { label: 'Sessions' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Sessions
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Session management and analytics
          </p>
        </div>
        <DataTable data={sessions} columns={columns} searchable />
      </div>
    </MainContent>
  )
}

