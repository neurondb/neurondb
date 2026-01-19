'use client'

import { useParams } from 'next/navigation'
import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import RetrievalStats from '@/components/RetrievalStats'
import KnowledgeRouter from '@/components/KnowledgeRouter'
import { factoryAPI } from '@/lib/api'

interface Agent {
  id: string
  name: string
  description: string
}

export default function AgentRetrievalPage() {
  const params = useParams()
  const agentId = params.agentId as string
  const [profileId, setProfileId] = useState<string>('')
  const [agent, setAgent] = useState<Agent | null>(null)
  const [days, setDays] = useState(30)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadProfileAndAgent()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId])

  const loadProfileAndAgent = async () => {
    try {
      // Get active profile
      const profilesResponse = await factoryAPI.get('/profiles')
      const profiles = profilesResponse.data.data || profilesResponse.data
      if (profiles && profiles.length > 0) {
        const activeProfile = profiles.find((p: any) => p.is_active) || profiles[0]
        setProfileId(activeProfile.id)

        // Load agent details
        try {
          const agentResponse = await factoryAPI.get(
            `/profiles/${activeProfile.id}/agent/agents/${agentId}`
          )
          setAgent(agentResponse.data.data || agentResponse.data)
        } catch (err) {
          console.error('Failed to load agent:', err)
        }
      }
    } catch (err) {
      console.error('Failed to load profile:', err)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <MainContent>
        <div className="min-h-full bg-transparent p-6">
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-slate-200 dark:bg-slate-700 rounded w-1/4"></div>
            <div className="h-64 bg-slate-200 dark:bg-slate-700 rounded"></div>
          </div>
        </div>
      </MainContent>
    )
  }

  if (!profileId) {
    return (
      <MainContent>
        <div className="min-h-full bg-transparent p-6">
          <div className="text-red-600 dark:text-red-400">
            No active profile found. Please create or select a profile.
          </div>
        </div>
      </MainContent>
    )
  }

  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'Agents', href: '/agents' },
            { label: agent?.name || 'Agent', href: `/agents/${agentId}` },
            { label: 'Retrieval' },
          ]}
          className="mb-6"
        />

        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Agentic RAG - Retrieval Analytics
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Monitor retrieval decisions, knowledge routing, and retrieval quality
          </p>
        </div>

        <div className="mb-4 flex items-center gap-4">
          <label className="text-sm font-medium">Time Period:</label>
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
            <option value={365}>Last year</option>
          </select>
        </div>

        <div className="space-y-6">
          <RetrievalStats profileId={profileId} agentId={agentId} days={days} />
          <KnowledgeRouter profileId={profileId} agentId={agentId} />
        </div>
      </div>
    </MainContent>
  )
}
