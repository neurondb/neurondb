'use client'

import { useParams } from 'next/navigation'
import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import MemoryQualityMetrics from '@/components/MemoryQualityMetrics'
import { factoryAPI } from '@/lib/api'

interface Agent {
  id: string
  name: string
}

export default function MemoryQualityPage() {
  const params = useParams()
  const agentId = params.agentId as string
  const [profileId, setProfileId] = useState<string>('')
  const [agent, setAgent] = useState<Agent | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadProfileAndAgent()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId])

  const loadProfileAndAgent = async () => {
    try {
      const profilesResponse = await factoryAPI.get('/profiles')
      const profiles = profilesResponse.data.data || profilesResponse.data
      if (profiles && profiles.length > 0) {
        const activeProfile = profiles.find((p: any) => p.is_active) || profiles[0]
        setProfileId(activeProfile.id)

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

  if (loading || !profileId) {
    return (
      <MainContent>
        <div className="min-h-full bg-transparent p-6">
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-slate-200 dark:bg-slate-700 rounded w-1/4"></div>
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
            { label: 'Memory', href: `/agents/${agentId}/memory` },
            { label: 'Quality' },
          ]}
          className="mb-6"
        />

        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Memory Quality Metrics
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Monitor memory quality, retrieval performance, and user feedback
          </p>
        </div>

        <MemoryQualityMetrics profileId={profileId} agentId={agentId} />
      </div>
    </MainContent>
  )
}
