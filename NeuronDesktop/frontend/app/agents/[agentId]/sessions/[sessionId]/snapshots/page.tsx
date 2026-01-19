'use client'

import { useParams } from 'next/navigation'
import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'
import { factoryAPI } from '@/lib/api'

interface Snapshot {
  id: string
  session_id: string
  agent_id: string
  user_message: string
  created_at: string
  deterministic_mode: boolean
}

export default function SnapshotsPage() {
  const params = useParams()
  const agentId = params.agentId as string
  const sessionId = params.sessionId as string
  const [profileId, setProfileId] = useState<string>('')
  const [snapshots, setSnapshots] = useState<Snapshot[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedSnapshot, setSelectedSnapshot] = useState<Snapshot | null>(null)

  useEffect(() => {
    loadProfile()
  }, [])

  useEffect(() => {
    if (profileId && sessionId) {
      loadSnapshots()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileId, sessionId])

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

  const loadSnapshots = async () => {
    try {
      const response = await factoryAPI.get(
        `/profiles/${profileId}/agent/sessions/${sessionId}/snapshots`
      )
      const data = response.data.data || response.data
      setSnapshots(Array.isArray(data) ? data : [])
    } catch (err) {
      console.error('Failed to load snapshots:', err)
    }
  }

  const handleCreateSnapshot = async () => {
    try {
      await factoryAPI.post(`/profiles/${profileId}/agent/sessions/${sessionId}/snapshots`, {
        user_message: 'Current state',
        deterministic_mode: false,
      })
      loadSnapshots()
    } catch (err) {
      console.error('Failed to create snapshot:', err)
    }
  }

  const handleReplay = async (snapshotId: string) => {
    try {
      const response = await factoryAPI.post(
        `/profiles/${profileId}/agent/snapshots/${snapshotId}/replay`
      )
      console.log('Replay started:', response.data)
    } catch (err) {
      console.error('Failed to replay snapshot:', err)
    }
  }

  const handleDelete = async (snapshotId: string) => {
    try {
      await factoryAPI.delete(`/profiles/${profileId}/agent/snapshots/${snapshotId}`)
      loadSnapshots()
    } catch (err) {
      console.error('Failed to delete snapshot:', err)
    }
  }

  const columns: Column<Snapshot>[] = [
    { key: 'id', label: 'Snapshot ID', sortable: true },
    { key: 'user_message', label: 'User Message', sortable: true },
    {
      key: 'deterministic_mode',
      label: 'Deterministic',
      render: (value) => (value ? 'Yes' : 'No'),
    },
    { key: 'created_at', label: 'Created', sortable: true },
    {
      key: 'actions',
      label: 'Actions',
      render: (value, row) => (
        <div className="flex gap-2">
          <button
            onClick={() => handleReplay(row.id)}
            className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Replay
          </button>
          <button
            onClick={() => handleDelete(row.id)}
            className="px-2 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700"
          >
            Delete
          </button>
        </div>
      ),
    },
  ]

  if (loading) {
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
            { label: 'Agent', href: `/agents/${agentId}` },
            { label: 'Sessions', href: `/agents/${agentId}/sessions` },
            { label: 'Snapshots' },
          ]}
          className="mb-6"
        />

        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              Execution Snapshots
            </h1>
            <p className="text-slate-600 dark:text-slate-400 mt-1">
              Capture and replay agent execution states
            </p>
          </div>
          <button
            onClick={handleCreateSnapshot}
            className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
          >
            Create Snapshot
          </button>
        </div>

        {snapshots.length > 0 ? (
          <DataTable data={snapshots} columns={columns} searchable />
        ) : (
          <div className="card p-6 text-center text-slate-600 dark:text-slate-400">
            No snapshots yet. Create a snapshot to capture the current execution state.
          </div>
        )}
      </div>
    </MainContent>
  )
}
