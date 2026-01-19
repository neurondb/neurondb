'use client'

import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'
import { factoryAPI } from '@/lib/api'

interface Resource {
  uri: string
  name: string
  description: string
  mimeType?: string
}

export default function MCPResourcesPage() {
  const [profileId, setProfileId] = useState<string>('')
  const [resources, setResources] = useState<Resource[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedResource, setSelectedResource] = useState<Resource | null>(null)
  const [resourceContent, setResourceContent] = useState<any>(null)

  useEffect(() => {
    loadProfile()
  }, [])

  useEffect(() => {
    if (profileId) {
      loadResources()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileId])

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
    }
  }

  const loadResources = async () => {
    try {
      setLoading(true)
      const response = await factoryAPI.get(`/profiles/${profileId}/mcp/resources`)
      const data = response.data.data || response.data
      setResources(data.resources || [])
    } catch (err) {
      console.error('Failed to load resources:', err)
    } finally {
      setLoading(false)
    }
  }

  const loadResourceContent = async (resource: Resource) => {
    try {
      setSelectedResource(resource)
      const response = await factoryAPI.get(
        `/profiles/${profileId}/mcp/resources/${encodeURIComponent(resource.uri)}`
      )
      const data = response.data.data || response.data
      setResourceContent(data)
    } catch (err) {
      console.error('Failed to load resource content:', err)
      setResourceContent(null)
    }
  }

  const columns: Column<Resource>[] = [
    { key: 'name', label: 'Name', sortable: true },
    { key: 'description', label: 'Description', sortable: true },
    { key: 'uri', label: 'URI', sortable: true },
  ]

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

  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'MCP', href: '/mcp' },
            { label: 'Resources' },
          ]}
          className="mb-6"
        />

        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            MCP Resources
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Browse and access MCP resources including schema, models, indexes, and system stats
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <DataTable
              data={resources}
              columns={columns}
              onRowClick={(resource) => loadResourceContent(resource)}
              searchable
            />
          </div>

          {selectedResource && (
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-4">{selectedResource.name}</h3>
              {resourceContent ? (
                <pre className="bg-slate-50 dark:bg-slate-800 p-4 rounded overflow-auto max-h-96">
                  {JSON.stringify(resourceContent, null, 2)}
                </pre>
              ) : (
                <div className="text-slate-600 dark:text-slate-400">Loading content...</div>
              )}
            </div>
          )}
        </div>
      </div>
    </MainContent>
  )
}
