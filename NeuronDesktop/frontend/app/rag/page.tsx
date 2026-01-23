'use client'

import { useState, useEffect, useCallback } from 'react'
import { profilesAPI, ragAPI, neurondbAPI, type Profile, type CollectionInfo } from '@/lib/api'
import DocumentIngestion from '@/components/rag/DocumentIngestion'
import RAGQuery from '@/components/rag/RAGQuery'
import PipelineManager from '@/components/rag/PipelineManager'
import MainContent from '@/components/MainContent'
import Breadcrumbs from '@/components/Breadcrumbs'
import ProfileSelector from '@/components/ProfileSelector'
import { 
  DocumentTextIcon,
  MagnifyingGlassIcon,
  Cog6ToothIcon
} from '@/components/Icons'

type TabType = 'ingestion' | 'query' | 'pipelines'

export default function RAGPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [collections, setCollections] = useState<CollectionInfo[]>([])
  const [activeTab, setActiveTab] = useState<TabType>('query')
  const [loading, setLoading] = useState(false)

  const loadProfiles = useCallback(async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0 && !selectedProfile) {
        const activeProfileId = localStorage.getItem('active_profile_id')
        if (activeProfileId) {
          const activeProfile = response.data.find((p: Profile) => p.id === activeProfileId)
          if (activeProfile) {
            setSelectedProfile(activeProfileId)
            return
          }
        }
        const defaultProfile = response.data.find((p: Profile) => p.is_default)
        setSelectedProfile(defaultProfile ? defaultProfile.id : response.data[0].id)
      }
    } catch (error) {
      console.error('Failed to load profiles:', error)
    }
  }, [selectedProfile])

  const loadCollections = useCallback(async () => {
    if (!selectedProfile) return
    setLoading(true)
    try {
      const response = await neurondbAPI.listCollections(selectedProfile)
      setCollections(response.data)
    } catch (error) {
      console.error('Failed to load collections:', error)
    } finally {
      setLoading(false)
    }
  }, [selectedProfile])

  useEffect(() => {
    loadProfiles()
  }, [loadProfiles])

  useEffect(() => {
    if (selectedProfile) {
      loadCollections()
      localStorage.setItem('active_profile_id', selectedProfile)
    }
  }, [selectedProfile, loadCollections])

  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'RAG', href: '/rag' },
          ]}
          className="mb-6"
        />

        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            RAG Management
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Complete RAG pipeline: document ingestion, query, and pipeline management
          </p>
        </div>

        <div className="mb-4">
          <ProfileSelector
            profiles={profiles}
            selectedProfile={selectedProfile}
            onSelect={setSelectedProfile}
          />
        </div>

        {/* Tabs */}
        <div className="mb-6 border-b border-slate-200 dark:border-slate-700">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('ingestion')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'ingestion'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300 dark:text-slate-400 dark:hover:text-slate-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <DocumentTextIcon className="w-5 h-5" />
                Document Ingestion
              </div>
            </button>
            <button
              onClick={() => setActiveTab('query')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'query'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300 dark:text-slate-400 dark:hover:text-slate-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <MagnifyingGlassIcon className="w-5 h-5" />
                RAG Query
              </div>
            </button>
            <button
              onClick={() => setActiveTab('pipelines')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'pipelines'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300 dark:text-slate-400 dark:hover:text-slate-300'
              }`}
            >
              <div className="flex items-center gap-2">
                <Cog6ToothIcon className="w-5 h-5" />
                Pipeline Management
              </div>
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="mt-6">
          {activeTab === 'ingestion' && (
            <DocumentIngestion
              profileId={selectedProfile}
              collections={collections}
              onIngestComplete={loadCollections}
            />
          )}
          {activeTab === 'query' && (
            <RAGQuery
              profileId={selectedProfile}
              collections={collections}
            />
          )}
          {activeTab === 'pipelines' && (
            <PipelineManager
              profileId={selectedProfile}
            />
          )}
        </div>
      </div>
    </MainContent>
  )
}
