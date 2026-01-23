'use client'

import { useState, useEffect } from 'react'
import { ragAPI, type RAGPipeline } from '@/lib/api'
import LoadingSpinner from '@/components/LoadingSpinner'
import { Cog6ToothIcon, PlusIcon } from '@/components/Icons'

interface PipelineManagerProps {
  profileId: string
}

export default function PipelineManager({ profileId }: PipelineManagerProps) {
  const [pipelines, setPipelines] = useState<RAGPipeline[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newPipeline, setNewPipeline] = useState({
    pipeline_name: '',
    embedding_model: 'default',
    chunk_size: 512,
    chunk_overlap: 128,
  })

  const loadPipelines = async () => {
    if (!profileId) return
    setLoading(true)
    setError('')
    try {
      const response = await ragAPI.listPipelines(profileId)
      setPipelines(response.data.pipelines || [])
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to load pipelines')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPipelines()
  }, [profileId])

  const handleCreatePipeline = async () => {
    if (!newPipeline.pipeline_name.trim()) {
      setError('Pipeline name is required')
      return
    }

    setLoading(true)
    setError('')
    try {
      await ragAPI.createPipeline(profileId, newPipeline)
      setShowCreateForm(false)
      setNewPipeline({
        pipeline_name: '',
        embedding_model: 'default',
        chunk_size: 512,
        chunk_overlap: 128,
      })
      loadPipelines()
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to create pipeline')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100">
            RAG Pipelines
          </h2>
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            <PlusIcon className="w-5 h-5" />
            Create Pipeline
          </button>
        </div>

        {showCreateForm && (
          <div className="mb-6 p-4 bg-slate-50 dark:bg-slate-900 rounded-md border border-slate-200 dark:border-slate-700">
            <h3 className="text-lg font-medium mb-4 text-slate-900 dark:text-slate-100">
              Create New Pipeline
            </h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Pipeline Name
                </label>
                <input
                  type="text"
                  value={newPipeline.pipeline_name}
                  onChange={(e) => setNewPipeline({ ...newPipeline, pipeline_name: e.target.value })}
                  className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
                  placeholder="my-rag-pipeline"
                />
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Embedding Model
                  </label>
                  <input
                    type="text"
                    value={newPipeline.embedding_model}
                    onChange={(e) => setNewPipeline({ ...newPipeline, embedding_model: e.target.value })}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Chunk Size
                  </label>
                  <input
                    type="number"
                    value={newPipeline.chunk_size}
                    onChange={(e) => setNewPipeline({ ...newPipeline, chunk_size: parseInt(e.target.value) || 512 })}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Chunk Overlap
                  </label>
                  <input
                    type="number"
                    value={newPipeline.chunk_overlap}
                    onChange={(e) => setNewPipeline({ ...newPipeline, chunk_overlap: parseInt(e.target.value) || 128 })}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
                  />
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleCreatePipeline}
                  disabled={loading}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-slate-400"
                >
                  {loading ? <LoadingSpinner className="w-5 h-5" /> : 'Create'}
                </button>
                <button
                  onClick={() => setShowCreateForm(false)}
                  className="px-4 py-2 bg-slate-200 dark:bg-slate-700 text-slate-900 dark:text-slate-100 rounded-md hover:bg-slate-300 dark:hover:bg-slate-600"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="mb-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
            <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {loading && !pipelines.length ? (
          <div className="flex justify-center py-8">
            <LoadingSpinner className="w-8 h-8" />
          </div>
        ) : pipelines.length === 0 ? (
          <div className="text-center py-8 text-slate-500 dark:text-slate-400">
            No pipelines found. Create one to get started.
          </div>
        ) : (
          <div className="space-y-2">
            {pipelines.map((pipeline) => (
              <div
                key={pipeline.pipeline_id}
                className="p-4 bg-slate-50 dark:bg-slate-900 rounded-md border border-slate-200 dark:border-slate-700"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium text-slate-900 dark:text-slate-100">
                      {pipeline.pipeline_name}
                    </h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Model: {pipeline.embedding_model} | Created: {new Date(pipeline.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <Cog6ToothIcon className="w-5 h-5 text-slate-400" />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
