'use client'

import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import { factoryAPI } from '@/lib/api'

export default function MCPDatasetsPage() {
  const [profileId, setProfileId] = useState<string>('')
  const [sourceType, setSourceType] = useState('huggingface')
  const [sourcePath, setSourcePath] = useState('')
  const [format, setFormat] = useState('json')
  const [autoEmbed, setAutoEmbed] = useState(true)
  const [createIndexes, setCreateIndexes] = useState(true)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadProfile()
  }, [])

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

  const handleLoad = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!sourcePath.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await factoryAPI.post(`/profiles/${profileId}/mcp/datasets/load`, {
        source_type: sourceType,
        source_path: sourcePath,
        format: sourceType === 'url' ? format : undefined,
        auto_embed: autoEmbed,
        create_indexes: createIndexes,
      })
      setResult(response.data.data || response.data)
    } catch (err: any) {
      setError(err.message || 'Failed to load dataset')
    } finally {
      setLoading(false)
    }
  }

  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'MCP', href: '/mcp' },
            { label: 'Datasets' },
          ]}
          className="mb-6"
        />

        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Dataset Loading
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Load datasets from HuggingFace, URLs, GitHub, S3, or local files with automatic embedding
          </p>
        </div>

        <div className="max-w-2xl">
          <form onSubmit={handleLoad} className="card p-6 space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Source Type</label>
              <select
                value={sourceType}
                onChange={(e) => setSourceType(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
              >
                <option value="huggingface">HuggingFace</option>
                <option value="url">URL</option>
                <option value="github">GitHub</option>
                <option value="s3">S3</option>
                <option value="local">Local File</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Source Path</label>
              <input
                type="text"
                value={sourcePath}
                onChange={(e) => setSourcePath(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
                placeholder={
                  sourceType === 'huggingface'
                    ? 'sentence-transformers/embedding-training-data'
                    : sourceType === 'url'
                    ? 'https://example.com/data.csv'
                    : sourceType === 'github'
                    ? 'owner/repo/path/to/data.json'
                    : sourceType === 's3'
                    ? 's3://my-bucket/data.parquet'
                    : '/path/to/local/file.jsonl'
                }
                required
              />
            </div>

            {sourceType === 'url' && (
              <div>
                <label className="block text-sm font-medium mb-2">Format</label>
                <select
                  value={format}
                  onChange={(e) => setFormat(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
                >
                  <option value="csv">CSV</option>
                  <option value="json">JSON</option>
                  <option value="jsonl">JSONL</option>
                  <option value="parquet">Parquet</option>
                </select>
              </div>
            )}

            <div className="flex items-center space-x-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={autoEmbed}
                  onChange={(e) => setAutoEmbed(e.target.checked)}
                  className="mr-2"
                />
                <span>Auto-embed text columns</span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={createIndexes}
                  onChange={(e) => setCreateIndexes(e.target.checked)}
                  className="mr-2"
                />
                <span>Create indexes</span>
              </label>
            </div>

            {error && (
              <div className="text-red-600 dark:text-red-400 text-sm">{error}</div>
            )}

            {result && (
              <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-md">
                <div className="text-green-800 dark:text-green-400 font-semibold mb-2">
                  Dataset loaded successfully!
                </div>
                <pre className="text-sm overflow-auto">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !sourcePath.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Loading...' : 'Load Dataset'}
            </button>
          </form>
        </div>
      </div>
    </MainContent>
  )
}
