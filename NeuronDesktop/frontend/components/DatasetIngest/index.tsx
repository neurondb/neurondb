'use client'

import { useState } from 'react'
import { profilesAPI, neurondbAPI } from '@/lib/api'
import { showSuccessToast, showErrorToast } from '@/lib/errors'
import { 
  ArrowUpTrayIcon,
  LinkIcon,
  CloudIcon,
  CodeBracketIcon,
  DocumentTextIcon,
  CheckIcon
} from '@/components/Icons'

interface IngestSource {
  type: 'file' | 'url' | 's3' | 'github' | 'huggingface'
  name: string
  icon: React.ComponentType<any>
  description: string
}

const sources: IngestSource[] = [
  {
    type: 'file',
    name: 'Upload File',
    icon: ArrowUpTrayIcon,
    description: 'Upload CSV, JSON, JSONL, or Parquet files'
  },
  {
    type: 'url',
    name: 'From URL',
    icon: LinkIcon,
    description: 'Load from HTTP/HTTPS URL'
  },
  {
    type: 's3',
    name: 'S3 Bucket',
    icon: CloudIcon,
    description: 'Load from AWS S3 bucket'
  },
  {
    type: 'github',
    name: 'GitHub',
    icon: CodeBracketIcon,
    description: 'Load from GitHub repository'
  },
  {
    type: 'huggingface',
    name: 'HuggingFace',
    icon: DocumentTextIcon,
    description: 'Load from HuggingFace datasets'
  },
]

export default function DatasetIngest({ 
  profileId, 
  onComplete 
}: { 
  profileId: string
  onComplete?: () => void 
}) {
  const [selectedSource, setSelectedSource] = useState<IngestSource | null>(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  
  // Form state
  const [sourcePath, setSourcePath] = useState('')
  const [format, setFormat] = useState('auto')
  const [tableName, setTableName] = useState('')
  const [autoEmbed, setAutoEmbed] = useState(true)
  const [embeddingModel, setEmbeddingModel] = useState('text-embedding-3-small')
  const [createIndex, setCreateIndex] = useState(true)

  const handleSourceSelect = (source: IngestSource) => {
    setSelectedSource(source)
    setSourcePath('')
    setTableName('')
  }

  const handleIngest = async () => {
    if (!selectedSource || !sourcePath) {
      showErrorToast('Please select a source and provide a path')
      return
    }

    setLoading(true)
    setProgress(0)

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90))
      }, 500)

      const response = await neurondbAPI.ingestDataset(profileId, {
        source_type: selectedSource.type,
        source_path: sourcePath,
        format: format !== 'auto' ? format : undefined,
        table_name: tableName || undefined,
        auto_embed: autoEmbed,
        embedding_model: embeddingModel,
        create_index: createIndex,
      })

      clearInterval(progressInterval)
      setProgress(100)

      const jobId = response.data?.job_id || response.data?.id || 'unknown'
      showSuccessToast(`Dataset ingestion started: ${jobId}`)
      
      setTimeout(() => {
        setLoading(false)
        if (onComplete) onComplete()
      }, 1000)
    } catch (error: any) {
      setLoading(false)
      showErrorToast('Failed to ingest dataset: ' + (error.response?.data?.error || error.message))
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">Ingest Dataset</h2>
        <p className="text-slate-600 dark:text-slate-400">Load data from various sources into NeuronDB</p>
      </div>

      {/* Source Selection */}
      {!selectedSource && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sources.map((source) => {
            const Icon = source.icon
            return (
              <button
                key={source.type}
                onClick={() => handleSourceSelect(source)}
                className="p-6 border border-slate-200 dark:border-slate-700 rounded-lg hover:border-purple-500 hover:bg-purple-50 dark:hover:bg-purple-900/20 transition-all text-left"
              >
                <Icon className="w-8 h-8 text-purple-600 dark:text-purple-400 mb-3" />
                <h3 className="font-semibold text-slate-900 dark:text-slate-100 mb-1">
                  {source.name}
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  {source.description}
                </p>
              </button>
            )
          })}
        </div>
      )}

      {/* Ingest Form */}
      {selectedSource && (
        <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-6 space-y-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              {selectedSource.name}
            </h3>
            <button
              onClick={() => setSelectedSource(null)}
              className="text-slate-600 hover:text-slate-800 dark:text-slate-400"
            >
              ← Back
            </button>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Source Path *
            </label>
            <input
              type="text"
              value={sourcePath}
              onChange={(e) => setSourcePath(e.target.value)}
              className="input w-full"
              placeholder={
                selectedSource.type === 'file' ? '/path/to/file.csv' :
                selectedSource.type === 'url' ? 'https://example.com/data.json' :
                selectedSource.type === 's3' ? 's3://bucket/path/to/file.parquet' :
                selectedSource.type === 'github' ? 'owner/repo/path/to/data.json' :
                'dataset-name'
              }
            />
          </div>

          {selectedSource.type === 'file' && (
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Format
              </label>
              <select
                value={format}
                onChange={(e) => setFormat(e.target.value)}
                className="input w-full"
              >
                <option value="auto">Auto-detect</option>
                <option value="csv">CSV</option>
                <option value="json">JSON</option>
                <option value="jsonl">JSONL</option>
                <option value="parquet">Parquet</option>
              </select>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Table Name (optional)
            </label>
            <input
              type="text"
              value={tableName}
              onChange={(e) => setTableName(e.target.value)}
              className="input w-full"
              placeholder="documents"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Auto-generated if not provided
            </p>
          </div>

          <div className="space-y-3">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={autoEmbed}
                onChange={(e) => setAutoEmbed(e.target.checked)}
                className="w-4 h-4 text-purple-600 rounded"
              />
              <span className="ml-2 text-sm text-slate-700 dark:text-slate-300">
                Auto-generate embeddings for text columns
              </span>
            </label>

            {autoEmbed && (
              <div className="ml-6">
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Embedding Model
                </label>
                <select
                  value={embeddingModel}
                  onChange={(e) => setEmbeddingModel(e.target.value)}
                  className="input w-full"
                >
                  <option value="text-embedding-3-small">text-embedding-3-small</option>
                  <option value="text-embedding-3-large">text-embedding-3-large</option>
                  <option value="text-embedding-ada-002">text-embedding-ada-002</option>
                </select>
              </div>
            )}

            <label className="flex items-center">
              <input
                type="checkbox"
                checked={createIndex}
                onChange={(e) => setCreateIndex(e.target.checked)}
                className="w-4 h-4 text-purple-600 rounded"
              />
              <span className="ml-2 text-sm text-slate-700 dark:text-slate-300">
                Create HNSW index for embeddings
              </span>
            </label>
          </div>

          {loading && (
            <div className="space-y-2">
              <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                <div
                  className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-sm text-slate-600 dark:text-slate-400 text-center">
                {progress}% complete
              </p>
            </div>
          )}

          <div className="flex gap-2">
            <button
              onClick={() => setSelectedSource(null)}
              className="btn btn-secondary flex-1"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              onClick={handleIngest}
              disabled={loading || !sourcePath}
              className="btn btn-primary flex-1"
            >
              {loading ? 'Ingesting...' : 'Start Ingestion'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}







