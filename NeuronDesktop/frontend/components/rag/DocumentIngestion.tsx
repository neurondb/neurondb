'use client'

import { useState } from 'react'
import { ragAPI, type CollectionInfo, type RAGIngestRequest } from '@/lib/api'
import LoadingSpinner from '@/components/LoadingSpinner'
import { DocumentTextIcon, ArrowUpTrayIcon } from '@/components/Icons'

interface DocumentIngestionProps {
  profileId: string
  collections: CollectionInfo[]
  onIngestComplete?: () => void
}

export default function DocumentIngestion({ profileId, collections, onIngestComplete }: DocumentIngestionProps) {
  const [documentText, setDocumentText] = useState('')
  const [selectedCollection, setSelectedCollection] = useState<string>('')
  const [chunkSize, setChunkSize] = useState(512)
  const [chunkOverlap, setChunkOverlap] = useState(128)
  const [embeddingModel, setEmbeddingModel] = useState('default')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string>('')

  const handleIngest = async () => {
    if (!profileId || !selectedCollection || !documentText.trim()) {
      setError('Please fill in all required fields')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const request: RAGIngestRequest = {
        document_text: documentText,
        table_name: selectedCollection,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        embedding_model: embeddingModel,
      }

      const response = await ragAPI.ingest(profileId, request)
      setResult(response.data)
      setDocumentText('')
      if (onIngestComplete) {
        onIngestComplete()
      }
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Failed to ingest document')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-slate-900 dark:text-slate-100">
          Ingest Document
        </h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Collection (Table)
            </label>
            <select
              value={selectedCollection}
              onChange={(e) => setSelectedCollection(e.target.value)}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
            >
              <option value="">Select a collection</option>
              {collections.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.schema ? `${col.schema}.${col.name}` : col.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Document Text
            </label>
            <textarea
              value={documentText}
              onChange={(e) => setDocumentText(e.target.value)}
              rows={10}
              className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100 font-mono text-sm"
              placeholder="Enter or paste document text here..."
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Chunk Size
              </label>
              <input
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(parseInt(e.target.value) || 512)}
                min={1}
                max={10000}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Chunk Overlap
              </label>
              <input
                type="number"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(parseInt(e.target.value) || 128)}
                min={0}
                max={chunkSize - 1}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Embedding Model
              </label>
              <input
                type="text"
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-slate-900 dark:text-slate-100"
                placeholder="default"
              />
            </div>
          </div>

          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
              <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
            </div>
          )}

          {result && (
            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md p-4">
              <p className="text-sm text-green-800 dark:text-green-200">
                {result.message || `Successfully ingested ${result.chunks_created} chunks`}
              </p>
            </div>
          )}

          <button
            onClick={handleIngest}
            disabled={loading || !selectedCollection || !documentText.trim()}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-slate-400 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <LoadingSpinner className="w-5 h-5" />
                Ingesting...
              </>
            ) : (
              <>
                <ArrowUpTrayIcon className="w-5 h-5" />
                Ingest Document
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
