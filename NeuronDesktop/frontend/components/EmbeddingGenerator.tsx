'use client'

import { useState } from 'react'
import { PlayIcon } from '@heroicons/react/24/outline'

export default function EmbeddingGenerator() {
  const [text, setText] = useState('')
  const [model, setModel] = useState('text-embedding-ada-002')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  
  const handleGenerate = async () => {
    setLoading(true)
    try {
      // This would call the actual API
      // const response = await neurondbAPI.generateEmbedding(text, model)
      // setResult(response)
      setResult({ vector: [0.1, 0.2, 0.3], dimensions: 1536 })
    } catch (error) {
      setResult({ error: String(error) })
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Text</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="input"
          rows={5}
          placeholder="Enter text to generate embedding..."
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Model</label>
        <input
          type="text"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="input"
          placeholder="text-embedding-ada-002"
        />
      </div>
      
      <button
        onClick={handleGenerate}
        disabled={loading || !text}
        className="btn-primary flex items-center gap-2"
      >
        <PlayIcon className="w-4 h-4" />
        {loading ? 'Generating...' : 'Generate Embedding'}
      </button>
      
      {result && (
        <div>
          <label className="block text-sm font-medium mb-2">Result</label>
          <pre className="input font-mono text-sm overflow-auto max-h-96">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}


