'use client'

interface ContextViewerProps {
  documents: string[]
  method?: string
  similarities?: number[]
}

export default function ContextViewer({ documents, method, similarities }: ContextViewerProps) {
  if (!documents || documents.length === 0) {
    return null
  }

  return (
    <div>
      <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
        Retrieved Context ({documents.length} documents)
        {method && (
          <span className="ml-2 text-xs text-slate-500 dark:text-slate-400">
            (Method: {method})
          </span>
        )}
      </h4>
      <div className="space-y-2">
        {documents.map((doc, index) => (
          <div
            key={index}
            className="bg-slate-50 dark:bg-slate-900 rounded-md p-3 border border-slate-200 dark:border-slate-700"
          >
            <div className="flex items-start justify-between mb-1">
              <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
                Document {index + 1}
              </span>
              {similarities && similarities[index] !== undefined && (
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  Similarity: {(similarities[index] * 100).toFixed(1)}%
                </span>
              )}
            </div>
            <p className="text-sm text-slate-900 dark:text-slate-100 line-clamp-3">
              {doc}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
