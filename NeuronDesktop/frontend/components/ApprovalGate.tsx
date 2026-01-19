'use client'

import { useState } from 'react'

interface ApprovalGateProps {
  executionId: string
  workflowId: string
  stepId: string
  message: string
  metadata?: Record<string, any>
  onApprove: (executionId: string, stepId: string) => Promise<void>
  onReject: (executionId: string, stepId: string, reason?: string) => Promise<void>
}

export default function ApprovalGate({
  executionId,
  workflowId,
  stepId,
  message,
  metadata,
  onApprove,
  onReject,
}: ApprovalGateProps) {
  const [rejectReason, setRejectReason] = useState('')
  const [processing, setProcessing] = useState(false)

  const handleApprove = async () => {
    setProcessing(true)
    try {
      await onApprove(executionId, stepId)
    } finally {
      setProcessing(false)
    }
  }

  const handleReject = async () => {
    setProcessing(true)
    try {
      await onReject(executionId, stepId, rejectReason)
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="card p-6 border-2 border-yellow-400 dark:border-yellow-600">
      <div className="flex items-start mb-4">
        <div className="flex-shrink-0">
          <div className="w-10 h-10 bg-yellow-100 dark:bg-yellow-900/30 rounded-full flex items-center justify-center">
            <span className="text-yellow-600 dark:text-yellow-400 text-xl">⚠️</span>
          </div>
        </div>
        <div className="ml-4 flex-1">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            Approval Required
          </h3>
          <p className="text-slate-600 dark:text-slate-400 mt-1">{message}</p>
        </div>
      </div>

      {metadata && Object.keys(metadata).length > 0 && (
        <div className="mb-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-md">
          <h4 className="text-sm font-semibold mb-2">Context</h4>
          <pre className="text-xs overflow-auto">
            {JSON.stringify(metadata, null, 2)}
          </pre>
        </div>
      )}

      <div className="space-y-3">
        <div>
          <label className="block text-sm font-medium mb-2">Rejection Reason (optional)</label>
          <textarea
            value={rejectReason}
            onChange={(e) => setRejectReason(e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
            rows={2}
            placeholder="Provide a reason for rejection..."
          />
        </div>

        <div className="flex gap-3">
          <button
            onClick={handleApprove}
            disabled={processing}
            className="flex-1 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {processing ? 'Processing...' : 'Approve'}
          </button>
          <button
            onClick={handleReject}
            disabled={processing}
            className="flex-1 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {processing ? 'Processing...' : 'Reject'}
          </button>
        </div>
      </div>
    </div>
  )
}
