'use client'

import { useState } from 'react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import { exportData, type ExportFormat } from '@/lib/export'

interface ExportDialogProps {
  isOpen: boolean
  onClose: () => void
  data: any[]
  defaultFilename?: string
}

export default function ExportDialog({
  isOpen,
  onClose,
  data,
  defaultFilename = 'export',
}: ExportDialogProps) {
  const [format, setFormat] = useState<ExportFormat>('csv')
  const [filename, setFilename] = useState(defaultFilename)

  if (!isOpen) return null

  const handleExport = () => {
    exportData(data, {
      format,
      filename: `${filename}.${format}`,
      includeHeaders: true,
    })
    onClose()
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-slate-800 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-700 w-full max-w-md mx-4 animate-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            Export Data
          </h2>
          <button
            onClick={onClose}
            className="p-1 rounded-md hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        <div className="p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Format
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(['csv', 'json', 'pdf'] as ExportFormat[]).map((fmt) => (
                <button
                  key={fmt}
                  onClick={() => setFormat(fmt)}
                  className={`
                    px-4 py-2 rounded-md text-sm font-medium
                    transition-colors duration-150
                    ${
                      format === fmt
                        ? 'bg-purple-600 text-white'
                        : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
                    }
                  `}
                >
                  {fmt.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              Filename
            </label>
            <input
              type="text"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              className="input"
              placeholder="export"
            />
          </div>

          <div className="text-sm text-slate-600 dark:text-slate-400">
            Exporting {data.length} row{data.length !== 1 ? 's' : ''}
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 p-4 border-t border-slate-200 dark:border-slate-700">
          <button onClick={onClose} className="btn-secondary">
            Cancel
          </button>
          <button onClick={handleExport} className="btn-primary">
            Export
          </button>
        </div>
      </div>
    </div>
  )
}

