'use client'

import { useState } from 'react'
import { XMarkIcon, QuestionMarkCircleIcon } from '@heroicons/react/24/outline'
import { shortcuts, getShortcutString } from '@/lib/keyboard'

interface HelpCenterProps {
  isOpen: boolean
  onClose: () => void
}

export default function HelpCenter({ isOpen, onClose }: HelpCenterProps) {
  const groupedShortcuts = shortcuts.reduce((acc, shortcut) => {
    const category = shortcut.category || 'Other'
    if (!acc[category]) {
      acc[category] = []
    }
    acc[category].push(shortcut)
    return acc
  }, {} as Record<string, typeof shortcuts>)

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-slate-800 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-700 w-full max-w-3xl mx-4 max-h-[80vh] overflow-hidden animate-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-2">
            <QuestionMarkCircleIcon className="w-5 h-5 text-purple-600" />
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Help Center
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        <div className="overflow-y-auto max-h-[calc(80vh-80px)] p-4">
          <div className="space-y-6">
            {/* Keyboard Shortcuts */}
            <section>
              <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 mb-3">
                Keyboard Shortcuts
              </h3>
              {Object.entries(groupedShortcuts).map(([category, categoryShortcuts]) => (
                <div key={category} className="mb-4">
                  <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    {category}
                  </h4>
                  <div className="space-y-2">
                    {categoryShortcuts.map((shortcut) => (
                      <div
                        key={shortcut.action}
                        className="flex items-center justify-between p-2 rounded-md hover:bg-slate-50 dark:hover:bg-slate-700/50"
                      >
                        <span className="text-sm text-slate-600 dark:text-slate-400">
                          {shortcut.description}
                        </span>
                        <kbd className="px-2 py-1 text-xs font-semibold text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded">
                          {getShortcutString(shortcut)}
                        </kbd>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </section>

            {/* Getting Started */}
            <section>
              <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100 mb-3">
                Getting Started
              </h3>
              <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
                <p>
                  NeuronDesktop provides a unified interface for managing NeuronDB, NeuronMCP, and
                  NeuronAgent.
                </p>
                <ul className="list-disc list-inside space-y-1 ml-2">
                  <li>Use the command palette (⌘K) to quickly access features</li>
                  <li>Use global search to find tools, collections, and agents</li>
                  <li>Right-click on items for context menus</li>
                  <li>Use keyboard shortcuts for faster navigation</li>
                </ul>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  )
}

