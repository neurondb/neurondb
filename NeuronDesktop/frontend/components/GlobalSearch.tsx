'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { MagnifyingGlassIcon, CommandLineIcon } from '@heroicons/react/24/outline'
import { search, type SearchResult } from '@/lib/search'

interface GlobalSearchProps {
  isOpen: boolean
  onClose: () => void
}

export default function GlobalSearch({ isOpen, onClose }: GlobalSearchProps) {
  const router = useRouter()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus()
      setQuery('')
      setResults([])
      setSelectedIndex(0)
    }
  }, [isOpen])

  useEffect(() => {
    if (query.trim()) {
      const searchResults = search(query)
      setResults(searchResults)
      setSelectedIndex(0)
    } else {
      setResults([])
    }
  }, [query])

  useEffect(() => {
    if (listRef.current && selectedIndex >= 0 && results.length > 0) {
      const selectedElement = listRef.current.children[selectedIndex] as HTMLElement
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
      }
    }
  }, [selectedIndex, results])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex((prev) => Math.min(prev + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex((prev) => Math.max(prev - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (results[selectedIndex]?.url) {
        router.push(results[selectedIndex].url!)
        onClose()
      }
    } else if (e.key === 'Escape') {
      onClose()
    }
  }

  const handleSelect = (result: SearchResult) => {
    if (result.url) {
      router.push(result.url)
      onClose()
    }
  }

  const getTypeIcon = (type: SearchResult['type']) => {
    switch (type) {
      case 'mcp-tool':
        return '🔧'
      case 'neurondb-collection':
        return '🗄️'
      case 'agent':
        return '🤖'
      case 'log':
        return '📋'
      case 'page':
        return '📄'
      default:
        return '📌'
    }
  }

  const getTypeLabel = (type: SearchResult['type']) => {
    switch (type) {
      case 'mcp-tool':
        return 'MCP Tool'
      case 'neurondb-collection':
        return 'Collection'
      case 'agent':
        return 'Agent'
      case 'log':
        return 'Log'
      case 'page':
        return 'Page'
      default:
        return 'Item'
    }
  }

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl mx-4 bg-white dark:bg-slate-800 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-700 overflow-hidden animate-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-slate-200 dark:border-slate-700">
          <MagnifyingGlassIcon className="w-5 h-5 text-slate-400" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search tools, collections, agents, logs..."
            className="flex-1 bg-transparent border-none outline-none text-slate-900 dark:text-slate-100 placeholder-slate-400"
          />
          <kbd className="px-2 py-1 text-xs font-semibold text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded">
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div ref={listRef} className="max-h-96 overflow-y-auto">
          {results.length > 0 ? (
            results.map((result, index) => (
              <button
                key={result.id}
                onClick={() => handleSelect(result)}
                className={`
                  w-full px-4 py-3 text-left hover:bg-slate-100 dark:hover:bg-slate-700
                  transition-colors duration-150
                  flex items-start gap-3
                  ${index === selectedIndex ? 'bg-purple-50 dark:bg-purple-900/20' : ''}
                `}
              >
                <span className="text-xl mt-0.5">{getTypeIcon(result.type)}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <div className="text-sm font-medium text-slate-900 dark:text-slate-100 truncate">
                      {result.title}
                    </div>
                    <span className="text-xs text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-700 px-2 py-0.5 rounded">
                      {getTypeLabel(result.type)}
                    </span>
                  </div>
                  {result.description && (
                    <div className="text-xs text-slate-500 dark:text-slate-400 line-clamp-2">
                      {result.description}
                    </div>
                  )}
                </div>
              </button>
            ))
          ) : query.trim() ? (
            <div className="px-4 py-8 text-center text-slate-500 dark:text-slate-400">
              No results found for &quot;{query}&quot;
            </div>
          ) : (
            <div className="px-4 py-8 text-center text-slate-500 dark:text-slate-400">
              Start typing to search...
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-slate-200 dark:bg-slate-700 rounded">↑</kbd>
              <kbd className="px-1.5 py-0.5 bg-slate-200 dark:bg-slate-700 rounded">↓</kbd>
              <span>Navigate</span>
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-slate-200 dark:bg-slate-700 rounded">Enter</kbd>
              <span>Select</span>
            </span>
          </div>
          <div className="flex items-center gap-1">
            <CommandLineIcon className="w-3 h-3" />
            <span>Global Search</span>
          </div>
        </div>
      </div>
    </div>
  )
}

