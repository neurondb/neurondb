'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { MagnifyingGlassIcon, CommandLineIcon } from '@heroicons/react/24/outline'
import { shortcuts, getShortcutString, type KeyboardShortcut } from '@/lib/keyboard'

interface Command {
  id: string
  label: string
  description?: string
  category: string
  action: () => void
  shortcut?: KeyboardShortcut
}

interface CommandPaletteProps {
  isOpen: boolean
  onClose: () => void
}

export default function CommandPalette({ isOpen, onClose }: CommandPaletteProps) {
  const router = useRouter()
  const [search, setSearch] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const commands: Command[] = [
    {
      id: 'dashboard',
      label: 'Go to Dashboard',
      description: 'Navigate to the main dashboard',
      category: 'Navigation',
      action: () => router.push('/dashboard'),
      shortcut: shortcuts.find((s) => s.action === 'view-dashboard'),
    },
    {
      id: 'neurondb',
      label: 'Go to NeuronDB',
      description: 'Open NeuronDB console',
      category: 'Navigation',
      action: () => router.push('/neurondb'),
      shortcut: shortcuts.find((s) => s.action === 'view-neurondb'),
    },
    {
      id: 'mcp',
      label: 'Go to MCP Console',
      description: 'Open MCP console',
      category: 'Navigation',
      action: () => router.push('/mcp'),
      shortcut: shortcuts.find((s) => s.action === 'view-mcp'),
    },
    {
      id: 'agents',
      label: 'Go to Agents',
      description: 'Open agents page',
      category: 'Navigation',
      action: () => router.push('/agents'),
      shortcut: shortcuts.find((s) => s.action === 'view-agents'),
    },
    {
      id: 'new-agent',
      label: 'Create New Agent',
      description: 'Create a new AI agent',
      category: 'Actions',
      action: () => router.push('/agents/create'),
    },
    {
      id: 'settings',
      label: 'Open Settings',
      description: 'Go to settings page',
      category: 'Actions',
      action: () => router.push('/settings'),
    },
  ]

  const filteredCommands = commands.filter((cmd) => {
    const searchLower = search.toLowerCase()
    return (
      cmd.label.toLowerCase().includes(searchLower) ||
      cmd.description?.toLowerCase().includes(searchLower) ||
      cmd.category.toLowerCase().includes(searchLower)
    )
  })

  const groupedCommands = filteredCommands.reduce((acc, cmd) => {
    if (!acc[cmd.category]) {
      acc[cmd.category] = []
    }
    acc[cmd.category].push(cmd)
    return acc
  }, {} as Record<string, Command[]>)

  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus()
      setSearch('')
      setSelectedIndex(0)
    }
  }, [isOpen])

  useEffect(() => {
    if (listRef.current && selectedIndex >= 0) {
      const selectedElement = listRef.current.children[selectedIndex] as HTMLElement
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
      }
    }
  }, [selectedIndex])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex((prev) => Math.min(prev + 1, filteredCommands.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex((prev) => Math.max(prev - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (filteredCommands[selectedIndex]) {
        filteredCommands[selectedIndex].action()
        onClose()
      }
    } else if (e.key === 'Escape') {
      onClose()
    }
  }

  const handleSelect = (command: Command) => {
    command.action()
    onClose()
  }

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]"
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
            value={search}
            onChange={(e) => {
              setSearch(e.target.value)
              setSelectedIndex(0)
            }}
            onKeyDown={handleKeyDown}
            placeholder="Type a command or search..."
            className="flex-1 bg-transparent border-none outline-none text-slate-900 dark:text-slate-100 placeholder-slate-400"
          />
          <kbd className="px-2 py-1 text-xs font-semibold text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded">
            ESC
          </kbd>
        </div>

        {/* Commands List */}
        <div
          ref={listRef}
          className="max-h-96 overflow-y-auto"
        >
          {Object.entries(groupedCommands).map(([category, categoryCommands]) => (
            <div key={category}>
              <div className="px-4 py-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider bg-slate-50 dark:bg-slate-900">
                {category}
              </div>
              {categoryCommands.map((command, index) => {
                const globalIndex = filteredCommands.indexOf(command)
                return (
                  <button
                    key={command.id}
                    onClick={() => handleSelect(command)}
                    className={`
                      w-full px-4 py-3 text-left hover:bg-slate-100 dark:hover:bg-slate-700
                      transition-colors duration-150
                      flex items-center justify-between
                      ${globalIndex === selectedIndex ? 'bg-purple-50 dark:bg-purple-900/20' : ''}
                    `}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-slate-900 dark:text-slate-100">
                        {command.label}
                      </div>
                      {command.description && (
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 truncate">
                          {command.description}
                        </div>
                      )}
                    </div>
                    {command.shortcut && (
                      <kbd className="ml-4 px-2 py-1 text-xs font-semibold text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded whitespace-nowrap">
                        {getShortcutString(command.shortcut)}
                      </kbd>
                    )}
                  </button>
                )
              })}
            </div>
          ))}
          {filteredCommands.length === 0 && (
            <div className="px-4 py-8 text-center text-slate-500 dark:text-slate-400">
              No commands found
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
            <span>Command Palette</span>
          </div>
        </div>
      </div>
    </div>
  )
}


