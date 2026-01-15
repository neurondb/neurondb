'use client'

import { useState } from 'react'
import DataTable, { type Column } from './DataTable'

interface Tool {
  name: string
  category: 'sql' | 'http' | 'code' | 'shell' | 'browser' | 'visualization' | 'filesystem' | 'memory' | 'collaboration' | 'neurondb' | 'multimodal'
  description: string
  enabled: boolean
}

const allTools: Tool[] = [
  { name: 'SQL', category: 'sql', description: 'Execute SQL queries', enabled: true },
  { name: 'HTTP', category: 'http', description: 'Make HTTP requests', enabled: true },
  { name: 'Code', category: 'code', description: 'Execute code', enabled: false },
  { name: 'Shell', category: 'shell', description: 'Execute shell commands', enabled: false },
  { name: 'Browser', category: 'browser', description: 'Browser automation with Playwright', enabled: false },
  { name: 'Visualization', category: 'visualization', description: 'Create visualizations', enabled: false },
  { name: 'Filesystem', category: 'filesystem', description: 'Virtual filesystem operations', enabled: false },
  { name: 'Memory', category: 'memory', description: 'Memory operations', enabled: true },
  { name: 'Collaboration', category: 'collaboration', description: 'Agent collaboration', enabled: false },
  { name: 'NeuronDB RAG', category: 'neurondb', description: 'RAG operations', enabled: true },
  { name: 'NeuronDB Hybrid Search', category: 'neurondb', description: 'Hybrid search', enabled: true },
  { name: 'NeuronDB Vector', category: 'neurondb', description: 'Vector operations', enabled: true },
  { name: 'NeuronDB ML', category: 'neurondb', description: 'ML operations', enabled: true },
  { name: 'Multimodal', category: 'multimodal', description: 'Multimodal processing', enabled: false },
]

export default function ToolRegistry() {
  const [tools, setTools] = useState<Tool[]>(allTools)
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  
  const categories = ['all', ...Array.from(new Set(tools.map(t => t.category)))]
  
  const filtered = selectedCategory === 'all'
    ? tools
    : tools.filter(t => t.category === selectedCategory)
  
  const toggleTool = (toolName: string) => {
    setTools(tools.map(t => t.name === toolName ? { ...t, enabled: !t.enabled } : t))
  }
  
  const columns: Column<Tool>[] = [
    { key: 'name', label: 'Tool', sortable: true },
    { key: 'category', label: 'Category', sortable: true },
    { key: 'description', label: 'Description', sortable: true },
    {
      key: 'enabled',
      label: 'Enabled',
      render: (value, row) => (
        <button
          onClick={() => toggleTool(row.name)}
          className={`
            px-3 py-1 rounded text-xs font-medium
            ${
              value
                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
            }
          `}
        >
          {value ? 'Enabled' : 'Disabled'}
        </button>
      ),
    },
  ]
  
  return (
    <div className="space-y-4">
      <div className="flex gap-2 flex-wrap">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`
              px-4 py-2 rounded-lg text-sm font-medium
              ${
                selectedCategory === cat
                  ? 'bg-purple-600 text-white'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
              }
            `}
          >
            {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </button>
        ))}
      </div>
      <DataTable data={filtered} columns={columns} searchable />
    </div>
  )
}


