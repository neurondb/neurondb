'use client'

import { ReactNode } from 'react'

interface MCPToolCategoryProps {
  category: string
  tools: Array<{ name: string; description: string }>
  onToolSelect?: (toolName: string) => void
}

export default function MCPToolCategory({
  category,
  tools,
  onToolSelect,
}: MCPToolCategoryProps) {
  return (
    <div className="mb-6">
      <h3 className="text-lg font-semibold mb-3 capitalize">{category}</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {tools.map((tool) => (
          <button
            key={tool.name}
            onClick={() => onToolSelect?.(tool.name)}
            className="p-4 text-left bg-slate-50 dark:bg-slate-800 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
          >
            <div className="font-semibold text-sm mb-1">{tool.name}</div>
            <div className="text-xs text-slate-600 dark:text-slate-400">
              {tool.description}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}


