'use client'

import { useState } from 'react'
import DataTable, { type Column } from './DataTable'
import JSONViewer from './JSONViewer'

interface MCPTool {
  name: string
  description: string
  category: string
  inputSchema: Record<string, any>
}

export default function MCPToolBrowser({ tools }: { tools: MCPTool[] }) {
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null)
  
  const columns: Column<MCPTool>[] = [
    { key: 'name', label: 'Tool Name', sortable: true },
    { key: 'description', label: 'Description', sortable: true },
    { key: 'category', label: 'Category', sortable: true },
  ]
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
      <div>
        <DataTable
          data={tools}
          columns={columns}
          onRowClick={setSelectedTool}
          searchable
        />
      </div>
      <div className="card">
        {selectedTool ? (
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-semibold mb-2">{selectedTool.name}</h3>
              <p className="text-slate-600 dark:text-slate-400">{selectedTool.description}</p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Input Schema</h4>
              <JSONViewer data={selectedTool.inputSchema} />
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-slate-500">
            Select a tool to view details
          </div>
        )}
      </div>
    </div>
  )
}

