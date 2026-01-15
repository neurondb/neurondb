'use client'

import { useState } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'
import SplitPanel from '@/components/SplitPanel'
import JSONViewer from '@/components/JSONViewer'

interface MCPTool {
  name: string
  description: string
  category: 'vector' | 'embedding' | 'ml' | 'analytics' | 'rag' | 'postgresql'
  parameters: Record<string, any>
}

export default function MCPToolsPage() {
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null)
  const [tools] = useState<MCPTool[]>([
    { name: 'vector_search', description: 'Vector search with cosine distance', category: 'vector', parameters: {} },
    { name: 'vector_search_l2', description: 'Vector search with L2 distance', category: 'vector', parameters: {} },
    { name: 'generate_embedding', description: 'Generate embeddings', category: 'embedding', parameters: {} },
    { name: 'train_model', description: 'Train ML model', category: 'ml', parameters: {} },
    { name: 'cluster_data', description: 'Cluster data', category: 'analytics', parameters: {} },
  ])
  
  const columns: Column<MCPTool>[] = [
    { key: 'name', label: 'Tool Name', sortable: true },
    { key: 'description', label: 'Description', sortable: true },
    {
      key: 'category',
      label: 'Category',
      render: (value) => (
        <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded text-xs">
          {value}
        </span>
      ),
    },
  ]
  
  const leftPanel = (
    <div className="h-full p-4">
      <h2 className="text-lg font-semibold mb-4">MCP Tools</h2>
      <DataTable
        data={tools}
        columns={columns}
        onRowClick={setSelectedTool}
        pageSize={20}
        searchable
      />
    </div>
  )
  
  const rightPanel = (
    <div className="h-full p-4">
      {selectedTool ? (
        <div className="space-y-4">
          <div>
            <h2 className="text-xl font-bold mb-2">{selectedTool.name}</h2>
            <p className="text-slate-600 dark:text-slate-400">{selectedTool.description}</p>
          </div>
          <div>
            <h3 className="font-semibold mb-2">Parameters</h3>
            <JSONViewer data={selectedTool.parameters} />
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center h-full text-slate-500">
          Select a tool to view details
        </div>
      )}
    </div>
  )
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'MCP Console', href: '/mcp' },
            { label: 'Tools' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            MCP Tools Browser
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Browse and test all available MCP tools
          </p>
        </div>
        <div className="h-[calc(100vh-300px)]">
          <SplitPanel left={leftPanel} right={rightPanel} defaultLeftWidth={60} />
        </div>
      </div>
    </MainContent>
  )
}


