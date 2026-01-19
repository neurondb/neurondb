'use client'

import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'
import SplitPanel from '@/components/SplitPanel'
import JSONViewer from '@/components/JSONViewer'
import { factoryAPI } from '@/lib/api'

interface MCPTool {
  name: string
  description: string
  category?: string
  inputSchema?: Record<string, any>
  parameters?: Record<string, any>
}

const CATEGORIES = [
  'all',
  'vector',
  'embedding',
  'ml',
  'analytics',
  'rag',
  'postgresql',
  'quantization',
  'hybrid_search',
  'reranking',
  'dataset',
]

export default function MCPToolsPage() {
  const [profileId, setProfileId] = useState<string>('')
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null)
  const [tools, setTools] = useState<MCPTool[]>([])
  const [filteredTools, setFilteredTools] = useState<MCPTool[]>([])
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadProfile()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (profileId) {
      loadTools()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileId])

  useEffect(() => {
    filterTools()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tools, selectedCategory, searchQuery])

  const loadProfile = async () => {
    try {
      const profilesResponse = await factoryAPI.get('/profiles')
      const profiles = profilesResponse.data.data || profilesResponse.data
      if (profiles && profiles.length > 0) {
        const activeProfile = profiles.find((p: any) => p.is_active) || profiles[0]
        setProfileId(activeProfile.id)
      }
    } catch (err) {
      console.error('Failed to load profile:', err)
    }
  }

  const loadTools = async () => {
    try {
      setLoading(true)
      const response = await factoryAPI.get(`/profiles/${profileId}/mcp/tools`)
      const data = response.data.data || response.data
      const toolsList = Array.isArray(data) ? data : data.tools || []
      
      /* Categorize tools based on name patterns */
      const categorizedTools = toolsList.map((tool: any) => {
        let category = 'other'
        const name = tool.name?.toLowerCase() || ''
        if (name.includes('vector') || name.includes('search')) {
          category = 'vector'
        } else if (name.includes('embed')) {
          category = 'embedding'
        } else if (name.includes('train') || name.includes('model') || name.includes('predict')) {
          category = 'ml'
        } else if (name.includes('cluster') || name.includes('analyze') || name.includes('outlier')) {
          category = 'analytics'
        } else if (name.includes('rag') || name.includes('retrieve') || name.includes('context')) {
          category = 'rag'
        } else if (name.includes('quantize')) {
          category = 'quantization'
        } else if (name.includes('hybrid') || name.includes('rrf')) {
          category = 'hybrid_search'
        } else if (name.includes('rerank')) {
          category = 'reranking'
        } else if (name.includes('dataset') || name.includes('load')) {
          category = 'dataset'
        } else if (name.includes('execute') || name.includes('query') || name.includes('table')) {
          category = 'postgresql'
        }
        return { ...tool, category }
      })
      
      setTools(categorizedTools)
    } catch (err) {
      console.error('Failed to load tools:', err)
      setTools([])
    } finally {
      setLoading(false)
    }
  }

  const filterTools = () => {
    let filtered = tools

    if (selectedCategory !== 'all') {
      filtered = filtered.filter((tool) => tool.category === selectedCategory)
    }

    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(
        (tool) =>
          tool.name?.toLowerCase().includes(query) ||
          tool.description?.toLowerCase().includes(query)
      )
    }

    setFilteredTools(filtered)
  }
  
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
    <div className="h-full p-4 space-y-4">
      <div>
        <h2 className="text-lg font-semibold mb-4">MCP Tools ({filteredTools.length})</h2>
        
        <div className="mb-4">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search tools..."
            className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800 mb-2"
          />
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Category</label>
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
          >
            {CATEGORIES.map((cat) => (
              <option key={cat} value={cat}>
                {cat.charAt(0).toUpperCase() + cat.slice(1).replace('_', ' ')}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div className="text-slate-600 dark:text-slate-400">Loading tools...</div>
      ) : (
        <DataTable
          data={filteredTools}
          columns={columns}
          onRowClick={setSelectedTool}
          pageSize={20}
        />
      )}
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
          {selectedTool.category && (
            <div>
              <span className="px-2 py-1 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded text-xs">
                {selectedTool.category}
              </span>
            </div>
          )}
          <div>
            <h3 className="font-semibold mb-2">Input Schema</h3>
            <JSONViewer data={selectedTool.inputSchema || selectedTool.parameters || {}} />
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




