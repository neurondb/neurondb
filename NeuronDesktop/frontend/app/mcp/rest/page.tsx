'use client'

import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import MCPRESTExplorer from '@/components/MCPRESTExplorer'

export default function MCPRESTPage() {
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'MCP Console', href: '/mcp' },
            { label: 'REST API Explorer' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            REST API Explorer
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Test MCP tools via REST API
          </p>
        </div>
        <MCPRESTExplorer />
      </div>
    </MainContent>
  )
}


