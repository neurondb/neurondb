'use client'

import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import MCPGraphQLBuilder from '@/components/MCPGraphQLBuilder'

export default function MCPGraphQLPage() {
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'MCP Console', href: '/mcp' },
            { label: 'GraphQL Query Builder' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            GraphQL Query Builder
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Flexible GraphQL queries for MCP tools
          </p>
        </div>
        <MCPGraphQLBuilder />
      </div>
    </MainContent>
  )
}

