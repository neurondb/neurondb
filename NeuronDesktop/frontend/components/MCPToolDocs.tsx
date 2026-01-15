'use client'

import { ReactNode } from 'react'

interface MCPToolDocsProps {
  toolName: string
  description: string
  parameters: Record<string, any>
  examples?: Array<{ title: string; code: string }>
}

export default function MCPToolDocs({
  toolName,
  description,
  parameters,
  examples = [],
}: MCPToolDocsProps) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold mb-2">{toolName}</h3>
        <p className="text-slate-600 dark:text-slate-400">{description}</p>
      </div>
      
      <div>
        <h4 className="font-semibold mb-2">Parameters</h4>
        <div className="space-y-2">
          {Object.entries(parameters).map(([key, schema]: [string, any]) => (
            <div key={key} className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
              <div className="font-mono text-sm font-semibold">{key}</div>
              <div className="text-xs text-slate-600 dark:text-slate-400 mt-1">
                {schema.description || 'No description'}
              </div>
              {schema.type && (
                <div className="text-xs text-slate-500 mt-1">
                  Type: <code>{schema.type}</code>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
      
      {examples.length > 0 && (
        <div>
          <h4 className="font-semibold mb-2">Examples</h4>
          <div className="space-y-3">
            {examples.map((example, idx) => (
              <div key={idx}>
                <div className="text-sm font-medium mb-1">{example.title}</div>
                <pre className="code-block text-xs">{example.code}</pre>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}


