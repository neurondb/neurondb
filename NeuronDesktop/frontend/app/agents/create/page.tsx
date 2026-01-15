'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import AgentWizard from '@/components/AgentWizard'
import TemplateGallery from '@/components/TemplateGallery'

export default function AgentCreatePage() {
  const router = useRouter()
  const [mode, setMode] = useState<'wizard' | 'template' | null>(null)

  if (mode === 'wizard') {
    return <AgentWizard onComplete={() => router.push('/agents')} onCancel={() => setMode(null)} />
  }

  if (mode === 'template') {
    return <TemplateGallery onSelect={() => router.push('/agents')} onCancel={() => setMode(null)} />
  }

  return (
    <div className="h-full overflow-auto bg-transparent p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-100 mb-2">Create New Agent</h1>
          <p className="text-gray-700 dark:text-slate-400">Choose how you&apos;d like to create your agent</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Step-by-Step Wizard Card */}
          <div
            onClick={() => setMode('wizard')}
            className="card hover:shadow-xl transition-all cursor-pointer border-2 border-transparent hover:border-blue-500 dark:hover:border-blue-400"
          >
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <svg className="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-slate-100 mb-2">Step-by-Step Wizard</h3>
                <p className="text-gray-600 dark:text-slate-400 mb-4">
                  Guided creation process. Perfect for beginners or when you want full control over every setting.
                </p>
                <ul className="text-sm text-gray-600 dark:text-slate-400 space-y-1">
                  <li>• Configure all agent settings</li>
                  <li>• Select tools and capabilities</li>
                  <li>• Set up memory and workflows</li>
                  <li>• Test before deployment</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Template Gallery Card */}
          <div
            onClick={() => setMode('template')}
            className="card hover:shadow-xl transition-all cursor-pointer border-2 border-transparent hover:border-green-500 dark:hover:border-green-400"
          >
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center flex-shrink-0">
                <svg className="w-6 h-6 text-green-600 dark:text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-slate-100 mb-2">Template Gallery</h3>
                <p className="text-gray-600 dark:text-slate-400 mb-4">
                  Start from pre-built templates. Quickly deploy common agent types with best practices.
                </p>
                <ul className="text-sm text-gray-600 dark:text-slate-400 space-y-1">
                  <li>• Customer support agents</li>
                  <li>• Data analysis pipelines</li>
                  <li>• Research assistants</li>
                  <li>• Document Q&A systems</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Tips */}
        <div className="mt-8 card bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
          <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">💡 Quick Tips</h3>
          <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
            <li>• Use templates for common use cases - they are well-tested defaults</li>
            <li>• The wizard is great for custom agents with specific requirements</li>
            <li>• You can always customize templates after deployment</li>
            <li>• Need help? Check the documentation or examples</li>
          </ul>
        </div>
      </div>
    </div>
  )
}


