'use client'

import VectorOperations from '@/components/VectorOperations'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'

export default function VectorOperationsPage() {
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'NeuronDB', href: '/neurondb' },
            { label: 'Vector Operations' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Vector Operations
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            All 133+ vector functions with distance metrics, math operations, and utilities
          </p>
        </div>
        <VectorOperations />
      </div>
    </MainContent>
  )
}


