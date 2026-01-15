'use client'

import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import { BarChart, LineChart } from '@/components/Charts'

export default function AnalyticsPage() {
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'NeuronDB', href: '/neurondb' },
            { label: 'Analytics & Quality Metrics' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            Analytics & Quality Metrics
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            Quality metrics, drift detection, and topic discovery
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">Quality Metrics</h3>
            <p className="text-slate-600 dark:text-slate-400">
              Recall@K, Precision@K, F1@K, MRR, Davies-Bouldin Index
            </p>
          </div>
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">Drift Detection</h3>
            <p className="text-slate-600 dark:text-slate-400">
              Centroid drift, Distribution divergence, Temporal monitoring
            </p>
          </div>
        </div>
      </div>
    </MainContent>
  )
}

