'use client'

import { useState } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'

interface MLAlgorithm {
  name: string
  category: 'classification' | 'regression' | 'clustering' | 'dimensionality' | 'outlier'
  description: string
  gpu: boolean
}

const mlAlgorithms: MLAlgorithm[] = [
  { name: 'Random Forest', category: 'classification', description: 'Ensemble learning for classification', gpu: true },
  { name: 'Gradient Boosting', category: 'regression', description: 'XGBoost, LightGBM, CatBoost', gpu: true },
  { name: 'K-Means', category: 'clustering', description: 'K-means clustering', gpu: true },
  { name: 'DBSCAN', category: 'clustering', description: 'Density-based clustering', gpu: false },
  { name: 'PCA', category: 'dimensionality', description: 'Principal Component Analysis', gpu: true },
  { name: 'SVM', category: 'classification', description: 'Support Vector Machine', gpu: true },
  { name: 'Logistic Regression', category: 'classification', description: 'Logistic regression', gpu: false },
  { name: 'Linear Regression', category: 'regression', description: 'Linear regression', gpu: false },
  { name: 'Z-Score Outlier', category: 'outlier', description: 'Z-score based outlier detection', gpu: false },
]

export default function MLWorkbenchPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  
  const categories = ['all', 'classification', 'regression', 'clustering', 'dimensionality', 'outlier']
  
  const filteredAlgorithms = selectedCategory === 'all'
    ? mlAlgorithms
    : mlAlgorithms.filter(alg => alg.category === selectedCategory)
  
  const columns: Column<MLAlgorithm>[] = [
    { key: 'name', label: 'Algorithm', sortable: true },
    { key: 'category', label: 'Category', sortable: true },
    { key: 'description', label: 'Description', sortable: true },
    {
      key: 'gpu',
      label: 'GPU Support',
      render: (value) => (
        <span className={value ? 'text-green-600 dark:text-green-400' : 'text-slate-400'}>
          {value ? '✓' : '✗'}
        </span>
      ),
    },
  ]
  
  return (
    <MainContent>
      <div className="min-h-full bg-transparent p-6">
        <Breadcrumbs
          items={[
            { label: 'NeuronDB', href: '/neurondb' },
            { label: 'ML Workbench' },
          ]}
          className="mb-6"
        />
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">
            ML Workbench
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-1">
            All ML algorithms: Random Forest, Gradient Boosting, Clustering, and more
          </p>
        </div>
        
        <div className="mb-4 flex items-center gap-2 flex-wrap">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`
                px-4 py-2 rounded-lg text-sm font-medium transition-colors
                ${
                  selectedCategory === cat
                    ? 'bg-purple-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
                }
              `}
            >
              {cat.charAt(0).toUpperCase() + cat.slice(1)}
            </button>
          ))}
        </div>
        
        <DataTable
          data={filteredAlgorithms}
          columns={columns}
          pageSize={20}
          searchable
          searchPlaceholder="Search ML algorithms..."
        />
      </div>
    </MainContent>
  )
}


