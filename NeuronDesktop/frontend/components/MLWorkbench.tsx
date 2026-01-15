'use client'

import { useState } from 'react'
import DataTable, { type Column } from './DataTable'

interface MLAlgorithm {
  name: string
  category: string
  description: string
  gpu: boolean
}

const algorithms: MLAlgorithm[] = [
  { name: 'Random Forest', category: 'classification', description: 'Ensemble learning', gpu: true },
  { name: 'Gradient Boosting', category: 'regression', description: 'XGBoost, LightGBM, CatBoost', gpu: true },
  { name: 'K-Means', category: 'clustering', description: 'K-means clustering', gpu: true },
  { name: 'DBSCAN', category: 'clustering', description: 'Density-based clustering', gpu: false },
  { name: 'PCA', category: 'dimensionality', description: 'Principal Component Analysis', gpu: true },
  { name: 'SVM', category: 'classification', description: 'Support Vector Machine', gpu: true },
]

export default function MLWorkbench() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  
  const categories = ['all', 'classification', 'regression', 'clustering', 'dimensionality', 'outlier']
  
  const filtered = selectedCategory === 'all'
    ? algorithms
    : algorithms.filter(alg => alg.category === selectedCategory)
  
  const columns: Column<MLAlgorithm>[] = [
    { key: 'name', label: 'Algorithm', sortable: true },
    { key: 'category', label: 'Category', sortable: true },
    { key: 'description', label: 'Description', sortable: true },
    {
      key: 'gpu',
      label: 'GPU',
      render: (value) => (
        <span className={value ? 'text-green-600' : 'text-slate-400'}>
          {value ? '✓' : '✗'}
        </span>
      ),
    },
  ]
  
  return (
    <div className="space-y-4">
      <div className="flex gap-2 flex-wrap">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`
              px-4 py-2 rounded-lg text-sm font-medium
              ${
                selectedCategory === cat
                  ? 'bg-purple-600 text-white'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
              }
            `}
          >
            {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </button>
        ))}
      </div>
      <DataTable data={filtered} columns={columns} searchable />
    </div>
  )
}

