'use client'

import { useState } from 'react'
import DataTable from './DataTable'
import { type Column } from './DataTable'

interface VectorOperation {
  name: string
  description: string
  category: 'distance' | 'math' | 'statistics' | 'preprocessing' | 'access'
  syntax: string
}

const vectorOperations: VectorOperation[] = [
  // Distance Metrics
  { name: 'L2 Distance', description: 'Euclidean distance', category: 'distance', syntax: 'vector <-> vector' },
  { name: 'Cosine Distance', description: 'Cosine similarity distance', category: 'distance', syntax: 'vector <=> vector' },
  { name: 'Inner Product', description: 'Negative dot product', category: 'distance', syntax: 'vector <#> vector' },
  { name: 'L1 Distance', description: 'Manhattan distance', category: 'distance', syntax: 'vector <+> vector' },
  { name: 'Hamming Distance', description: 'Hamming distance', category: 'distance', syntax: 'vector <~> vector' },
  
  // Math Operations
  { name: 'Add', description: 'Element-wise addition', category: 'math', syntax: 'vector_add(a, b)' },
  { name: 'Subtract', description: 'Element-wise subtraction', category: 'math', syntax: 'vector_sub(a, b)' },
  { name: 'Multiply', description: 'Element-wise multiplication', category: 'math', syntax: 'vector_mul(a, b)' },
  { name: 'Hadamard', description: 'Hadamard product', category: 'math', syntax: 'vector_hadamard(a, b)' },
  
  // Statistics
  { name: 'Mean', description: 'Calculate mean', category: 'statistics', syntax: 'vector_mean(vector)' },
  { name: 'Variance', description: 'Calculate variance', category: 'statistics', syntax: 'vector_variance(vector)' },
  { name: 'StdDev', description: 'Calculate standard deviation', category: 'statistics', syntax: 'vector_stddev(vector)' },
  
  // Preprocessing
  { name: 'Standardize', description: 'Standardize vector', category: 'preprocessing', syntax: 'vector_standardize(vector)' },
  { name: 'Normalize', description: 'Min-max normalization', category: 'preprocessing', syntax: 'vector_minmax_normalize(vector)' },
  { name: 'Clip', description: 'Clip values', category: 'preprocessing', syntax: 'vector_clip(vector, min, max)' },
  
  // Access
  { name: 'Get', description: 'Get element at index', category: 'access', syntax: 'vector_get(vector, index)' },
  { name: 'Set', description: 'Set element at index', category: 'access', syntax: 'vector_set(vector, index, value)' },
  { name: 'Slice', description: 'Slice vector', category: 'access', syntax: 'vector_slice(vector, start, end)' },
]

export default function VectorOperations() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  
  const categories = ['all', 'distance', 'math', 'statistics', 'preprocessing', 'access']
  
  const filteredOps = selectedCategory === 'all'
    ? vectorOperations
    : vectorOperations.filter(op => op.category === selectedCategory)
  
  const columns: Column<VectorOperation>[] = [
    { key: 'name', label: 'Operation', sortable: true },
    { key: 'description', label: 'Description', sortable: true },
    { key: 'category', label: 'Category', sortable: true },
    {
      key: 'syntax',
      label: 'Syntax',
      render: (value) => (
        <code className="text-xs bg-slate-100 dark:bg-slate-700 px-2 py-1 rounded">
          {value}
        </code>
      ),
    },
  ]
  
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 flex-wrap">
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
        data={filteredOps}
        columns={columns}
        pageSize={20}
        searchable
        searchPlaceholder="Search vector operations..."
      />
    </div>
  )
}


