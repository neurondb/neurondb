'use client'

import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import DataTable, { type Column } from '@/components/DataTable'
import { factoryAPI } from '@/lib/api'

interface MLAlgorithm {
  name: string
  category: 'classification' | 'regression' | 'clustering' | 'dimensionality' | 'outlier'
  description: string
  gpu: boolean
}

interface MLModel {
  id: string
  name: string
  algorithm: string
  status: string
  accuracy?: number
  created_at: string
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
  const [profileId, setProfileId] = useState<string>('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [models, setModels] = useState<MLModel[]>([])
  const [view, setView] = useState<'algorithms' | 'models' | 'train' | 'predict'>('algorithms')
  const [trainingConfig, setTrainingConfig] = useState({
    algorithm: '',
    table_name: '',
    target_column: '',
    feature_columns: [] as string[],
  })
  
  const categories = ['all', 'classification', 'regression', 'clustering', 'dimensionality', 'outlier']
  
  const filteredAlgorithms = selectedCategory === 'all'
    ? mlAlgorithms
    : mlAlgorithms.filter(alg => alg.category === selectedCategory)

  useEffect(() => {
    loadProfile()
  }, [])

  useEffect(() => {
    if (profileId && view === 'models') {
      loadModels()
    }
  }, [profileId, view])

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

  const loadModels = async () => {
    try {
      /* Load models via MCP or direct SQL */
      /* For now, placeholder */
      setModels([])
    } catch (err) {
      console.error('Failed to load models:', err)
    }
  }

  const handleTrain = async () => {
    try {
      /* Call MCP train_model tool */
      await factoryAPI.post(`/profiles/${profileId}/mcp/tools/call`, {
        name: 'train_model',
        arguments: {
          algorithm: trainingConfig.algorithm,
          table_name: trainingConfig.table_name,
          target_column: trainingConfig.target_column,
          feature_columns: trainingConfig.feature_columns,
        },
      })
      setView('models')
      loadModels()
    } catch (err) {
      console.error('Failed to train model:', err)
    }
  }
  
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
        
        <div className="mb-4 flex gap-2">
          <button
            onClick={() => setView('algorithms')}
            className={`px-4 py-2 rounded-md ${
              view === 'algorithms'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            Algorithms
          </button>
          <button
            onClick={() => setView('models')}
            className={`px-4 py-2 rounded-md ${
              view === 'models'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            Models
          </button>
          <button
            onClick={() => setView('train')}
            className={`px-4 py-2 rounded-md ${
              view === 'train'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            Train Model
          </button>
          <button
            onClick={() => setView('predict')}
            className={`px-4 py-2 rounded-md ${
              view === 'predict'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            Predict
          </button>
        </div>

        {view === 'algorithms' && (
          <>
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
          </>
        )}

        {view === 'models' && (
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4">Trained Models</h2>
            {models.length > 0 ? (
              <DataTable
                data={models}
                columns={[
                  { key: 'name', label: 'Name', sortable: true },
                  { key: 'algorithm', label: 'Algorithm', sortable: true },
                  { key: 'status', label: 'Status', sortable: true },
                  {
                    key: 'accuracy',
                    label: 'Accuracy',
                    sortable: true,
                    render: (value) => (value ? `${(Number(value) * 100).toFixed(2)}%` : 'N/A'),
                  },
                  { key: 'created_at', label: 'Created', sortable: true },
                ]}
              />
            ) : (
              <div className="text-center text-slate-600 dark:text-slate-400 py-8">
                No trained models yet. Train a model to get started.
              </div>
            )}
          </div>
        )}

        {view === 'train' && (
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4">Train Model</h2>
            <div className="space-y-4 max-w-2xl">
              <div>
                <label className="block text-sm font-medium mb-2">Algorithm</label>
                <select
                  value={trainingConfig.algorithm}
                  onChange={(e) =>
                    setTrainingConfig({ ...trainingConfig, algorithm: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
                >
                  <option value="">Select algorithm...</option>
                  {mlAlgorithms.map((alg) => (
                    <option key={alg.name} value={alg.name}>
                      {alg.name} - {alg.description}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Table Name</label>
                <input
                  type="text"
                  value={trainingConfig.table_name}
                  onChange={(e) =>
                    setTrainingConfig({ ...trainingConfig, table_name: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
                  placeholder="schema.table_name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Target Column</label>
                <input
                  type="text"
                  value={trainingConfig.target_column}
                  onChange={(e) =>
                    setTrainingConfig({ ...trainingConfig, target_column: e.target.value })
                  }
                  className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-800"
                  placeholder="target_column"
                />
              </div>
              <button
                onClick={handleTrain}
                disabled={!trainingConfig.algorithm || !trainingConfig.table_name}
                className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Train Model
              </button>
            </div>
          </div>
        )}

        {view === 'predict' && (
          <div className="card p-6">
            <h2 className="text-xl font-semibold mb-4">Make Predictions</h2>
            <div className="text-slate-600 dark:text-slate-400">
              Prediction interface coming soon. Use the MCP tools to make predictions.
            </div>
          </div>
        )}
      </div>
    </MainContent>
  )
}




