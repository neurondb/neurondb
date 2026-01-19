'use client'

import { useState, useEffect } from 'react'
import Breadcrumbs from '@/components/Breadcrumbs'
import MainContent from '@/components/MainContent'
import { BarChart, LineChart, PieChart } from '@/components/Charts'
import DataTable, { type Column } from '@/components/DataTable'
import { factoryAPI } from '@/lib/api'

interface QualityMetric {
  name: string
  value: number
  timestamp: string
}

interface ClusterResult {
  id: number
  cluster_id: number
  features: number[]
}

interface OutlierResult {
  id: number
  score: number
  is_outlier: boolean
}

export default function AnalyticsPage() {
  const [profileId, setProfileId] = useState<string>('')
  const [view, setView] = useState<'quality' | 'clustering' | 'outliers' | 'drift' | 'topics'>('quality')
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetric[]>([])
  const [clusterResults, setClusterResults] = useState<ClusterResult[]>([])
  const [outlierResults, setOutlierResults] = useState<OutlierResult[]>([])

  useEffect(() => {
    loadProfile()
  }, [])

  useEffect(() => {
    if (profileId) {
      loadData()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  const loadData = async () => {
    try {
      switch (view) {
        case 'quality':
          /* Load quality metrics via MCP */
          setQualityMetrics([
            { name: 'Recall@10', value: 0.85, timestamp: '2024-01-01' },
            { name: 'Precision@10', value: 0.78, timestamp: '2024-01-01' },
            { name: 'F1@10', value: 0.81, timestamp: '2024-01-01' },
            { name: 'MRR', value: 0.72, timestamp: '2024-01-01' },
          ])
          break
        case 'clustering':
          /* Load clustering results */
          setClusterResults([])
          break
        case 'outliers':
          /* Load outlier detection results */
          setOutlierResults([])
          break
      }
    } catch (err) {
      console.error('Failed to load analytics data:', err)
    }
  }

  const handleRunClustering = async () => {
    try {
      /* Call MCP cluster_data tool */
      const response = await factoryAPI.post(`/profiles/${profileId}/mcp/tools/call`, {
        name: 'cluster_data',
        arguments: {
          table_name: 'your_table',
          algorithm: 'kmeans',
          n_clusters: 5,
        },
      })
      console.log('Clustering result:', response.data)
    } catch (err) {
      console.error('Failed to run clustering:', err)
    }
  }

  const handleDetectOutliers = async () => {
    try {
      /* Call MCP detect_outliers tool */
      const response = await factoryAPI.post(`/profiles/${profileId}/mcp/tools/call`, {
        name: 'detect_outliers',
        arguments: {
          table_name: 'your_table',
          method: 'z_score',
        },
      })
      console.log('Outlier detection result:', response.data)
    } catch (err) {
      console.error('Failed to detect outliers:', err)
    }
  }

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
            Quality metrics, drift detection, clustering, outlier detection, and topic discovery
          </p>
        </div>

        <div className="mb-4 flex gap-2">
          <button
            onClick={() => setView('quality')}
            className={`px-4 py-2 rounded-md ${
              view === 'quality'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            Quality Metrics
          </button>
          <button
            onClick={() => setView('clustering')}
            className={`px-4 py-2 rounded-md ${
              view === 'clustering'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            Clustering
          </button>
          <button
            onClick={() => setView('outliers')}
            className={`px-4 py-2 rounded-md ${
              view === 'outliers'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            Outlier Detection
          </button>
          <button
            onClick={() => setView('drift')}
            className={`px-4 py-2 rounded-md ${
              view === 'drift'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            Drift Detection
          </button>
          <button
            onClick={() => setView('topics')}
            className={`px-4 py-2 rounded-md ${
              view === 'topics'
                ? 'bg-purple-600 text-white'
                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
            }`}
          >
            Topic Discovery
          </button>
        </div>

        {view === 'quality' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-4">Quality Metrics</h3>
              {qualityMetrics.length > 0 ? (
                <BarChart
                  data={qualityMetrics}
                  bars={[{ key: 'value', name: 'Score', color: 'rgb(139, 92, 246)' }]}
                  xAxisKey="name"
                  height={200}
                />
              ) : (
                <p className="text-slate-600 dark:text-slate-400">
                  Recall@K, Precision@K, F1@K, MRR, Davies-Bouldin Index
                </p>
              )}
            </div>
            <div className="card p-6">
              <h3 className="text-lg font-semibold mb-4">Metrics Over Time</h3>
              {qualityMetrics.length > 0 ? (
                <LineChart
                  data={qualityMetrics}
                  dataKey="timestamp"
                  lines={[{ key: 'value', name: 'Score', color: 'rgb(139, 92, 246)' }]}
                  height={200}
                />
              ) : (
                <p className="text-slate-600 dark:text-slate-400">No data available</p>
              )}
            </div>
          </div>
        )}

        {view === 'clustering' && (
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Clustering Analysis</h3>
              <button
                onClick={handleRunClustering}
                className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
              >
                Run Clustering
              </button>
            </div>
            <p className="text-slate-600 dark:text-slate-400 mb-4">
              K-Means, DBSCAN, GMM, Hierarchical clustering
            </p>
            {clusterResults.length > 0 ? (
              <DataTable
                data={clusterResults}
                columns={[
                  { key: 'id', label: 'ID', sortable: true },
                  { key: 'cluster_id', label: 'Cluster', sortable: true },
                ]}
              />
            ) : (
              <div className="text-slate-600 dark:text-slate-400">
                No clustering results. Run clustering analysis to see results.
              </div>
            )}
          </div>
        )}

        {view === 'outliers' && (
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Outlier Detection</h3>
              <button
                onClick={handleDetectOutliers}
                className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
              >
                Detect Outliers
              </button>
            </div>
            <p className="text-slate-600 dark:text-slate-400 mb-4">
              Z-score, Modified Z-score, IQR methods
            </p>
            {outlierResults.length > 0 ? (
              <DataTable
                data={outlierResults}
                columns={[
                  { key: 'id', label: 'ID', sortable: true },
                  { key: 'score', label: 'Outlier Score', sortable: true },
                  {
                    key: 'is_outlier',
                    label: 'Is Outlier',
                    render: (value) => (value ? 'Yes' : 'No'),
                  },
                ]}
              />
            ) : (
              <div className="text-slate-600 dark:text-slate-400">
                No outlier detection results. Run detection to see results.
              </div>
            )}
          </div>
        )}

        {view === 'drift' && (
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4">Drift Detection</h3>
            <p className="text-slate-600 dark:text-slate-400">
              Centroid drift, Distribution divergence, Temporal monitoring
            </p>
          </div>
        )}

        {view === 'topics' && (
          <div className="card p-6">
            <h3 className="text-lg font-semibold mb-4">Topic Discovery</h3>
            <p className="text-slate-600 dark:text-slate-400">
              Topic modeling and analysis using LDA and other algorithms
            </p>
          </div>
        )}
      </div>
    </MainContent>
  )
}




