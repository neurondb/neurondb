'use client'

import { useState, useEffect, useCallback } from 'react'
import { profilesAPI, neurondbAPI, type Profile, type CollectionInfo, type SearchRequest, type SearchResult } from '@/lib/api'
import JSONViewer from '@/components/JSONViewer'
import SQLEditor from '@/components/SQLEditor'
import { 
  MagnifyingGlassIcon,
  TableCellsIcon,
  ChartBarIcon,
  ArrowPathIcon,
  CodeBracketIcon,
  DocumentTextIcon
} from '@/components/Icons'
import Footer from '@/components/Footer'

type TabType = 'search' | 'sql' | 'collections'

export default function NeuronDBPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [collections, setCollections] = useState<CollectionInfo[]>([])
  const [selectedCollection, setSelectedCollection] = useState<CollectionInfo | null>(null)
  const [activeTab, setActiveTab] = useState<TabType>('search')
  
  // Vector Search state
  const [queryText, setQueryText] = useState('')
  const [limit, setLimit] = useState(10)
  const [distanceType, setDistanceType] = useState<'l2' | 'cosine' | 'inner_product'>('cosine')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingCollections, setLoadingCollections] = useState(false)
  
  // SQL Editor state
  const [sqlQuery, setSqlQuery] = useState('')
  const [sqlResults, setSqlResults] = useState<any>(null)
  const [sqlError, setSqlError] = useState<string>('')
  const [sqlLoading, setSqlLoading] = useState(false)
  const [sqlHistory, setSqlHistory] = useState<string[]>([])

  const loadProfiles = useCallback(async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0 && !selectedProfile) {
        const activeProfileId = localStorage.getItem('active_profile_id')
        if (activeProfileId) {
          const activeProfile = response.data.find((p: Profile) => p.id === activeProfileId)
          if (activeProfile) {
            setSelectedProfile(activeProfileId)
            return
          }
        }
        const defaultProfile = response.data.find((p: Profile) => p.is_default)
        setSelectedProfile(defaultProfile ? defaultProfile.id : response.data[0].id)
      }
    } catch (error) {
      console.error('Failed to load profiles:', error)
    }
  }, [selectedProfile])

  const loadCollections = useCallback(async () => {
    if (!selectedProfile) return
    setLoadingCollections(true)
    try {
      const response = await neurondbAPI.listCollections(selectedProfile)
      setCollections(response.data)
      if (response.data.length > 0 && !selectedCollection) {
        setSelectedCollection(response.data[0])
      }
    } catch (error) {
      console.error('Failed to load collections:', error)
    } finally {
      setLoadingCollections(false)
    }
  }, [selectedProfile, selectedCollection])

  useEffect(() => {
    loadProfiles()
  }, [loadProfiles])

  useEffect(() => {
    if (selectedProfile) {
      loadCollections()
    }
  }, [selectedProfile, loadCollections])

  const handleSearch = async () => {
    if (!selectedProfile || !selectedCollection || !queryText) return
    
    setLoading(true)
    try {
      const request: SearchRequest = {
        collection: selectedCollection.name,
        schema: selectedCollection.schema,
        query_text: queryText,
        limit,
        distance_type: distanceType,
        query_vector: [],
      }
      
      const response = await neurondbAPI.search(selectedProfile, request)
      setResults(response.data)
    } catch (error: any) {
      console.error('Search failed:', error)
      console.error(error.response?.data?.error || error.message || 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const handleSQLExecute = async () => {
    if (!selectedProfile || !sqlQuery.trim()) {
      console.warn('Cannot execute SQL: missing profile or query', { selectedProfile, hasQuery: !!sqlQuery.trim() })
      return
    }
    
    setSqlLoading(true)
    setSqlError('')
    setSqlResults(null)
    
    try {
      console.log('Executing SQL query:', { profileId: selectedProfile, query: sqlQuery.trim() })
      const response = await neurondbAPI.executeSQLFull(selectedProfile, sqlQuery.trim())
      console.log('SQL execution response:', response)
      console.log('SQL execution data:', response.data)
      
      // Handle the response properly
      let resultData = response.data
      
      // If data is null or undefined, check if response itself has the data
      if (resultData === null || resultData === undefined) {
        // For empty SELECT results, show empty array
        if (sqlQuery.trim().toUpperCase().startsWith('SELECT')) {
          resultData = []
        } else {
          throw new Error('No data returned from server')
        }
      }
      
      setSqlResults(resultData)
      
      // Add to history (keep last 20)
      setSqlHistory(prev => {
        const newHistory = [sqlQuery.trim(), ...prev.filter(q => q !== sqlQuery.trim())]
        return newHistory.slice(0, 20)
      })
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || error.response?.data?.error || error.message || 'SQL execution failed'
      setSqlError(errorMessage)
      console.error('SQL execution failed:', error)
      console.error('Error response:', error.response?.data)
      console.error('Error details:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        message: error.message,
      })
    } finally {
      setSqlLoading(false)
    }
  }

  const handleSQLHistorySelect = (query: string) => {
    setSqlQuery(query)
  }

  return (
    // Page background should inherit from the global app background (set in layout/AuthGuard/MainContent)
    <div className="h-full flex flex-col bg-transparent">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500/10 to-indigo-500/10 dark:from-blue-500/20 dark:to-indigo-500/20 border-b border-slate-200 dark:border-slate-700 px-8 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-100">NeuronDB Console</h1>
            <p className="text-base text-gray-700 dark:text-slate-400 mt-2">Manage collections, search vectors, and execute SQL queries</p>
          </div>
          <div className="text-sm text-gray-700 dark:text-slate-300">
            Profile: <span className="text-gray-900 dark:text-slate-100 font-medium">{profiles.find((p) => p.id === selectedProfile)?.name || '...'}</span>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Collections Selector */}
        <div className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm border-b border-slate-200 dark:border-slate-700 px-8 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-100">Collections</h2>
                <p className="text-base text-gray-700 dark:text-slate-400 mt-1">
                  {loadingCollections ? (
                    <span className="flex items-center gap-2">
                      <span className="animate-spin">⏳</span> Loading...
                    </span>
                  ) : collections.length === 0 ? (
                    <span className="text-yellow-600 dark:text-yellow-500">No collections found</span>
                  ) : (
                    `${collections.length} collection${collections.length !== 1 ? 's' : ''}`
                  )}
                </p>
              </div>
              {collections.length > 0 ? (
                <select
                  value={selectedCollection?.name || ''}
                  onChange={(e) => {
                    const collection = collections.find(c => c.name === e.target.value)
                    setSelectedCollection(collection || null)
                  }}
                  className="input text-base px-4 py-3 min-w-[300px]"
                >
                  <option value="">Select a collection</option>
                  {collections.map((collection) => (
                    <option key={collection.name} value={collection.name}>
                      {collection.name} ({collection.schema}) {collection.row_count !== undefined ? `- ${collection.row_count.toLocaleString()} rows` : ''}
                    </option>
                  ))}
                </select>
              ) : (
                <div className="px-4 py-3 min-w-[300px] bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-gray-700 dark:text-slate-400 text-base space-y-2">
                  {loadingCollections ? (
                    'Loading collections...'
                  ) : (
                    <>
                      <p className="font-semibold text-gray-800 dark:text-slate-200 mb-2">No collections found</p>
                      <p className="text-sm text-gray-700 dark:text-slate-400 mb-3">
                        Collections are PostgreSQL tables with vector columns. Create a table with a vector column to get started:
                      </p>
                      <pre className="text-xs bg-slate-200 dark:bg-slate-800 p-3 rounded border border-slate-300 dark:border-slate-600 text-gray-800 dark:text-slate-300 overflow-x-auto">
{`CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(384)
);`}
                      </pre>
                      <p className="text-xs text-gray-700 dark:text-slate-400 mt-2">
                        <strong>Note:</strong> NeuronDB is a PostgreSQL extension. You&apos;re connected to PostgreSQL, but you need tables with <code className="bg-slate-200 dark:bg-slate-800 px-1 rounded">vector</code> type columns for vector search.
                      </p>
                    </>
                  )}
                </div>
              )}
            </div>
            <button
              onClick={loadCollections}
              disabled={loadingCollections}
              className="p-3 text-gray-700 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-700 rounded-lg transition-all disabled:opacity-50"
              title="Refresh collections"
            >
              <ArrowPathIcon className={`w-6 h-6 ${loadingCollections ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm border-b border-slate-200 dark:border-slate-700 px-8">
          <div className="flex gap-1">
            <button
              onClick={() => setActiveTab('search')}
              className={`px-6 py-3 font-medium text-sm transition-all border-b-2 ${
                activeTab === 'search'
                  ? 'border-blue-600 text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30'
                  : 'border-transparent text-gray-700 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700/50'
              }`}
            >
              <div className="flex items-center gap-2">
                <MagnifyingGlassIcon className="w-5 h-5" />
                <span>Vector Search</span>
              </div>
            </button>
            <button
              onClick={() => setActiveTab('sql')}
              className={`px-6 py-3 font-medium text-sm transition-all border-b-2 ${
                activeTab === 'sql'
                  ? 'border-blue-600 text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30'
                  : 'border-transparent text-gray-700 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700/50'
              }`}
            >
              <div className="flex items-center gap-2">
                <CodeBracketIcon className="w-5 h-5" />
                <span>SQL Editor</span>
              </div>
            </button>
            <button
              onClick={() => setActiveTab('collections')}
              className={`px-6 py-3 font-medium text-sm transition-all border-b-2 ${
                activeTab === 'collections'
                  ? 'border-blue-600 text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30'
                  : 'border-transparent text-gray-700 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700/50'
              }`}
            >
              <div className="flex items-center gap-2">
                <TableCellsIcon className="w-5 h-5" />
                <span>Collections</span>
              </div>
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden bg-transparent">
          {/* Vector Search Tab */}
          {activeTab === 'search' && (
            <>
              <div className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm border-b border-slate-200 dark:border-slate-700 p-8">
                <div className="max-w-5xl mx-auto">
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100 mb-6">Vector Search</h2>
                  <div className="space-y-6">
                    <div>
                      <label className="block text-base font-semibold text-gray-800 dark:text-slate-200 mb-3">
                        Query Text *
                      </label>
                      <textarea
                        value={queryText}
                        onChange={(e) => setQueryText(e.target.value)}
                        className="input w-full text-base px-4 py-4 resize-y min-h-[180px] font-mono"
                        rows={8}
                        placeholder="Enter your search query here...&#10;You can enter multiple lines of text for semantic search."
                        style={{ minHeight: '180px', lineHeight: '1.6' }}
                      />
                      <p className="text-sm text-gray-700 dark:text-slate-400 mt-2">
                        Enter natural language text to search for similar content in the collection
                      </p>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-base font-semibold text-gray-800 dark:text-slate-200 mb-3">
                          Result Limit: <span className="text-blue-600 dark:text-blue-400">{limit}</span>
                        </label>
                        <div className="flex items-center gap-4">
                          <input
                            type="range"
                            min="1"
                            max="100"
                            value={limit}
                            onChange={(e) => setLimit(parseInt(e.target.value))}
                            className="flex-1 h-2 bg-slate-300 dark:bg-slate-600 rounded-lg appearance-none cursor-pointer"
                          />
                          <input
                            type="number"
                            min="1"
                            max="100"
                            value={limit}
                            onChange={(e) => {
                              const val = parseInt(e.target.value) || 1
                              setLimit(Math.min(Math.max(val, 1), 100))
                            }}
                            className="w-20 input text-center"
                          />
                        </div>
                        <p className="text-xs text-gray-700 dark:text-slate-400 mt-2">Number of results to return (1-100)</p>
                      </div>
                      
                      <div>
                        <label className="block text-base font-semibold text-gray-800 dark:text-slate-200 mb-3">
                          Distance Metric
                        </label>
                        <select
                          value={distanceType}
                          onChange={(e) => setDistanceType(e.target.value as any)}
                          className="input w-full text-base px-4 py-3"
                        >
                          <option value="cosine">Cosine Similarity</option>
                          <option value="l2">L2 (Euclidean Distance)</option>
                          <option value="inner_product">Inner Product</option>
                        </select>
                        <p className="text-xs text-gray-700 dark:text-slate-400 mt-2">How to measure similarity between vectors</p>
                      </div>
                    </div>
                    
                    <button
                      onClick={handleSearch}
                      disabled={loading || !queryText || !selectedCollection}
                      className="px-8 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white rounded-xl font-semibold text-base transition-all shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/60 flex items-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <MagnifyingGlassIcon className="w-5 h-5" />
                      {loading ? 'Searching...' : 'Search'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Search Results */}
              <div className="flex-1 overflow-y-auto">
                <div className="max-w-6xl mx-auto px-4 py-6">
                  {results.length === 0 && !loading && (
                    <div className="flex flex-col items-center justify-center min-h-[60vh]">
                      <div className="w-20 h-20 bg-slate-100 dark:bg-slate-700 rounded-full flex items-center justify-center mb-6">
                        <MagnifyingGlassIcon className="w-10 h-10 text-gray-700 dark:text-slate-400" />
                      </div>
                      <h2 className="text-2xl font-semibold text-gray-900 dark:text-slate-100 mb-2">No results yet</h2>
                      <p className="text-gray-700 dark:text-slate-400 mb-6">Enter a query above and click Search to find similar content</p>
                    </div>
                  )}
                  
                  {loading && (
                    <div className="flex flex-col items-center justify-center min-h-[60vh]">
                      <div className="w-16 h-16 border-4 border-blue-600 dark:border-blue-400 border-t-transparent rounded-full mb-6 animate-spin"></div>
                      <p className="text-lg text-gray-700 dark:text-slate-300">Searching...</p>
                      <p className="text-base text-gray-700 dark:text-slate-400 mt-2">Finding similar content in the collection</p>
                    </div>
                  )}
                  
                  {results.length > 0 && (
                    <>
                      <div className="mb-6">
                        <h3 className="text-xl font-bold text-gray-900 dark:text-slate-100 mb-1">Search Results</h3>
                        <p className="text-base text-gray-700 dark:text-slate-400">
                          Found {results.length} result{results.length !== 1 ? 's' : ''}
                        </p>
                      </div>
                      
                      <div className="space-y-6">
                        {results.map((result, idx) => (
                          <div key={idx} className="bg-white/70 dark:bg-slate-800/70 backdrop-blur-sm border border-slate-200 dark:border-slate-700 rounded-xl p-6 hover:border-slate-300 dark:hover:border-slate-600 transition-all shadow-lg">
                            <div className="flex items-start justify-between mb-4">
                              <div className="flex-1">
                                <div className="flex items-center gap-3 mb-2">
                                  <span className="text-lg font-bold text-blue-600 dark:text-blue-400">#{idx + 1}</span>
                                  <span className="text-base text-gray-700 dark:text-slate-400">ID: <code className="text-gray-800 dark:text-slate-300">{String(result.id)}</code></span>
                                </div>
                              </div>
                              <div className="flex items-center gap-6">
                                <div className="text-right bg-green-50 dark:bg-green-900/30 px-4 py-2 rounded-lg border border-green-200 dark:border-green-700">
                                  <div className="text-xs text-gray-700 dark:text-slate-400 uppercase tracking-wider mb-1">Score</div>
                                  <div className="text-xl font-bold text-green-600 dark:text-green-400">{result.score.toFixed(4)}</div>
                                </div>
                                <div className="text-right bg-blue-50 dark:bg-blue-900/30 px-4 py-2 rounded-lg border border-blue-200 dark:border-blue-700">
                                  <div className="text-xs text-gray-700 dark:text-slate-400 uppercase tracking-wider mb-1">Distance</div>
                                  <div className="text-xl font-bold text-blue-600 dark:text-blue-400">{result.distance.toFixed(4)}</div>
                                </div>
                              </div>
                            </div>
                            <div className="mt-4">
                              <JSONViewer data={result.data} title="Data" defaultExpanded={false} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              </div>
            </>
          )}

          {/* SQL Editor Tab */}
          {activeTab === 'sql' && (
            <div className="flex-1 flex flex-col overflow-hidden">
              <div className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm border-b border-slate-200 dark:border-slate-700 p-8">
                <div className="max-w-6xl mx-auto">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100">SQL Query Editor</h2>
                    <div className="text-sm text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/30 px-4 py-2 rounded-lg border border-amber-200 dark:border-amber-700">
                      ⚠️ Full database access - Use with caution
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-base font-semibold text-gray-800 dark:text-slate-200 mb-3">
                        SQL Query
                      </label>
                      <SQLEditor
                        value={sqlQuery}
                        onChange={setSqlQuery}
                        placeholder={`-- Enter any SQL query here
-- Examples:

-- Create a table with vector column:
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(384)
);

-- Insert data:
INSERT INTO documents (content, embedding) 
VALUES ('Hello world', '[0.1, 0.2, 0.3]'::vector);

-- Query data:
SELECT * FROM documents LIMIT 10;

-- Update data:
UPDATE documents SET content = 'Updated' WHERE id = 1;

-- Delete data:
DELETE FROM documents WHERE id = 1;`}
                        minHeight="300px"
                        className="w-full"
                      />
                    </div>
                    
                    {sqlHistory.length > 0 && (
                      <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">Query History</label>
                        <div className="flex flex-wrap gap-2">
                          {sqlHistory.slice(0, 10).map((query, idx) => (
                            <button
                              key={idx}
                              onClick={() => handleSQLHistorySelect(query)}
                              className="px-3 py-1 text-xs bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-gray-700 dark:text-slate-300 rounded border border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500 transition-all max-w-xs truncate"
                              title={query}
                            >
                              {query.substring(0, 50)}{query.length > 50 ? '...' : ''}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    <div className="flex items-center gap-4">
                      <button
                        onClick={handleSQLExecute}
                        disabled={sqlLoading || !sqlQuery.trim()}
                        className="px-8 py-3 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white rounded-xl font-semibold text-base transition-all shadow-lg shadow-green-500/30 hover:shadow-xl hover:shadow-green-500/60 flex items-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <CodeBracketIcon className="w-5 h-5" />
                        {sqlLoading ? 'Executing...' : 'Execute Query'}
                      </button>
                      <button
                        onClick={() => {
                          setSqlQuery('')
                          setSqlResults(null)
                          setSqlError('')
                        }}
                        className="px-6 py-3 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-gray-700 dark:text-slate-300 rounded-xl font-medium text-base transition-all border border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500"
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* SQL Results */}
              <div className="flex-1 overflow-y-auto bg-transparent">
                <div className="max-w-6xl mx-auto px-4 py-6">
                  {sqlError && (
                    <div className="mb-6 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 rounded-xl p-6">
                      <h3 className="text-lg font-bold text-red-700 dark:text-red-400 mb-2">Error</h3>
                      <pre className="text-sm text-red-800 dark:text-red-300 font-mono whitespace-pre-wrap">{sqlError}</pre>
                    </div>
                  )}
                  
                  {sqlResults && (
                    <div className="mb-6">
                      <h3 className="text-xl font-bold text-gray-900 dark:text-slate-100 mb-4">Query Results</h3>
                      {Array.isArray(sqlResults) ? (
                        <div className="bg-white/70 dark:bg-slate-800/70 backdrop-blur-sm border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
                          <div className="overflow-x-auto">
                            <table className="w-full">
                              <thead className="bg-slate-100 dark:bg-slate-700 border-b border-slate-300 dark:border-slate-600">
                                <tr>
                                  {sqlResults.length > 0 && Object.keys(sqlResults[0]).map((col) => (
                                    <th key={col} className="px-4 py-3 text-left text-sm font-semibold text-gray-700 dark:text-slate-200">
                                      {col}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {sqlResults.map((row: any, idx: number) => (
                                  <tr key={idx} className="border-b border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700/50">
                                    {Object.values(row).map((val: any, colIdx: number) => (
                                      <td key={colIdx} className="px-4 py-3 text-sm text-gray-600 dark:text-slate-300">
                                        {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          <div className="px-4 py-3 bg-slate-50 dark:bg-slate-700 border-t border-slate-200 dark:border-slate-600 text-sm text-gray-700 dark:text-slate-400">
                            {sqlResults.length} row{sqlResults.length !== 1 ? 's' : ''} returned
                          </div>
                        </div>
                      ) : (
                        <div className="bg-white/70 dark:bg-slate-800/70 backdrop-blur-sm border border-slate-200 dark:border-slate-700 rounded-xl p-6">
                          <JSONViewer data={sqlResults} title="Result" defaultExpanded={true} />
                        </div>
                      )}
                    </div>
                  )}
                  
                  {!sqlResults && !sqlError && !sqlLoading && (
                    <div className="flex flex-col items-center justify-center min-h-[60vh]">
                      <div className="w-20 h-20 bg-slate-100 dark:bg-slate-700 rounded-full flex items-center justify-center mb-6">
                        <CodeBracketIcon className="w-10 h-10 text-gray-700 dark:text-slate-400" />
                      </div>
                      <h2 className="text-2xl font-semibold text-gray-900 dark:text-slate-100 mb-2">No query executed</h2>
                      <p className="text-gray-700 dark:text-slate-400 mb-6">Enter a SQL query above and click Execute Query</p>
                    </div>
                  )}
                  
                  {sqlLoading && (
                    <div className="flex flex-col items-center justify-center min-h-[60vh]">
                      <div className="w-16 h-16 border-4 border-green-600 dark:border-green-400 border-t-transparent rounded-full mb-6 animate-spin"></div>
                      <p className="text-lg text-gray-700 dark:text-slate-300">Executing query...</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Collections Tab */}
          {activeTab === 'collections' && (
            <div className="flex-1 overflow-y-auto">
              <div className="max-w-6xl mx-auto px-4 py-6">
                {collections.length === 0 && !loadingCollections ? (
                  <div className="flex flex-col items-center justify-center min-h-[60vh]">
                    <div className="w-20 h-20 bg-slate-100 dark:bg-slate-700 rounded-full flex items-center justify-center mb-6">
                      <TableCellsIcon className="w-10 h-10 text-gray-700 dark:text-slate-400" />
                    </div>
                    <h2 className="text-2xl font-semibold text-gray-900 dark:text-slate-100 mb-2">No collections found</h2>
                    <p className="text-gray-700 dark:text-slate-400 mb-6">Create tables with vector columns to see them here</p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100">Collections</h2>
                      <button
                        onClick={loadCollections}
                        disabled={loadingCollections}
                        className="px-4 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-gray-700 dark:text-slate-300 rounded-lg font-medium transition-all border border-slate-300 dark:border-slate-600 hover:border-slate-400 dark:hover:border-slate-500 disabled:opacity-50 flex items-center gap-2"
                      >
                        <ArrowPathIcon className={`w-5 h-5 ${loadingCollections ? 'animate-spin' : ''}`} />
                        Refresh
                      </button>
                    </div>
                    
                    {collections.map((collection) => (
                      <div key={collection.name} className="bg-white/70 dark:bg-slate-800/70 backdrop-blur-sm border border-slate-200 dark:border-slate-700 rounded-xl p-6 hover:border-slate-300 dark:hover:border-slate-600 transition-all shadow-lg">
                        <div className="flex items-start justify-between mb-4">
                          <div>
                            <h3 className="text-xl font-bold text-gray-900 dark:text-slate-100 mb-1">{collection.name}</h3>
                            <p className="text-sm text-gray-700 dark:text-slate-400">Schema: <code className="text-gray-800 dark:text-slate-300">{collection.schema}</code></p>
                            {collection.vector_col && (
                              <p className="text-sm text-gray-700 dark:text-slate-400 mt-1">Vector Column: <code className="text-blue-600 dark:text-blue-400">{collection.vector_col}</code></p>
                            )}
                            {collection.row_count !== undefined && (
                              <p className="text-sm text-gray-700 dark:text-slate-400 mt-1">Rows: <span className="text-gray-800 dark:text-slate-300 font-medium">{collection.row_count.toLocaleString()}</span></p>
                            )}
                          </div>
                        </div>
                        
                        {collection.indexes && collection.indexes.length > 0 && (
                          <div className="mt-4">
                            <h4 className="text-sm font-semibold text-gray-700 dark:text-slate-300 mb-2">Indexes</h4>
                            <div className="space-y-2">
                              {collection.indexes.map((index, idx) => (
                                <div key={idx} className="bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg p-3">
                                  <div className="flex items-center justify-between mb-1">
                                    <span className="text-sm font-medium text-gray-800 dark:text-slate-200">{index.name}</span>
                                    <span className="text-xs text-gray-700 dark:text-slate-400 bg-slate-200 dark:bg-slate-600 px-2 py-1 rounded">{index.type}</span>
                                  </div>
                                  <code className="text-xs text-gray-700 dark:text-slate-300">{index.definition}</code>
                                  {index.size && (
                                    <div className="text-xs text-gray-700 dark:text-slate-400 mt-1">Size: {index.size}</div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Footer */}
      <Footer />
    </div>
  )
}
