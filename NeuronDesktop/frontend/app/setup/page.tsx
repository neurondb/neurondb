'use client'

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { factoryAPI, profilesAPI, type Profile } from '@/lib/api'
import { getErrorMessage } from '@/lib/errors'
import {
  SparklesIcon,
  CheckCircleIcon,
  ArrowRightIcon,
  DatabaseIcon,
  ServerIcon,
  WrenchScrewdriverIcon,
} from '@/components/Icons'

type Step = 'welcome' | 'postgresql' | 'profile' | 'status' | 'complete'

export default function SetupPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const isNewUser = searchParams?.get('new_user') === 'true'
  
  const [step, setStep] = useState<Step>(isNewUser ? 'welcome' : 'status')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // PostgreSQL Settings - Load from signup if available
  const [pgHost, setPgHost] = useState('localhost')
  const [pgPort, setPgPort] = useState('5432')
  const [pgDatabase, setPgDatabase] = useState('neurondb')
  const [pgUser, setPgUser] = useState('neurondb')
  const [pgPassword, setPgPassword] = useState('')

  useEffect(() => {
    // Load PostgreSQL settings from login/signup if available
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('pg_settings')
      if (stored) {
        try {
          const settings = JSON.parse(stored)
          setPgHost(settings.host || 'localhost')
          setPgPort(settings.port || '5432')
          setPgDatabase(settings.database || 'neurondb')
          setPgUser(settings.user || 'neurondb')
          setPgPassword(settings.password || '')
        } catch (e) {
          console.error('Failed to parse stored PostgreSQL settings:', e)
        }
      }
    }
  }, [])

  // Profile Settings
  const [profileName, setProfileName] = useState('Default')
  const [mcpCommand, setMcpCommand] = useState('')
  const [agentEndpoint, setAgentEndpoint] = useState('')

  // Status
  const [status, setStatus] = useState<any>(null)

  useEffect(() => {
    if (!isNewUser) {
      loadStatus()
    }
  }, [isNewUser])

  const loadStatus = async () => {
    try {
      const response = await factoryAPI.getStatus()
      setStatus(response.data)
      setStep('status')
    } catch (error: any) {
      console.error('Failed to load status:', error)
      // Don't set error state here - let user proceed
    }
  }

  const buildPostgreSQLDSN = () => {
    const encodedUser = encodeURIComponent(pgUser)
    const encodedPassword = encodeURIComponent(pgPassword)
    return `postgresql://${encodedUser}:${encodedPassword}@${pgHost}:${pgPort}/${pgDatabase}`
  }

  const handleCreateProfile = async () => {
    setLoading(true)
    setError(null)

    try {
      const profile: Partial<Profile> = {
        name: profileName,
        neurondb_dsn: buildPostgreSQLDSN(),
        mcp_config: mcpCommand ? {
          command: mcpCommand,
          args: [],
        } : undefined,
        agent_endpoint: agentEndpoint || undefined,
        is_default: true,
      }

      await profilesAPI.create(profile as Profile)
      setStep('complete')
    } catch (error: any) {
      setError(error.response?.data?.error || error.message || 'Failed to create profile')
    } finally {
      setLoading(false)
    }
  }

  const handleComplete = async () => {
    try {
      await factoryAPI.setSetupState(true)
      router.push('/')
    } catch (error: any) {
      console.error('Failed to complete setup:', error)
      // Still redirect even if setting state fails
      router.push('/')
    }
  }

  // Welcome Step (for new users)
  if (step === 'welcome') {
    return (
      <div className="min-h-screen bg-slate-50 dark:bg-gradient-to-br dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 flex items-center justify-center px-4">
        <div className="max-w-2xl w-full text-center">
          <div className="w-24 h-24 bg-gradient-to-br from-purple-500 via-purple-600 to-indigo-600 rounded-3xl flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-purple-500/30">
            <SparklesIcon className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 dark:from-purple-400 dark:to-indigo-400 bg-clip-text text-transparent mb-4">
            Welcome to NeuronDesktop
          </h1>
          <p className="text-xl text-slate-700 dark:text-slate-300 mb-8">
            Let&apos;s set up your profile and configure your connections. This will only take a few minutes.
          </p>
          <button
            onClick={() => setStep('postgresql')}
            className="px-8 py-4 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-lg font-semibold hover:from-purple-600 hover:to-indigo-700 transition-all duration-200 shadow-lg shadow-purple-500/20 inline-flex items-center gap-2"
          >
            Get Started
            <ArrowRightIcon className="w-5 h-5" />
          </button>
        </div>
      </div>
    )
  }

  // PostgreSQL Configuration Step
  if (step === 'postgresql') {
    return (
      <div className="min-h-screen bg-transparent flex items-center justify-center px-4 py-12">
        <div className="max-w-2xl w-full">
          <div className="bg-white dark:bg-slate-900 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-800 p-8">
            <div className="flex items-center gap-3 mb-6">
              <DatabaseIcon className="w-8 h-8 text-purple-600 dark:text-purple-400" />
              <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100">PostgreSQL Configuration</h2>
                <p className="text-gray-600 dark:text-slate-400 text-sm">Configure your PostgreSQL connection</p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">Host</label>
                  <input
                    type="text"
                    value={pgHost}
                    onChange={(e) => setPgHost(e.target.value)}
                    className="w-full px-4 py-3 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="localhost"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Port</label>
                  <input
                    type="text"
                    value={pgPort}
                    onChange={(e) => setPgPort(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="5432"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">Database</label>
                  <input
                    type="text"
                    value={pgDatabase}
                    onChange={(e) => setPgDatabase(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="neurondb"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">User</label>
                  <input
                    type="text"
                    value={pgUser}
                    onChange={(e) => setPgUser(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="neurondb"
                  />
                </div>

                <div className="col-span-2">
                  <label className="block text-sm font-medium text-slate-300 mb-2">Password</label>
                  <input
                    type="password"
                    value={pgPassword}
                    onChange={(e) => setPgPassword(e.target.value)}
                    className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    placeholder="Enter PostgreSQL password"
                  />
                </div>
              </div>

              <div className="mt-4 p-3 bg-slate-100 dark:bg-slate-800/50 border border-slate-300 dark:border-slate-700 rounded-lg">
                <p className="text-xs text-gray-700 dark:text-slate-400 mb-1">Connection String:</p>
                <code className="text-xs text-gray-800 dark:text-slate-300 break-all font-mono">
                  {buildPostgreSQLDSN()}
                </code>
              </div>

              {error && (
                <div className="bg-red-500/10 border border-red-500/50 text-red-400 px-4 py-3 rounded-lg text-sm">
                  {error}
                </div>
              )}

              <div className="flex gap-4 pt-4">
                <button
                  onClick={() => setStep('welcome')}
                  className="flex-1 px-4 py-3 bg-slate-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300 rounded-lg font-medium hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                >
                  Back
                </button>
                <button
                  onClick={() => setStep('profile')}
                  className="flex-1 px-4 py-3 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-lg font-medium hover:from-purple-600 hover:to-indigo-700 transition-all inline-flex items-center justify-center gap-2"
                >
                  Next
                  <ArrowRightIcon className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Profile Creation Step
  if (step === 'profile') {
    return (
      <div className="min-h-screen bg-transparent flex items-center justify-center px-4 py-12">
        <div className="max-w-2xl w-full">
          <div className="bg-white dark:bg-slate-900 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-800 p-8">
            <div className="flex items-center gap-3 mb-6">
              <ServerIcon className="w-8 h-8 text-purple-600 dark:text-purple-400" />
              <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100">Create Profile</h2>
                <p className="text-gray-600 dark:text-slate-400 text-sm">Set up your connection profile</p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">Profile Name</label>
                <input
                  type="text"
                  value={profileName}
                  onChange={(e) => setProfileName(e.target.value)}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="Default"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  MCP Command <span className="text-gray-500 dark:text-slate-500 text-xs">(Optional)</span>
                </label>
                <input
                  type="text"
                  value={mcpCommand}
                  onChange={(e) => setMcpCommand(e.target.value)}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="/path/to/neurondb-mcp"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Agent Endpoint <span className="text-gray-500 dark:text-slate-500 text-xs">(Optional)</span>
                </label>
                <input
                  type="text"
                  value={agentEndpoint}
                  onChange={(e) => setAgentEndpoint(e.target.value)}
                  className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="http://localhost:8080"
                />
              </div>

              {error && (
                <div className="bg-red-500/10 border border-red-500/50 text-red-400 px-4 py-3 rounded-lg text-sm">
                  {error}
                </div>
              )}

              <div className="flex gap-4 pt-4">
                <button
                  onClick={() => setStep('postgresql')}
                  className="flex-1 px-4 py-3 bg-slate-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300 rounded-lg font-medium hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                >
                  Back
                </button>
                <button
                  onClick={handleCreateProfile}
                  disabled={loading}
                  className="flex-1 px-4 py-3 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-lg font-medium hover:from-purple-600 hover:to-indigo-700 transition-all disabled:opacity-50 inline-flex items-center justify-center gap-2"
                >
                  {loading ? 'Creating...' : 'Create Profile'}
                  {!loading && <ArrowRightIcon className="w-5 h-5" />}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Status/Complete Step
  if (step === 'status' || step === 'complete') {
    return (
      <div className="min-h-screen bg-transparent flex items-center justify-center px-4 py-12">
        <div className="max-w-2xl w-full">
          <div className="bg-white dark:bg-slate-900 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-800 p-8 text-center">
            <div className="w-20 h-20 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center mx-auto mb-6">
              <CheckCircleIcon className="w-12 h-12 text-white" />
            </div>
            <h2 className="text-3xl font-bold text-gray-900 dark:text-slate-100 mb-4">Setup Complete!</h2>
            <p className="text-gray-700 dark:text-slate-400 mb-8">
              Your profile has been created successfully. Default models have been added automatically.
              You can configure model API keys in Settings.
            </p>
            <button
              onClick={handleComplete}
              className="px-8 py-4 bg-gradient-to-r from-purple-500 to-indigo-600 text-white rounded-lg font-semibold hover:from-purple-600 hover:to-indigo-700 transition-all duration-200 shadow-lg shadow-purple-500/20"
            >
              Go to Dashboard
            </button>
          </div>
        </div>
      </div>
    )
  }

  return null
}
