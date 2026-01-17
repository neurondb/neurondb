'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { setAuthToken } from '@/lib/auth'
import { DatabaseIcon } from '@/components/Icons'
import { getErrorMessage } from '@/lib/errors'
import { authAPI } from '@/lib/auth_api'
import { databaseTestAPI } from '@/lib/api'
import { logger } from '@/lib/logger'

export default function LoginPage() {
  const router = useRouter()
  const [isSignup, setIsSignup] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [testingConnection, setTestingConnection] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'success' | 'error' | 'schema_missing'>('idle')
  const [schemaMessage, setSchemaMessage] = useState('')
  
  // PostgreSQL Settings
  const [pgHost, setPgHost] = useState('localhost')
  const [pgPort, setPgPort] = useState('5432')
  const [pgDatabase, setPgDatabase] = useState('neurondb')
  const [pgUser, setPgUser] = useState('neurondb')
  const [pgPassword, setPgPassword] = useState('')

  // Load PostgreSQL settings from localStorage on mount
  useEffect(() => {
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
          logger.error('Failed to parse stored PostgreSQL settings', e)
        }
      }
    }
  }, [])

  const handleOIDCLogin = async () => {
    setError('')
    setLoading(true)

    try {
      const response = await authAPI.startOIDC()
      // Redirect to OIDC provider
      window.location.href = response.data.auth_url
    } catch (err) {
      const errorMessage = getErrorMessage(err)
      setError(errorMessage)
      setLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      // Build DSN from PostgreSQL settings for signup
      let neurondbDSN = ''
      if (isSignup) {
        if (pgPassword) {
          neurondbDSN = `postgresql://${encodeURIComponent(pgUser)}:${encodeURIComponent(pgPassword)}@${pgHost}:${pgPort}/${pgDatabase}`
        } else {
          neurondbDSN = `postgresql://${encodeURIComponent(pgUser)}@${pgHost}:${pgPort}/${pgDatabase}`
        }
      }

      // Use axios-based auth API for better error handling
      logger.debug('Attempting login', { username, isSignup })
      const response = isSignup 
        ? await authAPI.register({ username, password, neurondb_dsn: neurondbDSN })
        : await authAPI.login({ username, password })

      logger.debug('Login response received', { status: response.status, hasToken: !!response.data?.token })

      // For JWT mode, store token (for backward compatibility)
      // For cookie-based sessions, token is not needed but we store it for compatibility
      const token = response.data?.token
      if (!token) {
        logger.error('No token in response', undefined, { responseData: response.data })
        setError('Login failed: No token received from server. Check console for details.')
        setLoading(false)
        return
      }
      
      setAuthToken(token)
      logger.debug('Token stored successfully', { tokenLength: token.length })

      // Immediately verify the token works before redirecting (makes failures visible instead of "looping back" to /login)
      try {
        const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081/api/v1'
        const meResp = await fetch(`${apiBase}/auth/me`, {
          method: 'GET',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
        })

        if (!meResp.ok) {
          const text = await meResp.text()
          logger.error('Auth check failed after login', undefined, { status: meResp.status, response: text })
          setError(`Login succeeded but auth verification failed (${meResp.status}). ${text}`)
          setLoading(false)
          return
        }
      } catch (e) {
        logger.error('Auth check error after login', e)
        setError('Login succeeded but auth verification failed (network error).')
        setLoading(false)
        return
      }

      // Store PostgreSQL settings for both signup and login
      const pgSettings = {
        host: pgHost,
        port: pgPort,
        database: pgDatabase,
        user: pgUser,
        password: pgPassword,
      }
      localStorage.setItem('pg_settings', JSON.stringify(pgSettings))

      // Store active profile so pages can land directly on it (no selectors needed)
      if (response.data.profile_id) {
        localStorage.setItem('active_profile_id', response.data.profile_id)
      }

      // Verify token is stored before redirecting
      const storedToken = localStorage.getItem('neurondesk_auth_token')
      if (!storedToken) {
        logger.error('Token was not stored in localStorage')
        setError('Failed to store authentication token')
        setLoading(false)
        return
      }

      logger.info('Login successful, redirecting')
      
      // Small delay to ensure everything is saved
      await new Promise(resolve => setTimeout(resolve, 200))
      
      // Use window.location.replace to avoid adding to history
      // This ensures a clean redirect
      if (isSignup) {
        window.location.replace('/setup?new_user=true')
      } else {
        window.location.replace('/')
      }
      
      // This should never execute, but just in case
      return
    } catch (err: unknown) {
      // Extract proper error message using our error handler
      const errorObj = err as { response?: { status?: number, data?: { error?: string, message?: string } } }
      logger.error('Login error', err, { hasResponse: !!errorObj?.response })
      const errorMessage = getErrorMessage(err)
      
      // Provide more helpful error messages for common issues
      let displayMessage = errorMessage
      if (errorObj?.response?.status === 500) {
        // Check if there's a more specific error message in the response
        const responseData = errorObj.response?.data
        if (responseData?.error && responseData.error !== 'Internal Server Error') {
          displayMessage = responseData.error
        } else if (responseData?.message) {
          displayMessage = responseData.message
        } else {
          displayMessage = 'Server error occurred. Please check:\n1. The API server is running\n2. The database connection is configured correctly\n3. Check the server logs for more details'
        }
      }
      
      setError(displayMessage)
      setLoading(false)
    }
  }

  const buildPostgreSQLDSN = () => {
    const encodedUser = encodeURIComponent(pgUser)
    const encodedPassword = encodeURIComponent(pgPassword)
    return `postgresql://${encodedUser}:${encodedPassword}@${pgHost}:${pgPort}/${pgDatabase}`
  }

  const testConnection = async () => {
    setTestingConnection(true)
    // Don't clear login errors when testing connection
    // setError('')
    setConnectionStatus('idle')
    setSchemaMessage('')

    try {
      logger.debug('Testing database connection', { host: pgHost, port: pgPort, database: pgDatabase, user: pgUser })
      const response = await databaseTestAPI.test({
        host: pgHost,
        port: pgPort,
        database: pgDatabase,
        user: pgUser,
        password: pgPassword,
      })
      logger.debug('Database connection test response', { success: response.data.success, schemaExists: response.data.schema_exists })

      if (response.data.success) {
        if (response.data.schema_exists) {
          setConnectionStatus('success')
          setSchemaMessage('Connection successful and NeuronDesktop schema is configured.')
        } else {
          setConnectionStatus('schema_missing')
          const missingTables = response.data.missing_tables || []
          setSchemaMessage(
            `Connection successful, but NeuronDesktop schema is not configured. ` +
            `Missing tables: ${missingTables.join(', ')}. ` +
            `Please run neurondesktop.sql on this database to set up the schema.`
          )
        }
      } else {
        setConnectionStatus('error')
        setError(response.data.message || 'Connection test failed')
      }
    } catch (err) {
      logger.error('Database connection test error', err)
      setConnectionStatus('error')
      const errorMessage = getErrorMessage(err)
      setError(errorMessage)
    } finally {
      setTestingConnection(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-slate-50 to-slate-100 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 px-4 py-12">
      <div className="max-w-7xl w-full">
        <div className="max-w-2xl mx-auto space-y-6">
          {/* Header */}
          <div className="text-center mb-8">
          <div className="w-20 h-20 bg-gradient-to-br from-purple-500 via-purple-600 to-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-2xl shadow-purple-500/30">
            <DatabaseIcon className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent mb-2">
            NeuronDesktop
          </h1>
          <p className="text-gray-700 dark:text-slate-400">
            {isSignup ? 'Create your account' : 'Sign in to your account'}
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white dark:bg-slate-800 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-700 p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Authentication Section */}
            <div className="space-y-4">
              <div>
                <label htmlFor="username" className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                  Username
                </label>
                <input
                  id="username"
                  type="text"
                  required
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full px-4 py-3 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                  placeholder="Enter your username"
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                  placeholder="Enter your password"
                  minLength={isSignup ? 6 : undefined}
                />
                {isSignup && (
                  <p className="mt-1 text-sm text-gray-700 dark:text-slate-400">Password must be at least 6 characters</p>
                )}
              </div>
            </div>

            {/* PostgreSQL Settings - Always show for database connection */}
            <div className="border-t border-slate-200 dark:border-slate-700 pt-6 mt-6">
              <div className="flex items-center gap-2 mb-4">
                <DatabaseIcon className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                <h2 className="text-lg font-semibold text-gray-900 dark:text-slate-100">PostgreSQL Settings</h2>
              </div>
              <p className="text-sm text-gray-700 dark:text-slate-400 mb-4">
                Configure your PostgreSQL connection settings. These will be used for database connections.
              </p>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="pg-host" className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                    Host
                  </label>
                  <input
                    id="pg-host"
                    type="text"
                    value={pgHost}
                    onChange={(e) => setPgHost(e.target.value)}
                    className="w-full px-4 py-3 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                    placeholder="localhost"
                  />
                </div>

                <div>
                  <label htmlFor="pg-port" className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                    Port
                  </label>
                  <input
                    id="pg-port"
                    type="text"
                    value={pgPort}
                    onChange={(e) => setPgPort(e.target.value)}
                    className="w-full px-4 py-3 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                    placeholder="5432"
                  />
                </div>

                <div>
                  <label htmlFor="pg-database" className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                    Database
                  </label>
                  <input
                    id="pg-database"
                    type="text"
                    value={pgDatabase}
                    onChange={(e) => setPgDatabase(e.target.value)}
                    className="w-full px-4 py-3 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                    placeholder="neurondb"
                  />
                </div>

                <div>
                  <label htmlFor="pg-user" className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                    User
                  </label>
                  <input
                    id="pg-user"
                    type="text"
                    value={pgUser}
                    onChange={(e) => setPgUser(e.target.value)}
                    className="w-full px-4 py-3 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                    placeholder="neurondb"
                  />
                </div>

                <div className="col-span-2">
                  <label htmlFor="pg-password" className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                    Password
                  </label>
                  <input
                    id="pg-password"
                    type="password"
                    value={pgPassword}
                    onChange={(e) => setPgPassword(e.target.value)}
                    className="w-full px-4 py-3 bg-white dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-gray-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                    placeholder="Enter PostgreSQL password"
                  />
                </div>
              </div>

              <div className="mt-4 p-3 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg">
                <p className="text-xs text-gray-700 dark:text-slate-400 mb-1">Connection String:</p>
                <code className="text-xs text-gray-800 dark:text-slate-200 break-all font-mono">
                  {buildPostgreSQLDSN()}
                </code>
              </div>

              {/* Connection Test Button */}
              <div className="mt-4">
                <button
                  type="button"
                  onClick={testConnection}
                  disabled={testingConnection || !pgHost || !pgDatabase || !pgUser}
                  className="w-full px-4 py-3 bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg text-gray-700 dark:text-slate-300 font-medium hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {testingConnection ? (
                    <>
                      <svg className="animate-spin h-5 w-5 text-purple-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Testing Connection...
                    </>
                  ) : (
                    <>
                      <DatabaseIcon className="w-5 h-5" />
                      Test Connection
                    </>
                  )}
                </button>
              </div>

              {/* Connection Status Messages */}
              {connectionStatus === 'success' && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-start gap-3">
                    <svg className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <div>
                      <p className="text-sm font-medium text-green-700">Connection Successful</p>
                      <p className="text-xs text-green-600 mt-1">{schemaMessage}</p>
                    </div>
                  </div>
                </div>
              )}

              {connectionStatus === 'schema_missing' && (
                <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <div className="flex items-start gap-3">
                    <svg className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-yellow-700">Schema Not Configured</p>
                      <p className="text-xs text-yellow-600 mt-1">{schemaMessage}</p>
                      <div className="mt-3 p-3 bg-slate-50 dark:bg-slate-800 rounded border border-slate-300 dark:border-slate-600">
                        <p className="text-xs text-gray-700 dark:text-slate-300 mb-2">To set up the schema, run:</p>
                        <code className="text-xs text-gray-800 dark:text-slate-200 font-mono block bg-white dark:bg-slate-900 p-2 rounded border border-slate-300 dark:border-slate-700">
                          psql -h {pgHost} -p {pgPort} -U {pgUser} -d {pgDatabase} -f neurondesktop.sql
                        </code>
                        <p className="text-xs text-gray-700 dark:text-slate-400 mt-2">
                          Or download neurondesktop.sql from the NeuronDesktop API directory and run it on this database.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {connectionStatus === 'error' && error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-start gap-3">
                    <svg className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                    <div>
                      <p className="text-sm font-medium text-red-700">Connection Failed</p>
                      <p className="text-xs text-red-600 mt-1">{error}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Auth errors (e.g. wrong username/password). Note: connectionStatus is always a string ('idle'|'success'|...), so we must not gate on it being falsy. */}
            {error && connectionStatus !== 'error' && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm flex items-start gap-3">
                <svg className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <div className="flex-1">
                  <p className="font-medium">Authentication Error</p>
                  <p className="text-sm mt-1 whitespace-pre-line">{error}</p>
                  {error.includes('Server error') && (
                    <p className="text-xs mt-2 text-red-600">
                      Tip: Check the browser console (F12) for detailed error information.
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* OIDC SSO Button (Primary) */}
            <button
              type="button"
              onClick={handleOIDCLogin}
              disabled={loading}
              className="w-full bg-gradient-to-r from-purple-500 to-indigo-600 text-white py-3 rounded-lg font-medium hover:from-purple-600 hover:to-indigo-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-purple-500/20 mb-3"
            >
              {loading ? 'Please wait...' : 'Continue with SSO'}
            </button>

            {/* Local Auth Button (Secondary) */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-slate-100 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 text-gray-700 dark:text-slate-300 py-3 rounded-lg font-medium hover:bg-slate-200 dark:hover:bg-slate-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Please wait...' : (isSignup ? 'Sign Up & Continue' : 'Sign In with Password')}
            </button>

            <div className="text-center">
              <button
                type="button"
                onClick={() => {
                  setIsSignup(!isSignup)
                  setError('')
                }}
                className="text-sm text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors"
              >
                {isSignup ? 'Already have an account? Sign in' : "Don't have an account? Sign up"}
              </button>
            </div>
          </form>
        </div>
        </div>
      </div>
    </div>
  )
}
