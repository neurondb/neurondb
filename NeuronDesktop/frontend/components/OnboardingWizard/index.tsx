'use client'

import { useState, useEffect } from 'react'
import { profilesAPI, agentAPI, neurondbAPI, factoryAPI } from '@/lib/api'
import { CheckIcon, XMarkIcon, ArrowRightIcon, ArrowLeftIcon } from '@/components/Icons'
import { showSuccessToast, showErrorToast } from '@/lib/errors'

interface WizardStep {
  id: number
  name: string
  description: string
}

const steps: WizardStep[] = [
  { id: 1, name: 'Database Connection', description: 'Connect to NeuronDB PostgreSQL' },
  { id: 2, name: 'MCP Configuration', description: 'Configure MCP server settings' },
  { id: 3, name: 'Agent Setup', description: 'Set up NeuronAgent connection' },
  { id: 4, name: 'Demo Dataset', description: 'Load sample data to get started' },
  { id: 5, name: 'Complete', description: 'You\'re all set!' },
]

export default function OnboardingWizard({ onComplete }: { onComplete?: () => void }) {
  const [currentStep, setCurrentStep] = useState(0)
  const [loading, setLoading] = useState(false)
  
  // Step 1: Database Connection
  const [dbHost, setDbHost] = useState('localhost')
  const [dbPort, setDbPort] = useState('5433')
  const [dbName, setDbName] = useState('neurondb')
  const [dbUser, setDbUser] = useState('neurondb')
  const [dbPassword, setDbPassword] = useState('')
  const [dbTestResult, setDbTestResult] = useState<'idle' | 'testing' | 'success' | 'error'>('idle')
  const [dbTestError, setDbTestError] = useState('')
  
  // Step 2: MCP Configuration
  const [mcpCommand, setMcpCommand] = useState('')
  const [mcpArgs, setMcpArgs] = useState<string[]>([''])
  const [mcpConfig, setMcpConfig] = useState<any>({})
  
  // Step 3: Agent Setup
  const [agentEndpoint, setAgentEndpoint] = useState('http://localhost:8080')
  const [agentApiKey, setAgentApiKey] = useState('')
  const [agentTestResult, setAgentTestResult] = useState<'idle' | 'testing' | 'success' | 'error'>('idle')
  
  // Step 4: Demo Dataset
  const [datasetLoading, setDatasetLoading] = useState(false)
  const [datasetProgress, setDatasetProgress] = useState(0)

  const handleNext = async () => {
    if (currentStep < steps.length - 1) {
      // Validate current step before proceeding
      if (currentStep === 0 && dbTestResult !== 'success') {
        showErrorToast('Please test database connection first')
        return
      }
      if (currentStep === 2 && agentTestResult !== 'success') {
        showErrorToast('Please test agent connection first')
        return
      }
      
      setCurrentStep(currentStep + 1)
    }
  }

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const testDatabaseConnection = async () => {
    setDbTestResult('testing')
    setDbTestError('')
    
    try {
      const dsn = `postgresql://${dbUser}:${dbPassword}@${dbHost}:${dbPort}/${dbName}`
      await neurondbAPI.testConnection({ dsn })
      setDbTestResult('success')
      showSuccessToast('Database connection successful!')
    } catch (error: any) {
      setDbTestResult('error')
      setDbTestError(error.response?.data?.error || error.message || 'Connection failed')
      showErrorToast('Database connection failed: ' + (error.response?.data?.error || error.message))
    }
  }

  const testAgentConnection = async () => {
    setAgentTestResult('testing')
    
    try {
      await agentAPI.health(agentEndpoint, agentApiKey)
      setAgentTestResult('success')
      showSuccessToast('Agent connection successful!')
    } catch (error: any) {
      setAgentTestResult('error')
      showErrorToast('Agent connection failed: ' + (error.response?.data?.error || error.message))
    }
  }

  const loadDemoDataset = async () => {
    setDatasetLoading(true)
    setDatasetProgress(0)
    
    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setDatasetProgress(prev => Math.min(prev + 10, 90))
      }, 500)
      
      // Load demo dataset (using default profile)
      const dsn = `postgresql://${dbUser}:${dbPassword}@${dbHost}:${dbPort}/${dbName}`
      await neurondbAPI.loadDemoDataset({ dsn })
      
      clearInterval(progressInterval)
      setDatasetProgress(100)
      showSuccessToast('Demo dataset loaded successfully!')
      
      setTimeout(() => {
        setDatasetLoading(false)
        handleNext()
      }, 1000)
    } catch (error: any) {
      setDatasetLoading(false)
      showErrorToast('Failed to load demo dataset: ' + (error.response?.data?.error || error.message))
    }
  }

  const saveProfile = async () => {
    setLoading(true)
    
    try {
      const dsn = `postgresql://${dbUser}:${dbPassword}@${dbHost}:${dbPort}/${dbName}`
      const mcpConfigObj = {
        command: mcpCommand || 'neurondb-mcp',
        args: mcpArgs.filter(a => a.trim() !== ''),
        ...mcpConfig,
      }
      
      await profilesAPI.create({
        name: 'Default Profile',
        neurondb_dsn: dsn,
        mcp_config: mcpConfigObj,
        agent_endpoint: agentEndpoint || undefined,
        agent_api_key: agentApiKey || undefined,
        is_default: true,
      })
      
      showSuccessToast('Profile created successfully!')
      if (onComplete) onComplete()
    } catch (error: any) {
      showErrorToast('Failed to save profile: ' + (error.response?.data?.error || error.message))
    } finally {
      setLoading(false)
    }
  }

  const canProceed = () => {
    switch (currentStep) {
      case 0:
        return dbTestResult === 'success'
      case 1:
        return true // MCP config is optional
      case 2:
        return agentTestResult === 'success' || !agentEndpoint
      case 3:
        return true
      default:
        return true
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-slate-900 mb-2">Welcome to NeuronDesktop</h1>
          <p className="text-slate-600">Let&apos;s get you set up in a few simple steps</p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => (
              <div key={step.id} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all ${
                      index === currentStep
                        ? 'bg-purple-600 border-purple-600 text-white'
                        : index < currentStep
                        ? 'bg-green-600 border-green-600 text-white'
                        : 'bg-white border-slate-300 text-slate-400'
                    }`}
                  >
                    {index < currentStep ? (
                      <CheckIcon className="w-6 h-6" />
                    ) : (
                      <span className="font-semibold">{step.id}</span>
                    )}
                  </div>
                  <span className={`mt-2 text-sm font-medium text-center ${
                    index === currentStep ? 'text-purple-600' : index < currentStep ? 'text-green-600' : 'text-slate-400'
                  }`}>
                    {step.name}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div className={`h-0.5 flex-1 mx-2 ${
                    index < currentStep ? 'bg-green-600' : 'bg-slate-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
          {currentStep === 0 && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 mb-2">Database Connection</h2>
                <p className="text-slate-600">Connect to your NeuronDB PostgreSQL instance</p>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Host</label>
                  <input
                    type="text"
                    value={dbHost}
                    onChange={(e) => setDbHost(e.target.value)}
                    className="input w-full"
                    placeholder="localhost"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Port</label>
                  <input
                    type="text"
                    value={dbPort}
                    onChange={(e) => setDbPort(e.target.value)}
                    className="input w-full"
                    placeholder="5433"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Database</label>
                  <input
                    type="text"
                    value={dbName}
                    onChange={(e) => setDbName(e.target.value)}
                    className="input w-full"
                    placeholder="neurondb"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">User</label>
                  <input
                    type="text"
                    value={dbUser}
                    onChange={(e) => setDbUser(e.target.value)}
                    className="input w-full"
                    placeholder="neurondb"
                  />
                </div>
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-slate-700 mb-2">Password</label>
                  <input
                    type="password"
                    value={dbPassword}
                    onChange={(e) => setDbPassword(e.target.value)}
                    className="input w-full"
                    placeholder="Enter password"
                  />
                </div>
              </div>
              
              <div className="flex items-center gap-4">
                <button
                  onClick={testDatabaseConnection}
                  disabled={dbTestResult === 'testing'}
                  className="btn btn-primary"
                >
                  {dbTestResult === 'testing' ? 'Testing...' : 'Test Connection'}
                </button>
                
                {dbTestResult === 'success' && (
                  <div className="flex items-center text-green-600">
                    <CheckIcon className="w-5 h-5 mr-2" />
                    <span>Connection successful</span>
                  </div>
                )}
                
                {dbTestResult === 'error' && (
                  <div className="flex items-center text-red-600">
                    <XMarkIcon className="w-5 h-5 mr-2" />
                    <span>{dbTestError || 'Connection failed'}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {currentStep === 1 && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 mb-2">MCP Configuration</h2>
                <p className="text-slate-600">Configure your MCP server (optional)</p>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">MCP Command</label>
                <input
                  type="text"
                  value={mcpCommand}
                  onChange={(e) => setMcpCommand(e.target.value)}
                  className="input w-full"
                  placeholder="neurondb-mcp"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Arguments (one per line)</label>
                <textarea
                  value={mcpArgs.join('\n')}
                  onChange={(e) => setMcpArgs(e.target.value.split('\n'))}
                  className="input w-full min-h-[100px]"
                  placeholder="--config\n/path/to/config.json"
                />
              </div>
            </div>
          )}

          {currentStep === 2 && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 mb-2">Agent Setup</h2>
                <p className="text-slate-600">Connect to NeuronAgent (optional)</p>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">Agent Endpoint</label>
                <input
                  type="text"
                  value={agentEndpoint}
                  onChange={(e) => setAgentEndpoint(e.target.value)}
                  className="input w-full"
                  placeholder="http://localhost:8080"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-2">API Key (optional)</label>
                <input
                  type="password"
                  value={agentApiKey}
                  onChange={(e) => setAgentApiKey(e.target.value)}
                  className="input w-full"
                  placeholder="Enter API key"
                />
              </div>
              
              <div className="flex items-center gap-4">
                <button
                  onClick={testAgentConnection}
                  disabled={agentTestResult === 'testing' || !agentEndpoint}
                  className="btn btn-primary"
                >
                  {agentTestResult === 'testing' ? 'Testing...' : 'Test Connection'}
                </button>
                
                {agentTestResult === 'success' && (
                  <div className="flex items-center text-green-600">
                    <CheckIcon className="w-5 h-5 mr-2" />
                    <span>Connection successful</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {currentStep === 3 && (
            <div className="space-y-6">
              <div>
                <h2 className="text-2xl font-bold text-slate-900 mb-2">Demo Dataset</h2>
                <p className="text-slate-600">Load sample data to explore NeuronDB features</p>
              </div>
              
              <div className="bg-slate-50 rounded-lg p-6">
                <h3 className="font-semibold text-slate-900 mb-2">What will be loaded:</h3>
                <ul className="list-disc list-inside space-y-1 text-slate-600">
                  <li>Sample documents with embeddings</li>
                  <li>Vector search examples</li>
                  <li>Pre-configured indexes</li>
                </ul>
              </div>
              
              {datasetLoading && (
                <div className="space-y-2">
                  <div className="w-full bg-slate-200 rounded-full h-2">
                    <div
                      className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${datasetProgress}%` }}
                    />
                  </div>
                  <p className="text-sm text-slate-600 text-center">{datasetProgress}% complete</p>
                </div>
              )}
              
              <button
                onClick={loadDemoDataset}
                disabled={datasetLoading}
                className="btn btn-primary w-full"
              >
                {datasetLoading ? 'Loading...' : 'Load Demo Dataset'}
              </button>
            </div>
          )}

          {currentStep === 4 && (
            <div className="text-center space-y-6">
              <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                <CheckIcon className="w-12 h-12 text-green-600" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-slate-900 mb-2">Setup Complete!</h2>
                <p className="text-slate-600">You&apos;re all set to start using NeuronDesktop</p>
              </div>
              <button
                onClick={saveProfile}
                disabled={loading}
                className="btn btn-primary"
              >
                {loading ? 'Saving...' : 'Get Started'}
              </button>
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex justify-between">
          <button
            onClick={handleBack}
            disabled={currentStep === 0}
            className="btn btn-secondary"
          >
            <ArrowLeftIcon className="w-4 h-4 mr-2" />
            Back
          </button>
          {currentStep < steps.length - 1 && (
            <button
              onClick={handleNext}
              disabled={!canProceed()}
              className="btn btn-primary"
            >
              Next
              <ArrowRightIcon className="w-4 h-4 ml-2" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

