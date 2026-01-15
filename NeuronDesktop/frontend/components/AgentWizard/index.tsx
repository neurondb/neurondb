'use client'

import { useState, useEffect, useCallback } from 'react'
import { agentAPI, profilesAPI, type Profile, type CreateAgentRequest, type Model } from '@/lib/api'
import { CheckIcon, XMarkIcon } from '@/components/Icons'
import { showSuccessToast, showErrorToast } from '@/lib/errors'

interface WizardStep {
  id: number
  name: string
  component: React.ReactNode
}

export default function AgentWizard({ onComplete, onCancel }: { onComplete?: () => void; onCancel?: () => void }) {
  const [currentStep, setCurrentStep] = useState(0)
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(false)

  // Form state
  const [formData, setFormData] = useState<CreateAgentRequest>({
    name: '',
    description: '',
    system_prompt: '',
    model_name: 'gpt-4',
    enabled_tools: [],
    config: {},
  })

  const steps = [
    { id: 1, name: 'Basic Info' },
    { id: 2, name: 'Profile' },
    { id: 3, name: 'Tools' },
    { id: 4, name: 'Memory' },
    { id: 5, name: 'Review' },
  ]

  const loadProfiles = useCallback(async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0) {
        const defaultProfile = response.data.find((p: Profile) => p.is_default) || response.data[0]
        setSelectedProfile(defaultProfile.id)
      }
    } catch (error: any) {
      showErrorToast('Failed to load profiles: ' + (error.response?.data?.error || error.message))
    }
  }, [])

  const loadModels = useCallback(async () => {
    if (!selectedProfile) return
    try {
      const response = await agentAPI.listModels(selectedProfile)
      setModels(response.data.models)
    } catch (error: any) {
      showErrorToast('Failed to load models: ' + (error.response?.data?.error || error.message))
    }
  }, [selectedProfile])

  useEffect(() => {
    loadProfiles()
  }, [loadProfiles])

  useEffect(() => {
    if (selectedProfile) {
      loadModels()
    }
  }, [selectedProfile, loadModels])

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleCreate = async () => {
    if (!selectedProfile) {
      showErrorToast('Please select a profile first')
      return
    }
    
    if (!formData.name.trim()) {
      showErrorToast('Agent name is required')
      return
    }
    
    if (!formData.model_name) {
      showErrorToast('Please select a model')
      return
    }

    setLoading(true)
    try {
      await agentAPI.createAgent(selectedProfile, formData)
      showSuccessToast('Agent created successfully')
      if (onComplete) onComplete()
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || error.response?.data?.error || error.message || 'Failed to create agent'
      showErrorToast('Failed to create agent: ' + errorMessage)
    } finally {
      setLoading(false)
    }
  }

  const canProceed = () => {
    switch (currentStep) {
      case 0:
        return formData.name.trim() !== ''
      case 1:
        return formData.model_name !== ''
      case 2:
        return true // Tools are optional
      case 3:
        return true // Memory is optional
      default:
        return true
    }
  }

  return (
    <div className="h-full overflow-auto bg-transparent p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          {onCancel && (
            <button onClick={onCancel} className="text-blue-600 dark:text-blue-400 hover:underline mb-4">
              ← Cancel
            </button>
          )}
          <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-100 mb-2">Create Agent Wizard</h1>
          <p className="text-gray-700 dark:text-slate-400">Step-by-step agent creation</p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            {steps.map((step, index) => (
              <div key={step.id} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center border-2 ${
                      index === currentStep
                        ? 'bg-blue-600 border-blue-600 text-white'
                        : index < currentStep
                        ? 'bg-green-600 border-green-600 text-white'
                        : 'bg-gray-200 border-gray-300 text-gray-600 dark:bg-slate-700 dark:border-slate-600 dark:text-slate-300'
                    }`}
                  >
                    {index < currentStep ? (
                      <CheckIcon className="w-6 h-6" />
                    ) : (
                      <span className="font-semibold">{step.id}</span>
                    )}
                  </div>
                  <span
                    className={`mt-2 text-sm font-medium ${
                      index === currentStep
                        ? 'text-blue-600 dark:text-blue-400'
                        : index < currentStep
                        ? 'text-green-600 dark:text-green-400'
                        : 'text-gray-500 dark:text-slate-400'
                    }`}
                  >
                    {step.name}
                  </span>
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={`h-0.5 flex-1 mx-2 ${
                      index < currentStep ? 'bg-green-600' : 'bg-gray-200 dark:bg-slate-700'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <div className="card min-h-[400px]">
          {currentStep === 0 && <StepBasicInfo formData={formData} setFormData={setFormData} />}
          {currentStep === 1 && (
            <StepProfile
              formData={formData}
              setFormData={setFormData}
              profiles={profiles}
              selectedProfile={selectedProfile}
              setSelectedProfile={setSelectedProfile}
              models={models}
            />
          )}
          {currentStep === 2 && <StepTools formData={formData} setFormData={setFormData} />}
          {currentStep === 3 && <StepMemory formData={formData} setFormData={setFormData} />}
          {currentStep === 4 && (
            <StepReview formData={formData} selectedProfile={selectedProfile} profiles={profiles} models={models} />
          )}
        </div>

        {/* Navigation */}
        <div className="mt-6 flex justify-between">
          <button
            onClick={handleBack}
            disabled={currentStep === 0}
            className="btn btn-secondary"
          >
            Back
          </button>
          {currentStep < steps.length - 1 ? (
            <button
              onClick={handleNext}
              disabled={!canProceed()}
              className="btn btn-primary"
            >
              Next
            </button>
          ) : (
            <button
              onClick={handleCreate}
              disabled={loading || !canProceed()}
              className="btn btn-primary"
            >
              {loading ? 'Creating...' : 'Create Agent'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function StepBasicInfo({
  formData,
  setFormData,
}: {
  formData: CreateAgentRequest
  setFormData: (data: CreateAgentRequest) => void
}) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100 mb-2">Basic Information</h2>
        <p className="text-gray-600 dark:text-slate-400">Set up the basic details for your agent</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
          Agent Name *
        </label>
        <input
          type="text"
          value={formData.name}
          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
          className="input"
          placeholder="my-agent"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">Description</label>
        <input
          type="text"
          value={formData.description || ''}
          onChange={(e) => setFormData({ ...formData, description: e.target.value })}
          className="input"
          placeholder="A helpful AI agent"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">System Prompt</label>
        <textarea
          value={formData.system_prompt || ''}
          onChange={(e) => setFormData({ ...formData, system_prompt: e.target.value })}
          className="input min-h-[120px]"
          placeholder="You are a helpful assistant..."
        />
      </div>
    </div>
  )
}

function StepProfile({
  formData,
  setFormData,
  profiles,
  selectedProfile,
  setSelectedProfile,
  models,
}: {
  formData: CreateAgentRequest
  setFormData: (data: CreateAgentRequest) => void
  profiles: Profile[]
  selectedProfile: string
  setSelectedProfile: (id: string) => void
  models: Model[]
}) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100 mb-2">Profile & Model</h2>
        <p className="text-gray-600 dark:text-slate-400">Select profile and model configuration</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">Profile</label>
        <select
          value={selectedProfile}
          onChange={(e) => setSelectedProfile(e.target.value)}
          className="input"
        >
          {profiles.map((profile) => (
            <option key={profile.id} value={profile.id}>
              {profile.name} {profile.is_default && '(Default)'}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">Model *</label>
        <select
          value={formData.model_name || ''}
          onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
          className="input"
        >
          <option value="">Select a model</option>
          {models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.display_name} ({model.provider})
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}

function StepTools({
  formData,
  setFormData,
}: {
  formData: CreateAgentRequest
  setFormData: (data: CreateAgentRequest) => void
}) {
  const tools = ['sql', 'http', 'code', 'shell', 'browser']

  const toggleTool = (tool: string) => {
    const enabled = formData.enabled_tools || []
    if (enabled.includes(tool)) {
      setFormData({ ...formData, enabled_tools: enabled.filter((t) => t !== tool) })
    } else {
      setFormData({ ...formData, enabled_tools: [...enabled, tool] })
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100 mb-2">Tools</h2>
        <p className="text-gray-600 dark:text-slate-400">Select the tools your agent can use</p>
      </div>

      <div className="space-y-3">
        {tools.map((tool) => {
          const enabled = (formData.enabled_tools || []).includes(tool)
          return (
            <label
              key={tool}
              className="flex items-center p-4 border rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-800"
            >
              <input
                type="checkbox"
                checked={enabled}
                onChange={() => toggleTool(tool)}
                className="w-5 h-5 text-blue-600 rounded"
              />
              <div className="ml-3">
                <div className="font-medium text-gray-900 dark:text-slate-100">{tool.toUpperCase()}</div>
                <div className="text-sm text-gray-500 dark:text-slate-400">{getToolDescription(tool)}</div>
              </div>
            </label>
          )
        })}
      </div>
    </div>
  )
}

function StepMemory({
  formData,
  setFormData,
}: {
  formData: CreateAgentRequest
  setFormData: (data: CreateAgentRequest) => void
}) {
  const memoryEnabled = formData.config?.memory_enabled !== false

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100 mb-2">Memory Settings</h2>
        <p className="text-gray-600 dark:text-slate-400">Configure long-term memory for your agent</p>
      </div>

      <div className="space-y-4">
        <label className="flex items-center p-4 border rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-800">
          <input
            type="checkbox"
            checked={memoryEnabled}
            onChange={(e) =>
              setFormData({
                ...formData,
                config: { ...formData.config, memory_enabled: e.target.checked },
              })
            }
            className="w-5 h-5 text-blue-600 rounded"
          />
          <div className="ml-3">
            <div className="font-medium text-gray-900 dark:text-slate-100">Enable Long-term Memory</div>
            <div className="text-sm text-gray-500 dark:text-slate-400">
              Allow agent to remember past conversations and context
            </div>
          </div>
        </label>
      </div>
    </div>
  )
}

function StepReview({
  formData,
  selectedProfile,
  profiles,
  models,
}: {
  formData: CreateAgentRequest
  selectedProfile: string
  profiles: Profile[]
  models: Model[]
}) {
  const profile = profiles.find((p) => p.id === selectedProfile)
  const model = models.find((m) => m.name === formData.model_name)

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100 mb-2">Review & Create</h2>
        <p className="text-gray-600 dark:text-slate-400">Review your agent configuration before creating</p>
      </div>

      <div className="space-y-4">
        <div>
          <h3 className="font-semibold text-gray-900 dark:text-slate-100 mb-2">Basic Information</h3>
          <div className="bg-gray-50 dark:bg-slate-800 p-4 rounded">
            <p>
              <strong>Name:</strong> {formData.name}
            </p>
            {formData.description && (
              <p>
                <strong>Description:</strong> {formData.description}
              </p>
            )}
          </div>
        </div>

        <div>
          <h3 className="font-semibold text-gray-900 dark:text-slate-100 mb-2">Configuration</h3>
          <div className="bg-gray-50 dark:bg-slate-800 p-4 rounded space-y-1">
            <p>
              <strong>Profile:</strong> {profile?.name}
            </p>
            <p>
              <strong>Model:</strong> {model?.display_name || formData.model_name} ({model?.provider || 'N/A'})
            </p>
            <p>
              <strong>Tools:</strong> {formData.enabled_tools?.join(', ') || 'None'}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function getToolDescription(tool: string): string {
  const descriptions: Record<string, string> = {
    sql: 'Execute SQL queries on the database',
    http: 'Make HTTP requests to external APIs',
    code: 'Execute code in a sandboxed environment',
    shell: 'Execute shell commands (restricted)',
    browser: 'Browser automation with Playwright',
  }
  return descriptions[tool] || 'Tool description'
}



