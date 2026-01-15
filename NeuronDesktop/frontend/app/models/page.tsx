'use client'

import { useState, useEffect, useCallback } from 'react'
import { profilesAPI, modelConfigAPI, type Profile, type ModelConfig } from '@/lib/api'
import { 
  KeyIcon,
  PlusIcon,
  TrashIcon,
  PencilIcon,
  CheckCircleIcon,
  XCircleIcon,
  SparklesIcon
} from '@/components/Icons'

// Available model providers and their models
const MODEL_PROVIDERS = {
  openai: {
    name: 'OpenAI',
    models: [
      { name: 'gpt-4', displayName: 'GPT-4' },
      { name: 'gpt-4-turbo', displayName: 'GPT-4 Turbo' },
      { name: 'gpt-4o', displayName: 'GPT-4o' },
      { name: 'gpt-3.5-turbo', displayName: 'GPT-3.5 Turbo' },
    ],
    requiresKey: true,
    requiresBaseUrl: false,
  },
  anthropic: {
    name: 'Anthropic',
    models: [
      { name: 'claude-3-opus-20240229', displayName: 'Claude 3 Opus' },
      { name: 'claude-3-sonnet-20240229', displayName: 'Claude 3 Sonnet' },
      { name: 'claude-3-haiku-20240307', displayName: 'Claude 3 Haiku' },
      { name: 'claude-3-5-sonnet-20241022', displayName: 'Claude 3.5 Sonnet' },
    ],
    requiresKey: true,
    requiresBaseUrl: false,
  },
  google: {
    name: 'Google',
    models: [
      { name: 'gemini-pro', displayName: 'Gemini Pro' },
      { name: 'gemini-pro-vision', displayName: 'Gemini Pro Vision' },
      { name: 'gemini-1.5-pro', displayName: 'Gemini 1.5 Pro' },
    ],
    requiresKey: true,
    requiresBaseUrl: false,
  },
  ollama: {
    name: 'Ollama (Free)',
    models: [
      { name: 'llama2', displayName: 'Llama 2' },
      { name: 'llama2:13b', displayName: 'Llama 2 13B' },
      { name: 'llama2:70b', displayName: 'Llama 2 70B' },
      { name: 'mistral', displayName: 'Mistral' },
      { name: 'codellama', displayName: 'Code Llama' },
      { name: 'phi', displayName: 'Phi' },
    ],
    requiresKey: false,
    requiresBaseUrl: true,
    defaultBaseUrl: 'http://localhost:11434',
  },
  custom: {
    name: 'Custom',
    models: [],
    requiresKey: true,
    requiresBaseUrl: true,
  },
}

export default function ModelsPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [modelConfigs, setModelConfigs] = useState<ModelConfig[]>([])
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editingConfig, setEditingConfig] = useState<ModelConfig | null>(null)
  const [loading, setLoading] = useState(false)

  // Create form state
  const [newProvider, setNewProvider] = useState<string>('openai')
  const [newModelName, setNewModelName] = useState<string>('')
  const [newApiKey, setNewApiKey] = useState<string>('')
  const [newBaseUrl, setNewBaseUrl] = useState<string>('')
  const [newIsDefault, setNewIsDefault] = useState(false)
  const [showApiKey, setShowApiKey] = useState(false)

  const loadProfiles = useCallback(async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0 && !selectedProfile) {
        setSelectedProfile(response.data[0].id)
      }
    } catch (error) {
      console.error('Failed to load profiles:', error)
    }
  }, [selectedProfile])

  const loadModelConfigs = useCallback(async () => {
    if (!selectedProfile) return
    try {
      const response = await modelConfigAPI.list(selectedProfile, false)
      setModelConfigs(response.data)
    } catch (error) {
      console.error('Failed to load model configs:', error)
    }
  }, [selectedProfile])

  useEffect(() => {
    loadProfiles()
  }, [loadProfiles])

  useEffect(() => {
    if (selectedProfile) {
      loadModelConfigs()
    }
  }, [selectedProfile, loadModelConfigs])

  const handleCreate = async () => {
    if (!selectedProfile) {
      alert('Please select a profile first')
      return
    }

    const provider = MODEL_PROVIDERS[newProvider as keyof typeof MODEL_PROVIDERS]
    if (!provider) {
      alert('Invalid provider')
      return
    }

    if (provider.requiresKey && !newApiKey.trim()) {
      alert('API key is required for this provider')
      return
    }

    if (provider.requiresBaseUrl && !newBaseUrl.trim()) {
      alert('Base URL is required for this provider')
      return
    }

    if (!newModelName.trim()) {
      alert('Model name is required')
      return
    }

    setLoading(true)
    try {
      await modelConfigAPI.create(selectedProfile, {
        model_provider: newProvider,
        model_name: newModelName,
        api_key: newApiKey || undefined,
        base_url: newBaseUrl || (provider as any).defaultBaseUrl || undefined,
        is_default: newIsDefault,
        is_free: newProvider === 'ollama',
      })
      await loadModelConfigs()
      setShowCreateModal(false)
      resetForm()
      alert('Model configuration created successfully!')
    } catch (error: any) {
      console.error('Failed to create model config:', error)
      alert('Failed to create model configuration: ' + (error.response?.data?.error || error.message))
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this model configuration?')) {
      return
    }

    if (!selectedProfile) return
    try {
      await modelConfigAPI.delete(selectedProfile, id)
      await loadModelConfigs()
      alert('Model configuration deleted successfully!')
    } catch (error: any) {
      console.error('Failed to delete model config:', error)
      alert('Failed to delete model configuration: ' + (error.response?.data?.error || error.message))
    }
  }

  const handleSetDefault = async (id: string) => {
    if (!selectedProfile) return
    try {
      await modelConfigAPI.setDefault(selectedProfile, id)
      await loadModelConfigs()
      alert('Default model updated!')
    } catch (error: any) {
      console.error('Failed to set default model:', error)
      alert('Failed to set default model: ' + (error.response?.data?.error || error.message))
    }
  }

  const resetForm = () => {
    setNewProvider('openai')
    setNewModelName('')
    setNewApiKey('')
    setNewBaseUrl('')
    setNewIsDefault(false)
    setShowApiKey(false)
  }

  const getProviderInfo = (provider: string) => {
    return MODEL_PROVIDERS[provider as keyof typeof MODEL_PROVIDERS] || null
  }

  return (
    <div className="h-full overflow-auto bg-[#1a1a1a]">
      <div className="max-w-6xl mx-auto p-6">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-[#e0e0e0] mb-2">Model Settings</h1>
            <p className="text-[#999999]">Configure API keys and settings for AI models</p>
          </div>
          {selectedProfile && (
            <button
              onClick={() => setShowCreateModal(true)}
              className="btn-primary flex items-center gap-2"
            >
              <PlusIcon className="w-4 h-4" />
              Add Model
            </button>
          )}
        </div>

        {/* Profile Selector */}
        <div className="card mb-6">
          <label className="block text-sm font-medium text-[#c8c8c8] mb-2">
            Profile
          </label>
          <select
            value={selectedProfile}
            onChange={(e) => setSelectedProfile(e.target.value)}
            className="input w-full"
          >
            <option value="">Select a profile</option>
            {profiles.map((profile) => (
              <option key={profile.id} value={profile.id}>
                {profile.name}
              </option>
            ))}
          </select>
        </div>

        {/* Model Configs List */}
        {selectedProfile && (
          <div className="space-y-4">
            {modelConfigs.length === 0 ? (
              <div className="card text-center py-12">
                <SparklesIcon className="w-12 h-12 mx-auto mb-4 text-[#999999]" />
                <p className="text-[#999999] mb-4">No model configurations yet</p>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="btn-primary"
                >
                  Add Your First Model
                </button>
              </div>
            ) : (
              modelConfigs.map((config) => {
                const provider = getProviderInfo(config.model_provider)
                return (
                  <div key={config.id} className="card">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="text-lg font-semibold text-[#e0e0e0]">
                            {provider?.name || config.model_provider} - {config.model_name}
                          </h3>
                          {config.is_default && (
                            <span className="badge badge-success">Default</span>
                          )}
                          {config.is_free && (
                            <span className="badge badge-info">Free</span>
                          )}
                        </div>
                        <div className="space-y-1 text-sm text-[#999999]">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">Provider:</span>
                            <span>{config.model_provider}</span>
                          </div>
                          {config.base_url && (
                            <div className="flex items-center gap-2">
                              <span className="font-medium">Base URL:</span>
                              <span className="font-mono text-xs">{config.base_url}</span>
                            </div>
                          )}
                          <div className="flex items-center gap-2">
                            <span className="font-medium">API Key:</span>
                            <span>{config.api_key ? '✓ Set' : '✗ Not set'}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {!config.is_default && (
                          <button
                            onClick={() => handleSetDefault(config.id)}
                            className="px-3 py-1.5 text-sm border border-[#333333] rounded-lg hover:bg-[#2d2d2d] text-[#c8c8c8]"
                            title="Set as default"
                          >
                            Set Default
                          </button>
                        )}
                        <button
                          onClick={() => handleDelete(config.id)}
                          className="p-2 text-[#999999] hover:text-red-400"
                          title="Delete"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        )}

        {!selectedProfile && (
          <div className="card text-center py-12">
            <KeyIcon className="w-12 h-12 mx-auto mb-4 text-[#999999]" />
            <p className="text-[#999999]">Please select a profile to manage model configurations</p>
          </div>
        )}
      </div>

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-[#252525] rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-[#333333]">
              <h2 className="text-2xl font-bold text-[#e0e0e0]">Add Model Configuration</h2>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-[#c8c8c8] mb-2">
                  Provider *
                </label>
                <select
                  value={newProvider}
                  onChange={(e) => {
                    setNewProvider(e.target.value)
                    setNewModelName('')
                    const provider = MODEL_PROVIDERS[e.target.value as keyof typeof MODEL_PROVIDERS]
                    if ((provider as any)?.defaultBaseUrl) {
                      setNewBaseUrl((provider as any).defaultBaseUrl)
                    } else {
                      setNewBaseUrl('')
                    }
                  }}
                  className="input w-full"
                >
                  {Object.entries(MODEL_PROVIDERS).map(([key, provider]) => (
                    <option key={key} value={key}>
                      {provider.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-[#c8c8c8] mb-2">
                  Model *
                </label>
                {newProvider === 'custom' ? (
                  <input
                    type="text"
                    value={newModelName}
                    onChange={(e) => setNewModelName(e.target.value)}
                    className="input w-full"
                    placeholder="Enter custom model name"
                  />
                ) : (
                  <select
                    value={newModelName}
                    onChange={(e) => setNewModelName(e.target.value)}
                    className="input w-full"
                  >
                    <option value="">Select a model</option>
                    {MODEL_PROVIDERS[newProvider as keyof typeof MODEL_PROVIDERS]?.models.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.displayName}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {MODEL_PROVIDERS[newProvider as keyof typeof MODEL_PROVIDERS]?.requiresKey && (
                <div>
                  <label className="block text-sm font-medium text-[#c8c8c8] mb-2">
                    API Key *
                  </label>
                  <div className="flex gap-2">
                    <input
                      type={showApiKey ? 'text' : 'password'}
                      value={newApiKey}
                      onChange={(e) => setNewApiKey(e.target.value)}
                      className="input flex-1"
                      placeholder="Enter API key"
                    />
                    <button
                      onClick={() => setShowApiKey(!showApiKey)}
                      className="px-4 py-2 border border-[#333333] rounded-lg hover:bg-[#2d2d2d]"
                    >
                      {showApiKey ? 'Hide' : 'Show'}
                    </button>
                  </div>
                </div>
              )}

              {MODEL_PROVIDERS[newProvider as keyof typeof MODEL_PROVIDERS]?.requiresBaseUrl && (
                <div>
                  <label className="block text-sm font-medium text-[#c8c8c8] mb-2">
                    Base URL *
                  </label>
                  <input
                    type="text"
                    value={newBaseUrl}
                    onChange={(e) => setNewBaseUrl(e.target.value)}
                    className="input w-full"
                    placeholder="http://localhost:11434"
                  />
                </div>
              )}

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="isDefault"
                  checked={newIsDefault}
                  onChange={(e) => setNewIsDefault(e.target.checked)}
                  className="w-4 h-4 rounded border-[#333333] bg-[#252525]"
                />
                <label htmlFor="isDefault" className="text-sm text-[#c8c8c8]">
                  Set as default model for this profile
                </label>
              </div>
            </div>

            <div className="p-6 border-t border-[#333333] flex justify-end gap-3">
              <button
                onClick={() => {
                  setShowCreateModal(false)
                  resetForm()
                }}
                className="btn-secondary"
                disabled={loading}
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={loading}
                className="btn-primary"
              >
                {loading ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

