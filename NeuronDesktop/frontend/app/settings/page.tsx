'use client'

import { useState, useEffect, useCallback } from 'react'
import { profilesAPI, mcpAPI, agentAPI, modelConfigAPI, factoryAPI, type Profile, type ModelConfig, type FactoryStatus, type ComponentStatus } from '@/lib/api'
import { getErrorMessage } from '@/lib/errors'
import { useTheme } from '@/contexts/ThemeContext'
// Authentication is now handled via JWT tokens, no need for API key management
import { 
  ServerIcon,
  DatabaseIcon,
  PlusIcon,
  TrashIcon,
  PencilIcon,
  SparklesIcon,
  CpuIcon,
  CheckCircleIcon,
  XCircleIcon,
  ArrowDownTrayIcon,
  ArrowUpTrayIcon
} from '@/components/Icons'
import { showSuccessToast, showErrorToast } from '@/lib/errors'

export default function SettingsPage() {
  const { theme, setTheme } = useTheme()
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [expandedProfile, setExpandedProfile] = useState<string | null>(null)
  const [editingProfile, setEditingProfile] = useState<Profile | null>(null)
  const [profileError, setProfileError] = useState<string | null>(null)
  const [notice, setNotice] = useState<{ type: 'success' | 'error'; message: string } | null>(null)

  // MCP Config state
  const [mcpCommand, setMcpCommand] = useState('')
  const [mcpArgs, setMcpArgs] = useState<string[]>([''])
  const [mcpEnv, setMcpEnv] = useState<Array<{ key: string; value: string }>>([{ key: '', value: '' }])

  // Agent Config state
  const [agentEndpoint, setAgentEndpoint] = useState('')
  const [agentApiKey, setAgentApiKey] = useState('')
  const [showAgentApiKey, setShowAgentApiKey] = useState(false)

  // Test state
  const [mcpTestPassed, setMcpTestPassed] = useState(false)
  const [mcpTestLoading, setMcpTestLoading] = useState(false)
  const [mcpTestError, setMcpTestError] = useState<string | null>(null)
  const [agentTestPassed, setAgentTestPassed] = useState(false)
  const [agentTestLoading, setAgentTestLoading] = useState(false)
  const [agentTestError, setAgentTestError] = useState<string | null>(null)

  // Connection Status state
  const [connectionStatus, setConnectionStatus] = useState<FactoryStatus | null>(null)
  const [statusLoading, setStatusLoading] = useState(false)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [testingService, setTestingService] = useState<'neurondb' | 'neuronagent' | 'neuronmcp' | null>(null)
  
  // Configuration editing state
  const [editingNeuronDB, setEditingNeuronDB] = useState(false)
  const [editingNeuronAgent, setEditingNeuronAgent] = useState(false)
  const [neurondbDSN, setNeurondbDSN] = useState('')
  const [neurondbSaving, setNeurondbSaving] = useState(false)
  const [neuronagentEndpoint, setNeuronagentEndpoint] = useState('')
  const [neuronagentApiKey, setNeuronagentApiKey] = useState('')
  const [neuronagentSaving, setNeuronagentSaving] = useState(false)
  const [showNeuronagentApiKey, setShowNeuronagentApiKey] = useState(false)

  // Model Config state
  const [modelConfigs, setModelConfigs] = useState<ModelConfig[]>([])
  const [selectedProfileForModels, setSelectedProfileForModels] = useState<string>('')
  const [showModelModal, setShowModelModal] = useState(false)
  const [editingModel, setEditingModel] = useState<ModelConfig | null>(null)
  const [newModelProvider, setNewModelProvider] = useState<string>('openai')
  const [newModelName, setNewModelName] = useState<string>('')
  const [newModelApiKey, setNewModelApiKey] = useState<string>('')
  const [newModelBaseUrl, setNewModelBaseUrl] = useState<string>('')
  const [newModelIsDefault, setNewModelIsDefault] = useState(false)
  const [showModelApiKey, setShowModelApiKey] = useState(false)
  
  // Inline editing state
  const [editingApiKeyId, setEditingApiKeyId] = useState<string | null>(null)
  const [inlineApiKey, setInlineApiKey] = useState<string>('')
  const [savingApiKey, setSavingApiKey] = useState<string | null>(null)
  const [settingDefault, setSettingDefault] = useState<string | null>(null)
  const [togglingEnabled, setTogglingEnabled] = useState<string | null>(null)

  // Active settings section with nested support
  const [activeSection, setActiveSection] = useState<'modules' | 'appearance' | 'profiles'>('modules')
  const [activeSubSection, setActiveSubSection] = useState<'neurondb' | 'neuronagent' | 'neuronmcp' | 'models' | null>('neurondb')
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['modules', 'profiles']))

  // Model providers configuration
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
        { name: 'llama3:8b', displayName: 'Llama 3 8B' },
        { name: 'llama3:70b', displayName: 'Llama 3 70B ⭐' },
        { name: 'mistral', displayName: 'Mistral' },
        { name: 'mistral:7b', displayName: 'Mistral 7B' },
        { name: 'mixtral:8x7b', displayName: 'Mixtral 8x7B' },
        { name: 'codellama', displayName: 'Code Llama' },
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

  const loadProfiles = useCallback(async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0 && !selectedProfileForModels) {
        const defaultProfile = response.data.find(p => p.is_default)
        setSelectedProfileForModels(defaultProfile ? defaultProfile.id : response.data[0].id)
      }
    } catch (error: any) {
      showErrorToast('Failed to load profiles: ' + (error.response?.data?.error || error.message))
    }
  }, [selectedProfileForModels])

  const loadModelConfigs = useCallback(async () => {
    if (!selectedProfileForModels) return
    try {
      // Load with API keys for inline editing
      const response = await modelConfigAPI.list(selectedProfileForModels, true)
      setModelConfigs(response.data)
    } catch (error: any) {
      showErrorToast('Failed to load model configs: ' + (error.response?.data?.error || error.message))
    }
  }, [selectedProfileForModels])

  const loadConnectionStatus = useCallback(async () => {
    setStatusLoading(true)
    setStatusError(null)
    try {
      const response = await factoryAPI.getStatus()
      setConnectionStatus(response.data)
    } catch (error: any) {
      showErrorToast('Failed to load connection status: ' + (error.response?.data?.error || error.message))
      setStatusError(error.response?.data?.error || error.message || 'Failed to load connection status')
    } finally {
      setStatusLoading(false)
    }
  }, [])

  useEffect(() => {
    loadProfiles()
    loadConnectionStatus()
    // Refresh status every 30 seconds
    const interval = setInterval(loadConnectionStatus, 30000)
    return () => clearInterval(interval)
  }, [loadProfiles, loadConnectionStatus])

  // Load current configurations when status is loaded
  useEffect(() => {
    if (connectionStatus && profiles.length > 0) {
      const defaultProfile = profiles.find(p => p.is_default) || profiles[0]
      if (defaultProfile) {
        if (!editingNeuronDB && defaultProfile.neurondb_dsn) {
          setNeurondbDSN(defaultProfile.neurondb_dsn)
        }
        if (!editingNeuronAgent) {
          if (defaultProfile.agent_endpoint) {
            setNeuronagentEndpoint(defaultProfile.agent_endpoint)
          }
          if (defaultProfile.agent_api_key) {
            setNeuronagentApiKey(defaultProfile.agent_api_key)
          }
        }
      }
    }
  }, [connectionStatus, profiles, editingNeuronDB, editingNeuronAgent])

  useEffect(() => {
    if (selectedProfileForModels) {
      loadModelConfigs()
    }
  }, [selectedProfileForModels, loadModelConfigs])

  const testServiceConnection = async (service: 'neurondb' | 'neuronagent' | 'neuronmcp') => {
    setTestingService(service)
    try {
      // Reload status to get fresh connection test
      await loadConnectionStatus()
      setNotice({ type: 'success', message: `${service} connection status updated.` })
    } catch (error: any) {
      setNotice({ type: 'error', message: `Failed to test ${service}: ${error.message}` })
    } finally {
      setTestingService(null)
    }
  }

  const getStatusColor = (status: ComponentStatus | undefined) => {
    if (!status) return 'text-slate-600 dark:text-slate-400'
    if (status.status === 'running' && status.reachable) return 'text-green-400'
    if (status.status === 'reachable') return 'text-yellow-400'
    return 'text-red-400'
  }

  const getStatusBadge = (status: ComponentStatus | undefined) => {
    if (!status) return { text: 'Unknown', color: 'bg-slate-600' }
    if (status.status === 'running' && status.reachable) return { text: 'Connected', color: 'bg-green-600' }
    if (status.status === 'reachable') return { text: 'Reachable', color: 'bg-yellow-600' }
    if (status.status === 'installed') return { text: 'Installed', color: 'bg-blue-600' }
    return { text: 'Disconnected', color: 'bg-red-600' }
  }

  const handleSaveNeuronDBConfig = async () => {
    const defaultProfile = profiles.find(p => p.is_default) || profiles[0]
    if (!defaultProfile || !neurondbDSN.trim()) {
      setNotice({ type: 'error', message: 'Please enter a valid DSN' })
      return
    }

    setNeurondbSaving(true)
    try {
      const updatedProfile = {
        ...defaultProfile,
        neurondb_dsn: neurondbDSN.trim()
      }
      await profilesAPI.update(defaultProfile.id, updatedProfile)
      await loadProfiles()
      await loadConnectionStatus()
      setEditingNeuronDB(false)
      setNotice({ type: 'success', message: 'NeuronDB DSN updated successfully' })
    } catch (error: any) {
      setNotice({ type: 'error', message: 'Failed to update DSN: ' + (error.response?.data?.error || error.message) })
    } finally {
      setNeurondbSaving(false)
    }
  }

  const handleSaveNeuronAgentConfig = async () => {
    const defaultProfile = profiles.find(p => p.is_default) || profiles[0]
    if (!defaultProfile || !neuronagentEndpoint.trim()) {
      setNotice({ type: 'error', message: 'Please enter a valid endpoint' })
      return
    }

    setNeuronagentSaving(true)
    try {
      const updatedProfile = {
        ...defaultProfile,
        agent_endpoint: neuronagentEndpoint.trim(),
        agent_api_key: neuronagentApiKey.trim() || undefined
      }
      await profilesAPI.update(defaultProfile.id, updatedProfile)
      await loadProfiles()
      await loadConnectionStatus()
      setEditingNeuronAgent(false)
      setNotice({ type: 'success', message: 'NeuronAgent configuration updated successfully' })
    } catch (error: any) {
      setNotice({ type: 'error', message: 'Failed to update configuration: ' + (error.response?.data?.error || error.message) })
    } finally {
      setNeuronagentSaving(false)
    }
  }

  const handleCreateModel = async () => {
    if (!selectedProfileForModels) {
      setNotice({ type: 'error', message: 'Please select a profile first.' })
      return
    }

    const provider = MODEL_PROVIDERS[newModelProvider as keyof typeof MODEL_PROVIDERS]
    if (!provider) {
      setNotice({ type: 'error', message: 'Invalid provider.' })
      return
    }

    if (provider.requiresKey && !newModelApiKey.trim()) {
      setNotice({ type: 'error', message: 'API key is required for this provider.' })
      return
    }

    if (provider.requiresBaseUrl && !newModelBaseUrl.trim()) {
      setNotice({ type: 'error', message: 'Base URL is required for this provider.' })
      return
    }

    if (!newModelName.trim()) {
      setNotice({ type: 'error', message: 'Model name is required.' })
      return
    }

    try {
      await modelConfigAPI.create(selectedProfileForModels, {
        model_provider: newModelProvider,
        model_name: newModelName,
        api_key: newModelApiKey || undefined,
        base_url: newModelBaseUrl || (provider as any).defaultBaseUrl || undefined,
        is_default: newModelIsDefault,
        is_free: newModelProvider === 'ollama',
      })
      await loadModelConfigs()
      setShowModelModal(false)
      setNewModelProvider('openai')
      setNewModelName('')
      setNewModelApiKey('')
      setNewModelBaseUrl('')
      setNewModelIsDefault(false)
      setNotice({ type: 'success', message: 'Model configuration created.' })
    } catch (error: any) {
      showErrorToast('Failed to create model config: ' + (error.response?.data?.error || error.message))
      setNotice({ type: 'error', message: 'Failed to create model configuration: ' + (error.response?.data?.error || error.message) })
    }
  }

  const handleDeleteModel = async (id: string) => {
    if (!confirm('Are you sure you want to delete this model configuration?')) {
      return
    }
    if (!selectedProfileForModels) return
    try {
      await modelConfigAPI.delete(selectedProfileForModels, id)
      await loadModelConfigs()
      setNotice({ type: 'success', message: 'Model configuration deleted.' })
    } catch (error: any) {
      showErrorToast('Failed to delete model config: ' + (error.response?.data?.error || error.message))
      setNotice({ type: 'error', message: 'Failed to delete model configuration: ' + (error.response?.data?.error || error.message) })
    }
  }

  const handleEditModel = (config: ModelConfig) => {
    setEditingModel(config)
    setNewModelProvider(config.model_provider)
    setNewModelName(config.model_name)
    setNewModelApiKey(config.api_key || '')
    setNewModelBaseUrl(config.base_url || '')
    setNewModelIsDefault(config.is_default)
    setShowModelModal(true)
  }

  const handleUpdateModel = async () => {
    if (!editingModel || !selectedProfileForModels) return

    const provider = MODEL_PROVIDERS[newModelProvider as keyof typeof MODEL_PROVIDERS]
    if (provider?.requiresKey && !newModelApiKey.trim()) {
      setNotice({ type: 'error', message: 'API key is required for this provider.' })
      return
    }

    try {
      await modelConfigAPI.update(selectedProfileForModels, editingModel.id, {
        model_provider: newModelProvider,
        model_name: newModelName,
        api_key: newModelApiKey || undefined,
        base_url: newModelBaseUrl || undefined,
        is_default: newModelIsDefault,
      })
      await loadModelConfigs()
      setShowModelModal(false)
      setEditingModel(null)
      setNotice({ type: 'success', message: 'Model configuration updated.' })
    } catch (error: any) {
      showErrorToast('Failed to update model config: ' + (error.response?.data?.error || error.message))
      setNotice({ type: 'error', message: 'Failed to update model configuration: ' + (error.response?.data?.error || error.message) })
    }
  }

  const handleSetDefault = async (modelId: string) => {
    if (!selectedProfileForModels) return
    setSettingDefault(modelId)
    try {
      await modelConfigAPI.setDefault(selectedProfileForModels, modelId)
      await loadModelConfigs()
      setNotice({ type: 'success', message: 'Default model updated.' })
    } catch (error: any) {
      showErrorToast('Failed to set default model: ' + (error.response?.data?.error || error.message))
      setNotice({ type: 'error', message: 'Failed to set default model: ' + (error.response?.data?.error || error.message) })
    } finally {
      setSettingDefault(null)
    }
  }

  const handleSaveInlineApiKey = async (modelId: string) => {
    if (!selectedProfileForModels) return
    setSavingApiKey(modelId)
    try {
      await modelConfigAPI.update(selectedProfileForModels, modelId, {
        api_key: inlineApiKey || undefined,
      })
      await loadModelConfigs()
      setEditingApiKeyId(null)
      setInlineApiKey('')
      setNotice({ type: 'success', message: 'API key updated.' })
    } catch (error: any) {
      showErrorToast('Failed to update API key: ' + (error.response?.data?.error || error.message))
      setNotice({ type: 'error', message: 'Failed to update API key: ' + (error.response?.data?.error || error.message) })
    } finally {
      setSavingApiKey(null)
    }
  }

  const handleToggleEnabled = async (modelId: string, currentEnabled: boolean) => {
    if (!selectedProfileForModels) return
    setTogglingEnabled(modelId)
    try {
      const config = modelConfigs.find(m => m.id === modelId)
      if (!config) return
      
      await modelConfigAPI.update(selectedProfileForModels, modelId, {
        metadata: {
          ...(config.metadata || {}),
          enabled: !currentEnabled,
        },
      })
      await loadModelConfigs()
      setNotice({ type: 'success', message: `Model ${!currentEnabled ? 'enabled' : 'disabled'}.` })
    } catch (error: any) {
      showErrorToast('Failed to toggle model enabled state: ' + (error.response?.data?.error || error.message))
      setNotice({ type: 'error', message: 'Failed to update model: ' + (error.response?.data?.error || error.message) })
    } finally {
      setTogglingEnabled(null)
    }
  }

  const isModelEnabled = (config: ModelConfig): boolean => {
    return config.metadata?.enabled !== false // Default to enabled if not specified
  }


  const handleEditProfile = (profile: Profile) => {
    setEditingProfile(profile)
    setExpandedProfile(profile.id)
    
    // Reset test states
    setMcpTestPassed(false)
    setMcpTestError(null)
    setAgentTestPassed(false)
    setAgentTestError(null)
    
    // Load MCP config
    if (profile.mcp_config) {
      setMcpCommand((profile.mcp_config.command as string) || 'neurondb-mcp')
      const args = profile.mcp_config.args as string[] || []
      setMcpArgs(args.length > 0 ? args : [''])
      
      const env = profile.mcp_config.env as Record<string, string> || {}
      const envEntries = Object.entries(env)
      setMcpEnv(envEntries.length > 0 ? envEntries.map(([k, v]) => ({ key: k, value: v })) : [{ key: '', value: '' }])
    } else {
      setMcpCommand('neurondb-mcp')
      setMcpArgs([''])
      setMcpEnv([{ key: '', value: '' }])
    }
    
    // Load Agent config
    setAgentEndpoint(profile.agent_endpoint || '')
    setAgentApiKey(profile.agent_api_key || '')
  }

  const handleTestMCP = async () => {
    setMcpTestLoading(true)
    setMcpTestError(null)
    setMcpTestPassed(false)
    
    try {
      const env: Record<string, string> = {}
      mcpEnv.forEach(({ key, value }) => {
        if (key.trim() && value.trim()) {
          env[key.trim()] = value.trim()
        }
      })
      
      const config = {
        command: mcpCommand || 'neurondb-mcp',
        args: mcpArgs.filter(arg => arg.trim() !== ''),
        env: env
      }
      
      await mcpAPI.testConfig(config)
      setMcpTestPassed(true)
      setMcpTestError(null)
    } catch (error: any) {
      setMcpTestPassed(false)
      const errorMsg = error.response?.data?.message || error.response?.data?.error || error.message || 'Test failed'
      setMcpTestError(errorMsg)
    } finally {
      setMcpTestLoading(false)
    }
  }

  const handleTestAgent = async () => {
    setAgentTestLoading(true)
    setAgentTestError(null)
    setAgentTestPassed(false)
    
    try {
      if (!agentEndpoint) {
        throw new Error('Agent endpoint is required')
      }
      
      await agentAPI.testConfig({
        endpoint: agentEndpoint,
        api_key: agentApiKey
      })
      setAgentTestPassed(true)
      setAgentTestError(null)
    } catch (error: any) {
      setAgentTestPassed(false)
      const errorMsg = error.response?.data?.message || error.response?.data?.error || error.message || 'Test failed'
      setAgentTestError(errorMsg)
    } finally {
      setAgentTestLoading(false)
    }
  }

  const handleSaveMCPConfig = async () => {
    if (!editingProfile) return
    
    try {
      const mcpConfig: Record<string, any> = {
        command: mcpCommand,
        args: mcpArgs.filter(arg => arg.trim() !== ''),
        env: {}
      }
      
      mcpEnv.forEach(({ key, value }) => {
        if (key.trim() && value.trim()) {
          mcpConfig.env[key.trim()] = value.trim()
        }
      })
      
      const updatedProfile = {
        ...editingProfile,
        mcp_config: mcpConfig
      }
      
      await profilesAPI.update(editingProfile.id, updatedProfile)
      await loadProfiles()
      setNotice({ type: 'success', message: `MCP configuration saved to profile "${editingProfile.name}".` })
    } catch (error: any) {
      showErrorToast('Failed to save MCP config: ' + (error.response?.data?.error || error.message))
      setNotice({ type: 'error', message: 'Failed to save MCP configuration.' })
    }
  }

  const handleSaveAgentConfig = async () => {
    if (!editingProfile) return
    
    try {
      const updatedProfile = {
        ...editingProfile,
        agent_endpoint: agentEndpoint,
        agent_api_key: agentApiKey
      }
      
      await profilesAPI.update(editingProfile.id, updatedProfile)
      await loadProfiles()
      setNotice({ type: 'success', message: `Agent configuration saved to profile "${editingProfile.name}".` })
    } catch (error: any) {
      showErrorToast('Failed to save agent config: ' + (error.response?.data?.error || error.message))
      setNotice({ type: 'error', message: 'Failed to save agent configuration.' })
    }
  }

  const addMcpArg = () => {
    setMcpArgs([...mcpArgs, ''])
    setMcpTestPassed(false)
    setMcpTestError(null)
  }

  const removeMcpArg = (index: number) => {
    setMcpArgs(mcpArgs.filter((_, i) => i !== index))
    setMcpTestPassed(false)
    setMcpTestError(null)
  }

  const updateMcpArg = (index: number, value: string) => {
    const newArgs = [...mcpArgs]
    newArgs[index] = value
    setMcpArgs(newArgs)
  }

  const addMcpEnv = () => {
    setMcpEnv([...mcpEnv, { key: '', value: '' }])
    setMcpTestPassed(false)
    setMcpTestError(null)
  }

  const removeMcpEnv = (index: number) => {
    setMcpEnv(mcpEnv.filter((_, i) => i !== index))
    setMcpTestPassed(false)
    setMcpTestError(null)
  }

  const updateMcpEnv = (index: number, field: 'key' | 'value', value: string) => {
    const newEnv = [...mcpEnv]
    newEnv[index] = { ...newEnv[index], [field]: value }
    setMcpEnv(newEnv)
  }


  const settingsSections = [
    { 
      id: 'modules' as const, 
      name: 'Modules', 
      icon: ServerIcon,
      subSections: [
        { id: 'neurondb' as const, name: 'NeuronDB', icon: DatabaseIcon },
        { id: 'neuronagent' as const, name: 'NeuronAgent', icon: CpuIcon },
        { id: 'neuronmcp' as const, name: 'NeuronMCP', icon: SparklesIcon },
      ]
    },
    { id: 'appearance' as const, name: 'Appearance', icon: SparklesIcon },
    { 
      id: 'profiles' as const, 
      name: 'Profiles', 
      icon: ServerIcon,
      subSections: [
        { id: 'models' as const, name: 'Models', icon: SparklesIcon },
      ]
    },
  ]

  const toggleSection = (sectionId: string) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(sectionId)) {
      newExpanded.delete(sectionId)
    } else {
      newExpanded.add(sectionId)
    }
    setExpandedSections(newExpanded)
  }

  const handleSectionClick = (sectionId: string, subSectionId?: string) => {
    setActiveSection(sectionId as any)
    if (subSectionId) {
      setActiveSubSection(subSectionId as any)
    } else {
      // Clear sub-section when clicking a section without submenus
      const section = settingsSections.find(s => s.id === sectionId)
      if (!section?.subSections || section.subSections.length === 0) {
        setActiveSubSection(null)
      }
    }
    // Auto-expand section when clicking
    if (!expandedSections.has(sectionId)) {
      setExpandedSections(new Set([...expandedSections, sectionId]))
    }
  }

  return (
    <div className="w-full -mx-6 px-6" style={{ minHeight: 'calc(100vh - 4rem)', display: 'flex' }}>
      <div className="flex flex-1" style={{ minHeight: '100%' }}>
        {/* Sidebar Navigation */}
        <aside className="w-64 flex-shrink-0 py-6 border-r border-slate-700 flex flex-col" style={{ minHeight: '100%' }}>
          <h1 className="text-2xl font-bold text-slate-100 mb-6 px-4">Settings</h1>
          <nav className="space-y-1">
            {settingsSections.map((section) => {
              const Icon = section.icon
              const isExpanded = expandedSections.has(section.id)
              const hasSubSections = section.subSections && section.subSections.length > 0
              const isActive = activeSection === section.id && (!hasSubSections || activeSubSection !== null)
              
              return (
                <div key={section.id}>
                  <button
                    onClick={() => {
                      if (hasSubSections) {
                        if (!isExpanded) {
                          // Expand and select first submenu
                          setExpandedSections(new Set([...expandedSections, section.id]))
                          if (section.subSections && section.subSections.length > 0) {
                            handleSectionClick(section.id, section.subSections[0].id)
                          }
                        } else {
                          // Just toggle collapse
                          toggleSection(section.id)
                        }
                      } else {
                        handleSectionClick(section.id)
                      }
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${
                      isActive && !hasSubSections
                        ? 'bg-purple-600 text-white'
                        : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                    }`}
                  >
                    <Icon className="w-5 h-5 flex-shrink-0" />
                    <span className="font-medium truncate flex-1 min-w-0">{section.name}</span>
                    {hasSubSections && (
                      <span className={`text-xs transition-transform flex-shrink-0 ${isExpanded ? 'rotate-90' : ''}`}>
                        ›
                      </span>
                    )}
                  </button>
                  
                  {/* Sub-sections */}
                  {hasSubSections && isExpanded && section.subSections && (
                    <div className="ml-4 space-y-1">
                      {section.subSections.map((subSection) => {
                        const SubIcon = subSection.icon
                        const isSubActive = activeSection === section.id && activeSubSection === subSection.id
                        return (
                          <button
                            key={subSection.id}
                            onClick={() => handleSectionClick(section.id, subSection.id)}
                            className={`w-full flex items-center gap-3 px-4 py-2 text-left transition-colors text-sm ${
                              isSubActive
                                ? 'bg-purple-600 text-white'
                                : 'text-slate-700 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-200'
                            }`}
                          >
                            <SubIcon className="w-4 h-4 flex-shrink-0" />
                            <span className="truncate flex-1 min-w-0">{subSection.name}</span>
                          </button>
                        )
                      })}
                    </div>
                  )}
                </div>
              )
            })}
          </nav>
        </aside>

        {/* Main Content Area */}
        <div className="flex-1 min-w-0 py-6 overflow-y-auto px-6">
          <div className="w-full">
            {notice && (
              <div
                className={`mb-6 px-4 py-3 rounded-lg border ${
                  notice.type === 'success'
                    ? 'bg-green-500/10 border-green-500/50 text-green-300'
                    : 'bg-red-500/10 border-red-500/50 text-red-300'
                }`}
              >
                <div className="flex items-start gap-3">
                  {notice.type === 'success' ? (
                    <CheckCircleIcon className="w-5 h-5 mt-0.5 flex-shrink-0" />
                  ) : (
                    <XCircleIcon className="w-5 h-5 mt-0.5 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <p className="text-sm">{notice.message}</p>
                  </div>
                  <button
                    onClick={() => setNotice(null)}
                    className="text-slate-700 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 text-sm"
                    aria-label="Dismiss notice"
                  >
                    ×
                  </button>
                </div>
              </div>
            )}

            {/* Modules Section - NeuronDB */}
            {activeSection === 'modules' && activeSubSection === 'neurondb' && (
              <div className="card mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <DatabaseIcon className="w-6 h-6 text-slate-600 dark:text-slate-400" />
              <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-100">NeuronDB Connection</h2>
            </div>
            <button
              onClick={loadConnectionStatus}
              disabled={statusLoading}
              className="btn btn-secondary text-sm"
            >
              {statusLoading ? 'Refreshing...' : 'Refresh Status'}
            </button>
          </div>

          {statusError && (
            <div className="mb-4 px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/50 text-red-300 text-sm">
              {statusError}
            </div>
          )}

          <div className="grid grid-cols-1 gap-4">
            {/* NeuronDB Status */}
            <div className="border border-slate-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <DatabaseIcon className="w-5 h-5 text-slate-600 dark:text-slate-400" />
                  <h3 className="font-semibold text-slate-100">NeuronDB</h3>
                </div>
                <div className="flex items-center gap-2">
                  {testingService === 'neurondb' ? (
                    <div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <div className={`w-3 h-3 rounded-full ${getStatusBadge(connectionStatus?.neurondb).color}`} />
                  )}
                  {!editingNeuronDB && (
                    <button
                      onClick={() => setEditingNeuronDB(true)}
                      className="p-1 text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400"
                      title="Edit configuration"
                    >
                      <PencilIcon className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-700 dark:text-slate-400">Status:</span>
                  <span className={`font-medium ${getStatusColor(connectionStatus?.neurondb)}`}>
                    {getStatusBadge(connectionStatus?.neurondb).text}
                  </span>
                </div>
                
                {editingNeuronDB ? (
                  <div className="space-y-2 mt-3">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">DSN</label>
                      <input
                        type="text"
                        value={neurondbDSN}
                        onChange={(e) => setNeurondbDSN(e.target.value)}
                        className="input w-full text-xs"
                        placeholder="postgresql://user:password@host:port/database"
                      />
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={handleSaveNeuronDBConfig}
                        disabled={neurondbSaving}
                        className="flex-1 btn btn-primary text-xs py-1.5"
                      >
                        {neurondbSaving ? 'Saving...' : 'Save'}
                      </button>
                      <button
                        onClick={() => {
                          setEditingNeuronDB(false)
                          const defaultProfile = profiles.find(p => p.is_default) || profiles[0]
                          if (defaultProfile?.neurondb_dsn) {
                            setNeurondbDSN(defaultProfile.neurondb_dsn)
                          }
                        }}
                        className="btn btn-secondary text-xs py-1.5"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {connectionStatus?.neurondb?.details && (
                      <div className="text-xs text-slate-700 dark:text-slate-500 space-y-1">
                        {connectionStatus.neurondb.details.dsn && (
                          <div className="break-all">DSN: {String(connectionStatus.neurondb.details.dsn)}</div>
                        )}
                        {connectionStatus.neurondb.details.postgres_version && (
                          <div>PostgreSQL: {String(connectionStatus.neurondb.details.postgres_version).split(' ')[0]}</div>
                        )}
                        {connectionStatus.neurondb.details.extension_version && (
                          <div>Extension: {String(connectionStatus.neurondb.details.extension_version)}</div>
                        )}
                      </div>
                    )}
                    {connectionStatus?.neurondb?.error_message && (
                      <div className="text-xs text-red-400 mt-2 p-2 bg-red-500/10 rounded">
                        {connectionStatus.neurondb.error_message}
                      </div>
                    )}
                    <button
                      onClick={() => testServiceConnection('neurondb')}
                      disabled={testingService === 'neurondb'}
                      className="w-full mt-2 btn btn-secondary text-xs py-1.5"
                    >
                      Test Connection
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
            )}

            {/* Modules Section - NeuronAgent */}
            {activeSection === 'modules' && activeSubSection === 'neuronagent' && (
              <div className="card mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <CpuIcon className="w-6 h-6 text-slate-600 dark:text-slate-400" />
              <h2 className="text-xl font-semibold text-slate-100">NeuronAgent Connection</h2>
            </div>
            <button
              onClick={loadConnectionStatus}
              disabled={statusLoading}
              className="btn btn-secondary text-sm"
            >
              {statusLoading ? 'Refreshing...' : 'Refresh Status'}
            </button>
          </div>

          {statusError && (
            <div className="mb-4 px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/50 text-red-300 text-sm">
              {statusError}
            </div>
          )}

          <div className="grid grid-cols-1 gap-4">
            {/* NeuronAgent Status */}
            <div className="border border-slate-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <CpuIcon className="w-5 h-5 text-slate-400" />
                  <h3 className="font-semibold text-slate-100">NeuronAgent</h3>
                </div>
                <div className="flex items-center gap-2">
                  {testingService === 'neuronagent' ? (
                    <div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <div className={`w-3 h-3 rounded-full ${getStatusBadge(connectionStatus?.neuronagent).color}`} />
                  )}
                  {!editingNeuronAgent && (
                    <button
                      onClick={() => setEditingNeuronAgent(true)}
                      className="p-1 text-slate-600 dark:text-slate-400 hover:text-blue-600 dark:hover:text-blue-400"
                      title="Edit configuration"
                    >
                      <PencilIcon className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-700 dark:text-slate-400">Status:</span>
                  <span className={`font-medium ${getStatusColor(connectionStatus?.neuronagent)}`}>
                    {getStatusBadge(connectionStatus?.neuronagent).text}
                  </span>
                </div>
                
                {editingNeuronAgent ? (
                  <div className="space-y-2 mt-3">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Endpoint</label>
                      <input
                        type="text"
                        value={neuronagentEndpoint}
                        onChange={(e) => setNeuronagentEndpoint(e.target.value)}
                        className="input w-full text-xs"
                        placeholder="http://localhost:8080"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">API Key (optional)</label>
                      <div className="flex gap-2">
                        <input
                          type={showNeuronagentApiKey ? 'text' : 'password'}
                          value={neuronagentApiKey}
                          onChange={(e) => setNeuronagentApiKey(e.target.value)}
                          className="input flex-1 text-xs"
                          placeholder="Enter API key"
                        />
                        <button
                          onClick={() => setShowNeuronagentApiKey(!showNeuronagentApiKey)}
                          className="px-2 py-1 border border-slate-600 rounded text-xs"
                        >
                          {showNeuronagentApiKey ? 'Hide' : 'Show'}
                        </button>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={handleSaveNeuronAgentConfig}
                        disabled={neuronagentSaving}
                        className="flex-1 btn btn-primary text-xs py-1.5"
                      >
                        {neuronagentSaving ? 'Saving...' : 'Save'}
                      </button>
                      <button
                        onClick={() => {
                          setEditingNeuronAgent(false)
                          const defaultProfile = profiles.find(p => p.is_default) || profiles[0]
                          if (defaultProfile) {
                            if (defaultProfile.agent_endpoint) {
                              setNeuronagentEndpoint(defaultProfile.agent_endpoint)
                            }
                            if (defaultProfile.agent_api_key) {
                              setNeuronagentApiKey(defaultProfile.agent_api_key)
                            }
                          }
                        }}
                        className="btn btn-secondary text-xs py-1.5"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    {connectionStatus?.neuronagent?.details && (
                      <div className="text-xs text-slate-700 dark:text-slate-500 space-y-1">
                        {connectionStatus.neuronagent.details.endpoint && (
                          <div className="break-all">Endpoint: {String(connectionStatus.neuronagent.details.endpoint)}</div>
                        )}
                        {connectionStatus.neuronagent.details.health_check && (
                          <div>Health: {String(connectionStatus.neuronagent.details.health_check)}</div>
                        )}
                      </div>
                    )}
                    {connectionStatus?.neuronagent?.error_message && (
                      <div className="text-xs text-red-400 mt-2 p-2 bg-red-500/10 rounded">
                        {connectionStatus.neuronagent.error_message}
                      </div>
                    )}
                    <button
                      onClick={() => testServiceConnection('neuronagent')}
                      disabled={testingService === 'neuronagent'}
                      className="w-full mt-2 btn btn-secondary text-xs py-1.5"
                    >
                      Test Connection
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
            )}

            {/* Modules Section - NeuronMCP */}
            {activeSection === 'modules' && activeSubSection === 'neuronmcp' && (
              <div className="card mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <SparklesIcon className="w-6 h-6 text-slate-400" />
              <h2 className="text-xl font-semibold text-slate-100">NeuronMCP Connection</h2>
            </div>
            <button
              onClick={loadConnectionStatus}
              disabled={statusLoading}
              className="btn btn-secondary text-sm"
            >
              {statusLoading ? 'Refreshing...' : 'Refresh Status'}
            </button>
          </div>

          {statusError && (
            <div className="mb-4 px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/50 text-red-300 text-sm">
              {statusError}
            </div>
          )}

          <div className="grid grid-cols-1 gap-4">
            {/* NeuronMCP Status */}
            <div className="border border-slate-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <SparklesIcon className="w-5 h-5 text-slate-400" />
                  <h3 className="font-semibold text-slate-100">NeuronMCP</h3>
                </div>
                {testingService === 'neuronmcp' ? (
                  <div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
                ) : (
                  <div className={`w-3 h-3 rounded-full ${getStatusBadge(connectionStatus?.neuronmcp).color}`} />
                )}
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-700 dark:text-slate-400">Status:</span>
                  <span className={`font-medium ${getStatusColor(connectionStatus?.neuronmcp)}`}>
                    {getStatusBadge(connectionStatus?.neuronmcp).text}
                  </span>
                </div>
                {connectionStatus?.neuronmcp?.details && (
                  <div className="text-xs text-slate-700 dark:text-slate-500 space-y-1">
                    {connectionStatus.neuronmcp.details.binary_path && (
                      <div className="break-all">Binary: {String(connectionStatus.neuronmcp.details.binary_path)}</div>
                    )}
                  </div>
                )}
                {connectionStatus?.neuronmcp?.error_message && (
                  <div className="text-xs text-red-400 mt-2 p-2 bg-red-500/10 rounded break-words">
                    {connectionStatus.neuronmcp.error_message}
                  </div>
                )}
                <button
                  onClick={() => testServiceConnection('neuronmcp')}
                  disabled={testingService === 'neuronmcp'}
                  className="w-full mt-2 btn btn-secondary text-xs py-1.5"
                >
                  Test Connection
                </button>
              </div>
            </div>
          </div>
        </div>
            )}

            {/* Appearance Section */}
            {activeSection === 'appearance' && (
              <div className="card mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <SparklesIcon className="w-6 h-6 text-purple-500" />
              <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-100">Appearance</h2>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between py-3 border-b border-slate-200 dark:border-slate-700">
              <div>
                <h3 className="font-medium text-gray-900 dark:text-slate-100">Theme</h3>
                <p className="text-sm text-gray-700 dark:text-slate-400 mt-1">
                  Choose between light and dark theme
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setTheme('light')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    theme === 'light'
                      ? 'bg-purple-500 text-white'
                      : 'bg-slate-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }`}
                >
                  Light
                </button>
                <button
                  onClick={() => setTheme('dark')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    theme === 'dark'
                      ? 'bg-purple-500 text-white'
                      : 'bg-slate-100 dark:bg-slate-800 text-gray-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }`}
                >
                  Dark
                </button>
              </div>
            </div>
          </div>
        </div>
            )}

            {/* Profiles Section */}
            {activeSection === 'profiles' && activeSubSection !== 'models' && (
              <div className="card mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <ServerIcon className="w-6 h-6 text-slate-400" />
              <h2 className="text-xl font-semibold text-slate-100">Connection Profiles</h2>
            </div>
            <p className="text-sm text-slate-400">
              Profiles are automatically created when users sign up. Admin users can view and manage all profiles.
            </p>
          </div>
          
          <div className="space-y-3">
            {profiles.map((profile) => (
              <div key={profile.id} className="border border-slate-700 rounded-lg p-4 hover:border-slate-700 transition-colors">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h3 className="font-semibold text-slate-100">{profile.name}</h3>
                      <span className="text-xs text-slate-400">({profile.id.slice(0, 8)}...)</span>
                      {profile.is_default && (
                        <span className="px-2 py-0.5 text-xs bg-blue-600 text-white rounded">Default</span>
                      )}
                      {profile.mcp_config && profile.mcp_config.command && (
                        <span className="px-2 py-0.5 text-xs bg-purple-600 text-white rounded flex items-center gap-1">
                          <SparklesIcon className="w-3 h-3" />
                          MCP
                        </span>
                      )}
                      {profile.agent_endpoint && (
                        <span className="px-2 py-0.5 text-xs bg-green-600 text-white rounded flex items-center gap-1">
                          <CpuIcon className="w-3 h-3" />
                          Agent
                        </span>
                      )}
                    </div>
                    <div className="space-y-1 text-sm text-slate-400">
                      <div className="flex items-center gap-2">
                        <DatabaseIcon className="w-4 h-4 flex-shrink-0" />
                        <span className="font-mono text-xs break-all">{profile.neurondb_dsn.split('@')[1] || profile.neurondb_dsn}</span>
                      </div>
                      {profile.agent_endpoint && (
                        <div className="flex items-center gap-2">
                          <ServerIcon className="w-4 h-4 flex-shrink-0" />
                          <span className="break-all">{profile.agent_endpoint}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button 
                      onClick={async () => {
                        try {
                          await profilesAPI.export(profile.id)
                          showSuccessToast(`Profile "${profile.name}" exported successfully`)
                        } catch (error: any) {
                          showErrorToast('Failed to export profile: ' + (error.response?.data?.error || error.message))
                        }
                      }}
                      className="p-2 text-slate-700 dark:text-slate-500 hover:text-green-600 dark:hover:text-green-400"
                      title="Export Profile"
                    >
                      <ArrowDownTrayIcon className="w-4 h-4" />
                    </button>
                    <button 
                      onClick={() => handleEditProfile(profile)}
                      className="p-2 text-slate-700 dark:text-slate-500 hover:text-blue-600 dark:hover:text-blue-400"
                      title="Edit Profile"
                    >
                      <PencilIcon className="w-4 h-4" />
                    </button>
                    <button 
                      onClick={async () => {
                        if (confirm(`Are you sure you want to delete profile "${profile.name}"?`)) {
                          try {
                            await profilesAPI.delete(profile.id)
                            await loadProfiles()
                            if (expandedProfile === profile.id) {
                              setExpandedProfile(null)
                              setEditingProfile(null)
                            }
                            showSuccessToast('Profile deleted successfully')
                          } catch (error: any) {
                            showErrorToast('Failed to delete profile: ' + (error.response?.data?.error || error.message))
                          }
                        }
                      }}
                      className="p-2 text-slate-700 dark:text-slate-500 hover:text-red-600 dark:hover:text-red-400"
                      title="Delete Profile"
                    >
                      <TrashIcon className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                
                {/* Expanded Configuration Sections */}
                {expandedProfile === profile.id && editingProfile?.id === profile.id && (
                  <div className="mt-4 pt-4 border-t border-slate-700 space-y-6">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-slate-400">Configuration for {profile.name}</span>
                      <button
                        onClick={() => {
                          setExpandedProfile(null)
                          setEditingProfile(null)
                        }}
                        className="text-sm text-slate-700 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200"
                      >
                        Close
                      </button>
                    </div>
                    
                    {/* MCP Configuration */}
                    <div>
                      <div className="flex items-center gap-2 mb-3">
                        <SparklesIcon className="w-5 h-5 text-slate-400" />
                        <h3 className="text-lg font-semibold text-slate-100">MCP Configuration</h3>
                      </div>
                      <p className="text-sm text-slate-700 dark:text-slate-400 mb-4">
                        Configure NeuronMCP settings for this profile. These settings will be saved and used when connecting to MCP tools.
                      </p>
                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm font-medium text-slate-200 mb-2">
                            Command
                          </label>
                          <input
                            type="text"
                            value={mcpCommand}
                            onChange={(e) => {
                              setMcpCommand(e.target.value)
                              setMcpTestPassed(false)
                              setMcpTestError(null)
                            }}
                            className="input w-full"
                            placeholder="neurondb-mcp"
                          />
                        </div>
                        
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="block text-sm font-medium text-slate-200">
                              Arguments
                            </label>
                            <button
                              onClick={addMcpArg}
                              className="text-xs text-blue-400 hover:text-blue-300"
                            >
                              + Add Argument
                            </button>
                          </div>
                          <div className="space-y-2">
                            {mcpArgs.map((arg, index) => (
                              <div key={index} className="flex gap-2">
                                <input
                                  type="text"
                                  value={arg}
                                  onChange={(e) => {
                                    updateMcpArg(index, e.target.value)
                                    setMcpTestPassed(false)
                                    setMcpTestError(null)
                                  }}
                                  className="input flex-1"
                                  placeholder="--config /path/to/config.json"
                                />
                                {mcpArgs.length > 1 && (
                                  <button
                                    onClick={() => removeMcpArg(index)}
                                    className="px-3 py-2 text-red-400 hover:text-red-300"
                                  >
                                    <TrashIcon className="w-4 h-4" />
                                  </button>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="block text-sm font-medium text-slate-200">
                              Environment Variables
                            </label>
                            <button
                              onClick={addMcpEnv}
                              className="text-xs text-blue-400 hover:text-blue-300"
                            >
                              + Add Variable
                            </button>
                          </div>
                          <div className="space-y-2">
                            {mcpEnv.map((env, index) => (
                              <div key={index} className="flex gap-2">
                                <input
                                  type="text"
                                  value={env.key}
                                  onChange={(e) => {
                                    updateMcpEnv(index, 'key', e.target.value)
                                    setMcpTestPassed(false)
                                    setMcpTestError(null)
                                  }}
                                  className="input flex-1"
                                  placeholder="Variable name"
                                />
                                <input
                                  type="text"
                                  value={env.value}
                                  onChange={(e) => {
                                    updateMcpEnv(index, 'value', e.target.value)
                                    setMcpTestPassed(false)
                                    setMcpTestError(null)
                                  }}
                                  className="input flex-1"
                                  placeholder="Variable value"
                                />
                                {mcpEnv.length > 1 && (
                                  <button
                                    onClick={() => removeMcpEnv(index)}
                                    className="px-3 py-2 text-red-400 hover:text-red-300"
                                  >
                                    <TrashIcon className="w-4 h-4" />
                                  </button>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-3 flex-wrap">
                          <button
                            onClick={handleTestMCP}
                            disabled={mcpTestLoading}
                            className="btn border border-slate-600 hover:bg-slate-700 disabled:opacity-50"
                          >
                            {mcpTestLoading ? 'Testing...' : 'Test MCP Configuration'}
                          </button>
                          {mcpTestPassed && (
                            <div className="flex items-center gap-2 text-green-400">
                              <CheckCircleIcon className="w-5 h-5" />
                              <span className="text-sm">Test passed</span>
                            </div>
                          )}
                          {mcpTestError && (
                            <div className="flex items-center gap-2 text-red-400">
                              <XCircleIcon className="w-5 h-5" />
                              <span className="text-sm">{mcpTestError}</span>
                            </div>
                          )}
                        </div>
                        <div className="flex items-center gap-3">
                          <button
                            onClick={handleSaveMCPConfig}
                            className="btn btn-primary"
                          >
                            Save MCP Configuration to Profile
                          </button>
                          <span className="text-xs text-slate-400">
                            Settings will be saved to {editingProfile?.name || 'this profile'}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Agent Configuration */}
                    <div>
                      <div className="flex items-center gap-2 mb-3">
                        <CpuIcon className="w-5 h-5 text-slate-400" />
                        <h3 className="text-lg font-semibold text-slate-100">Agent Configuration</h3>
                      </div>
                      <p className="text-sm text-slate-700 dark:text-slate-400 mb-4">
                        Configure NeuronAgent endpoint and API key for this profile. These settings will be saved and used when connecting to the agent service.
                      </p>
                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm font-medium text-slate-200 mb-2">
                            Agent Endpoint
                          </label>
                          <input
                            type="text"
                            value={agentEndpoint}
                            onChange={(e) => {
                              setAgentEndpoint(e.target.value)
                              setAgentTestPassed(false)
                              setAgentTestError(null)
                            }}
                            className="input w-full"
                            placeholder="http://localhost:8080"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-slate-200 mb-2">
                            Agent API Key
                          </label>
                          <div className="flex gap-2">
                            <input
                              type={showAgentApiKey ? 'text' : 'password'}
                              value={agentApiKey}
                              onChange={(e) => {
                                setAgentApiKey(e.target.value)
                                setAgentTestPassed(false)
                                setAgentTestError(null)
                              }}
                              className="input flex-1"
                              placeholder="Enter agent API key"
                            />
                            <button
                              onClick={() => setShowAgentApiKey(!showAgentApiKey)}
                              className="px-4 py-2 border border-slate-700 rounded-lg hover:bg-slate-800"
                            >
                              {showAgentApiKey ? 'Hide' : 'Show'}
                            </button>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-3 flex-wrap">
                          <button
                            onClick={handleTestAgent}
                            disabled={agentTestLoading}
                            className="btn border border-slate-600 hover:bg-slate-700 disabled:opacity-50"
                          >
                            {agentTestLoading ? 'Testing...' : 'Test Agent Configuration'}
                          </button>
                          {agentTestPassed && (
                            <div className="flex items-center gap-2 text-green-400">
                              <CheckCircleIcon className="w-5 h-5" />
                              <span className="text-sm">Test passed</span>
                            </div>
                          )}
                          {agentTestError && (
                            <div className="flex items-center gap-2 text-red-400">
                              <XCircleIcon className="w-5 h-5" />
                              <span className="text-sm">{agentTestError}</span>
                            </div>
                          )}
                        </div>
                        <div className="flex items-center gap-3">
                          <button
                            onClick={handleSaveAgentConfig}
                            className="btn btn-primary"
                          >
                            Save Agent Configuration to Profile
                          </button>
                          <span className="text-xs text-slate-400">
                            Settings will be saved to {editingProfile?.name || 'this profile'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
            
            {profiles.length === 0 && (
              <div className="text-center py-8 text-slate-400">
                <ServerIcon className="w-12 h-12 mx-auto mb-2 text-slate-500" />
                <p>No profiles found. Profiles are created automatically during signup.</p>
              </div>
            )}
          </div>
        </div>
            )}

            {/* Models Section - Submenu of Profiles */}
            {activeSection === 'profiles' && activeSubSection === 'models' && (
              <div className="card mb-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <SparklesIcon className="w-6 h-6 text-slate-400" />
                    <h2 className="text-xl font-semibold text-slate-100">Model Configurations</h2>
                  </div>
                  <button
                    onClick={() => setShowModelModal(true)}
                    className="btn btn-primary text-sm"
                  >
                    <PlusIcon className="w-4 h-4 mr-2" />
                    Add Model
                  </button>
                </div>

                <div className="mb-4">
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Select Profile
                  </label>
                  <select
                    value={selectedProfileForModels}
                    onChange={(e) => setSelectedProfileForModels(e.target.value)}
                    className="input w-full max-w-md"
                  >
                    {profiles.map((profile) => (
                      <option key={profile.id} value={profile.id}>
                        {profile.name} {profile.is_default && '(Default)'}
                      </option>
                    ))}
                  </select>
                </div>

                {selectedProfileForModels && (() => {
                  // Get all available models from all providers (excluding custom)
                  const allAvailableModels: Array<{
                    provider: string
                    providerName: string
                    modelName: string
                    displayName: string
                    requiresKey: boolean
                    requiresBaseUrl: boolean
                    defaultBaseUrl?: string
                    isFree: boolean
                  }> = []
                  
                  Object.entries(MODEL_PROVIDERS).forEach(([providerKey, provider]) => {
                    if (providerKey !== 'custom') {
                      provider.models.forEach((model) => {
                        allAvailableModels.push({
                          provider: providerKey,
                          providerName: provider.name,
                          modelName: model.name,
                          displayName: model.displayName,
                          requiresKey: provider.requiresKey,
                          requiresBaseUrl: provider.requiresBaseUrl,
                          defaultBaseUrl: (provider as any).defaultBaseUrl,
                          isFree: providerKey === 'ollama',
                        })
                      })
                    }
                  })
                  
                  // Create a map of configured models by provider+modelName
                  const configuredModelsMap = new Map<string, ModelConfig>()
                  const availableModelsSet = new Set<string>()
                  allAvailableModels.forEach((model) => {
                    availableModelsSet.add(`${model.provider}:${model.modelName}`)
                  })
                  
                  modelConfigs.forEach((config) => {
                    const key = `${config.model_provider}:${config.model_name}`
                    configuredModelsMap.set(key, config)
                    
                    // Add custom models (not in predefined list) to the display
                    if (!availableModelsSet.has(key)) {
                      const provider = MODEL_PROVIDERS[config.model_provider as keyof typeof MODEL_PROVIDERS]
                      if (provider) {
                        allAvailableModels.push({
                          provider: config.model_provider,
                          providerName: provider.name,
                          modelName: config.model_name,
                          displayName: config.model_name,
                          requiresKey: provider.requiresKey,
                          requiresBaseUrl: provider.requiresBaseUrl,
                          defaultBaseUrl: (provider as any).defaultBaseUrl,
                          isFree: config.is_free || false,
                        })
                        availableModelsSet.add(key)
                      }
                    }
                  })
                  
                  return (
                    <div>
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b border-slate-700">
                              <th className="text-left py-3 px-4 text-sm font-semibold text-slate-300">Model</th>
                              <th className="text-left py-3 px-4 text-sm font-semibold text-slate-300">Provider</th>
                              <th className="text-left py-3 px-4 text-sm font-semibold text-slate-300">API Key</th>
                              <th className="text-center py-3 px-4 text-sm font-semibold text-slate-300">Status</th>
                              <th className="text-center py-3 px-4 text-sm font-semibold text-slate-300">Default</th>
                              <th className="text-center py-3 px-4 text-sm font-semibold text-slate-300">Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {allAvailableModels.map((availableModel) => {
                              const configKey = `${availableModel.provider}:${availableModel.modelName}`
                              const config = configuredModelsMap.get(configKey)
                              const isConfigured = !!config
                              const isEnabled = config ? isModelEnabled(config) : false
                              const isEditingApiKey = config && editingApiKeyId === config.id
                              
                              return (
                                <tr 
                                  key={configKey} 
                                  className={`border-b border-slate-800 hover:bg-slate-800/30 transition-colors ${
                                    !isConfigured ? 'opacity-50' : !isEnabled ? 'opacity-60' : ''
                                  }`}
                                >
                                  {/* Model Name */}
                                  <td className="py-3 px-4">
                                    <div className="flex items-center gap-2">
                                      <span className="font-medium text-slate-100">{availableModel.displayName}</span>
                                      {availableModel.isFree && (
                                        <span className="px-1.5 py-0.5 text-xs bg-green-600 text-white rounded">Free</span>
                                      )}
                                      {!isConfigured && (
                                        <span className="px-1.5 py-0.5 text-xs bg-slate-600 text-slate-300 rounded">Not Configured</span>
                                      )}
                                    </div>
                                    <div className="text-xs text-slate-700 dark:text-slate-500 mt-1">{availableModel.modelName}</div>
                                    {config?.base_url && (
                                      <div className="text-xs text-slate-700 dark:text-slate-500 mt-1 break-all">{config.base_url}</div>
                                    )}
                                  </td>
                                  
                                  {/* Provider */}
                                  <td className="py-3 px-4">
                                    <span className="text-sm text-slate-300">{availableModel.providerName}</span>
                                  </td>
                                  
                                  {/* API Key */}
                                  <td className="py-3 px-4">
                                    {!isConfigured ? (
                                      <span className="text-sm text-slate-500">-</span>
                                    ) : availableModel.requiresKey ? (
                                      isEditingApiKey ? (
                                        <div className="flex items-center gap-2">
                                          <input
                                            type="password"
                                            value={inlineApiKey}
                                            onChange={(e) => setInlineApiKey(e.target.value)}
                                            placeholder="Enter API key"
                                            className="input text-sm w-48"
                                            autoFocus
                                          />
                                          <button
                                            onClick={() => handleSaveInlineApiKey(config!.id)}
                                            disabled={savingApiKey === config!.id}
                                            className="btn btn-primary text-xs px-2 py-1"
                                          >
                                            {savingApiKey === config!.id ? '...' : 'Save'}
                                          </button>
                                          <button
                                            onClick={() => {
                                              setEditingApiKeyId(null)
                                              setInlineApiKey('')
                                            }}
                                            className="btn btn-secondary text-xs px-2 py-1"
                                          >
                                            Cancel
                                          </button>
                                        </div>
                                      ) : (
                                        <div className="flex items-center gap-2">
                                          <span className="text-sm text-slate-400">
                                            {config.api_key ? '••••••••••••' : 'Not set'}
                                          </span>
                                          <button
                                            onClick={() => {
                                              setEditingApiKeyId(config.id)
                                              setInlineApiKey(config.api_key || '')
                                            }}
                                            className="text-xs text-blue-400 hover:text-blue-300"
                                          >
                                            {config.api_key ? 'Change' : 'Set'}
                                          </button>
                                        </div>
                                      )
                                    ) : (
                                      <span className="text-sm text-slate-500">N/A</span>
                                    )}
                                  </td>
                                  
                                  {/* Enable/Disable Status */}
                                  <td className="py-3 px-4 text-center">
                                    {!isConfigured ? (
                                      <span className="text-sm text-slate-500">-</span>
                                    ) : (
                                      <button
                                        onClick={() => handleToggleEnabled(config!.id, isEnabled)}
                                        disabled={togglingEnabled === config!.id}
                                        className={`px-3 py-1 text-xs rounded transition-colors ${
                                          isEnabled
                                            ? 'bg-green-600/20 text-green-400 hover:bg-green-600/30'
                                            : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                                        }`}
                                        title={isEnabled ? 'Disable model' : 'Enable model'}
                                      >
                                        {togglingEnabled === config!.id ? (
                                          <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin mx-auto" />
                                        ) : (
                                          isEnabled ? 'Enabled' : 'Disabled'
                                        )}
                                      </button>
                                    )}
                                  </td>
                                  
                                  {/* Default */}
                                  <td className="py-3 px-4 text-center">
                                    {!isConfigured ? (
                                      <span className="text-sm text-slate-500">-</span>
                                    ) : config.is_default ? (
                                      <span className="px-2 py-1 text-xs bg-blue-600 text-white rounded">Default</span>
                                    ) : (
                                      <button
                                        onClick={() => handleSetDefault(config.id)}
                                        disabled={settingDefault === config.id}
                                        className="px-3 py-1 text-xs bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 rounded transition-colors"
                                        title="Set as default model"
                                      >
                                        {settingDefault === config.id ? (
                                          <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin mx-auto" />
                                        ) : (
                                          'Set Default'
                                        )}
                                      </button>
                                    )}
                                  </td>
                                  
                                  {/* Actions */}
                                  <td className="py-3 px-4">
                                    <div className="flex items-center justify-center gap-2">
                                      {!isConfigured ? (
                                        <button
                                          onClick={() => {
                                            setNewModelProvider(availableModel.provider)
                                            setNewModelName(availableModel.modelName)
                                            setNewModelBaseUrl(availableModel.defaultBaseUrl || '')
                                            setNewModelApiKey('')
                                            setNewModelIsDefault(false)
                                            setShowModelModal(true)
                                          }}
                                          className="btn btn-primary text-xs px-3 py-1"
                                          title="Add this model"
                                        >
                                          <PlusIcon className="w-3 h-3 mr-1 inline" />
                                          Add
                                        </button>
                                      ) : (
                                        <>
                                          <button
                                            onClick={() => handleEditModel(config)}
                                            className="p-1.5 text-slate-700 dark:text-slate-500 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                                            title="Edit model configuration"
                                          >
                                            <PencilIcon className="w-4 h-4" />
                                          </button>
                                          <button
                                            onClick={() => handleDeleteModel(config.id)}
                                            className="p-1.5 text-slate-700 dark:text-slate-500 hover:text-red-600 dark:hover:text-red-400 transition-colors"
                                            title="Delete model"
                                          >
                                            <TrashIcon className="w-4 h-4" />
                                          </button>
                                        </>
                                      )}
                                    </div>
                                  </td>
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )
                })()}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model Configuration Modal - Always available */}
      {showModelModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-900 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-slate-800">
              <h2 className="text-2xl font-bold text-slate-100">
                {editingModel ? 'Edit Model Configuration' : 'Add Model Configuration'}
              </h2>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Provider *
                </label>
                <select
                  value={newModelProvider}
                  onChange={(e) => {
                    setNewModelProvider(e.target.value)
                    setNewModelName('')
                    const provider = MODEL_PROVIDERS[e.target.value as keyof typeof MODEL_PROVIDERS]
                    if ((provider as any)?.defaultBaseUrl) {
                      setNewModelBaseUrl((provider as any).defaultBaseUrl)
                    } else {
                      setNewModelBaseUrl('')
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
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Model *
                </label>
                {newModelProvider === 'custom' ? (
                  <input
                    type="text"
                    value={newModelName}
                    onChange={(e) => setNewModelName(e.target.value)}
                    className="input w-full"
                    placeholder="Enter custom model name"
                  />
                ) : (
                  <div className="space-y-2 max-h-64 overflow-y-auto border border-slate-700 rounded-lg p-3 bg-slate-800/50">
                    {MODEL_PROVIDERS[newModelProvider as keyof typeof MODEL_PROVIDERS]?.models.map((model) => (
                      <button
                        key={model.name}
                        type="button"
                        onClick={() => setNewModelName(model.name)}
                        className={`w-full text-left px-4 py-2 rounded-lg border transition-colors ${
                          newModelName === model.name
                            ? 'border-purple-500 bg-purple-500/20 text-purple-300'
                            : 'border-slate-600 bg-slate-800/50 text-slate-300 hover:border-slate-500 hover:bg-slate-700/50'
                        }`}
                      >
                        <div className="font-medium">{model.displayName}</div>
                        <div className="text-xs text-slate-700 dark:text-slate-500 mt-0.5">{model.name}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {MODEL_PROVIDERS[newModelProvider as keyof typeof MODEL_PROVIDERS]?.requiresKey && (
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    API Key *
                  </label>
                  <div className="flex gap-2">
                    <input
                      type={showModelApiKey ? 'text' : 'password'}
                      value={newModelApiKey}
                      onChange={(e) => setNewModelApiKey(e.target.value)}
                      className="input flex-1"
                      placeholder="Enter API key"
                    />
                    <button
                      onClick={() => setShowModelApiKey(!showModelApiKey)}
                      className="px-4 py-2 border border-slate-700 rounded-lg hover:bg-slate-800"
                    >
                      {showModelApiKey ? 'Hide' : 'Show'}
                    </button>
                  </div>
                </div>
              )}

              {MODEL_PROVIDERS[newModelProvider as keyof typeof MODEL_PROVIDERS]?.requiresBaseUrl && (
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Base URL *
                  </label>
                  <input
                    type="text"
                    value={newModelBaseUrl}
                    onChange={(e) => setNewModelBaseUrl(e.target.value)}
                    className="input w-full"
                    placeholder="http://localhost:11434"
                  />
                </div>
              )}

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="isDefaultModel"
                  checked={newModelIsDefault}
                  onChange={(e) => setNewModelIsDefault(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-700 bg-slate-800"
                />
                <label htmlFor="isDefaultModel" className="text-sm text-slate-300">
                  Set as default model for this profile
                </label>
              </div>
            </div>

            <div className="p-6 border-t border-slate-800 flex justify-end gap-3">
              <button
                onClick={() => {
                  setShowModelModal(false)
                  setEditingModel(null)
                  setNewModelProvider('openai')
                  setNewModelName('')
                  setNewModelApiKey('')
                  setNewModelBaseUrl('')
                  setNewModelIsDefault(false)
                }}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={editingModel ? handleUpdateModel : handleCreateModel}
                className="btn btn-primary"
              >
                {editingModel ? 'Update' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

