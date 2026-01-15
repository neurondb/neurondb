'use client'

import { useState, useEffect } from 'react'
import { agentAPI, profilesAPI, templateAPI, type Profile, type CreateAgentRequest, type Template } from '@/lib/api'
import { PlusIcon, MagnifyingGlassIcon } from '@/components/Icons'
import { showSuccessToast, showErrorToast } from '@/lib/errors'

const HARDCODED_TEMPLATES: Template[] = [
  {
    id: 'customer-support',
    name: 'Customer Support',
    description: 'Multi-tier customer support agent with escalation workflow',
    category: 'support',
    configuration: {},
  },
  {
    id: 'data-pipeline',
    name: 'Data Pipeline',
    description: 'Data ingestion, processing, and analysis pipeline workflow',
    category: 'data',
    configuration: {},
  },
  {
    id: 'research-assistant',
    name: 'Research Assistant',
    description: 'Multi-source research assistant with web search and document analysis',
    category: 'research',
    configuration: {},
  },
  {
    id: 'document-qa',
    name: 'Document Q&A',
    description: 'RAG-based document Q&A agent with vector search',
    category: 'rag',
    configuration: {},
  },
  {
    id: 'report-generator',
    name: 'Report Generator',
    description: 'Automated report generation workflow with data analysis and visualization',
    category: 'analytics',
    configuration: {},
  },
]

export default function TemplateGallery({ onSelect, onCancel }: { onSelect?: () => void; onCancel?: () => void }) {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null)
  const [customizing, setCustomizing] = useState(false)
  const [loading, setLoading] = useState(false)
  const [agentName, setAgentName] = useState('')
  const [templates, setTemplates] = useState<Template[]>([])
  const [loadingTemplates, setLoadingTemplates] = useState(false)

  useEffect(() => {
    loadProfiles()
    loadTemplates()
  }, [])

  const loadProfiles = async () => {
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
  }

  const loadTemplates = async () => {
    setLoadingTemplates(true)
    try {
      const response = await templateAPI.list()
      if (response.data && response.data.length > 0) {
        setTemplates(response.data)
      } else {
        // Fallback to hardcoded templates if API returns empty
        setTemplates(HARDCODED_TEMPLATES)
      }
    } catch (error: any) {
      // Fallback to hardcoded templates on error
      setTemplates(HARDCODED_TEMPLATES)
    } finally {
      setLoadingTemplates(false)
    }
  }

  const filteredTemplates = templates.filter(
    (template) =>
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.category.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleDeploy = async (template: Template) => {
    if (!selectedProfile) {
      showErrorToast('Please select a profile')
      return
    }

    const defaultName = `${template.name.toLowerCase().replace(/\s+/g, '-')}-agent`
    const name = agentName.trim() || defaultName

    if (!name) {
      showErrorToast('Agent name is required')
      return
    }

    setLoading(true)
    try {
      // Try to deploy from template API first
      try {
        const response = await templateAPI.deploy(selectedProfile, template.id, name)
        showSuccessToast('Agent created successfully from template')
        if (onSelect) onSelect()
        return
      } catch (templateError: any) {
        // If template deployment fails, fall back to manual creation
        if (templateError.response?.status !== 404) {
          throw templateError
        }
      }

      // Fallback: Map template to agent configuration and create manually
      const agentConfig: CreateAgentRequest = getTemplateConfig(template)
      const response = await agentAPI.createAgent(selectedProfile, {
        ...agentConfig,
        name,
      })

      showSuccessToast('Agent created successfully')
      if (onSelect) onSelect()
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || error.response?.data?.error || error.message || 'Failed to create agent'
      showErrorToast('Failed to create agent: ' + errorMessage)
    } finally {
      setLoading(false)
    }
  }

  const getTemplateConfig = (template: Template): CreateAgentRequest => {
    // If template has configuration from API, use it
    if (template.configuration && typeof template.configuration === 'object') {
      const config = template.configuration
      return {
        name: '',
        description: config.description || template.description || '',
        system_prompt: config.system_prompt || config.systemPrompt || '',
        model_name: config.model?.name || config.model_name || 'gpt-4',
        enabled_tools: config.tools || config.enabled_tools || [],
        config: config.config || {},
      }
    }

    // Fallback to hardcoded configs
    const configs: Record<string, CreateAgentRequest> = {
      'customer-support': {
        name: '',
        description: 'Multi-tier customer support agent with escalation workflow',
        system_prompt: 'You are a customer support agent. Help customers with their questions and issues.',
        model_name: 'gpt-4',
        enabled_tools: ['sql', 'http', 'browser'],
        config: { temperature: 0.7, max_tokens: 1500 },
      },
      'data-pipeline': {
        name: '',
        description: 'Data ingestion, processing, and analysis pipeline workflow',
        system_prompt: 'You are a data pipeline agent. Process data, run analysis, and generate reports.',
        model_name: 'gpt-4',
        enabled_tools: ['sql', 'code'],
        config: { temperature: 0.3, max_tokens: 2000 },
      },
      'research-assistant': {
        name: '',
        description: 'Multi-source research assistant with web search and document analysis',
        system_prompt: 'You are a research assistant. Gather information from multiple sources and provide comprehensive research summaries.',
        model_name: 'gpt-4',
        enabled_tools: ['http', 'browser', 'sql'],
        config: { temperature: 0.5, max_tokens: 2000 },
      },
      'document-qa': {
        name: '',
        description: 'RAG-based document Q&A agent with vector search',
        system_prompt: 'You are a document Q&A assistant. Answer questions based on provided document context.',
        model_name: 'gpt-4',
        enabled_tools: ['sql'],
        config: { temperature: 0.3, max_tokens: 1500 },
      },
      'report-generator': {
        name: '',
        description: 'Automated report generation workflow with data analysis and visualization',
        system_prompt: 'You are a report generation agent. Analyze data, generate insights, and compile comprehensive reports.',
        model_name: 'gpt-4',
        enabled_tools: ['sql', 'code', 'http'],
        config: { temperature: 0.4, max_tokens: 2500 },
      },
    }

    return configs[template.id] || {
      name: '',
      description: template.description,
      model_name: 'gpt-4',
      enabled_tools: [],
    }
  }

  if (customizing && selectedTemplate) {
    return (
      <div className="h-full overflow-auto bg-transparent p-6">
        <div className="max-w-4xl mx-auto">
          <div className="mb-6">
            <button
              onClick={() => {
                setCustomizing(false)
                setSelectedTemplate(null)
                setAgentName('')
              }}
              className="text-blue-600 dark:text-blue-400 hover:underline mb-4"
            >
              ← Back to templates
            </button>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-slate-100">Customize Template</h2>
            <p className="text-gray-600 dark:text-slate-400 mt-2">{selectedTemplate.description}</p>
          </div>

          <div className="card space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                Agent Name *
              </label>
              <input
                type="text"
                value={agentName}
                onChange={(e) => setAgentName(e.target.value)}
                className="input"
                placeholder={`${selectedTemplate.name.toLowerCase().replace(/\s+/g, '-')}-agent`}
              />
            </div>

            <div className="flex gap-3 pt-4">
              <button
                onClick={() => handleDeploy(selectedTemplate)}
                disabled={loading}
                className="btn btn-primary flex-1"
              >
                {loading ? 'Creating...' : 'Create Agent'}
              </button>
              <button onClick={() => setCustomizing(false)} className="btn btn-secondary">
                Cancel
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto bg-transparent p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-100 mb-2">Template Gallery</h1>
            <p className="text-gray-700 dark:text-slate-400">Choose a template to get started quickly</p>
          </div>
          {onCancel && (
            <button onClick={onCancel} className="btn btn-secondary">
              Cancel
            </button>
          )}
        </div>

        {/* Profile Selector */}
        {profiles.length > 0 && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">Profile</label>
            <select
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value)}
              className="input max-w-md"
            >
              {profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name} {profile.is_default && '(Default)'}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Search */}
        <div className="mb-6">
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search templates..."
              className="input pl-10"
            />
          </div>
        </div>

        {/* Template Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredTemplates.map((template) => (
            <div
              key={template.id}
              className="card hover:shadow-xl transition-shadow cursor-pointer"
              onClick={() => {
                setSelectedTemplate(template)
                setCustomizing(true)
                setAgentName('')
              }}
            >
              <div className="mb-4">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-slate-100 mb-2">{template.name}</h3>
                <p className="text-sm text-gray-600 dark:text-slate-400">{template.description}</p>
              </div>
              <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-slate-700">
                <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded">
                  {template.category}
                </span>
                <button className="text-blue-600 dark:text-blue-400 hover:underline text-sm">Use Template →</button>
              </div>
            </div>
          ))}
        </div>

        {filteredTemplates.length === 0 && (
          <div className="card text-center py-12">
            <p className="text-gray-600 dark:text-slate-400">No templates found matching &quot;{searchQuery}&quot;</p>
          </div>
        )}
      </div>
    </div>
  )
}



