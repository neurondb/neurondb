import axios, { AxiosError } from 'axios'
import { getErrorMessage, showErrorToast } from './errors'
import { logger } from './logger'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081/api/v1'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
  withCredentials: true, // Enable cookies for session-based auth
})

// No longer need to add Authorization header - cookies are sent automatically
// Keep for backward compatibility with JWT if needed
api.interceptors.request.use((config) => {
  // Only add Authorization header if we have a token in localStorage (JWT fallback)
  const token = localStorage.getItem('neurondesk_auth_token')
  if (token && !config.headers.Authorization) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Add error response interceptor for better error messages
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    logger.error('API Error', error, {
      status: error.response?.status,
      statusText: error.response?.statusText,
      code: error.code,
      hasResponse: !!error.response,
      hasRequest: !!error.request
    })

    // Enhance error with better message
    if (error.response) {
      const data = error.response.data as any
      // Store user-friendly error message in error object
      if (data?.error) {
        error.message = data.error
      } else if (data?.message) {
        error.message = data.message
      } else if (error.response.status === 401) {
        error.message = 'Authentication required. Your session may have expired. Please log in again.'
        // Show toast for auth errors
        if (typeof window !== 'undefined' && !window.location.pathname.includes('/login')) {
          showErrorToast(error, error.message)
          // Try to refresh session first
          api.post('/api/v1/auth/refresh').catch(() => {
            // Refresh failed, clear any stored tokens and redirect
            localStorage.removeItem('neurondesk_auth_token')
            setTimeout(() => {
              window.location.href = '/login'
            }, 1000)
          })
        }
      } else if (error.response.status === 403) {
        error.message = 'Access denied. You do not have permission to perform this action.'
        // Show toast for permission errors
        if (typeof window !== 'undefined') {
          showErrorToast(error, error.message)
        }
      } else if (error.response.status >= 500) {
        error.message = 'Server error. Please try again later.'
        // Show toast for server errors
        if (typeof window !== 'undefined') {
          showErrorToast(error, error.message)
        }
      }
    } else if (error.request) {
      // Network error - check if it might be CORS or auth issue
      const token = localStorage.getItem('neurondesk_auth_token')
      if (!token) {
        error.message = 'Not authenticated. Please log in first.'
        if (typeof window !== 'undefined' && !window.location.pathname.includes('/login')) {
          setTimeout(() => {
            window.location.href = '/login'
          }, 1000)
        }
      } else {
        // Check if this might be a preflight CORS failure (which can hide 401 errors)
        // Try to provide more helpful error message
        if (error.code === 'ECONNREFUSED' || error.message.includes('ECONNREFUSED')) {
          error.message = 'Cannot connect to server. Please check if the API server is running on http://localhost:8081'
        } else if (error.code === 'ETIMEDOUT' || error.code === 'ECONNABORTED') {
          error.message = 'Request timeout. The server took too long to respond.'
        } else if (error.code === 'ERR_NETWORK' || error.message.includes('Network Error')) {
          // Network error could be CORS blocking a 401 response
          error.message = 'Network error. This might be an authentication issue. Please try: 1) Log out and log back in, 2) Check if API server is running on http://localhost:8081, 3) Check browser console (F12) for detailed errors.'
        } else {
          error.message = 'Network error. Please check your connection and try again.'
        }
      }
    } else {
      // No response and no request - likely a configuration issue
      error.message = 'Request failed. Please check your configuration and try again.'
    }
    
    return Promise.reject(error)
  }
)

export interface Profile {
  id: string
  name: string
  user_id: string
  profile_username?: string
  mcp_config: Record<string, any>
  neurondb_dsn: string
  agent_endpoint?: string
  agent_api_key?: string
  default_collection?: string
  is_default?: boolean
  created_at: string
  updated_at: string
}

export interface ToolDefinition {
  name: string
  description: string
  inputSchema: Record<string, any>
}

export interface ToolResult {
  content: Array<{ type: string; text: string }>
  isError?: boolean
  metadata?: any
}

export interface CollectionInfo {
  name: string
  schema: string
  vector_col?: string
  indexes?: Array<{
    name: string
    type: string
    definition: string
    size?: string
  }>
  row_count?: number
}

export interface SearchRequest {
  collection: string
  schema?: string
  query_vector: number[]
  query_text?: string
  limit?: number
  filter?: Record<string, any>
  distance_type?: 'l2' | 'cosine' | 'inner_product'
}

export interface SearchResult {
  id: any
  score: number
  distance: number
  data: Record<string, any>
}

export interface ModelConfig {
  id: string
  profile_id: string
  model_provider: string // 'openai', 'anthropic', 'google', 'ollama', 'custom'
  model_name: string
  api_key?: string
  base_url?: string
  is_default: boolean
  is_free: boolean
  metadata?: Record<string, any>
  created_at: string
  updated_at: string
}

// Profiles API
export const profilesAPI = {
  list: () => api.get<Profile[]>('/profiles'),
  get: (id: string) => api.get<Profile>(`/profiles/${id}`),
  create: (profile: Partial<Profile>) => api.post<Profile>('/profiles', profile),
  update: (id: string, profile: Partial<Profile>) => api.put<Profile>(`/profiles/${id}`, profile),
  delete: (id: string) => api.delete(`/profiles/${id}`),
  export: async (id: string) => {
    const response = await api.get(`/profiles/${id}/export`, { responseType: 'blob' })
    const blob = new Blob([response.data], { type: 'application/json' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `profile-${id}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  },
  import: (data: any) => api.post<Profile>('/profiles/import', data),
}

// MCP API
export interface MCPThread {
  id: string
  profile_id: string
  title: string
  created_at: string
  updated_at: string
}

export interface MCPMessage {
  id: string
  thread_id: string
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  tool_name?: string
  data?: any
  created_at: string
}

export interface MCPThreadWithMessages extends MCPThread {
  messages: MCPMessage[]
}

export const mcpAPI = {
  listConnections: () => api.get('/mcp/connections'),
  listTools: (profileId: string) => api.get<{ tools: ToolDefinition[] }>(`/profiles/${profileId}/mcp/tools`),
  callTool: (profileId: string, name: string, args: Record<string, any>) =>
    api.post<ToolResult>(`/profiles/${profileId}/mcp/tools/call`, { name, arguments: args }),
  testConfig: (config: { command: string; args?: string[]; env?: Record<string, string> }) =>
    api.post('/mcp/test', config),
  // Chat thread endpoints
  listThreads: (profileId: string) => api.get<MCPThread[]>(`/profiles/${profileId}/mcp/threads`),
  createThread: (profileId: string, title?: string) => api.post<MCPThread>(`/profiles/${profileId}/mcp/threads`, { title: title || 'New chat' }),
  getThread: (profileId: string, threadId: string) => api.get<MCPThreadWithMessages>(`/profiles/${profileId}/mcp/threads/${threadId}`),
  updateThread: (profileId: string, threadId: string, title: string) => api.put<MCPThread>(`/profiles/${profileId}/mcp/threads/${threadId}`, { title }),
  deleteThread: (profileId: string, threadId: string) => api.delete(`/profiles/${profileId}/mcp/threads/${threadId}`),
  addMessage: (profileId: string, threadId: string, role: string, content: string, toolName?: string, data?: any) =>
    api.post<MCPMessage>(`/profiles/${profileId}/mcp/threads/${threadId}/messages`, { role, content, tool_name: toolName, data }),
}

// NeuronDB API
export const neurondbAPI = {
  listCollections: (profileId: string) => api.get<CollectionInfo[]>(`/profiles/${profileId}/neurondb/collections`),
  search: (profileId: string, request: SearchRequest) =>
    api.post<SearchResult[]>(`/profiles/${profileId}/neurondb/search`, request),
  executeSQL: (profileId: string, query: string) =>
    api.post(`/profiles/${profileId}/neurondb/sql`, { query }),
  executeSQLFull: (profileId: string, query: string) =>
    api.post(`/profiles/${profileId}/neurondb/sql/execute`, { query }),
  ingestDataset: (profileId: string, data: { 
    source_type: string; 
    source_path?: string; 
    source?: string;
    collection?: string; 
    format?: string;
    table_name?: string;
    auto_embed?: boolean;
    embedding_model?: string;
    create_index?: boolean;
  }) =>
    api.post(`/profiles/${profileId}/neurondb/datasets/ingest`, data),
  testConnection: (data: { dsn: string }) =>
    api.post('/neurondb/test-connection', data),
  loadDemoDataset: (data: { dsn: string }) =>
    api.post('/neurondb/datasets/load-demo', data),
}

// Agent API
export interface Agent {
  id: string
  name: string
  description?: string
  system_prompt?: string
  model_name?: string
  enabled_tools?: string[]
  config?: Record<string, any>
  created_at?: string
}

export interface Model {
  name: string
  display_name: string
  provider: string
  description: string
}

export interface CreateAgentRequest {
  name: string
  description?: string
  system_prompt?: string
  model_name: string
  enabled_tools?: string[]
  config?: Record<string, any>
}

export interface Session {
  id: string
  agent_id: string
  external_user_id?: string
  metadata?: Record<string, any>
  created_at?: string
}

export interface Message {
  id: string
  session_id: string
  role: string
  content: string
  metadata?: Record<string, any>
  created_at?: string
}

export const agentAPI = {
  listAgents: (profileId: string) => api.get<Agent[]>(`/profiles/${profileId}/agent/agents`),
  getAgent: (profileId: string, agentId: string) => api.get<Agent>(`/profiles/${profileId}/agent/agents/${agentId}`),
  createAgent: (profileId: string, request: CreateAgentRequest) =>
    api.post<Agent>(`/profiles/${profileId}/agent/agents`, request),
  updateAgent: (profileId: string, agentId: string, request: CreateAgentRequest) =>
    api.put<Agent>(`/profiles/${profileId}/agent/agents/${agentId}`, request),
  deleteAgent: (profileId: string, agentId: string) =>
    api.delete(`/profiles/${profileId}/agent/agents/${agentId}`),
  exportAgent: async (profileId: string, agentId: string) => {
    const response = await api.get(`/profiles/${profileId}/agent/agents/${agentId}/export`, { responseType: 'blob' })
    const blob = new Blob([response.data], { type: 'application/json' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `agent-${agentId}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  },
  importAgent: (profileId: string, data: any) =>
    api.post<Agent>(`/profiles/${profileId}/agent/agents/import`, data),
  listModels: (profileId: string) => api.get<{ models: Model[] }>(`/profiles/${profileId}/agent/models`),
  createSession: (profileId: string, agentId: string, externalUserId?: string) =>
    api.post<Session>(`/profiles/${profileId}/agent/sessions`, { agent_id: agentId, external_user_id: externalUserId }),
  listSessions: (profileId: string, agentId: string) =>
    api.get<Session[]>(`/profiles/${profileId}/agent/agents/${agentId}/sessions`),
  getSession: (profileId: string, sessionId: string) =>
    api.get<Session>(`/profiles/${profileId}/agent/sessions/${sessionId}`),
  sendMessage: (profileId: string, sessionId: string, content: string, stream?: boolean) =>
    api.post<Message>(`/profiles/${profileId}/agent/sessions/${sessionId}/messages`, {
      role: 'user',
      content,
      stream: stream || false,
    }),
  getMessages: (profileId: string, sessionId: string) =>
    api.get<Message[]>(`/profiles/${profileId}/agent/sessions/${sessionId}/messages`),
  testConfig: (config: { endpoint: string; api_key: string }) =>
    api.post('/agent/test', config),
  health: (endpoint: string, apiKey: string) =>
    api.get('/agent/health', { headers: { 'X-Agent-Endpoint': endpoint, 'X-Agent-API-Key': apiKey } }),
}

// Model Config API
export const modelConfigAPI = {
  list: (profileId: string, includeApiKey?: boolean) => 
    api.get<ModelConfig[]>(`/profiles/${profileId}/models`, { params: { include_api_key: includeApiKey } }),
  get: (profileId: string, id: string) => 
    api.get<ModelConfig>(`/profiles/${profileId}/models/${id}`),
  create: (profileId: string, config: Partial<ModelConfig>) => 
    api.post<ModelConfig>(`/profiles/${profileId}/models`, config),
  update: (profileId: string, id: string, config: Partial<ModelConfig>) => 
    api.put<ModelConfig>(`/profiles/${profileId}/models/${id}`, config),
  delete: (profileId: string, id: string) => 
    api.delete(`/profiles/${profileId}/models/${id}`),
  getDefault: (profileId: string) => 
    api.get<ModelConfig>(`/profiles/${profileId}/models/default`),
  setDefault: (profileId: string, id: string) => 
    api.post(`/profiles/${profileId}/models/${id}/set-default`),
}

// Factory API
export interface FactoryStatus {
  os: {
    type: string
    distro: string
    version: string
    arch: string
  }
  docker: {
    available: boolean
    version?: string
  }
  neurondb: ComponentStatus
  neuronagent: ComponentStatus
  neuronmcp: ComponentStatus
  install_commands: {
    docker?: string[]
    deb?: string[]
    rpm?: string[]
    macpkg?: string[]
  }
}

export interface ComponentStatus {
  installed: boolean
  running: boolean
  reachable: boolean
  status: string
  error_message?: string
  details?: Record<string, any>
}

export interface SetupState {
  setup_complete: boolean
}

export const factoryAPI = {
  getStatus: () => api.get<FactoryStatus>('/factory/status'),
  getSetupState: () => api.get<SetupState>('/factory/setup-state'),
  setSetupState: (completed: boolean) => 
    api.post<SetupState>('/factory/setup-state', { completed }),
}

// Database Test API (public endpoint, no auth required)
const databaseTestApi = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
  withCredentials: false, // Don't send credentials for database test
})

databaseTestApi.interceptors.request.use(
  (config) => {
    logger.debug('Database test request', {
      method: config.method,
      url: config.url,
      baseURL: config.baseURL,
      fullURL: (config.baseURL || '') + (config.url || ''),
    })
    return config
  },
  (error) => {
    logger.error('Database test request error', error)
    return Promise.reject(error)
  }
)

// Add error response interceptor for better error messages
databaseTestApi.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response) {
      const data = error.response.data as any
      if (data?.error) {
        error.message = data.error
      } else if (data?.message) {
        error.message = data.message
      } else if (error.response.status >= 500) {
        error.message = 'Server error. Please try again later.'
      }
    } else if (error.request) {
      // Network error - request was made but no response received
      if (error.code === 'ECONNREFUSED' || error.message.includes('ECONNREFUSED')) {
        error.message = 'Cannot connect to server. Please check if the API server is running on ' + API_BASE_URL
      } else if (error.code === 'ETIMEDOUT' || error.code === 'ECONNABORTED') {
        error.message = 'Request timeout. The server took too long to respond.'
      } else if (error.code === 'ERR_NETWORK' || error.message.includes('Network Error')) {
        error.message = 'Network error. Please check if the API server is running on ' + API_BASE_URL + '. If it is, check browser console for CORS errors.'
      } else {
        error.message = 'Network error: ' + (error.message || error.code || 'Unknown error') + '. Please check if the API server is running on ' + API_BASE_URL
      }
    }
    return Promise.reject(error)
  }
)

export interface DatabaseTestRequest {
  host: string
  port: string
  database: string
  user: string
  password: string
}

export interface DatabaseTestResponse {
  success: boolean
  message: string
  schema_exists: boolean
  missing_tables?: string[]
  dsn?: string
}

export const databaseTestAPI = {
  test: (request: DatabaseTestRequest) => 
    databaseTestApi.post<DatabaseTestResponse>('/database/test', request),
}

export interface Template {
  id: string
  name: string
  description: string
  category: string
  configuration: any
  workflow?: any
}

export const templateAPI = {
  list: () => api.get<Template[]>('/templates'),
  get: (id: string) => api.get<Template>(`/templates/${id}`),
  deploy: (profileId: string, templateId: string, name: string) =>
    api.post<Agent>(`/profiles/${profileId}/templates/${templateId}/deploy`, { name }),
}

export interface RequestLog {
  id: string
  profile_id?: string
  endpoint: string
  method: string
  request_body?: any
  response_body?: any
  status_code: number
  duration_ms: number
  created_at: string
}

export const requestLogsAPI = {
  listLogs: (profileId: string, params?: { limit?: number; status_code?: number; endpoint?: string; start_date?: string; end_date?: string }) => {
    const queryParams = new URLSearchParams()
    if (params?.limit) queryParams.append('limit', params.limit.toString())
    if (params?.status_code) queryParams.append('status_code', params.status_code.toString())
    if (params?.endpoint) queryParams.append('endpoint', params.endpoint)
    if (params?.start_date) queryParams.append('start_date', params.start_date)
    if (params?.end_date) queryParams.append('end_date', params.end_date)
    const query = queryParams.toString()
    return api.get<RequestLog[]>(`/profiles/${profileId}/logs${query ? '?' + query : ''}`)
  },
  getLog: (profileId: string, logId: string) => 
    api.get<RequestLog>(`/profiles/${profileId}/logs/${logId}`),
  deleteLog: (profileId: string, logId: string) => 
    api.delete(`/profiles/${profileId}/logs/${logId}`),
  exportLogs: (profileId: string, format: 'json' | 'csv' = 'json') => 
    api.get(`/profiles/${profileId}/logs/export?format=${format}`, { responseType: 'blob' }),
}

export interface DashboardStats {
  system_metrics: Record<string, any>
  neurondb_stats?: {
    collections_count: number
    total_vectors: number
    indexes_count: number
    query_count: number
    avg_query_time: number
  }
  neuronagent_stats?: {
    agents_count: number
    sessions_count: number
    messages_count: number
    avg_response_time: number
  }
  mcp_stats?: {
    tools_count: number
    tools_called: number
    active_connections: number
  }
  recent_activity: Array<{
    id: string
    type: string
    description: string
    timestamp: string
    user?: string
  }>
  health_status: {
    status: string
    components: Record<string, string>
    last_checked: string
  }
}

export const dashboardAPI = {
  getDashboard: (profileId: string) => 
    api.get<DashboardStats>(`/profiles/${profileId}/dashboard`),
}

export default api

