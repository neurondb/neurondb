'use client'

import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { mcpAPI, profilesAPI, modelConfigAPI, type Profile, type ToolDefinition, type ToolResult, type ModelConfig, type MCPThread, type MCPMessage, type MCPThreadWithMessages } from '@/lib/api'
import StatusBadge from '@/components/StatusBadge'
import JSONViewer from '@/components/JSONViewer'
import MarkdownContent from '@/components/MarkdownContent'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import { 
  PaperAirplaneIcon, 
  WrenchScrewdriverIcon,
  CheckCircleIcon,
  XCircleIcon,
  ChatBubbleLeftRightIcon,
  SparklesIcon,
  PlusIcon,
  MagnifyingGlassIcon,
  TrashIcon
} from '@/components/Icons'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
  toolName?: string
  timestamp: number
  data?: any
}

interface Thread {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: Message[]
}

// Main MCP Console component
// This component is wrapped with ErrorBoundary for error handling
function MCPPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [tools, setTools] = useState<ToolDefinition[]>([])
  const [modelConfigs, setModelConfigs] = useState<ModelConfig[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [modelError, setModelError] = useState<string | null>(null)
  const [threads, setThreads] = useState<Thread[]>([])
  const [activeThreadId, setActiveThreadId] = useState<string>('')
  const [threadSearch, setThreadSearch] = useState('')
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const [showTools, setShowTools] = useState(false)
  const [profileError, setProfileError] = useState<string | null>(null)
  const [loadingProfiles, setLoadingProfiles] = useState(true)
  const [mcpError, setMcpError] = useState<string | null>(null)
  const [loadingThreads, setLoadingThreads] = useState(false)
  const errorShownRef = useRef<boolean>(false)
  const reconnectAttemptRef = useRef<number>(0)
  const shouldAutoConnectRef = useRef<boolean>(true)
  const initializedRef = useRef<boolean>(false)
  const isMountedRef = useRef<boolean>(true)
  const lastLoadedProfileRef = useRef<string>('')
  const threadsLoadingRef = useRef<boolean>(false)
  const lastConnectedProfileRef = useRef<string>('')
  const conversationThreadRef = useRef<string>('') // Track thread ID for current conversation

  // Comprehensive error logging function - NEVER fails silently
  const logError = useCallback((context: string, error: any, additionalInfo?: Record<string, any>) => {
    const errorDetails = {
      context,
      timestamp: new Date().toISOString(),
      error: {
        message: error?.message || String(error),
        stack: error?.stack,
        name: error?.name,
        code: error?.code,
        response: error?.response ? {
          status: error.response.status,
          statusText: error.response.statusText,
          data: error.response.data,
        } : undefined,
      },
      additionalInfo,
      userAgent: typeof window !== 'undefined' ? window.navigator.userAgent : undefined,
      url: typeof window !== 'undefined' ? window.location.href : undefined,
    }
    
    // Always log to console with full details
    console.error(`[ERROR] ${context}:`, errorDetails)
    
    // Display error to user if component is mounted
    if (isMountedRef.current) {
      const userMessage = parseError(error)
      setMcpError(userMessage)
    }
    
    // In production, send to error reporting service
    // Example: if (process.env.NODE_ENV === 'production') { Sentry.captureException(error, { extra: errorDetails }) }
    
    return errorDetails
  }, [])

  // Helper to parse error and return simple one-line message
  const parseError = (error: any): string => {
    if (!error) return 'Unknown error'
    
    const msg = error.response?.data?.error || error.response?.data?.message || error.message || String(error)
    const msgLower = msg.toLowerCase()
    
    // Check if NeuronMCP is not running
    if (msgLower.includes('connection refused') || 
        msgLower.includes('connection reset') ||
        msgLower.includes('econnrefused') ||
        msgLower.includes('failed to connect') ||
        msgLower.includes('not running') ||
        msgLower.includes('cannot connect')) {
      return 'NeuronMCP not running'
    }
    
    // Check for timeout
    if (msgLower.includes('timeout') || msgLower.includes('timed out')) {
      return 'NeuronMCP not responding'
    }
    
    // Check for database migration errors
    if (msgLower.includes('does not exist') || msgLower.includes('relation') || msgLower.includes('mcp_chat_threads') || msgLower.includes('database tables not found')) {
      return 'Database migration required. Run: psql -d neurondb -f NeuronDesktop/api/migrations/007_mcp_chat_threads.sql'
    }
    
    // Return the actual error message (first line only)
    return msg.split('\n')[0].trim() || 'Configuration error'
  }

  // Track component mount state to prevent state updates after unmount
  useEffect(() => {
    isMountedRef.current = true
    
    // Detect if we're in a reload loop (component mounting too frequently)
    const mountTime = Date.now()
    const lastMountTime = sessionStorage.getItem('mcp_page_last_mount')
    if (lastMountTime) {
      const timeSinceLastMount = mountTime - parseInt(lastMountTime, 10)
      if (timeSinceLastMount < 1000) {
        console.warn('[MCP Page] Component mounting too frequently - possible reload loop detected', {
          timeSinceLastMount,
          timestamp: new Date().toISOString()
        })
      }
    }
    sessionStorage.setItem('mcp_page_last_mount', mountTime.toString())
    
    // Restore activeThreadId from sessionStorage if available (helps survive Fast Refresh)
    const savedThreadId = sessionStorage.getItem('mcp_active_thread_id')
    if (savedThreadId && !activeThreadId) {
      console.log('[MCP Page] Restoring active thread ID from sessionStorage:', savedThreadId)
      setActiveThreadId(savedThreadId)
    }
    
    return () => {
      isMountedRef.current = false
    }
  }, [activeThreadId])

  // Save activeThreadId to sessionStorage whenever it changes (survives Fast Refresh)
  useEffect(() => {
    if (activeThreadId) {
      sessionStorage.setItem('mcp_active_thread_id', activeThreadId)
      console.log('[MCP Page] Saved active thread ID to sessionStorage:', activeThreadId)
    }
  }, [activeThreadId])

  const loadProfiles = useCallback(async () => {
    if (!isMountedRef.current) return
    
    setLoadingProfiles(true)
    setProfileError(null)
    try {
      const response = await profilesAPI.list()
      if (!isMountedRef.current) return
      
      console.log('Profiles loaded:', response.data)
      setProfiles(response.data)
      
      if (response.data.length > 0 && !selectedProfile) {
        // Use active profile (set at login); otherwise default/first profile.
        const activeProfileId = localStorage.getItem('active_profile_id')
        if (activeProfileId) {
          const activeProfile = response.data.find((p: Profile) => p.id === activeProfileId)
          if (activeProfile) {
            setSelectedProfile(activeProfileId)
            if (isMountedRef.current) setLoadingProfiles(false)
            return
          }
        }

        const defaultProfile = response.data.find((p: Profile) => p.is_default)
        setSelectedProfile(defaultProfile ? defaultProfile.id : response.data[0].id)
      } else if (response.data.length === 0) {
        setProfileError('No profiles found. Profiles are created automatically during signup.')
      }
    } catch (error: any) {
      logError('loadProfiles', error, { selectedProfile })
      if (isMountedRef.current) {
        const errorMsg = parseError(error)
        setProfileError(errorMsg)
      }
    } finally {
      if (isMountedRef.current) {
        setLoadingProfiles(false)
      }
    }
  }, [selectedProfile, logError])

  useEffect(() => {
    // Prevent double initialization in React StrictMode
    if (initializedRef.current) return
    initializedRef.current = true
    
    loadProfiles().catch((error) => {
      logError('loadProfiles initialization', error, { phase: 'initialization' })
    })
  }, [loadProfiles, logError])

  // Prevent page reloads from unhandled promise rejections
  useEffect(() => {
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error('Unhandled promise rejection:', event.reason)
      // Prevent default browser behavior (page reload) for non-critical errors
      if (event.reason && typeof event.reason === 'object') {
        const error = event.reason as any
        // Only prevent reload for API errors, not critical system errors
        if (error?.response?.status || error?.code === 'ERR_BAD_RESPONSE' || error?.message?.includes('500')) {
          event.preventDefault()
          return
        }
      }
      event.preventDefault() // Prevent default for all promise rejections
    }
    
    const handleError = (event: ErrorEvent) => {
      console.error('Unhandled error:', event.error)
      // Prevent default for non-critical errors
      if (event.error && typeof event.error === 'object') {
        const error = event.error as any
        if (error?.response?.status || error?.code === 'ERR_BAD_RESPONSE') {
          event.preventDefault()
        }
      }
    }
    
    window.addEventListener('unhandledrejection', handleUnhandledRejection)
    window.addEventListener('error', handleError)
    
    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection)
      window.removeEventListener('error', handleError)
    }
  }, [])

  // Create new thread in database
  const createNewThreadInDB = useCallback(async (): Promise<Thread> => {
    if (!selectedProfile) {
      throw new Error('No profile selected')
    }
    
    const response = await mcpAPI.createThread(selectedProfile, 'New chat')
    const apiThread = response.data
    
    return {
      id: apiThread.id,
      title: apiThread.title,
      createdAt: new Date(apiThread.created_at).getTime(),
      updatedAt: new Date(apiThread.updated_at).getTime(),
      messages: [],
    }
  }, [selectedProfile])

  // Convert API thread format to local format
  const convertAPIThreadToLocal = (apiThread: MCPThreadWithMessages | MCPThread | null | undefined): Thread | null => {
    if (!apiThread || !apiThread.id) {
      return null
    }
    return {
      id: apiThread.id,
      title: apiThread.title,
      createdAt: new Date(apiThread.created_at).getTime(),
      updatedAt: new Date(apiThread.updated_at).getTime(),
      messages: ('messages' in apiThread && apiThread.messages) ? apiThread.messages.map((msg) => ({
        id: msg.id,
        role: msg.role as Message['role'],
        content: msg.content,
        toolName: typeof msg.tool_name === 'string' ? msg.tool_name : (msg.tool_name as any)?.String || undefined,
        timestamp: new Date(msg.created_at).getTime(),
        data: msg.data,
      })) : [],
    }
  }


  const activeThread = useMemo(() => 
    threads.find((t) => t.id === activeThreadId) || null,
    [threads, activeThreadId]
  )
  
  // Deduplicate messages by ID to prevent showing the same message multiple times
  const messages = useMemo(() => {
    if (!activeThread?.messages) return []
    return activeThread.messages.filter((msg, index, self) => 
      index === self.findIndex((m) => m.id === msg.id)
    )
  }, [activeThread])

  useEffect(() => {
    if (messages.length > 0 && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages.length, activeThreadId]) // Only depend on length and thread ID, not the full array

  const loadTools = useCallback(async () => {
    if (!selectedProfile) return
    try {
      const response = await mcpAPI.listTools(selectedProfile)
      setTools(response.data.tools || [])
      setMcpError(null)
    } catch (error: any) {
      console.error('Failed to load tools:', error)
      const errorMsg = parseError(error)
      setMcpError(errorMsg)
    }
  }, [selectedProfile])

  const loadModelConfigs = useCallback(async () => {
    if (!selectedProfile) return
    try {
      const response = await modelConfigAPI.list(selectedProfile, false)
      setModelConfigs(response.data)
      const defaultModel = response.data.find(m => m.is_default)
      if (defaultModel) {
        setSelectedModel(defaultModel.id)
      } else if (response.data.length > 0) {
        setSelectedModel(response.data[0].id)
      }
    } catch (error) {
      console.error('Failed to load model configs:', error)
    }
  }, [selectedProfile])

  const validateModel = useCallback(async () => {
    if (!selectedModel || !selectedProfile) {
      setModelError(null)
      return
    }

    const model = modelConfigs.find(m => m.id === selectedModel)
    if (!model) {
      setModelError('Model not found')
      return
    }

    const provider = model.model_provider
    if (provider !== 'ollama' && !model.is_free) {
      try {
        const response = await modelConfigAPI.list(selectedProfile, true)
        const fullModel = response.data.find(m => m.id === selectedModel)
        if (!fullModel || !fullModel.api_key || fullModel.api_key.trim() === '') {
          setModelError(`API key not set for ${model.model_name}. Please configure it in Model Settings.`)
          return
        }
      } catch (error) {
        setModelError('Failed to validate model configuration')
        return
      }
    }

    setModelError(null)
  }, [selectedModel, selectedProfile, modelConfigs])


  // Load conversation threads from API when profile is selected
  const loadThreads = useCallback(async (retryCount: number = 0): Promise<void> => {
    const maxRetries = 2
    
    if (!isMountedRef.current || !selectedProfile) {
      threadsLoadingRef.current = false
      return
    }
    
    setLoadingThreads(true)
    try {
      const response = await mcpAPI.listThreads(selectedProfile)
      if (!isMountedRef.current) return
      
      const apiThreads = response.data || []
      
      if (apiThreads.length > 0) {
        // Load full thread data with messages (with timeout per thread)
        const loadThreadWithTimeout = async (apiThread: MCPThread, timeout: number = 5000): Promise<Thread | null> => {
          try {
            const threadResponse = await Promise.race([
              mcpAPI.getThread(selectedProfile, apiThread.id),
              new Promise<never>((_, reject) => 
                setTimeout(() => reject(new Error('Thread load timeout')), timeout)
              )
            ])
            
            return convertAPIThreadToLocal(threadResponse.data)
          } catch (error) {
            console.warn(`Thread ${apiThread.id} load timeout or error, using basic info:`, error)
            return convertAPIThreadToLocal(apiThread)
          }
        }
        
        const loadedThreads = await Promise.all(
          apiThreads.map(thread => loadThreadWithTimeout(thread))
        )
        
        const validThreads = loadedThreads.filter((t): t is Thread => t !== null)
        
        if (isMountedRef.current) {
          setThreads(validThreads)
          
          // Restore active thread if it exists
          if (activeThreadId) {
            const activeThread = validThreads.find(t => t.id === activeThreadId)
            if (!activeThread && validThreads.length > 0) {
              setActiveThreadId(validThreads[0].id)
            }
          } else if (validThreads.length > 0) {
            setActiveThreadId(validThreads[0].id)
          }
        }
      } else {
        // No threads exist - create one
        if (activeThreadId) {
          // If we have an activeThreadId but no threads, try to create it
          try {
            const newThread = await createNewThreadInDB()
            if (isMountedRef.current) {
              setThreads([newThread])
              setActiveThreadId(newThread.id)
            }
          } catch (error) {
            logError('loadThreads - createNewThread fallback', error, { selectedProfile })
            if (isMountedRef.current) {
              setMcpError('Failed to load threads and create new thread. Please check database connection.')
            }
          }
        } else {
          // Create initial thread if none exist (only once)
          if (threads.length === 0) {
            try {
              const newThread = await createNewThreadInDB()
              if (isMountedRef.current) {
                setThreads([newThread])
                setActiveThreadId(newThread.id)
              }
            } catch (error) {
              logError('loadThreads - createInitialThread', error, { selectedProfile })
              if (isMountedRef.current) {
                setMcpError('Failed to create initial thread. Please check database connection.')
              }
            }
          }
        }
      }
    } catch (error: any) {
      logError('loadThreads', error, { selectedProfile })
      
      // Only retry if we haven't exceeded max retries and don't have threads
      if (threads.length === 0) {
        // Retry once after a short delay
        setTimeout(() => {
          if (isMountedRef.current && selectedProfile) {
            loadThreads().catch((retryError) => {
              logError('loadThreads - retry', retryError, { selectedProfile })
              if (isMountedRef.current) {
                setMcpError('Failed to load threads after retry. Please check database connection.')
                setLoadingThreads(false)
              }
            })
          }
        }, 1000)
      } else {
        if (isMountedRef.current) {
          setLoadingThreads(false)
        }
      }
    } finally {
      threadsLoadingRef.current = false
      if (isMountedRef.current) {
        setLoadingThreads(false)
      }
    }
  }, [selectedProfile, activeThreadId, threads.length, createNewThreadInDB, logError])

  useEffect(() => {
    if (selectedProfile && activeThreadId === null && threads.length === 0) {
      loadThreads().catch((error) => {
        logError('loadThreads - unhandled error', error, { selectedProfile })
        threadsLoadingRef.current = false
        if (isMountedRef.current) {
          setLoadingThreads(false)
        }
      })
    }
  }, [selectedProfile, activeThreadId, threads.length, loadThreads, logError])

  // This useEffect will be moved after connectWebSocket is defined

  useEffect(() => {
    if (selectedModel) {
      validateModel()
    } else {
      setModelError(null)
    }
  }, [selectedModel, validateModel])

  const createNewThread = async () => {
    if (!selectedProfile) return
    
    try {
      const newThread = await createNewThreadInDB()
      setThreads((prev) => [newThread, ...prev])
      setActiveThreadId(newThread.id)
      setInput('')
    } catch (error: any) {
      console.error('Failed to create thread:', error)
      setMcpError(parseError(error))
    }
  }

  const deleteThread = async (threadId: string) => {
    if (!selectedProfile) return
    
    try {
      await mcpAPI.deleteThread(selectedProfile, threadId)
      setThreads((prev) => {
        const next = prev.filter((t) => t.id !== threadId)
        // Ensure we always have at least one thread.
        if (next.length === 0) {
          createNewThreadInDB().then((t) => {
            setThreads([t])
            setActiveThreadId(t.id)
          })
          return []
        }

        if (activeThreadId === threadId) {
          setActiveThreadId(next[0].id)
        }
        return next
      })
    } catch (error: any) {
      logError('deleteThread', error, { threadId, selectedProfile })
      if (isMountedRef.current) {
        setMcpError(parseError(error))
      }
    }
  }

  const addMessage = useCallback(async (role: Message['role'], content: string, toolName?: string, data?: any) => {
    if (!selectedProfile || !isMountedRef.current) return
    
    // For assistant/tool messages, use the conversation thread if available to ensure they go to the right thread
    const targetThreadId = (role === 'assistant' || role === 'tool') && conversationThreadRef.current 
      ? conversationThreadRef.current 
      : activeThreadId
    
    console.log('[addMessage]', { role, targetThreadId, activeThreadId, conversationThread: conversationThreadRef.current })
    
    // If no active thread, create one
    if (!targetThreadId) {
      try {
        const newThread = await createNewThreadInDB()
        if (!isMountedRef.current) return
        setThreads((prev) => [newThread, ...prev])
        setActiveThreadId(newThread.id)
        conversationThreadRef.current = newThread.id
      } catch (error) {
        if (!isMountedRef.current) return
        logError('addMessage - createThread', error, { role, contentLength: content.length })
        // Create temp thread in memory
        const tempThread: Thread = {
          id: `temp-${Date.now()}`,
          title: 'New chat',
          createdAt: Date.now(),
          updatedAt: Date.now(),
          messages: [],
        }
        setThreads((prev) => [tempThread, ...prev])
        setActiveThreadId(tempThread.id)
        conversationThreadRef.current = tempThread.id
      }
      return // Exit and let the function be called again with the new thread
    }

    // Create message with temp ID immediately (don't wait for DB)
    const now = Date.now()
    const msg: Message = {
      id: `temp-${now}-${Math.random().toString(36).slice(2, 9)}`,
      role,
      content,
      toolName,
      timestamp: now,
      data,
    }

    // Add to UI immediately (only if still mounted)
    if (!isMountedRef.current) return
    setThreads((prev) => {
      return prev.map((t) => {
        if (t.id !== targetThreadId) return t
        
        // Check if message already exists to prevent duplicates
        const messageExists = t.messages.some(m => m.content === content && m.role === role && m.timestamp > now - 1000)
        if (messageExists) {
          return t
        }
        
        return {
          ...t,
          updatedAt: now,
          messages: [...t.messages, msg],
        }
      })
    })

    // Try to save to DB in background (don't block UI)
    // Only save if thread ID is not a temp ID (starts with 'temp-')
    if (targetThreadId && !targetThreadId.startsWith('temp-')) {
      try {
        const response = await mcpAPI.addMessage(selectedProfile, targetThreadId, role, content, toolName, data)
        if (!isMountedRef.current) return
        
        const apiMessage = response.data
        
        // Update with real ID from DB
        setThreads((prev) => {
          return prev.map((t) => {
            if (t.id !== targetThreadId) return t
            return {
              ...t,
              messages: t.messages.map((m) => 
                m.id === msg.id ? { ...m, id: apiMessage.id } : m
              ),
            }
          })
        })
      } catch (error: any) {
        if (!isMountedRef.current) return
        logError('addMessage - saveToDB', error, { 
          role, 
          contentLength: content.length, 
          threadId: targetThreadId,
          toolName,
          note: 'Message displayed in UI but not saved to DB - non-fatal'
        })
        // Message is already in UI, so this is non-fatal
      }
    } else {
      // Thread is temporary, don't try to save to DB
      console.log('Message added to temp thread, skipping DB save')
    }
  }, [selectedProfile, activeThreadId, createNewThreadInDB, logError])

  const connectWebSocket = useCallback(() => {
    if (!selectedProfile) return
    
    // Prevent multiple simultaneous connection attempts
    if (connecting || connected) return
    
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    
    setConnecting(true)
    setConnected(false)
    setMcpError(null)
    errorShownRef.current = false
    
    const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081/api/v1'
    const wsBaseUrl = apiBaseUrl.replace(/^http/, 'ws')
    // Browser WebSockets can't set Authorization headers; pass JWT via query param (?token=...)
    const token = localStorage.getItem('neurondesk_auth_token')
    const tokenParam = token ? `?token=${encodeURIComponent(token)}` : ''
    const wsUrl = `${wsBaseUrl}/profiles/${selectedProfile}/mcp/ws${tokenParam}`
    
    console.log('Connecting to WebSocket:', wsUrl.replace(/token=[^&]+/, 'token=***'))
    console.log('Token present:', !!token)
    
    const ws = new WebSocket(wsUrl)
    
    ws.onopen = () => {
      if (!isMountedRef.current) return
      setConnected(true)
      setConnecting(false)
      setMcpError(null)
      errorShownRef.current = false
      reconnectAttemptRef.current = 0
      ws.send(JSON.stringify({ type: 'list_tools' }))
    }
    
    ws.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('WebSocket message received:', data)
        
        if (data.type === 'connected') {
          // Skip adding 'connected' message if threads aren't loaded yet
          // It will be added once threads are ready
          if (activeThreadId && !activeThreadId.startsWith('temp-')) {
            try {
              await addMessage('system', data.message)
            } catch (error) {
              logError('WebSocket - addConnectedMessage', error, { activeThreadId, message: data.message })
            }
          }
        } else if (data.type === 'tool_result') {
          try {
            await addMessage('tool', JSON.stringify(data.result, null, 2), undefined, data.result)
            if (isMountedRef.current) setLoading(false) // Clear loading state after tool result
          } catch (error) {
            logError('WebSocket - addToolResultMessage', error, { result: data.result })
            if (isMountedRef.current) setLoading(false)
          }
        } else if (data.type === 'tools') {
          if (isMountedRef.current) setTools(data.tools || [])
        } else if (data.type === 'error') {
          if (!isMountedRef.current) return
          setConnecting(false)
          setConnected(false)
          setLoading(false) // Clear loading state on error
          const errorMsg = parseError({ message: data.error || 'Unknown error' })
          logError('WebSocket - error message', new Error(data.error || 'Unknown error'), { errorData: data })
          try {
            await addMessage('system', `Error: ${errorMsg}`)
          } catch (error) {
            logError('WebSocket - addErrorMessage', error, { errorMsg })
          }
          if (!errorShownRef.current) {
            setMcpError(errorMsg)
            errorShownRef.current = true
          }
          shouldAutoConnectRef.current = false
        } else if (data.type === 'assistant') {
          const content = data.content || data.message || data.text || ''
          if (content) {
            try {
              await addMessage('assistant', content)
              if (isMountedRef.current) setLoading(false) // Clear loading state after receiving assistant response
            } catch (error) {
              logError('WebSocket - addAssistantMessage', error, { contentLength: content.length })
              if (isMountedRef.current) setLoading(false)
            }
          } else {
            logError('WebSocket - emptyAssistantMessage', new Error('Received assistant message with no content'), { data })
            if (isMountedRef.current) setLoading(false)
            try {
              await addMessage('system', 'Received empty response from server. Check browser console for details.')
            } catch (error) {
              logError('WebSocket - addEmptyResponseMessage', error)
            }
          }
        } else if (data.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong' }))
        } else {
          // Handle unknown message types
          logError('WebSocket - unknownMessageType', new Error(`Unknown message type: ${data.type}`), { messageType: data.type, data })
          // Try to extract any content that might be there
          if (data.content || data.message || data.text) {
            try {
              await addMessage('assistant', data.content || data.message || data.text || '')
              if (isMountedRef.current) setLoading(false)
            } catch (error) {
              logError('WebSocket - addUnknownTypeMessage', error, { messageType: data.type })
              if (isMountedRef.current) setLoading(false)
            }
          } else {
            if (isMountedRef.current) setLoading(false)
            try {
              await addMessage('system', `Received unexpected message type: ${data.type}. Check console for details.`)
            } catch (error) {
              logError('WebSocket - addUnexpectedTypeMessage', error, { messageType: data.type })
            }
          }
        }
      } catch (error) {
        logError('WebSocket - parseMessage', error, { 
          rawData: typeof event.data === 'string' ? event.data.substring(0, 500) : String(event.data).substring(0, 500),
          dataLength: typeof event.data === 'string' ? event.data.length : String(event.data).length
        })
        if (isMountedRef.current) {
          setLoading(false) // Clear loading state on parse error
          if (!errorShownRef.current) {
            setMcpError('Invalid response from NeuronMCP')
            errorShownRef.current = true
          }
        }
      }
    }
    
    ws.onerror = (error) => {
      logError('WebSocket onerror', error, { 
        selectedProfile, 
        wsState: ws.readyState,
        wsUrl: wsUrl.replace(/token=[^&]+/, 'token=***')
      })
      if (!isMountedRef.current) return
      setConnecting(false)
      setConnected(false)
      if (!errorShownRef.current) {
        setMcpError('NeuronMCP not running')
        errorShownRef.current = true
      }
      shouldAutoConnectRef.current = false // Stop auto-reconnecting
    }
    
    ws.onclose = (event) => {
      console.log('WebSocket closed:', { code: event.code, reason: event.reason, wasClean: event.wasClean })
      if (!isMountedRef.current) return
      setConnecting(false)
      setConnected(false)
      
      // Handle different close codes with reconnection logic
      if (event.code === 1000 || event.code === 1001) {
        // Normal closure - allow auto-reconnect
        shouldAutoConnectRef.current = true
        reconnectAttemptRef.current = 0
      } else if (event.code === 1006) {
        // Abnormal closure - attempt reconnection with exponential backoff
        if (shouldAutoConnectRef.current && reconnectAttemptRef.current < 5) {
          reconnectAttemptRef.current++
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current - 1), 30000) // Max 30s
          console.log(`WebSocket closed abnormally. Reconnecting in ${delay}ms (attempt ${reconnectAttemptRef.current}/5)...`)
          setTimeout(() => {
            if (isMountedRef.current && shouldAutoConnectRef.current && selectedProfile) {
              connectWebSocket()
            }
          }, delay)
          if (!errorShownRef.current) {
            setMcpError(`Connection lost. Reconnecting... (attempt ${reconnectAttemptRef.current}/5)`)
            errorShownRef.current = true
          }
        } else {
          // Max retries reached
          if (!errorShownRef.current) {
            setMcpError('Connection lost. Maximum reconnection attempts reached. Please refresh the page.')
            errorShownRef.current = true
          }
          shouldAutoConnectRef.current = false
        }
      } else {
        // Other error codes - show error but allow manual reconnect
        if (!errorShownRef.current) {
          // Common close codes that indicate server not running
          if (event.code === 1006 || event.code === 1001) {
            setMcpError('Connection failed - check auth token or MCP server')
          } else {
            setMcpError(`Connection error (code: ${event.code})`)
          }
          errorShownRef.current = true
        }
        shouldAutoConnectRef.current = false
      }
    }
    
    wsRef.current = ws
  }, [selectedProfile, connecting, connected, logError, activeThreadId, addMessage])

  useEffect(() => {
    if (selectedModel) {
      validateModel()
    } else {
      setModelError(null)
    }
  }, [selectedModel, validateModel])

  useEffect(() => {
    if (!selectedProfile || !shouldAutoConnectRef.current) return
    
    // Prevent reconnecting if already connected to the same profile
    if (lastConnectedProfileRef.current === selectedProfile && (connecting || connected)) return
    
    // Prevent multiple simultaneous initializations
    if (connecting || connected) {
      // If connected to a different profile, close and reconnect
      if (lastConnectedProfileRef.current && lastConnectedProfileRef.current !== selectedProfile) {
        if (wsRef.current) {
          wsRef.current.close()
          wsRef.current = null
        }
        lastConnectedProfileRef.current = ''
        setConnected(false)
        setConnecting(false)
      } else {
        return
      }
    }
    
    lastConnectedProfileRef.current = selectedProfile
    
    try {
      loadTools().catch((error) => {
        logError('useEffect - loadTools', error, { selectedProfile })
      })
      loadModelConfigs().catch((error) => {
        logError('useEffect - loadModelConfigs', error, { selectedProfile })
      })
      connectWebSocket()
    } catch (error) {
      logError('useEffect - connectWebSocket', error, { selectedProfile })
      if (isMountedRef.current) {
        setConnected(false)
        setConnecting(false)
      }
    }
  }, [selectedProfile, connecting, connected, loadTools, loadModelConfigs, connectWebSocket, logError])

  useEffect(() => {
    if (!selectedProfile || !shouldAutoConnectRef.current) return
    
    // Prevent reconnecting if already connected to the same profile
    if (lastConnectedProfileRef.current === selectedProfile && (connecting || connected)) return
    
    // Prevent multiple simultaneous initializations
    if (connecting || connected) {
      // If connected to a different profile, close and reconnect
      if (lastConnectedProfileRef.current && lastConnectedProfileRef.current !== selectedProfile) {
        if (wsRef.current) {
          wsRef.current.close()
          wsRef.current = null
        }
        lastConnectedProfileRef.current = ''
        setConnected(false)
        setConnecting(false)
      } else {
        return
      }
    }
    
    lastConnectedProfileRef.current = selectedProfile
    
    try {
      loadTools().catch((error) => {
        logError('useEffect - loadTools', error, { selectedProfile })
      })
      loadModelConfigs().catch((error) => {
        logError('useEffect - loadModelConfigs', error, { selectedProfile })
      })
      connectWebSocket()
    } catch (error) {
      logError('useEffect - initializeMCPConnection', error, { selectedProfile })
      // Don't let initialization errors crash the component
    }
    
    return () => {
      if (wsRef.current) {
        try {
          wsRef.current.close()
        } catch (error) {
          logError('useEffect cleanup - closeWebSocket', error, { selectedProfile })
        }
        wsRef.current = null
      }
      if (isMountedRef.current) {
        setConnected(false)
        setConnecting(false)
      }
    }
  }, [selectedProfile, connectWebSocket, connected, connecting, loadModelConfigs, loadTools, logError])

  const handleSendMessage = async () => {
    if (!input.trim() || loading || !selectedModel || modelError) return
    
    if (!connected) {
      const errorMsg = 'WebSocket not connected. Please wait for connection or click "Connect to MCP" button.'
      setMcpError(errorMsg)
      addMessage('system', `Error: ${errorMsg}`)
      return
    }
    
    const userMessage = input.trim()
    setInput('')
    
    // Store the current thread ID for this conversation
    conversationThreadRef.current = activeThreadId
    console.log('[MCP] Starting conversation in thread:', conversationThreadRef.current)
    
    await addMessage('user', userMessage)
    setLoading(true)
    setMcpError(null) // Clear any previous errors
    
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const message = {
          type: 'chat',
          content: userMessage,
          model_id: selectedModel
        }
        console.log('Sending WebSocket message:', message)
        wsRef.current.send(JSON.stringify(message))
        
        // Set a timeout to clear loading state if no response is received
        const timeoutId = setTimeout(() => {
          setLoading((prevLoading) => {
            if (prevLoading) {
              console.warn('No response received within timeout')
              const timeoutMsg = 'No response received within 60 seconds. The MCP server may be slow or not responding.'
              setMcpError(timeoutMsg)
              addMessage('system', timeoutMsg)
              return false
            }
            return prevLoading
          })
        }, 60000) // 60 second timeout
        
        // Store timeout ID to clear it if response comes early (handled in onmessage)
        // Note: In a production app, you'd want to track multiple timeouts, but for simplicity we'll let it run
      } else {
        setLoading(false)
        const errorMsg = `WebSocket connection is not open (state: ${wsRef.current?.readyState}). Please refresh the page or reconnect.`
        setMcpError(errorMsg)
        addMessage('system', `Error: ${errorMsg}`)
        console.error('WebSocket state:', {
          wsRefExists: !!wsRef.current,
          readyState: wsRef.current?.readyState,
          connected,
          connecting
        })
      }
    } catch (error: any) {
      setLoading(false)
      const errorMsg = parseError(error)
      setMcpError(errorMsg)
      addMessage('system', `Error: ${errorMsg}`)
      console.error('Error sending message:', error)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const handleToolCall = async (tool: ToolDefinition, args: Record<string, any>) => {
    if (!selectedProfile || !selectedModel || modelError) return
    
    setLoading(true)
    addMessage('user', `Calling tool: ${tool.name}`)
    
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'tool_call',
          name: tool.name,
          arguments: args,
          model_id: selectedModel
        }))
      } else {
        const result = await mcpAPI.callTool(selectedProfile, tool.name, args)
        addMessage('tool', JSON.stringify(result.data, null, 2), tool.name, result.data)
      }
    } catch (error: any) {
      const errorMsg = parseError(error)
      setMcpError(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const handleToolCallWithResult = async (tool: ToolDefinition, args: Record<string, any>): Promise<any> => {
    if (!selectedProfile) {
      throw new Error('No profile selected')
    }
    
    try {
      // Always use API call for ToolCard to get immediate results
      const result = await mcpAPI.callTool(selectedProfile, tool.name, args)
      // Also add to chat messages
      addMessage('tool', JSON.stringify(result.data, null, 2), tool.name, result.data)
      return result.data
    } catch (error: any) {
      const errorMsg = error.response?.data?.error || error.message || 'Tool call failed'
      throw new Error(errorMsg)
    }
  }

  const filteredThreads = threads
    .slice()
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .filter((t) => {
      const q = threadSearch.trim().toLowerCase()
      if (!q) return true
      return (t.title || '').toLowerCase().includes(q)
    })

  return (
    <div className="h-full w-full flex overflow-hidden bg-white dark:bg-slate-950">
      {/* Conversation Sidebar (ChatGPT-style) */}
      <aside className="w-72 flex flex-col bg-white dark:bg-slate-950 border-r border-slate-200 dark:border-slate-800">
        <div className="p-3 border-b border-slate-200 dark:border-slate-800">
          <button
            onClick={createNewThread}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600 text-gray-900 dark:text-slate-100 transition-colors"
          >
            <PlusIcon className="w-4 h-4" />
            New chat
          </button>

          <div className="mt-3 space-y-2">
            <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-slate-400">
              <StatusBadge status={connected ? 'connected' : connecting ? 'connecting' : 'disconnected'} />
              <span className="truncate">
                {profileError
                  ? 'Profile error'
                  : loadingProfiles
                    ? 'Loading profile…'
                    : `Profile: ${profiles.find((p) => p.id === selectedProfile)?.name || '…'}`}
              </span>
            </div>
            {!connected && !connecting && selectedProfile && (
              <button
                onClick={() => {
                  shouldAutoConnectRef.current = true
                  errorShownRef.current = false
                  connectWebSocket()
                }}
                className="w-full px-3 py-1.5 text-xs rounded-lg bg-purple-600 hover:bg-purple-700 text-white transition-colors flex items-center justify-center gap-2"
              >
                <span>Connect to MCP</span>
              </button>
            )}
            {connecting && (
              <div className="w-full px-3 py-1.5 text-xs text-center text-gray-600 dark:text-slate-400">
                Connecting...
              </div>
            )}
          </div>

          <div className="mt-3 relative">
            <MagnifyingGlassIcon className="w-4 h-4 text-gray-500 dark:text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              value={threadSearch}
              onChange={(e) => setThreadSearch(e.target.value)}
              placeholder="Search chats"
              className="w-full pl-9 pr-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-gray-900 dark:text-slate-200 placeholder-gray-500 dark:placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500/30 focus:border-purple-500/50"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {filteredThreads.map((t) => {
            const active = t.id === activeThreadId
            return (
              <div
                key={t.id}
                className={`group flex items-center gap-2 rounded-lg px-2 py-2 cursor-pointer transition-colors ${
                  active ? 'bg-slate-200 dark:bg-slate-800' : 'hover:bg-slate-100 dark:hover:bg-slate-900'
                }`}
                onClick={() => setActiveThreadId(t.id)}
                role="button"
                tabIndex={0}
              >
                <ChatBubbleLeftRightIcon className="w-4 h-4 text-gray-500 dark:text-slate-500 flex-shrink-0" />
                <div className="min-w-0 flex-1">
                  <div className={`text-sm truncate ${active ? 'text-gray-900 dark:text-slate-100' : 'text-gray-700 dark:text-slate-200'}`}>
                    {t.title || 'New chat'}
                  </div>
                  <div className="text-[11px] text-gray-500 dark:text-slate-500 truncate">
                    {t.messages.length} messages
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteThread(t.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded-md hover:bg-slate-200 dark:hover:bg-slate-700 text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-200 transition-all"
                  title="Delete chat"
                >
                  <TrashIcon className="w-4 h-4" />
                </button>
              </div>
            )
          })}
        </div>

        <div className="p-3 border-t border-slate-200 dark:border-slate-800 text-xs text-gray-500 dark:text-slate-500">
          Chats are stored locally in this browser.
        </div>
      </aside>

      {/* Main chat column */}
      <section className="flex-1 flex flex-col min-w-0">
        {/* Error Banner */}
        {mcpError && (
          <div className="px-4 py-2 bg-red-50 dark:bg-red-900/20 border-b border-red-200 dark:border-red-800/50 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 flex-1 min-w-0">
              <XCircleIcon className="w-4 h-4 text-red-600 dark:text-red-400 flex-shrink-0" />
              <span className="text-sm text-red-700 dark:text-red-300 truncate">{mcpError}</span>
            </div>
            {mcpError.includes('not running') && (
              <a
                href="/settings"
                className="text-xs text-red-600 dark:text-red-300 hover:text-red-800 dark:hover:text-red-200 underline whitespace-nowrap"
              >
                Run NeuronMCP
              </a>
            )}
          </div>
        )}
        
        {/* Header */}
        <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-950/30">
          <div className="min-w-0">
            <div className="text-sm text-gray-900 dark:text-slate-200 truncate">
              {activeThread?.title || 'New chat'}
            </div>
            {profileError && (
              <div className="text-xs text-red-600 dark:text-red-400 truncate">
                {profileError} <a href="/settings" className="underline text-red-500 dark:text-red-300 hover:text-red-700 dark:hover:text-red-200">Settings</a>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Model selector (ChatGPT-style placement: top) */}
            <div className="flex items-center gap-2">
              <SparklesIcon className="w-4 h-4 text-gray-500 dark:text-slate-500" />
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="px-3 py-1.5 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-sm text-gray-900 dark:text-slate-200 hover:border-slate-400 dark:hover:border-slate-600 focus:outline-none focus:ring-2 focus:ring-purple-500/30 focus:border-purple-500/50 transition-colors"
              >
                <option value="">Select a model</option>
                {modelConfigs.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.model_provider} - {model.model_name} {model.is_default && '(Default)'}
                  </option>
                ))}
              </select>
            </div>

            {modelError ? (
              <div className="hidden sm:flex items-center gap-2 text-red-400 text-xs">
                <XCircleIcon className="w-4 h-4" />
                <span className="max-w-56 truncate">{modelError}</span>
              </div>
            ) : selectedModel ? (
              <div className="hidden sm:flex items-center gap-2 text-green-400 text-xs">
                <CheckCircleIcon className="w-4 h-4" />
                <span>Ready</span>
              </div>
            ) : null}

            <button
              onClick={() => setShowTools((v) => !v)}
              className="p-2 text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-900 rounded-lg transition-colors"
              title="Tools"
            >
              <WrenchScrewdriverIcon className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-6">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full min-h-[45vh]">
                <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-full flex items-center justify-center mb-4">
                  <SparklesIcon className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-slate-100 mb-2">How can I help you today?</h2>
                <p className="text-gray-600 dark:text-slate-400 mb-6">Pick a model, then send a message. Tools are in the right drawer.</p>
              </div>
            )}

            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}

            {loading && messages.length > 0 && (
              <div className="flex items-center gap-2 text-gray-500 dark:text-slate-400 py-4">
                <div className="w-2 h-2 bg-gray-400 dark:bg-slate-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 dark:bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                <div className="w-2 h-2 bg-gray-400 dark:bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Composer */}
        <div className="border-t border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-950/30">
          <div className="max-w-3xl mx-auto px-4 py-4">
            <div className="relative">
              <div className="relative flex items-end bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-2xl shadow-lg hover:border-slate-400 dark:hover:border-slate-600 focus-within:border-purple-500 focus-within:ring-2 focus-within:ring-purple-500/20 transition-all">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={selectedModel ? 'Message…' : 'Select a model first…'}
                  disabled={!selectedModel || !!modelError || loading}
                  rows={1}
                  className="flex-1 px-4 py-3 bg-transparent text-gray-900 dark:text-slate-100 placeholder-gray-500 dark:placeholder-slate-500 resize-none focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed max-h-32 overflow-y-auto"
                  style={{ height: 'auto' }}
                  onInput={(e) => {
                    const target = e.target as HTMLTextAreaElement
                    target.style.height = 'auto'
                    target.style.height = `${Math.min(target.scrollHeight, 128)}px`
                  }}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!input.trim() || !selectedModel || !!modelError || loading}
                  className="m-2 p-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-300 dark:disabled:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
                  title="Send message"
                >
                  <PaperAirplaneIcon className="w-5 h-5" />
                </button>
              </div>
              {!selectedModel && (
                <p className="text-xs text-red-600 dark:text-red-400 mt-2 text-center">Please select a model to start chatting</p>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Tools Drawer (right) */}
      {showTools && (
        <aside className="w-96 bg-white dark:bg-slate-950 border-l border-slate-200 dark:border-slate-800 flex flex-col">
          <div className="p-4 border-b border-slate-200 dark:border-slate-800">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-slate-100">Tools</h2>
              <button
                onClick={() => setShowTools(false)}
                className="text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-200"
              >
                <XCircleIcon className="w-5 h-5" />
              </button>
            </div>
            <p className="text-sm text-gray-600 dark:text-slate-400 mt-1">{tools.length} available</p>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {tools.map((tool) => (
              <ToolCard
                key={tool.name}
                tool={tool}
                onCall={handleToolCallWithResult}
                disabled={!selectedModel || !!modelError || loading}
              />
            ))}
          </div>
        </aside>
      )}
    </div>
  )
}

// Export wrapped component with ErrorBoundary
// Using a stable export to prevent Fast Refresh issues
const MCPPageWithErrorBoundary = () => (
  <ErrorBoundary>
    <MCPPage />
  </ErrorBoundary>
)

export default MCPPageWithErrorBoundary

// Tool Card Component
function ToolCard({ tool, onCall, disabled }: { tool: ToolDefinition, onCall: (tool: ToolDefinition, args: Record<string, any>) => Promise<any>, disabled: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const [args, setArgs] = useState<Record<string, any>>({})
  const [result, setResult] = useState<any>(null)
  const [calling, setCalling] = useState(false)

  const handleCall = async () => {
    setCalling(true)
    setResult(null)
    try {
      const toolResult = await onCall(tool, args)
      setResult(toolResult)
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : 'Tool call failed' })
    } finally {
      setCalling(false)
    }
  }

  return (
    <div className="border border-slate-300 dark:border-slate-700 rounded-lg bg-slate-50 dark:bg-slate-800 flex flex-col">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left p-3 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
      >
        <div className="font-medium text-sm text-gray-900 dark:text-slate-100">{tool.name}</div>
        <div className="text-xs text-gray-600 dark:text-slate-400 mt-1 line-clamp-2">{tool.description}</div>
      </button>
      {expanded && (
        <div className="flex flex-col border-t border-slate-300 dark:border-slate-700" style={{ maxHeight: '600px' }}>
          {/* Results Display - Top */}
          <div className="flex-1 p-3 border-b border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900/50 min-h-[200px] max-h-[300px] overflow-y-auto">
            <div className="text-xs font-semibold text-gray-700 dark:text-slate-300 mb-2">Results:</div>
            {result ? (
              <div className="text-xs">
                <JSONViewer data={result} defaultExpanded={false} />
              </div>
            ) : calling ? (
              <div className="flex items-center gap-2 text-gray-500 dark:text-slate-400">
                <div className="w-2 h-2 bg-gray-400 dark:bg-slate-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 dark:bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                <div className="w-2 h-2 bg-gray-400 dark:bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                <span className="ml-2">Calling tool...</span>
              </div>
            ) : (
              <div className="text-xs text-gray-500 dark:text-slate-500 italic">No results yet. Fill in the test parameters below and click Call Tool.</div>
            )}
          </div>
          
          {/* Test Inputs - Bottom */}
          <div className="p-3 space-y-3">
            {Object.entries(tool.inputSchema?.properties || {}).map(([key, schema]: [string, any]) => (
              <div key={key}>
                <label className="block text-xs font-medium text-gray-700 dark:text-slate-300 mb-1">
                  {key} {schema.required && <span className="text-red-500">*</span>}
                </label>
                <input
                  type="text"
                  value={args[key] || ''}
                  onChange={(e) => {
                    let value: any = e.target.value
                    if (value.startsWith('{') || value.startsWith('[')) {
                      try {
                        value = JSON.parse(value)
                      } catch {}
                    }
                    setArgs({ ...args, [key]: value })
                  }}
                  className="w-full px-3 py-2 text-sm bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-gray-900 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder={schema.description || schema.type || key}
                />
              </div>
            ))}
            <button
              onClick={handleCall}
              disabled={disabled || calling}
              className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-300 dark:disabled:bg-slate-700 disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors"
            >
              {calling ? 'Calling...' : 'Call Tool'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// Message Bubble Component - ChatGPT Style
function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user'
  const isSystem = message.role === 'system'
  const isTool = message.role === 'tool'
  const isAssistant = message.role === 'assistant'
  
  // Check if content looks like an incomplete tool call
  const isIncompleteToolCall = isAssistant && message.content && 
    (message.content.includes('TOOL_CALL:') || message.content.startsWith('TOOL_CALL'))

  if (isSystem) {
    return (
      <div className="flex justify-center my-4">
        <div className="px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-lg text-sm text-blue-300">
          {message.content}
        </div>
      </div>
    )
  }
  
  // Handle incomplete tool calls
  if (isIncompleteToolCall) {
    return (
      <div className="flex justify-center my-4">
        <div className="px-4 py-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-sm">
          <div className="flex items-center gap-2 mb-2">
            <WrenchScrewdriverIcon className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
            <span className="font-semibold text-yellow-700 dark:text-yellow-300">Tool Call Incomplete</span>
          </div>
          <p className="text-yellow-600 dark:text-yellow-400 text-xs">
            The AI started to call a tool but the response was truncated. Try asking again or being more specific.
          </p>
          <details className="mt-2">
            <summary className="text-xs text-yellow-600 dark:text-yellow-400 cursor-pointer hover:underline">
              Show raw output
            </summary>
            <pre className="mt-1 text-xs text-yellow-700 dark:text-yellow-300 whitespace-pre-wrap break-words">
              {message.content}
            </pre>
          </details>
        </div>
      </div>
    )
  }

  return (
    <div className={`flex gap-4 py-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center flex-shrink-0">
          {isTool ? (
            <WrenchScrewdriverIcon className="w-4 h-4 text-white" />
          ) : (
            <SparklesIcon className="w-4 h-4 text-white" />
          )}
        </div>
      )}
      
      <div className={`flex-1 ${isUser ? 'max-w-[85%] flex justify-end' : 'max-w-[85%]'}`}>
        <div className={`rounded-2xl px-4 py-3 ${
          isUser 
            ? 'bg-purple-600 text-white' 
            : isTool
            ? 'bg-green-500/10 border border-green-500/30 text-green-700 dark:text-green-100'
            : 'bg-slate-100 dark:bg-slate-800 text-gray-900 dark:text-slate-100'
        }`}>
          {message.toolName && (
            <div className="text-xs font-semibold mb-2 opacity-70">
              Tool: {message.toolName}
            </div>
          )}
          {message.data ? (
            <JSONViewer data={message.data} defaultExpanded={!isSystem} />
          ) : isAssistant ? (
            <MarkdownContent content={message.content} />
          ) : (
            <div className="whitespace-pre-wrap leading-relaxed">{message.content}</div>
          )}
        </div>
        <div className={`text-xs text-gray-500 dark:text-slate-500 mt-1 ${isUser ? 'text-right' : 'text-left'}`}>
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-slate-300 dark:bg-slate-700 flex items-center justify-center flex-shrink-0">
          <span className="text-xs font-semibold text-gray-700 dark:text-slate-300">U</span>
        </div>
      )}
    </div>
  )
}

