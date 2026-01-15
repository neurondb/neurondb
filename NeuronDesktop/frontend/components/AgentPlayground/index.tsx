'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { agentAPI, profilesAPI, type Profile, type Agent, type CreateAgentRequest } from '@/lib/api'
import { ChatBubbleLeftRightIcon, PaperAirplaneIcon, ArrowPathIcon, XMarkIcon } from '@/components/Icons'
import MarkdownContent from '@/components/MarkdownContent'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  toolCalls?: Array<{
    tool: string
    input: any
    output?: any
    error?: string
  }>
}

interface AgentPlaygroundProps {
  agentId?: string
  onClose?: () => void
}

export default function AgentPlayground({ agentId, onClose }: AgentPlaygroundProps) {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [agents, setAgents] = useState<Agent[]>([])
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const loadProfiles = useCallback(async () => {
    try {
      const response = await profilesAPI.list()
      setProfiles(response.data)
      if (response.data.length > 0 && !selectedProfile) {
        const activeProfileId = localStorage.getItem('active_profile_id')
        if (activeProfileId) {
          const active = response.data.find((p: Profile) => p.id === activeProfileId)
          if (active) {
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

  const createSession = useCallback(async (agentId: string) => {
    if (!selectedProfile) return
    try {
      const response = await agentAPI.createSession(selectedProfile, agentId)
      setSessionId(response.data.id)
      setMessages([])
    } catch (error) {
      console.error('Failed to create session:', error)
      alert('Failed to create session: ' + (error as any).message)
    }
  }, [selectedProfile])

  const loadAgents = useCallback(async () => {
    if (!selectedProfile) return
    try {
      const response = await agentAPI.listAgents(selectedProfile)
      setAgents(response.data)
      if (agentId) {
        const agent = response.data.find((a: Agent) => a.id === agentId)
        if (agent) {
          setSelectedAgent(agent)
          createSession(agent.id)
        }
      }
    } catch (error) {
      console.error('Failed to load agents:', error)
    }
  }, [selectedProfile, agentId, createSession])

  useEffect(() => {
    loadProfiles()
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [loadProfiles])

  useEffect(() => {
    if (selectedProfile) {
      loadAgents()
    }
  }, [selectedProfile, loadAgents])

  useEffect(() => {
    if (agentId && agents.length > 0) {
      const agent = agents.find((a) => a.id === agentId)
      if (agent) {
        setSelectedAgent(agent)
        createSession(agent.id)
      }
    }
  }, [agentId, agents, createSession])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])

  const handleAgentSelect = (agent: Agent) => {
    setSelectedAgent(agent)
    createSession(agent.id)
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !sessionId || loading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputMessage('')
    setLoading(true)
    setStreaming(true)
    setStreamingContent('')

    try {
      const profile = profiles.find((p) => p.id === selectedProfile)
      if (!profile?.agent_endpoint) {
        throw new Error('Agent endpoint not configured')
      }

      // Try WebSocket first for streaming
      const wsUrl = profile.agent_endpoint.replace('http://', 'ws://').replace('https://', 'wss://')
      const ws = new WebSocket(`${wsUrl}/api/v1/ws?session_id=${sessionId}`)

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            type: 'message',
            content: userMessage.content,
            role: 'user',
          })
        )
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'content') {
            setStreamingContent((prev) => prev + (data.content || ''))
          } else if (data.type === 'tool_call') {
            // Handle tool calls
            console.log('Tool call:', data)
          } else if (data.type === 'done') {
            setStreaming(false)
            setLoading(false)
            if (streamingContent) {
              const assistantMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: streamingContent,
                timestamp: new Date(),
              }
              setMessages((prev) => [...prev, assistantMessage])
              setStreamingContent('')
            }
            ws.close()
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.onerror = () => {
        // Fallback to HTTP if WebSocket fails
        ws.close()
        sendMessageHTTP(userMessage.content)
      }

      wsRef.current = ws
    } catch (error) {
      // Fallback to HTTP
      sendMessageHTTP(userMessage.content)
    }
  }

  const sendMessageHTTP = async (content: string) => {
    if (!sessionId || !selectedProfile) return

    try {
      const response = await agentAPI.sendMessage(selectedProfile, sessionId, content, false)

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.content || 'No response',
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error: any) {
      console.error('Failed to send message:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: `Error: ${error.response?.data?.error || error.message || 'Failed to send message'}`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
      setStreaming(false)
      setStreamingContent('')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const resetSession = () => {
    if (selectedAgent) {
      createSession(selectedAgent.id)
    }
  }

  return (
    <div className="h-full flex flex-col bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-slate-700 p-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-100">Agent Playground</h2>
          {selectedAgent && (
            <span className="text-sm text-gray-600 dark:text-slate-400">
              Testing: <span className="font-medium">{selectedAgent.name}</span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {!selectedAgent && (
            <select
              value={selectedProfile}
              onChange={(e) => setSelectedProfile(e.target.value)}
              className="input text-sm"
            >
              {profiles.map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.name}
                </option>
              ))}
            </select>
          )}
          {selectedAgent && (
            <button
              onClick={resetSession}
              className="btn btn-secondary flex items-center gap-2 text-sm"
              title="Reset Session"
            >
              <ArrowPathIcon className="w-4 h-4" />
              Reset
            </button>
          )}
          {onClose && (
            <button onClick={onClose} className="p-2 text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-100">
              <XMarkIcon className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Agent Selection */}
      {!selectedAgent && (
        <div className="border-b border-gray-200 dark:border-slate-700 p-4">
          <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
            Select Agent to Test
          </label>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {agents.map((agent) => (
              <button
                key={agent.id}
                onClick={() => handleAgentSelect(agent)}
                className="card hover:shadow-md transition-all text-left"
              >
                <h3 className="font-semibold text-gray-900 dark:text-slate-100 mb-1">{agent.name}</h3>
                {agent.description && (
                  <p className="text-sm text-gray-600 dark:text-slate-400">{agent.description}</p>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !streaming && (
          <div className="text-center py-12">
            <ChatBubbleLeftRightIcon className="w-16 h-16 text-gray-400 dark:text-slate-600 mx-auto mb-4" />
            <p className="text-gray-600 dark:text-slate-400">
              {selectedAgent ? 'Start a conversation with your agent' : 'Select an agent to begin testing'}
            </p>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-3xl rounded-lg p-4 ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : message.role === 'system'
                  ? 'bg-yellow-100 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200'
                  : 'bg-gray-100 dark:bg-slate-800 text-gray-900 dark:text-slate-100'
              }`}
            >
              {message.role === 'assistant' ? (
                <MarkdownContent content={message.content} />
              ) : (
                <p className="whitespace-pre-wrap">{message.content}</p>
              )}
              <p className="text-xs mt-2 opacity-70">
                {message.timestamp.toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

        {streaming && streamingContent && (
          <div className="flex justify-start">
            <div className="max-w-3xl rounded-lg p-4 bg-gray-100 dark:bg-slate-800 text-gray-900 dark:text-slate-100">
              <MarkdownContent content={streamingContent} />
              <span className="inline-block w-2 h-4 bg-blue-600 animate-pulse ml-1" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      {selectedAgent && sessionId && (
        <div className="border-t border-gray-200 dark:border-slate-700 p-4">
          <div className="flex gap-2">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="input flex-1 min-h-[60px] resize-none"
              disabled={loading}
            />
            <button
              onClick={handleSendMessage}
              disabled={loading || !inputMessage.trim()}
              className="btn btn-primary flex items-center gap-2 px-6"
            >
              <PaperAirplaneIcon className="w-5 h-5" />
              {loading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

