'use client'

import { useState, useEffect, useCallback } from 'react'
import { Message, Agent } from '@/lib/api'
import { useAgentWebSocket, WebSocketMessage } from '@/lib/useAgentWebSocket'
import { logger } from '@/lib/logger'
import MessageList from './MessageList'
import MessageInput from './MessageInput'
import StatusBadge from './StatusBadge'

interface ChatInterfaceProps {
  profileId: string
  agent: Agent
  sessionId: string | null
  initialMessages?: Message[]
}

export default function ChatInterface({
  profileId,
  agent,
  sessionId,
  initialMessages = [],
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [streamingContent, setStreamingContent] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)

  const handleWebSocketMessage = useCallback((wsMessage: WebSocketMessage) => {
    if (wsMessage.error) {
      logger.error('WebSocket error in ChatInterface', undefined, { error: wsMessage.error })
      setIsStreaming(false)
      setStreamingContent('')
      return
    }

    // Handle NeuronAgent WebSocket message format
    // NeuronAgent sends: { type: "response", content: "...", complete: true }
    if (wsMessage.type === 'response' && wsMessage.content) {
      const content = wsMessage.content as string
      if (wsMessage.complete) {
        // Complete response received
        const assistantMessage: Message = {
          id: `msg-${Date.now()}`,
          session_id: sessionId || '',
          role: 'assistant',
          content: content,
          created_at: new Date().toISOString(),
        }
        setMessages((prev) => [...prev, assistantMessage])
        setStreamingContent('')
        setIsStreaming(false)
      } else {
        // Streaming chunk
        setStreamingContent((prev) => prev + content)
        setIsStreaming(true)
      }
    } else if (wsMessage.type === 'chunk' || (wsMessage.content && !wsMessage.type)) {
      // Handle streaming chunks or generic content
      const content = (wsMessage.content || wsMessage.text || '') as string
      setStreamingContent((prev) => prev + content)
      setIsStreaming(true)
    } else if (wsMessage.type === 'done' || wsMessage.type === 'complete') {
      // Finalize the streaming message
      if (streamingContent) {
        const assistantMessage: Message = {
          id: `msg-${Date.now()}`,
          session_id: sessionId || '',
          role: 'assistant',
          content: streamingContent,
          created_at: new Date().toISOString(),
        }
        setMessages((prev) => [...prev, assistantMessage])
        setStreamingContent('')
        setIsStreaming(false)
      }
    } else if (wsMessage.type === 'message' && wsMessage.role === 'assistant') {
      // Complete message received
      const assistantMessage: Message = {
        id: (wsMessage.id as string) || `msg-${Date.now()}`,
        session_id: sessionId || '',
        role: 'assistant',
        content: (wsMessage.content || wsMessage.text || '') as string,
        created_at: (wsMessage.created_at as string) || new Date().toISOString(),
      }
      setMessages((prev) => [...prev, assistantMessage])
      setStreamingContent('')
      setIsStreaming(false)
    }
  }, [sessionId, streamingContent])

  const { sendMessage, isConnected, isConnecting, error, reconnect } = useAgentWebSocket({
    profileId,
    sessionId,
    enabled: !!sessionId,
    onMessage: handleWebSocketMessage,
    onError: (err) => {
      logger.error('WebSocket error in ChatInterface', err)
      setIsStreaming(false)
      setStreamingContent('')
    },
  })

  const handleSend = useCallback(
    async (content: string) => {
      if (!sessionId) {
        logger.error('No session ID available in ChatInterface')
        return
      }

      // Add user message to UI immediately
      const userMessage: Message = {
        id: `temp-user-${Date.now()}`,
        session_id: sessionId,
        role: 'user',
        content,
        created_at: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, userMessage])
      setStreamingContent('')
      setIsStreaming(true)

      // Send via WebSocket
      sendMessage(content)
    },
    [sessionId, sendMessage]
  )

  // Load messages when session changes
  useEffect(() => {
    if (sessionId && initialMessages.length === 0) {
      // Messages will be loaded by parent component
      setMessages([])
    } else {
      setMessages(initialMessages)
    }
  }, [sessionId, initialMessages])

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="border-b border-slate-800 p-4 bg-slate-900">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-100">{agent.name}</h2>
            {agent.description && (
              <p className="text-sm text-slate-400 mt-1">{agent.description}</p>
            )}
          </div>
          <div className="flex items-center gap-2">
            <StatusBadge
              status={isConnected ? 'connected' : isConnecting ? 'loading' : 'disconnected'}
            />
            {error && (
              <button
                onClick={reconnect}
                className="text-xs text-blue-400 hover:text-blue-300 underline"
              >
                Reconnect
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Messages */}
      <MessageList
        messages={messages}
        streamingContent={streamingContent}
        isStreaming={isStreaming}
      />

      {/* Input */}
      <MessageInput
        onSend={handleSend}
        disabled={!isConnected || !sessionId}
        placeholder={!sessionId ? 'No session available' : isConnected ? 'Type your message...' : 'Connecting...'}
      />
    </div>
  )
}

