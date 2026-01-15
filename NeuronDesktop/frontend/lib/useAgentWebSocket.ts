'use client'

import { useEffect, useRef, useState, useCallback } from 'react'

export interface WebSocketMessage {
  type?: string
  content?: string
  error?: string
  [key: string]: any
}

export interface UseAgentWebSocketOptions {
  profileId: string
  sessionId: string | null
  enabled?: boolean
  onMessage?: (message: WebSocketMessage) => void
  onError?: (error: Error) => void
  onConnect?: () => void
  onDisconnect?: () => void
}

export interface UseAgentWebSocketReturn {
  sendMessage: (content: string) => void
  isConnected: boolean
  isConnecting: boolean
  error: string | null
  reconnect: () => void
  disconnect: () => void
}

export function useAgentWebSocket({
  profileId,
  sessionId,
  enabled = true,
  onMessage,
  onError,
  onConnect,
  onDisconnect,
}: UseAgentWebSocketOptions): UseAgentWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 5
  const reconnectDelay = 3000

  const getWebSocketUrl = useCallback((token?: string | null) => {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081'
    const wsProtocol = baseUrl.startsWith('https') ? 'wss' : 'ws'
    // Remove protocol and path, keep host:port
    const wsBaseUrl = baseUrl.replace(/^https?:\/\//, '').split('/')[0]
    const tokenParam = token ? `&token=${encodeURIComponent(token)}` : ''
    return `${wsProtocol}://${wsBaseUrl}/api/v1/profiles/${profileId}/agent/ws?session_id=${sessionId}${tokenParam}`
  }, [profileId, sessionId])

  const connect = useCallback(() => {
    if (!enabled || !sessionId || wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    if (wsRef.current?.readyState === WebSocket.CONNECTING) {
      return
    }

    setIsConnecting(true)
    setError(null)

    try {
      const token = localStorage.getItem('neurondesk_auth_token')
      const url = getWebSocketUrl(token)
      
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnected(true)
        setIsConnecting(false)
        reconnectAttemptsRef.current = 0
        onConnect?.()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessage?.(data)
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err)
          onError?.(new Error('Failed to parse message'))
        }
      }

      ws.onerror = (event) => {
        console.error('WebSocket error:', event)
        const errorMessage = 'WebSocket connection error'
        setError(errorMessage)
        setIsConnecting(false)
        onError?.(new Error(errorMessage))
      }

      ws.onclose = (event) => {
        setIsConnected(false)
        setIsConnecting(false)
        wsRef.current = null

        if (event.code !== 1000 && enabled && reconnectAttemptsRef.current < maxReconnectAttempts) {
          // Attempt to reconnect
          reconnectAttemptsRef.current++
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectDelay * reconnectAttemptsRef.current)
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          setError('Failed to reconnect after multiple attempts')
          onError?.(new Error('Connection lost'))
        }

        onDisconnect?.()
      }
    } catch (err) {
      setIsConnecting(false)
      const errorMessage = err instanceof Error ? err.message : 'Failed to create WebSocket connection'
      setError(errorMessage)
      onError?.(err instanceof Error ? err : new Error(errorMessage))
    }
  }, [enabled, sessionId, getWebSocketUrl, onMessage, onError, onConnect, onDisconnect])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect')
      wsRef.current = null
    }
    
    setIsConnected(false)
    setIsConnecting(false)
    reconnectAttemptsRef.current = 0
  }, [])

  const reconnect = useCallback(() => {
    disconnect()
    reconnectAttemptsRef.current = 0
    setTimeout(() => {
      connect()
    }, 1000)
  }, [disconnect, connect])

  const sendMessage = useCallback((content: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('WebSocket is not connected')
      return
    }

    try {
      wsRef.current.send(JSON.stringify({ content }))
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message'
      setError(errorMessage)
      onError?.(err instanceof Error ? err : new Error(errorMessage))
    }
  }, [onError])

  useEffect(() => {
    if (enabled && sessionId) {
      connect()
    } else {
      disconnect()
    }

    return () => {
      disconnect()
    }
  }, [enabled, sessionId, connect, disconnect])

  return {
    sendMessage,
    isConnected,
    isConnecting,
    error,
    reconnect,
    disconnect,
  }
}

