'use client'

import { useState, useEffect, useRef } from 'react'
import { PlayIcon, StopIcon } from '@heroicons/react/24/outline'

export default function MCPSSEStream() {
  const [isConnected, setIsConnected] = useState(false)
  const [messages, setMessages] = useState<string[]>([])
  const eventSourceRef = useRef<EventSource | null>(null)
  
  const connect = () => {
    if (eventSourceRef.current) return
    
    const eventSource = new EventSource('/api/v1/mcp/stream')
    eventSourceRef.current = eventSource
    
    eventSource.onopen = () => {
      setIsConnected(true)
    }
    
    eventSource.onmessage = (event) => {
      setMessages((prev) => [...prev, event.data])
    }
    
    eventSource.onerror = () => {
      setIsConnected(false)
      eventSource.close()
      eventSourceRef.current = null
    }
  }
  
  const disconnect = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
      setIsConnected(false)
    }
  }
  
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [])
  
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <button
          onClick={isConnected ? disconnect : connect}
          className={`btn-primary flex items-center gap-2 ${
            isConnected ? 'bg-red-600 hover:bg-red-700' : ''
          }`}
        >
          {isConnected ? (
            <>
              <StopIcon className="w-4 h-4" />
              Disconnect
            </>
          ) : (
            <>
              <PlayIcon className="w-4 h-4" />
              Connect
            </>
          )}
        </button>
        <span
          className={`px-3 py-1 rounded text-sm ${
            isConnected
              ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
              : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
          }`}
        >
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
      
      <div className="card">
        <h3 className="font-semibold mb-2">Stream Messages</h3>
        <div className="max-h-96 overflow-y-auto space-y-2">
          {messages.length > 0 ? (
            messages.map((msg, idx) => (
              <div key={idx} className="p-2 bg-slate-100 dark:bg-slate-700 rounded text-sm font-mono">
                {msg}
              </div>
            ))
          ) : (
            <p className="text-slate-500 text-sm">No messages yet</p>
          )}
        </div>
      </div>
    </div>
  )
}


