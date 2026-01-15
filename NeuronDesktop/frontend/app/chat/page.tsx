'use client'

import { useState, useEffect, useCallback } from 'react'
import { agentAPI, profilesAPI, type Profile, type Agent, type Session, type Message } from '@/lib/api'
import ChatInterface from '@/components/ChatInterface'
import { PlusIcon, ChatBubbleLeftRightIcon } from '@/components/Icons'

export default function ChatPage() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [selectedProfile, setSelectedProfile] = useState<string>('')
  const [agents, setAgents] = useState<Agent[]>([])
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [selectedSession, setSelectedSession] = useState<Session | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
      setError('Failed to load profiles')
    }
  }, [selectedProfile])

  const loadAgents = useCallback(async () => {
    if (!selectedProfile) return
    setLoading(true)
    setError(null)
    try {
      const response = await agentAPI.listAgents(selectedProfile)
      setAgents(response.data)
      if (response.data.length > 0 && !selectedAgent) {
        setSelectedAgent(response.data[0])
      } else if (response.data.length === 0) {
        setSelectedAgent(null)
      }
    } catch (error: any) {
      console.error('Failed to load agents:', error)
      setError('Failed to load agents. Make sure NeuronAgent is running and configured.')
      setSelectedAgent(null)
    } finally {
      setLoading(false)
    }
  }, [selectedProfile, selectedAgent])

  const loadSessions = useCallback(async () => {
    if (!selectedProfile || !selectedAgent) return
    setLoading(true)
    try {
      const response = await agentAPI.listSessions(selectedProfile, selectedAgent.id)
      setSessions(response.data)
    } catch (error) {
      console.error('Failed to load sessions:', error)
      // Don't set error here, sessions might not exist yet
    } finally {
      setLoading(false)
    }
  }, [selectedProfile, selectedAgent])

  const loadMessages = useCallback(async () => {
    if (!selectedProfile || !selectedSession) return
    setLoading(true)
    try {
      const response = await agentAPI.getMessages(selectedProfile, selectedSession.id)
      setMessages(response.data)
    } catch (error) {
      console.error('Failed to load messages:', error)
      setError('Failed to load messages')
    } finally {
      setLoading(false)
    }
  }, [selectedProfile, selectedSession])

  useEffect(() => {
    loadProfiles()
  }, [loadProfiles])

  useEffect(() => {
    if (selectedProfile) {
      loadAgents()
    }
  }, [selectedProfile, loadAgents])

  useEffect(() => {
    if (selectedProfile && selectedAgent) {
      loadSessions()
    }
  }, [selectedProfile, selectedAgent, loadSessions])

  useEffect(() => {
    if (selectedSession) {
      loadMessages()
    } else {
      setMessages([])
    }
  }, [selectedSession, loadMessages])

  const handleCreateSession = async () => {
    if (!selectedProfile || !selectedAgent) return
    setLoading(true)
    setError(null)
    try {
      const response = await agentAPI.createSession(selectedProfile, selectedAgent.id)
      const newSession = response.data
      setSelectedSession(newSession)
      setSessions((prev) => [newSession, ...prev])
      setMessages([])
    } catch (error: any) {
      console.error('Failed to create session:', error)
      setError('Failed to create session: ' + (error.response?.data?.error || error.message))
    } finally {
      setLoading(false)
    }
  }

  const handleSelectSession = (session: Session) => {
    setSelectedSession(session)
  }

  const handleAgentChange = (agent: Agent) => {
    setSelectedAgent(agent)
    setSelectedSession(null)
    setMessages([])
  }

  if (!selectedProfile) {
    return (
      <div className="h-full overflow-auto bg-transparent p-6">
        <div className="card text-center py-12">
          <p className="text-slate-400">Loading profile…</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full w-full flex flex-col bg-slate-950 overflow-hidden">
      {/* Header */}
      <div className="border-b border-slate-800 p-6 bg-slate-900">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-100 mb-2">Chat</h1>
            <p className="text-slate-400">Chat with AI agents using NeuronAgent</p>
          </div>
          <div className="text-sm text-slate-300">
            Profile: <span className="text-slate-100 font-medium">{profiles.find((p) => p.id === selectedProfile)?.name || '...'}</span>
          </div>
        </div>

        {/* Agent Selection */}
        {agents.length > 0 && (
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium text-slate-300">Agent:</label>
            <select
              value={selectedAgent?.id || ''}
              onChange={(e) => {
                const agent = agents.find((a) => a.id === e.target.value)
                if (agent) handleAgentChange(agent)
              }}
              className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {agents.map((agent) => (
                <option key={agent.id} value={agent.id}>
                  {agent.name}
                </option>
              ))}
            </select>
            <button
              onClick={handleCreateSession}
              disabled={loading || !selectedAgent}
              className="btn btn-primary flex items-center gap-2"
            >
              <PlusIcon className="w-4 h-4" />
              New Chat
            </button>
          </div>
        )}

        {/* Session Selection */}
        {sessions.length > 0 && selectedAgent && (
          <div className="flex items-center gap-2 mt-4 overflow-x-auto">
            <label className="text-sm font-medium text-slate-300 flex-shrink-0">Sessions:</label>
            <div className="flex gap-2">
              {sessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => handleSelectSession(session)}
                  className={`px-3 py-1 rounded-lg text-sm transition-colors flex-shrink-0 ${
                    selectedSession?.id === session.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {new Date(session.created_at || '').toLocaleDateString()}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/20 border border-red-700 text-red-300 px-4 py-3 mx-6 mt-4 rounded-lg">
          {error}
        </div>
      )}

      {/* Chat Interface or Empty State */}
      {loading && !selectedSession ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-slate-400">Loading...</p>
          </div>
        </div>
      ) : !selectedAgent ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <ChatBubbleLeftRightIcon className="w-16 h-16 text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-200 mb-2">No Agent Selected</h3>
            <p className="text-slate-400 mb-6">
              {agents.length === 0
                ? 'No agents available. Create an agent first in the Agents page.'
                : 'Please select an agent to start chatting'}
            </p>
            {agents.length === 0 && (
              <a
                href="/agents"
                className="btn btn-primary inline-flex items-center gap-2"
              >
                Go to Agents
              </a>
            )}
          </div>
        </div>
      ) : !selectedSession ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <ChatBubbleLeftRightIcon className="w-16 h-16 text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-200 mb-2">Start a New Conversation</h3>
            <p className="text-slate-400 mb-6">Click &quot;New Chat&quot; to begin chatting with {selectedAgent.name}</p>
            <button onClick={handleCreateSession} className="btn btn-primary">
              Start Chat
            </button>
          </div>
        </div>
      ) : (
        <div className="flex-1 min-h-0">
          <ChatInterface
            profileId={selectedProfile}
            agent={selectedAgent}
            sessionId={selectedSession.id}
            initialMessages={messages}
          />
        </div>
      )}
    </div>
  )
}

