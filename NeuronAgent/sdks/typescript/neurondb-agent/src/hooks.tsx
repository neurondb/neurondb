/**
 * React hooks for NeuronAgent
 */

import { useState, useEffect, useCallback } from 'react';
import { NeuronAgentClient } from './client';
import { Agent, Session, Message, StreamingResponse } from './types';

/**
 * Hook for managing agents
 */
export function useAgents(client: NeuronAgentClient) {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchAgents = useCallback(async () => {
    try {
      setLoading(true);
      const data = await client.listAgents();
      setAgents(data);
      setError(null);
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
    }
  }, [client]);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  return { agents, loading, error, refetch: fetchAgents };
}

/**
 * Hook for managing a session with streaming
 */
export function useSession(client: NeuronAgentClient, sessionId?: string) {
  const [session, setSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [streaming, setStreaming] = useState(false);

  const sendMessage = useCallback(async (content: string, stream = false) => {
    if (!sessionId) return;

    try {
      setLoading(true);
      setError(null);

      if (stream) {
        setStreaming(true);
        // TODO: Implement WebSocket streaming
        const message = await client.sendMessage(sessionId, content, true);
        setMessages(prev => [...prev, message]);
      } else {
        const message = await client.sendMessage(sessionId, content, false);
        setMessages(prev => [...prev, message]);
      }
    } catch (err) {
      setError(err as Error);
    } finally {
      setLoading(false);
      setStreaming(false);
    }
  }, [client, sessionId]);

  useEffect(() => {
    if (sessionId) {
      client.getSession(sessionId).then(setSession);
      client.getMessages(sessionId).then(setMessages);
    }
  }, [client, sessionId]);

  return {
    session,
    messages,
    loading,
    error,
    streaming,
    sendMessage,
  };
}

