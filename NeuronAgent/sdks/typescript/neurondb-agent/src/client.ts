/**
 * NeuronAgent HTTP Client
 */

import { ClientConfig, Agent, Session, Message, Tool, Workflow } from './types';

export class NeuronAgentClient {
  private baseURL: string;
  private apiKey: string;
  private timeout: number;
  private retries: number;

  constructor(config: ClientConfig) {
    this.baseURL = config.baseURL || 'http://localhost:8080';
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 30000;
    this.retries = config.retries || 3;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: any
  ): Promise<T> {
    const url = `${this.baseURL}/api/v1${path}`;
    const headers: HeadersInit = {
      'Authorization': `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
    };

    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this.retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        const response = await fetch(url, {
          method,
          headers,
          body: body ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const error = await response.json().catch(() => ({ message: response.statusText }));
          throw new Error(error.message || `HTTP ${response.status}`);
        }

        return await response.json();
      } catch (error: any) {
        lastError = error;
        if (attempt < this.retries && this.isRetryable(error)) {
          await this.delay(Math.pow(2, attempt) * 100);
          continue;
        }
        throw error;
      }
    }

    throw lastError || new Error('Request failed');
  }

  private isRetryable(error: any): boolean {
    return error.name === 'AbortError' || 
           (error.message && error.message.includes('timeout'));
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Agent methods
  async createAgent(agent: Partial<Agent>): Promise<Agent> {
    return this.request<Agent>('POST', '/agents', agent);
  }

  async listAgents(): Promise<Agent[]> {
    return this.request<Agent[]>('GET', '/agents');
  }

  async getAgent(id: string): Promise<Agent> {
    return this.request<Agent>('GET', `/agents/${id}`);
  }

  async updateAgent(id: string, agent: Partial<Agent>): Promise<Agent> {
    return this.request<Agent>('PUT', `/agents/${id}`, agent);
  }

  async deleteAgent(id: string): Promise<void> {
    return this.request<void>('DELETE', `/agents/${id}`);
  }

  // Session methods
  async createSession(session: Partial<Session>): Promise<Session> {
    return this.request<Session>('POST', '/sessions', session);
  }

  async getSession(id: string): Promise<Session> {
    return this.request<Session>('GET', `/sessions/${id}`);
  }

  // Message methods
  async sendMessage(sessionId: string, content: string, stream?: boolean): Promise<Message> {
    return this.request<Message>('POST', `/sessions/${sessionId}/messages`, {
      role: 'user',
      content,
      stream: stream || false,
    });
  }

  async getMessages(sessionId: string): Promise<Message[]> {
    return this.request<Message[]>('GET', `/sessions/${sessionId}/messages`);
  }

  // Tool methods
  async listTools(): Promise<Tool[]> {
    return this.request<Tool[]>('GET', '/tools');
  }

  // Workflow methods
  async createWorkflow(workflow: Partial<Workflow>): Promise<Workflow> {
    return this.request<Workflow>('POST', '/workflows', workflow);
  }
}

