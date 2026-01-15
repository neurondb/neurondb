/**
 * Type definitions for NeuronAgent SDK
 */

export interface Agent {
  id: string;
  name: string;
  description?: string;
  system_prompt: string;
  model_name: string;
  enabled_tools: string[];
  config: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface Session {
  id: string;
  agent_id: string;
  external_user_id?: string;
  metadata: Record<string, any>;
  created_at: string;
  last_activity_at: string;
}

export interface Message {
  id: string;
  session_id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  tool_name?: string;
  tool_call_id?: string;
  token_count?: number;
  metadata: Record<string, any>;
  created_at: string;
}

export interface Tool {
  name: string;
  description: string;
  schema: Record<string, any>;
  enabled: boolean;
  created_at: string;
}

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  steps: WorkflowStep[];
  status: 'active' | 'paused' | 'completed';
  created_at: string;
}

export interface WorkflowStep {
  id: string;
  type: 'agent' | 'tool' | 'http' | 'approval' | 'conditional';
  config: Record<string, any>;
  dependencies: string[];
}

export interface StreamingResponse {
  type: 'token' | 'tool_call' | 'tool_result' | 'done' | 'error';
  content?: string;
  data?: any;
}

export interface ClientConfig {
  baseURL?: string;
  apiKey: string;
  timeout?: number;
  retries?: number;
}

