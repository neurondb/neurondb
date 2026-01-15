/**
 * NeuronMCP TypeScript/JavaScript SDK
 * 
 * Provides full MCP protocol support for Node.js and browser environments.
 */

export interface MCPRequest {
  jsonrpc: "2.0";
  method: string;
  params: Record<string, any>;
  id: string;
}

export interface MCPResponse {
  jsonrpc: "2.0";
  result?: Record<string, any>;
  error?: {
    code: number | string;
    message: string;
    data?: any;
  };
  id?: string;
}

export interface ToolResult {
  content: Array<Record<string, any>>;
  isError: boolean;
  metadata?: Record<string, any>;
}

export interface ClientConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  maxRetries?: number;
  retryBackoff?: number;
}

export class MCPError extends Error {
  constructor(
    message: string,
    public code?: string | number
  ) {
    super(message);
    this.name = "MCPError";
  }
}

export class MCPConnectionError extends MCPError {
  constructor(message: string) {
    super(message);
    this.name = "MCPConnectionError";
  }
}

export class MCPTimeoutError extends MCPError {
  constructor(message: string) {
    super(message);
    this.name = "MCPTimeoutError";
  }
}

export class MCPToolError extends MCPError {
  constructor(message: string, code?: string | number) {
    super(message, code);
    this.name = "MCPToolError";
  }
}

export class NeuronMCPClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;
  private maxRetries: number;
  private retryBackoff: number;

  constructor(config: ClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 30000;
    this.maxRetries = config.maxRetries || 3;
    this.retryBackoff = config.retryBackoff || 1.5;
  }

  /**
   * Call a tool on the MCP server
   */
  async callTool(
    toolName: string,
    params: Record<string, any>
  ): Promise<ToolResult> {
    const request: MCPRequest = {
      jsonrpc: "2.0",
      method: "tools/call",
      params: {
        name: toolName,
        arguments: params,
      },
      id: this.generateId(),
    };

    return this.callWithRetry(request);
  }

  /**
   * List all available tools
   */
  async listTools(): Promise<Array<Record<string, any>>> {
    const request: MCPRequest = {
      jsonrpc: "2.0",
      method: "tools/list",
      params: {},
      id: this.generateId(),
    };

    const response = await this.callOnce(request);
    return response.result?.tools || [];
  }

  /**
   * Get schema for a specific tool
   */
  async getToolSchema(toolName: string): Promise<Record<string, any>> {
    const tools = await this.listTools();
    const tool = tools.find((t) => t.name === toolName);
    if (!tool) {
      throw new MCPToolError(`Tool '${toolName}' not found`);
    }
    return tool;
  }

  private async callWithRetry(request: MCPRequest): Promise<ToolResult> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        return await this.callOnce(request);
      } catch (error) {
        if (
          error instanceof MCPConnectionError ||
          error instanceof MCPTimeoutError
        ) {
          lastError = error;
          if (attempt < this.maxRetries - 1) {
            const waitTime = Math.pow(this.retryBackoff, attempt) * 1000;
            await this.sleep(waitTime);
          } else {
            throw error;
          }
        } else {
          throw error;
        }
      }
    }

    throw lastError || new MCPError("Unknown error");
  }

  private async callOnce(request: MCPRequest): Promise<ToolResult> {
    const url = `${this.baseUrl}/mcp`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.status === 200) {
        const data: MCPResponse = await response.json();

        if (data.error) {
          throw new MCPToolError(
            data.error.message || "Unknown error",
            data.error.code
          );
        }

        return {
          content: data.result?.content || [],
          isError: data.result?.isError || false,
          metadata: data.result?.metadata,
        };
      } else if (response.status === 408 || response.status === 504) {
        throw new MCPTimeoutError(`Request timeout: ${response.status}`);
      } else {
        const errorText = await response.text();
        throw new MCPConnectionError(`HTTP ${response.status}: ${errorText}`);
      }
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof MCPError) {
        throw error;
      }
      if (error instanceof Error && error.name === "AbortError") {
        throw new MCPTimeoutError("Request timeout");
      }
      throw new MCPConnectionError(`Connection error: ${error}`);
    }
  }

  private generateId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}




