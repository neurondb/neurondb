# NeuronDesktop Frontend Architecture

**Complete frontend architecture documentation for NeuronDesktop.**

> **Version:** 1.0  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Component Architecture](#component-architecture)
- [State Management](#state-management)
- [API Client](#api-client)
- [WebSocket Integration](#websocket-integration)
- [Theme System](#theme-system)

---

## Technology Stack

### Core Technologies

- **Framework:** Next.js 14+ (React)
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **State Management:** React Context + Zustand
- **API Client:** Axios
- **WebSocket:** Native WebSocket API
- **Forms:** React Hook Form
- **Charts:** Recharts

### Build Tools

- **Bundler:** Next.js (Webpack/Turbopack)
- **Type Checking:** TypeScript
- **Linting:** ESLint
- **Formatting:** Prettier

---

## Project Structure

```
NeuronDesktop/
├── app/                    # Next.js App Router
│   ├── (auth)/            # Auth routes
│   ├── (dashboard)/       # Dashboard routes
│   └── api/               # API routes (proxy)
├── components/            # React components
│   ├── ui/               # UI components
│   ├── agents/           # Agent components
│   ├── databases/        # Database components
│   └── mcp/              # MCP components
├── lib/                   # Utilities
│   ├── api/              # API client
│   ├── websocket/        # WebSocket client
│   └── utils/            # Utilities
├── hooks/                 # React hooks
├── store/                 # State management
└── types/                 # TypeScript types
```

---

## Component Architecture

### Page Components

**Dashboard:**
- `app/(dashboard)/dashboard/page.tsx`: Main dashboard
- `app/(dashboard)/agents/page.tsx`: Agent management
- `app/(dashboard)/databases/page.tsx`: Database management
- `app/(dashboard)/mcp/page.tsx`: MCP integration

### UI Components

**Common Components:**
- `components/ui/Button.tsx`: Button component
- `components/ui/Input.tsx`: Input component
- `components/ui/Modal.tsx`: Modal component
- `components/ui/Table.tsx`: Table component

**Feature Components:**
- `components/agents/AgentList.tsx`: Agent list
- `components/agents/AgentForm.tsx`: Agent form
- `components/databases/DatabaseList.tsx`: Database list
- `components/mcp/ToolList.tsx`: MCP tool list

---

## State Management

### Context API

**Profile Context:**
```typescript
interface ProfileContext {
  profiles: Profile[];
  currentProfile: Profile | null;
  setCurrentProfile: (profile: Profile) => void;
}
```

**Auth Context:**
```typescript
interface AuthContext {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}
```

### Zustand Store

**Profile Store:**
```typescript
interface ProfileStore {
  profiles: Profile[];
  currentProfile: Profile | null;
  setCurrentProfile: (profile: Profile) => void;
  fetchProfiles: () => Promise<void>;
}
```

---

## API Client

### API Client Structure

**Base Client:**
```typescript
class APIClient {
  private baseURL: string;
  private token: string | null;

  async get<T>(path: string): Promise<T>;
  async post<T>(path: string, data: any): Promise<T>;
  async put<T>(path: string, data: any): Promise<T>;
  async delete<T>(path: string): Promise<T>;
}
```

**Endpoints:**
- `profiles`: Profile management
- `agents`: Agent management
- `databases`: Database management
- `mcp`: MCP integration

---

## WebSocket Integration

### WebSocket Client

**Connection:**
```typescript
class WebSocketClient {
  connect(url: string, token: string): void;
  send(message: any): void;
  onMessage(callback: (message: any) => void): void;
  disconnect(): void;
}
```

**Usage:**
```typescript
const ws = new WebSocketClient();
ws.connect('ws://localhost:8081/api/v1/ws', token);
ws.onMessage((message) => {
  // Handle message
});
```

---

## Theme System

### Theme Configuration

**Tailwind Config:**
```typescript
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {...},
        secondary: {...},
      },
    },
  },
};
```

**Dark Mode:**
- System preference detection
- Manual toggle
- Persistent storage

---

## Related Documentation

- [NeuronDesktop API Reference](../reference/neurondesktop-api-complete.md)
- [NeuronDesktop Deployment](../../NeuronDesktop/docs/deployment.md)

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0



