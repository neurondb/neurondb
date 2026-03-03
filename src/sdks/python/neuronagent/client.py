"""
NeuronAgent HTTP client: agents, sessions, and streaming messages.
"""

import json
import requests
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .types import Agent, Message, SendMessageResponse, Session


def _iter_sse_lines(response: requests.Response) -> Iterator[Tuple[str, str]]:
    """Parse SSE stream; yield (event_type, data) for each event."""
    event_type = ""
    data_buf: List[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line == "":
            if data_buf:
                yield (event_type or "message", "\n".join(data_buf))
            event_type = ""
            data_buf = []
            continue
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_buf.append(line[5:].strip())
    if data_buf:
        yield (event_type or "message", "\n".join(data_buf))


class _AgentsNamespace:
    """Namespace for agent operations."""

    def __init__(self, client: "NeuronAgentClient") -> None:
        self._client = client

    def create_agent(
        self,
        name: str,
        description: str = "",
        system_prompt: str = "",
        model_name: str = "gpt-4",
        enabled_tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "Agent":
        return self._client.create_agent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            model_name=model_name,
            enabled_tools=enabled_tools,
            config=config,
        )


class _SessionsNamespace:
    """Namespace for session operations."""

    def __init__(self, client: "NeuronAgentClient") -> None:
        self._client = client

    def create_session(
        self,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        external_user_id: Optional[str] = None,
    ) -> Session:
        return self._client.create_session(
            agent_id=agent_id,
            metadata=metadata,
            external_user_id=external_user_id,
        )

    def send_message(
        self,
        session_id: str,
        content: str,
        role: str = "user",
        stream: bool = False,
    ) -> SendMessageResponse:
        return self._client.send_message(
            session_id=session_id,
            content=content,
            role=role,
            stream=stream,
        )

    def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Message]:
        return self._client.get_messages(
            session_id=session_id,
            limit=limit,
            offset=offset,
        )


class NeuronAgentClient:
    """
    Client for the NeuronAgent HTTP API (sessions, messages, streaming).
    Use client.agents.create_agent(), client.sessions.create_session(), etc.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers.setdefault("Content-Type", "application/json")
        self.agents = _AgentsNamespace(self)
        self.sessions = _SessionsNamespace(self)

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1{path}"

    def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ):
        url = self._url(path)
        resp = self._session.request(
            method,
            url,
            json=json_data,
            timeout=self.timeout,
            stream=stream,
        )
        resp.raise_for_status()
        return resp

    # --- Agents (minimal for examples) ---

    def create_agent(
        self,
        name: str,
        description: str = "",
        system_prompt: str = "",
        model_name: str = "gpt-4",
        enabled_tools: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """Create an agent. Returns Agent with id, name, etc."""
        body = {
            "name": name,
            "description": description,
            "system_prompt": system_prompt,
            "model_name": model_name,
            "enabled_tools": enabled_tools or [],
            "config": config or {},
        }
        r = self._request("POST", "/agents", json_data=body)
        data = r.json()
        return Agent(
            id=str(data.get("id", "")),
            name=data.get("name", ""),
            description=data.get("description"),
            system_prompt=data.get("system_prompt", ""),
            model_name=data.get("model_name", "gpt-4"),
            enabled_tools=data.get("enabled_tools") or [],
            config=data.get("config") or {},
        )

    # --- Sessions ---

    def create_session(
        self,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        external_user_id: Optional[str] = None,
    ) -> Session:
        """Create a session for an agent."""
        body = {
            "agent_id": agent_id,
            "metadata": metadata or {},
        }
        if external_user_id is not None:
            body["external_user_id"] = external_user_id
        r = self._request("POST", "/sessions", json_data=body)
        data = r.json()
        return Session(
            id=data["id"],
            agent_id=data["agent_id"],
            external_user_id=data.get("external_user_id"),
            metadata=data.get("metadata") or {},
            created_at=data.get("created_at", ""),
            last_activity_at=data.get("last_activity_at", ""),
        )

    def send_message(
        self,
        session_id: str,
        content: str,
        role: str = "user",
        stream: bool = False,
    ) -> SendMessageResponse:
        """
        Send a message and return the full response (non-stream).
        For streaming, use send_message_stream() instead.
        """
        body = {
            "role": role,
            "content": content,
            "stream": False,
        }
        r = self._request(
            "POST",
            f"/sessions/{session_id}/messages",
            json_data=body,
        )
        data = r.json()
        return SendMessageResponse(
            session_id=str(data.get("session_id", session_id)),
            agent_id=str(data.get("agent_id", "")),
            response=data.get("response", ""),
            tokens_used=data.get("tokens_used"),
            tool_calls=data.get("tool_calls") or [],
            tool_results=data.get("tool_results") or [],
        )

    def send_message_stream(
        self,
        session_id: str,
        content: str,
        role: str = "user",
    ) -> Iterator[str]:
        """
        Send a message with stream=True and yield SSE chunks (content only).
        Yields assistant text chunks; other event types (tool_calls, etc.) can be
        extended later.
        """
        body = {
            "role": role,
            "content": content,
            "stream": True,
        }
        url = self._url(f"/sessions/{session_id}/messages")
        resp = self._session.post(
            url,
            json=body,
            timeout=self.timeout,
            stream=True,
            headers={
                **self._session.headers,
                "Accept": "text/event-stream",
            },
        )
        resp.raise_for_status()

        for ev_type, payload in _iter_sse_lines(resp):
            if ev_type == "chunk" and payload:
                try:
                    data = json.loads(payload)
                    if "content" in data:
                        yield data["content"]
                except json.JSONDecodeError:
                    pass
            elif ev_type == "error" and payload:
                try:
                    err = json.loads(payload)
                    raise RuntimeError(err.get("error", payload))
                except json.JSONDecodeError:
                    raise RuntimeError(payload)

    def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Message]:
        """Get messages for a session."""
        path = f"/sessions/{session_id}/messages?limit={limit}&offset={offset}"
        r = self._request("GET", path)
        data = r.json()
        out = []
        for m in data:
            out.append(
                Message(
                    id=m["id"],
                    session_id=str(m["session_id"]),
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    tool_name=m.get("tool_name"),
                    tool_call_id=m.get("tool_call_id"),
                    token_count=m.get("token_count"),
                    metadata=m.get("metadata") or {},
                    created_at=m.get("created_at", ""),
                )
            )
        return out

    def get_session(self, session_id: str) -> Session:
        """Get a session by ID."""
        r = self._request("GET", f"/sessions/{session_id}")
        data = r.json()
        return Session(
            id=data["id"],
            agent_id=data["agent_id"],
            external_user_id=data.get("external_user_id"),
            metadata=data.get("metadata") or {},
            created_at=data.get("created_at", ""),
            last_activity_at=data.get("last_activity_at", ""),
        )

    def list_sessions(self, agent_id: str) -> List[Session]:
        """List sessions for an agent."""
        r = self._request("GET", f"/agents/{agent_id}/sessions")
        data = r.json()
        return [
            Session(
                id=s["id"],
                agent_id=s["agent_id"],
                external_user_id=s.get("external_user_id"),
                metadata=s.get("metadata") or {},
                created_at=s.get("created_at", ""),
                last_activity_at=s.get("last_activity_at", ""),
            )
            for s in data
        ]

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "NeuronAgentClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# Optional: run() helper for one-off message (create session, send, return response)
def run(
    base_url: str,
    agent_id: str,
    message: str,
    api_key: Optional[str] = None,
) -> str:
    """
    One-off: create a session, send one message, return the assistant response text.
    """
    with NeuronAgentClient(base_url=base_url, api_key=api_key) as client:
        session = client.create_session(agent_id=agent_id)
        response = client.send_message(session_id=session.id, content=message)
        return response.response
