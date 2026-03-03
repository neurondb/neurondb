"""Simple types for NeuronAgent API responses."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Agent:
    """Agent from CreateAgent (minimal for compatibility)."""

    id: str
    name: str
    description: Optional[str]
    system_prompt: str
    model_name: str
    enabled_tools: List[str]
    config: Dict[str, Any]


@dataclass
class Session:
    """Session from CreateSession / GetSession."""

    id: str
    agent_id: str
    external_user_id: Optional[str]
    metadata: Dict[str, Any]
    created_at: str
    last_activity_at: str


@dataclass
class Message:
    """Message from GetMessages."""

    id: int
    session_id: str
    role: str
    content: str
    tool_name: Optional[str]
    tool_call_id: Optional[str]
    token_count: Optional[int]
    metadata: Dict[str, Any]
    created_at: str


@dataclass
class SendMessageResponse:
    """Response from SendMessage (non-stream)."""

    session_id: str
    agent_id: str
    response: str
    tokens_used: Optional[int]
    tool_calls: List[Any]
    tool_results: List[Any]
