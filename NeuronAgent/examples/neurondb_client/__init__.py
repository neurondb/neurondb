"""
NeuronAgent Python Client Library

A modular, production-ready client library for NeuronAgent.

Usage:
    from neurondb_client import NeuronAgentClient
    from neurondb_client.agents import AgentManager
    from neurondb_client.sessions import SessionManager
    
    client = NeuronAgentClient()
    agent_mgr = AgentManager(client)
    session_mgr = SessionManager(client)
"""

from .core.client import NeuronAgentClient
from .core.exceptions import (
    NeuronAgentError,
    AuthenticationError,
    NotFoundError,
    ServerError,
    ValidationError
)
from .agents import AgentManager, AgentProfile
from .sessions import SessionManager, ConversationManager
from .utils.config import ConfigLoader
from .utils.logging import setup_logging
from .utils.metrics import MetricsCollector

__version__ = "1.0.0"
__all__ = [
    "NeuronAgentClient",
    "AgentManager",
    "AgentProfile",
    "SessionManager",
    "ConversationManager",
    "ConfigLoader",
    "MetricsCollector",
    "setup_logging",
    "NeuronAgentError",
    "AuthenticationError",
    "NotFoundError",
    "ServerError",
    "ValidationError",
]

