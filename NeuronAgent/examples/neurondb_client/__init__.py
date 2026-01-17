"""
NeuronAgent Python Client Library

A modular client library for NeuronAgent.

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
    ValidationError,
    ConnectionError,
    TimeoutError
)
from .agents import AgentManager, AgentProfile
from .sessions import SessionManager, ConversationManager
from .tools import ToolManager
from .workflows import WorkflowManager
from .budgets import BudgetManager
from .evaluation import EvaluationManager
from .replay import ReplayManager
from .specializations import SpecializationManager
from .plans import PlanManager
from .webhooks import WebhookManager
from .memory import MemoryManager
from .vfs import VFSManager
from .collaboration import CollaborationManager
from .utils.config import ConfigLoader
from .utils.logging import setup_logging
from .utils.metrics import MetricsCollector

__version__ = "2.1.0"
__all__ = [
    "NeuronAgentClient",
    "AgentManager",
    "AgentProfile",
    "SessionManager",
    "ConversationManager",
    "ToolManager",
    "WorkflowManager",
    "BudgetManager",
    "EvaluationManager",
    "ReplayManager",
    "SpecializationManager",
    "PlanManager",
    "WebhookManager",
    "MemoryManager",
    "VFSManager",
    "CollaborationManager",
    "ConfigLoader",
    "MetricsCollector",
    "setup_logging",
    "NeuronAgentError",
    "AuthenticationError",
    "NotFoundError",
    "ServerError",
    "ValidationError",
    "ConnectionError",
    "TimeoutError",
]









