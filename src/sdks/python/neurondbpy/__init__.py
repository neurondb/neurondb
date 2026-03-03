"""
neurondbpy: Python SDK for NeuronDB.

Provides:
- NeuronDBClient / AsyncNeuronDBClient: sync and async PostgreSQL clients with vector support
- vectors: vector search, insert, index helpers
- LLM SQL and pipeline: install neurondbpy-llm for LLM client (generate, explain, optimize, translate) and training pipeline
- CLI: neurondbpy command-line interface
"""

__version__ = "1.0.0"

from neurondbpy.client import NeuronDBClient


def __getattr__(name: str):
    if name == "LLMSQLClient":
        try:
            from neurondbpy_llm.llm.client import LLMSQLClient
            return LLMSQLClient
        except ImportError:
            raise ImportError(
                "LLM client requires neurondbpy-llm. Install with: pip install neurondbpy-llm"
            ) from None
    if name in ("GenerateSQLResponse", "ModelInfo", "OptimizeSQLResponse", "DebugSQLResponse"):
        try:
            from neurondbpy_llm import llm as _llm
            return getattr(_llm, name)
        except ImportError:
            raise ImportError(
                "LLM types require neurondbpy-llm. Install with: pip install neurondbpy-llm"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "NeuronDBClient",
    "LLMSQLClient",
    "GenerateSQLResponse",
    "ModelInfo",
    "OptimizeSQLResponse",
    "DebugSQLResponse",
]
