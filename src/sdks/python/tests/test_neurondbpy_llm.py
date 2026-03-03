"""Tests for neurondbpy.llm client and types."""

import pytest

from neurondbpy.llm.types import (
    GenerateSQLResponse,
    ModelInfo,
    OptimizeSQLResponse,
    DebugSQLResponse,
    GenerationOptions,
)


def test_generate_sql_response() -> None:
    r = GenerateSQLResponse(sql="SELECT 1", explanation="Count", confidence=0.9, warnings=[])
    assert r.sql == "SELECT 1"
    assert r.confidence == 0.9


def test_model_info() -> None:
    m = ModelInfo(id="x", name="SQL LLM", version="1.0", dialect="postgresql")
    assert m.dialect == "postgresql"


def test_llm_client_extract_sql() -> None:
    pytest.importorskip("httpx")
    from neurondbpy.llm.client import LLMSQLClient
    content = "Here is the query:\n<sql>SELECT * FROM t;</sql>\nExplanation: Simple."
    assert LLMSQLClient._extract_sql(content) == "SELECT * FROM t;"
    assert LLMSQLClient._extract_explanation(content) == "Simple."


def test_llm_client_init() -> None:
    pytest.importorskip("httpx")
    from neurondbpy.llm.client import LLMSQLClient
    c = LLMSQLClient(base_url="http://localhost:8080")
    assert c.base_url == "http://localhost:8080"
    assert c.agent_api is True
    c2 = LLMSQLClient(base_url="http://localhost:8000", agent_api=False)
    assert c2.agent_api is False
