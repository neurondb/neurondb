"""Pydantic types for LLM SQL API responses."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class GenerateSQLResponse(BaseModel):
    sql: str
    explanation: str = ""
    confidence: float = 1.0
    warnings: List[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    dialect: str


class OptimizeSQLResponse(BaseModel):
    optimized_sql: str
    explanation: str = ""
    suggestions: List[str] = Field(default_factory=list)


class DebugSQLResponse(BaseModel):
    analysis: str = ""
    issues: List[str] = Field(default_factory=list)


class GenerationOptions(BaseModel):
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    extra: dict[str, Any] = Field(default_factory=dict)
