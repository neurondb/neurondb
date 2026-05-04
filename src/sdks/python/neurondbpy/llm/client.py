"""HTTP client for NeuronAgent / LLM SQL HTTP APIs."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

import httpx

from neurondbpy.llm.types import GenerateSQLResponse, OptimizeSQLResponse


class LLMSQLClient:
    """Client for generate / explain / optimize SQL via HTTP."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        *,
        agent_api: bool = True,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.agent_api = agent_api
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, headers=self._headers(), json=dict(payload))
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _extract_sql(content: str) -> str:
        m = re.search(r"<sql>(.*?)</sql>", content, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_explanation(content: str) -> str:
        m = re.search(r"Explanation:\s*(.+?)(?:\n\n|\Z)", content, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def generate_sql(
        self,
        *,
        prompt: str,
        dialect: str = "postgresql",
        schema: Optional[Any] = None,
    ) -> GenerateSQLResponse:
        body: dict[str, Any] = {"prompt": prompt, "dialect": dialect}
        if schema is not None:
            body["schema"] = schema
        path = "/v1/sql/generate" if self.agent_api else "/api/generate"
        data = self._post_json(path, body)
        if isinstance(data, dict) and "sql" in data:
            return GenerateSQLResponse.model_validate(data)
        if isinstance(data, str):
            return GenerateSQLResponse(
                sql=self._extract_sql(data) or data,
                explanation=self._extract_explanation(data),
            )
        return GenerateSQLResponse.model_validate(data)

    def explain_sql(self, *, sql: str, detail_level: str = "detailed") -> str:
        path = "/v1/sql/explain" if self.agent_api else "/api/explain"
        data = self._post_json(path, {"sql": sql, "detail_level": detail_level})
        if isinstance(data, str):
            return data
        if isinstance(data, dict) and "explanation" in data:
            return str(data["explanation"])
        return json.dumps(data)

    def optimize_sql(self, *, sql: str) -> OptimizeSQLResponse:
        path = "/v1/sql/optimize" if self.agent_api else "/api/optimize"
        data = self._post_json(path, {"sql": sql})
        if isinstance(data, dict):
            return OptimizeSQLResponse.model_validate(data)
        return OptimizeSQLResponse(optimized_sql=str(data))
