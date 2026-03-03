"""
Unit tests for NeuronAgent Python client (sessions, streaming, run).
"""

import json
import unittest
from unittest.mock import MagicMock, patch

import requests

from neuronagent import NeuronAgentClient, run
from neuronagent.client import _iter_sse_lines
from neuronagent.types import Agent, Message, SendMessageResponse, Session


def _mock_response(status_code=200, json_data=None, text="", iter_lines=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status_code
    r.raise_for_status = MagicMock()
    if status_code >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(response=r)
    r.json = MagicMock(return_value=json_data or {})
    r.text = text
    if iter_lines is not None:
        r.iter_lines = MagicMock(decode_unicode=True)
        r.iter_lines.return_value = iter_lines
    return r


class TestIterSseLines(unittest.TestCase):
    def test_sse_parsing_single_event(self):
        lines = ["event: chunk", "data: {\"content\": \"hi\"}", ""]
        out = list(_iter_sse_lines(_mock_response(iter_lines=lines)))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "chunk")
        self.assertEqual(out[0][1], "{\"content\": \"hi\"}")

    def test_sse_parsing_multiple_events(self):
        lines = [
            "event: chunk", "data: {\"content\": \"a\"}", "",
            "event: chunk", "data: {\"content\": \"b\"}", "",
        ]
        out = list(_iter_sse_lines(_mock_response(iter_lines=lines)))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][1], "{\"content\": \"a\"}")
        self.assertEqual(out[1][1], "{\"content\": \"b\"}")

    def test_sse_parsing_no_final_newline(self):
        lines = ["event: chunk", "data: {\"content\": \"x\"}"]
        out = list(_iter_sse_lines(_mock_response(iter_lines=lines)))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1], "{\"content\": \"x\"}")


class TestNeuronAgentClient(unittest.TestCase):
    def setUp(self):
        self.client = NeuronAgentClient(base_url="http://localhost:8080", api_key="test-key")

    def tearDown(self):
        self.client.close()

    @patch.object(requests.Session, "request")
    def test_create_agent_returns_agent(self, mock_request):
        mock_request.return_value = _mock_response(json_data={
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "my-agent",
            "description": "desc",
            "system_prompt": "You are helpful.",
            "model_name": "gpt-4",
            "enabled_tools": ["sql"],
            "config": {},
        })
        agent = self.client.agents.create_agent(
            name="my-agent",
            description="desc",
            system_prompt="You are helpful.",
            enabled_tools=["sql"],
        )
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.id, "550e8400-e29b-41d4-a716-446655440000")
        self.assertEqual(agent.name, "my-agent")
        self.assertEqual(agent.enabled_tools, ["sql"])

    @patch.object(requests.Session, "request")
    def test_create_session_returns_session(self, mock_request):
        mock_request.return_value = _mock_response(json_data={
            "id": "660e8400-e29b-41d4-a716-446655440001",
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "external_user_id": None,
            "metadata": {"user_id": "u1"},
            "created_at": "2025-01-01T00:00:00Z",
            "last_activity_at": "2025-01-01T00:00:00Z",
        })
        session = self.client.sessions.create_session(
            agent_id="550e8400-e29b-41d4-a716-446655440000",
            metadata={"user_id": "u1"},
        )
        self.assertIsInstance(session, Session)
        self.assertEqual(session.id, "660e8400-e29b-41d4-a716-446655440001")
        self.assertEqual(session.agent_id, "550e8400-e29b-41d4-a716-446655440000")

    @patch.object(requests.Session, "request")
    def test_send_message_returns_response(self, mock_request):
        mock_request.return_value = _mock_response(json_data={
            "session_id": "660e8400-e29b-41d4-a716-446655440001",
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "response": "Hello back!",
            "tokens_used": 10,
            "tool_calls": [],
            "tool_results": [],
        })
        out = self.client.sessions.send_message(
            session_id="660e8400-e29b-41d4-a716-446655440001",
            content="Hello",
        )
        self.assertIsInstance(out, SendMessageResponse)
        self.assertEqual(out.response, "Hello back!")
        self.assertEqual(out.tokens_used, 10)

    @patch.object(requests.Session, "request")
    def test_get_messages_returns_list(self, mock_request):
        mock_request.return_value = _mock_response(json_data=[
            {
                "id": 1,
                "session_id": "660e8400-e29b-41d4-a716-446655440001",
                "role": "user",
                "content": "Hi",
                "tool_name": None,
                "tool_call_id": None,
                "token_count": None,
                "metadata": {},
                "created_at": "2025-01-01T00:00:00Z",
            },
            {
                "id": 2,
                "session_id": "660e8400-e29b-41d4-a716-446655440001",
                "role": "assistant",
                "content": "Hello!",
                "tool_name": None,
                "tool_call_id": None,
                "token_count": None,
                "metadata": {},
                "created_at": "2025-01-01T00:00:01Z",
            },
        ])
        messages = self.client.sessions.get_messages(session_id="660e8400-e29b-41d4-a716-446655440001")
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], Message)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "Hi")
        self.assertEqual(messages[1].content, "Hello!")

    @patch.object(requests.Session, "post")
    def test_send_message_stream_yields_chunks(self, mock_post):
        sse_body = [
            b"event: chunk",
            b'data: {"content": "Hel"}',
            b"",
            b"event: chunk",
            b'data: {"content": "lo"}',
            b"",
        ]
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=iter([line.decode("utf-8") for line in sse_body]))
        mock_post.return_value = resp

        chunks = list(
            self.client.send_message_stream(
                session_id="660e8400-e29b-41d4-a716-446655440001",
                content="Hi",
            )
        )
        self.assertEqual(chunks, ["Hel", "lo"])

    @patch.object(requests.Session, "post")
    def test_send_message_stream_error_event_raises(self, mock_post):
        sse_body = ["event: error", 'data: {"error": "Something failed"}', ""]
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.iter_lines = MagicMock(return_value=iter(sse_body))
        mock_post.return_value = resp

        with self.assertRaises(RuntimeError) as ctx:
            list(
                self.client.send_message_stream(
                    session_id="sid",
                    content="Hi",
                )
            )
        self.assertIn("Something failed", str(ctx.exception))

    @patch.object(requests.Session, "request")
    def test_run_returns_response_text(self, mock_request):
        mock_request.side_effect = [
            _mock_response(json_data={
                "id": "660e8400-e29b-41d4-a716-446655440001",
                "agent_id": "550e8400-e29b-41d4-a716-446655440000",
                "external_user_id": None,
                "metadata": {},
                "created_at": "2025-01-01T00:00:00Z",
                "last_activity_at": "2025-01-01T00:00:00Z",
            }),
            _mock_response(json_data={
                "session_id": "660e8400-e29b-41d4-a716-446655440001",
                "agent_id": "550e8400-e29b-41d4-a716-446655440000",
                "response": "One-off reply",
                "tokens_used": 5,
                "tool_calls": [],
                "tool_results": [],
            }),
        ]
        result = run(
            base_url="http://localhost:8080",
            agent_id="550e8400-e29b-41d4-a716-446655440000",
            message="Hello",
        )
        self.assertEqual(result, "One-off reply")


if __name__ == "__main__":
    unittest.main()
