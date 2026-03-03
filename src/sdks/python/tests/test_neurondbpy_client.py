"""Tests for neurondbpy.client."""

import pytest

from neurondbpy.client import NeuronDBClient, _DISTANCE_OPS


def test_distance_ops() -> None:
    assert _DISTANCE_OPS["cosine"] == "<=>"
    assert _DISTANCE_OPS["l2"] == "<->"
    assert _DISTANCE_OPS["inner_product"] == "<#>"
    assert _DISTANCE_OPS.get("unknown", "<=>") == "<=>"


def test_client_init_dsn() -> None:
    client = NeuronDBClient(dsn="postgresql://localhost/test")
    assert client._dsn == "postgresql://localhost/test"


def test_client_init_params() -> None:
    client = NeuronDBClient(host="localhost", port=5433, dbname="neurondb", user="u", password="p")
    assert "host=localhost" in client._dsn
    assert "port=5433" in client._dsn
    assert "dbname=neurondb" in client._dsn


def test_client_context_manager() -> None:
    """Context manager should not connect without a real DB."""
    client = NeuronDBClient(dsn="postgresql://invalid:5432/nonexistent")
    with pytest.raises(Exception):
        with client:
            client.execute("SELECT 1")
