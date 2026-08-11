"""Integration tests — full pipeline orchestration.

These require all three services running. Skipped unless INTEGRATION_TESTS=1.
"""
import os
import pytest
import requests

from orchestrator import chatbot, admin_client, mcp_client
from orchestrator.config import config

INT = os.getenv("INTEGRATION_TESTS") == "1"
REASON = "set INTEGRATION_TESTS=1 (requires chatbot Qdrant+Ollama, admin service, MCP server)"


@pytest.mark.skipif(not INT, reason=REASON)
def test_all_services_up():
    assert admin_client.is_available(), "admin service not running"
    assert mcp_client.is_available(), "mcp server not running"
    # chatbot Qdrant
    try:
        from qdrant_client import QdrantClient
        QdrantClient(url=config.qdrant_url).get_collections()
    except Exception as e:
        pytest.fail(f"qdrant not available: {e}")


@pytest.mark.skipif(not INT, reason=REASON)
def test_full_pipeline():
    """End-to-end: chatbot RAG → admin escalation → MCP recording."""
    # 1. chatbot answers a RAG question
    answer = chatbot.answer("where is the parking?")
    assert len(answer) > 10, f"chatbot answer too short: {answer}"

    # 2. submit reservation to admin
    res = {"name": "Integration", "surname": "Test",
           "car_number": "BC1234AB",
           "start_time": "2025-12-30 10:00",
           "end_time": "2025-12-30 12:00"}
    created = admin_client.submit_reservation(res)
    assert created and created["id"] > 0, "admin submit failed"
    rid = created["id"]

    # 3. admin approves
    decided = admin_client.decide(rid, "approve", "integration test")
    assert decided and decided["status"] == "approved", "admin approve failed"

    # 4. MCP writes the confirmed reservation
    period = f"{res['start_time']} - {res['end_time']}"
    approval_time = decided.get("decided_at") or "2026-08-11T12:00:00Z"
    mcp_resp = mcp_client.write_confirmed(
        f"{res['name']} {res['surname']}", res["car_number"],
        period, approval_time,
    )
    assert mcp_resp and "saved" in mcp_resp.lower(), f"mcp write failed: {mcp_resp}"
    assert "Integration Test" in mcp_resp
    assert "BC1234AB" in mcp_resp