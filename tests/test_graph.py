"""Unit tests for orchestrator graph logic — no external services needed."""
import pytest
from unittest.mock import patch, MagicMock
from langgraph.graph import END

from orchestrator.graph import (
    build_graph, _missing, _matches, _summary,
    RESERVATION_TRIGGERS, CONFIRM_TRIGGERS, CANCEL_TRIGGERS,
)


def test_matches_word_boundary():
    assert _matches("book", RESERVATION_TRIGGERS)
    assert _matches("I want to book", RESERVATION_TRIGGERS)
    assert not _matches("bookmark", RESERVATION_TRIGGERS)  # "bookmark" → {"bookmark"}, no "book"
    assert _matches("yes", CONFIRM_TRIGGERS)
    assert _matches("no", CANCEL_TRIGGERS)
    assert not _matches("Pakhno", CANCEL_TRIGGERS)  # regression: substring bug


def test_missing_fields():
    assert _missing({}) == ["name", "surname", "car_number", "start_time", "end_time"]
    d = {"name": "x"}
    assert _missing(d) == ["surname", "car_number", "start_time", "end_time"]
    full = {"name": "a", "surname": "b", "car_number": "c",
            "start_time": "d", "end_time": "e"}
    assert _missing(full) == []


def test_summary():
    res = {"name": "N", "surname": "P", "car_number": "BC1234AB",
           "start_time": "2025-12-01 10:00", "end_time": "2025-12-01 12:00"}
    s = _summary(res)
    assert "N P" in s
    assert "BC1234AB" in s
    assert "2025-12-01 10:00" in s


def test_graph_compiles():
    g = build_graph()
    assert g is not None


def test_user_interaction_info_path(monkeypatch):
    """A plain question should go through RAG and return to END."""
    from orchestrator.graph import user_interaction_node
    monkeypatch.setattr("orchestrator.graph.chatbot.answer", lambda q: "Kyiv, 14 Independence Square")
    state = {"user_input": "where is the parking?"}
    out = user_interaction_node(state)
    assert "Kyiv" in out["bot_reply"]
    assert out["phase"] == "chat"


def test_user_interaction_book_starts_collection(monkeypatch):
    from orchestrator.graph import user_interaction_node
    state = {"user_input": "book"}
    out = user_interaction_node(state)
    assert out["phase"] == "collecting"
    assert "first name" in out["bot_reply"].lower()


def test_user_interaction_collects_fields(monkeypatch):
    from orchestrator.graph import user_interaction_node
    state = {"user_input": "Nikita", "phase": "collecting",
             "reservation": {}}
    out = user_interaction_node(state)
    assert out["reservation"]["name"] == "Nikita"
    assert "last name" in out["bot_reply"].lower()


def test_user_interaction_cancel_during_collection(monkeypatch):
    from orchestrator.graph import user_interaction_node
    state = {"user_input": "no", "phase": "collecting", "reservation": {"name": "x"}}
    out = user_interaction_node(state)
    assert out["phase"] == "chat"
    assert out["reservation"] == {}


def test_user_interaction_confirm_triggers_escalation(monkeypatch):
    from orchestrator.graph import user_interaction_node
    state = {"user_input": "yes", "phase": "collecting",
             "reservation": {"name": "N", "surname": "P", "car_number": "BC1234AB",
                             "start_time": "2025-12-01 10:00", "end_time": "2025-12-01 12:00"}}
    # mock admin_client
    monkeypatch.setattr("orchestrator.graph.admin_client.is_available", lambda: True)
    monkeypatch.setattr("orchestrator.graph.admin_client.submit_reservation",
                        lambda r: {"id": 42, "status": "pending"})
    out = user_interaction_node(state)
    assert out["phase"] == "awaiting_admin"
    assert out["admin_request_id"] == 42
    assert "escalated to admin" in out["bot_reply"]


def test_escalate_admin_unavailable(monkeypatch):
    from orchestrator.graph import _escalate
    monkeypatch.setattr("orchestrator.graph.admin_client.is_available", lambda: False)
    state = {"reservation": {"name": "N", "surname": "P", "car_number": "BC1234AB",
                             "start_time": "x", "end_time": "y"}}
    out = _escalate(state)
    assert out["phase"] == "chat"
    assert "not available" in out["bot_reply"].lower()


def test_poll_admin_approved(monkeypatch):
    from orchestrator.graph import _poll_admin
    monkeypatch.setattr("orchestrator.graph.admin_client.get_status",
                        lambda rid: {"id": rid, "status": "approved", "decided_at": "2026-08-11T12:00:00Z"})
    state = {"admin_request_id": 5}
    out = _poll_admin(state)
    assert "Approved by admin" in out["bot_reply"]


def test_poll_admin_rejected(monkeypatch):
    from orchestrator.graph import _poll_admin
    monkeypatch.setattr("orchestrator.graph.admin_client.get_status",
                        lambda rid: {"id": rid, "status": "rejected", "admin_comment": "bad"})
    state = {"admin_request_id": 5}
    out = _poll_admin(state)
    assert "rejected" in out["bot_reply"].lower()
    assert out["phase"] == "done"


def test_poll_admin_pending(monkeypatch):
    from orchestrator.graph import _poll_admin
    monkeypatch.setattr("orchestrator.graph.admin_client.get_status",
                        lambda rid: {"id": rid, "status": "pending"})
    state = {"admin_request_id": 5, "phase": "awaiting_admin"}
    out = _poll_admin(state)
    assert "pending" in out["bot_reply"].lower()


def test_data_recording_node(monkeypatch):
    from orchestrator.graph import data_recording_node
    monkeypatch.setattr("orchestrator.graph.mcp_client.write_confirmed",
                        lambda n, c, p, t: "reservation saved: ok")
    state = {"reservation": {"name": "N", "surname": "P", "car_number": "BC1234AB",
                             "start_time": "2025-12-01 10:00", "end_time": "2025-12-01 12:00"},
             "admin_request": {"decided_at": "2026-08-11T12:00:00Z"}}
    out = data_recording_node(state)
    assert "approved and recorded" in out["bot_reply"].lower()
    assert out["phase"] == "done"
    assert out["last_mcp_response"] == "reservation saved: ok"


def test_data_recording_node_failure(monkeypatch):
    from orchestrator.graph import data_recording_node
    monkeypatch.setattr("orchestrator.graph.mcp_client.write_confirmed",
                        lambda n, c, p, t: None)
    state = {"reservation": {"name": "N", "surname": "P", "car_number": "BC1234AB",
                             "start_time": "x", "end_time": "y"},
             "admin_request": {}}
    out = data_recording_node(state)
    assert "failed" in out["bot_reply"].lower()


def test_route_after_user_approved(monkeypatch):
    from orchestrator.graph import _route_after_user
    state = {"phase": "awaiting_admin", "bot_reply": "Approved by admin! Recording..."}
    assert _route_after_user(state) == "data_recording"


def test_route_after_user_pending():
    from orchestrator.graph import _route_after_user
    state = {"phase": "awaiting_admin", "bot_reply": "Still pending admin review."}
    # LangGraph uses "__end__" as the END sentinel
    assert _route_after_user(state) in (END, "__end__")


def test_route_after_admin_approved():
    from orchestrator.graph import _route_after_admin
    state = {"bot_reply": "Approved by admin! Recording..."}
    assert _route_after_admin(state) == "data_recording"


def test_route_after_admin_rejected():
    from orchestrator.graph import _route_after_admin
    state = {"bot_reply": "Reservation rejected by admin."}
    assert _route_after_admin(state) in (END, "__end__")