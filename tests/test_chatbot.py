"""Unit tests for the chatbot RAG logic — mocked where needed."""
import os
import tempfile
import pytest
from unittest.mock import MagicMock

from orchestrator import chatbot


@pytest.fixture(autouse=True)
def tmp_sqlite(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(chatbot.config, "sqlite_path", os.path.join(d, "t.db"))
        chatbot.init_sqlite()
        yield


def test_is_dynamic():
    assert chatbot._is_dynamic("what are the prices?")
    assert chatbot._is_dynamic("working hours")
    assert not chatbot._is_dynamic("where is the parking?")


def test_retrieve_dynamic_prices():
    out = chatbot.retrieve_dynamic("what are the prices?")
    assert out and "Prices" in out
    assert "Zone A" in out


def test_retrieve_dynamic_hours():
    out = chatbot.retrieve_dynamic("working hours?")
    assert out and "Working hours" in out
    assert "monday" in out


def test_retrieve_dynamic_slots():
    out = chatbot.retrieve_dynamic("available slots?")
    assert out and "Available" in out


def test_retrieve_dynamic_non_dynamic_returns_none():
    assert chatbot.retrieve_dynamic("where is the parking?") is None


def test_answer_dynamic_path():
    out = chatbot.answer("what are the prices?")
    assert "Zone" in out
    assert "UAH" in out


def test_answer_static_fallback(monkeypatch):
    monkeypatch.setattr(chatbot, "retrieve_static", lambda q, top_k=5: "Central Parking is at 14 Independence Sq.")
    monkeypatch.setattr(chatbot, "retrieve_dynamic", lambda q: None)
    monkeypatch.setattr(chatbot, "get_llm", lambda: MagicMock(invoke=lambda p: "It is at 14 Independence Sq."))
    out = chatbot.answer("where is the parking?")
    assert "14 Independence" in out


def test_answer_no_context(monkeypatch):
    monkeypatch.setattr(chatbot, "retrieve_static", lambda q, top_k=5: "")
    monkeypatch.setattr(chatbot, "retrieve_dynamic", lambda q: None)
    out = chatbot.answer("random question?")
    assert "don't have information" in out.lower()


def test_init_sqlite_seeds_once():
    # calling init again should not duplicate rows
    chatbot.init_sqlite()
    with chatbot.sqlite_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM working_hours").fetchone()[0] == 7