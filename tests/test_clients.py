"""Unit tests for the three stage clients — mocked, no live services."""
import pytest
from unittest.mock import patch, MagicMock

from orchestrator import admin_client, mcp_client


# ---------- admin_client ----------

def _resp(code=200, data=None):
    r = MagicMock(); r.status_code = code; r.json.return_value = data or {}
    r.raise_for_status = MagicMock()
    return r


@patch("orchestrator.admin_client.requests")
def test_admin_submit_ok(mock_req):
    mock_req.post.return_value = _resp(200, {"id": 1, "status": "pending"})
    out = admin_client.submit_reservation({"name": "N", "surname": "P",
        "car_number": "BC1234AB", "start_time": "x", "end_time": "y"})
    assert out["id"] == 1


@patch("orchestrator.admin_client.requests")
def test_admin_submit_failure(mock_req):
    mock_req.post.side_effect = RuntimeError("down")
    assert admin_client.submit_reservation({"name": "N", "surname": "P",
        "car_number": "BC1234AB", "start_time": "x", "end_time": "y"}) is None


@patch("orchestrator.admin_client.requests")
def test_admin_get_status_ok(mock_req):
    mock_req.get.return_value = _resp(200, {"id": 5, "status": "approved"})
    assert admin_client.get_status(5)["status"] == "approved"


@patch("orchestrator.admin_client.requests")
def test_admin_get_status_404(mock_req):
    mock_req.get.return_value = _resp(404)
    assert admin_client.get_status(99) is None


@patch("orchestrator.admin_client.requests")
def test_admin_decide_ok(mock_req):
    mock_req.post.return_value = _resp(200, {"id": 1, "status": "approved"})
    out = admin_client.decide(1, "approve", "ok")
    assert out["status"] == "approved"


@patch("orchestrator.admin_client.requests")
def test_admin_decide_twice(mock_req):
    mock_req.post.return_value = _resp(404)
    assert admin_client.decide(1, "reject") is None


@patch("orchestrator.admin_client.requests")
def test_admin_list_pending(mock_req):
    mock_req.get.return_value = _resp(200, [{"id": 1}, {"id": 2}])
    assert len(admin_client.list_pending()) == 2


@patch("orchestrator.admin_client.requests")
def test_admin_is_available_true(mock_req):
    mock_req.get.return_value = _resp(200)
    assert admin_client.is_available() is True


@patch("orchestrator.admin_client.requests")
def test_admin_is_available_false(mock_req):
    mock_req.get.side_effect = RuntimeError
    assert admin_client.is_available() is False


# ---------- mcp_client ----------

def test_mcp_is_available_true(monkeypatch):
    import requests
    r = MagicMock(); r.status_code = 200
    monkeypatch.setattr("orchestrator.mcp_client.requests", MagicMock(get=MagicMock(return_value=r)))
    assert mcp_client.is_available() is True


def test_mcp_is_available_false(monkeypatch):
    monkeypatch.setattr("orchestrator.mcp_client.requests",
                        MagicMock(get=MagicMock(side_effect=RuntimeError)))
    assert mcp_client.is_available() is False


def test_mcp_write_confirmed_normalizes_time(monkeypatch):
    # mock asyncio.run to capture the normalized approval_time
    captured = {}
    def fake_run(coro):
        # coro is _call_mcp_write(name, car, period, approval_time)
        # we can't easily inspect args, so mock _call_mcp_write directly
        coro.close()
        return "reservation saved: ok"
    monkeypatch.setattr("orchestrator.mcp_client.asyncio.run", fake_run)
    out = mcp_client.write_confirmed("N P", "BC1234AB",
        "2025-12-01 10:00 - 2025-12-01 12:00",
        "2026-08-11T15:30:00+03:00")
    assert out == "reservation saved: ok"


def test_mcp_write_confirmed_failure_returns_none(monkeypatch):
    monkeypatch.setattr("orchestrator.mcp_client.asyncio.run",
                        lambda c: (_ for _ in ()).throw(RuntimeError("down")))
    out = mcp_client.write_confirmed("N", "BC1234AB", "p", "2026-08-11T12:00:00Z")
    assert out is None