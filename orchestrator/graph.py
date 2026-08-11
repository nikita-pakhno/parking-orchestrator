"""LangGraph orchestrator — the unified pipeline.

Graph structure:

      start
        │
        ▼
   user_interaction  ◄──── (user asks / books / gives data)
        │
   ┌────┴────┐
   │         │
info_only   reservation_complete
   │         │
END     admin_approval  ◄──── (admin service decides)
            │
       ┌────┴────┐
       │         │
   rejected   approved
       │         │
      END    data_recording  ◄──── (MCP write_reservation)
                 │
                END

Each node is one stage of the pipeline. The orchestrator drives the full
flow: RAG answer → collect reservation → escalate to admin → record on
approval. The user interaction node is re-entered as many times as needed
to collect reservation fields.
"""
import logging
import re
from typing import TypedDict, Optional, Dict, Any, Annotated
from operator import add
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END

from . import chatbot
from . import admin_client
from . import mcp_client

logger = logging.getLogger(__name__)


REQUIRED_FIELDS = ["name", "surname", "car_number", "start_time", "end_time"]
QUESTIONS = {
    "name": "What's your first name?",
    "surname": "What's your last name?",
    "car_number": "What's your car plate number? (e.g. BC1234AB)",
    "start_time": "When do you want to start? (YYYY-MM-DD HH:MM)",
    "end_time": "When do you want to finish? (YYYY-MM-DD HH:MM)",
}

RESERVATION_TRIGGERS = {"book", "reserve", "reservation", "booking"}
CONFIRM_TRIGGERS = {"yes", "confirm", "submit", "ok", "sure", "y", "yep", "yeah"}
CANCEL_TRIGGERS = {"no", "cancel", "stop", "abort", "n", "nope"}
STATUS_QUERY = {"status", "check"}


def _matches(text: str, words: set) -> bool:
    tokens = set(re.split(r"\W+", text.lower().strip()))
    return bool(tokens & words)


class OrchestratorState(TypedDict, total=False):
    user_input: str
    bot_reply: str
    phase: str   # "chat" | "collecting" | "awaiting_admin" | "done"
    reservation: Dict[str, Any]
    admin_request_id: Optional[int]
    last_mcp_response: Optional[str]
    history: Annotated[list, add]


def _missing(res: Dict[str, Any]) -> list:
    return [f for f in REQUIRED_FIELDS if not res.get(f)]


# ---------- Nodes ----------

def user_interaction_node(state: OrchestratorState) -> OrchestratorState:
    """Handle user input — either answer via RAG or detect reservation intent."""
    msg = state.get("user_input", "").strip()
    res = state.setdefault("reservation", {})

    # status poll while awaiting admin
    if state.get("phase") == "awaiting_admin":
        if _matches(msg, STATUS_QUERY) or _matches(msg, CANCEL_TRIGGERS):
            return _poll_admin(state)
        # user said something else — still pending
        state["bot_reply"] = ("Your reservation is pending admin review. "
                              "Type 'status' to check again.")
        return state

    # mid-collection
    if state.get("phase") == "collecting" and _missing(res):
        if _matches(msg, CANCEL_TRIGGERS) and len(msg) <= 6:
            state["reservation"] = {}
            state["phase"] = "chat"
            state["bot_reply"] = "Reservation cancelled. Anything else?"
            return state
        # fill next field
        field = _missing(res)[0]
        res[field] = msg
        if _missing(res):
            state["bot_reply"] = QUESTIONS[_missing(res)[0]]
            return state
        # all fields filled
        state["bot_reply"] = _summary(res) + "\n\nType 'yes' to submit, 'no' to cancel."
        return state

    # waiting for confirm after collection complete
    if state.get("phase") == "collecting" and not _missing(res):
        if _matches(msg, CONFIRM_TRIGGERS):
            return _escalate(state)
        if _matches(msg, CANCEL_TRIGGERS):
            state["reservation"] = {}
            state["phase"] = "chat"
            state["bot_reply"] = "Reservation cancelled. Anything else?"
            return state
        state["bot_reply"] = "Please type 'yes' to submit or 'no' to cancel."
        return state

    # fresh input — reservation trigger or info question
    if _matches(msg, RESERVATION_TRIGGERS):
        state["phase"] = "collecting"
        state["reservation"] = {}
        state["bot_reply"] = QUESTIONS["name"]
        return state

    # default: RAG answer
    state["phase"] = "chat"
    state["bot_reply"] = chatbot.answer(msg)
    return state


def admin_approval_node(state: OrchestratorState) -> OrchestratorState:
    """Escalation happens in _escalate(); this node polls for the decision."""
    return _poll_admin(state)


def data_recording_node(state: OrchestratorState) -> OrchestratorState:
    """On approval, call the MCP server to write the reservation to file."""
    res = state.get("reservation", {})
    req = state.get("admin_request", {})
    name = f"{res.get('name','')} {res.get('surname','')}".strip()
    car = res.get("car_number", "")
    period = f"{res.get('start_time','')} - {res.get('end_time','')}"
    approval_time = req.get("decided_at") or datetime.now(timezone.utc).isoformat()
    resp = mcp_client.write_confirmed(name, car, period, approval_time)
    state["last_mcp_response"] = resp
    if resp and "saved" in resp.lower():
        state["bot_reply"] = f"Reservation approved and recorded! {resp}"
    else:
        state["bot_reply"] = f"Approved, but file write failed: {resp}"
    state["phase"] = "done"
    return state


# ---------- Helpers (called from nodes) ----------

def _summary(res: Dict[str, Any]) -> str:
    return (f"Name: {res['name']} {res['surname']}\n"
            f"Car: {res['car_number']}\n"
            f"From: {res['start_time']}\nTo:   {res['end_time']}")


def _escalate(state: OrchestratorState) -> OrchestratorState:
    res = state["reservation"]
    if not admin_client.is_available():
        state["bot_reply"] = ("Admin service is not available. "
                              "Please try again later.")
        state["phase"] = "chat"
        return state
    created = admin_client.submit_reservation(res)
    if not created:
        state["bot_reply"] = "Failed to submit reservation to admin."
        state["phase"] = "chat"
        return state
    state["admin_request_id"] = created["id"]
    state["admin_request"] = created
    state["phase"] = "awaiting_admin"
    state["bot_reply"] = (
        f"Reservation escalated to admin (request #{created['id']}). "
        "Type 'status' to check the decision."
    )
    return state


def _poll_admin(state: OrchestratorState) -> OrchestratorState:
    rid = state.get("admin_request_id")
    if rid is None:
        state["bot_reply"] = "No pending reservation."
        state["phase"] = "chat"
        return state
    info = admin_client.get_status(rid)
    if info is None:
        state["bot_reply"] = "Couldn't reach the admin service."
        return state
    status = info["status"]
    if status == "approved":
        state["admin_request"] = info
        state["bot_reply"] = "Approved by admin! Recording..."
        # fall through to data_recording via routing
        return state
    if status == "rejected":
        state["bot_reply"] = f"Reservation rejected by admin. Comment: {info.get('admin_comment') or 'n/a'}"
        state["phase"] = "done"
        return state
    state["bot_reply"] = "Still pending admin review. Type 'status' to check again."
    return state


# ---------- Graph ----------

def _route_after_user(state: OrchestratorState) -> str:
    """Decide whether to go to admin_approval, data_recording, or end."""
    # If we just got approved via status poll → record
    if state.get("phase") == "awaiting_admin":
        reply = state.get("bot_reply", "")
        if "Approved by admin" in reply:
            return "data_recording"
        return END
    return END


def _route_after_admin(state: OrchestratorState) -> str:
    reply = state.get("bot_reply", "")
    if "Approved by admin" in reply:
        return "data_recording"
    if "rejected" in reply.lower():
        return END
    return END


def build_graph():
    g = StateGraph(OrchestratorState)
    g.add_node("user_interaction", user_interaction_node)
    g.add_node("admin_approval", admin_approval_node)
    g.add_node("data_recording", data_recording_node)

    g.set_entry_point("user_interaction")
    g.add_conditional_edges("user_interaction", _route_after_user,
                            {END: END, "data_recording": "data_recording"})
    g.add_conditional_edges("admin_approval", _route_after_admin,
                            {END: END, "data_recording": "data_recording"})
    g.add_edge("data_recording", END)
    return g.compile()