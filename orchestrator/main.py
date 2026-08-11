"""Interactive CLI for the parking orchestrator.

Runs the unified LangGraph pipeline. The user chats with the bot, books
reservations, and the orchestrator escalates to the admin service and
records approved reservations via the MCP server — all in one flow.
"""
import logging

from .graph import build_graph
from . import chatbot

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

INTRO = """
========================================
  Parking Orchestrator (Stage 4)
  Unified: chatbot → admin → MCP file
========================================
Ask about prices, hours, slots, location, rules.
Say 'book' to start a reservation — it will be escalated to an admin
and, once approved, written to the reservations file via the MCP server.
Type 'quit' to exit.
========================================
"""


def main():
    # make sure the chatbot's SQLite exists
    chatbot.init_sqlite()
    graph = build_graph()
    state = {}

    print(INTRO)
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("bye")
            break

        state["user_input"] = user
        result = graph.invoke(state)
        state = result
        reply = state.get("bot_reply")
        if reply:
            print(f"bot > {reply}")
        state.pop("bot_reply", None)
        state.pop("user_input", None)


if __name__ == "__main__":
    main()