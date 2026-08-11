# Parking Orchestrator (Stage 4)

LangGraph-based orchestration that unifies the three earlier stages into a
single pipeline:

1. **Chatbot (RAG agent)** — answers user questions and collects reservation
   data (Stage 1 logic, embedded as a library).
2. **Admin approval (human-in-the-loop)** — escalates reservations to the
   admin service and polls for the decision (Stage 2 REST API).
3. **Data recording (MCP server)** — once approved, calls the MCP
   `write_reservation` tool to persist the reservation to a file (Stage 3).

## Architecture

```
                         ┌─────────────────────────────┐
                         │   Orchestrator (LangGraph)   │
                         │                              │
   user ──────────────►  │  user_interaction            │
                         │     │                        │
                         │     ├─ info? ──► RAG answer  │
                         │     └─ book? ─► collect data │
                         │                  │           │
                         │            _escalate         │
                         │                  │           │
                         │            admin_approval    │
                         │             (polls Stage 2)  │
                         │                  │           │
                         │       ┌──────────┴────────┐  │
                         │       │                   │  │
                         │    rejected            approved│
                         │       │                   │  │
                         │      END            data_recording
                         │                        │      │
                         │                   MCP write   │
                         │                        │      │
                         └────────────────────────┴──────┘
                                   │
                            confirmed_reservations.txt
```

### Graph nodes

| Node                | Responsibility                                           |
|---------------------|----------------------------------------------------------|
| `user_interaction`  | RAG answer or reservation data collection (Stage 1)      |
| `admin_approval`    | Escalate to admin service + poll for decision (Stage 2)  |
| `data_recording`    | Call MCP `write_reservation` on approval (Stage 3)       |

### State

```python
OrchestratorState = {
    "user_input": str,
    "bot_reply": str,
    "phase": "chat" | "collecting" | "awaiting_admin" | "done",
    "reservation": {name, surname, car_number, start_time, end_time},
    "admin_request_id": int,
    "last_mcp_response": str,
}
```

## Prerequisites

The orchestrator depends on three services running:

1. **Qdrant** + **Ollama** (Stage 1 chatbot backend):
   ```bash
   docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
   ollama pull llama3.2:3b && ollama pull nomic-embed-text
   # seed Qdrant — use the Stage 1 repo's app/seed.py
   ```

2. **Admin service** (Stage 2):
   ```bash
   cd ../parking-admin-agent
   python -m admin_service.server   # http://localhost:8001
   ```

3. **MCP server** (Stage 3):
   ```bash
   cd ../parking-mcp-server
   python -m mcp_server.run          # http://localhost:8002/mcp
   ```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m orchestrator.main
```

Example session:

```
you > where is the parking?
bot > Central Parking is at 14 Independence Square, Kyiv...
you > book
bot > What's your first name?
you > Nikita
bot > What's your last name?
you > Pakhno
bot > What's your car plate number? (e.g. BC1234AB)
you > BC1234AB
bot > When do you want to start? (YYYY-MM-DD HH:MM)
you > 2025-12-01 10:00
bot > When do you want to finish? (YYYY-MM-DD HH:MM)
you > 2025-12-01 12:00
bot > Name: Nikita Pakhno
     Car: BC1234AB
     From: 2025-12-01 10:00
     To:   2025-12-01 12:00
     Type 'yes' to submit, 'no' to cancel.
you > yes
bot > Reservation escalated to admin (request #5). Type 'status' to check.
# admin approves via the admin service or admin CLI
you > status
bot > Reservation approved and recorded! reservation saved: Nikita Pakhno | ...
```

## Project layout

```
orchestrator/
  config.py           # env config
  chatbot.py          # Stage 1 — RAG logic (Qdrant + SQL + LLM)
  admin_client.py     # Stage 2 — admin service REST client
  mcp_client.py       # Stage 3 — MCP write_reservation client
  graph.py            # LangGraph orchestrator (the unified pipeline)
  main.py             # interactive CLI
tests/
  test_graph.py       # unit tests for graph logic (mocked, no services)
  test_clients.py     # unit tests for admin_client + mcp_client (mocked)
  test_chatbot.py     # unit tests for chatbot RAG (real SQLite, mocked LLM)
  test_load.py        # load tests (set LOAD_TESTS=1)
  test_integration.py # full pipeline integration (set INTEGRATION_TESTS=1)
```

## Tests

```bash
# unit tests — no external services needed
pytest -q tests/test_graph.py tests/test_clients.py tests/test_chatbot.py

# load tests — requires all three services running
LOAD_TESTS=1 pytest -q tests/test_load.py

# integration test — requires all three services running
INTEGRATION_TESTS=1 pytest -q tests/test_integration.py
```

## Deployment

All four repos are designed to run on a single host for development. For
production:

- Run Qdrant in Docker, Ollama as a service.
- Run `admin_service.server` behind a reverse proxy with TLS.
- Run `mcp_server.run` on an internal network only (bearer-token protected).
- Run `orchestrator.main` as a long-running service or wrap it in a FastAPI
  endpoint for multi-user access.

Each repo has its own `docker-compose.yml` / `.github/workflows/ci.yml` for
CI.