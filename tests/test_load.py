"""Load tests for each component of the pipeline.

Run with: pytest tests/test_load.py -q

These tests measure throughput and latency under concurrent load. They
require all three services running (chatbot Qdrant+Ollama, admin service,
MCP server). Set LOAD_TESTS=1 to enable them — skipped otherwise.
"""
import os
import time
import pytest
import requests
import concurrent.futures

from orchestrator import chatbot, admin_client, mcp_client
from orchestrator.config import config

LOAD = os.getenv("LOAD_TESTS") == "1"
REASON = "set LOAD_TESTS=1 to run load tests"


@pytest.mark.skipif(not LOAD, reason=REASON)
def test_chatbot_rag_under_load():
    """RAG answer latency under 10 concurrent queries."""
    queries = [
        "where is the parking?",
        "what are the prices?",
        "working hours?",
        "is there EV charging?",
        "what zones are available?",
        "how do I pay?",
        "can I cancel?",
        "monthly passes?",
        "max vehicle height?",
        "booking process?",
    ]

    def one(q):
        t0 = time.perf_counter()
        a = chatbot.answer(q)
        return time.perf_counter() - t0, len(a)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(one, queries * 3))  # 30 queries

    times = [r[0] for r in results]
    avg = sum(times) / len(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    print(f"\nchatbot: 30 queries, avg={avg:.2f}s, p95={p95:.2f}s")
    assert avg < 10.0, f"avg latency too high: {avg}"


@pytest.mark.skipif(not LOAD, reason=REASON)
def test_admin_service_under_load():
    """Admin service POST throughput — 20 concurrent reservation submissions."""
    payload = {"name": "Load", "surname": "Test",
               "car_number": "BC1234AB",
               "start_time": "2025-12-20 10:00",
               "end_time": "2025-12-20 12:00"}

    def one(_):
        t0 = time.perf_counter()
        r = requests.post(f"{config.admin_api_url}/requests", json=payload, timeout=10)
        return time.perf_counter() - t0, r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(one, range(20)))

    times = [r[0] for r in results]
    codes = [r[1] for r in results]
    avg = sum(times) / len(times)
    print(f"\nadmin: 20 POSTs, avg={avg:.3f}s, all 200: {all(c == 200 for c in codes)}")
    assert all(c == 200 for c in codes), f"some requests failed: {codes}"
    assert avg < 2.0, f"admin latency too high: {avg}"


@pytest.mark.skipif(not LOAD, reason=REASON)
def test_mcp_server_under_load():
    """MCP write_reservation tool under 10 concurrent calls."""
    import asyncio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def one(i):
        t0 = time.perf_counter()
        async with streamablehttp_client(
            config.mcp_url,
            headers={"Authorization": f"Bearer {config.mcp_token}"},
        ) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool("write_reservation", {
                    "name": f"Load Test {i}", "car_number": "BC1234AB",
                    "reservation_period": "2025-12-25 10:00 - 2025-12-25 12:00",
                    "approval_time": "2026-08-11T12:00:00Z",
                })
                ok = "saved" in (res.content[0].text if res.content else "").lower()
        return time.perf_counter() - t0, ok

    async def run_all():
        tasks = [one(i) for i in range(10)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_all())
    times = [r[0] for r in results]
    avg = sum(times) / len(times)
    all_ok = all(r[1] for r in results)
    print(f"\nmcp: 10 writes, avg={avg:.3f}s, all saved: {all_ok}")
    assert all_ok, "some MCP writes failed"
    assert avg < 3.0, f"mcp latency too high: {avg}"