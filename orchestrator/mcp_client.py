"""MCP server write_reservation tool client."""
import asyncio
import logging
import requests
from typing import Optional
from datetime import datetime, timezone

from .config import config

logger = logging.getLogger(__name__)


async def _call_mcp_write(name: str, car_number: str,
                          reservation_period: str, approval_time: str) -> Optional[str]:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        logger.error("mcp package not installed")
        return None

    headers = {"Authorization": f"Bearer {config.mcp_token}"}
    async with streamablehttp_client(config.mcp_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("write_reservation", {
                "name": name, "car_number": car_number,
                "reservation_period": reservation_period,
                "approval_time": approval_time,
            })
            return result.content[0].text if result.content else None


def write_confirmed(name: str, car_number: str, period: str, approval_time: str) -> Optional[str]:
    """Sync wrapper. Returns MCP tool response or None on failure."""
    # Normalize ISO 8601 with offset → YYYY-MM-DDTHH:MM:SSZ
    try:
        dt = datetime.fromisoformat(approval_time)
        approval_time = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass
    try:
        return asyncio.run(_call_mcp_write(name, car_number, period, approval_time))
    except Exception as e:
        logger.error("MCP write failed: %s", e)
        return None


def is_available() -> bool:
    """Check MCP server health (no auth needed for /health)."""
    try:
        url = config.mcp_url.rsplit("/", 1)[0]  # strip /mcp
        return requests.get(f"{url}/health", timeout=3).status_code == 200
    except Exception:
        return False