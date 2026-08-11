"""Admin service REST API client."""
import logging
import requests
from typing import Optional, Dict, Any, List

from .config import config

logger = logging.getLogger(__name__)


def submit_reservation(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """POST a reservation to the admin service. Returns the created request."""
    try:
        r = requests.post(f"{config.admin_api_url}/requests", json={
            "name": data["name"], "surname": data["surname"],
            "car_number": data["car_number"],
            "start_time": data["start_time"], "end_time": data["end_time"],
        }, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("admin submit failed: %s", e)
        return None


def get_status(rid: int) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{config.admin_api_url}/requests/{rid}", timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("admin get_status failed: %s", e)
        return None


def list_pending() -> List[Dict[str, Any]]:
    try:
        r = requests.get(f"{config.admin_api_url}/requests?status=pending", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("admin list failed: %s", e)
        return []


def decide(rid: int, action: str, comment: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(f"{config.admin_api_url}/requests/{rid}/decide",
                          json={"action": action, "comment": comment}, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("admin decide failed: %s", e)
        return None


def is_available() -> bool:
    try:
        return requests.get(f"{config.admin_api_url}/health", timeout=3).status_code == 200
    except Exception:
        return False