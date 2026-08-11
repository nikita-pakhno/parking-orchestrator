"""RAG chatbot logic embedded as a library."""
import logging
import sqlite3
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from .config import config

logger = logging.getLogger(__name__)

_llm: Optional[OllamaLLM] = None
_embeddings: Optional[OllamaEmbeddings] = None
_qdrant: Optional[QdrantClient] = None


def get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(base_url=config.ollama_base_url, model=config.llm_model, temperature=0.3)
    return _llm


def get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(base_url=config.ollama_base_url, model=config.embeddings_model)
    return _embeddings


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=config.qdrant_url)
    return _qdrant


@contextmanager
def sqlite_conn():
    conn = sqlite3.connect(config.sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_sqlite():
    os.makedirs(os.path.dirname(config.sqlite_path) or ".", exist_ok=True)
    with sqlite_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS slots (
                id INTEGER PRIMARY KEY, zone TEXT, slot_number TEXT,
                available INTEGER DEFAULT 1, price_per_hour REAL);
            CREATE TABLE IF NOT EXISTS working_hours (
                day TEXT PRIMARY KEY, open_time TEXT, close_time TEXT);
            CREATE TABLE IF NOT EXISTS reservations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT, surname TEXT, car_number TEXT,
                start_time TEXT, end_time TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        """)
        # seed if empty
        if conn.execute("SELECT COUNT(*) FROM slots").fetchone()[0] == 0:
            zones = [("A", "10", 25.0), ("A", "11", 25.0), ("B", "20", 18.0),
                     ("B", "21", 18.0), ("C", "30", 12.0), ("C", "31", 12.0)]
            conn.executemany("INSERT INTO slots (zone, slot_number, price_per_hour) VALUES (?,?,?)",
                             [(z, n, p) for z, n, p in zones])
        if conn.execute("SELECT COUNT(*) FROM working_hours").fetchone()[0] == 0:
            hours = [("monday","07:00","22:00"),("tuesday","07:00","22:00"),
                     ("wednesday","07:00","22:00"),("thursday","07:00","22:00"),
                     ("friday","07:00","22:00"),("saturday","08:00","20:00"),
                     ("sunday","08:00","20:00")]
            conn.executemany("INSERT OR REPLACE INTO working_hours VALUES (?,?,?)", hours)
        conn.commit()


DYNAMIC_KEYWORDS = {"available", "free", "vacant", "slot", "price", "cost",
                    "hours", "working", "open", "close"}


def _is_dynamic(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in DYNAMIC_KEYWORDS)


def retrieve_static(query: str, top_k: int = 5) -> str:
    try:
        vec = get_embeddings().embed_query(query)
        hits = get_qdrant().query_points(
            collection_name=config.qdrant_collection,
            query=vec, limit=top_k,
        ).points
        return "\n\n".join(p.payload.get("text", "") for p in hits if p.payload)
    except Exception as e:
        logger.warning("qdrant retrieval failed: %s", e)
        return ""


def retrieve_dynamic(query: str) -> Optional[str]:
    if not _is_dynamic(query):
        return None
    parts = []
    lowered = query.lower()
    with sqlite_conn() as conn:
        if any(w in lowered for w in ("hour", "open", "close", "working")):
            rows = conn.execute("SELECT day, open_time, close_time FROM working_hours ORDER BY rowid").fetchall()
            parts.append("Working hours:\n" + "\n".join(f"  {r['day']}: {r['open_time']}-{r['close_time']}" for r in rows))
        if any(w in lowered for w in ("price", "cost", "how much")):
            rows = conn.execute("SELECT zone, MIN(price_per_hour) min_p, MAX(price_per_hour) max_p FROM slots GROUP BY zone").fetchall()
            parts.append("Prices:\n" + "\n".join(f"  Zone {r['zone']}: {r['min_p']:.0f}-{r['max_p']:.0f} UAH/h" for r in rows))
        if any(w in lowered for w in ("available", "free", "vacant", "slot")):
            rows = conn.execute("SELECT zone, slot_number, price_per_hour FROM slots WHERE available=1 ORDER BY zone").fetchall()
            if rows:
                parts.append("Available slots:\n" + "\n".join(f"  {r['zone']}-{r['slot_number']} ({r['price_per_hour']:.0f} UAH/h)" for r in rows))
            else:
                parts.append("No available slots.")
    return "\n\n".join(parts) if parts else None


def answer(query: str) -> str:
    """RAG answer — dynamic SQL path or static vector path with LLM."""
    dynamic = retrieve_dynamic(query)
    if dynamic:
        return dynamic
    context = retrieve_static(query)
    if not context:
        return "I don't have information about that. Ask about prices, hours, slots, or booking."
    prompt = ("You are a helpful parking assistant. Answer using only the context below. Be concise.\n\n"
              f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")
    try:
        return get_llm().invoke(prompt).strip()
    except Exception as e:
        logger.warning("llm failed: %s", e)
        return context[:500]