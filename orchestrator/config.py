import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # LLM
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")

    # Chatbot RAG
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "parking_static")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./data/parking.db")

    # Admin service
    admin_api_url: str = os.getenv("ADMIN_API_URL", "http://localhost:8001")

    # MCP server
    mcp_url: str = os.getenv("MCP_URL", "http://localhost:8002/mcp")
    mcp_token: str = os.getenv("MCP_TOKEN", "change-me-parking-mcp-secret")

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()