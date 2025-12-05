from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = ""

    # Vector DB
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # Graph DB (Neo4j)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # SQL DB
    DATABASE_URL: str = "sqlite:///./brain.db"
    # For PostgreSQL: "postgresql://user:password@localhost/brain"

    # Ollama (optional)
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:latest"

    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = "openai"
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Retrieval Settings
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
