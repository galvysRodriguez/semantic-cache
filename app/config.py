from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración leída de variables de entorno (prefijo SEMCACHE_) o de .env."""

    model_config = SettingsConfigDict(env_prefix="SEMCACHE_", env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    index_name: str = "semcache_idx"
    key_prefix: str = "semcache:"

    # Modelo multilingüe (384 dims): funciona bien con consultas en español e inglés.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Similitud de coseno mínima para considerar un HIT (distancia = 1 - similitud).
    similarity_threshold: float = 0.95
    ttl_seconds: int = 3600

    # Rutas interceptadas y campo JSON del body que contiene la consulta.
    cached_paths: list[str] = ["/ask"]
    query_field: str = "query"


@lru_cache
def get_settings() -> Settings:
    return Settings()
