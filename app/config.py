from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración leída de variables de entorno (prefijo SEMCACHE_) o de .env."""

    model_config = SettingsConfigDict(
        env_prefix="SEMCACHE_", env_file=".env", extra="ignore", populate_by_name=True
    )

    redis_url: str = "redis://localhost:6379/0"
    index_name: str = "semcache_idx"
    key_prefix: str = "semcache:"

    # Modelo multilingüe (384 dims): funciona bien con consultas en español e inglés.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # False desactiva el middleware (útil para medir la línea base sin caché).
    enabled: bool = True
    # Similitud de coseno mínima para considerar un HIT (distancia = 1 - similitud).
    similarity_threshold: float = 0.95
    ttl_seconds: int = 3600
    # TTL por ruta, p. ej. {"/news": 300}; las rutas no listadas usan ttl_seconds.
    route_ttls: dict[str, int] = {}

    # Rutas interceptadas y campo JSON del body que contiene la consulta.
    cached_paths: list[str] = ["/ask"]
    query_field: str = "query"
    # Cabecera que aísla la caché por cliente; sin ella se usa el espacio "public".
    tenant_header: str = "x-tenant-id"
    # Cuánto espera una petición duplicada a que termine la que está en vuelo.
    inflight_wait_ms: int = 30_000

    # LLM: se usa el primer proveedor con clave (Groq, Grok, Gemini); sin ninguna, un
    # backend simulado (demo, tests y CI sin coste).
    groq_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GROQ_API_KEY", "SEMCACHE_GROQ_API_KEY")
    )
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    xai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("XAI_API_KEY", "SEMCACHE_XAI_API_KEY")
    )
    xai_model: str = "grok-4.3"
    xai_base_url: str = "https://api.x.ai/v1"
    gemini_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "SEMCACHE_GEMINI_API_KEY")
    )
    gemini_model: str = "gemini-2.5-flash"
    system_prompt: str = "Responde de forma breve y precisa, en el idioma de la pregunta."
    # USD por millón de tokens, para estimar el ahorro. Sin valor se usa la tarifa del modelo
    # por defecto de cada proveedor (ver app/llm.py); defínelo si cambias de modelo.
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None
    simulated_latency_s: float = 1.5

    def ttl_for(self, path: str) -> int:
        return self.route_ttls.get(path, self.ttl_seconds)


@lru_cache
def get_settings() -> Settings:
    return Settings()
