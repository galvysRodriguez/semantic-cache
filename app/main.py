import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from redis.asyncio import Redis

from app.cache import SemanticCache
from app.config import Settings, get_settings
from app.embeddings import Embedder, FastEmbedEmbedder
from app.middleware import SemanticCacheMiddleware

logging.basicConfig(level=logging.INFO)


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    query: str
    answer: str


async def expensive_backend(query: str) -> str:
    """Simula una llamada lenta (LLM, RAG, API externa). Sustitúyela por la real."""
    await asyncio.sleep(1.5)
    return f"Respuesta generada para: {query!r}"


def create_app(settings: Settings | None = None, *, embedder: Embedder | None = None,
               redis: Redis | None = None) -> FastAPI:
    settings = settings or get_settings()
    embedder = embedder or FastEmbedEmbedder(settings.embedding_model)
    redis = redis or Redis.from_url(settings.redis_url)
    cache = SemanticCache(
        redis,
        index_name=settings.index_name,
        prefix=settings.key_prefix,
        dim=embedder.dim,
        threshold=settings.similarity_threshold,
        ttl=settings.ttl_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await cache.ensure_index()
        yield
        await redis.aclose()

    app = FastAPI(title="Semantic Cache", lifespan=lifespan)
    app.state.cache = cache
    app.add_middleware(
        SemanticCacheMiddleware,
        cache=cache,
        embedder=embedder,
        paths=settings.cached_paths,
        query_field=settings.query_field,
    )

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        return AskResponse(query=req.query, answer=await expensive_backend(req.query))

    @app.get("/health")
    async def health() -> dict:
        return {"redis": await redis.ping()}

    @app.delete("/cache")
    async def clear_cache() -> dict:
        return {"deleted": await cache.clear()}

    return app

