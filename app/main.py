import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel
from redis.asyncio import Redis

from app.cache import SemanticCache
from app.config import Settings, get_settings
from app.embeddings import Embedder, FastEmbedEmbedder
from app.llm import LLMBackend, build_backend
from app.middleware import LLM_COST_HEADER, LLM_TOKENS_HEADER, SemanticCacheMiddleware
from app.stats import CacheStats

logging.basicConfig(level=logging.INFO)
STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    query: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float


class AskResponse(BaseModel):
    query: str
    answer: str
    model: str
    usage: Usage


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    redis: Redis | None = None,
    llm: LLMBackend | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    embedder = embedder or FastEmbedEmbedder(settings.embedding_model)
    redis = redis or Redis.from_url(settings.redis_url)
    llm = llm or build_backend(settings)
    cache = SemanticCache(
        redis,
        index_name=settings.index_name,
        prefix=settings.key_prefix,
        dim=embedder.dim,
        threshold=settings.similarity_threshold,
        ttl=settings.ttl_seconds,
    )
    stats = CacheStats(redis, f"{settings.index_name}:stats")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await cache.ensure_index()
        logging.getLogger("semcache").info(
            "LLM backend: %s | cache enabled: %s", llm.name, settings.enabled
        )
        yield
        await redis.aclose()

    app = FastAPI(title="Semantic Cache", lifespan=lifespan)
    app.state.cache = cache
    app.state.stats = stats
    if settings.enabled:
        app.add_middleware(
            SemanticCacheMiddleware, cache=cache, embedder=embedder, stats=stats, settings=settings
        )
    app.mount("/metrics", make_asgi_app())

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest, response: Response) -> AskResponse:
        result = await llm.generate(req.query)
        # El middleware lee estas cabeceras para contabilizar gasto (MISS) y ahorro (HIT).
        response.headers[LLM_TOKENS_HEADER] = str(result.input_tokens + result.output_tokens)
        response.headers[LLM_COST_HEADER] = f"{result.cost_usd:.8f}"
        return AskResponse(
            query=req.query,
            answer=result.answer,
            model=result.model,
            usage=Usage(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
            ),
        )

    @app.get("/", include_in_schema=False)
    async def demo() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict:
        return {"redis": await redis.ping(), "llm": llm.name, "cache_enabled": settings.enabled}

    @app.get("/stats")
    async def get_stats() -> dict:
        return {
            **await stats.snapshot(),
            "llm": llm.name,
            "similarity_threshold": settings.similarity_threshold,
        }

    @app.delete("/stats")
    async def reset_stats() -> dict:
        await stats.reset()
        return {"reset": True}

    @app.delete("/cache")
    async def invalidate(request: Request, path: str | None = None) -> dict:
        """Sin `path` vacía toda la caché; con `path` solo esa ruta para el tenant actual."""
        if path is None:
            return {"deleted": await cache.clear()}
        tenant = request.headers.get(settings.tenant_header, "public")
        scope = SemanticCache.scope_for("POST", path, tenant)
        return {"deleted": await cache.invalidate_scope(scope), "path": path, "tenant": tenant}

    return app
