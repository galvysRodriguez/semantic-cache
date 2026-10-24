import asyncio
import uuid
from contextlib import asynccontextmanager

import httpx
import pytest
from redis.asyncio import Redis

from app.config import Settings
from app.llm import LLMResult
from app.main import create_app

# El motor de búsqueda de Redis solo indexa la db 0; el aislamiento va por índice/prefijo.
REDIS_URL = "redis://localhost:6379/0"


class FakeLLM:
    name = "fake"

    def __init__(self, delay_s: float = 0.0):
        self.calls = 0
        self.delay_s = delay_s

    async def generate(self, prompt: str) -> LLMResult:
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return LLMResult(
            answer=f"answer:{prompt}",
            model=self.name,
            input_tokens=10,
            output_tokens=40,
            cost_usd=0.0001,
        )


@pytest.fixture
async def redis():
    client = Redis.from_url(REDIS_URL)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis no disponible en localhost:6379")
    yield client
    await client.aclose()


@pytest.fixture
def make_client(redis):
    """Fábrica: crea una app aislada (índice y prefijo únicos) y su cliente HTTP."""

    @asynccontextmanager
    async def factory(embedder, llm=None, **overrides):
        suffix = uuid.uuid4().hex[:8]
        settings = Settings(
            redis_url=REDIS_URL,
            index_name=f"test_idx_{suffix}",
            key_prefix=f"test_{suffix}:",
            gemini_api_key=None,
            xai_api_key=None,
            **overrides,
        )
        llm = llm or FakeLLM()
        app = create_app(settings, embedder=embedder, redis=redis, llm=llm)
        await app.state.cache.ensure_index()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                client.app = app
                client.llm = llm
                yield client
        finally:
            await app.state.cache.clear()
            await app.state.stats.reset()
            await redis.ft(settings.index_name).dropindex()

    return factory
