import uuid

import httpx
import numpy as np
import pytest
from redis.asyncio import Redis

from app.config import Settings
from app.main import create_app

REDIS_URL = "redis://localhost:6379/0"  # el motor de búsqueda solo indexa db 0


class FakeEmbedder:
    """Vectores deterministas: cada texto conocido mapea a un vector fijo."""

    dim = 4

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls = 0

    def embed(self, text: str) -> np.ndarray:
        self.calls += 1
        v = np.array(self.vectors[text], dtype=np.float32)
        return v / np.linalg.norm(v)


VECTORS = {
    "capital de francia": [1, 0, 0, 0],
    "cual es la capital de francia": [1, 0.1, 0, 0],   # coseno ≈ 0.995 -> HIT
    "capital francesa": [1, 0.5, 0, 0],                # coseno ≈ 0.894 -> MISS
    "receta de paella": [0, 0, 1, 0],                  # ortogonal -> MISS
}


@pytest.fixture
async def client():
    redis = Redis.from_url(REDIS_URL)
    try:
        await redis.ping()
    except Exception:
        pytest.skip("Redis no disponible en localhost:6379")

    suffix = uuid.uuid4().hex[:8]
    settings = Settings(redis_url=REDIS_URL, index_name=f"test_idx_{suffix}",
                        key_prefix=f"test_{suffix}:", similarity_threshold=0.95)
    embedder = FakeEmbedder(VECTORS)
    app = create_app(settings, embedder=embedder, redis=redis)

    calls = {"n": 0}
    import app.main as main_mod

    async def fake_backend(query: str) -> str:
        calls["n"] += 1
        return f"answer:{query}"

    original = main_mod.expensive_backend
    main_mod.expensive_backend = fake_backend
    await app.state.cache.ensure_index()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test") as c:
        c.backend_calls = calls
        c.embedder = embedder
        yield c

    main_mod.expensive_backend = original
    await app.state.cache.clear()
    await redis.ft(settings.index_name).dropindex()
    await redis.aclose()


async def test_miss_then_exact_hit(client):
    r1 = await client.post("/ask", json={"query": "capital de francia"})
    assert r1.status_code == 200
    assert r1.headers["x-cache"] == "MISS"

    r2 = await client.post("/ask", json={"query": "capital de francia"})
    assert r2.headers["x-cache"] == "HIT"
    assert float(r2.headers["x-cache-similarity"]) > 0.999
    assert r2.json() == r1.json()
    assert client.backend_calls["n"] == 1


async def test_exact_repeat_skips_embedding(client):
    await client.post("/ask", json={"query": "capital de francia"})
    calls = client.embedder.calls
    r = await client.post("/ask", json={"query": "  capital   de francia "})  # se normaliza
    assert r.headers["x-cache"] == "HIT"
    assert client.embedder.calls == calls


async def test_semantic_hit_above_threshold(client):
    await client.post("/ask", json={"query": "capital de francia"})
    r = await client.post("/ask", json={"query": "cual es la capital de francia"})
    assert r.headers["x-cache"] == "HIT"
    assert r.headers["x-cache-matched-query"] == "capital de francia"
    assert r.json()["answer"] == "answer:capital de francia"
    assert client.backend_calls["n"] == 1


async def test_below_threshold_is_miss(client):
    await client.post("/ask", json={"query": "capital de francia"})
    r = await client.post("/ask", json={"query": "capital francesa?"})
    assert r.headers["x-cache"] == "MISS"
    r = await client.post("/ask", json={"query": "receta de paella"})
    assert r.headers["x-cache"] == "MISS"
    assert client.backend_calls["n"] == 3


async def test_errors_are_not_cached(client):
    r = await client.post("/ask", json={"wrong": "x"})
    assert r.status_code == 422
    assert "x-cache" not in r.headers


async def test_non_cached_routes_pass_through(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert "x-cache" not in r.headers
