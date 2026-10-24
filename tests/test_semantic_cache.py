"""Tests del middleware con embeddings falsos y deterministas (rápidos, sin modelo)."""

import asyncio

import numpy as np
import pytest

from tests.conftest import FakeLLM


class FakeEmbedder:
    """Cada texto (ya normalizado) mapea a un vector fijo."""

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
    "cual es la capital de francia": [1, 0.1, 0, 0],  # coseno ≈ 0.995 -> HIT
    "capital francesa": [1, 0.5, 0, 0],  # coseno ≈ 0.894 -> MISS
    "receta de paella": [0, 0, 1, 0],  # ortogonal -> MISS
}


@pytest.fixture
async def client(make_client):
    async with make_client(FakeEmbedder(VECTORS)) as c:
        yield c


async def ask(client, query, **headers):
    return await client.post("/ask", json={"query": query}, headers=headers)


async def test_miss_then_exact_hit(client):
    r1 = await ask(client, "capital de francia")
    assert r1.status_code == 200
    assert r1.headers["x-cache"] == "MISS"

    r2 = await ask(client, "capital de francia")
    assert r2.headers["x-cache"] == "HIT"
    assert float(r2.headers["x-cache-similarity"]) > 0.999
    assert r2.json() == r1.json()
    assert client.llm.calls == 1


async def test_exact_repeat_skips_embedding(client):
    await ask(client, "capital de francia")
    embedder = client.app.user_middleware[0].kwargs["embedder"]
    calls = embedder.calls
    r = await ask(client, "  Capital   de FRANCIA? ")  # se normaliza a la misma clave
    assert r.headers["x-cache"] == "HIT"
    assert embedder.calls == calls


async def test_semantic_hit_above_threshold(client):
    await ask(client, "capital de francia")
    r = await ask(client, "cual es la capital de francia")
    assert r.headers["x-cache"] == "HIT"
    assert r.headers["x-cache-matched-query"] == "capital de francia"
    assert r.json()["answer"] == "answer:capital de francia"
    assert client.llm.calls == 1


async def test_below_threshold_is_miss(client):
    await ask(client, "capital de francia")
    assert (await ask(client, "capital francesa?")).headers["x-cache"] == "MISS"
    assert (await ask(client, "receta de paella")).headers["x-cache"] == "MISS"
    assert client.llm.calls == 3


async def test_errors_are_not_cached(client):
    r = await client.post("/ask", json={"wrong": "x"})
    assert r.status_code == 422
    assert "x-cache" not in r.headers


async def test_non_cached_routes_pass_through(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert "x-cache" not in r.headers


# --- tenants -------------------------------------------------------------------------


async def test_tenants_do_not_share_entries(client):
    await ask(client, "capital de francia", **{"x-tenant-id": "acme"})
    r = await ask(client, "capital de francia", **{"x-tenant-id": "globex"})
    assert r.headers["x-cache"] == "MISS"
    r = await ask(client, "capital de francia", **{"x-tenant-id": "acme"})
    assert r.headers["x-cache"] == "HIT"
    assert client.llm.calls == 2


# --- single-flight -------------------------------------------------------------------


async def test_concurrent_identical_requests_call_backend_once(make_client):
    async with make_client(FakeEmbedder(VECTORS), llm=FakeLLM(delay_s=0.3)) as c:
        responses = await asyncio.gather(*(ask(c, "capital de francia") for _ in range(10)))
        assert c.llm.calls == 1
        assert sorted(r.headers["x-cache"] for r in responses) == ["HIT"] * 9 + ["MISS"]
        assert len({r.text for r in responses}) == 1
        stats = (await c.get("/stats")).json()
        assert stats["by_outcome"]["hit_coalesced"] == 9


async def test_waiters_fall_back_when_leader_fails(make_client):
    class FlakyLLM(FakeLLM):
        async def generate(self, prompt):
            if self.calls == 0:
                self.calls += 1
                await asyncio.sleep(self.delay_s)  # el seguidor ya está esperando al líder
                raise RuntimeError("upstream caído")
            return await super().generate(prompt)

    async with make_client(FakeEmbedder(VECTORS), llm=FlakyLLM(delay_s=0.1)) as c:
        # El cliente ASGI de pruebas propaga la excepción del líder; el seguidor debe
        # llamar él mismo al backend en lugar de quedarse colgado.
        results = await asyncio.gather(
            ask(c, "capital de francia"), ask(c, "capital de francia"), return_exceptions=True
        )
        ok = [r for r in results if not isinstance(r, Exception)]
        assert len(ok) == 1 and ok[0].status_code == 200
        assert c.llm.calls == 2


# --- métricas ------------------------------------------------------------------------


async def test_stats_track_hits_misses_and_savings(client):
    await ask(client, "capital de francia")  # miss
    await ask(client, "capital de francia")  # hit exacto
    r = await ask(client, "cual es la capital de francia")  # hit semántico
    assert r.headers["x-cache-saved-tokens"] == "50"
    assert "x-llm-tokens" not in r.headers

    s = (await client.get("/stats")).json()
    assert s["hits"] == 2 and s["misses"] == 1
    assert s["by_outcome"]["hit_exact"] == 1 and s["by_outcome"]["hit_semantic"] == 1
    assert s["hit_rate"] == pytest.approx(2 / 3)
    assert s["tokens_spent"] == 50 and s["tokens_saved"] == 100
    assert s["cost_saved_usd"] == pytest.approx(0.0002)
    assert s["llm_calls_saved"] == 2


async def test_prometheus_endpoint(client):
    await ask(client, "capital de francia")
    r = await client.get("/metrics/")
    assert r.status_code == 200
    assert 'semcache_requests_total{outcome="miss"}' in r.text


# --- invalidación y TTL --------------------------------------------------------------


async def test_invalidate_single_route(client):
    await ask(client, "capital de francia")
    r = await client.delete("/cache", params={"path": "/ask"})
    assert r.json()["deleted"] == 1
    assert (await ask(client, "capital de francia")).headers["x-cache"] == "MISS"


async def test_invalidate_is_scoped_to_tenant(client):
    await ask(client, "capital de francia", **{"x-tenant-id": "acme"})
    await ask(client, "capital de francia")
    r = await client.delete("/cache", params={"path": "/ask"}, headers={"x-tenant-id": "acme"})
    assert r.json()["deleted"] == 1
    assert (await ask(client, "capital de francia")).headers["x-cache"] == "HIT"


async def test_route_ttl_override(make_client, redis):
    async with make_client(FakeEmbedder(VECTORS), route_ttls={"/ask": 42}) as c:
        await ask(c, "capital de francia")
        keys = [k async for k in redis.scan_iter(match=f"{c.app.state.cache.prefix}*")]
        assert len(keys) == 1
        assert 0 < await redis.ttl(keys[0]) <= 42


# --- resiliencia ---------------------------------------------------------------------


async def test_redis_failure_falls_through_to_backend(client, monkeypatch):
    async def boom(*args, **kwargs):
        raise ConnectionError("redis caído")

    monkeypatch.setattr(client.app.state.cache, "exact_lookup", boom)
    r = await ask(client, "capital de francia")
    assert r.status_code == 200
    assert r.json()["answer"] == "answer:capital de francia"
