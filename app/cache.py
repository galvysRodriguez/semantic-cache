import asyncio
import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np
from redis.asyncio import Redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError
from redis.typing import EncodableT, FieldT


@dataclass
class CachedResponse:
    status: int
    headers: list[tuple[str, str]]
    body: bytes
    similarity: float
    matched_query: str


class SemanticCache:
    """Índice vectorial HNSW en Redis con métrica COSINE.

    Cada entrada es un HASH con el embedding de la consulta y la respuesta HTTP serializada.
    El campo `scope` (TAG) separa entradas por método + ruta + tenant para no mezclar
    endpoints ni clientes. Claves: `{prefix}{scope}:{sha256(consulta)}`.
    """

    def __init__(
        self, redis: Redis, *, index_name: str, prefix: str, dim: int, threshold: float, ttl: int
    ):
        self.redis = redis
        self.index_name = index_name
        self.prefix = prefix
        self.dim = dim
        self.threshold = threshold
        self.ttl = ttl
        # Fuera de `prefix` para que el índice no los recoja.
        self.lock_prefix = f"{index_name}:lock:"

    async def ensure_index(self) -> None:
        try:
            await self.redis.ft(self.index_name).info()
            return
        except ResponseError:
            pass
        schema = [
            TagField("scope"),
            TextField("query"),
            NumericField("created_at"),
            VectorField(
                "embedding",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": self.dim, "DISTANCE_METRIC": "COSINE"},
            ),
        ]
        definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
        await self.redis.ft(self.index_name).create_index(schema, definition=definition)

    @staticmethod
    def scope_for(method: str, path: str, tenant: str = "public") -> str:
        # Hash hex: evita escapar caracteres especiales ("/", "-") en la consulta TAG.
        return hashlib.sha1(f"{method} {path} {tenant}".encode()).hexdigest()[:16]

    @staticmethod
    def _query_hash(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:32]

    def key_for(self, scope: str, query: str) -> str:
        # Clave determinista: una consulta repetida se resuelve con un HMGET, sin embedding.
        return f"{self.prefix}{scope}:{self._query_hash(query)}"

    async def exact_lookup(self, scope: str, query: str) -> CachedResponse | None:
        status, headers, body = await self.redis.hmget(
            self.key_for(scope, query), "status", "headers", "body"
        )
        if status is None or headers is None or body is None:
            return None
        return CachedResponse(
            status=int(status),
            headers=[tuple(h) for h in json.loads(headers)],
            body=_text(body).encode("latin-1"),
            similarity=1.0,
            matched_query=query,
        )

    async def lookup(
        self, scope: str, vector: np.ndarray, query: str, k: int = 3
    ) -> CachedResponse | None:
        q = (
            Query(f"(@scope:{{{scope}}})=>[KNN {k} @embedding $vec AS distance]")
            .sort_by("distance")
            .return_fields("distance", "query", "status", "headers", "body")
            .dialect(2)
        )
        res = await self.redis.ft(self.index_name).search(
            q, query_params={"vec": vector.astype(np.float32).tobytes()}
        )
        wanted = entity_tokens(query)
        for doc in res.docs:
            similarity = 1.0 - float(doc.distance)
            if similarity < self.threshold:
                break  # ordenados por distancia: el resto tampoco llega
            # El modelo apenas distingue números: "pedido 10023" vs "pedido 10024" da 0.96.
            if entity_tokens(doc.query) != wanted:
                continue
            return CachedResponse(
                status=int(doc.status),
                headers=[tuple(h) for h in json.loads(doc.headers)],
                body=doc.body.encode("latin-1"),
                similarity=similarity,
                matched_query=doc.query,
            )
        return None

    async def store(
        self,
        scope: str,
        query: str,
        vector: np.ndarray,
        status: int,
        headers: list[tuple[str, str]],
        body: bytes,
        ttl: int | None = None,
    ) -> str:
        key = self.key_for(scope, query)
        ttl = self.ttl if ttl is None else ttl
        mapping: dict[FieldT, EncodableT] = {
            "scope": scope,
            "query": query,
            "created_at": time.time(),
            "embedding": vector.astype(np.float32).tobytes(),
            "status": status,
            "headers": json.dumps(headers),
            # latin-1 es biyectivo byte<->char: guarda cualquier body sin pérdida.
            "body": body.decode("latin-1"),
        }
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=mapping)
            if ttl > 0:
                pipe.expire(key, ttl)
            await pipe.execute()
        return key

    # --- single-flight: una sola llamada al backend por consulta idéntica en vuelo -------

    def _lock_key(self, scope: str, query: str) -> str:
        return f"{self.lock_prefix}{scope}:{self._query_hash(query)}"

    async def acquire_inflight(self, scope: str, query: str, ttl_ms: int) -> bool:
        return bool(await self.redis.set(self._lock_key(scope, query), "1", nx=True, px=ttl_ms))

    async def release_inflight(self, scope: str, query: str) -> None:
        await self.redis.delete(self._lock_key(scope, query))

    async def wait_inflight(
        self, scope: str, query: str, timeout_s: float, interval_s: float = 0.02
    ) -> CachedResponse | None:
        """Espera a que la petición líder guarde su respuesta.

        Devuelve None si el líder termina sin cachear (p. ej. error 5xx) o si vence el plazo.
        """
        lock_key = self._lock_key(scope, query)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            hit = await self.exact_lookup(scope, query)
            if hit is not None:
                return hit
            if not await self.redis.exists(lock_key):
                return await self.exact_lookup(scope, query)
            await asyncio.sleep(interval_s)
        return None

    # --- invalidación --------------------------------------------------------------------

    async def invalidate_scope(self, scope: str) -> int:
        return await self._delete_matching(f"{self.prefix}{scope}:*")

    async def clear(self) -> int:
        return await self._delete_matching(f"{self.prefix}*")

    async def _delete_matching(self, pattern: str) -> int:
        deleted = 0
        async for key in self.redis.scan_iter(match=pattern, count=500):
            deleted += await self.redis.delete(key)
        return deleted


def _text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value


def entity_tokens(query: str) -> frozenset[str]:
    """Tokens con algún dígito (IDs, cantidades, años): deben coincidir para aceptar un HIT."""
    return frozenset(t for t in query.split() if any(c.isdigit() for c in t))
