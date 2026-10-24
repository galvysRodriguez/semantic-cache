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
    El campo `scope` (TAG) separa entradas por ruta/método para no mezclar endpoints.
    """

    def __init__(self, redis: Redis, *, index_name: str, prefix: str, dim: int,
                 threshold: float, ttl: int):
        self.redis = redis
        self.index_name = index_name
        self.prefix = prefix
        self.dim = dim
        self.threshold = threshold
        self.ttl = ttl

    async def ensure_index(self) -> None:
        try:
            await self.redis.ft(self.index_name).info()
            return
        except ResponseError:
            pass
        schema = (
            TagField("scope"),
            TextField("query"),
            NumericField("created_at"),
            VectorField(
                "embedding",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": self.dim, "DISTANCE_METRIC": "COSINE"},
            ),
        )
        definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
        await self.redis.ft(self.index_name).create_index(schema, definition=definition)

    @staticmethod
    def scope_for(method: str, path: str) -> str:
        # Hash hex: evita escapar caracteres especiales ("/", "-") en la consulta TAG.
        return hashlib.sha1(f"{method} {path}".encode()).hexdigest()[:16]

    def key_for(self, scope: str, query: str) -> str:
        # Clave determinista: una consulta repetida se resuelve con un HMGET, sin embedding.
        return f"{self.prefix}{scope}:{hashlib.sha256(query.encode()).hexdigest()[:32]}"

    async def exact_lookup(self, scope: str, query: str) -> CachedResponse | None:
        status, headers, body = await self.redis.hmget(
            self.key_for(scope, query), "status", "headers", "body"
        )
        if status is None:
            return None
        return CachedResponse(
            status=int(status),
            headers=[tuple(h) for h in json.loads(headers)],
            body=body.decode("utf-8").encode("latin-1"),
            similarity=1.0,
            matched_query=query,
        )

    async def lookup(self, scope: str, vector: np.ndarray) -> CachedResponse | None:
        q = (
            Query(f"(@scope:{{{scope}}})=>[KNN 1 @embedding $vec AS distance]")
            .sort_by("distance")
            .return_fields("distance", "query", "status", "headers", "body")
            .dialect(2)
        )
        res = await self.redis.ft(self.index_name).search(
            q, query_params={"vec": vector.astype(np.float32).tobytes()}
        )
        if not res.docs:
            return None
        doc = res.docs[0]
        similarity = 1.0 - float(doc.distance)
        if similarity < self.threshold:
            return None
        return CachedResponse(
            status=int(doc.status),
            headers=[tuple(h) for h in json.loads(doc.headers)],
            body=doc.body.encode("latin-1"),
            similarity=similarity,
            matched_query=doc.query,
        )

    async def store(self, scope: str, query: str, vector: np.ndarray, status: int,
                    headers: list[tuple[str, str]], body: bytes) -> str:
        key = self.key_for(scope, query)
        mapping = {
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
            if self.ttl > 0:
                pipe.expire(key, self.ttl)
            await pipe.execute()
        return key

    async def clear(self) -> int:
        deleted = 0
        async for key in self.redis.scan_iter(match=f"{self.prefix}*", count=500):
            deleted += await self.redis.delete(key)
        return deleted
