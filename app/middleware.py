import json
import logging
import time

from anyio import to_thread
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.cache import SemanticCache
from app.embeddings import Embedder

logger = logging.getLogger("semcache")

# Cabeceras que no deben reutilizarse desde caché (se recalculan o son por-respuesta).
_SKIP_HEADERS = {"content-length", "date", "server", "x-cache", "x-cache-similarity",
                 "x-cache-latency-ms", "x-cache-matched-query"}


class SemanticCacheMiddleware:
    """Middleware ASGI puro (no BaseHTTPMiddleware) para interceptar el body sin overhead.

    Flujo: leer body -> extraer consulta -> embedding -> KNN en Redis.
      - similitud >= umbral: responder desde caché (X-Cache: HIT), sin tocar el endpoint.
      - si no: ejecutar el endpoint, capturar la respuesta 2xx y guardarla (X-Cache: MISS).
    Si Redis falla, el middleware se degrada a pass-through: la caché nunca tumba la API.
    """

    def __init__(self, app: ASGIApp, *, cache: SemanticCache, embedder: Embedder,
                 paths: list[str], query_field: str = "query"):
        self.app = app
        self.cache = cache
        self.embedder = embedder
        self.paths = set(paths)
        self.query_field = query_field

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in self.paths:
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        body = await _read_body(receive)
        replay = _replay_receive(body, receive)

        query = self._extract_query(body)
        if query is None:
            await self.app(scope, replay, send)
            return

        cache_scope = SemanticCache.scope_for(scope["method"], scope["path"])
        vector = None
        try:
            hit = await self.cache.exact_lookup(cache_scope, query)
            if hit is None:
                vector = await to_thread.run_sync(self.embedder.embed, query)
                hit = await self.cache.lookup(cache_scope, vector)
        except Exception:
            logger.exception("Semantic cache lookup failed; falling through")
            await self.app(scope, replay, send)
            return

        if hit is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000
            headers = [(k.encode(), v.encode()) for k, v in hit.headers]
            headers += [
                (b"content-length", str(len(hit.body)).encode()),
                (b"x-cache", b"HIT"),
                (b"x-cache-similarity", f"{hit.similarity:.4f}".encode()),
                (b"x-cache-latency-ms", f"{elapsed_ms:.2f}".encode()),
                # Los valores de cabecera HTTP son latin-1; se escapa lo demás.
                (b"x-cache-matched-query",
                 hit.matched_query.encode("latin-1", "backslashreplace")),
            ]
            await send({"type": "http.response.start", "status": hit.status, "headers": headers})
            await send({"type": "http.response.body", "body": hit.body})
            return

        status = 500
        resp_headers: list[tuple[bytes, bytes]] = []
        chunks: list[bytes] = []

        async def capture_send(message: Message) -> None:
            nonlocal status, resp_headers
            if message["type"] == "http.response.start":
                status = message["status"]
                resp_headers = list(message.get("headers", []))
                message = {**message, "headers": resp_headers + [(b"x-cache", b"MISS")]}
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
            await send(message)

        await self.app(scope, replay, capture_send)

        if 200 <= status < 300:
            try:
                headers = [
                    (k.decode("latin-1"), v.decode("latin-1"))
                    for k, v in resp_headers
                    if k.decode("latin-1").lower() not in _SKIP_HEADERS
                ]
                await self.cache.store(cache_scope, query, vector, status, headers, b"".join(chunks))
            except Exception:
                logger.exception("Semantic cache store failed")

    def _extract_query(self, body: bytes) -> str | None:
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        query = payload.get(self.query_field)
        if not isinstance(query, str) or not query.strip():
            return None
        return " ".join(query.split())


async def _read_body(receive: Receive) -> bytes:
    chunks = []
    while True:
        message = await receive()
        if message["type"] == "http.request":
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        elif message["type"] == "http.disconnect":
            break
    return b"".join(chunks)


def _replay_receive(body: bytes, receive: Receive) -> Receive:
    """Devuelve el body ya consumido al endpoint y delega el resto (p. ej. disconnect)."""
    sent = False

    async def wrapped() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return wrapped
