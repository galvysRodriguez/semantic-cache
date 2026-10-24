import contextlib
import json
import logging
import re
import time
import unicodedata

from anyio import to_thread
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.cache import CachedResponse, SemanticCache
from app.config import Settings
from app.embeddings import Embedder
from app.stats import CacheStats

logger = logging.getLogger("semcache")

# Cabeceras que no deben reutilizarse desde caché (se recalculan o son por-respuesta).
_SKIP_HEADERS = {
    "content-length",
    "date",
    "server",
    "x-cache",
    "x-cache-similarity",
    "x-cache-latency-ms",
    "x-cache-matched-query",
}
# El endpoint informa del consumo del LLM con estas cabeceras; en un HIT se convierten en ahorro.
LLM_TOKENS_HEADER = "x-llm-tokens"
LLM_COST_HEADER = "x-llm-cost-usd"
_PUNCT_RE = re.compile(r"[^\w\s]")


class SemanticCacheMiddleware:
    """Middleware ASGI puro (no BaseHTTPMiddleware) para interceptar el body sin overhead.

    Flujo: leer body -> normalizar consulta -> clave exacta -> embedding -> KNN en Redis.
      - coincidencia exacta o similitud >= umbral: responder desde caché (X-Cache: HIT).
      - misma consulta ya en vuelo: esperar su respuesta en vez de llamar otra vez al backend.
      - si no: ejecutar el endpoint, capturar la respuesta 2xx y guardarla (X-Cache: MISS).
    Si Redis falla, el middleware se degrada a pass-through: la caché nunca tumba la API.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        cache: SemanticCache,
        embedder: Embedder,
        stats: CacheStats,
        settings: Settings,
    ):
        self.app = app
        self.cache = cache
        self.embedder = embedder
        self.stats = stats
        self.settings = settings
        self.paths = set(settings.cached_paths)
        self.tenant_header = settings.tenant_header.lower().encode()

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

        cache_scope = SemanticCache.scope_for(scope["method"], scope["path"], self._tenant(scope))
        vector = None
        leader = False
        try:
            outcome = "hit_exact"
            hit = await self.cache.exact_lookup(cache_scope, query)
            if hit is None:
                outcome = "hit_semantic"
                vector = await to_thread.run_sync(self.embedder.embed, query)
                hit = await self.cache.lookup(cache_scope, vector, query)
            if hit is None:
                outcome = "hit_coalesced"
                leader = await self.cache.acquire_inflight(
                    cache_scope, query, self.settings.inflight_wait_ms
                )
                if leader:
                    # Doble comprobación: el líder anterior pudo guardar y soltar el lock
                    # entre nuestra búsqueda y el SET NX.
                    hit = await self.cache.exact_lookup(cache_scope, query)
                    if hit is not None:
                        await self.cache.release_inflight(cache_scope, query)
                        leader = False
                else:
                    hit = await self.cache.wait_inflight(
                        cache_scope, query, self.settings.inflight_wait_ms / 1000
                    )
        except Exception:
            logger.exception("Semantic cache lookup failed; falling through")
            if leader:
                with contextlib.suppress(Exception):
                    await self.cache.release_inflight(cache_scope, query)
            await self.app(scope, replay, send)
            await self._record("bypass", start)
            return

        if hit is not None:
            await self._send_hit(send, hit, start)
            tokens, cost = _llm_usage(hit.headers)
            await self._record(outcome, start, tokens, cost)
            return

        try:
            status, headers, resp_body = await self._call_and_capture(scope, replay, send)
            if 200 <= status < 300 and vector is not None:
                try:
                    await self.cache.store(
                        cache_scope,
                        query,
                        vector,
                        status,
                        headers,
                        resp_body,
                        ttl=self.settings.ttl_for(scope["path"]),
                    )
                except Exception:
                    logger.exception("Semantic cache store failed")
        finally:
            if leader:
                await self.cache.release_inflight(cache_scope, query)
        tokens, cost = _llm_usage(headers)
        await self._record("miss", start, tokens, cost)

    def _tenant(self, scope: Scope) -> str:
        for name, value in scope["headers"]:
            if name == self.tenant_header and value:
                return value.decode("latin-1")
        return "public"

    async def _send_hit(self, send: Send, hit: CachedResponse, start: float) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000
        tokens, cost = _llm_usage(hit.headers)
        headers = [
            (k.encode(), v.encode())
            for k, v in hit.headers
            if k.lower() not in (LLM_TOKENS_HEADER, LLM_COST_HEADER)
        ]
        headers += [
            (b"content-length", str(len(hit.body)).encode()),
            (b"x-cache", b"HIT"),
            (b"x-cache-similarity", f"{hit.similarity:.4f}".encode()),
            (b"x-cache-latency-ms", f"{elapsed_ms:.2f}".encode()),
            # Los valores de cabecera HTTP son latin-1; se escapa lo demás.
            (b"x-cache-matched-query", hit.matched_query.encode("latin-1", "backslashreplace")),
            (b"x-cache-saved-tokens", str(tokens).encode()),
            (b"x-cache-saved-usd", f"{cost:.8f}".encode()),
        ]
        await send({"type": "http.response.start", "status": hit.status, "headers": headers})
        await send({"type": "http.response.body", "body": hit.body})

    async def _call_and_capture(
        self, scope: Scope, receive: Receive, send: Send
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        status = 500
        resp_headers: list[tuple[bytes, bytes]] = []
        chunks: list[bytes] = []

        async def capture_send(message: Message) -> None:
            nonlocal status, resp_headers
            if message["type"] == "http.response.start":
                status = message["status"]
                resp_headers = list(message.get("headers", []))
                message = {**message, "headers": [*resp_headers, (b"x-cache", b"MISS")]}
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
            await send(message)

        await self.app(scope, receive, capture_send)
        headers = [
            (k.decode("latin-1"), v.decode("latin-1"))
            for k, v in resp_headers
            if k.decode("latin-1").lower() not in _SKIP_HEADERS
        ]
        return status, headers, b"".join(chunks)

    async def _record(self, outcome: str, start: float, tokens: int = 0, cost: float = 0.0) -> None:
        try:
            await self.stats.record(
                outcome, time.perf_counter() - start, tokens=tokens, cost_usd=cost
            )
        except Exception:
            logger.exception("Failed to record cache stats")

    def _extract_query(self, body: bytes) -> str | None:
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        query = payload.get(self.settings.query_field)
        if not isinstance(query, str):
            return None
        return normalize_query(query) or None


def normalize_query(text: str) -> str:
    """Minúsculas, sin tildes, sin puntuación y con espacios colapsados.

    El modelo de embeddings es sensible a mayúsculas y tildes: "¿Cuál es la capital de
    Francia?" vs "dime la capital de francia" da 0.88 en crudo y 0.97 normalizado.
    """
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def _llm_usage(headers: list[tuple[str, str]]) -> tuple[int, float]:
    values = {k.lower(): v for k, v in headers}
    try:
        return int(values.get(LLM_TOKENS_HEADER, 0)), float(values.get(LLM_COST_HEADER, 0.0))
    except ValueError:
        return 0, 0.0


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
