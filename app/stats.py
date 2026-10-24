from prometheus_client import Counter, Histogram
from redis.asyncio import Redis

# Métricas Prometheus (por proceso), expuestas en /metrics.
REQUESTS = Counter("semcache_requests_total", "Peticiones interceptadas", ["outcome"])
LATENCY = Histogram(
    "semcache_latency_seconds",
    "Latencia de extremo a extremo en el middleware",
    ["outcome"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
TOKENS = Counter("semcache_llm_tokens_total", "Tokens de LLM", ["kind"])  # spent | saved
COST = Counter("semcache_llm_cost_usd_total", "Coste estimado del LLM en USD", ["kind"])

HIT_OUTCOMES = ("hit_exact", "hit_semantic", "hit_coalesced")
OUTCOMES = (*HIT_OUTCOMES, "miss", "bypass")


class CacheStats:
    """Contadores agregados en Redis: compartidos entre réplicas y persistentes entre reinicios."""

    def __init__(self, redis: Redis, key: str):
        self.redis = redis
        self.key = key

    async def record(
        self, outcome: str, latency_s: float, *, tokens: int = 0, cost_usd: float = 0.0
    ) -> None:
        """`tokens`/`cost_usd`: gastados si es miss, ahorrados si es hit."""
        kind = "saved" if outcome in HIT_OUTCOMES else "spent"
        REQUESTS.labels(outcome).inc()
        LATENCY.labels(outcome).observe(latency_s)
        if tokens:
            TOKENS.labels(kind).inc(tokens)
            COST.labels(kind).inc(cost_usd)

        async with self.redis.pipeline(transaction=False) as pipe:
            pipe.hincrby(self.key, f"{outcome}:count", 1)
            pipe.hincrbyfloat(self.key, f"{outcome}:latency_ms", latency_s * 1000)
            if tokens:
                pipe.hincrby(self.key, f"tokens_{kind}", tokens)
                pipe.hincrbyfloat(self.key, f"cost_{kind}_usd", cost_usd)
            await pipe.execute()

    async def snapshot(self) -> dict:
        raw = {_text(k): float(v) for k, v in (await self.redis.hgetall(self.key)).items()}
        counts = {o: int(raw.get(f"{o}:count", 0)) for o in OUTCOMES}
        avg = {
            o: raw.get(f"{o}:latency_ms", 0.0) / counts[o] if counts[o] else None for o in OUTCOMES
        }

        hits = sum(counts[o] for o in HIT_OUTCOMES)
        total = hits + counts["miss"]
        hit_ms = sum(raw.get(f"{o}:latency_ms", 0.0) for o in HIT_OUTCOMES) / hits if hits else None
        miss_ms = avg["miss"]
        return {
            "requests": total + counts["bypass"],
            "hits": hits,
            "misses": counts["miss"],
            "hit_rate": hits / total if total else 0.0,
            "by_outcome": counts,
            "avg_latency_ms": {"hit": hit_ms, "miss": miss_ms, **avg},
            "speedup": miss_ms / hit_ms if hit_ms and miss_ms else None,
            # Cada hit es una llamada al LLM que no se hizo.
            "llm_calls_saved": hits,
            "tokens_spent": int(raw.get("tokens_spent", 0)),
            "tokens_saved": int(raw.get("tokens_saved", 0)),
            "cost_spent_usd": raw.get("cost_spent_usd", 0.0),
            "cost_saved_usd": raw.get("cost_saved_usd", 0.0),
        }

    async def reset(self) -> None:
        await self.redis.delete(self.key)


def _text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value
