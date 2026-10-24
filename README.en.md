# Semantic cache for LLMs · FastAPI + Redis

[Español](README.md) · **English**

[![CI](https://github.com/galvysRodriguez/semantic-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/galvysRodriguez/semantic-cache/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12+-3776AB)
![Redis](https://img.shields.io/badge/Redis_8-vector_search-DC382D)
![LLM](https://img.shields.io/badge/LLM-Groq_%7C_Grok_%7C_Gemini-F55036)
![License](https://img.shields.io/badge/license-MIT-blue)

FastAPI middleware that intercepts questions sent to an LLM (Groq by default; xAI Grok or Gemini
also supported). If a **semantically equivalent** question has already been answered, it returns
the stored answer from Redis in milliseconds instead of paying for another call.

![Demo](docs/demo.gif)

The demo UI is in Spanish. Repeated and reworded questions come back as **HIT** in milliseconds.

| Under load (50 users, 60 s) | no cache | with cache |
|---|---:|---:|
| Throughput | 26 req/s | **88 req/s** |
| p50 latency | 1500 ms | **45 ms** |
| p50 latency, repeated or reworded questions | 1500 ms | **37 ms** |
| Requests that reach the LLM | 100 % | **10 %** |

With real Groq (`openai/gpt-oss-20b`, one of the fastest LLMs), a MISS takes ~500 ms and a HIT
15-40 ms: **15 to 30 times faster**. The load test uses a simulated 1.5 s backend, closer to a large
LLM, so it doesn't spend API quota.

## How it works

```
POST /ask {"query": "dime la capital de francia"}
        │
        ▼
SemanticCacheMiddleware (pure ASGI)
  1. normalize: lowercase, strip accents and punctuation
  2. exact same query? ───────── HMGET on a sha256 key ───────────────► HIT  (~1 ms)
  3. local embedding (multilingual MiniLM, ONNX, 384 dims)
  4. KNN 3 on a Redis HNSW COSINE index
       └─ similarity ≥ 0.95 and same numbers/IDs ────────────────────► HIT  (~15-30 ms)
  5. same query already in flight? ── wait for its answer ───────────► HIT  (single-flight)
  6. otherwise: call the LLM, store the 2xx response with a TTL ──────► MISS (~0.5-3 s)
```

Every response says what happened: `X-Cache: HIT|MISS`, `X-Cache-Similarity`,
`X-Cache-Matched-Query`, `X-Cache-Latency-Ms` and, on a HIT, `X-Cache-Saved-Tokens` /
`X-Cache-Saved-Usd`.

## Getting started

```bash
cp .env.example .env            # add GROQ_API_KEY (without a key a simulated LLM is used)
docker compose up --build       # or: podman compose up --build
```

- Demo: http://localhost:8000
- API (Swagger): http://localhost:8000/docs
- Metrics: http://localhost:8000/stats (JSON) and http://localhost:8000/metrics (Prometheus)

The image ships with the embedding model, so it starts in seconds and only needs network access to
call the LLM.

## Design decisions and lessons learned

Things you won't find in a tutorial, found by measuring:

**1. The model is sensitive to case and accents.** "¿Cuál es la capital de Francia?" vs
"dime la capital de francia" ("what is the capital of France?" vs "tell me the capital of france")
scored **0.876**, below the threshold, so the user never got the HIT. Normalizing the text before
embedding raises it to **0.968**. Variants that only differ in spelling become the same key and are
answered without computing an embedding.
→ `tests/test_real_similarity.py`

**2. The model barely tells numbers apart: a dangerous false positive.** Found during the load
test. "Estado del pedido 10023" vs "Estado del pedido 10024" (order status) scores **0.964**, so
the cache would have given one customer another customer's order status. A semantic HIT now
requires tokens containing digits (IDs, amounts, years) to match exactly. The search returns the 3
nearest neighbours so a rejected one doesn't hide a valid one.
→ `test_different_ids_never_share_an_answer`, `test_end_to_end_numbers_never_hit`

**3. Why the threshold is 0.95 and not lower.** Questions on different topics stay far apart
(≤ 0.66 after normalization). But questions that differ by a single word get very close: "translate
'hola' into English" vs "translate 'adiós' into English" scores **0.935**. That's why I didn't lower
the threshold, even though it misses rewordings that use different words (0.78 for "how big is
Spain's population?" vs "how many inhabitants…?"). It's deliberate: an extra MISS costs one LLM
call, while an extra HIT returns a wrong answer. The tests mark these cases as known limits
(`xfail`).

**4. Concurrent requests for the same thing (cache stampede).** If 10 users ask the same new
question at once, an unprotected cache makes 10 LLM calls. A `SET NX PX` lock in Redis lets one
request through and the rest wait for its answer. If that request fails, the waiting ones call the
LLM themselves instead of hanging.

The first version had a race: a request could miss the cache and then, right after, take the lock
the previous request had just released after storing its answer, calling the LLM a second time.
The fix is a double check: after taking the lock, look in the cache again.
→ `test_concurrent_identical_requests_call_backend_once`,
`test_leader_finishing_before_lock_is_not_called_twice`

**5. The cost of the cache: CPU-bound embeddings.** Under load, new questions are somewhat slower
with the cache (p50 1.7 s vs 1.5 s) because each one pays for an embedding and competes for CPU. If
traffic had few repeats, the cache wouldn't pay off. The next steps for that case would be a GPU
embedding model or a separate embedding service.

**Other decisions:**
- **Pure ASGI**, not `BaseHTTPMiddleware`: reads and replays the body without overhead and captures
  the response to store it.
- **Per-tenant isolation** (`X-Tenant-ID`) and per-endpoint isolation, so one client's answers are
  never served to another.
- **Fault tolerant**: if Redis goes down, the middleware passes requests straight to the LLM.
- **Only 2xx responses are cached**, with a TTL configurable per route.
- **Metrics stored in Redis**, shared across replicas, plus per-process Prometheus metrics.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/ask` | `{"query": "..."}` → LLM answer, cached |
| `GET` | `/stats` | hit rate, latencies, tokens and USD saved |
| `DELETE` | `/stats` | resets the metrics |
| `DELETE` | `/cache?path=/ask` | invalidates one route for the request's tenant (no `path`: everything) |
| `GET` | `/metrics` | Prometheus format |
| `GET` | `/health` | Redis status and active LLM backend |

## Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `GROQ_API_KEY` | – | Groq (`gsk_…` keys); the first provider with a key is used |
| `SEMCACHE_GROQ_MODEL` | `openai/gpt-oss-20b` | |
| `XAI_API_KEY` | – | xAI Grok (`xai-…` keys) |
| `SEMCACHE_XAI_MODEL` | `grok-4.3` | |
| `GEMINI_API_KEY` | – | without any key the simulated backend is used |
| `SEMCACHE_GEMINI_MODEL` | `gemini-2.5-flash` | |
| `SEMCACHE_SIMILARITY_THRESHOLD` | `0.95` | |
| `SEMCACHE_TTL_SECONDS` | `3600` | 0 = never expire |
| `SEMCACHE_ROUTE_TTLS` | `{}` | e.g. `{"/news": 300}` |
| `SEMCACHE_CACHED_PATHS` | `["/ask"]` | intercepted routes |
| `SEMCACHE_TENANT_HEADER` | `x-tenant-id` | |
| `SEMCACHE_ENABLED` | `true` | `false` to measure the baseline |
| `SEMCACHE_PRICE_INPUT_PER_MTOK` / `_OUTPUT_` | default model's pricing | USD per million tokens, used to estimate savings; set it if you change models |

Only Groq has been tested against its real API. Grok and Gemini are implemented from their
documentation and covered by offline tests, but have not been run with a real key.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
docker compose up -d redis

.venv/bin/pytest                    # 74 tests: middleware, LLM, real model and Redis integration
.venv/bin/ruff check . && .venv/bin/mypy
.venv/bin/uvicorn app.main:create_app --factory --reload
```

Load test (results are in `loadtest/results/`):

```bash
.venv/bin/locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 60s
```

The simulated traffic is 90 % known questions (with spelling variants and rewordings) and 10 % new
questions. The simulated backend takes 1.5 s, so measuring doesn't spend LLM quota.

To re-record the GIF and screenshot in `docs/` (with the app running):

```bash
pip install playwright pillow && playwright install chromium
python scripts/record_demo.py
```

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `unknown command 'FT.CREATE'` or `FT._LIST` | Your Redis has no vector search. Use Redis 8 or Redis Stack, e.g. `docker compose up -d redis`; Valkey or a plain `redis-server` won't work. |
| `Cannot create index on db != 0` | Indexes only work on database 0: use `redis://…:6379/0`. |
| Groq returns `403 Access denied. Please check your network settings` | Groq doesn't serve your region. It works from the host through a VPN. |
| …only inside the container, with a VPN on | Rootless podman networking goes out through the physical interface and bypasses the VPN. Create a local `compose.override.yaml` with `network_mode: host` for the `app` service and `SEMCACHE_REDIS_URL=redis://localhost:6379/0`. |
| xAI returns `Incorrect API key provided` for a `gsk_…` key | That's a **Groq** key (groq.com), not a **Grok** key (x.ai): put it in `GROQ_API_KEY`. |
| `/health` shows `"llm": "simulated"` | There's no key in `.env`, or the app wasn't restarted after adding it: `docker compose up -d --force-recreate app`. |

## Project layout

```
app/
  middleware.py   ASGI interception, normalization, single-flight
  cache.py        Redis vector index, KNN search, entity guard, invalidation
  embeddings.py   local embeddings (fastembed / ONNX)
  llm.py          Groq, Grok (xAI), Gemini and simulated backends
  stats.py        Redis and Prometheus metrics
  main.py         FastAPI app
  static/         demo page
tests/            unit (fake embeddings) and integration (real model and Redis)
loadtest/         Locust scenario and results
scripts/          quick benchmark and GIF recording
```

## License

[MIT](LICENSE)
