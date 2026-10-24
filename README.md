# Caché semántico de alta velocidad (FastAPI + Redis)

Middleware ASGI que intercepta consultas a FastAPI y responde desde Redis si ya se respondió
una pregunta **semánticamente equivalente** (similitud de coseno ≥ 0.95), sin ejecutar el endpoint.

```
POST /ask {"query": "..."}
        │
        ▼
SemanticCacheMiddleware
  1. normaliza la consulta (minúsculas, sin tildes ni puntuación)
  2. ¿coincidencia exacta? ── HMGET por clave sha256 ──► HIT (<1 ms)
  3. embedding (fastembed/ONNX, 384 dims)
  4. KNN 1 en índice HNSW COSINE (Redis 8) ── sim ≥ 0.95 ──► HIT (~15-50 ms)
  5. si no: ejecuta el endpoint, guarda la respuesta 2xx con TTL ──► MISS
```

Cabeceras de respuesta: `X-Cache: HIT|MISS`, `X-Cache-Similarity`, `X-Cache-Latency-Ms`,
`X-Cache-Matched-Query`.

## Puesta en marcha

```bash
docker compose up -d          # o: podman run -d -p 6379:6379 docker.io/redis:8.2
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:create_app --factory --port 8000
.venv/bin/python scripts/benchmark.py
```

Hace falta Redis 8 o Redis Stack: incluye el motor de búsqueda vectorial. Un `redis-server`
o Valkey normal no lo trae. La primera vez se descarga el modelo de embeddings (~220 MB).

## Resultados medidos (CPU, sin GPU)

| consulta                                   | resultado | similitud | latencia |
|--------------------------------------------|-----------|-----------|----------|
| ¿Cuál es la capital de Francia?            | MISS      | –         | 1545 ms (backend simulado) |
| ¿cuál es la capital de Francia             | HIT       | 0.9886    | 11 ms    |
| Dime la capital de Francia                 | HIT       | 0.9598    | 23 ms    |
| ¿Cómo se prepara una paella valenciana?    | MISS      | –         | 1565 ms  |
| repetición exacta                          | HIT       | 1.0       | 0.7 ms   |

En un HIT semántico casi todo el tiempo es el embedding (~20 ms en CPU); la búsqueda en Redis
tarda ~1 ms.

## Configuración (`.env`, prefijo `SEMCACHE_`)

| variable               | por defecto                                                   |
|------------------------|---------------------------------------------------------------|
| `REDIS_URL`            | `redis://localhost:6379/0` (los índices solo funcionan en db 0) |
| `SIMILARITY_THRESHOLD` | `0.95`                                                        |
| `TTL_SECONDS`          | `3600` (0 = sin expiración)                                   |
| `EMBEDDING_MODEL`      | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `CACHED_PATHS`         | `["/ask"]`                                                    |
| `QUERY_FIELD`          | `query`                                                       |

## Integrarlo en tu app

Sustituye `expensive_backend` en `app/main.py` por tu llamada real (LLM, RAG…), o añade el
middleware a tu propia app:

```python
app.add_middleware(SemanticCacheMiddleware, cache=cache, embedder=embedder,
                   paths=["/chat"], query_field="prompt")
```

## Decisiones de diseño

- **ASGI puro**, no `BaseHTTPMiddleware`: lee y reinyecta el body sin overhead y captura la
  respuesta en streaming.
- **Separado por endpoint**: un campo TAG `scope` (hash de método + ruta) evita que `/ask` y
  `/chat` compartan respuestas.
- **Tolerante a fallos**: si Redis cae, el middleware deja pasar la petición al endpoint.
- **Normalización antes del embedding**: el modelo es sensible a mayúsculas y tildes
  ("Francia" vs "francia": 0.88 → 0.97 normalizado). Ver `tests/test_real_similarity.py`.
- **Solo se cachean las respuestas 2xx.**
- **El embedding corre en un threadpool** para no bloquear el event loop.

## Tests

```bash
.venv/bin/pytest      # requiere Redis en localhost:6379; usa un índice temporal aislado
```
