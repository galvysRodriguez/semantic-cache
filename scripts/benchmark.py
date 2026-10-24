"""Mide latencia MISS vs HIT contra el servidor en marcha.

Uso: python scripts/benchmark.py [http://localhost:8000]
"""
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

QUERIES = [
    "¿Cuál es la capital de Francia?",
    "¿cuál es la capital de Francia",       # casi idéntica
    "Cuál es la capital de Francia",        # paráfrasis mínima
    "Dime la capital de Francia",           # paráfrasis
    "¿Cómo se prepara una paella valenciana?",
]


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=30) as c:
        c.delete("/cache")
        print(f"{'consulta':45} {'cache':5} {'sim':>7} {'ms cliente':>11} {'ms middleware':>14}")
        for q in QUERIES:
            t0 = time.perf_counter()
            r = c.post("/ask", json={"query": q})
            ms = (time.perf_counter() - t0) * 1000
            h = r.headers
            print(f"{q:45} {h.get('x-cache', '-'):5} {h.get('x-cache-similarity', '-'):>7} "
                  f"{ms:11.1f} {h.get('x-cache-latency-ms', '-'):>14}")


if __name__ == "__main__":
    main()
