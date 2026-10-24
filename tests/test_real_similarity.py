"""Tests con el modelo de embeddings real: verifican qué pares detecta como similares (≥ 0.95).

Cargan el modelo una vez por módulo (~2 s). Los de integración necesitan Redis en localhost:6379.
"""

import pytest

from app.config import Settings
from app.embeddings import FastEmbedEmbedder
from app.middleware import normalize_query

THRESHOLD = 0.95
MODEL = Settings(_env_file=None).embedding_model

# Misma intención, distinta forma de escribirla: deben dar HIT.
SAME = [
    ("¿Cuál es la capital de Francia?", "dime la capital de francia"),
    ("¿Cuál es la capital de Francia?", "Dime la capital de Francia"),
    ("¿Cuál es la capital de Francia?", "cual es la capital de francia"),
    ("¿Cuál es la capital de Francia?", "¿cuál es la capital de Francia"),
    ("¿Cuál es la capital de Francia?", "¿Qué ciudad es la capital de Francia?"),
    ("¿Cuál es la capital de Francia?", "CUAL ES LA CAPITAL DE FRANCIA"),
    ("¿Cuántos habitantes tiene España?", "cuantos habitantes tiene espana"),
    ("¿Qué hora es en Tokio?", "que hora es en tokio"),
]

# Paráfrasis con otras palabras: el modelo pequeño se queda por debajo de 0.95.
PARAPHRASE_BELOW_THRESHOLD = [
    ("¿Cómo se prepara una paella valenciana?", "como preparar una paella valenciana"),
    ("¿Cuántos habitantes tiene España?", "¿cuánta población tiene España?"),
]

# Parecidas en la forma pero con distinta respuesta: NUNCA deben dar HIT.
DIFFERENT = [
    ("¿Cuál es la capital de Francia?", "¿Cuál es la capital de Suecia?"),
    ("dime la capital de francia", "dime la capital de suecia"),
    ("¿Cuál es la capital de Francia?", "¿Cuál es la capital de Alemania?"),
    ("¿Cuántos habitantes tiene España?", "¿Cuántos habitantes tiene Portugal?"),
    ("¿Cómo se prepara una paella valenciana?", "¿Cómo se prepara una tortilla española?"),
    ("¿Qué hora es en Tokio?", "¿Qué hora es en Madrid?"),
    ("¿Cuál es la capital de Francia?", "¿Cómo se prepara una paella valenciana?"),
    ("Traduce 'hola' al inglés", "Traduce 'adiós' al inglés"),  # 0.935: el caso más justo
]


@pytest.fixture(scope="module")
def embedder():
    return FastEmbedEmbedder(MODEL)


def similarity(embedder, a: str, b: str) -> float:
    return float(embedder.embed(normalize_query(a)) @ embedder.embed(normalize_query(b)))


def _id(pair):
    return f"{pair[0]} | {pair[1]}"


# Distintas solo por un número o ID: el modelo las ve casi iguales, así que el middleware
# exige que los tokens con dígitos coincidan antes de aceptar un HIT semántico.
DIFFERENT_BY_NUMBER = [
    ("Estado del pedido 10023", "Estado del pedido 10024"),
    ("¿Cuánto es 15 por 3?", "¿Cuánto es 16 por 3?"),
    ("¿Cuál es la raíz cuadrada de 1234?", "¿Cuál es la raíz cuadrada de 5678?"),
    ("¿Qué pasó en 1945?", "¿Qué pasó en 1989?"),
]


# --- normalización -------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("¿Cuál es la capital de Francia?", "cual es la capital de francia"),
        ("  Dime   la CAPITAL de Francia!! ", "dime la capital de francia"),
        ("¿Cuántos habitantes tiene España?", "cuantos habitantes tiene espana"),
        ("¡Hola, mundo!", "hola mundo"),
        ("?¿!", ""),
    ],
)
def test_normalize_query(raw, expected):
    assert normalize_query(raw) == expected


# --- similitud con el modelo real -----------------------------------------------------


@pytest.mark.parametrize("a, b", SAME, ids=[_id(p) for p in SAME])
def test_same_intent_is_above_threshold(embedder, a, b):
    sim = similarity(embedder, a, b)
    assert sim >= THRESHOLD, f"similitud {sim:.4f} < {THRESHOLD}"


@pytest.mark.parametrize("a, b", DIFFERENT, ids=[_id(p) for p in DIFFERENT])
def test_different_questions_stay_below_threshold(embedder, a, b):
    sim = similarity(embedder, a, b)
    assert sim < THRESHOLD, f"falso positivo: similitud {sim:.4f} >= {THRESHOLD}"


@pytest.mark.xfail(strict=True, reason="límite conocido del modelo MiniLM con umbral 0.95")
@pytest.mark.parametrize(
    "a, b", PARAPHRASE_BELOW_THRESHOLD, ids=[_id(p) for p in PARAPHRASE_BELOW_THRESHOLD]
)
def test_reworded_paraphrase_known_limit(embedder, a, b):
    assert similarity(embedder, a, b) >= THRESHOLD


def test_normalization_improves_case_variants(embedder):
    a, b = "¿Cuál es la capital de Francia?", "dime la capital de francia"
    raw = float(embedder.embed(a) @ embedder.embed(b))
    assert raw < THRESHOLD <= similarity(embedder, a, b)


# --- integración: middleware + Redis + modelo real -------------------------------------


@pytest.fixture
async def client(make_client, embedder):
    async with make_client(embedder, similarity_threshold=THRESHOLD) as c:
        yield c


@pytest.mark.parametrize("a, b", SAME, ids=[_id(p) for p in SAME])
async def test_end_to_end_hit(client, a, b):
    r1 = await client.post("/ask", json={"query": a})
    assert r1.headers["x-cache"] == "MISS"
    r2 = await client.post("/ask", json={"query": b})
    assert r2.headers["x-cache"] == "HIT", "la segunda pregunta debería salir de caché"
    assert r2.json() == r1.json()
    assert client.llm.calls == 1


@pytest.mark.parametrize("a, b", DIFFERENT, ids=[_id(p) for p in DIFFERENT])
async def test_end_to_end_no_false_hit(client, a, b):
    await client.post("/ask", json={"query": a})
    r = await client.post("/ask", json={"query": b})
    assert r.headers["x-cache"] == "MISS"
    assert client.llm.calls == 2


def test_model_alone_confuses_order_ids(embedder):
    """Documenta por qué existe el guard: sin él, esto sería un HIT con la respuesta ajena."""
    assert similarity(embedder, "Estado del pedido 10023", "Estado del pedido 10024") >= THRESHOLD


@pytest.mark.parametrize("a, b", DIFFERENT_BY_NUMBER, ids=[_id(p) for p in DIFFERENT_BY_NUMBER])
async def test_end_to_end_numbers_never_hit(client, a, b):
    await client.post("/ask", json={"query": a})
    r = await client.post("/ask", json={"query": b})
    assert r.headers["x-cache"] == "MISS"
    assert client.llm.calls == 2
