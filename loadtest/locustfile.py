"""Carga realista: preguntas repetidas, variantes de escritura, paráfrasis y preguntas nuevas.

Uso:
  locust -f loadtest/locustfile.py --host http://localhost:8000 \
         --headless -u 50 -r 10 -t 60s --csv results/run
"""

import random
import uuid

from locust import FastHttpUser, between, task

# Grupos de preguntas equivalentes: la primera "siembra" la caché, el resto deberían ser HIT.
GROUPS = [
    [
        "¿Cuál es la capital de Francia?",
        "dime la capital de francia",
        "¿Qué ciudad es la capital de Francia?",
        "cual es la capital de francia",
    ],
    [
        "¿Cuántos habitantes tiene España?",
        "cuantos habitantes tiene espana",
        "¿Cuántos habitantes tiene España",
    ],
    ["¿Qué hora es en Tokio?", "que hora es en tokio", "QUÉ HORA ES EN TOKIO"],
    ["¿Cómo se prepara una paella valenciana?", "como se prepara una paella valenciana"],
    [
        "¿Quién escribió Cien años de soledad?",
        "quien escribio cien anos de soledad",
        "¿Quién es el autor de Cien años de soledad?",
    ],
    ["¿Qué es la fotosíntesis?", "que es la fotosintesis", "Explica qué es la fotosíntesis"],
]


class CacheUser(FastHttpUser):
    wait_time = between(0.1, 0.5)

    @task(9)
    def known_question(self):
        query = random.choice(random.choice(GROUPS))
        self.client.post("/ask", json={"query": query}, name="/ask [conocida]")

    @task(1)
    def new_question(self):
        # Pregunta única: siempre MISS, representa el tráfico que la caché no puede absorber.
        query = f"Pregunta única {uuid.uuid4().hex[:8]} sobre historia"
        self.client.post("/ask", json={"query": query}, name="/ask [nueva]")
