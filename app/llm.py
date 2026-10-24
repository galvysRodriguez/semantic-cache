import asyncio
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings


@dataclass
class LLMResult:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LLMBackend(Protocol):
    name: str

    async def generate(self, prompt: str) -> LLMResult: ...


# USD por millón de tokens (entrada, salida) del modelo por defecto de cada proveedor, oct. 2026.
# SEMCACHE_PRICE_INPUT_PER_MTOK / _OUTPUT_ los sustituyen si usas otro modelo.
GROQ_PRICES = (0.075, 0.30)  # openai/gpt-oss-20b
GROK_PRICES = (1.25, 2.50)  # grok-4.3
GEMINI_PRICES = (0.30, 2.50)  # gemini-2.5-flash


def _cost(
    settings: Settings, defaults: tuple[float, float], input_tokens: int, output_tokens: int
) -> float:
    price_in = settings.price_input_per_mtok
    price_out = settings.price_output_per_mtok
    price_in = defaults[0] if price_in is None else price_in
    price_out = defaults[1] if price_out is None else price_out
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def _http_client(
    base_url: str, api_key: str | None, transport: httpx.AsyncBaseTransport | None
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(60.0, connect=5.0),
        transport=transport,
    )


class GroqBackend:
    """Groq (API compatible con OpenAI: POST /chat/completions) vía httpx."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._client = _http_client(settings.groq_base_url, settings.groq_api_key, transport)
        self._settings = settings
        self.name = settings.groq_model

    async def generate(self, prompt: str) -> LLMResult:
        resp = await self._client.post(
            "/chat/completions",
            json={
                "model": self.name,
                "messages": [
                    {"role": "system", "content": self._settings.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        # En el formato OpenAI, completion_tokens ya incluye los tokens de razonamiento.
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        return LLMResult(
            answer=answer,
            model=data.get("model", self.name),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost(self._settings, GROQ_PRICES, input_tokens, output_tokens),
        )


class GrokBackend:
    """xAI Responses API (POST /v1/responses) vía httpx, sin SDK adicional."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._client = _http_client(settings.xai_base_url, settings.xai_api_key, transport)
        self._settings = settings
        self.name = settings.xai_model

    async def generate(self, prompt: str) -> LLMResult:
        resp = await self._client.post(
            "/responses",
            json={
                "model": self.name,
                "input": [
                    {"role": "system", "content": self._settings.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "store": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        answer = "".join(
            part.get("text", "")
            for item in data.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        usage = data.get("usage") or {}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        # Los tokens de razonamiento se facturan como salida; según la respuesta pueden venir
        # dentro de output_tokens o solo en total_tokens, así que se toma el mayor.
        output_tokens = max(output_tokens, usage.get("total_tokens", 0) - input_tokens)
        return LLMResult(
            answer=answer,
            model=data.get("model", self.name),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost(self._settings, GROK_PRICES, input_tokens, output_tokens),
        )


class GeminiBackend:
    def __init__(self, settings: Settings):
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._config = types.GenerateContentConfig(system_instruction=settings.system_prompt)
        self._settings = settings
        self.name = settings.gemini_model

    async def generate(self, prompt: str) -> LLMResult:
        resp = await self._client.aio.models.generate_content(
            model=self.name, contents=prompt, config=self._config
        )
        usage = resp.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        # Los tokens de razonamiento ("thinking") también se facturan como salida.
        output_tokens = (
            (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0) if usage else 0
        )
        return LLMResult(
            answer=resp.text or "",
            model=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost(self._settings, GEMINI_PRICES, input_tokens, output_tokens),
        )


class SimulatedBackend:
    """Sustituto sin red: misma interfaz y latencia configurable."""

    name = "simulated"

    def __init__(self, settings: Settings):
        self._settings = settings

    async def generate(self, prompt: str) -> LLMResult:
        await asyncio.sleep(self._settings.simulated_latency_s)
        answer = (
            f"Respuesta simulada para: {prompt!r}. Configura GROQ_API_KEY para usar un LLM real."
        )
        # ~4 caracteres por token: suficiente para que las métricas de ahorro tengan sentido.
        input_tokens, output_tokens = len(prompt) // 4 + 1, len(answer) // 4 + 1
        return LLMResult(
            answer,
            self.name,
            input_tokens,
            output_tokens,
            _cost(self._settings, GROQ_PRICES, input_tokens, output_tokens),
        )


def build_backend(settings: Settings) -> LLMBackend:
    """Usa el primer proveedor con clave configurada: Groq, Grok (xAI), Gemini."""
    if settings.groq_api_key:
        return GroqBackend(settings)
    if settings.xai_api_key:
        return GrokBackend(settings)
    if settings.gemini_api_key:
        return GeminiBackend(settings)
    return SimulatedBackend(settings)
