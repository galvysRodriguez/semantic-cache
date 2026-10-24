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


def _cost(settings: Settings, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * settings.price_input_per_mtok
        + output_tokens * settings.price_output_per_mtok
    ) / 1_000_000


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
            cost_usd=_cost(self._settings, input_tokens, output_tokens),
        )


class GrokBackend:
    """xAI Responses API (POST /v1/responses) vía httpx, sin SDK adicional."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._client = httpx.AsyncClient(
            base_url=settings.xai_base_url,
            headers={"Authorization": f"Bearer {settings.xai_api_key}"},
            timeout=httpx.Timeout(60.0, connect=5.0),
            transport=transport,
        )
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
            cost_usd=_cost(self._settings, input_tokens, output_tokens),
        )


class SimulatedBackend:
    """Sustituto sin red: misma interfaz y latencia configurable."""

    name = "simulated"

    def __init__(self, settings: Settings):
        self._settings = settings

    async def generate(self, prompt: str) -> LLMResult:
        await asyncio.sleep(self._settings.simulated_latency_s)
        answer = f"Respuesta simulada para: {prompt!r}. Configura XAI_API_KEY para usar Grok."
        # ~4 caracteres por token: suficiente para que las métricas de ahorro tengan sentido.
        input_tokens, output_tokens = len(prompt) // 4 + 1, len(answer) // 4 + 1
        return LLMResult(
            answer,
            self.name,
            input_tokens,
            output_tokens,
            _cost(self._settings, input_tokens, output_tokens),
        )


def build_backend(settings: Settings) -> LLMBackend:
    if settings.xai_api_key:
        return GrokBackend(settings)
    if settings.gemini_api_key:
        return GeminiBackend(settings)
    return SimulatedBackend(settings)
