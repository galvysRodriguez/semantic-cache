import asyncio
from dataclasses import dataclass
from typing import Protocol

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


class SimulatedBackend:
    """Sustituto sin red: misma interfaz y latencia configurable."""

    name = "simulated"

    def __init__(self, settings: Settings):
        self._settings = settings

    async def generate(self, prompt: str) -> LLMResult:
        await asyncio.sleep(self._settings.simulated_latency_s)
        answer = f"Respuesta simulada para: {prompt!r}. Configura GEMINI_API_KEY para usar Gemini."
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
    return GeminiBackend(settings) if settings.gemini_api_key else SimulatedBackend(settings)
