"""Backends LLM: forma de la petición a xAI y lectura de la respuesta, sin red."""

import json

import httpx
import pytest

from app.config import Settings
from app.llm import GeminiBackend, GrokBackend, SimulatedBackend, build_backend


def grok_with(handler) -> GrokBackend:
    settings = Settings(xai_api_key="xai-test", gemini_api_key=None)
    return GrokBackend(settings, transport=httpx.MockTransport(handler))


def responses_payload(text: str, usage: dict) -> dict:
    return {
        "model": "grok-4.3",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
        "usage": usage,
    }


async def test_grok_request_shape_and_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        usage = {"input_tokens": 20, "output_tokens": 30, "total_tokens": 50}
        return httpx.Response(200, json=responses_payload("París.", usage))

    result = await grok_with(handler).generate("¿Capital de Francia?")

    assert seen["url"] == "https://api.x.ai/v1/responses"
    assert seen["auth"] == "Bearer xai-test"
    assert seen["body"]["model"] == "grok-4.3"
    assert seen["body"]["input"][-1] == {"role": "user", "content": "¿Capital de Francia?"}
    assert seen["body"]["input"][0]["role"] == "system"
    assert result.answer == "París."
    assert (result.input_tokens, result.output_tokens) == (20, 30)
    assert result.cost_usd == pytest.approx((20 * 1.25 + 30 * 2.50) / 1_000_000)


async def test_grok_counts_reasoning_tokens_reported_only_in_total():
    def handler(request):
        usage = {
            "input_tokens": 32,
            "output_tokens": 9,
            "total_tokens": 151,
            "output_tokens_details": {"reasoning_tokens": 110},
        }
        return httpx.Response(200, json=responses_payload("ok", usage))

    result = await grok_with(handler).generate("x")
    assert result.output_tokens == 119


async def test_grok_http_errors_propagate():
    backend = grok_with(lambda request: httpx.Response(401, json={"error": "bad key"}))
    with pytest.raises(httpx.HTTPStatusError):
        await backend.generate("x")


@pytest.mark.parametrize(
    "keys, expected",
    [
        ({"xai_api_key": "x", "gemini_api_key": "g"}, GrokBackend),
        ({"xai_api_key": None, "gemini_api_key": "g"}, GeminiBackend),
        ({"xai_api_key": None, "gemini_api_key": None}, SimulatedBackend),
    ],
)
def test_backend_selection(keys, expected):
    assert isinstance(build_backend(Settings(**keys)), expected)
