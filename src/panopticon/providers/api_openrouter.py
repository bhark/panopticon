"""OpenRouter over its OpenAI-compatible chat completions endpoint.

One system message and one user message, never a growing array: the harness owns the
transcript and renders it into that single user message, so the prefix it cached last
turn is the prefix it sends this turn.

`strict: true` puts the schema through OpenAI strict mode, which cannot express a
free-form `args` object, so args travels as a JSON string. See `strict_action_schema`.
"""

from __future__ import annotations

import asyncio
import os

import httpx

from panopticon.model import Usage
from panopticon.providers.base import TurnRequest, TurnResponse, parse_action, strict_action_schema

RETRY_STATUS = (408, 429, 500, 502, 503, 504)


class OpenRouter:
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        context_window: int = 128_000,
        timeout: float = 180.0,
        attempts: int = 3,
        backoff: float = 1.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.context_window = context_window
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.attempts = attempts
        self.backoff = backoff
        self._client = client

    async def act(self, req: TurnRequest) -> TurnResponse:
        schema = strict_action_schema(req.tools)
        body = _body(self.model, req.system, req.prompt)
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "action", "strict": True, "schema": schema},
        }
        data, error = await self._post(body)
        if data is None:
            return TurnResponse(error=error)
        content = _content(data)
        usage = _usage(data)
        if not content:
            return TurnResponse(error="openrouter returned no content", usage=usage)
        action, parse_error = parse_action(content, req.tools)
        return TurnResponse(action=action, usage=usage, error=parse_error)

    async def summarize(self, system: str, text: str) -> str | None:
        data, _ = await self._post(_body(self.model, system, text))
        return _content(data) or None if data is not None else None

    async def _post(self, body: dict) -> tuple[dict | None, str | None]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return None, f"{self.api_key_env} is not set"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"

        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            error = "openrouter made no attempt"
            for attempt in range(self.attempts):
                try:
                    response = await client.post(url, json=body, headers=headers)
                except httpx.HTTPError as exc:
                    error = f"openrouter request failed: {exc}"
                else:
                    if response.status_code < 400:
                        try:
                            return response.json(), None
                        except ValueError as exc:
                            return None, f"openrouter sent invalid JSON: {exc}"
                    error = f"openrouter {response.status_code}: {response.text[:200]}"
                    if response.status_code not in RETRY_STATUS:
                        return None, error
                if attempt < self.attempts - 1:
                    await asyncio.sleep(self.backoff * 2**attempt)
            return None, error
        finally:
            if self._client is None:
                await client.aclose()


def _body(model: str, system: str, prompt: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "usage": {"include": True},
    }


def _content(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "")


def _usage(data: dict) -> Usage:
    u = data.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    got = Usage(
        input_tokens=u.get("prompt_tokens", 0),
        output_tokens=u.get("completion_tokens", 0),
        cache_read=cached,
        cost_usd=float(u.get("cost", 0.0) or 0.0),
    )
    got.context_tokens = u.get("total_tokens", 0) or got.input_tokens + got.output_tokens
    return got
