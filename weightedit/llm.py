"""OpenRouter chat client (stdlib only) and an offline mock client."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request

import numpy as np

BASE_URL = "https://openrouter.ai/api/v1"


class LLMError(Exception):
    pass


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL, timeout: int = 900):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise LLMError("Set OPENROUTER_API_KEY (or put it in a .env file).")
        self.base_url = base_url
        self.timeout = timeout

    def chat(self, model_id: str, messages: list[dict], max_tokens: int, seed: int | None = None,
             context: dict | None = None, retries: int = 6) -> dict:
        body = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        last = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                if "error" in resp:
                    raise LLMError(json.dumps(resp["error"]))
                choice = resp["choices"][0]
                return {
                    "content": choice["message"].get("content") or "",
                    "reasoning": choice["message"].get("reasoning"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": resp.get("usage", {}),
                    "provider": resp.get("provider"),
                    "model": resp.get("model"),
                }
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:2000]
                last = LLMError(f"HTTP {e.code}: {detail}")
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise last  # e.g. 400 context too long: not retryable
            except (urllib.error.URLError, TimeoutError, LLMError, KeyError) as e:
                last = LLMError(str(e))
            time.sleep(min(60, 2**attempt) + random.random())
        raise last


class MockClient:
    """Pretends to be an LLM: answers with a magnitude-pruned student plus noise, for pipeline tests."""

    def chat(self, model_id: str, messages: list[dict], max_tokens: int, seed: int | None = None,
             context: dict | None = None, **_) -> dict:
        from .baselines import magnitude_prune

        rng = np.random.default_rng((seed or 0) * 100 + len(messages))
        student = magnitude_prune(context["teacher"])
        student = {k: v + rng.normal(0, 0.05, v.shape) for k, v in student.items()}
        text = "Here is my student.\n```json\n" + json.dumps({k: np.round(v, 3).tolist() for k, v in student.items()}) + "\n```"
        return {"content": text, "reasoning": None, "finish_reason": "stop",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
                "provider": "mock", "model": "mock"}


def make_client(model_name: str):
    return MockClient() if model_name == "mock" else OpenRouterClient()
