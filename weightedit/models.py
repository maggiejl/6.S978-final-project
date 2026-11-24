"""Models under test. Short name -> OpenRouter model id and request settings.

Check ids at https://openrouter.ai/models before running; fill in the TODOs.
"""

MODELS: dict[str, dict] = {
    "gpt-3.5": {"id": "openai/gpt-3.5-turbo", "max_tokens": 4096},
    "gpt-4": {"id": "openai/gpt-4", "max_tokens": 4096},
    "gpt-4o": {"id": "openai/gpt-4o", "max_tokens": 16384},
    "o3": {"id": "openai/o3", "max_tokens": 32000},
    "sol": {"id": "TODO", "max_tokens": 32000},
    "astra": {"id": "TODO", "max_tokens": 32000},
    # Offline stand-in for smoke tests (--dry-run): returns a noisy magnitude-pruned student.
    "mock": {"id": "mock", "max_tokens": 0},
}

# Order used for plots (weakest -> strongest).
MODEL_ORDER = ["gpt-3.5", "gpt-4", "gpt-4o", "o3", "sol", "astra"]
