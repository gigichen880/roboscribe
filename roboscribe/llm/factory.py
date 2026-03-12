"""LLM backend factory."""

from __future__ import annotations

from roboscribe.config import Config
from roboscribe.exceptions import ConfigError
from roboscribe.llm.base import LLMBackend


def create_backend(config: Config) -> LLMBackend:
    """Create an LLM backend from config."""
    if config.llm_backend == "anthropic":
        from roboscribe.llm.anthropic_backend import AnthropicBackend
        return AnthropicBackend(api_key=config.api_key, model=config.llm_model)
    elif config.llm_backend in ("openai", "qwen", "qwen-beijing", "qwen-singapore", "deepseek"):
        # All OpenAI-compatible providers use the same backend
        from roboscribe.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(
            api_key=config.api_key,
            model=config.llm_model,
            base_url=config.base_url,
        )
    else:
        raise ConfigError(f"Unknown LLM backend: {config.llm_backend}")
