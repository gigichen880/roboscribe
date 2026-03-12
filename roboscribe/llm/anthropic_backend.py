"""Anthropic Claude LLM backend."""

from __future__ import annotations

from roboscribe.exceptions import LLMError
from roboscribe.llm.base import LLMBackend, LLMResponse


class AnthropicBackend(LLMBackend):
    """Claude / Anthropic backend."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise LLMError("anthropic package not installed. Run: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.2,
            )
        except Exception as e:
            raise LLMError(f"Anthropic API error: {e}") from e

        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text

        tokens = resp.usage.input_tokens + resp.usage.output_tokens

        return LLMResponse(
            raw_text=text,
            model=self.model,
            tokens_used=tokens,
        )
