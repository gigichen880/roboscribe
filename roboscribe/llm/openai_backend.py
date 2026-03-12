"""OpenAI LLM backend."""

from __future__ import annotations

from roboscribe.exceptions import LLMError
from roboscribe.llm.base import LLMBackend, LLMResponse


class OpenAIBackend(LLMBackend):
    """GPT-4o / OpenAI-compatible backend."""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = ""):
        try:
            from openai import OpenAI
        except ImportError:
            raise LLMError("openai package not installed. Run: pip install openai")

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
        except Exception as e:
            raise LLMError(f"OpenAI API error: {e}") from e

        choice = resp.choices[0]
        tokens = resp.usage.total_tokens if resp.usage else 0

        return LLMResponse(
            raw_text=choice.message.content or "",
            model=self.model,
            tokens_used=tokens,
        )
