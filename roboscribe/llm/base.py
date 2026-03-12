"""Abstract LLM backend and response types."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    """Response from an LLM backend."""

    raw_text: str
    model: str
    tokens_used: int = 0

    @property
    def code(self) -> str:
        """Extract Python code from the response."""
        return extract_code(self.raw_text)

    @property
    def reasoning(self) -> str:
        """Extract reasoning text (everything outside code blocks)."""
        return re.sub(r"```python.*?```", "", self.raw_text, flags=re.DOTALL).strip()


def extract_code(text: str) -> str:
    """Extract Python code from markdown-formatted LLM output.

    Looks for ```python ... ``` blocks. If multiple blocks exist,
    joins them. If no blocks found, returns the raw text.
    """
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    # Fallback: try generic code blocks
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)
    return text.strip()


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate a response from the LLM."""
        ...
