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


@dataclass
class ToolCall:
    """A single tool call from the LLM."""

    id: str
    name: str
    arguments: dict


@dataclass
class LLMToolResponse:
    """Response from an LLM backend that may include tool calls."""

    content: str  # text response (reasoning)
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: object = None  # raw API response for provider-specific handling
    model: str = ""
    tokens_used: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


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

    def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> LLMToolResponse:
        """Generate a response with tool-use support.

        Subclasses should override this. The default falls back to
        a plain generate() call with no tool support.
        """
        # Build a single user prompt from messages
        user_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                user_parts.append(content)
            elif role == "tool":
                user_parts.append(f"[Tool result for {msg.get('name', '?')}]:\n{content}")

        resp = self.generate(system_prompt, "\n\n".join(user_parts))
        return LLMToolResponse(
            content=resp.raw_text,
            model=resp.model,
            tokens_used=resp.tokens_used,
        )
