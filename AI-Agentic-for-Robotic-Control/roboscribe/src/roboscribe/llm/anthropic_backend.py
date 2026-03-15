"""Anthropic Claude LLM backend with tool-use support."""

from __future__ import annotations

import json

from roboscribe.exceptions import LLMError
from roboscribe.llm.base import LLMBackend, LLMResponse, LLMToolResponse, ToolCall


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert our tool definitions to Anthropic tool_use format."""
    anthropic_tools = []
    for tool in tools:
        anthropic_tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        })
    return anthropic_tools


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert our message format to Anthropic messages format.

    Anthropic requires strict user/assistant alternation.
    Tool results must be sent as user messages with tool_result content blocks.
    """
    anthropic_msgs = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "user":
            anthropic_msgs.append({"role": "user", "content": msg["content"]})

        elif role == "assistant":
            content_blocks = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
            if content_blocks:
                anthropic_msgs.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            # Anthropic expects tool results as user messages
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg["content"],
            }
            if msg.get("is_error"):
                tool_result_block["is_error"] = True

            # Merge into previous user message if it exists, else create one
            if anthropic_msgs and anthropic_msgs[-1]["role"] == "user":
                prev = anthropic_msgs[-1]
                if isinstance(prev["content"], str):
                    prev["content"] = [
                        {"type": "text", "text": prev["content"]},
                        tool_result_block,
                    ]
                elif isinstance(prev["content"], list):
                    prev["content"].append(tool_result_block)
            else:
                anthropic_msgs.append({
                    "role": "user",
                    "content": [tool_result_block],
                })

    return anthropic_msgs


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

    def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> LLMToolResponse:
        anthropic_messages = _to_anthropic_messages(messages)
        anthropic_tools = _to_anthropic_tools(tools)

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=anthropic_messages,
                tools=anthropic_tools,
                temperature=0.2,
            )
        except Exception as e:
            raise LLMError(f"Anthropic API error: {e}") from e

        tokens = resp.usage.input_tokens + resp.usage.output_tokens

        # Extract text content and tool calls from response blocks
        content_parts = []
        tool_calls = []

        for block in resp.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        return LLMToolResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            raw_response=resp,
            model=self.model,
            tokens_used=tokens,
        )
