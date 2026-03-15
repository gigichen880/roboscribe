"""OpenAI LLM backend (also covers Qwen, DeepSeek via compatible API)."""

from __future__ import annotations

import json

from roboscribe.exceptions import LLMError
from roboscribe.llm.base import LLMBackend, LLMResponse, LLMToolResponse, ToolCall


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert our tool definitions to OpenAI function calling format."""
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        })
    return openai_tools


def _to_openai_messages(
    system_prompt: str,
    messages: list[dict],
) -> list[dict]:
    """Convert our message format to OpenAI messages format."""
    openai_msgs = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        role = msg.get("role", "user")

        if role == "user":
            openai_msgs.append({"role": "user", "content": msg["content"]})

        elif role == "assistant":
            assistant_msg: dict = {"role": "assistant"}
            if msg.get("content"):
                assistant_msg["content"] = msg["content"]
            if msg.get("tool_calls"):
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["arguments"])
                                if isinstance(tc["arguments"], dict)
                                else tc["arguments"]
                            ),
                        },
                    }
                    for tc in msg["tool_calls"]
                ]
            openai_msgs.append(assistant_msg)

        elif role == "tool":
            openai_msgs.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", ""),
                "content": msg["content"],
            })

    return openai_msgs


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

    def generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> LLMToolResponse:
        openai_messages = _to_openai_messages(system_prompt, messages)
        openai_tools = _to_openai_tools(tools)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=openai_tools,
                temperature=0.2,
                max_tokens=4096,
            )
        except Exception as e:
            raise LLMError(f"OpenAI API error: {e}") from e

        choice = resp.choices[0]
        tokens = resp.usage.total_tokens if resp.usage else 0

        # Extract text content
        content = choice.message.content or ""

        # Extract tool calls
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {"_raw": tc.function.arguments}

                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        return LLMToolResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=resp,
            model=self.model,
            tokens_used=tokens,
        )
