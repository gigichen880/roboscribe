"""Configuration for RoboScribe."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from roboscribe.exceptions import ConfigError


@dataclass
class Config:
    """RoboScribe configuration."""

    # LLM settings
    llm_backend: str = "openai"
    llm_model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""  # Custom API base URL for OpenAI-compatible providers (e.g. Qwen)

    # Generation settings
    max_attempts: int = 5
    success_threshold: float = 0.8

    # Simulation settings
    robot: str = "Panda"
    controller: str = "OSC_POSE"
    num_episodes: int = 10
    max_episode_steps: int = 200
    sim_timeout: int = 120  # seconds per simulation run

    # Output settings
    output_dir: str = "."
    verbose: bool = False
    interactive: bool = False

    # Video display settings
    record_attempt_videos: bool = True

    @classmethod
    def from_env(cls, **overrides) -> Config:
        """Create config from environment variables + overrides."""
        cfg = cls(
            llm_backend=os.environ.get("ROBOSCRIBE_LLM_BACKEND", "openai"),
            llm_model=os.environ.get("ROBOSCRIBE_LLM_MODEL", "gpt-4o"),
            api_key=os.environ.get("ROBOSCRIBE_API_KEY", ""),
            base_url=os.environ.get("ROBOSCRIBE_BASE_URL", ""),
        )
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    # Known OpenAI-compatible providers: (backend_name, default_model, base_url, env_var)
    PROVIDERS = {
        "openai": ("gpt-4o", "", "OPENAI_API_KEY"),
        "anthropic": ("claude-sonnet-4-20250514", "", "ANTHROPIC_API_KEY"),
        "qwen": ("qwen-plus", "https://dashscope-us.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
        "qwen-beijing": ("qwen-plus", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
        "qwen-singapore": ("qwen-plus", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
        "deepseek": ("deepseek-chat", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    }

    def validate(self) -> None:
        """Validate configuration."""
        # Apply provider defaults
        provider = self.PROVIDERS.get(self.llm_backend)
        if provider:
            default_model, default_url, env_var = provider
            if not self.llm_model or self.llm_model == "gpt-4o":
                if self.llm_backend != "openai":
                    self.llm_model = default_model
            if not self.base_url and default_url:
                self.base_url = default_url
            if not self.api_key:
                self.api_key = os.environ.get(env_var, "")

        # Generic fallback
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if not self.api_key:
            raise ConfigError(
                "No API key found. Set ROBOSCRIBE_API_KEY or a provider-specific key "
                "(OPENAI_API_KEY, ANTHROPIC_API_KEY, DASHSCOPE_API_KEY, DEEPSEEK_API_KEY)"
            )

        valid_backends = set(self.PROVIDERS.keys())
        if self.llm_backend not in valid_backends:
            raise ConfigError(
                f"Unknown LLM backend: {self.llm_backend}. "
                f"Supported: {', '.join(sorted(valid_backends))}"
            )

        if self.max_attempts < 1:
            raise ConfigError("max_attempts must be >= 1")
