"""Configuration for RoboScribe."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from roboscribe.exceptions import ConfigError


def _load_dotenv() -> None:
    """Load key=value pairs from .env files into os.environ (if not already set).

    Searches (in order):
      1. <project-root>/roboscribe/.env   (next to pyproject.toml)
      2. <project-root>/.env              (repo root)
      3. ~/.roboscribe/.env               (user home)
    First file found wins.  Existing env vars are never overwritten.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / ".env",   # roboscribe/.env
        Path(__file__).resolve().parents[3] / ".env",   # repo root .env
        Path.home() / ".roboscribe" / ".env",           # ~/.roboscribe/.env
    ]
    for path in candidates:
        if path.is_file():
            _parse_dotenv(path)
            return


def _parse_dotenv(path: Path) -> None:
    """Parse a .env file and inject into os.environ (skip existing keys)."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load .env on import so keys are available everywhere
_load_dotenv()


# Path where the UI (and CLI) can persist settings
def get_env_file_path() -> Path:
    """Return the canonical .env path (roboscribe/.env, next to pyproject.toml)."""
    return Path(__file__).resolve().parents[2] / ".env"


def save_env_vars(values: dict[str, str]) -> Path:
    """Write/update key=value pairs in the .env file.

    Existing keys are updated in-place; new keys are appended.
    Empty values remove the key.  Returns the path written to.
    """
    path = get_env_file_path()
    existing_lines: list[str] = []
    if path.exists():
        existing_lines = path.read_text().splitlines()

    written_keys: set[str] = set()
    new_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in values:
                written_keys.add(key)
                if values[key]:  # non-empty → update
                    new_lines.append(f"{key}={values[key]}")
                # empty → remove (skip the line)
                continue
        new_lines.append(line)

    # Append new keys that weren't already in the file
    for key, value in values.items():
        if key not in written_keys and value:
            new_lines.append(f"{key}={value}")

    path.write_text("\n".join(new_lines) + "\n")

    # Also inject into current process so they take effect immediately
    for key, value in values.items():
        if value:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

    return path


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
    max_episode_steps: int = 1000
    sim_timeout: int = 180  # seconds per simulation run

    # Output settings
    output_dir: str = "."
    verbose: bool = False
    interactive: bool = False

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
