"""Shared test fixtures."""

import pytest

from roboscribe.config import Config


@pytest.fixture
def config():
    """Default test config."""
    return Config(
        llm_backend="openai",
        llm_model="gpt-4o",
        api_key="test-key-123",
        max_attempts=3,
        num_episodes=5,
        sim_timeout=30,
    )
