"""Tests for the agent loop (with mocked LLM and simulation)."""

from unittest.mock import MagicMock, patch

import pytest

from roboscribe.agent.loop import AgentLoop, GenerationResult
from roboscribe.agent.interactive import ReviewAction, ReviewResult
from roboscribe.config import Config
from roboscribe.llm.base import LLMResponse
from roboscribe.sim.trajectory import TrajectoryResult


MOCK_POLICY = """\
import numpy as np
state = 0
def reset():
    global state
    state = 0
def get_action(obs):
    return np.zeros(7)
"""


@pytest.fixture
def mock_config():
    return Config(
        llm_backend="openai",
        llm_model="gpt-4o",
        api_key="test-key",
        max_attempts=3,
        num_episodes=5,
        verbose=False,
        interactive=False,
    )


def test_loop_succeeds_first_attempt(mock_config):
    """Loop returns success if simulation passes on first try."""
    with patch("roboscribe.agent.loop.create_backend") as mock_llm, \
         patch("roboscribe.agent.loop.SimulationRunner") as mock_runner_cls, \
         patch("roboscribe.agent.loop.PolicyWriter") as mock_writer_cls:

        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(
            raw_text=f"```python\n{MOCK_POLICY}\n```",
            model="gpt-4o",
            tokens_used=100,
        )
        mock_llm.return_value = mock_backend

        mock_runner = MagicMock()
        mock_runner.run_policy.return_value = TrajectoryResult(
            success_rate=0.9,
            successes=9,
            total_episodes=10,
        )
        mock_runner_cls.return_value = mock_runner

        mock_writer = MagicMock()
        mock_writer.write.return_value = "lift_policy.py"
        mock_writer_cls.return_value = mock_writer

        loop = AgentLoop(mock_config)
        result = loop.run("pick up the cube", "Lift")

        assert result.success is True
        assert result.attempts == 1
        assert result.success_rate == 0.9


def test_loop_retries_on_failure(mock_config):
    """Loop retries when simulation fails, then succeeds."""
    with patch("roboscribe.agent.loop.create_backend") as mock_llm, \
         patch("roboscribe.agent.loop.SimulationRunner") as mock_runner_cls, \
         patch("roboscribe.agent.loop.PolicyWriter") as mock_writer_cls:

        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(
            raw_text=f"```python\n{MOCK_POLICY}\n```",
            model="gpt-4o",
            tokens_used=100,
        )
        mock_llm.return_value = mock_backend

        mock_runner = MagicMock()
        # First attempt fails, second succeeds
        mock_runner.run_policy.side_effect = [
            TrajectoryResult(success_rate=0.0, total_episodes=10, episode_rewards=[0.01]*10),
            TrajectoryResult(success_rate=0.9, successes=9, total_episodes=10),
        ]
        mock_runner_cls.return_value = mock_runner

        mock_writer = MagicMock()
        mock_writer.write.return_value = "lift_policy.py"
        mock_writer_cls.return_value = mock_writer

        loop = AgentLoop(mock_config)
        result = loop.run("pick up the cube", "Lift")

        assert result.success is True
        assert result.attempts == 2
        assert mock_backend.generate.call_count == 2


def test_loop_rejects_unknown_env(mock_config):
    """Loop raises error for unknown environment."""
    with patch("roboscribe.agent.loop.create_backend"), \
         patch("roboscribe.agent.loop.SimulationRunner"), \
         patch("roboscribe.agent.loop.PolicyWriter"):

        loop = AgentLoop(mock_config)
        with pytest.raises(Exception, match="Unknown environment"):
            loop.run("do something", "FakeEnv")


# -- Interactive mode tests --

@pytest.fixture
def interactive_config():
    return Config(
        llm_backend="openai",
        llm_model="gpt-4o",
        api_key="test-key",
        max_attempts=3,
        num_episodes=5,
        verbose=False,
        interactive=True,
    )


def test_interactive_quit_stops_loop(interactive_config):
    """User pressing quit in interactive mode stops the loop."""
    with patch("roboscribe.agent.loop.create_backend") as mock_llm, \
         patch("roboscribe.agent.loop.SimulationRunner") as mock_runner_cls, \
         patch("roboscribe.agent.loop.PolicyWriter") as mock_writer_cls, \
         patch("roboscribe.agent.loop.InteractiveReviewer") as mock_reviewer_cls:

        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(
            raw_text=f"```python\n{MOCK_POLICY}\n```",
            model="gpt-4o",
            tokens_used=100,
        )
        mock_llm.return_value = mock_backend

        mock_runner = MagicMock()
        mock_runner.run_policy.return_value = TrajectoryResult(
            success_rate=0.0, total_episodes=10, episode_rewards=[0.01] * 10,
        )
        mock_runner_cls.return_value = mock_runner

        mock_writer = MagicMock()
        mock_writer.write.return_value = "lift_policy.py"
        mock_writer_cls.return_value = mock_writer

        mock_reviewer = MagicMock()
        mock_reviewer.review.return_value = ReviewResult(action=ReviewAction.QUIT)
        mock_reviewer_cls.return_value = mock_reviewer

        loop = AgentLoop(interactive_config)
        result = loop.run("pick up the cube", "Lift")

        assert result.success is False
        assert result.attempts == 1
        assert mock_backend.generate.call_count == 1  # no revision attempted


def test_interactive_feedback_passed_to_revision(interactive_config):
    """User feedback in interactive mode is passed to the revision prompt."""
    with patch("roboscribe.agent.loop.create_backend") as mock_llm, \
         patch("roboscribe.agent.loop.SimulationRunner") as mock_runner_cls, \
         patch("roboscribe.agent.loop.PolicyWriter") as mock_writer_cls, \
         patch("roboscribe.agent.loop.InteractiveReviewer") as mock_reviewer_cls:

        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(
            raw_text=f"```python\n{MOCK_POLICY}\n```",
            model="gpt-4o",
            tokens_used=100,
        )
        mock_llm.return_value = mock_backend

        mock_runner = MagicMock()
        mock_runner.run_policy.side_effect = [
            TrajectoryResult(success_rate=0.0, total_episodes=10, episode_rewards=[0.01] * 10),
            TrajectoryResult(success_rate=0.9, successes=9, total_episodes=10),
        ]
        mock_runner_cls.return_value = mock_runner

        mock_writer = MagicMock()
        mock_writer.write.return_value = "lift_policy.py"
        mock_writer_cls.return_value = mock_writer

        mock_reviewer = MagicMock()
        mock_reviewer.review.return_value = ReviewResult(
            action=ReviewAction.FEEDBACK,
            human_feedback="lower the arm further before grasping",
        )
        mock_reviewer_cls.return_value = mock_reviewer

        loop = AgentLoop(interactive_config)
        result = loop.run("pick up the cube", "Lift")

        assert result.success is True
        assert result.attempts == 2
        # Verify build_revision_prompt was called with human_feedback
        # (the second generate call includes the revision prompt)
        assert mock_backend.generate.call_count == 2


def test_interactive_save_stops_loop(interactive_config):
    """User pressing save in interactive mode stops and saves best result."""
    with patch("roboscribe.agent.loop.create_backend") as mock_llm, \
         patch("roboscribe.agent.loop.SimulationRunner") as mock_runner_cls, \
         patch("roboscribe.agent.loop.PolicyWriter") as mock_writer_cls, \
         patch("roboscribe.agent.loop.InteractiveReviewer") as mock_reviewer_cls:

        mock_backend = MagicMock()
        mock_backend.generate.return_value = LLMResponse(
            raw_text=f"```python\n{MOCK_POLICY}\n```",
            model="gpt-4o",
            tokens_used=100,
        )
        mock_llm.return_value = mock_backend

        mock_runner = MagicMock()
        mock_runner.run_policy.return_value = TrajectoryResult(
            success_rate=0.3, successes=3, total_episodes=10,
            episode_rewards=[0.5] * 10,
        )
        mock_runner_cls.return_value = mock_runner

        mock_writer = MagicMock()
        mock_writer.write.return_value = "lift_policy.py"
        mock_writer_cls.return_value = mock_writer

        mock_reviewer = MagicMock()
        mock_reviewer.review.return_value = ReviewResult(action=ReviewAction.SAVE)
        mock_reviewer_cls.return_value = mock_reviewer

        loop = AgentLoop(interactive_config)
        result = loop.run("pick up the cube", "Lift")

        assert result.success is False
        assert result.attempts == 1
        assert result.success_rate == 0.3
        mock_writer.write.assert_called_once()
