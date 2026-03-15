"""Tests for simulation runner (without requiring robosuite)."""

import json
import os
import tempfile

from roboscribe.sim.runner import SimulationRunner, HARNESS_TEMPLATE
from roboscribe.sim.env_registry import ENV_REGISTRY
from roboscribe.sim.trajectory import TrajectoryResult
from roboscribe.config import Config


def test_env_registry_has_required_envs():
    required = ["Lift", "Stack", "Door", "Wipe"]
    for env in required:
        assert env in ENV_REGISTRY
        info = ENV_REGISTRY[env]
        assert info.description
        assert info.goal_description
        assert info.tips


def test_env_info_obs_keys():
    info = ENV_REGISTRY["Lift"]
    assert "robot0_eef_pos" in info.obs_keys
    assert "cube_pos" in info.obs_keys
    assert info.action_dim == 7


def test_trajectory_result_properties():
    result = TrajectoryResult(success_rate=0.0, error="test error")
    assert result.failed is True
    assert result.partial_success is False

    result2 = TrajectoryResult(success_rate=0.5)
    assert result2.partial_success is True

    result3 = TrajectoryResult(success_rate=1.0)
    assert result3.failed is False


def test_runner_rejects_unknown_env():
    cfg = Config(api_key="test")
    runner = SimulationRunner(cfg)
    try:
        runner.run_policy("def get_action(obs): pass", "FakeEnv123")
        assert False, "Should have raised"
    except Exception as e:
        assert "Unknown environment" in str(e)


def test_harness_template_is_valid_python():
    """The harness template should be syntactically valid Python."""
    compile(HARNESS_TEMPLATE, "<harness>", "exec")
