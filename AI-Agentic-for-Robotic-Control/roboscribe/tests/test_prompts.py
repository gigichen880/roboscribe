"""Tests for prompt generation."""

from roboscribe.agent.prompts import (
    SYSTEM_PROMPT,
    build_generation_prompt,
    build_revision_prompt,
)
from roboscribe.sim.env_registry import ENV_REGISTRY
from roboscribe.sim.diagnostics import Diagnosis


def test_system_prompt_has_key_info():
    assert "OSC_POSE" in SYSTEM_PROMPT
    assert "get_action" in SYSTEM_PROMPT
    assert "gripper" in SYSTEM_PROMPT.lower()
    assert "7" in SYSTEM_PROMPT  # 7-dimensional


def test_generation_prompt_lift():
    env_info = ENV_REGISTRY["Lift"]
    prompt = build_generation_prompt("pick up the cube", env_info)
    assert "pick up the cube" in prompt
    assert "Lift" in prompt
    assert "cube_pos" in prompt
    assert "robot0_eef_pos" in prompt
    # Should include few-shot example
    assert "get_action" in prompt
    assert "APPROACH" in prompt


def test_generation_prompt_door():
    env_info = ENV_REGISTRY["Door"]
    prompt = build_generation_prompt("open the door", env_info)
    assert "open the door" in prompt
    assert "Door" in prompt
    assert "handle" in prompt.lower()


def test_revision_prompt():
    env_info = ENV_REGISTRY["Lift"]
    diag = Diagnosis(
        category="MISSED_GRASP",
        summary="Robot moves toward object but fails to grasp",
        details="Gripper closes too early",
        suggestions="Lower further before closing gripper",
    )
    prompt = build_revision_prompt(
        "pick up the cube", env_info,
        "def get_action(obs): return np.zeros(7)",
        diag,
        "Episode 0: reward=0.5, success=False",
    )
    assert "MISSED_GRASP" in prompt
    assert "Lower further" in prompt
    assert "FAILED" in prompt
    assert "def get_action" in prompt


def test_revision_prompt_with_human_feedback():
    """Human feedback should appear in the revision prompt."""
    env_info = ENV_REGISTRY["Lift"]
    diag = Diagnosis(
        category="MISSED_GRASP",
        summary="Robot fails to grasp",
        details="",
        suggestions="Lower further",
    )
    prompt = build_revision_prompt(
        "pick up the cube", env_info,
        "def get_action(obs): return np.zeros(7)",
        diag,
        "",
        human_feedback="Try closing the gripper more slowly",
    )
    assert "Human Feedback" in prompt
    assert "high-priority" in prompt
    assert "Try closing the gripper more slowly" in prompt
    assert "Pay special attention" in prompt


def test_revision_prompt_without_human_feedback():
    """No human feedback section when feedback is empty."""
    env_info = ENV_REGISTRY["Lift"]
    diag = Diagnosis(
        category="MISSED_GRASP",
        summary="Robot fails to grasp",
        details="",
        suggestions="Lower further",
    )
    prompt = build_revision_prompt(
        "pick up the cube", env_info,
        "def get_action(obs): return np.zeros(7)",
        diag,
        "",
        human_feedback="",
    )
    assert "Human Feedback" not in prompt
    assert "Pay special attention" not in prompt
