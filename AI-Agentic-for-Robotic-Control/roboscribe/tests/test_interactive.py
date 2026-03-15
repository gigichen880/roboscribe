"""Tests for the interactive review module."""

from unittest.mock import patch, MagicMock

import pytest

from roboscribe.agent.interactive import (
    InteractiveReviewer,
    ReviewAction,
    ReviewResult,
)
from roboscribe.sim.trajectory import TrajectoryResult
from roboscribe.sim.diagnostics import Diagnosis


SAMPLE_POLICY = """\
import numpy as np

APPROACH = 0
LOWER = 1
GRASP = 2
LIFT = 3

state = APPROACH

def reset():
    global state
    state = APPROACH

def get_action(obs):
    global state
    eef = obs['robot0_eef_pos']
    cube = obs['cube_pos']
    action = np.zeros(7)
    if state == APPROACH:
        action[:3] = 10.0 * (cube - eef)
        action[2] = 0
        if np.linalg.norm(cube[:2] - eef[:2]) < 0.02:
            state = LOWER
    elif state == LOWER:
        action[2] = -0.5
        if eef[2] < cube[2] + 0.01:
            state = GRASP
    elif state == GRASP:
        action[6] = 1.0
        state = LIFT
    elif state == LIFT:
        action[2] = 0.5
        action[6] = 1.0
    return np.clip(action, -1, 1)
"""


@pytest.fixture
def reviewer():
    return InteractiveReviewer()


@pytest.fixture
def sim_result():
    return TrajectoryResult(
        success_rate=0.3,
        successes=3,
        total_episodes=10,
        episode_rewards=[0.5, 1.2, 0.3, 2.45, 0.1, 0.8, 1.0, 0.6, 0.9, 0.4],
    )


@pytest.fixture
def diagnosis():
    return Diagnosis(
        category="INCONSISTENT",
        summary="Policy succeeds 30% of the time",
        details="Timing issues",
        suggestions="Widen position thresholds",
    )


class TestPhaseExtraction:
    """Tests for _extract_phases."""

    def test_extracts_constant_definitions(self, reviewer):
        phases = reviewer._extract_phases(SAMPLE_POLICY)
        names = [p.name for p in phases]
        assert "APPROACH" in names
        assert "LOWER" in names
        assert "GRASP" in names
        assert "LIFT" in names

    def test_extracts_correct_line_numbers(self, reviewer):
        phases = reviewer._extract_phases(SAMPLE_POLICY)
        phase_map = {p.name: p.line_number for p in phases}
        # APPROACH = 0 is line 3 of the sample
        assert phase_map["APPROACH"] == 3
        assert phase_map["LOWER"] == 4

    def test_no_duplicates(self, reviewer):
        phases = reviewer._extract_phases(SAMPLE_POLICY)
        names = [p.name for p in phases]
        assert len(names) == len(set(names))

    def test_empty_code(self, reviewer):
        assert reviewer._extract_phases("") == []

    def test_no_state_machine(self, reviewer):
        code = "def get_action(obs):\n    return np.zeros(7)\n"
        assert reviewer._extract_phases(code) == []


class TestReviewActions:
    """Tests for the review menu with mocked input."""

    def test_continue_action(self, reviewer, sim_result, diagnosis):
        with patch("click.prompt", return_value="c"):
            result = reviewer.review(
                SAMPLE_POLICY, sim_result, diagnosis, 1, 5,
            )
        assert result.action == ReviewAction.CONTINUE
        assert result.human_feedback == ""

    def test_quit_action(self, reviewer, sim_result, diagnosis):
        with patch("click.prompt", return_value="q"):
            result = reviewer.review(
                SAMPLE_POLICY, sim_result, diagnosis, 1, 5,
            )
        assert result.action == ReviewAction.QUIT

    def test_save_action(self, reviewer, sim_result, diagnosis):
        with patch("click.prompt", return_value="s"):
            result = reviewer.review(
                SAMPLE_POLICY, sim_result, diagnosis, 1, 5,
            )
        assert result.action == ReviewAction.SAVE

    def test_feedback_action(self, reviewer, sim_result, diagnosis):
        with patch("click.prompt", return_value="f"), \
             patch("builtins.input", side_effect=["try lowering more", "", ""]):
            result = reviewer.review(
                SAMPLE_POLICY, sim_result, diagnosis, 1, 5,
            )
        assert result.action == ReviewAction.FEEDBACK
        assert "try lowering more" in result.human_feedback

    def test_feedback_empty_falls_back(self, reviewer, sim_result, diagnosis):
        """Empty feedback still returns FEEDBACK action."""
        with patch("click.prompt", return_value="f"), \
             patch("builtins.input", side_effect=["", ""]):
            result = reviewer.review(
                SAMPLE_POLICY, sim_result, diagnosis, 1, 5,
            )
        assert result.action == ReviewAction.FEEDBACK
        assert result.human_feedback == ""


class TestCodeStyling:
    """Tests for _style_code_line."""

    def test_constant_gets_yellow(self, reviewer):
        styled = reviewer._style_code_line("APPROACH = 0")
        assert "\x1b[" in styled  # contains ANSI escape

    def test_phase_check_gets_styled(self, reviewer):
        styled = reviewer._style_code_line("    if state == APPROACH:")
        assert "\x1b[" in styled

    def test_def_gets_bold(self, reviewer):
        styled = reviewer._style_code_line("def get_action(obs):")
        assert "\x1b[" in styled

    def test_plain_line_unchanged(self, reviewer):
        line = "    x = 42"
        styled = reviewer._style_code_line(line)
        assert styled == line
