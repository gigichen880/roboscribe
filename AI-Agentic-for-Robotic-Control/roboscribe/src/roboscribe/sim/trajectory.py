"""Trajectory result dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryResult:
    """Result of running a policy in simulation."""

    success_rate: float = 0.0
    successes: int = 0
    total_episodes: int = 0
    error: str = ""
    error_type: str = ""  # CODE_ERROR, TIMEOUT, RUNTIME_ERROR
    trajectory_summary: str = ""  # text summary of what happened
    final_obs: dict = field(default_factory=dict)  # last observation
    episode_rewards: list[float] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.error) or self.success_rate < 0.5

    @property
    def partial_success(self) -> bool:
        return 0.0 < self.success_rate < 1.0
