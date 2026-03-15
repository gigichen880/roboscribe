"""PID controllers for smooth robotic control.

Provides position (3D) and rotation (1D angular) PID controllers
used by generated scripted policies.
"""

from __future__ import annotations

import numpy as np


class PID:
    """3D position PID controller."""

    def __init__(self, kp: float, ki: float, kd: float, target=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = np.array(target) if target is not None else np.zeros(3)
        self.integral_error = np.zeros(3)
        self.previous_error = None

    def reset(self, target=None):
        """Reset internal error history. Optionally set new target."""
        self.integral_error = np.zeros(3)
        self.previous_error = None
        if target is not None:
            self.target = np.array(target)

    def get_error(self) -> float:
        """Euclidean distance from current position to target."""
        if self.previous_error is None:
            return 0.0
        return float(np.linalg.norm(self.previous_error))

    def update(self, current_pos, dt: float) -> np.ndarray:
        """Compute PID control signal.

        Args:
            current_pos: Current 3D position.
            dt: Time step in seconds.

        Returns:
            3D control output vector.
        """
        current_pos = np.array(current_pos)
        error = self.target - current_pos

        P = self.kp * error

        self.integral_error += error * dt
        I = self.ki * self.integral_error

        if self.previous_error is None:
            D = np.zeros(3)
        else:
            D = self.kd * (error - self.previous_error) / dt

        self.previous_error = error
        return P + I + D


class RotationPID:
    """1D angular PID controller with proper angle wrapping."""

    def __init__(self, kp: float, ki: float, kd: float, target: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral_error = 0.0
        self.previous_error = None
        self.integral_limit = np.pi

    def reset(self, target: float = None):
        """Reset internal state. Optionally set new target."""
        self.integral_error = 0.0
        self.previous_error = None
        if target is not None:
            self.target = target

    def get_error(self) -> float:
        """Absolute angular error in radians."""
        if self.previous_error is None:
            return 0.0
        return abs(self.previous_error)

    def update(self, current_angle: float, dt: float) -> float:
        """Compute angular PID control signal.

        Args:
            current_angle: Current angle in radians.
            dt: Time step in seconds.

        Returns:
            Rotation rate command.
        """
        error = np.arctan2(
            np.sin(self.target - current_angle),
            np.cos(self.target - current_angle),
        )

        P = self.kp * error

        self.integral_error += error * dt
        self.integral_error = np.clip(
            self.integral_error, -self.integral_limit, self.integral_limit,
        )
        I = self.ki * self.integral_error

        if self.previous_error is None:
            D = 0.0
        else:
            error_diff = np.arctan2(
                np.sin(error - self.previous_error),
                np.cos(error - self.previous_error),
            )
            D = self.kd * error_diff / dt

        self.previous_error = error
        return P + I + D
