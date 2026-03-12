"""RoboScribe exceptions."""


class RoboScribeError(Exception):
    """Base exception for RoboScribe."""


class LLMError(RoboScribeError):
    """Error communicating with LLM backend."""


class SimulationError(RoboScribeError):
    """Error running simulation."""


class SimulationTimeout(SimulationError):
    """Simulation exceeded time limit."""


class PolicyError(RoboScribeError):
    """Error in generated policy code."""


class ConfigError(RoboScribeError):
    """Invalid configuration."""
