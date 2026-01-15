"""Local exceptions for state trajectory module."""


class StateTrajectoryError(Exception):
    """Base exception for state trajectory errors."""


class TimebaseError(StateTrajectoryError):
    """Timebase calculation errors."""


class InputGenerationError(StateTrajectoryError):
    """Input generation errors."""


class SimulationError(StateTrajectoryError):
    """Backend simulation errors."""


class DatasetServiceError(StateTrajectoryError):
    """Dataset loading/access errors."""
