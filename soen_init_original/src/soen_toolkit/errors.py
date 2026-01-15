"""Project-wide exception types for clearer error handling."""


class SoenToolkitError(Exception):
    """Base class for all SOEN-Toolkit custom exceptions."""


class ConfigError(SoenToolkitError):
    """Configuration-related errors."""


class DatasetError(SoenToolkitError):
    """Errors related to dataset loading or creation."""


class ModelStateError(SoenToolkitError):
    """Errors related to model availability or invalid state."""


__all__ = [
    "SoenToolkitError",
    "ConfigError",
    "DatasetError",
    "ModelStateError",
]
