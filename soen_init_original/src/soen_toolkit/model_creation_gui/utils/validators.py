"""Tiny placeholder – add numeric/logic validators here."""


def positive_float(value: float, name: str = "value") -> None:
    if value <= 0:
        msg = f"{name} must be > 0 (got {value})"
        raise ValueError(msg)
