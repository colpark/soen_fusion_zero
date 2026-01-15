"""Small helpers for working with S3 URIs in training configs.

This module avoids importing heavy AWS SDKs. It only normalizes and parses
`s3://bucket/key` strings and provides a simple heuristic for file-like keys.
"""

from __future__ import annotations


def is_s3_uri(text: str | None) -> bool:
    if not text:
        return False
    t = str(text).strip()
    return t.startswith("s3://")


def normalize_s3_uri(text: str) -> str:
    """Return a canonical `s3://bucket/key` form without trailing slashes.

    Does not validate existence.
    """
    t = str(text).strip()
    if not t.startswith("s3://"):
        return t
    # Remove redundant slashes but keep scheme and bucket delimiter
    while "//" in t[5:]:
        t = t[:5] + t[5:].replace("//", "/")
    return t.rstrip("/")


def split_s3_uri(uri: str) -> tuple[str, str]:
    """Split `s3://bucket/prefix` into (bucket, key). Key may be empty.
    Raises ValueError if not an s3 URI.
    """
    u = normalize_s3_uri(uri)
    if not is_s3_uri(u):
        msg = f"Not an S3 URI: {uri}"
        raise ValueError(msg)
    rest = u[len("s3://") :]
    if "/" in rest:
        bucket, key = rest.split("/", 1)
    else:
        bucket, key = rest, ""
    return bucket, key


def looks_like_file_uri(uri: str) -> bool:
    """Heuristic to decide if an S3 URI likely points to a file object.

    We consider it file-like when the last path segment has an extension we
    commonly use (e.g., .h5, .hdf5, .soen, .ckpt, .pt, .pth, .yaml, .yml).
    This remains a heuristic
    callers should keep behavior forgiving and log
    clearly when assumptions might not hold.
    """
    try:
        _, key = split_s3_uri(uri)
    except Exception:
        return False
    if not key:
        return False
    last = key.split("/")[-1]
    lower = last.lower()
    file_exts = (
        ".h5",
        ".hdf5",
        ".soen",
        ".ckpt",
        ".pt",
        ".pth",
        ".yaml",
        ".yml",
        ".csv",
    )
    return any(lower.endswith(ext) for ext in file_exts)
