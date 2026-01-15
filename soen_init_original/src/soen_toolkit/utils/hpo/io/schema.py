#!/usr/bin/env python3
"""Helpers for building standardized trials DataFrames."""

from __future__ import annotations

import contextlib
import json

import numpy as np
import pandas as pd


def build_trials_df_from_jsonl(jsonl_path: str) -> pd.DataFrame:
    rows = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rows.append(rec)
    except FileNotFoundError:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Normalize base columns
    if "trial" in df.columns and "trial_number" not in df.columns:
        df = df.rename(columns={"trial": "trial_number"})
    # Ensure present
    for col in ("trial_number", "state", "value"):
        if col not in df.columns:
            df[col] = None
    # Flags
    try:
        df["is_complete"] = df["state"].astype(str) == "COMPLETE"
        df["is_failed"] = df["state"].astype(str) == "FAIL"
        df["is_pruned"] = df["state"].astype(str) == "PRUNED"
    except Exception:
        df["is_complete"] = False
        df["is_failed"] = False
        df["is_pruned"] = False
    # Value to float
    with contextlib.suppress(Exception):
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Expand params/metrics
    def _expand(frame: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
        if col not in frame.columns:
            return frame
        try:
            expanded = pd.json_normalize(frame[col].fillna({}))
            expanded.columns = [f"{prefix}{c}" for c in expanded.columns]
            frame = frame.drop(columns=[col])
            frame = frame.reset_index(drop=True)
            expanded = expanded.reindex(frame.index, fill_value=np.nan)
            for c in expanded.columns:
                frame[c] = expanded[c]
        except Exception:
            pass
        return frame

    df = _expand(df, "params", "param_")
    df = _expand(df, "user_attrs", "metric_")

    # Rank/percentile for complete trials
    try:
        mask = df["is_complete"] & ~df["value"].isna()
        if mask.any():
            df.loc[mask, "rank"] = df.loc[mask, "value"].rank(ascending=False)
            df.loc[mask, "percentile"] = df.loc[mask, "rank"] / mask.sum()
    except Exception:
        pass
    return df
