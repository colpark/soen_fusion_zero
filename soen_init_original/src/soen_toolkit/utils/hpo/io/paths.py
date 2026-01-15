#!/usr/bin/env python3
"""Study path utilities for HPO runs.

Centralizes directory structure and file naming so callers don't reimplement it.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

DEFAULT_OUT_DIR = os.path.join(os.getcwd(), "criticality_runs")


@dataclass
class StudyPaths:
    """Resolved filesystem locations for a single study.

    Use this instead of ad-hoc path concatenation.
    """

    out_dir: str
    study_name: str

    @property
    def study_dir(self) -> str:
        p = os.path.join(self.out_dir, f"optuna_report_{self.study_name}")
        Path(p).mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> str:
        return os.path.join(self.study_dir, "optuna_studies.db")

    @property
    def summary_json(self) -> str:
        return os.path.join(self.study_dir, f"optuna_summary_{self.study_name}.json")

    @property
    def best_spec_yaml(self) -> str:
        return os.path.join(self.study_dir, f"best_spec_{self.study_name}.yaml")

    @property
    def trials_jsonl(self) -> str:
        return os.path.join(self.study_dir, "trials.jsonl")


def get_study_paths(study_name: str, *, out_dir: str | None = None) -> StudyPaths:
    """Create a StudyPaths helper for consistent file layout."""
    return StudyPaths(out_dir=(out_dir or DEFAULT_OUT_DIR), study_name=study_name)
