#!/usr/bin/env python3
"""Data processing module for criticality hyperparameter optimization results.
Handles complex conditional parameter structures and extracts insights from Optuna studies.
"""

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrialData:
    """Structured representation of a single trial."""

    number: int
    value: float
    state: str
    params: dict[str, Any]
    user_attrs: dict[str, Any]
    datetime_start: str | None = None
    datetime_complete: str | None = None
    duration: float | None = None


@dataclass
class StudySummary:
    """High-level study statistics."""

    study_name: str
    n_trials: int
    n_complete: int
    n_failed: int
    n_pruned: int
    best_value: float
    best_trial: int
    elapsed_time: float
    direction: str


@dataclass
class CriticalityMetrics:
    """Organized criticality metrics from trials."""

    branching_sigma: list[float]
    branching_cost: list[float]
    branching_cost_v2: list[float]
    beta_temporal: list[float]
    beta_spatial: list[float]
    psd_temporal_cost: list[float]
    psd_spatial_cost: list[float]
    chi_variance: list[float]
    chi_inv_cost: list[float]
    avalanche_cost: list[float]
    autocorr_cost: list[float]
    jacobian_cost: list[float]
    jac_spectral_radius: list[float]
    total_cost: list[float]


@dataclass
class ParameterGroups:
    """Organized parameter groups with conditional awareness."""

    layer_params: dict[str, dict[str, Any]]  # param_name -> {trials, distributions, values}
    connection_params: dict[str, dict[str, Any]]
    weight_params: dict[str, dict[str, Any]]
    conditional_params: dict[str, dict[str, Any]]  # params that depend on others


class CriticalityDataProcessor:
    """Process and organize Optuna study data for visualization."""

    def __init__(self, study: optuna.Study, config: dict[str, Any] | None = None) -> None:
        """Initialize with Optuna study and optional configuration.

        Args:
            study: Optuna study object
            config: Configuration dict used for the optimization

        """
        self.study = study
        self.config = config or {}
        self.trials_df = None
        self._processed_data = {}

    def process_all(self) -> dict[str, Any]:
        """Process all data and return organized results."""
        if self._processed_data:
            return self._processed_data

        logger.info("Processing Optuna study data...")

        # Extract basic trial data
        trials_data = self._extract_trial_data()
        self.trials_df = self._create_trials_dataframe(trials_data)
        # Create a visualization-friendly filtered dataframe to drop extreme outliers
        trials_df_filtered = self._filter_extreme_objectives(self.trials_df)

        # Process different aspects
        study_summary = self._create_study_summary()
        criticality_metrics = self._extract_criticality_metrics()
        parameter_groups = self._organize_parameters()
        parameter_importance = self._calculate_parameter_importance()
        correlations = self._calculate_correlations()
        best_trials_analysis = self._analyze_best_trials()
        failure_analysis = self._analyze_failures()
        search_space_summary = self._summarize_search_space()

        self._processed_data = {
            "study_summary": study_summary,
            "trials_data": trials_data,
            "trials_df": self.trials_df,
            "trials_df_filtered": trials_df_filtered,
            "criticality_metrics": criticality_metrics,
            "parameter_groups": parameter_groups,
            "parameter_importance": parameter_importance,
            "correlations": correlations,
            "best_trials_analysis": best_trials_analysis,
            "failure_analysis": failure_analysis,
            "search_space_summary": search_space_summary,
            "config": self.config,
        }

        logger.info(f"Processed {len(trials_data)} trials")
        return self._processed_data

    def _filter_extreme_objectives(self, df: pd.DataFrame, *, bottom_frac: float = 0.05) -> pd.DataFrame:
        """Return a copy of df filtered for visualization.

        New rule: keep only COMPLETE + finite trials and drop the bottom
        `bottom_frac` of trials by objective value. Objective values are the
        Optuna 'value' field (our score), so bottom means worst-performing.
        """
        if df is None or df.empty or "value" not in df.columns:
            return df.copy() if df is not None else df

        try:
            complete = df[(df.get("is_complete", False)) & (~df["value"].isna())].copy()
            # Exclude failed-threshold trials: value <= -100
            if not complete.empty:
                complete = complete[complete["value"] > -100.0]
            if complete.empty:
                return complete
            vals = complete["value"].astype(float)
            # Drop non-finite
            finite_mask = np.isfinite(vals.values)
            complete = complete.loc[finite_mask]
            vals = complete["value"].astype(float)
            if len(vals) < 5:
                return complete
            # Cut off the bottom fraction of trials by value
            cutoff = float(np.percentile(vals, bottom_frac * 100.0))
            keep_mask = vals >= cutoff
            filtered = complete.loc[keep_mask]
            dropped = len(complete) - len(filtered)
            if dropped > 0:
                logger.info(f"Filtered bottom {int(bottom_frac * 100)}% trials for viz (cutoff {cutoff:.6g}, dropped {dropped})")
            return filtered
        except Exception as e:
            logger.warning(f"Objective filtering failed: {e}; returning unfiltered COMPLETE trials")
            out = df[df.get("is_complete", False) & ~df["value"].isna()].copy()
            if not out.empty:
                out = out[out["value"] > -100.0]
            return out

    def _extract_trial_data(self) -> list[TrialData]:
        """Extract structured data from all trials."""
        trials_data = []

        for trial in self.study.trials:
            trial_data = TrialData(
                number=trial.number,
                value=trial.value if trial.value is not None else float("nan"),
                state=trial.state.name,
                params=trial.params.copy(),
                user_attrs=trial.user_attrs.copy(),
                datetime_start=trial.datetime_start.isoformat() if trial.datetime_start else None,
                datetime_complete=trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                duration=trial.duration.total_seconds() if trial.duration else None,
            )
            trials_data.append(trial_data)

        return trials_data

    def _create_trials_dataframe(self, trials_data: list[TrialData]) -> pd.DataFrame:
        """Create a comprehensive DataFrame from trial data."""
        rows = []

        for trial in trials_data:
            row = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state,
                "duration": trial.duration,
            }

            # Add parameters
            for param, value in trial.params.items():
                row[f"param_{param}"] = value

            # Add metrics
            for metric, value in trial.user_attrs.items():
                row[f"metric_{metric}"] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        # Add derived columns
        if not df.empty:
            df["is_complete"] = df["state"] == "COMPLETE"
            df["is_failed"] = df["state"] == "FAIL"
            df["is_pruned"] = df["state"] == "PRUNED"

            # Add rank columns for complete trials
            complete_mask = df["is_complete"] & ~df["value"].isna()
            if complete_mask.any():
                df.loc[complete_mask, "rank"] = df.loc[complete_mask, "value"].rank(ascending=False)
                df.loc[complete_mask, "percentile"] = df.loc[complete_mask, "rank"] / complete_mask.sum()

        return df

    def _create_study_summary(self) -> StudySummary:
        """Create high-level study summary."""
        trials = self.study.trials
        complete_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
        pruned_trials = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]

        # Calculate elapsed time
        if trials:
            start_times = [t.datetime_start for t in trials if t.datetime_start]
            end_times = [t.datetime_complete for t in trials if t.datetime_complete]
            if start_times and end_times:
                elapsed = (max(end_times) - min(start_times)).total_seconds()
            else:
                elapsed = 0.0
        else:
            elapsed = 0.0

        return StudySummary(
            study_name=self.study.study_name,
            n_trials=len(trials),
            n_complete=len(complete_trials),
            n_failed=len(failed_trials),
            n_pruned=len(pruned_trials),
            best_value=self.study.best_value if complete_trials else float("nan"),
            best_trial=self.study.best_trial.number if complete_trials else -1,
            elapsed_time=elapsed,
            direction=self.study.direction.name,
        )

    def _extract_criticality_metrics(self) -> CriticalityMetrics:
        """Extract and organize criticality metrics."""
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        def extract_metric(metric_name: str) -> list[float]:
            values = []
            for trial in complete_trials:
                value = trial.user_attrs.get(metric_name)
                if value is not None and not math.isnan(float(value)):
                    values.append(float(value))
                else:
                    values.append(float("nan"))
            return values

        return CriticalityMetrics(
            branching_sigma=extract_metric("branching_sigma"),
            branching_cost=extract_metric("branching_cost"),
            branching_cost_v2=extract_metric("branching_cost_v2"),
            beta_temporal=extract_metric("beta_temporal"),
            beta_spatial=extract_metric("beta_spatial_mean"),
            psd_temporal_cost=extract_metric("psd_temporal_cost"),
            psd_spatial_cost=extract_metric("psd_spatial_cost"),
            chi_variance=extract_metric("chi_variance"),
            chi_inv_cost=extract_metric("chi_inv_cost"),
            avalanche_cost=extract_metric("avalanche_cost"),
            autocorr_cost=extract_metric("autocorr_cost"),
            jacobian_cost=extract_metric("jacobian_cost"),
            jac_spectral_radius=extract_metric("jac_spectral_radius"),
            total_cost=extract_metric("total_cost"),
        )

    def _organize_parameters(self) -> ParameterGroups:
        """Organize parameters by type and handle conditional relationships."""
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        layer_params = {}
        connection_params = {}
        weight_params = {}
        conditional_params = {}

        # Collect all parameter names
        all_param_names = set()
        for trial in complete_trials:
            all_param_names.update(trial.params.keys())

        # Categorize parameters
        for param_name in all_param_names:
            values = []
            distributions = []

            for trial in complete_trials:
                if param_name in trial.params:
                    values.append(trial.params[param_name])

                    # Extract distribution info if available
                    if param_name.endswith("_dist"):
                        distributions.append(trial.params[param_name])

            param_info = {
                "values": values,
                "distributions": distributions if distributions else None,
                "n_trials": len(values),
                "unique_values": len(set(values)) if values else 0,
            }

            # Categorize by parameter type
            if param_name.startswith("layer_"):
                layer_params[param_name] = param_info
            elif param_name in ["connection_type", "alpha", "sparsity", "expected_fan_out", "block_count", "connection_mode", "within_block_density", "cross_block_density", "allow_self_connections"]:
                connection_params[param_name] = param_info
            elif param_name.startswith("weight_") or param_name in ["init", "mean", "std", "gain", "a", "b"]:
                weight_params[param_name] = param_info
            # Check for conditional relationships
            elif self._is_conditional_parameter(param_name, complete_trials):
                conditional_params[param_name] = param_info

        return ParameterGroups(
            layer_params=layer_params,
            connection_params=connection_params,
            weight_params=weight_params,
            conditional_params=conditional_params,
        )

    def _is_conditional_parameter(self, param_name: str, trials: list) -> bool:
        """Check if a parameter is conditional (only appears in some trials)."""
        total_trials = len(trials)
        param_trials = sum(1 for trial in trials if param_name in trial.params)

        # If parameter appears in less than 50% of trials, consider it conditional
        return param_trials < total_trials * 0.5

    def _calculate_parameter_importance(self) -> dict[str, float]:
        """Calculate parameter importance using correlation with objective."""
        if self.trials_df is None or self.trials_df.empty:
            return {}

        complete_df = self.trials_df[self.trials_df["is_complete"] & ~self.trials_df["value"].isna()].copy()
        if len(complete_df) < 10:  # Need minimum trials for meaningful correlation
            return {}

        importance = {}
        param_columns = [col for col in complete_df.columns if col.startswith("param_")]

        for param_col in param_columns:
            param_name = param_col[6:]  # Remove 'param_' prefix

            # Skip non-numeric parameters for correlation
            if complete_df[param_col].dtype in ["object", "string"]:
                # For categorical parameters, use variance in objective across categories
                try:
                    grouped = complete_df.groupby(param_col)["value"].agg(["mean", "std", "count"])
                    if len(grouped) > 1:
                        # Weight by sample size and use coefficient of variation
                        weighted_std = np.average(grouped["std"].fillna(0), weights=grouped["count"])
                        mean_value = grouped["mean"].mean()
                        if mean_value != 0:
                            importance[param_name] = abs(weighted_std / mean_value)
                        else:
                            importance[param_name] = weighted_std
                    else:
                        importance[param_name] = 0.0
                except Exception:
                    importance[param_name] = 0.0
            else:
                # For numeric parameters, use absolute correlation
                try:
                    corr = complete_df[param_col].corr(complete_df["value"])
                    importance[param_name] = abs(corr) if not math.isnan(corr) else 0.0
                except Exception:
                    importance[param_name] = 0.0

        return importance

    def _calculate_correlations(self) -> dict[str, Any]:
        """Calculate correlations between parameters and metrics."""
        if self.trials_df is None or self.trials_df.empty:
            return {}

        complete_df = self.trials_df[self.trials_df["is_complete"]].copy()
        if len(complete_df) < 10:
            return {}

        # Get numeric columns only
        numeric_cols = complete_df.select_dtypes(include=[np.number]).columns
        param_cols = [col for col in numeric_cols if col.startswith("param_")]
        metric_cols = [col for col in numeric_cols if col.startswith("metric_")]

        correlations = {}

        # Parameter-objective correlations
        if "value" in numeric_cols:
            param_obj_corr = {}
            try:
                value_std = float(complete_df["value"].std(ddof=0))
            except Exception:
                value_std = 0.0
            for param_col in param_cols:
                try:
                    param_std = float(complete_df[param_col].std(ddof=0))
                    if param_std > 0.0 and value_std > 0.0:
                        corr = complete_df[param_col].corr(complete_df["value"])
                        param_obj_corr[param_col[6:]] = corr if not math.isnan(corr) else 0.0
                    else:
                        param_obj_corr[param_col[6:]] = 0.0
                except Exception:
                    param_obj_corr[param_col[6:]] = 0.0
            correlations["param_objective"] = param_obj_corr

        # Metric-objective correlations
        if "value" in numeric_cols:
            metric_obj_corr = {}
            try:
                value_std = float(complete_df["value"].std(ddof=0))
            except Exception:
                value_std = 0.0
            for metric_col in metric_cols:
                try:
                    metric_std = float(complete_df[metric_col].std(ddof=0))
                    if metric_std > 0.0 and value_std > 0.0:
                        corr = complete_df[metric_col].corr(complete_df["value"])
                        metric_obj_corr[metric_col[7:]] = corr if not math.isnan(corr) else 0.0
                    else:
                        metric_obj_corr[metric_col[7:]] = 0.0
                except Exception:
                    metric_obj_corr[metric_col[7:]] = 0.0
            correlations["metric_objective"] = metric_obj_corr

        # Inter-metric correlations
        # Drop constant/no-variance metric columns to avoid numpy warnings
        metric_cols_nonconst = [col for col in metric_cols if not math.isclose(float(complete_df[col].std(ddof=0) or 0.0), 0.0)]
        if len(metric_cols_nonconst) > 1:
            metric_corr = complete_df[metric_cols_nonconst].corr()
            # Clean column names
            metric_corr.columns = [col[7:] for col in metric_corr.columns]
            metric_corr.index = [idx[7:] for idx in metric_corr.index]
            correlations["metric_metric"] = metric_corr.to_dict()

        return correlations

    def _summarize_search_space(self) -> dict[str, Any]:
        """Summarize the search space based on observed parameter names across trials.

        We count the unique Optuna parameter keys (not bounds) that appeared in any trial.
        Categories:
          - layer: keys starting with 'layer_'
          - connections: keys matching 'J_<from>_to_<to>_*'
          - other: everything else (global or legacy keys)
        """
        trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            return {
                "total_params": 0,
                "by_category": {"layer": 0, "connections": 0, "other": 0},
                "by_connection": {},
                "by_type": {"continuous": 0, "categorical": 0, "mixed": 0, "other": 0},
            }

        all_param_names = set()
        # Track parameter types using Optuna distributions when available
        type_by_param: dict[str, str] = {}
        for t in trials:
            try:
                all_param_names.update(t.params.keys())
                for pname, dist in t.distributions.items():
                    try:
                        from optuna.distributions import (
                            CategoricalDistribution,
                            DiscreteUniformDistribution,
                            FloatDistribution,
                            IntDistribution,
                            LogUniformDistribution,
                            UniformDistribution,
                        )
                    except Exception:
                        FloatDistribution = IntDistribution = CategoricalDistribution = ()
                        UniformDistribution = LogUniformDistribution = DiscreteUniformDistribution = ()
                    # Determine coarse type
                    if isinstance(dist, (CategoricalDistribution,)):
                        tname = "categorical"
                    elif isinstance(dist, (FloatDistribution, IntDistribution, UniformDistribution, LogUniformDistribution, DiscreteUniformDistribution)):
                        tname = "continuous"
                    else:
                        tname = "other"
                    prev = type_by_param.get(pname)
                    if prev is None:
                        type_by_param[pname] = tname
                    elif prev != tname:
                        type_by_param[pname] = "mixed"
            except Exception:
                pass

        # Categorize
        import re

        layer_params = {p for p in all_param_names if str(p).startswith("layer_")}
        conn_pattern = re.compile(r"^J_\d+_to_\d+_")
        connection_params = {p for p in all_param_names if conn_pattern.match(str(p))}
        other_params = all_param_names - layer_params - connection_params

        # Per-connection breakdown
        by_connection: dict[str, int] = {}
        for p in connection_params:
            try:
                p.split("_", 4)  # J, from, to, rest...
                # safer: reconstruct J_<from>_to_<to>
                parts = str(p).split("_")
                # expect ['J', from, 'to', to, ...]
                if len(parts) >= 4 and parts[0] == "J" and parts[2] == "to":
                    key = f"J_{parts[1]}_to_{parts[3]}"
                else:
                    # fallback, split at first four segments
                    key = "_".join(parts[:4])
                by_connection[key] = by_connection.get(key, 0) + 1
            except Exception:
                pass

        # By coarse type counts (continuous = numeric, categorical = categorical distributions)
        counts_by_type = {"continuous": 0, "categorical": 0, "mixed": 0, "other": 0}
        for p in all_param_names:
            counts_by_type[type_by_param.get(p, "other")] = counts_by_type.get(type_by_param.get(p, "other"), 0) + 1

        return {
            "total_params": len(all_param_names),
            "by_category": {
                "layer": len(layer_params),
                "connections": len(connection_params),
                "other": len(other_params),
            },
            "by_type": counts_by_type,
            "by_connection": dict(sorted(by_connection.items(), key=lambda x: x[0])),
        }

    def _analyze_best_trials(self, top_n: int = 10) -> dict[str, Any]:
        """Analyze the best performing trials."""
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]

        if not complete_trials:
            return {}

        # Sort by value (assuming maximization - adjust if minimizing)
        if self.study.direction == optuna.study.StudyDirection.MAXIMIZE:
            best_trials = sorted(complete_trials, key=lambda x: x.value, reverse=True)[:top_n]
        else:
            best_trials = sorted(complete_trials, key=lambda x: x.value)[:top_n]

        # Extract common patterns in best trials
        best_params = {}
        best_metrics = {}

        for trial in best_trials:
            for param, value in trial.params.items():
                if param not in best_params:
                    best_params[param] = []
                best_params[param].append(value)

            for metric, value in trial.user_attrs.items():
                if metric not in best_metrics:
                    best_metrics[metric] = []
                best_metrics[metric].append(value)

        # Calculate statistics for best trials
        param_stats = {}
        for param, values in best_params.items():
            if all(isinstance(v, (int, float)) for v in values):
                param_stats[param] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                }
            else:
                # For categorical parameters
                from collections import Counter

                counter = Counter(values)
                param_stats[param] = {
                    "most_common": counter.most_common(),
                    "unique_count": len(counter),
                }

        metric_stats = {}
        for metric, values in best_metrics.items():
            if all(isinstance(v, (int, float)) and not math.isnan(v) for v in values):
                metric_stats[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                }

        return {
            "best_trials": [{"number": t.number, "value": t.value, "params": t.params, "metrics": t.user_attrs} for t in best_trials],
            "param_stats": param_stats,
            "metric_stats": metric_stats,
            "n_best": len(best_trials),
        }

    def _analyze_failures(self) -> dict[str, Any]:
        """Analyze failed trials to understand failure patterns."""
        failed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]

        if not failed_trials:
            return {"n_failed": 0, "failure_reasons": {}, "failure_patterns": {}}

        failure_reasons = {}
        failure_patterns = {}

        # Collect failure reasons
        for trial in failed_trials:
            reason = trial.user_attrs.get("failure_reason", "unknown")
            if reason not in failure_reasons:
                failure_reasons[reason] = 0
            failure_reasons[reason] += 1

        # Look for parameter patterns in failures
        failed_params = {}
        for trial in failed_trials:
            for param, value in trial.params.items():
                if param not in failed_params:
                    failed_params[param] = []
                failed_params[param].append(value)

        # Compare failed vs successful parameter ranges
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        complete_params = {}
        for trial in complete_trials:
            for param, value in trial.params.items():
                if param not in complete_params:
                    complete_params[param] = []
                complete_params[param].append(value)

        for param, failed_vals in failed_params.items():
            if param in complete_params:
                success_vals = complete_params[param]

                if all(isinstance(v, (int, float)) for v in failed_vals + success_vals):
                    failure_patterns[param] = {
                        "failed_mean": np.mean(failed_vals),
                        "success_mean": np.mean(complete_params[param]),
                        "failed_range": [np.min(failed_vals), np.max(failed_vals)],
                        "success_range": [np.min(success_vals), np.max(success_vals)],
                    }

        return {
            "n_failed": len(failed_trials),
            "failure_reasons": failure_reasons,
            "failure_patterns": failure_patterns,
        }


def load_study_from_db(study_name: str, storage_path: str) -> optuna.Study:
    """Load Optuna study from database."""
    try:
        return optuna.load_study(study_name=study_name, storage=storage_path)
    except Exception as e:
        logger.exception(f"Failed to load study {study_name}: {e}")
        raise


def load_study_from_directory(study_dir: str) -> tuple[optuna.Study, dict[str, Any]]:
    """Load study and config from a study directory."""
    study_path = Path(study_dir)

    # Look for database
    db_files = list(study_path.glob("*.db"))
    if not db_files:
        msg = f"No database file found in {study_dir}"
        raise FileNotFoundError(msg)

    storage_url = f"sqlite:///{db_files[0]}"

    # Extract study name from directory
    study_name = study_path.name.replace("optuna_report_", "")

    # Load config first to get study name if available
    config = {}
    json_files = list(study_path.glob("*summary*.json"))
    if json_files:
        import json

        with open(json_files[0]) as f:
            summary = json.load(f)
            config = summary.get("config", {})
            # Use study name from config if available
            if "study_name" in config:
                study_name = config["study_name"]

    # Load study with study name
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception:
        # If loading with study name fails, try to get available studies
        try:
            # Get all study names from the database
            study_summaries = optuna.get_all_study_summaries(storage=storage_url)
            if study_summaries:
                # Use the first (or only) study
                actual_study_name = study_summaries[0].study_name
                study = optuna.load_study(study_name=actual_study_name, storage=storage_url)
            else:
                msg = f"No studies found in database: {db_files[0]}"
                raise ValueError(msg)
        except Exception as e2:
            msg = f"Failed to load study from {db_files[0]}: {e2}"
            raise ValueError(msg)

    return study, config
