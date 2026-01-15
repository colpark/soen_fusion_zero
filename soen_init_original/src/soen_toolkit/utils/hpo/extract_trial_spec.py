#!/usr/bin/env python3
"""Extract a specific trial's configuration as a YAML model spec."""

import argparse
import os
import sys

import optuna
import yaml

from .tools.criticality_data_processor import load_study_from_directory


def extract_trial_spec(study_dir: str, trial_number: int, output_path: str | None = None):
    """Extract a specific trial's configuration and save as YAML."""
    # Load study
    study, config = load_study_from_directory(study_dir)

    # Find the trial
    target_trial = None
    for trial in study.trials:
        if trial.number == trial_number:
            target_trial = trial
            break

    if target_trial is None:
        msg = f"Trial #{trial_number} not found in study {study.study_name}"
        raise ValueError(msg)

    if target_trial.state != optuna.trial.TrialState.COMPLETE:
        pass

    # Load base spec
    hp_config_path = config.get("hp_config", "criticality_runs/HPO_config.yaml")
    if not os.path.exists(hp_config_path):
        msg = f"HPO config not found: {hp_config_path}"
        raise FileNotFoundError(msg)

    with open(hp_config_path) as f:
        hp_config = yaml.safe_load(f) or {}

    paths_cfg = hp_config.get("paths", {})
    base_spec_path = paths_cfg.get("base_model_spec", "/Users/matthewcox/Documents/GreatSky_Local/Q2/soen_v2_studies/Studies/Criticality/ModelSpecs/100D_base.yaml")

    if not os.path.exists(base_spec_path):
        msg = f"Base model spec not found: {base_spec_path}"
        raise FileNotFoundError(msg)

    with open(base_spec_path) as f:
        base_spec = yaml.safe_load(f) or {}

    # Create a simplified spec with trial parameters
    trial_spec = base_spec.copy()

    # Add trial parameters as a metadata section
    trial_spec["trial_parameters"] = dict(target_trial.params)
    trial_spec["trial_metrics"] = dict(target_trial.user_attrs)

    # Set output path
    if output_path is None:
        output_path = f"trial_{trial_number}_spec.yaml"

    # Add trial metadata to the spec
    trial_spec["_trial_metadata"] = {
        "trial_number": trial_number,
        "objective_value": target_trial.value,
        "state": target_trial.state.name,
        "study_name": study.study_name,
        "user_attrs": dict(target_trial.user_attrs),
    }

    # Save the spec
    with open(output_path, "w") as f:
        yaml.dump(trial_spec, f, default_flow_style=False, indent=2)

    # Show key info

    if target_trial.user_attrs:
        for key, _value in sorted(target_trial.user_attrs.items()):
            if key in ["branching_sigma", "beta_temporal", "total_cost"]:
                pass

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract specific trial configuration")
    parser.add_argument("study_dir", help="Path to study directory")
    parser.add_argument("trial_number", type=int, help="Trial number to extract")
    parser.add_argument("--output", "-o", help="Output YAML file path")

    args = parser.parse_args()

    try:
        extract_trial_spec(args.study_dir, args.trial_number, args.output)
    except Exception:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
