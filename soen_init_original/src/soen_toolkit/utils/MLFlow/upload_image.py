#!/usr/bin/env python3
"""Upload any file to a specific MLflow run.

Usage:
    python upload_file.py <run_id> <file_path> [artifact_folder]

Examples:
    python upload_file.py 24c019355bb04ef0ba0fdcd32510d8fa ~/Desktop/results.png plots
    python upload_file.py 24c019355bb04ef0ba0fdcd32510d8fa ~/data/config.yaml configs
    python upload_file.py 24c019355bb04ef0ba0fdcd32510d8fa ~/results/analysis.pdf reports

"""

import os
from pathlib import Path
import sys

import mlflow


def upload_file(run_id: str, file_path: str, artifact_folder: str = "uploads") -> bool | None:
    """Upload any file to MLflow run as artifact."""
    # Set up MLflow connection (keep default URI; require username/password)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "https://mlflow-greatsky.duckdns.org")
    if not os.environ.get("MLFLOW_TRACKING_USERNAME") or not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        return False

    # Validate inputs
    file_path_obj = Path(file_path).expanduser()
    if not file_path_obj.exists():
        return False

    try:
        client = mlflow.tracking.MlflowClient()

        # Check if run exists
        try:
            client.get_run(run_id)
        except Exception:
            return False

        # Upload the file
        client.log_artifact(run_id, str(file_path_obj), artifact_folder)

        # Show how to reference it (different for images vs other files)

        # Show appropriate reference syntax
        if file_path_obj.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"]:
            pass
        else:
            pass

        return True

    except Exception:
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)

    run_id = sys.argv[1]
    image_path = sys.argv[2]
    artifact_folder = sys.argv[3] if len(sys.argv) > 3 else "images"

    success = upload_file(run_id, image_path, artifact_folder)
    sys.exit(0 if success else 1)
