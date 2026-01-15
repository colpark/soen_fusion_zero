"""TensorBoard log reader utility.

Provides a clean API for reading TensorBoard event logs (tfevents files containing
metrics/scalars) into pandas DataFrames and exporting to CSV. Uses tbparse under
the hood, which works alongside PyTorch without environment conflicts.

Example usage:
    from soen_toolkit.utils.tb_reader import TBReader, read_tb_logs

    # Point at any level of your project hierarchy:
    # - A project root (reads all experiments/groups/repeats)
    reader = TBReader("experiments/project_MyProject")

    # - A specific group's logs directory
    reader = TBReader("experiments/project_MyProject/experiment_Foo/group_Bar/logs")

    # - A single repeat directory
    reader = TBReader("experiments/project_MyProject/.../logs/repeat_0")

    # Get scalar metrics as a DataFrame
    df = reader.scalars()  # Columns: step, tag, value, wall_time, dir_name, group, repeat

    # Filter by tag pattern (regex)
    accuracy_df = reader.scalars(tags="accuracy")
    loss_df = reader.scalars(tags=["loss", "perplexity"])

    # Summary info
    print(reader.summary())  # {'runs': [...], 'scalars': {'count': ..., 'tags': [...]}}
    print(reader.runs())     # List of run directories found
    print(reader.tags())     # List of metric tags

    # Export to CSV
    reader.to_csv("metrics.csv")
    reader.to_csv("output_dir/", per_run=True)  # One CSV per run
"""

from __future__ import annotations

import functools
from pathlib import Path
import re
from typing import TYPE_CHECKING

import pandas as pd
from tbparse import SummaryReader

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parse_dir_name(dir_name: str | None) -> tuple[str, str]:
    """Parse group and repeat from a dir_name path.

    Expected format: .../group_<name>/logs/<repeat>
    Returns: (group_name, repeat_name)
    """
    if not dir_name or pd.isna(dir_name):
        return "", ""

    dir_str = str(dir_name)

    # Extract group: look for 'group_<name>' pattern
    group_match = re.search(r"group_([^/]+)", dir_str)
    group = group_match.group(1) if group_match else ""

    # Extract repeat: everything after '/logs/'
    if "/logs/" in dir_str:
        repeat = dir_str.split("/logs/")[-1]
    else:
        # Fallback: use last path component
        repeat = Path(dir_str).name

    return group, repeat


class TBReader:
    """Reader for TensorBoard event logs.

    Wraps tbparse.SummaryReader with a cleaner API tailored to soen_toolkit's
    log directory structure. Includes caching to avoid repeated parsing.

    Args:
        path: Path to an event file, run directory, or parent directory
            containing multiple runs. Recursively discovers all event files.

    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"Path does not exist: {self._path}")

        # Create the underlying tbparse reader
        # Include extra columns for run identification and timestamps
        self._reader = SummaryReader(
            str(self._path),
            pivot=False,
            extra_columns={"dir_name", "wall_time", "file_name"},
        )

        # Cached DataFrames (populated on first access)
        self._scalars_cache: pd.DataFrame | None = None
        self._histograms_cache: pd.DataFrame | None = None

    @property
    def path(self) -> Path:
        """The root path being read."""
        return self._path

    def _get_scalars_raw(self) -> pd.DataFrame:
        """Get raw scalars DataFrame with caching."""
        if self._scalars_cache is not None:
            return self._scalars_cache

        df = self._reader.scalars
        if df is None or df.empty:
            self._scalars_cache = pd.DataFrame(
                columns=["step", "tag", "value", "wall_time", "dir_name", "file_name", "group", "repeat"]
            )
            return self._scalars_cache

        # Add parsed group and repeat columns
        parsed = df["dir_name"].apply(_parse_dir_name)
        df = df.copy()
        df["group"] = parsed.apply(lambda x: x[0])
        df["repeat"] = parsed.apply(lambda x: x[1])

        self._scalars_cache = df
        return self._scalars_cache

    def _get_histograms_raw(self) -> pd.DataFrame:
        """Get raw histograms DataFrame with caching."""
        if self._histograms_cache is not None:
            return self._histograms_cache

        df = self._reader.histograms
        if df is None or df.empty:
            self._histograms_cache = pd.DataFrame()
            return self._histograms_cache

        self._histograms_cache = df
        return self._histograms_cache

    def scalars(
        self,
        tags: str | Sequence[str] | None = None,
        step_min: int | None = None,
        step_max: int | None = None,
        groups: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Get scalar metrics as a DataFrame.

        Args:
            tags: Filter to specific tag(s). Supports regex patterns.
            step_min: Minimum step value (inclusive).
            step_max: Maximum step value (inclusive).
            groups: Filter to specific group(s). Supports regex patterns.

        Returns:
            DataFrame with columns: step, tag, value, wall_time, dir_name, file_name, group, repeat

        """
        df = self._get_scalars_raw()
        if df.empty:
            return df.copy()

        # Apply filters
        if tags is not None:
            df = self._filter_by_column(df, "tag", tags)

        if groups is not None:
            df = self._filter_by_column(df, "group", groups)

        if step_min is not None:
            df = df[df["step"] >= step_min]
        if step_max is not None:
            df = df[df["step"] <= step_max]

        return df.reset_index(drop=True)

    def histograms(
        self,
        tags: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Get histogram data as a DataFrame.

        Args:
            tags: Filter to specific tag(s). Supports regex patterns.

        Returns:
            DataFrame with histogram data.

        """
        df = self._get_histograms_raw()
        if df.empty:
            return df.copy()

        if tags is not None:
            df = self._filter_by_column(df, "tag", tags)

        return df.reset_index(drop=True)

    def images(
        self,
        tags: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Get image data as a DataFrame.

        Note: Requires tensorflow to be installed.

        Args:
            tags: Filter to specific tag(s). Supports regex patterns.

        Returns:
            DataFrame with image data (includes encoded image bytes).

        """
        try:
            df = self._reader.images
        except ModuleNotFoundError:
            # Image parsing requires tensorflow
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        if tags is not None:
            df = self._filter_by_column(df, "tag", tags)

        return df.reset_index(drop=True)

    def tensors(
        self,
        tags: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Get tensor data as a DataFrame.

        Note: Requires tensorflow to be installed.

        Args:
            tags: Filter to specific tag(s). Supports regex patterns.

        Returns:
            DataFrame with tensor data.

        """
        try:
            df = self._reader.tensors
        except ModuleNotFoundError:
            # Tensor parsing requires tensorflow
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        if tags is not None:
            df = self._filter_by_column(df, "tag", tags)

        return df.reset_index(drop=True)

    def hparams(self) -> pd.DataFrame:
        """Get hyperparameters as a DataFrame.

        Returns:
            DataFrame with hyperparameter configurations.

        """
        try:
            df = self._reader.hparams
        except ModuleNotFoundError:
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()
        return df.reset_index(drop=True)

    @functools.cached_property
    def _tags_scalars(self) -> list[str]:
        """Cached list of scalar tags."""
        df = self._get_scalars_raw()
        if df.empty or "tag" not in df.columns:
            return []
        return sorted(df["tag"].unique().tolist())

    @functools.cached_property
    def _runs_list(self) -> list[str]:
        """Cached list of runs."""
        df = self._get_scalars_raw()
        if df.empty or "dir_name" not in df.columns:
            return []
        unique_runs = df["dir_name"].dropna().unique().tolist()
        return sorted(str(r) for r in unique_runs if r)

    @functools.cached_property
    def _groups_list(self) -> list[str]:
        """Cached list of groups."""
        df = self._get_scalars_raw()
        if df.empty or "group" not in df.columns:
            return []
        unique_groups = df["group"].dropna().unique().tolist()
        return sorted(str(g) for g in unique_groups if g)

    def tags(self, data_type: str = "scalars") -> list[str]:
        """Get list of available tags.

        Args:
            data_type: One of "scalars", "histograms", "images", "tensors".

        Returns:
            List of unique tag names.

        """
        if data_type == "scalars":
            return self._tags_scalars

        # For other types, compute on demand (less common)
        df_map = {
            "histograms": self.histograms,
            "images": self.images,
            "tensors": self.tensors,
        }
        getter = df_map.get(data_type)
        if getter is None:
            return []

        df = getter()
        if df.empty or "tag" not in df.columns:
            return []
        return sorted(df["tag"].unique().tolist())

    def runs(self) -> list[str]:
        """Get list of run directory names.

        Returns:
            List of unique run identifiers (dir_name values).

        """
        return self._runs_list

    def groups(self) -> list[str]:
        """Get list of experiment groups.

        Returns:
            List of unique group names (extracted from dir_name).

        """
        return self._groups_list

    def to_csv(
        self,
        output: str | Path,
        per_run: bool = False,
        data_type: str = "scalars",
        **kwargs,
    ) -> Path | list[Path]:
        """Export data to CSV file(s).

        Args:
            output: Output file path, or directory if per_run=True.
            per_run: If True, write one CSV per run to the output directory.
            data_type: Data type to export: "scalars", "histograms", "tensors".
            **kwargs: Additional arguments passed to DataFrame.to_csv().

        Returns:
            Path to the written file, or list of paths if per_run=True.

        """
        output_path = Path(output).expanduser().resolve()

        # Get the appropriate data
        df_map = {
            "scalars": self.scalars,
            "histograms": self.histograms,
            "tensors": self.tensors,
        }
        if data_type not in df_map:
            raise ValueError(f"data_type must be one of {list(df_map.keys())}")

        df = df_map[data_type]()

        if df.empty:
            raise ValueError(f"No {data_type} data found to export")

        # Set sensible defaults for CSV export
        csv_kwargs = {"index": False}
        csv_kwargs.update(kwargs)

        if per_run:
            return self._export_per_run(df, output_path, data_type, csv_kwargs)
        else:
            # Write single CSV
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, **csv_kwargs)
            return output_path

    def _export_per_run(
        self,
        df: pd.DataFrame,
        output_path: Path,
        data_type: str,
        csv_kwargs: dict,
    ) -> list[Path]:
        """Export one CSV per run, handling filename collisions."""
        output_path.mkdir(parents=True, exist_ok=True)
        written_paths = []

        if "dir_name" not in df.columns or df["dir_name"].isna().all():
            # Single run - just use "run" as the name
            file_path = output_path / f"run_{data_type}.csv"
            df.to_csv(file_path, **csv_kwargs)
            written_paths.append(file_path)
            return written_paths

        # Track used filenames to avoid collisions
        used_names: dict[str, int] = {}

        for run_name in df["dir_name"].dropna().unique():
            run_df = df[df["dir_name"] == run_name]

            # Build filename from group and repeat for uniqueness
            group = run_df["group"].iloc[0] if "group" in run_df.columns else ""
            repeat = run_df["repeat"].iloc[0] if "repeat" in run_df.columns else ""

            if group and repeat:
                base_name = f"{group}_{repeat}"
            elif group:
                base_name = group
            elif repeat:
                base_name = repeat
            else:
                base_name = "run"

            # Sanitize for filesystem
            safe_name = re.sub(r"[^\w\-]", "_", base_name)

            # Handle collisions by appending counter
            if safe_name in used_names:
                used_names[safe_name] += 1
                safe_name = f"{safe_name}_{used_names[safe_name]}"
            else:
                used_names[safe_name] = 0

            file_path = output_path / f"{safe_name}_{data_type}.csv"
            run_df.to_csv(file_path, **csv_kwargs)
            written_paths.append(file_path)

        return written_paths

    def pivot(
        self,
        index: str = "step",
        columns: str = "tag",
        values: str = "value",
        data_type: str = "scalars",
    ) -> pd.DataFrame:
        """Pivot scalar data into wide format.

        Useful for comparing multiple metrics across steps.

        Args:
            index: Column to use as index (default: "step").
            columns: Column to pivot into columns (default: "tag").
            values: Column for cell values (default: "value").
            data_type: Data source (default: "scalars").

        Returns:
            Wide-format DataFrame with one column per tag.

        """
        df_map = {
            "scalars": self.scalars,
            "histograms": self.histograms,
            "tensors": self.tensors,
        }
        df = df_map.get(data_type, self.scalars)()

        if df.empty:
            return pd.DataFrame()

        # Handle multiple runs - include dir_name in index if present and has multiple values
        if "dir_name" in df.columns:
            unique_runs = df["dir_name"].dropna().nunique()
            if unique_runs > 1:
                return df.pivot_table(
                    index=["dir_name", index],
                    columns=columns,
                    values=values,
                    aggfunc="first",
                ).reset_index()

        return df.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc="first",
        ).reset_index()

    @functools.cached_property
    def _summary_data(self) -> dict:
        """Cached summary data."""
        scalars_df = self._get_scalars_raw()
        return {
            "path": str(self._path),
            "runs": self._runs_list,
            "groups": self._groups_list,
            "scalars": {
                "count": len(scalars_df),
                "tags": self._tags_scalars,
            },
            "histograms": {
                "count": len(self._get_histograms_raw()),
                "tags": self.tags("histograms"),
            },
            "images": {
                "count": len(self.images()),
                "tags": self.tags("images"),
            },
            "tensors": {
                "count": len(self.tensors()),
                "tags": self.tags("tensors"),
            },
            "hparams": {
                "count": len(self.hparams()),
            },
        }

    def summary(self) -> dict:
        """Get a summary of the data in this reader.

        Returns:
            Dict with counts and available tags for each data type.

        """
        return self._summary_data

    def _filter_by_column(
        self,
        df: pd.DataFrame,
        column: str,
        patterns: str | Sequence[str],
    ) -> pd.DataFrame:
        """Filter DataFrame by regex patterns on a column.

        Args:
            df: DataFrame to filter.
            column: Column name to filter on.
            patterns: Single pattern or list of patterns (regex).

        Returns:
            Filtered DataFrame.

        """
        if column not in df.columns:
            return df

        if isinstance(patterns, str):
            patterns = [patterns]

        # Build combined regex pattern
        compiled = [re.compile(p) for p in patterns]
        mask = df[column].apply(lambda x: any(p.search(str(x)) for p in compiled))
        return df[mask]

    def __repr__(self) -> str:
        return f"TBReader(path='{self._path}', runs={len(self._runs_list)}, scalars={len(self._get_scalars_raw())})"


def read_tb_logs(path: str | Path, **kwargs) -> pd.DataFrame:
    """Convenience function to read scalar metrics from TensorBoard event files.

    Args:
        path: Path to a project, experiment, group, or repeat directory containing
            tfevents files. Can be any level of the hierarchy - the reader will
            recursively find all event files.
        **kwargs: Arguments passed to TBReader.scalars() (e.g., tags, step_min, step_max, groups).

    Returns:
        DataFrame with columns: step, tag, value, wall_time, dir_name, file_name, group, repeat

    Example:
        df = read_tb_logs("experiments/project_MyProject", tags="val_accuracy")

    """
    return TBReader(path).scalars(**kwargs)
