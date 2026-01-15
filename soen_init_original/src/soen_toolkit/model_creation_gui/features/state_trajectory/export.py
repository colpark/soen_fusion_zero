"""Export service with atomic saves and metadata."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from .settings import StateTrajSettings
from .timebase import Timebase


class ExportService:
    """Atomic export with metadata for reproducibility."""

    def export_sequences(
        self,
        input_seq: torch.Tensor,
        output_seq: torch.Tensor,
        model,
        settings: StateTrajSettings,
        timebase: Timebase,
        output_path: Path,
    ) -> tuple[Path, Path, Path, Path]:
        """Export sequences, model weights, and metadata atomically.

        Writes to temporary files first, then atomically moves to final paths
        to ensure partial writes don't leave corrupted data.

        Args:
            input_seq: Input sequence tensor [seq_len, dim]
            output_seq: Output sequence tensor [seq_len, dim]
            model: Model instance
            settings: Settings used for simulation
            timebase: Timebase for metadata
            output_path: Base output path (without extension)

        Returns:
            Tuple of (sequences_path, weights_path, metadata_path, info_path)

        Raises:
            IOError: If export fails
        """
        base_path = Path(output_path)
        base_dir = base_path.parent
        base_name = base_path.stem

        # Ensure output directory exists
        base_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        sequences_path = base_dir / f"{base_name}_sequences.npz"
        weights_path = base_dir / f"{base_name}_weights.pth"
        metadata_path = base_dir / f"{base_name}_metadata.json"
        info_path = base_dir / f"{base_name}_info.txt"

        # Prepare data
        input_np = input_seq.detach().cpu().numpy()
        output_np = output_seq.detach().cpu().numpy()

        # Create metadata
        metadata = self._create_metadata(settings, timebase, input_np, output_np)

        # Write to temporary files first (atomic writes)
        try:
            # 1. Sequences (npz with named arrays)
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=base_dir) as tmp_seq:
                np.savez_compressed(tmp_seq, input=input_np, output=output_np)
                tmp_seq_path = tmp_seq.name
            os.replace(tmp_seq_path, sequences_path)

            # 2. Model weights
            with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=base_dir) as tmp_wgt:
                torch.save(model.state_dict(), tmp_wgt.name)
                tmp_wgt_path = tmp_wgt.name
            os.replace(tmp_wgt_path, weights_path)

            # 3. Metadata JSON
            with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=base_dir, encoding="utf-8") as tmp_meta:
                json.dump(metadata, tmp_meta, indent=2)
                tmp_meta_path = tmp_meta.name
            os.replace(tmp_meta_path, metadata_path)

            # 4. Human-readable info text
            info_text = self._create_info_text(metadata)
            with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=base_dir, encoding="utf-8") as tmp_info:
                tmp_info.write(info_text)
                tmp_info_path = tmp_info.name
            os.replace(tmp_info_path, info_path)

            return sequences_path, weights_path, metadata_path, info_path

        except Exception as e:
            # Clean up any partial temporary files
            for tmp_path in [tmp_seq_path, tmp_wgt_path, tmp_meta_path, tmp_info_path]:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass
            raise OSError(f"Export failed: {e}") from e

    def _create_metadata(
        self,
        settings: StateTrajSettings,
        timebase: Timebase,
        input_np: np.ndarray,
        output_np: np.ndarray,
    ) -> dict:
        """Create metadata dictionary for export."""
        return {
            "export_timestamp": datetime.now().isoformat(),
            "simulation": {
                "metric": settings.metric.value,
                "backend": settings.backend.value,
                "dt_dimensionless": timebase.dt,
                "omega_c_rad_per_s": timebase.omega_c,
                "step_ns": timebase.step_ns,
                "sampling_rate_hz": timebase.sampling_rate_hz,
                "nyquist_hz": timebase.nyquist_hz,
            },
            "input": {
                "kind": settings.input_kind.value,
                "seq_len": int(input_np.shape[0]),
                "dim": int(input_np.shape[1]),
                "shape": list(input_np.shape),
                "dtype": str(input_np.dtype),
            },
            "output": {
                "seq_len": int(output_np.shape[0]),
                "dim": int(output_np.shape[1]),
                "shape": list(output_np.shape),
                "dtype": str(output_np.dtype),
            },
            "padding": {
                "prepend_enabled": settings.prepend.enabled,
                "prepend_steps": settings.prepend.count_steps,
                "prepend_ns": settings.prepend.time_ns,
                "append_enabled": settings.append.enabled,
                "append_steps": settings.append.count_steps,
                "append_ns": settings.append.time_ns,
                "append_mode": settings.append.mode,
            },
            "dataset": {
                "path": settings.dataset_path,
                "group": settings.group,
                "task_type": settings.task_type.value,
            }
            if settings.input_kind.value == "dataset"
            else None,
            "encoding": {
                "mode": settings.encoding.mode,
                "vocab_size": settings.encoding.vocab_size,
            }
            if settings.encoding.mode != "raw"
            else None,
        }

    def _create_info_text(self, metadata: dict) -> str:
        """Create human-readable info text from metadata."""
        lines = [
            "=" * 60,
            "State Trajectory Export",
            "=" * 60,
            "",
            f"Exported: {metadata['export_timestamp']}",
            "",
            "Simulation Parameters:",
            f"  Metric: {metadata['simulation']['metric']}",
            f"  Backend: {metadata['simulation']['backend']}",
            f"  dt (dimensionless): {metadata['simulation']['dt_dimensionless']}",
            f"  ω_c (rad/s): {metadata['simulation']['omega_c_rad_per_s']:.6e}",
            f"  Step time: {metadata['simulation']['step_ns']:.6f} ns",
            f"  Sampling rate: {metadata['simulation']['sampling_rate_hz']:.6e} Hz",
            f"  Nyquist frequency: {metadata['simulation']['nyquist_hz']:.6e} Hz",
            "",
            "Input Sequence:",
            f"  Kind: {metadata['input']['kind']}",
            f"  Shape: {metadata['input']['shape']}",
            f"  Length: {metadata['input']['seq_len']} steps",
            f"  Dimensions: {metadata['input']['dim']}",
            f"  Total time: {metadata['input']['seq_len'] * metadata['simulation']['step_ns']:.3f} ns",
            "",
            "Output Sequence:",
            f"  Shape: {metadata['output']['shape']}",
            f"  Length: {metadata['output']['seq_len']} steps",
            f"  Dimensions: {metadata['output']['dim']}",
            f"  Total time: {metadata['output']['seq_len'] * metadata['simulation']['step_ns']:.3f} ns",
            "",
        ]

        # Add padding info if enabled
        padding = metadata.get("padding", {})
        if padding.get("prepend_enabled") or padding.get("append_enabled"):
            lines.append("Padding:")
            if padding.get("prepend_enabled"):
                lines.append(f"  Prepended: {padding['prepend_steps']} steps ({padding['prepend_ns']:.3f} ns)")
            if padding.get("append_enabled"):
                lines.append(f"  Appended: {padding['append_steps']} steps ({padding['append_ns']:.3f} ns) [{padding['append_mode']}]")
            lines.append("")

        # Add dataset info if available
        if metadata.get("dataset"):
            ds = metadata["dataset"]
            lines.append("Dataset:")
            lines.append(f"  Path: {ds['path']}")
            if ds.get("group"):
                lines.append(f"  Group: {ds['group']}")
            lines.append(f"  Task: {ds['task_type']}")
            lines.append("")

        # Add encoding info if available
        if metadata.get("encoding"):
            enc = metadata["encoding"]
            lines.append("Encoding:")
            lines.append(f"  Mode: {enc['mode']}")
            lines.append(f"  Vocab size: {enc['vocab_size']}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("Files:")
        lines.append("  *_sequences.npz  - Input/output arrays (use np.load)")
        lines.append("  *_weights.pth    - Model state dict (use torch.load)")
        lines.append("  *_metadata.json  - Machine-readable metadata")
        lines.append("  *_info.txt       - This file")
        lines.append("=" * 60)

        return "\n".join(lines)
