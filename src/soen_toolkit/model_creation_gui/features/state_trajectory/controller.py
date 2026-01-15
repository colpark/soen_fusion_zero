"""Controller: orchestrates simulation runs without Qt dependencies."""

from __future__ import annotations

import logging
import traceback

import numpy as np
import torch

from soen_toolkit.physics.constants import get_omega_c

from .classification import (
    ClassificationResult,
    concatenate_histories,
    extract_final_states,
    get_prediction,
)
from .errors import SimulationError
from .inputs import (
    ColoredNoise,
    ConstantInput,
    DatasetInput,
    GaussianNoise,
    InputSource,
    Sinusoid,
    SquareWave,
)
from .padding import apply_append, apply_prepend
from .settings import InputKind, StateTrajSettings, TaskType, TimeMode
from .timebase import Timebase

logger = logging.getLogger(__name__)


class StateTrajectoryController:
    """Orchestrates simulation runs (no Qt dependencies).

    This controller coordinates:
    - Timebase calculation
    - Input generation
    - Padding application
    - Backend selection and execution

    It's designed to be testable independently of the GUI.
    """

    def __init__(self, model_manager, dataset_service, backends, model_adapter):
        """Initialize controller.

        Args:
            model_manager: ModelManager instance
            dataset_service: Service for dataset access
            backends: Dict[Backend, BackendRunner] mapping
            model_adapter: ModelAdapter for model operations
        """
        self.mgr = model_manager
        self.dataset = dataset_service
        self.backends = backends
        self.adapter = model_adapter

    def build_timebase(self, settings: StateTrajSettings) -> Timebase:
        """Construct timebase from settings.

        Args:
            settings: Simulation settings

        Returns:
            Timebase instance
        """
        # Get omega_c (could be model-specific in the future)
        omega_c = get_omega_c()

        if settings.time_mode == TimeMode.DT:
            return Timebase.from_dt(settings.dt, omega_c)
        else:
            return Timebase.from_total_ns(settings.total_time_ns, settings.seq_len, omega_c)

    def run(self, settings: StateTrajSettings) -> tuple[Timebase, torch.Tensor, list, list, float, list | None, object | None]:
        """Execute simulation with given settings.

        This is the main entry point for running a simulation. It:
        1. Builds the timebase
        2. Applies dt to model (single source of truth)
        3. Generates input
        4. Applies padding
        5. Runs selected backend

        Args:
            settings: Complete simulation settings

        Returns:
            Tuple of (timebase, input_features, metric_histories, raw_state_histories, elapsed_sec, fft_data, criticality_metrics)

        Raises:
            SimulationError: If model not available or simulation fails
        """
        if self.mgr.model is None:
            raise SimulationError("No model available. Build or load a model first.")

        # Save tracking state and enable full tracking to ensure all metrics are captured
        old_flags = self.adapter.get_tracking_flags(self.mgr.model)
        self.adapter.enable_full_tracking(self.mgr.model)

        try:
            # 1. Build timebase
            tb = self.build_timebase(settings)

            # 2. Apply dt to model ONCE (single source of truth)
            self.adapter.set_dt(self.mgr.model, tb.dt)

            # 3. Generate input
            expected_dim = self.mgr.model.layers_config[0].params.get("dim")
            if expected_dim is None:
                raise SimulationError("Could not determine model input dimension")

            input_source = self._make_input_source(settings)
            rng = self.dataset.rng() if self.dataset else np.random.default_rng(1337)

            x = input_source.make(settings.seq_len, expected_dim, tb, rng)

            # 4. Apply padding
            x = apply_prepend(x, settings.prepend, settings.time_mode, tb)
            x = apply_append(x, settings.append, settings.time_mode, tb)

            # Add batch dimension
            x = x.unsqueeze(0)

            # 5. Run backend
            runner = self.backends.get(settings.backend)
            if runner is None:
                raise SimulationError(f"Backend {settings.backend} not available")

            try:
                logger.info(f"Running {settings.backend.value} simulation with seq_len={settings.seq_len}, dt={tb.dt:.6e}")
                metric_h, raw_h, elapsed = runner.run(x, settings.metric.value, settings.display.include_s0)
                logger.info(f"Simulation completed successfully in {elapsed:.4f}s")

                # Compute FFT data if needed
                fft_data = None
                from .settings import ViewMode
                if settings.fft.view_mode != ViewMode.TIME_ONLY:
                    fft_data = self._compute_fft_data(raw_h, tb.step_ns, settings.fft)

                # Compute Criticality if requested
                crit_metrics = None
                if settings.calculate_criticality:
                    try:
                        from soen_toolkit.utils.metrics import quantify_criticality
                        # Pass cached states directly to avoid re-running, AND inputs for Lyapunov calc
                        crit_metrics = quantify_criticality(self.mgr.model, inputs=x, states=raw_h)
                    except Exception as e:
                        logger.error(f"Criticality calculation failed: {e}")

                return tb, x, metric_h, raw_h, elapsed, fft_data, crit_metrics
            except Exception as e:
                logger.error("Simulation failed in controller.run()", exc_info=True)
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                logger.error(f"Simulation settings: backend={settings.backend}, metric={settings.metric}, seq_len={settings.seq_len}, dt={tb.dt}")
                raise
        finally:
            # Restore tracking state
            self.adapter.set_tracking_flags(self.mgr.model, old_flags)

    def _make_input_source(self, settings: StateTrajSettings) -> InputSource:
        """Create input source from settings.

        Args:
            settings: Settings containing input configuration

        Returns:
            InputSource instance

        Raises:
            SimulationError: If input kind not supported or dataset not loaded
        """
        kind = settings.input_kind
        params = settings.input_params

        if kind == InputKind.CONSTANT:
            return ConstantInput(value=params.constant_val)

        elif kind == InputKind.GAUSSIAN:
            return GaussianNoise(std=params.noise_std)

        elif kind == InputKind.COLORED:
            return ColoredNoise(beta=params.colored_beta)

        elif kind == InputKind.SINE:
            return Sinusoid(
                freq_mhz=params.sine_freq_mhz,
                amp=params.sine_amp,
                phase_deg=params.sine_phase_deg,
                offset=params.sine_offset,
            )

        elif kind == InputKind.SQUARE:
            return SquareWave(
                freq_mhz=params.sine_freq_mhz,
                amp=params.sine_amp,
                phase_deg=params.sine_phase_deg,
                offset=params.sine_offset,
            )

        elif kind == InputKind.DATASET:
            if self.dataset is None or not self.dataset.is_loaded():
                raise SimulationError("Dataset not loaded. Please load a dataset first.")

            # Get dataset indices for sample selection
            indices = self.dataset.get_indices_for_class(settings.class_id, settings.task_type)
            if not indices:
                raise SimulationError(f"No samples found for class {settings.class_id}")

            # Clamp sample index to valid range
            sample_idx = min(settings.sample_index, len(indices) - 1)
            data_idx = indices[sample_idx]

            return DatasetInput(
                dataset=self.dataset.get_dataset(),
                data_index=data_idx,
                encoding=settings.encoding,
                scaling=settings.scaling,
                target_length=settings.seq_len,
            )

        else:
            raise SimulationError(f"Unknown input kind: {kind}")

    def run_classification(self, settings: StateTrajSettings) -> tuple[Timebase, ClassificationResult, float]:
        """Run multiple samples with state carryover for classification visualization.

        This method:
        1. Randomly selects N samples from different classes
        2. Runs each sample sequentially, carrying forward ALL final states
        3. Applies time pooling and computes predictions
        4. Returns concatenated results with sample boundaries

        Args:
            settings: Complete simulation settings with classification config

        Returns:
            Tuple of (timebase, classification_result, total_elapsed_sec)

        Raises:
            SimulationError: If model not available, dataset not loaded, or sim fails
        """
        if self.mgr.model is None:
            raise SimulationError("No model available. Build or load a model first.")

        if not settings.classification.enabled:
            raise SimulationError("Classification mode not enabled in settings")

        if settings.input_kind != InputKind.DATASET:
            raise SimulationError("Classification mode only works with dataset input")

        if self.dataset is None or not self.dataset.is_loaded():
            raise SimulationError("Dataset not loaded for classification mode")

        # Save tracking state and enable full tracking
        old_flags = self.adapter.get_tracking_flags(self.mgr.model)
        self.adapter.enable_full_tracking(self.mgr.model)

        try:
            # Build timebase (same for all samples)
            tb = self.build_timebase(settings)
            self.adapter.set_dt(self.mgr.model, tb.dt)

            # Get backend
            runner = self.backends.get(settings.backend)
            if runner is None:
                raise SimulationError(f"Backend {settings.backend} not available")

            # Get expected input dimension
            expected_dim = self.mgr.model.layers_config[0].params.get("dim")
            if expected_dim is None:
                raise SimulationError("Could not determine model input dimension")

            # Get number of classes from dataset
            dataset_obj = self.dataset.get_dataset()
            if settings.task_type != TaskType.CLASSIFICATION:
                raise SimulationError("Classification mode requires classification task type")

            # Get num_classes from dataset labels (primary) or model output dim (fallback)
            num_classes = None
            try:
                if hasattr(dataset_obj, 'labels') and dataset_obj.labels is not None:
                    num_classes = len(set(dataset_obj.labels))
            except Exception as e:
                logger.warning(f"Could not get num_classes from dataset labels: {e}")

            if num_classes is None:
                # Fallback: use model's output layer dimension
                try:
                    output_layer_cfg = self.mgr.model.layers_config[-1]
                    num_classes = output_layer_cfg.params.get("dim")
                except Exception as e:
                    raise SimulationError(f"Could not determine num_classes from dataset or model: {e}") from e

            if num_classes is None or num_classes <= 0:
                raise SimulationError("Could not determine valid num_classes for classification")

            # Select random samples from different classes
            rng = self.dataset.rng()
            num_samples = settings.classification.num_samples
            selected_samples = []

            for _ in range(num_samples):
                # Pick random class
                class_id = rng.integers(0, num_classes)

                # Get indices for this class
                indices = self.dataset.get_indices_for_class(class_id, settings.task_type)
                if not indices:
                    # Skip if no samples for this class
                    continue

                # Pick random sample from this class
                sample_idx = rng.integers(0, len(indices))
                data_idx = indices[sample_idx]

                selected_samples.append((class_id, data_idx))

            if not selected_samples:
                raise SimulationError("No valid samples found")

            # Run samples sequentially with state carryover
            all_inputs = []
            all_metric_histories = []
            all_raw_histories = []
            sample_classes = []
            sample_boundaries = [0]  # Start of first sample
            predictions = []
            correct = []
            total_elapsed = 0.0

            initial_main_states: dict[int, torch.Tensor] | None = None
            initial_s1_states: dict[int, torch.Tensor] | None = None
            initial_s2_states: dict[int, torch.Tensor] | None = None

            for sample_idx, (class_id, data_idx) in enumerate(selected_samples):
                logger.info(f"Running classification sample {sample_idx + 1}/{len(selected_samples)}: class={class_id}")

                # Generate input for this sample
                input_source = DatasetInput(
                    dataset=dataset_obj,
                    data_index=data_idx,
                    encoding=settings.encoding,
                    scaling=settings.scaling,
                    target_length=settings.seq_len,
                )

                x = input_source.make(settings.seq_len, expected_dim, tb, rng)

                # Apply padding
                x = apply_prepend(x, settings.prepend, settings.time_mode, tb)
                x = apply_append(x, settings.append, settings.time_mode, tb)

                # Add batch dimension
                x = x.unsqueeze(0)  # [1, T, D]

                # Run with initial states from previous sample
                try:
                    metric_h, raw_h, elapsed = runner.run(
                        x,
                        settings.metric.value,
                        settings.display.include_s0,
                        initial_states=initial_main_states,
                        s1_states=initial_s1_states,
                        s2_states=initial_s2_states,
                    )
                    total_elapsed += elapsed
                except Exception as e:
                    logger.error(f"Failed to run sample {sample_idx}: {e}")
                    raise

                # Extract final states for next sample
                next_main: dict[int, torch.Tensor] = {}
                next_s1: dict[int, torch.Tensor] = {}
                next_s2: dict[int, torch.Tensor] = {}

                if hasattr(runner, "get_last_states"):
                    try:
                        next_main, next_s1, next_s2 = runner.get_last_states()
                    except Exception:
                        next_main = {}
                        next_s1 = {}
                        next_s2 = {}

                if not next_main:
                    next_main = extract_final_states(raw_h, self.mgr.model.layers_config)

                initial_main_states = {k: v.clone().detach() for k, v in next_main.items()} if next_main else None
                initial_s1_states = {k: v.clone().detach() for k, v in next_s1.items()} if next_s1 else None
                initial_s2_states = {k: v.clone().detach() for k, v in next_s2.items()} if next_s2 else None

                # Get prediction for this sample
                output_states = raw_h[-1]  # Last layer states [1, T+1, num_classes]
                pred_class = get_prediction(output_states, settings.classification.pooling_method)
                is_correct = pred_class == class_id

                # Store results
                all_inputs.append(x)
                all_metric_histories.append(metric_h)
                all_raw_histories.append(raw_h)
                sample_classes.append(class_id)
                predictions.append(pred_class)
                correct.append(is_correct)

                # Record boundary for next sample (cumulative time index)
                if sample_idx < len(selected_samples) - 1:
                    current_t = x.shape[1]  # Time steps in this sample
                    prev_boundary = sample_boundaries[-1]
                    sample_boundaries.append(prev_boundary + current_t)

            # Concatenate all histories along time dimension
            concatenated_metric = concatenate_histories(all_metric_histories)
            concatenated_raw = concatenate_histories(all_raw_histories)
            concatenated_input = torch.cat(all_inputs, dim=1)  # [1, total_T, D]

            # Create result object
            result = ClassificationResult(
                input_features=concatenated_input,
                metric_histories=concatenated_metric,
                raw_state_histories=concatenated_raw,
                sample_classes=sample_classes,
                sample_boundaries=sample_boundaries,
                predictions=predictions,
                correct=correct,
            )

            logger.info(f"Classification run completed: {sum(correct)}/{len(correct)} correct")

            return tb, result, total_elapsed

        finally:
            # Restore tracking state
            self.adapter.set_tracking_flags(self.mgr.model, old_flags)

    def _compute_fft_data(
        self,
        raw_histories: list[torch.Tensor],
        dt_ns: float,
        fft_settings,
    ) -> list[tuple[np.ndarray, np.ndarray, list | None]]:
        """Compute FFT data for all layers.

        Args:
            raw_histories: Per-layer state histories [batch, T+1, dim]
            dt_ns: Time step in nanoseconds
            fft_settings: FFT configuration

        Returns:
            List of (freqs_mhz, magnitudes, peaks) tuples per layer
        """
        from .fft_analysis import FFTAnalysisService

        fft_service = FFTAnalysisService()
        fft_data = []

        for hist in raw_histories:
            # Convert to numpy [T, dim]
            arr = hist[0].detach().cpu().numpy()

            # Compute spectrum
            freqs_mhz, magnitudes = fft_service.compute_spectrum(
                arr,
                dt_ns,
                fft_settings.window_function,
                fft_settings.aggregation_mode,
                fft_settings.remove_dc,
                fft_settings.normalize,
                fft_settings.y_scale,
            )

            # Find peaks if enabled
            peaks = None
            if fft_settings.show_peaks:
                try:
                    peaks = fft_service.find_peaks(
                        freqs_mhz,
                        magnitudes,
                        fft_settings.num_peaks,
                        min_prominence=10.0,  # 10 dB
                    )
                except Exception:
                    peaks = None  # Non-fatal if peak finding fails

            fft_data.append((freqs_mhz, magnitudes, peaks))

        return fft_data
