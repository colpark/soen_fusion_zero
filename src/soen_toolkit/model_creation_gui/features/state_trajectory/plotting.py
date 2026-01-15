"""Plotting renderer for state trajectory visualization."""

from __future__ import annotations

import contextlib

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QSplitter, QWidget
import pyqtgraph as pg
import torch

from soen_toolkit.physics.constants import DEFAULT_IC

from .classification import ClassificationResult
from .settings import ClassificationSettings, DisplaySettings, FFTSettings, Metric, TimeMode, ViewMode, ZeroPaddingSpec
from .timebase import Timebase


def _format_sim_time(elapsed: float) -> str:
    """Convert elapsed seconds to a human-friendly string with appropriate units."""
    if elapsed >= 1.0:
        return f"{elapsed:.3f} s"
    ms = elapsed * 1e3
    if ms >= 1.0:
        return f"{ms:.3f} ms"
    us = elapsed * 1e6
    if us >= 1.0:
        return f"{us:.3f} µs"
    ns = elapsed * 1e9
    return f"{ns:.3f} ns"


class PlotRenderer:
    """Pure plotting logic: arrays → plots.

    Renders layer-wise plots with proper axes, legends, and overlays.
    Stateless and testable independently of Qt event loops.
    """

    def render_layer_plots(
        self,
        histories: list[torch.Tensor],
        raw_state_histories: list[torch.Tensor],
        metric: Metric,
        timebase: Timebase,
        elapsed: float,
        layers_config: list,
        tabs_widget: pg.QtWidgets.QTabWidget,
        display_settings: DisplaySettings,
        fft_settings: FFTSettings | None = None,
        fft_data: list[tuple[np.ndarray, np.ndarray, list | None]] | None = None,
        dataset=None,
        data_index: int | None = None,
        is_seq2seq: bool = False,
        prepend_spec: ZeroPaddingSpec | None = None,
        append_spec: ZeroPaddingSpec | None = None,
        time_mode: TimeMode = TimeMode.DT,
        classification_result: ClassificationResult | None = None,
        classification_settings: ClassificationSettings | None = None,
    ) -> None:
        """Render all layer plots into tabs.

        Args:
            histories: Per-layer metric histories (format depends on metric)
            raw_state_histories: Per-layer state histories [batch, T+1, dim]
            metric: Selected metric for plotting
            timebase: Timebase for time axis conversions
            elapsed: Simulation elapsed time in seconds
            layers_config: Layer configuration list
            tabs_widget: Qt tab widget to populate
            display_settings: Display options (legend, total, etc.)
            fft_settings: FFT configuration (optional)
            fft_data: FFT data per layer (optional)
            dataset: Optional dataset for target overlay
            data_index: Optional dataset index for target overlay
            is_seq2seq: Whether task is seq2seq (for target overlay)
            prepend_spec: Padding spec for prepended zeros
            append_spec: Padding spec for appended zeros
            time_mode: Time mode for padding calculations
            classification_result: Optional ClassificationResult for multi-sample mode
            classification_settings: Display settings for classification visualization
        """
        # Save current tab index before clearing
        current_tab = tabs_widget.currentIndex()
        tabs_widget.clear()

        # Metric-specific y-axis label base
        y_label_map = {
            Metric.STATE: "s(t)",
            Metric.PHI: "phi(t)",
            Metric.G: "g(t)",
            Metric.POWER: "P(t) [nW]",
            Metric.ENERGY: "E(t) [nJ]",
        }
        y_label_base = y_label_map.get(metric, "Value")

        # Iterate over layers and create plots
        for idx, (cfg, hist) in enumerate(zip(layers_config, histories, strict=False)):
            layer_id = cfg.layer_id
            layer_type = getattr(cfg, "layer_type", None)

            if hist is None:
                # Create placeholder for missing data
                pw = self._create_placeholder_plot(layer_id)
                tabs_widget.addTab(pw, f"Layer {layer_id}")
                continue

            # Choose source history based on metric
            if metric == Metric.STATE:
                hist_source = hist  # Already filtered by include_s0 in backend
            else:
                hist_source = hist

            # Convert to numpy array [T, dim]
            hist_cpu = hist_source[0].detach().cpu()  # Remove batch dim
            arr = hist_cpu.numpy()

            # Create plot widget based on FFT view mode
            if fft_settings and fft_settings.view_mode != ViewMode.TIME_ONLY and fft_data:
                pw = self._create_fft_view_widget(
                    arr=arr,
                    layer_id=layer_id,
                    layer_type=layer_type,
                    metric=metric,
                    y_label_base=y_label_base,
                    timebase=timebase,
                    display_settings=display_settings,
                    elapsed=elapsed,
                    prepend_spec=prepend_spec,
                    append_spec=append_spec,
                    time_mode=time_mode,
                    fft_settings=fft_settings,
                    fft_tuple=fft_data[idx] if idx < len(fft_data) else None,
                )
            else:
                # Time only view (default)
                pw = self._create_plot_widget(
                    arr=arr,
                    layer_id=layer_id,
                    layer_type=layer_type,
                    metric=metric,
                    y_label_base=y_label_base,
                    timebase=timebase,
                    display_settings=display_settings,
                    elapsed=elapsed,
                    prepend_spec=prepend_spec,
                    append_spec=append_spec,
                    time_mode=time_mode,
                )

            # Overlay targets for output layer (only if pw is a PlotWidget)
            if isinstance(pw, pg.PlotWidget):
                if (
                    classification_result is None
                    and display_settings.overlay_targets
                    and idx == len(layers_config) - 1
                    and dataset is not None
                    and data_index is not None
                ):
                    if is_seq2seq:
                        self._overlay_targets_seq2seq(pw, dataset, data_index, arr, timebase, display_settings, prepend_spec, time_mode)
                    else:
                        # Classification: show expected class with horizontal reference line
                        self._overlay_targets_classification(pw, dataset, data_index, arr, timebase, display_settings)

                if classification_result is not None:
                    self._overlay_classification_annotations(
                        pw,
                        classification_result,
                        arr,
                        layer_index=idx,
                        total_layers=len(layers_config),
                        show_accuracy=(classification_settings.show_accuracy if classification_settings else True),
                    )

            tabs_widget.addTab(pw, f"Layer {layer_id}")

        # Restore tab selection if possible
        if current_tab >= 0 and current_tab < tabs_widget.count():
            tabs_widget.setCurrentIndex(current_tab)

    def _create_placeholder_plot(self, layer_id: int) -> pg.PlotWidget:
        """Create placeholder plot for missing data."""
        pw = pg.PlotWidget()
        pw.setBackground("w")
        txt = pg.TextItem("No data for this layer", color=(200, 0, 0), anchor=(0.5, 0.5))
        txt.setFont(QFont("", 14))
        txt.setPos(0, 0)
        pw.addItem(txt)
        return pw

    def _create_plot_widget(
        self,
        arr: np.ndarray,
        layer_id: int,
        layer_type: str | None,
        metric: Metric,
        y_label_base: str,
        timebase: Timebase,
        display_settings: DisplaySettings,
        elapsed: float,
        prepend_spec: ZeroPaddingSpec | None = None,
        append_spec: ZeroPaddingSpec | None = None,
        time_mode: TimeMode = TimeMode.DT,
    ) -> pg.PlotWidget:
        """Create a single plot widget for a layer.

        Args:
            arr: Data array [T, dim]
            layer_id: Layer identifier
            layer_type: Layer type string (e.g., "SingleDendrite", "Multiplier", etc.)
            metric: Metric being plotted
            y_label_base: Base Y-axis label (without units)
            timebase: Timebase for time axis
            display_settings: Display options
            elapsed: Simulation time in seconds
            prepend_spec: Padding spec for prepended zeros
            append_spec: Padding spec for appended zeros
            time_mode: Time mode for padding calculations

        Returns:
            Configured PlotWidget
        """
        # Check if this is a physical layer that should be converted to current units
        # Only convert STATE metric for: SingleDendrite, Multiplier (WICC/v1), MultiplierNOCC (NOCC/v2)
        # Exclude DendriteReadout as it's voltage with different normalization
        # Handle both "Multiplier" and "MultiplierWICC" (legacy alias)
        should_convert_to_current = metric == Metric.STATE and layer_type in ("SingleDendrite", "Multiplier", "MultiplierWICC", "MultiplierNOCC")

        if should_convert_to_current:
            # Convert dimensionless state to current in microamperes
            # I [uA] = s (dimensionless) * I_c [A] * 1e6 [uA/A]
            arr = arr * DEFAULT_IC * 1e6
            y_label = "I(t) [uA]"
        else:
            y_label = y_label_base

        pw = pg.PlotWidget()
        pw.setBackground("w")
        pw.showGrid(x=True, y=True)

        # Compute time axis shift based on metric
        if metric == Metric.STATE:
            shift_steps = 0 if display_settings.include_s0 else 1
        elif metric == Metric.ENERGY:
            shift_steps = 1  # Energy accumulated from first step
        else:
            shift_steps = 0

        # Calculate prepend steps for time axis offset
        prepend_steps = 0
        if prepend_spec and prepend_spec.enabled:
            if time_mode == TimeMode.DT:
                prepend_steps = prepend_spec.count_steps
            elif timebase.step_ns > 0:
                prepend_steps = int(round(prepend_spec.time_ns / timebase.step_ns))

        # Build time axis in nanoseconds
        # If prepended zeros exist, start time axis at negative time
        T_plot = arr.shape[0]
        dt_ns = timebase.step_ns
        time_axis = [(i + shift_steps - prepend_steps) * dt_ns for i in range(T_plot)]

        # Set axis labels with styling
        label_style = {"color": "#000", "font-size": "12pt"}
        pw.setLabel("left", f"{y_label} - Layer {layer_id}", **label_style)

        t_end = time_axis[-1] if len(time_axis) > 0 else 0.0
        pw.setLabel("bottom", f"Time (ns) [Δt={dt_ns:.3f} ns, end≈{t_end:.3f} ns]", **label_style)

        # Store axis metadata for downstream overlays
        pw._time_axis = time_axis
        pw._shift_steps = shift_steps
        pw._prepend_steps = prepend_steps
        pw._dt_ns = dt_ns

        # Style axis ticks
        tick_font = QFont()
        tick_font.setPointSize(10)
        for ax_name in ("left", "bottom"):
            ax = pw.getAxis(ax_name)
            ax.setPen(pg.mkPen(color="k", width=1))
            ax.setTextPen("k")
            with contextlib.suppress(Exception):
                ax.setStyle(tickFont=tick_font)

        # Add legend ONLY if needed (performance optimization)
        if display_settings.show_legend or display_settings.show_total:
            pw.addLegend()

        # Plot individual neuron traces
        for neuron in range(arr.shape[1]):
            color = pg.intColor(neuron, arr.shape[1], alpha=200)
            name = f"N{neuron}" if display_settings.show_legend else None
            pw.plot(
                time_axis,
                arr[:, neuron],
                pen=pg.mkPen(color=color, width=2),
                name=name,
            )

        # Plot total line if enabled
        if display_settings.show_total:
            total_series = arr.sum(axis=1)
            pw.plot(
                time_axis,
                total_series,
                pen=pg.mkPen(color="k", width=4),
                name="Total",
            )

        # Overlay simulation time label
        self._add_time_label(pw, elapsed)

        return pw

    def _add_time_label(self, pw: pg.PlotWidget, elapsed: float) -> None:
        """Add simulation time label to plot."""
        try:
            vb = pw.getViewBox()
            rect = vb.viewRect()
            tlabel = "Sim took:\n" + _format_sim_time(elapsed)
            ti = pg.TextItem(tlabel, color=(0, 0, 0), anchor=(1, 1))
            ti.setFont(QFont("", 16))
            ti.setPos(rect.right(), rect.bottom())
            pw.addItem(ti)
        except Exception:
            pass  # Non-fatal if label placement fails

    def _overlay_targets_seq2seq(
        self,
        pw: pg.PlotWidget,
        dataset,
        data_index: int,
        arr: np.ndarray,
        timebase: Timebase,
        display_settings: DisplaySettings,
        prepend_spec: ZeroPaddingSpec | None = None,
        time_mode: TimeMode = TimeMode.DT,
    ) -> None:
        """Overlay target sequences on output layer plot for seq2seq tasks.

        Args:
            pw: Plot widget to add targets to
            dataset: Dataset containing targets
            data_index: Index of sample
            arr: Prediction array [T, dim] for shape matching
            timebase: Timebase for time axis
            display_settings: Display settings
        """
        try:
            # Fetch target from dataset
            _, tgt = dataset[data_index]
            y = tgt.detach().cpu().numpy() if hasattr(tgt, "detach") else np.array(tgt)

            # Ensure 2D
            if y.ndim == 1:
                y = y[:, None]

            # Resample to match prediction length if needed
            T_plot = arr.shape[0]
            if y.shape[0] != T_plot and y.shape[0] > 0:
                x_old = np.linspace(0.0, 1.0, y.shape[0])
                x_new = np.linspace(0.0, 1.0, T_plot)
                y = np.vstack([np.interp(x_new, x_old, y[:, j]) for j in range(y.shape[1])]).T.astype(np.float32)

            # Build time axis
            shift_steps = 0 if display_settings.include_s0 else 1
            dt_ns = timebase.step_ns
            prepend_steps = 0
            if prepend_spec and prepend_spec.enabled:
                if time_mode == TimeMode.DT:
                    prepend_steps = prepend_spec.count_steps
                elif timebase.step_ns > 0:
                    prepend_steps = int(round(prepend_spec.time_ns / timebase.step_ns))
            time_axis = [(i + shift_steps - prepend_steps) * dt_ns for i in range(T_plot)]

            # Plot targets as dashed lines
            for j in range(min(y.shape[1], arr.shape[1])):
                pw.plot(
                    time_axis,
                    y[:, j],
                    pen=pg.mkPen(color=(0, 0, 0, 180), width=2, style=Qt.PenStyle.DashLine),
                    name=f"Target{j}",
                )
        except Exception:
            pass  # Non-fatal if target overlay fails

    def _overlay_targets_classification(
        self,
        pw: pg.PlotWidget,
        dataset,
        data_index: int,
        arr: np.ndarray,
        timebase: Timebase,
        display_settings: DisplaySettings,
    ) -> None:
        """Overlay expected class label as reference text on classification output plot.

        For classification, we show a text label indicating which class this sample belongs to.
        This helps visually verify if the model's output neurons match the expected class.

        Args:
            pw: Plot widget to add reference to
            dataset: Dataset containing labels
            data_index: Index of sample
            arr: Prediction array [T, dim] for shape matching
            timebase: Timebase for time axis
            display_settings: Display settings
        """
        try:
            # Fetch label from dataset
            _, label = dataset[data_index]
            class_id = int(label.item() if hasattr(label, "item") else label)

            # Add text label showing expected class
            try:
                vb = pw.getViewBox()
                rect = vb.viewRect()

                # Create text showing expected class
                text_str = f"Expected Class: {class_id}"
                ti = pg.TextItem(text_str, color=(255, 0, 0), anchor=(0, 1))
                ti.setFont(QFont("", 14, QFont.Weight.Bold))

                # Position in top-left corner
                ti.setPos(rect.left(), rect.top())
                pw.addItem(ti)

                # Also add a horizontal reference line at a reasonable height
                # to show which neuron should be active (if we have enough neurons)
                if class_id < arr.shape[1]:
                    # Find max value to scale reference line appropriately
                    max_val = np.max(arr)
                    ref_height = max_val * 0.8 if max_val > 0 else 1.0

                    # Add subtle reference line showing expected neuron position
                    ref_line = pg.InfiniteLine(
                        pos=ref_height,
                        angle=0,
                        pen=pg.mkPen(color=(255, 0, 0, 100), width=2, style=Qt.PenStyle.DashLine),
                        label=None,
                    )
                    pw.addItem(ref_line)

            except Exception:
                pass  # Non-fatal if positioning fails

        except Exception:
            pass  # Non-fatal if target overlay fails

    def _overlay_classification_annotations(
        self,
        pw: pg.PlotWidget,
        classification_result: ClassificationResult,
        arr: np.ndarray,
        *,
        layer_index: int,
        total_layers: int,
        show_accuracy: bool,
    ) -> None:
        """Overlay sample separators and accuracy markers for classification mode."""

        if classification_result is None or len(classification_result.sample_classes) == 0:
            return

        shift_steps = getattr(pw, "_shift_steps", 0)
        prepend_steps = getattr(pw, "_prepend_steps", 0)
        dt_ns = getattr(pw, "_dt_ns", 1.0)

        boundaries = classification_result.sample_boundaries or [0]
        total_steps = int(classification_result.input_features.shape[1])

        # Ensure boundaries are sorted and start at zero
        boundaries = sorted({int(b) for b in boundaries if b >= 0})
        if not boundaries or boundaries[0] != 0:
            boundaries.insert(0, 0)

        def _step_to_time_ns(step: int) -> float:
            return (step + shift_steps - prepend_steps) * dt_ns

        # Vertical separators
        for boundary in boundaries[1:]:
            time_ns = _step_to_time_ns(boundary)
            line = pg.InfiniteLine(
                pos=time_ns,
                angle=90,
                pen=pg.mkPen(color=(0, 0, 0, 160), width=2, style=Qt.PenStyle.DashLine),
            )
            pw.addItem(line)

        # Determine annotation placement
        vb = pw.getViewBox()
        _, y_range = vb.viewRange()
        y_min, y_max = y_range
        if y_max == y_min:
            y_max = y_min + 1.0
        y_span = y_max - y_min
        label_y = y_max - 0.08 * y_span

        is_output_layer = layer_index == (total_layers - 1)

        num_samples = len(classification_result.sample_classes)
        for idx in range(num_samples):
            start_step = boundaries[idx] if idx < len(boundaries) else 0
            end_step = boundaries[idx + 1] if idx + 1 < len(boundaries) else total_steps

            start_time = _step_to_time_ns(start_step)
            end_time = _step_to_time_ns(end_step)
            center_time = (start_time + end_time) / 2.0

            actual = classification_result.sample_classes[idx]
            label = f"Class {actual}"

            if is_output_layer and show_accuracy and idx < len(classification_result.predictions):
                pred = classification_result.predictions[idx]
                correct = classification_result.correct[idx] if idx < len(classification_result.correct) else False
                if correct:
                    label += " ✓"
                else:
                    label += f" ✗ (pred {pred})"

            ti = pg.TextItem(label, color=(0, 0, 0), anchor=(0.5, 0))
            ti.setFont(QFont("", 12))
            ti.setPos(center_time, label_y)
            pw.addItem(ti)

    # ========================================================================
    # FFT Plotting Methods
    # ========================================================================

    def _create_fft_view_widget(
        self,
        arr: np.ndarray,
        layer_id: int,
        layer_type: str | None,
        metric: Metric,
        y_label_base: str,
        timebase: Timebase,
        display_settings: DisplaySettings,
        elapsed: float,
        prepend_spec: ZeroPaddingSpec | None,
        append_spec: ZeroPaddingSpec | None,
        time_mode: TimeMode,
        fft_settings: FFTSettings,
        fft_tuple: tuple[np.ndarray, np.ndarray, list | None] | None,
    ) -> QWidget | pg.PlotWidget:
        """Create appropriate FFT view widget based on view mode.

        Args:
            arr: Time domain data [T, dim]
            layer_id: Layer identifier
            layer_type: Layer type string
            metric: Metric being plotted
            y_label_base: Base Y-axis label
            timebase: Timebase for time axis
            display_settings: Display options
            elapsed: Simulation time
            prepend_spec: Prepend padding spec
            append_spec: Append padding spec
            time_mode: Time mode
            fft_settings: FFT configuration
            fft_tuple: (freqs_mhz, magnitudes, peaks) from controller

        Returns:
            Plot widget or composite widget based on view mode
        """
        if fft_tuple is None:
            # Fallback to time only
            return self._create_plot_widget(
                arr, layer_id, layer_type, metric, y_label_base,
                timebase, display_settings, elapsed, prepend_spec, append_spec, time_mode
            )

        freqs_mhz, magnitudes, peaks = fft_tuple

        if fft_settings.view_mode == ViewMode.FREQUENCY_ONLY:
            # FFT only
            return self.create_fft_plot(freqs_mhz, magnitudes, layer_id, fft_settings, peaks)

        elif fft_settings.view_mode == ViewMode.SPLIT:
            # Side-by-side time + frequency
            time_pw = self._create_plot_widget(
                arr, layer_id, layer_type, metric, y_label_base,
                timebase, display_settings, elapsed, prepend_spec, append_spec, time_mode
            )
            fft_pw = self.create_fft_plot(freqs_mhz, magnitudes, layer_id, fft_settings, peaks)
            return self.create_split_view(time_pw, fft_pw)

        elif fft_settings.view_mode == ViewMode.WATERFALL:
            # Spectrogram
            from .fft_analysis import FFTAnalysisService
            fft_service = FFTAnalysisService()
            try:
                time_ns, freqs, Sxx = fft_service.compute_spectrogram(
                    arr,
                    timebase.step_ns,
                    fft_settings.window_function,
                )
                return self.create_waterfall_plot(time_ns, freqs, Sxx, layer_id)
            except Exception:
                # Fallback to FFT if spectrogram fails
                return self.create_fft_plot(freqs_mhz, magnitudes, layer_id, fft_settings, peaks)

        # Default fallback
        return self._create_plot_widget(
            arr, layer_id, layer_type, metric, y_label_base,
            timebase, display_settings, elapsed, prepend_spec, append_spec, time_mode
        )

    def create_fft_plot(
        self,
        freqs_mhz: np.ndarray,
        magnitudes: np.ndarray,
        layer_id: int,
        fft_settings: FFTSettings,
        peaks: list[tuple[float, float]] | None = None,
    ) -> pg.PlotWidget:
        """Create frequency domain plot.

        Args:
            freqs_mhz: Frequency axis in MHz [N_freqs]
            magnitudes: Spectrum magnitudes [N_freqs, channels] or [N_freqs] if aggregated
            layer_id: Layer identifier
            fft_settings: FFT configuration
            peaks: Optional list of (frequency_mhz, magnitude) peak tuples

        Returns:
            Configured PlotWidget with FFT spectrum
        """
        pw = pg.PlotWidget()
        pw.setBackground("w")
        pw.showGrid(x=True, y=True)

        # Axis labels
        y_label = "Magnitude" if fft_settings.y_scale == "linear" else "Magnitude (dB)"
        label_style = {"color": "#000", "font-size": "12pt"}
        pw.setLabel("left", f"{y_label} - Layer {layer_id}", **label_style)
        pw.setLabel("bottom", "Frequency (MHz)", **label_style)

        # Style axis ticks
        tick_font = QFont()
        tick_font.setPointSize(10)
        for ax_name in ("left", "bottom"):
            ax = pw.getAxis(ax_name)
            ax.setPen(pg.mkPen(color="k", width=1))
            ax.setTextPen("k")
            with contextlib.suppress(Exception):
                ax.setStyle(tickFont=tick_font)

        # Add legend if multiple channels
        if magnitudes.ndim > 1 and magnitudes.shape[1] > 1:
            pw.addLegend()

        # Plot spectra
        if magnitudes.ndim == 1 or magnitudes.shape[1] == 1:
            # Single trace (aggregated or single channel)
            mag_plot = magnitudes.flatten()
            pw.plot(freqs_mhz, mag_plot, pen=pg.mkPen(color="k", width=2))
        else:
            # Individual channels
            for ch in range(magnitudes.shape[1]):
                color = pg.intColor(ch, magnitudes.shape[1], alpha=200)
                pw.plot(
                    freqs_mhz,
                    magnitudes[:, ch],
                    pen=pg.mkPen(color=color, width=1.5),
                    name=f"N{ch}",
                )

        # Add peak markers
        if peaks and fft_settings.show_peaks:
            for freq_mhz, mag in peaks:
                # Vertical line at peak
                peak_line = pg.InfiniteLine(
                    pos=freq_mhz,
                    angle=90,
                    pen=pg.mkPen(color=(255, 0, 0, 120), width=1, style=Qt.PenStyle.DashLine),
                )
                pw.addItem(peak_line)

                # Text label
                label_text = f"{freq_mhz:.2f} MHz"
                try:
                    vb = pw.getViewBox()
                    y_range = vb.viewRange()[1]
                    label_y = y_range[1] * 0.95

                    label = pg.TextItem(label_text, color=(255, 0, 0), anchor=(0.5, 1))
                    label.setFont(QFont("", 9))
                    label.setPos(freq_mhz, label_y)
                    pw.addItem(label)
                except Exception:
                    pass

        # Set frequency range if specified
        if fft_settings.freq_min_mhz is not None or fft_settings.freq_max_mhz is not None:
            x_min = fft_settings.freq_min_mhz if fft_settings.freq_min_mhz is not None else freqs_mhz[0]
            x_max = fft_settings.freq_max_mhz if fft_settings.freq_max_mhz is not None else freqs_mhz[-1]
            pw.setXRange(x_min, x_max, padding=0.02)

        return pw

    def create_split_view(
        self,
        time_plot: pg.PlotWidget,
        fft_plot: pg.PlotWidget,
    ) -> QWidget:
        """Create side-by-side time + frequency view.

        Args:
            time_plot: Time domain plot widget
            fft_plot: Frequency domain plot widget

        Returns:
            QSplitter with both plots
        """
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(time_plot)
        splitter.addWidget(fft_plot)
        splitter.setSizes([60, 40])  # 60% time, 40% FFT
        splitter.setChildrenCollapsible(False)

        return splitter

    def create_waterfall_plot(
        self,
        time_ns: np.ndarray,
        freqs_mhz: np.ndarray,
        spectrogram: np.ndarray,
        layer_id: int,
    ) -> pg.PlotWidget:
        """Create spectrogram (time-frequency heatmap).

        Args:
            time_ns: Time axis in nanoseconds
            freqs_mhz: Frequency axis in MHz
            spectrogram: Spectrogram data [freqs, time] in dB
            layer_id: Layer identifier

        Returns:
            PlotWidget with spectrogram image
        """
        pw = pg.PlotWidget()
        pw.setBackground("w")

        # Create image item for spectrogram
        img = pg.ImageItem()
        pw.addItem(img)

        # Set data (transpose for correct orientation)
        img.setImage(spectrogram)

        # Set transform to map image coords to data coords
        # Image is [freqs, time], we want x=time, y=freq
        tr = pg.QtGui.QTransform()
        tr.translate(time_ns[0], freqs_mhz[0])
        tr.scale(
            (time_ns[-1] - time_ns[0]) / spectrogram.shape[1],
            (freqs_mhz[-1] - freqs_mhz[0]) / spectrogram.shape[0],
        )
        img.setTransform(tr)

        # Color map
        try:
            img.setColorMap(pg.colormap.get("viridis"))
        except Exception:
            # Fallback if viridis not available
            pass

        # Axis labels
        label_style = {"color": "#000", "font-size": "12pt"}
        pw.setLabel("left", f"Frequency (MHz) - Layer {layer_id}", **label_style)
        pw.setLabel("bottom", "Time (ns)", **label_style)

        # Style axis ticks
        tick_font = QFont()
        tick_font.setPointSize(10)
        for ax_name in ("left", "bottom"):
            ax = pw.getAxis(ax_name)
            ax.setPen(pg.mkPen(color="k", width=1))
            ax.setTextPen("k")
            with contextlib.suppress(Exception):
                ax.setStyle(tickFont=tick_font)

        # Add colorbar
        try:
            colorbar = pg.ColorBarItem(
                values=(spectrogram.min(), spectrogram.max()),
                colorMap=pg.colormap.get("viridis"),
                label="Magnitude (dB)",
            )
            colorbar.setImageItem(img, insert_in=pw)
        except Exception:
            pass  # Non-fatal if colorbar fails

        return pw
