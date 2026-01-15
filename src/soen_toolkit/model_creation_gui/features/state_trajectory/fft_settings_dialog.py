"""FFT settings configuration dialog."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .settings import FFTSettings


class FFTSettingsDialog(QDialog):
    """Dialog for configuring FFT analysis parameters."""

    def __init__(self, parent=None, initial_settings: FFTSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle("FFT Analysis Settings")
        self.setMinimumWidth(400)

        # Store settings
        self._settings = initial_settings or FFTSettings()

        self._init_ui()
        self._load_settings()

    def _init_ui(self) -> None:
        """Initialize UI components."""
        layout = QVBoxLayout(self)

        # Window function group
        window_group = QGroupBox("Window Function")
        window_layout = QFormLayout()

        self.window_cb = QComboBox()
        self.window_cb.addItems(["Hann", "Hamming", "Blackman", "Rectangular"])
        self.window_cb.setToolTip("Window function applied before FFT to reduce spectral leakage")
        window_layout.addRow("Window Type:", self.window_cb)

        window_group.setLayout(window_layout)
        layout.addWidget(window_group)

        # Channel aggregation group
        agg_group = QGroupBox("Channel Aggregation")
        agg_layout = QFormLayout()

        self.agg_mode_cb = QComboBox()
        self.agg_mode_cb.addItems([
            "Individual Channels",
            "Average (post-FFT)",
            "Average (pre-FFT)",
            "RMS",
            "Max Hold",
        ])
        self.agg_mode_cb.setToolTip(
            "How to combine multiple channels:\n"
            "• Individual: Plot each channel separately\n"
            "• Average (post-FFT): Compute FFT per channel, then average\n"
            "• Average (pre-FFT): Average time traces, then compute FFT\n"
            "• RMS: Root mean square across channels\n"
            "• Max Hold: Maximum magnitude across channels"
        )
        agg_layout.addRow("Mode:", self.agg_mode_cb)

        agg_group.setLayout(agg_layout)
        layout.addWidget(agg_group)

        # Frequency range group
        freq_group = QGroupBox("Frequency Range")
        freq_layout = QFormLayout()

        freq_row_min = QHBoxLayout()
        self.freq_min_spin = QDoubleSpinBox()
        self.freq_min_spin.setRange(0.0, 1e6)
        self.freq_min_spin.setSuffix(" MHz")
        self.freq_min_spin.setDecimals(3)
        self.freq_min_spin.setSpecialValueText("Auto (0)")
        self.freq_min_spin.setValue(0.0)
        freq_row_min.addWidget(self.freq_min_spin)
        freq_layout.addRow("Min Frequency:", freq_row_min)

        freq_row_max = QHBoxLayout()
        self.freq_max_spin = QDoubleSpinBox()
        self.freq_max_spin.setRange(0.0, 1e6)
        self.freq_max_spin.setSuffix(" MHz")
        self.freq_max_spin.setDecimals(3)
        self.freq_max_spin.setSpecialValueText("Auto (Nyquist)")
        self.freq_max_spin.setValue(0.0)
        freq_row_max.addWidget(self.freq_max_spin)
        freq_layout.addRow("Max Frequency:", freq_row_max)

        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)

        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()

        # Y-axis scale
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Y-Axis Scale:"))
        self.scale_cb = QComboBox()
        self.scale_cb.addItems(["Linear", "Logarithmic (dB)"])
        self.scale_cb.setToolTip("dB scale: 20*log10(magnitude)")
        scale_row.addWidget(self.scale_cb)
        scale_row.addStretch()
        display_layout.addLayout(scale_row)

        # Checkboxes
        self.detrend_cb = QCheckBox("Remove DC offset")
        self.detrend_cb.setToolTip("Subtract mean value before FFT")
        self.detrend_cb.setChecked(True)
        display_layout.addWidget(self.detrend_cb)

        self.normalize_cb = QCheckBox("Normalize to max = 1")
        self.normalize_cb.setToolTip("Scale spectrum so maximum value is 1")
        display_layout.addWidget(self.normalize_cb)

        self.show_peaks_cb = QCheckBox("Show dominant peaks")
        self.show_peaks_cb.setToolTip("Annotate prominent frequency peaks")
        self.show_peaks_cb.setChecked(True)
        display_layout.addWidget(self.show_peaks_cb)

        # Number of peaks
        peaks_row = QHBoxLayout()
        peaks_row.addWidget(QLabel("  Max peaks:"))
        self.num_peaks_spin = QSpinBox()
        self.num_peaks_spin.setRange(1, 20)
        self.num_peaks_spin.setValue(5)
        self.num_peaks_spin.setEnabled(True)
        peaks_row.addWidget(self.num_peaks_spin)
        peaks_row.addStretch()
        display_layout.addLayout(peaks_row)

        # Connect peak checkbox to num_peaks enablement
        self.show_peaks_cb.toggled.connect(self.num_peaks_spin.setEnabled)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Info label
        info_label = QLabel(
            "💡 Tip: Longer sequences provide better frequency resolution.\n"
            "    Resolution = 1 / (sequence_length × dt)"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 9pt; padding: 5px;")
        layout.addWidget(info_label)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _load_settings(self) -> None:
        """Load settings into UI."""
        # Window function
        window_map = {
            "hann": "Hann",
            "hamming": "Hamming",
            "blackman": "Blackman",
            "rectangular": "Rectangular",
        }
        window_text = window_map.get(self._settings.window_function.lower(), "Hann")
        self.window_cb.setCurrentText(window_text)

        # Aggregation mode
        agg_map = {
            "individual": "Individual Channels",
            "average_post_fft": "Average (post-FFT)",
            "average_pre_fft": "Average (pre-FFT)",
            "rms": "RMS",
            "max_hold": "Max Hold",
        }
        agg_text = agg_map.get(self._settings.aggregation_mode, "Individual Channels")
        self.agg_mode_cb.setCurrentText(agg_text)

        # Frequency range
        if self._settings.freq_min_mhz is not None:
            self.freq_min_spin.setValue(self._settings.freq_min_mhz)
        else:
            self.freq_min_spin.setValue(0.0)

        if self._settings.freq_max_mhz is not None:
            self.freq_max_spin.setValue(self._settings.freq_max_mhz)
        else:
            self.freq_max_spin.setValue(0.0)

        # Y-axis scale
        if self._settings.y_scale == "db":
            self.scale_cb.setCurrentText("Logarithmic (dB)")
        else:
            self.scale_cb.setCurrentText("Linear")

        # Checkboxes
        self.detrend_cb.setChecked(self._settings.remove_dc)
        self.normalize_cb.setChecked(self._settings.normalize)
        self.show_peaks_cb.setChecked(self._settings.show_peaks)
        self.num_peaks_spin.setValue(self._settings.num_peaks)

    def _reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        self._settings = FFTSettings()
        self._load_settings()

    def get_settings(self) -> FFTSettings:
        """Extract settings from UI.

        Returns:
            FFTSettings with current UI values
        """
        # Window function
        window_reverse_map = {
            "Hann": "hann",
            "Hamming": "hamming",
            "Blackman": "blackman",
            "Rectangular": "rectangular",
        }
        window_func = window_reverse_map.get(self.window_cb.currentText(), "hann")

        # Aggregation mode
        agg_reverse_map = {
            "Individual Channels": "individual",
            "Average (post-FFT)": "average_post_fft",
            "Average (pre-FFT)": "average_pre_fft",
            "RMS": "rms",
            "Max Hold": "max_hold",
        }
        agg_mode = agg_reverse_map.get(self.agg_mode_cb.currentText(), "individual")

        # Frequency range (0.0 means None/auto)
        freq_min = self.freq_min_spin.value() if self.freq_min_spin.value() > 0.0 else None
        freq_max = self.freq_max_spin.value() if self.freq_max_spin.value() > 0.0 else None

        # Y-axis scale
        y_scale = "db" if self.scale_cb.currentText() == "Logarithmic (dB)" else "linear"

        return FFTSettings(
            view_mode=self._settings.view_mode,  # Not changed in this dialog
            window_function=window_func,
            aggregation_mode=agg_mode,
            freq_min_mhz=freq_min,
            freq_max_mhz=freq_max,
            y_scale=y_scale,
            remove_dc=self.detrend_cb.isChecked(),
            normalize=self.normalize_cb.isChecked(),
            show_peaks=self.show_peaks_cb.isChecked(),
            num_peaks=self.num_peaks_spin.value(),
        )

