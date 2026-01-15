"""FFT analysis service for frequency domain visualization."""

from __future__ import annotations

import numpy as np
from scipy import signal


class FFTAnalysisService:
    """Pure FFT computation and data preparation.

    Handles windowing, detrending, aggregation, and frequency axis scaling.
    """

    def compute_spectrum(
        self,
        time_series: np.ndarray,
        dt_ns: float,
        window_function: str = "hann",
        aggregation_mode: str = "individual",
        remove_dc: bool = True,
        normalize: bool = False,
        y_scale: str = "db",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute frequency spectrum from time series.

        Args:
            time_series: Time series data [T, channels]
            dt_ns: Time step in nanoseconds
            window_function: Window type ("hann", "hamming", "blackman", "rectangular")
            aggregation_mode: Channel aggregation ("individual", "average_post_fft",
                            "average_pre_fft", "rms", "max_hold")
            remove_dc: Remove DC offset before FFT
            normalize: Normalize to max = 1
            y_scale: Y-axis scale ("linear" or "db")

        Returns:
            freqs_mhz: Frequency axis in MHz [N_freqs]
            magnitudes: Spectrum magnitudes [N_freqs, channels] or [N_freqs] if aggregated
        """
        # Handle aggregation_pre_fft case
        if aggregation_mode == "average_pre_fft":
            time_series = time_series.mean(axis=1, keepdims=True)

        # Window application
        window = self._get_window(time_series.shape[0], window_function)
        windowed = time_series * window[:, None]

        # Detrending
        if remove_dc:
            windowed = windowed - windowed.mean(axis=0, keepdims=True)

        # FFT (use rfft for real-valued signals)
        fft_result = np.fft.rfft(windowed, axis=0)
        magnitudes = np.abs(fft_result)

        # Frequency axis (convert to MHz)
        sample_rate_hz = 1e9 / dt_ns  # ns -> Hz
        freqs_hz = np.fft.rfftfreq(time_series.shape[0], d=1/sample_rate_hz)
        freqs_mhz = freqs_hz / 1e6

        # Post-FFT aggregation
        if aggregation_mode == "average_post_fft":
            magnitudes = magnitudes.mean(axis=1, keepdims=True)
        elif aggregation_mode == "rms":
            magnitudes = np.sqrt((magnitudes ** 2).mean(axis=1, keepdims=True))
        elif aggregation_mode == "max_hold":
            magnitudes = magnitudes.max(axis=1, keepdims=True)
        # else: individual channels, keep as-is

        # Normalization
        if normalize:
            max_val = magnitudes.max()
            if max_val > 1e-10:
                magnitudes = magnitudes / max_val

        # Convert to dB if requested
        if y_scale == "db":
            # Add small epsilon to avoid log(0)
            magnitudes = 20 * np.log10(magnitudes + 1e-10)

        return freqs_mhz, magnitudes

    def compute_spectrogram(
        self,
        time_series: np.ndarray,
        dt_ns: float,
        window_function: str = "hann",
        nperseg: int | None = None,
        noverlap: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram using Short-Time Fourier Transform.

        Args:
            time_series: Time series data [T, channels]
            dt_ns: Time step in nanoseconds
            window_function: Window type
            nperseg: Length of each segment (default: T//8)
            noverlap: Number of points to overlap (default: nperseg//2)

        Returns:
            time_ns: Time axis in nanoseconds
            freqs_mhz: Frequency axis in MHz
            Sxx: Spectrogram [freqs, time, channels]
        """
        # Average across channels for spectrogram
        if time_series.shape[1] > 1:
            ts_avg = time_series.mean(axis=1)
        else:
            ts_avg = time_series[:, 0]

        # Default window size
        if nperseg is None:
            nperseg = min(256, time_series.shape[0] // 8)
        if noverlap is None:
            noverlap = nperseg // 2

        # Sample rate
        sample_rate_hz = 1e9 / dt_ns

        # Compute spectrogram
        freqs_hz, times_s, Sxx = signal.spectrogram(
            ts_avg,
            fs=sample_rate_hz,
            window=window_function,
            nperseg=nperseg,
            noverlap=noverlap,
            mode='magnitude',
        )

        # Convert units
        freqs_mhz = freqs_hz / 1e6
        time_ns = times_s * 1e9

        # Convert to dB
        Sxx_db = 20 * np.log10(Sxx + 1e-10)

        return time_ns, freqs_mhz, Sxx_db

    def find_peaks(
        self,
        freqs_mhz: np.ndarray,
        magnitudes: np.ndarray,
        num_peaks: int = 5,
        min_prominence: float = 10.0,
    ) -> list[tuple[float, float]]:
        """Find dominant frequency peaks.

        Args:
            freqs_mhz: Frequency axis
            magnitudes: Spectrum magnitudes (single channel or aggregated)
            num_peaks: Maximum number of peaks to return
            min_prominence: Minimum peak prominence in dB

        Returns:
            List of (frequency_mhz, magnitude) tuples
        """
        # Use first channel if multi-channel
        if magnitudes.ndim > 1 and magnitudes.shape[1] > 1:
            mag = magnitudes[:, 0]
        else:
            mag = magnitudes.flatten()

        # Find peaks
        peaks, properties = signal.find_peaks(
            mag,
            prominence=min_prominence,
            distance=max(1, len(mag) // 100),  # Avoid clustering
        )

        # Sort by prominence
        prominences = properties['prominences']
        sorted_indices = np.argsort(prominences)[::-1]

        # Return top peaks
        result = []
        for idx in sorted_indices[:num_peaks]:
            peak_idx = peaks[idx]
            result.append((freqs_mhz[peak_idx], mag[peak_idx]))

        return result

    def _get_window(self, N: int, name: str) -> np.ndarray:
        """Get window function.

        Args:
            N: Window length
            name: Window type

        Returns:
            Window array of length N
        """
        name_lower = name.lower()

        if name_lower == "hann":
            return np.hanning(N)
        elif name_lower == "hamming":
            return np.hamming(N)
        elif name_lower == "blackman":
            return np.blackman(N)
        elif name_lower == "rectangular":
            return np.ones(N)
        else:
            # Default to Hann
            return np.hanning(N)

    def get_frequency_resolution(self, seq_len: int, dt_ns: float) -> float:
        """Calculate frequency resolution for given parameters.

        Args:
            seq_len: Number of time steps
            dt_ns: Time step in nanoseconds

        Returns:
            Frequency resolution in MHz
        """
        1e9 / dt_ns
        total_time_s = seq_len * dt_ns / 1e9
        return 1.0 / total_time_s / 1e6  # Convert to MHz

    def get_nyquist_frequency(self, dt_ns: float) -> float:
        """Calculate Nyquist frequency for given time step.

        Args:
            dt_ns: Time step in nanoseconds

        Returns:
            Nyquist frequency in GHz
        """
        sample_rate_hz = 1e9 / dt_ns
        nyquist_hz = sample_rate_hz / 2.0
        return nyquist_hz / 1e9  # Convert to GHz

