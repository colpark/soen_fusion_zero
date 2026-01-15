"""Test FFT analysis service."""

from __future__ import annotations

import numpy as np

from soen_toolkit.model_creation_gui.features.state_trajectory.fft_analysis import FFTAnalysisService


class TestFFTAnalysisService:
    """Tests for FFT computation service."""

    def test_basic_fft(self):
        """Test basic FFT with known frequencies."""
        # Create test signal: 10 MHz and 25 MHz components
        fs_ghz = 100.0  # 100 GHz sample rate
        dt_ns = 1e3 / fs_ghz  # ~10 ns time step
        duration_ns = 1000.0  # 1 microsecond
        N = int(duration_ns / dt_ns)

        # Generate test signal
        t = np.arange(N) * dt_ns / 1e9  # time in seconds
        freq1_mhz = 10.0
        freq2_mhz = 25.0
        signal = (
            1.0 * np.sin(2 * np.pi * freq1_mhz * 1e6 * t) +
            0.5 * np.sin(2 * np.pi * freq2_mhz * 1e6 * t)
        )

        # Multi-channel
        time_series = np.stack([signal, signal * 0.8], axis=1)

        # Compute FFT
        fft_service = FFTAnalysisService()
        freqs_mhz, magnitudes = fft_service.compute_spectrum(
            time_series,
            dt_ns,
            window_function="hann",
            aggregation_mode="individual",
            remove_dc=True,
            normalize=False,
            y_scale="db",
        )

        # Verify output shapes
        assert freqs_mhz.shape[0] == N // 2 + 1, "Frequency axis wrong length"
        assert magnitudes.shape[0] == N // 2 + 1, "Magnitude axis wrong length"
        assert magnitudes.shape[1] == 2, "Should have 2 channels"

        # Find peaks
        peaks = fft_service.find_peaks(freqs_mhz, magnitudes, num_peaks=3)

        # Verify peaks near expected frequencies
        peak_freqs = [p[0] for p in peaks]
        assert any(abs(f - freq1_mhz) < 2.0 for f in peak_freqs), "10 MHz peak not found"
        assert any(abs(f - freq2_mhz) < 2.0 for f in peak_freqs), "25 MHz peak not found"

    def test_aggregation_modes(self):
        """Test different channel aggregation modes."""
        # Simple signal
        N = 100
        dt_ns = 10.0
        signal = np.random.randn(N, 3)  # 3 channels

        fft_service = FFTAnalysisService()

        # Test all aggregation modes
        modes = ["individual", "average_post_fft", "average_pre_fft", "rms", "max_hold"]

        for mode in modes:
            freqs, mags = fft_service.compute_spectrum(
                signal, dt_ns, aggregation_mode=mode
            )

            assert freqs.shape[0] > 0, f"Empty freq axis for {mode}"
            assert mags.shape[0] > 0, f"Empty magnitude for {mode}"

            if mode == "individual":
                assert mags.shape[1] == 3, "Individual should keep all channels"
            else:
                assert mags.shape[1] == 1, f"{mode} should aggregate to 1 channel"

    def test_window_functions(self):
        """Test different window functions."""
        N = 100
        fft_service = FFTAnalysisService()

        windows = ["hann", "hamming", "blackman", "rectangular"]

        for window in windows:
            w = fft_service._get_window(N, window)
            assert w.shape == (N,), f"Window {window} wrong shape"
            assert np.all(np.isfinite(w)), f"Window {window} has non-finite values"

    def test_frequency_calculations(self):
        """Test frequency resolution and Nyquist calculations."""
        fft_service = FFTAnalysisService()

        # Test frequency resolution
        res = fft_service.get_frequency_resolution(seq_len=1000, dt_ns=10.0)
        expected_res = 1.0 / (1000 * 10.0 / 1e3)  # MHz
        assert abs(res - expected_res) < 1e-6, "Frequency resolution incorrect"

        # Test Nyquist frequency
        nyquist = fft_service.get_nyquist_frequency(dt_ns=10.0)
        expected_nyquist = 1e9 / (2 * 10.0) / 1e9  # GHz
        assert abs(nyquist - expected_nyquist) < 1e-6, "Nyquist frequency incorrect"

    def test_spectrogram(self):
        """Test spectrogram computation."""
        # Create signal with time-varying frequency
        N = 500
        dt_ns = 10.0
        t = np.arange(N) * dt_ns / 1e9

        # Chirp signal (frequency increases over time)
        freq_start = 5e6
        freq_end = 50e6
        freq_sweep = freq_start + (freq_end - freq_start) * (t / t[-1])
        signal = np.sin(2 * np.pi * np.cumsum(freq_sweep) * dt_ns / 1e9)

        time_series = signal[:, None]  # Single channel

        fft_service = FFTAnalysisService()
        time_ns, freqs_mhz, Sxx = fft_service.compute_spectrogram(
            time_series,
            dt_ns,
            window_function="hann",
        )

        # Verify shapes
        assert time_ns.shape[0] > 0, "Empty time axis"
        assert freqs_mhz.shape[0] > 0, "Empty frequency axis"
        assert Sxx.shape == (freqs_mhz.shape[0], time_ns.shape[0]), "Spectrogram wrong shape"
        assert np.all(np.isfinite(Sxx)), "Spectrogram has non-finite values"

    def test_dc_removal(self):
        """Test DC offset removal."""
        N = 100
        dt_ns = 10.0

        # Signal with DC offset
        signal = np.ones((N, 2)) * 5.0  # Strong DC
        signal[:, 0] += 0.1 * np.sin(2 * np.pi * 10e6 * np.arange(N) * dt_ns / 1e9)

        fft_service = FFTAnalysisService()

        # Without DC removal
        freqs1, mags1 = fft_service.compute_spectrum(
            signal, dt_ns, remove_dc=False, y_scale="linear"
        )

        # With DC removal
        freqs2, mags2 = fft_service.compute_spectrum(
            signal, dt_ns, remove_dc=True, y_scale="linear"
        )

        # DC bin should be much smaller with removal
        assert mags2[0, 0] < mags1[0, 0] * 0.1, "DC removal ineffective"

    def test_normalization(self):
        """Test spectrum normalization."""
        N = 100
        dt_ns = 10.0
        signal = np.random.randn(N, 1)

        fft_service = FFTAnalysisService()

        freqs, mags = fft_service.compute_spectrum(
            signal, dt_ns, normalize=True, y_scale="linear"
        )

        # Max should be 1.0 (or very close)
        assert abs(mags.max() - 1.0) < 1e-6, "Normalization failed"

