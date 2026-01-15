# FFT Analysis Feature - Complete Implementation

## ✅ Overview

A comprehensive frequency-domain analysis tool has been integrated into the State Trajectory visualization dialog. This feature leverages the real physical time (nanoseconds) to provide accurate FFT analysis of layer states.

## 🎨 User Interface

### New Controls (Plots Tab, Top Bar)

**View Mode Dropdown:**
- `Time Only` (default) - Standard time-domain view
- `Frequency Only` - FFT spectrum only (maximized)
- `Time + Frequency` - Side-by-side split view (60/40 ratio)
- `Waterfall` - Spectrogram (time-frequency heatmap)

**FFT Settings Button:**
- Opens configuration dialog for advanced options

**FFT Info Label:**
- Displays frequency resolution and Nyquist frequency dynamically
- Example: "FFT: Res=0.100 MHz | Nyquist=5.00 GHz"

### FFT Settings Dialog

**Window Functions:**
- Hann (default) - Good all-purpose window
- Hamming - Slightly wider main lobe
- Blackman - Excellent sidelobe suppression
- Rectangular - No windowing (use with caution)

**Channel Aggregation:**
- `Individual Channels` - Plot each neuron separately (default)
- `Average (post-FFT)` - Compute FFT per channel, then average spectra
- `Average (pre-FFT)` - Average time traces first, then FFT
- `RMS` - Root mean square across channels
- `Max Hold` - Maximum magnitude across channels

**Frequency Range:**
- Min/Max frequency sliders (auto = 0 to Nyquist)
- Useful for zooming into specific frequency bands

**Display Options:**
- Y-Axis Scale: Linear or **Logarithmic (dB)** [default]
- Remove DC offset (enabled by default)
- Normalize to max = 1
- Show dominant peaks (enabled by default)
- Number of peaks to annotate (1-20)

## 📊 View Modes Explained

### Time Only
Standard behavior - no changes from before.

### Frequency Only
Shows FFT magnitude spectrum for all neurons/channels:
- X-axis: Frequency (MHz)
- Y-axis: Magnitude (dB) or linear
- Colors: Per-channel (same as time view)
- Red dashed lines: Dominant frequency peaks with labels

### Time + Frequency (Split View)
Side-by-side comparison:
- Left panel (60%): Time domain s(t)
- Right panel (40%): Frequency spectrum
- Both panels share same neuron color scheme
- Splitter is adjustable by dragging

### Waterfall (Spectrogram)
Time-frequency heatmap using STFT:
- X-axis: Time (ns)
- Y-axis: Frequency (MHz)
- Color: Magnitude (dB) - viridis colormap
- Shows how frequency content evolves over time

## 🔧 Implementation Architecture

### Files Created
```
state_trajectory/
├── fft_analysis.py          # FFT computation service
├── fft_settings_dialog.py   # Configuration modal
└── FFT_FEATURE.md           # This document
```

### Files Modified
```
state_trajectory/
├── settings.py              # Added FFTSettings dataclass + persistence
├── plotting.py              # Added FFT/waterfall plot methods
├── dialog.py                # Added UI controls + handlers
└── controller.py            # Added FFT computation in workflow
```

### Design Principles

**Separation of Concerns:**
- `FFTAnalysisService`: Pure computation (numpy/scipy)
- `PlotRenderer`: Visual presentation (pyqtgraph)
- `Controller`: Orchestration and data flow
- `Dialog`: UI state management

**Performance:**
- FFT computed only when frequency view active
- Lazy evaluation prevents unnecessary computation
- Results computed per-layer in controller for efficiency

**Persistence:**
- All FFT settings saved via QSettings
- User preferences persist across sessions
- Sensible defaults (dB scale, Hann window, DC removal)

## 📐 Technical Details

### Frequency Resolution
```
Resolution (MHz) = 1 / (sequence_length × dt_ns) × 10^3
```

Example: 100 steps × 100 ns/step → 0.1 MHz resolution

### Nyquist Frequency
```
Nyquist (GHz) = (1 / (2 × dt_ns)) × 10^-6
```

Example: dt = 100 ns → Nyquist = 5 GHz

### Signal Processing Chain

1. **Windowing**: Applied to reduce spectral leakage
2. **Detrending**: Remove DC offset (mean subtraction)
3. **FFT**: `np.fft.rfft()` for real-valued signals
4. **Aggregation**: Combine channels (if enabled)
5. **Scaling**: Convert to dB (if enabled)
6. **Peak Detection**: `scipy.signal.find_peaks()` with prominence threshold

### Spectrogram (Waterfall)

Uses Short-Time Fourier Transform (STFT):
- Window size: `min(256, T/8)` samples
- Overlap: 50% (nperseg/2)
- Always averaged across channels
- Output in dB scale

## 🎯 Usage Examples

### Quick Start
1. Load model and dataset
2. Click "Refresh Plot"
3. Change **View** dropdown from "Time Only" to "Time + Frequency"
4. Observe time and frequency domain side-by-side

### Analyzing Oscillations
1. Use **Frequency Only** view
2. Look for peaks in FFT spectrum
3. Peak labels show dominant frequencies
4. Adjust sequence length for better resolution

### Finding Transients
1. Use **Waterfall** view
2. Observe how frequency content changes over time
3. Useful for detecting startup transients or mode transitions

### Tuning FFT
1. Click **FFT Settings...**
2. Try different window functions (Blackman for clean spectra)
3. Adjust channel aggregation for cleaner visualization
4. Enable/disable peak detection as needed

## 🔬 Tested Scenarios

✅ **Basic FFT:**
- Synthetic sine waves correctly identified
- Peak detection finds 10 MHz and 25 MHz components
- All aggregation modes produce expected outputs

✅ **SOEN-like Signals:**
- Multi-channel exponentially decaying signals
- Frequency resolution calculations accurate
- Nyquist limits correctly enforced

✅ **Spectrogram:**
- STFT produces valid time-frequency representation
- Colormap and axes configured correctly

## 🚀 Future Enhancements (Optional)

### Potential Additions:
- **Export FFT**: Save spectrum to CSV/NPZ
- **Cursor Synchronization**: Crosshair in time plot highlights frequency
- **Phase Spectrum**: Show phase alongside magnitude
- **Coherence Analysis**: Cross-channel coherence
- **Bandwidth Markers**: Annotate -3dB bandwidth
- **Zoom Presets**: Quick zoom to common bands (< 1 MHz, 1-10 MHz, etc.)

### Performance Optimizations:
- Cache FFT results until inputs change
- Downsample very long sequences before FFT
- Parallel FFT for multi-layer networks

## 🎓 Notes for Users

### When to Use FFT:

**Good for:**
- Analyzing oscillatory behavior
- Detecting resonant frequencies
- Comparing spectral content across layers
- Finding noise characteristics

**Tips:**
- Longer sequences → better frequency resolution
- Remove DC offset for cleaner spectra
- Use dB scale to see weak components
- Hann window is good default choice

### Interpreting Results:

**Peaks indicate:**
- Resonant modes in the network
- External input frequencies
- Noise characteristics

**Spectrograms show:**
- Transient startup behavior
- Mode transitions over time
- Time-varying frequency content

## 🐛 Troubleshooting

**"No peaks detected":**
- Increase sequence length for better resolution
- Lower minimum prominence in FFT settings
- Check if signal is too noisy (try averaging)

**Aliasing warnings:**
- High-frequency components > Nyquist will fold back
- Reduce dt (increase sample rate) to push Nyquist higher
- Or low-pass filter input if aliasing is problematic

**Spectrogram looks wrong:**
- Very short sequences may not have enough samples
- Try increasing sequence length to 500+ steps
- Check that signal has time-varying frequency content

---

**Implementation Date:** November 2025  
**Python Version:** 3.12+  
**Dependencies:** numpy, scipy, pyqtgraph, PyQt6

