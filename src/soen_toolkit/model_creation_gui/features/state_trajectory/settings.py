"""Settings dataclasses and persistence adapter for state trajectory dialog."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from PyQt6.QtCore import QSettings


class Metric(str, Enum):
    """Available metrics for plotting."""

    STATE = "state"
    PHI = "phi"
    G = "g"
    POWER = "power"
    ENERGY = "energy"


class Backend(str, Enum):
    """Simulation backends."""

    TORCH = "torch"
    JAX = "jax"


class TaskType(str, Enum):
    """Dataset task types."""

    CLASSIFICATION = "classification"
    SEQ2SEQ = "seq2seq"


class InputKind(str, Enum):
    """Input source types."""

    DATASET = "dataset"
    CONSTANT = "constant"
    GAUSSIAN = "gaussian"
    COLORED = "colored"
    SINE = "sine"
    SQUARE = "square"


class TimeMode(str, Enum):
    """Time specification modes."""

    DT = "dt"
    TOTAL_NS = "total_ns"


class ViewMode(str, Enum):
    """FFT view modes."""

    TIME_ONLY = "time_only"
    FREQUENCY_ONLY = "frequency_only"
    SPLIT = "split"
    WATERFALL = "waterfall"


@dataclass
class FFTSettings:
    """FFT analysis configuration."""

    view_mode: ViewMode = ViewMode.TIME_ONLY
    window_function: str = "hann"  # "hann" | "hamming" | "blackman" | "rectangular"
    aggregation_mode: str = "individual"  # "individual" | "average_post_fft" | "average_pre_fft" | "rms" | "max_hold"
    freq_min_mhz: float | None = None  # None = auto (0)
    freq_max_mhz: float | None = None  # None = auto (Nyquist)
    y_scale: str = "db"  # "linear" | "db"
    remove_dc: bool = True
    normalize: bool = False
    show_peaks: bool = True
    num_peaks: int = 5


@dataclass
class EncodingSettings:
    """Input encoding configuration."""

    mode: str = "raw"  # "raw" | "one_hot"
    vocab_size: int = 65


@dataclass
class ZeroPaddingSpec:
    """Zero padding specification."""

    enabled: bool = False
    count_steps: int = 0  # used when TimeMode.DT
    time_ns: float = 0.0  # used when TimeMode.TOTAL_NS
    mode: str = "zeros"  # "zeros" | "hold_last" (append only)


@dataclass
class ClassificationSettings:
    """Settings for seq2static classification visualization."""

    enabled: bool = False  # Enable multi-sample classification mode
    num_samples: int = 3  # Number of random samples to concatenate
    pooling_method: str = "max"  # "max" or "final" for time pooling
    show_accuracy: bool = True  # Show tick/cross for correct/incorrect


@dataclass
class DisplaySettings:
    """Plot display options."""

    show_legend: bool = False
    show_total: bool = False
    include_s0: bool = True
    overlay_targets: bool = True


@dataclass
class InputParams:
    """Parameters for input generators."""

    constant_val: float = 1.0
    noise_std: float = 0.1
    colored_beta: float = 2.0
    sine_freq_mhz: float = 10.0
    sine_amp: float = 1.0
    sine_phase_deg: float = 0.0
    sine_offset: float = 0.0


@dataclass
class ScalingBounds:
    """Optional feature scaling bounds."""

    min_val: float | None = None
    max_val: float | None = None


@dataclass
class StateTrajSettings:
    """Complete settings for state trajectory simulation and plotting."""

    # Dataset
    dataset_path: str = ""
    group: str | None = None
    task_type: TaskType = TaskType.CLASSIFICATION

    # Simulation
    metric: Metric = Metric.STATE
    backend: Backend = Backend.TORCH
    input_kind: InputKind = InputKind.DATASET
    seq_len: int = 100

    # Time
    time_mode: TimeMode = TimeMode.DT
    dt: float = 37.0
    total_time_ns: float = 0.0

    # Padding
    prepend: ZeroPaddingSpec = field(default_factory=ZeroPaddingSpec)
    append: ZeroPaddingSpec = field(default_factory=ZeroPaddingSpec)

    # Encoding
    encoding: EncodingSettings = field(default_factory=EncodingSettings)
    scaling: ScalingBounds = field(default_factory=ScalingBounds)

    # Display
    display: DisplaySettings = field(default_factory=DisplaySettings)

    # Classification visualization
    classification: ClassificationSettings = field(default_factory=ClassificationSettings)

    # Input generator params
    input_params: InputParams = field(default_factory=InputParams)

    # FFT analysis
    fft: FFTSettings = field(default_factory=FFTSettings)

    # Criticality
    calculate_criticality: bool = False

    # Sample selection (for dataset input)
    class_id: int | None = 0
    sample_index: int = 0


class QSettingsAdapter:
    """Adapter for persisting StateTrajSettings to/from QSettings."""

    def __init__(self, organization: str = "GreatSky", application: str = "SOEN-Toolkit"):
        self._settings = QSettings(organization, application)
        self._prefix = "stateTraj/"

    def load(self) -> StateTrajSettings:
        """Load settings from persistent storage."""
        s = self._settings
        p = self._prefix

        # Helper to safely load enum
        def load_enum(key: str, enum_class, default):
            try:
                val = s.value(p + key, default.value, type=str)
                return enum_class(val)
            except (ValueError, KeyError):
                return default

        # Helper to safely load float
        def load_float(key: str, default: float) -> float:
            try:
                return float(s.value(p + key, default, type=float))
            except (ValueError, TypeError):
                return default

        # Helper to safely load int
        def load_int(key: str, default: int) -> int:
            try:
                return int(s.value(p + key, default, type=int))
            except (ValueError, TypeError):
                return default

        # Helper to safely load bool
        def load_bool(key: str, default: bool) -> bool:
            try:
                return bool(s.value(p + key, default, type=bool))
            except (ValueError, TypeError):
                return default

        # Helper to safely load str
        def load_str(key: str, default: str) -> str:
            try:
                return str(s.value(p + key, default, type=str))
            except (ValueError, TypeError):
                return default

        # Load all settings
        return StateTrajSettings(
            dataset_path=load_str("datasetPath", ""),
            group=s.value(p + "group", None, type=str) or None,
            task_type=load_enum("taskType", TaskType, TaskType.CLASSIFICATION),
            metric=load_enum("metric", Metric, Metric.STATE),
            backend=load_enum("backend", Backend, Backend.TORCH),
            input_kind=load_enum("inputSrc", InputKind, InputKind.DATASET),
            seq_len=load_int("seqLen", 100),
            time_mode=load_enum("timeMode", TimeMode, TimeMode.DT),
            dt=load_float("dt", 37.0),
            total_time_ns=load_float("totalTimeNs", 0.0),
            prepend=ZeroPaddingSpec(
                enabled=load_bool("prependZerosEnabled", False),
                count_steps=load_int("prependZerosCount", 0),
                time_ns=load_float("prependZerosNs", 0.0),
                mode="zeros",  # prepend always zeros
            ),
            append=ZeroPaddingSpec(
                enabled=load_bool("appendZerosEnabled", False),
                count_steps=load_int("appendZerosCount", 0),
                time_ns=load_float("appendZerosNs", 0.0),
                mode=load_str("appendMode", "zeros"),
            ),
            encoding=EncodingSettings(
                mode=load_str("encoding", "raw"),
                vocab_size=load_int("vocabSize", 65),
            ),
            scaling=ScalingBounds(
                min_val=self._load_optional_float("melMin"),
                max_val=self._load_optional_float("melMax"),
            ),
            display=DisplaySettings(
                show_legend=load_bool("showLegend", False),
                show_total=load_bool("showTotal", False),
                include_s0=load_bool("includeS0", True),
                overlay_targets=load_bool("overlayTargets", True),
            ),
            classification=ClassificationSettings(
                enabled=load_bool("classificationEnabled", False),
                num_samples=load_int("classificationNumSamples", 3),
                pooling_method=load_str("classificationPooling", "max"),
                show_accuracy=load_bool("classificationShowAccuracy", True),
            ),
            input_params=InputParams(
                constant_val=load_float("inputConstantVal", 1.0),
                noise_std=load_float("inputNoiseStd", 0.1),
                colored_beta=load_float("inputColoredBeta", 2.0),
                sine_freq_mhz=load_float("inputSineFreqMHz", 10.0),
                sine_amp=load_float("inputSineAmp", 1.0),
                sine_phase_deg=load_float("inputSinePhaseDeg", 0.0),
                sine_offset=load_float("inputSineOffset", 0.0),
            ),
            fft=FFTSettings(
                view_mode=load_enum("fftViewMode", ViewMode, ViewMode.TIME_ONLY),
                window_function=load_str("fftWindow", "hann"),
                aggregation_mode=load_str("fftAggregation", "individual"),
                freq_min_mhz=self._load_optional_float("fftFreqMin"),
                freq_max_mhz=self._load_optional_float("fftFreqMax"),
                y_scale=load_str("fftYScale", "db"),
                remove_dc=load_bool("fftRemoveDC", True),
                normalize=load_bool("fftNormalize", False),
                show_peaks=load_bool("fftShowPeaks", True),
                num_peaks=load_int("fftNumPeaks", 5),
            ),
            class_id=s.value(p + "digit", 0, type=int) if s.value(p + "digit", None) is not None else 0,
            sample_index=load_int("sampleIndex", 0),
            calculate_criticality=load_bool("calculateCriticality", False),
        )

    def _load_optional_float(self, key: str) -> float | None:
        """Load optional float from settings (empty string or 'auto' → None)."""
        try:
            val = self._settings.value(self._prefix + key, "", type=str)
            if not val or val.lower() == "auto":
                return None
            return float(val)
        except (ValueError, TypeError):
            return None

    def save(self, settings: StateTrajSettings) -> None:
        """Save settings to persistent storage."""
        s = self._settings
        p = self._prefix

        s.setValue(p + "datasetPath", settings.dataset_path)
        s.setValue(p + "group", settings.group or "")
        s.setValue(p + "taskType", settings.task_type.value)
        s.setValue(p + "metric", settings.metric.value)
        s.setValue(p + "backend", settings.backend.value)
        s.setValue(p + "inputSrc", settings.input_kind.value)
        s.setValue(p + "seqLen", settings.seq_len)
        s.setValue(p + "timeMode", settings.time_mode.value)
        s.setValue(p + "dt", settings.dt)
        s.setValue(p + "totalTimeNs", settings.total_time_ns)

        # Padding
        s.setValue(p + "prependZerosEnabled", settings.prepend.enabled)
        s.setValue(p + "prependZerosCount", settings.prepend.count_steps)
        s.setValue(p + "prependZerosNs", settings.prepend.time_ns)
        s.setValue(p + "appendZerosEnabled", settings.append.enabled)
        s.setValue(p + "appendZerosCount", settings.append.count_steps)
        s.setValue(p + "appendZerosNs", settings.append.time_ns)
        s.setValue(p + "appendMode", settings.append.mode)

        # Encoding
        s.setValue(p + "encoding", settings.encoding.mode)
        s.setValue(p + "vocabSize", settings.encoding.vocab_size)

        # Scaling (save as string, empty for None)
        s.setValue(p + "melMin", str(settings.scaling.min_val) if settings.scaling.min_val is not None else "")
        s.setValue(p + "melMax", str(settings.scaling.max_val) if settings.scaling.max_val is not None else "")

        # Display
        s.setValue(p + "showLegend", settings.display.show_legend)
        s.setValue(p + "showTotal", settings.display.show_total)
        s.setValue(p + "includeS0", settings.display.include_s0)
        s.setValue(p + "overlayTargets", settings.display.overlay_targets)

        # Classification
        s.setValue(p + "classificationEnabled", settings.classification.enabled)
        s.setValue(p + "classificationNumSamples", settings.classification.num_samples)
        s.setValue(p + "classificationPooling", settings.classification.pooling_method)
        s.setValue(p + "classificationShowAccuracy", settings.classification.show_accuracy)

        # Input params
        s.setValue(p + "inputConstantVal", settings.input_params.constant_val)
        s.setValue(p + "inputNoiseStd", settings.input_params.noise_std)
        s.setValue(p + "inputColoredBeta", settings.input_params.colored_beta)
        s.setValue(p + "inputSineFreqMHz", settings.input_params.sine_freq_mhz)
        s.setValue(p + "inputSineAmp", settings.input_params.sine_amp)
        s.setValue(p + "inputSinePhaseDeg", settings.input_params.sine_phase_deg)
        s.setValue(p + "inputSineOffset", settings.input_params.sine_offset)

        # FFT settings
        s.setValue(p + "fftViewMode", settings.fft.view_mode.value)
        s.setValue(p + "fftWindow", settings.fft.window_function)
        s.setValue(p + "fftAggregation", settings.fft.aggregation_mode)
        s.setValue(p + "fftFreqMin", str(settings.fft.freq_min_mhz) if settings.fft.freq_min_mhz is not None else "")
        s.setValue(p + "fftFreqMax", str(settings.fft.freq_max_mhz) if settings.fft.freq_max_mhz is not None else "")
        s.setValue(p + "fftYScale", settings.fft.y_scale)
        s.setValue(p + "fftRemoveDC", settings.fft.remove_dc)
        s.setValue(p + "fftNormalize", settings.fft.normalize)
        s.setValue(p + "fftShowPeaks", settings.fft.show_peaks)
        s.setValue(p + "fftNumPeaks", settings.fft.num_peaks)

        # Criticality
        s.setValue(p + "calculateCriticality", settings.calculate_criticality)

        # Sample selection
        s.setValue(p + "digit", settings.class_id if settings.class_id is not None else 0)
        s.setValue(p + "sampleIndex", settings.sample_index)
