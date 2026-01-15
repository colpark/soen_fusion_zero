"""SageMaker input/output channel management.

Handles the mapping between local/S3 paths and SageMaker channels,
with proper path rewriting for container environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..session import CloudSession

logger = logging.getLogger(__name__)


# SageMaker container paths
CONTAINER_INPUT_BASE = "/opt/ml/input/data"
CONTAINER_OUTPUT_BASE = "/opt/ml/output"
CONTAINER_MODEL_BASE = "/opt/ml/model"


@dataclass
class InputChannel:
    """Represents a SageMaker input channel.

    Attributes:
        name: Channel name (e.g., "training", "config", "model").
        source: S3 URI or local path to data.
        container_path: Path where data is mounted in container.
        content_type: MIME type of data (optional).
    """

    name: str
    source: str
    container_path: str = ""
    content_type: str | None = None

    def __post_init__(self) -> None:
        """Set default container path if not provided."""
        if not self.container_path:
            self.container_path = f"{CONTAINER_INPUT_BASE}/{self.name}"

    @property
    def is_s3(self) -> bool:
        """Check if source is an S3 URI."""
        return self.source.startswith("s3://")

    @property
    def is_local(self) -> bool:
        """Check if source is a local path."""
        return not self.is_s3

    def to_s3(self, session: CloudSession, s3_prefix: str) -> InputChannel:
        """Upload local source to S3 and return new channel.

        Args:
            session: Cloud session for S3 operations.
            s3_prefix: S3 prefix for uploaded files.

        Returns:
            New InputChannel with S3 source.
        """
        if self.is_s3:
            return self

        local_path = Path(self.source)
        if not local_path.exists():
            raise FileNotFoundError(f"Input channel source not found: {self.source}")

        if local_path.is_file():
            s3_key = f"{s3_prefix}/{self.name}/{local_path.name}"
            s3_uri = session.upload_file(str(local_path), s3_key)
        else:
            s3_key = f"{s3_prefix}/{self.name}"
            session.upload_directory(str(local_path), s3_key)
            s3_uri = f"s3://{session.bucket}/{s3_key}"

        return InputChannel(
            name=self.name,
            source=s3_uri,
            container_path=self.container_path,
            content_type=self.content_type,
        )


@dataclass
class OutputChannel:
    """Represents a SageMaker output channel.

    Attributes:
        name: Channel name.
        s3_uri: S3 URI where outputs will be written.
        container_path: Path in container where outputs are written.
    """

    name: str
    s3_uri: str
    container_path: str = CONTAINER_MODEL_BASE


@dataclass
class ChannelConfig:
    """Complete channel configuration for a job.

    Attributes:
        inputs: List of input channels.
        outputs: List of output channels.
        path_mapping: Mapping from original paths to container paths.
    """

    inputs: list[InputChannel] = field(default_factory=list)
    outputs: list[OutputChannel] = field(default_factory=list)
    path_mapping: dict[str, str] = field(default_factory=dict)

    def get_input_dict(self) -> dict[str, str]:
        """Get SageMaker-compatible input dictionary.

        Returns:
            Dict mapping channel names to S3 URIs.
        """
        return {ch.name: ch.source for ch in self.inputs if ch.is_s3}

    def add_input(
        self,
        name: str,
        source: str,
        content_type: str | None = None,
    ) -> InputChannel:
        """Add an input channel.

        Args:
            name: Channel name.
            source: S3 URI or local path.
            content_type: Optional MIME type.

        Returns:
            Created InputChannel.
        """
        channel = InputChannel(
            name=name,
            source=source,
            content_type=content_type,
        )
        self.inputs.append(channel)
        self.path_mapping[source] = channel.container_path
        return channel

    def add_output(
        self,
        name: str,
        s3_uri: str,
        container_path: str = CONTAINER_MODEL_BASE,
    ) -> OutputChannel:
        """Add an output channel.

        Args:
            name: Channel name.
            s3_uri: S3 URI for outputs.
            container_path: Path in container.

        Returns:
            Created OutputChannel.
        """
        channel = OutputChannel(
            name=name,
            s3_uri=s3_uri,
            container_path=container_path,
        )
        self.outputs.append(channel)
        return channel


def prepare_channels(
    session: CloudSession,
    *,
    data_path: str | None = None,
    config_path: str | None = None,
    model_path: str | None = None,
    output_path: str | None = None,
    job_name: str | None = None,
) -> ChannelConfig:
    """Prepare input/output channels for a SageMaker job.

    Uploads local files to S3 and creates channel configuration.

    Args:
        session: Cloud session for S3 operations.
        data_path: Training data path (S3 or local).
        config_path: Configuration file path (S3 or local).
        model_path: Model checkpoint path (S3 or local).
        output_path: Output S3 path (default: auto-generated).
        job_name: Job name for S3 prefix.

    Returns:
        ChannelConfig with prepared channels.
    """
    cloud_cfg = session.cloud_config
    project = cloud_cfg.project if cloud_cfg else "soen"
    experiment = cloud_cfg.experiment if cloud_cfg else "default"
    timestamp = time.strftime("%Y%m%d%H%M%S")
    job_id = job_name or f"job-{timestamp}"

    s3_prefix = f"soen/{project}/{experiment}/{job_id}"

    config = ChannelConfig()

    # Training data channel
    if data_path:
        channel = config.add_input("training", data_path)
        if channel.is_local:
            uploaded = channel.to_s3(session, s3_prefix)
            config.inputs[-1] = uploaded
            logger.info(f"Uploaded training data to {uploaded.source}")

    # Config channel
    if config_path:
        channel = config.add_input("config", config_path, content_type="application/x-yaml")
        if channel.is_local:
            uploaded = channel.to_s3(session, s3_prefix)
            config.inputs[-1] = uploaded
            logger.info(f"Uploaded config to {uploaded.source}")

    # Model channel (for fine-tuning)
    if model_path:
        channel = config.add_input("model", model_path)
        if channel.is_local:
            uploaded = channel.to_s3(session, s3_prefix)
            config.inputs[-1] = uploaded
            logger.info(f"Uploaded model to {uploaded.source}")

    # Output channel
    if output_path:
        output_s3 = output_path
    else:
        output_s3 = f"s3://{session.bucket}/{s3_prefix}/output"
    config.add_output("model", output_s3)

    return config


def rewrite_paths_for_container(
    config: dict,
    channel_config: ChannelConfig,
) -> dict:
    """Rewrite paths in a config dict for container environment.

    Args:
        config: Original configuration dictionary.
        channel_config: Channel configuration with path mappings.

    Returns:
        Modified configuration with container paths.
    """
    import copy

    result = copy.deepcopy(config)

    # Rewrite data.data_path
    if "data" in result and "data_path" in result["data"]:
        original = result["data"]["data_path"]
        if original in channel_config.path_mapping:
            result["data"]["data_path"] = channel_config.path_mapping[original]
        elif not original.startswith("/opt/ml/"):
            # Assume it maps to training channel
            result["data"]["data_path"] = f"{CONTAINER_INPUT_BASE}/training"

    # Rewrite model.base_model_path
    if "model" in result and "base_model_path" in result["model"]:
        original = result["model"]["base_model_path"]
        if original in channel_config.path_mapping:
            result["model"]["base_model_path"] = channel_config.path_mapping[original]
        elif not original.startswith("/opt/ml/"):
            result["model"]["base_model_path"] = f"{CONTAINER_INPUT_BASE}/model"

    # Set output directory
    if "logging" not in result:
        result["logging"] = {}
    result["logging"]["output_dir"] = CONTAINER_MODEL_BASE

    return result

