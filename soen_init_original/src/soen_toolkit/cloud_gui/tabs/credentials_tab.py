"""Credentials configuration tab."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import yaml

logger = logging.getLogger(__name__)


class CredentialsTab(QWidget):
    """Tab for configuring AWS credentials."""

    credentials_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()
        self._load_from_env()

    def _setup_ui(self) -> None:
        """Set up the UI."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Configure your AWS credentials for cloud training. "
            "Credentials are stored in environment variables (not persisted to disk)."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # AWS Credentials Group
        aws_group = QGroupBox("AWS Credentials")
        aws_layout = QFormLayout(aws_group)

        self.role_input = QLineEdit()
        self.role_input.setPlaceholderText("arn:aws:iam::123456789012:role/SageMakerRole")
        aws_layout.addRow("SageMaker Role ARN:", self.role_input)

        self.bucket_input = QLineEdit()
        self.bucket_input.setPlaceholderText("my-training-bucket")
        aws_layout.addRow("S3 Bucket:", self.bucket_input)

        self.region_input = QComboBox()
        self.region_input.addItems([
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
            "eu-west-1",
            "eu-west-2",
            "eu-central-1",
            "ap-northeast-1",
            "ap-southeast-1",
            "ap-southeast-2",
        ])
        aws_layout.addRow("Region:", self.region_input)

        layout.addWidget(aws_group)

        # MLflow Group (Optional)
        mlflow_group = QGroupBox("MLflow Tracking (Optional)")
        mlflow_layout = QFormLayout(mlflow_group)

        self.mlflow_uri_input = QLineEdit()
        self.mlflow_uri_input.setPlaceholderText("http://mlflow.example.com:5000")
        mlflow_layout.addRow("Tracking URI:", self.mlflow_uri_input)

        layout.addWidget(mlflow_group)

        # Docker Images Group (Optional)
        docker_group = QGroupBox("Docker Images (Optional)")
        docker_layout = QFormLayout(docker_group)

        self.pytorch_image_input = QLineEdit()
        self.pytorch_image_input.setPlaceholderText("account.dkr.ecr.region.amazonaws.com/repo:pytorch")
        docker_layout.addRow("PyTorch Image:", self.pytorch_image_input)

        self.jax_image_input = QLineEdit()
        self.jax_image_input.setPlaceholderText("account.dkr.ecr.region.amazonaws.com/repo:jax")
        docker_layout.addRow("JAX Image:", self.jax_image_input)

        layout.addWidget(docker_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_credentials)
        button_layout.addWidget(self.apply_btn)

        self.validate_btn = QPushButton("Validate")
        self.validate_btn.clicked.connect(self._validate_credentials)
        button_layout.addWidget(self.validate_btn)

        self.save_default_btn = QPushButton("Save as Default")
        self.save_default_btn.clicked.connect(self._save_as_default)
        button_layout.addWidget(self.save_default_btn)

        self.load_file_btn = QPushButton("Load File...")
        self.load_file_btn.clicked.connect(self._on_load_file)
        button_layout.addWidget(self.load_file_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Status area
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)

        layout.addStretch()

    def _load_from_env(self) -> None:
        """Load values from environment variables and auto-apply if valid."""
        # Load from environment
        role = os.environ.get("SOEN_SM_ROLE", "")
        bucket = os.environ.get("SOEN_SM_BUCKET", "")
        region = os.environ.get("AWS_REGION", "us-east-1")
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        pytorch_image = os.environ.get("SOEN_DOCKER_PYTORCH", "")
        jax_image = os.environ.get("SOEN_DOCKER_JAX", "")

        # Try to load from default config files if env is empty
        if not role or not bucket:
            self._try_load_default_config()
            return

        # Populate fields
        self.role_input.setText(role)
        self.bucket_input.setText(bucket)

        idx = self.region_input.findText(region)
        if idx >= 0:
            self.region_input.setCurrentIndex(idx)

        self.mlflow_uri_input.setText(mlflow_uri)
        self.pytorch_image_input.setText(pytorch_image)
        self.jax_image_input.setText(jax_image)

        # Auto-apply if we have valid credentials
        if role and bucket:
            self.status_text.append("Loaded credentials from environment")
            self.credentials_changed.emit()

    def _try_load_default_config(self) -> None:
        """Try to load config from default locations."""
        # Check common config file locations
        config_paths = [
            Path.home() / ".soen" / "cloud_config.yaml",
            Path.home() / ".config" / "soen" / "cloud_config.yaml",
            Path.cwd() / "cloud_config.yaml",
            Path.cwd() / "scripts" / "cloud_test" / "cloud_config.yaml",
        ]

        for path in config_paths:
            if path.exists():
                try:
                    self.load_from_file(str(path))
                    self.status_text.append(f"Auto-loaded config from: {path}")
                    # Apply the loaded config
                    self._apply_credentials()
                    return
                except Exception:
                    continue

        self.status_text.append("No saved credentials found. Please enter them above.")

    def _apply_credentials(self) -> None:
        """Apply credentials to environment variables."""
        role = self.role_input.text().strip()
        bucket = self.bucket_input.text().strip()
        region = self.region_input.currentText()
        mlflow_uri = self.mlflow_uri_input.text().strip()
        pytorch_image = self.pytorch_image_input.text().strip()
        jax_image = self.jax_image_input.text().strip()

        if not role:
            QMessageBox.warning(self, "Missing Field", "SageMaker Role ARN is required.")
            return
        if not bucket:
            QMessageBox.warning(self, "Missing Field", "S3 Bucket is required.")
            return

        # Validate role format
        if not role.startswith("arn:aws:iam::"):
            QMessageBox.warning(
                self,
                "Invalid Role",
                "Role ARN must start with 'arn:aws:iam::'\n\n"
                "Example: arn:aws:iam::123456789012:role/SageMakerRole",
            )
            return

        # Apply to environment
        os.environ["SOEN_SM_ROLE"] = role
        os.environ["SOEN_SM_BUCKET"] = bucket
        os.environ["AWS_REGION"] = region

        if mlflow_uri:
            os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

        if pytorch_image:
            os.environ["SOEN_DOCKER_PYTORCH"] = pytorch_image
        if jax_image:
            os.environ["SOEN_DOCKER_JAX"] = jax_image

        self.status_text.append("Applied credentials to environment")
        self.status_text.append(f"  Role: {role[:50]}...")
        self.status_text.append(f"  Bucket: {bucket}")
        self.status_text.append(f"  Region: {region}")

        self.credentials_changed.emit()

    def _save_as_default(self) -> None:
        """Save current config as default."""
        # Create default config directory
        config_dir = Path.home() / ".soen"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "cloud_config.yaml"

        self.save_to_file(str(config_path))
        self.status_text.append(f"Saved as default: {config_path}")
        QMessageBox.information(
            self,
            "Saved",
            f"Configuration saved to:\n{config_path}\n\n"
            "This will be loaded automatically next time.",
        )

    def _validate_credentials(self) -> None:
        """Validate AWS credentials."""
        self._apply_credentials()

        try:
            from soen_toolkit.cloud.config import AWSConfig, CloudConfig
            from soen_toolkit.cloud.session import CloudSession

            role = os.environ.get("SOEN_SM_ROLE", "")
            bucket = os.environ.get("SOEN_SM_BUCKET", "")
            region = os.environ.get("AWS_REGION", "us-east-1")

            config = CloudConfig(
                aws=AWSConfig(role=role, bucket=bucket, region=region)
            )

            self.status_text.append("\nValidating credentials...")

            session = CloudSession(config)

            self.status_text.append("✓ Authentication successful")
            self.status_text.append(f"  Account: {session.identity.account_id}")
            self.status_text.append(f"  User: {session.identity.user_arn}")
            self.status_text.append(f"✓ Bucket access verified: {bucket}")

            QMessageBox.information(
                self,
                "Validation Successful",
                f"Successfully connected to AWS!\n\n"
                f"Account: {session.identity.account_id}\n"
                f"Bucket: {bucket}",
            )

        except Exception as e:
            self.status_text.append(f"✗ Validation failed: {e}")
            QMessageBox.warning(
                self,
                "Validation Failed",
                f"Could not validate credentials:\n\n{e}",
            )

    def _on_load_file(self) -> None:
        """Load configuration from file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Cloud Configuration",
            str(Path.home()),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self.load_from_file(path)

    def load_from_file(self, path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}

            # Handle 'cloud' wrapper
            if "cloud" in data:
                data = data["cloud"]

            # AWS config
            aws = data.get("aws", {})
            if aws.get("role"):
                self.role_input.setText(aws["role"])
            if aws.get("bucket"):
                self.bucket_input.setText(aws["bucket"])
            if aws.get("region"):
                idx = self.region_input.findText(aws["region"])
                if idx >= 0:
                    self.region_input.setCurrentIndex(idx)

            # MLflow
            mlflow = data.get("mlflow", {})
            if mlflow.get("tracking_uri"):
                self.mlflow_uri_input.setText(mlflow["tracking_uri"])

            # Docker images
            docker = data.get("docker_images", {})
            if docker.get("pytorch"):
                self.pytorch_image_input.setText(docker["pytorch"])
            if docker.get("jax"):
                self.jax_image_input.setText(docker["jax"])

            self.status_text.append(f"✓ Loaded configuration from: {path}")

        except Exception as e:
            QMessageBox.warning(self, "Load Failed", f"Could not load config:\n\n{e}")

    def save_to_file(self, path: str) -> None:
        """Save configuration to YAML file."""
        try:
            data = {
                "aws": {
                    "role": self.role_input.text().strip(),
                    "bucket": self.bucket_input.text().strip(),
                    "region": self.region_input.currentText(),
                },
            }

            mlflow_uri = self.mlflow_uri_input.text().strip()
            if mlflow_uri:
                data["mlflow"] = {"tracking_uri": mlflow_uri}

            pytorch_image = self.pytorch_image_input.text().strip()
            jax_image = self.jax_image_input.text().strip()
            if pytorch_image or jax_image:
                data["docker_images"] = {}
                if pytorch_image:
                    data["docker_images"]["pytorch"] = pytorch_image
                if jax_image:
                    data["docker_images"]["jax"] = jax_image

            with open(path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)

            self.status_text.append(f"✓ Saved configuration to: {path}")

        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Could not save config:\n\n{e}")

