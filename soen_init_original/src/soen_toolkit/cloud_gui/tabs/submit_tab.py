"""Job submission tab."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class SubmitTab(QWidget):
    """Tab for submitting jobs."""

    job_submitted = pyqtSignal(str)  # Emits job name

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI."""
        layout = QVBoxLayout(self)

        # Training Config Group
        config_group = QGroupBox("Training Configuration")
        config_layout = QFormLayout(config_group)

        config_row = QHBoxLayout()
        self.config_path_input = QLineEdit()
        self.config_path_input.setPlaceholderText("Path to training config YAML")
        config_row.addWidget(self.config_path_input)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_config)
        config_row.addWidget(browse_btn)

        config_layout.addRow("Config File:", config_row)

        self.job_type_combo = QComboBox()
        self.job_type_combo.addItems(["training", "inference", "processing"])
        config_layout.addRow("Job Type:", self.job_type_combo)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["pytorch", "jax"])
        config_layout.addRow("Backend:", self.backend_combo)

        layout.addWidget(config_group)

        # Instance Configuration Group
        instance_group = QGroupBox("Instance Configuration")
        instance_layout = QFormLayout(instance_group)

        self.instance_type_combo = QComboBox()
        self.instance_type_combo.addItems([
            "ml.g5.xlarge",
            "ml.g5.2xlarge",
            "ml.g5.4xlarge",
            "ml.g5.8xlarge",
            "ml.g5.12xlarge",
            "ml.p4d.24xlarge",
            "ml.g4dn.xlarge",
            "ml.g4dn.2xlarge",
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
        ])
        self.instance_type_combo.currentTextChanged.connect(self._update_cost_estimate)
        instance_layout.addRow("Instance Type:", self.instance_type_combo)

        self.instance_count_spin = QSpinBox()
        self.instance_count_spin.setRange(1, 8)
        self.instance_count_spin.setValue(1)
        self.instance_count_spin.valueChanged.connect(self._update_cost_estimate)
        instance_layout.addRow("Instance Count:", self.instance_count_spin)

        self.max_runtime_spin = QDoubleSpinBox()
        self.max_runtime_spin.setRange(0.5, 72)
        self.max_runtime_spin.setValue(8.0)
        self.max_runtime_spin.setSuffix(" hours")
        self.max_runtime_spin.valueChanged.connect(self._update_cost_estimate)
        instance_layout.addRow("Max Runtime:", self.max_runtime_spin)

        self.use_spot_check = QCheckBox("Use Spot Instances (65% cheaper)")
        self.use_spot_check.setChecked(True)
        self.use_spot_check.stateChanged.connect(self._update_cost_estimate)
        instance_layout.addRow("", self.use_spot_check)

        layout.addWidget(instance_group)

        # Cost Estimate Group
        cost_group = QGroupBox("Cost Estimate")
        cost_layout = QVBoxLayout(cost_group)

        self.cost_label = QLabel()
        self.cost_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        cost_layout.addWidget(self.cost_label)

        self.cost_details_label = QLabel()
        self.cost_details_label.setWordWrap(True)
        cost_layout.addWidget(self.cost_details_label)

        layout.addWidget(cost_group)

        # Submit Buttons
        button_layout = QHBoxLayout()

        self.estimate_btn = QPushButton("📊 Estimate Cost Only")
        self.estimate_btn.clicked.connect(self._update_cost_estimate)
        button_layout.addWidget(self.estimate_btn)

        self.submit_btn = QPushButton("🚀 Submit Job")
        self.submit_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }"
        )
        self.submit_btn.clicked.connect(self._submit_job)
        button_layout.addWidget(self.submit_btn)

        layout.addLayout(button_layout)

        # Status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.status_text)

        layout.addStretch()

        # Initial cost estimate
        self._update_cost_estimate()

    def _browse_config(self) -> None:
        """Browse for training config file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Configuration",
            str(Path.home()),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self.config_path_input.setText(path)

    def _update_cost_estimate(self) -> None:
        """Update cost estimate based on current settings."""
        try:
            from soen_toolkit.cloud.cost import CostEstimator

            estimator = CostEstimator()
            estimate = estimator.estimate_from_params(
                instance_type=self.instance_type_combo.currentText(),
                instance_count=self.instance_count_spin.value(),
                max_runtime_hours=self.max_runtime_spin.value(),
                use_spot=self.use_spot_check.isChecked(),
            )

            if self.use_spot_check.isChecked():
                self.cost_label.setText(f"Estimated Cost: ${estimate.spot_total:.2f}")
            else:
                self.cost_label.setText(f"Estimated Cost: ${estimate.on_demand_total:.2f}")

            self.cost_details_label.setText(
                f"On-Demand: ${estimate.on_demand_hourly:.3f}/hr = ${estimate.on_demand_total:.2f} max\n"
                f"Spot (~65% off): ${estimate.spot_hourly:.3f}/hr = ${estimate.spot_total:.2f} max"
            )

        except Exception as e:
            self.cost_label.setText("Cost estimate unavailable")
            self.cost_details_label.setText(str(e))

    def _submit_job(self) -> None:
        """Submit the training job."""
        config_path = self.config_path_input.text().strip()
        if not config_path:
            QMessageBox.warning(self, "Missing Config", "Please select a training config file.")
            return

        if not Path(config_path).exists():
            QMessageBox.warning(self, "File Not Found", f"Config file not found:\n{config_path}")
            return

        # Check credentials
        role = os.environ.get("SOEN_SM_ROLE", "")
        bucket = os.environ.get("SOEN_SM_BUCKET", "")
        if not role or not bucket:
            QMessageBox.warning(
                self,
                "Missing Credentials",
                "AWS credentials not configured.\n\nGo to the Credentials tab to set up.",
            )
            return

        # Confirm submission
        instance_type = self.instance_type_combo.currentText()
        instance_count = self.instance_count_spin.value()
        use_spot = self.use_spot_check.isChecked()

        reply = QMessageBox.question(
            self,
            "Confirm Submission",
            f"Submit training job with:\n\n"
            f"Instance: {instance_type} x {instance_count}\n"
            f"Spot: {use_spot}\n"
            f"Config: {Path(config_path).name}\n\n"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self.status_text.append("Submitting job...")

        try:
            from soen_toolkit.cloud.config import (
                AWSConfig,
                Backend,
                CloudConfig,
                InstanceConfig,
                JobConfig,
                JobType,
            )
            from soen_toolkit.cloud.jobs import TrainingJob
            from soen_toolkit.cloud.session import CloudSession

            # Create configuration
            region = os.environ.get("AWS_REGION", "us-east-1")
            config = CloudConfig(
                aws=AWSConfig(role=role, bucket=bucket, region=region)
            )

            session = CloudSession(config)

            backend = Backend.PYTORCH if self.backend_combo.currentText() == "pytorch" else Backend.JAX

            job_config = JobConfig(
                job_type=JobType.TRAINING,
                backend=backend,
                aws=config.aws,
                instance=InstanceConfig(
                    instance_type=instance_type,
                    instance_count=instance_count,
                    use_spot=use_spot,
                    max_runtime_hours=self.max_runtime_spin.value(),
                ),
                training_config_path=Path(config_path),
            )

            job = TrainingJob(job_config)
            job.validate()

            job_name = job.submit(session)

            self.status_text.append(f"✓ Job submitted: {job_name}")
            self.status_text.append(f"  Instance: {instance_type} x {instance_count}")
            self.status_text.append(f"  Spot: {use_spot}")

            QMessageBox.information(
                self,
                "Job Submitted",
                f"Training job submitted successfully!\n\nJob Name: {job_name}",
            )

            self.job_submitted.emit(job_name)

        except Exception as e:
            self.status_text.append(f"✗ Submission failed: {e}")
            QMessageBox.warning(self, "Submission Failed", f"Could not submit job:\n\n{e}")

