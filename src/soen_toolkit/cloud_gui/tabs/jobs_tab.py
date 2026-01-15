"""Jobs monitoring tab with log streaming."""

from __future__ import annotations

from datetime import datetime
import logging
import os

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class JobsTab(QWidget):
    """Tab for monitoring jobs with log streaming."""

    def __init__(self) -> None:
        super().__init__()
        self._log_timer: QTimer | None = None
        self._current_log_job: str | None = None
        self._last_log_token: str | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI."""
        layout = QVBoxLayout(self)

        # Create splitter for jobs list and logs
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top section: Jobs table
        jobs_widget = QWidget()
        jobs_layout = QVBoxLayout(jobs_widget)
        jobs_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Recent Training Jobs"))
        header_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(self.refresh_btn)

        jobs_layout.addLayout(header_layout)

        # Jobs table
        self.jobs_table = QTableWidget()
        self.jobs_table.setColumnCount(6)
        self.jobs_table.setHorizontalHeaderLabels([
            "Job Name",
            "Status",
            "Message",
            "Instance",
            "Created",
            "Duration",
        ])
        self.jobs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.jobs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.jobs_table.itemSelectionChanged.connect(self._on_job_selected)

        # Make columns stretch
        header = self.jobs_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)  # Job name
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Status
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Message
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Instance
            header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Created
            header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # Duration
        self.jobs_table.setColumnWidth(0, 250)  # Job name default width

        jobs_layout.addWidget(self.jobs_table)

        # Action buttons
        button_layout = QHBoxLayout()

        self.stop_job_btn = QPushButton("Stop Job")
        self.stop_job_btn.clicked.connect(self._stop_job)
        button_layout.addWidget(self.stop_job_btn)

        self.open_console_btn = QPushButton("Open in Console")
        self.open_console_btn.clicked.connect(self._open_in_console)
        button_layout.addWidget(self.open_console_btn)

        button_layout.addStretch()

        self.status_label = QLabel()
        button_layout.addWidget(self.status_label)

        jobs_layout.addLayout(button_layout)

        splitter.addWidget(jobs_widget)

        # Bottom section: Log viewer
        logs_widget = QWidget()
        logs_layout = QVBoxLayout(logs_widget)
        logs_layout.setContentsMargins(0, 0, 0, 0)

        # Log header
        log_header = QHBoxLayout()
        self.log_title = QLabel("Logs: Select a job above")
        log_header.addWidget(self.log_title)
        log_header.addStretch()

        self.auto_scroll_check = QCheckBox("Auto-scroll")
        self.auto_scroll_check.setChecked(True)
        log_header.addWidget(self.auto_scroll_check)

        self.stream_logs_check = QCheckBox("Stream logs")
        self.stream_logs_check.setChecked(False)
        self.stream_logs_check.toggled.connect(self._toggle_log_streaming)
        log_header.addWidget(self.stream_logs_check)

        self.clear_logs_btn = QPushButton("Clear")
        self.clear_logs_btn.clicked.connect(self._clear_logs)
        log_header.addWidget(self.clear_logs_btn)

        logs_layout.addLayout(log_header)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monaco", 11))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
            }
        """)
        logs_layout.addWidget(self.log_text)

        splitter.addWidget(logs_widget)

        # Set initial splitter sizes (60% jobs, 40% logs)
        splitter.setSizes([300, 200])

        layout.addWidget(splitter)

    def refresh(self) -> None:
        """Refresh jobs list from SageMaker."""
        try:
            import boto3

            region = os.environ.get("AWS_REGION", "us-east-1")
            sm = boto3.client("sagemaker", region_name=region)

            response = sm.list_training_jobs(
                MaxResults=20,
                SortBy="CreationTime",
                SortOrder="Descending",
            )

            jobs = response.get("TrainingJobSummaries", [])

            self.jobs_table.setRowCount(len(jobs))

            for row, job_summary in enumerate(jobs):
                name = job_summary.get("TrainingJobName", "")
                status = job_summary.get("TrainingJobStatus", "Unknown")
                created = job_summary.get("CreationTime")
                end_time = job_summary.get("TrainingEndTime")

                # Job name
                self.jobs_table.setItem(row, 0, QTableWidgetItem(name))

                # Status with color
                status_item = QTableWidgetItem(status)
                if status == "Completed":
                    status_item.setForeground(Qt.GlobalColor.darkGreen)
                elif status == "Failed":
                    status_item.setForeground(Qt.GlobalColor.red)
                elif status == "InProgress":
                    status_item.setForeground(Qt.GlobalColor.blue)
                elif status == "Stopped":
                    status_item.setForeground(Qt.GlobalColor.darkYellow)
                self.jobs_table.setItem(row, 1, status_item)

                # Get detailed info for status message and instance type
                try:
                    detail = sm.describe_training_job(TrainingJobName=name)

                    # Use SecondaryStatus for current state (more accurate than message)
                    secondary_status = detail.get("SecondaryStatus", "")
                    transitions = detail.get("SecondaryStatusTransitions", [])

                    # Map secondary status to friendly message
                    status_messages = {
                        "Starting": "Starting...",
                        "Pending": "Pending...",
                        "Downloading": "Downloading image...",
                        "Training": "Training...",
                        "Uploading": "Uploading model...",
                        "Completed": "Completed",
                        "Failed": "Failed",
                        "Stopping": "Stopping...",
                        "Stopped": "Stopped",
                    }
                    message = status_messages.get(secondary_status, secondary_status)

                    # Check for capacity issues in the message
                    if transitions:
                        last_msg = transitions[-1].get("StatusMessage", "")
                        if "capacity" in last_msg.lower():
                            message = "Waiting for capacity..."

                    message_item = QTableWidgetItem(message)
                    if "capacity" in message.lower() or "waiting" in message.lower():
                        message_item.setForeground(Qt.GlobalColor.darkYellow)
                    elif "Training" in message:
                        message_item.setForeground(Qt.GlobalColor.darkGreen)
                    self.jobs_table.setItem(row, 2, message_item)

                    # Instance type
                    resource = detail.get("ResourceConfig", {})
                    instance_type = resource.get("InstanceType", "-")
                    self.jobs_table.setItem(row, 3, QTableWidgetItem(instance_type))

                except Exception:
                    self.jobs_table.setItem(row, 2, QTableWidgetItem("-"))
                    self.jobs_table.setItem(row, 3, QTableWidgetItem("-"))

                # Created time
                if created:
                    created_str = created.strftime("%Y-%m-%d %H:%M")
                    self.jobs_table.setItem(row, 4, QTableWidgetItem(created_str))

                # Duration
                if created and end_time:
                    duration = end_time - created
                    hours = duration.total_seconds() / 3600
                    self.jobs_table.setItem(row, 5, QTableWidgetItem(f"{hours:.1f}h"))
                elif created and status == "InProgress":
                    duration = datetime.now(created.tzinfo) - created
                    minutes = duration.total_seconds() / 60
                    self.jobs_table.setItem(row, 5, QTableWidgetItem(f"{minutes:.0f}m"))
                else:
                    self.jobs_table.setItem(row, 5, QTableWidgetItem("-"))

            self.status_label.setText(f"Loaded {len(jobs)} jobs")

        except Exception as e:
            logger.exception("Failed to refresh jobs")
            self.status_label.setText(f"✗ Error: {str(e)[:50]}")

    def _get_selected_job(self) -> str | None:
        """Get the selected job name."""
        selected = self.jobs_table.selectedItems()
        if not selected:
            return None
        row = selected[0].row()
        item = self.jobs_table.item(row, 0)
        return item.text() if item is not None else None

    def _on_job_selected(self) -> None:
        """Handle job selection change."""
        job_name = self._get_selected_job()
        if job_name:
            self.log_title.setText(f"Logs: {job_name}")
            self._current_log_job = job_name
            self._last_log_token = None
            # First show job details, then fetch logs
            self._show_job_details(job_name)
            self._fetch_logs(initial=True)

    def _show_job_details(self, job_name: str) -> None:
        """Show detailed job information."""
        try:
            import boto3

            region = os.environ.get("AWS_REGION", "us-east-1")
            sm = boto3.client("sagemaker", region_name=region)
            job = sm.describe_training_job(TrainingJobName=job_name)

            self.log_text.clear()

            # Status
            status = job.get("TrainingJobStatus", "Unknown")
            secondary = job.get("SecondaryStatus", "N/A")
            status_color = {
                "Completed": "#4caf50",
                "Failed": "#f44336",
                "Stopped": "#ff9800",
                "InProgress": "#2196f3",
            }.get(status, "#d4d4d4")

            self.log_text.append(f'<span style="color:#888">{"="*50}</span>')
            self.log_text.append(f'<span style="color:{status_color}; font-weight:bold">Status: {status} ({secondary})</span>')

            if job.get("FailureReason"):
                self.log_text.append(f'<span style="color:#f44336">Failure: {job["FailureReason"]}</span>')

            # Instance
            resource = job.get("ResourceConfig", {})
            instance_type = resource.get("InstanceType", "N/A")
            instance_count = resource.get("InstanceCount", 1)
            spot = "Yes" if job.get("EnableManagedSpotTraining") else "No"

            self.log_text.append(f'<span style="color:#888">Instance: {instance_type} x{instance_count} (Spot: {spot})</span>')

            # Timing & Cost
            billable = job.get("BillableTimeInSeconds", 0)
            if billable:
                minutes = billable / 60
                # Rough cost estimate
                hourly_prices = {"ml.g5.xlarge": 1.006, "ml.g5.2xlarge": 1.515}
                hourly = hourly_prices.get(instance_type, 1.0)
                cost = (billable / 3600) * hourly
                if job.get("EnableManagedSpotTraining"):
                    cost *= 0.35
                self.log_text.append(f'<span style="color:#888">Billable: {minutes:.1f} min | Est. Cost: ${cost:.2f}</span>')

            # Output
            artifacts = job.get("ModelArtifacts", {})
            if artifacts.get("S3ModelArtifacts"):
                self.log_text.append(f'<span style="color:#4fc3f7">Output: {artifacts["S3ModelArtifacts"]}</span>')

            # Show status history with messages (shows capacity issues, etc.)
            status_history = job.get("SecondaryStatusTransitions", [])
            if status_history:
                self.log_text.append("")
                self.log_text.append('<span style="color:#888">--- Status History ---</span>')
                for transition in status_history[-5:]:  # Last 5 transitions
                    ts = transition.get("StartTime")
                    ts_str = ts.strftime("%H:%M:%S") if ts else ""
                    status_name = transition.get("Status", "")
                    message = transition.get("StatusMessage", "")

                    # Color important messages
                    if "capacity" in message.lower() or "insufficient" in message.lower():
                        color = "#ff9800"  # Orange for capacity issues
                        self.log_text.append(f'<span style="color:{color}">[{ts_str}] {status_name}: {message}</span>')
                    elif "error" in message.lower() or "fail" in message.lower():
                        color = "#f44336"  # Red for errors
                        self.log_text.append(f'<span style="color:{color}">[{ts_str}] {status_name}: {message}</span>')
                    else:
                        self.log_text.append(f'<span style="color:#888">[{ts_str}] {status_name}: {message[:80]}</span>')

            self.log_text.append(f'<span style="color:#888">{"="*50}</span>')
            self.log_text.append("")

        except Exception as e:
            self.log_text.append(f'<span style="color:#f44336">Could not fetch job details: {e}</span>')

    def _toggle_log_streaming(self, enabled: bool) -> None:
        """Toggle automatic log streaming."""
        if enabled:
            if self._log_timer is None:
                self._log_timer = QTimer()
                self._log_timer.timeout.connect(self._fetch_new_logs)
            self._log_timer.start(3000)  # Refresh every 3 seconds
        elif self._log_timer:
            self._log_timer.stop()

    def _fetch_logs(self, initial: bool = False) -> None:
        """Fetch logs for the current job."""
        if not self._current_log_job:
            return

        try:
            import boto3

            region = os.environ.get("AWS_REGION", "us-east-1")
            logs_client = boto3.client("logs", region_name=region)

            log_group = "/aws/sagemaker/TrainingJobs"

            # Find the actual log stream for this job
            # Format is: {job_name}/algo-1-{timestamp}
            try:
                streams = logs_client.describe_log_streams(
                    logGroupName=log_group,
                    logStreamNamePrefix=f"{self._current_log_job}/",
                    limit=5,
                )
                if not streams.get("logStreams"):
                    self.log_text.append('<span style="color:#888">No logs yet - job may still be starting...</span>')
                    return
                # Get the most recent stream (sort by creation time)
                sorted_streams = sorted(
                    streams["logStreams"],
                    key=lambda s: s.get("creationTime", 0),
                    reverse=True
                )
                log_stream = sorted_streams[0]["logStreamName"]
            except logs_client.exceptions.ResourceNotFoundException:
                self.log_text.append('<span style="color:#888">No logs yet - job may still be starting...</span>')
                return
            except Exception as e:
                self.log_text.append(f'<span style="color:#f44336">Error finding logs: {e}</span>')
                return

            # Fetch log events
            kwargs = {
                "logGroupName": log_group,
                "logStreamName": log_stream,
                "startFromHead": True,
                "limit": 500 if initial else 100,
            }

            if not initial and self._last_log_token:
                kwargs["nextToken"] = self._last_log_token

            response = logs_client.get_log_events(**kwargs)

            events = response.get("events", [])
            self._last_log_token = response.get("nextForwardToken")

            if initial:
                self.log_text.clear()

            for event in events:
                timestamp = datetime.fromtimestamp(event["timestamp"] / 1000)
                time_str = timestamp.strftime("%H:%M:%S")
                message = event["message"]

                # Color code based on content
                if "ERROR" in message or "error" in message.lower():
                    self.log_text.append(f'<span style="color:#f44336">[{time_str}] {message}</span>')
                elif "WARNING" in message or "warn" in message.lower():
                    self.log_text.append(f'<span style="color:#ff9800">[{time_str}] {message}</span>')
                elif "INFO" in message:
                    self.log_text.append(f'<span style="color:#4fc3f7">[{time_str}] {message}</span>')
                else:
                    self.log_text.append(f'<span style="color:#d4d4d4">[{time_str}] {message}</span>')

            if events and self.auto_scroll_check.isChecked():
                scrollbar = self.log_text.verticalScrollBar()
                if scrollbar is not None:
                    scrollbar.setValue(scrollbar.maximum())

        except Exception as e:
            if initial:
                self.log_text.append('<span style="color:#888">No logs available yet. Job may still be starting...</span>')
                self.log_text.append(f'<span style="color:#666">{str(e)}</span>')

    def _fetch_new_logs(self) -> None:
        """Fetch only new logs (for streaming)."""
        self._fetch_logs(initial=False)

    def _clear_logs(self) -> None:
        """Clear the log display."""
        self.log_text.clear()
        self._last_log_token = None

    def _open_in_console(self) -> None:
        """Open job in AWS Console."""
        job_name = self._get_selected_job()
        if not job_name:
            QMessageBox.warning(self, "No Selection", "Please select a job first.")
            return

        import webbrowser

        region = os.environ.get("AWS_REGION", "us-east-1")
        url = (
            f"https://{region}.console.aws.amazon.com/sagemaker/home"
            f"?region={region}#/jobs/{job_name}"
        )
        webbrowser.open(url)

    def _stop_job(self) -> None:
        """Stop the selected job."""
        job_name = self._get_selected_job()
        if not job_name:
            QMessageBox.warning(self, "No Selection", "Please select a job first.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Stop",
            f"Stop job:\n{job_name}\n\nAre you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            from soen_toolkit.cloud.config import AWSConfig, CloudConfig
            from soen_toolkit.cloud.session import CloudSession

            role = os.environ.get("SOEN_SM_ROLE", "")
            bucket = os.environ.get("SOEN_SM_BUCKET", "")
            region = os.environ.get("AWS_REGION", "us-east-1")

            config = CloudConfig(
                aws=AWSConfig(role=role, bucket=bucket, region=region)
            )
            session = CloudSession(config, skip_validation=True)
            sm = session.sagemaker_client()

            sm.stop_training_job(TrainingJobName=job_name)

            QMessageBox.information(self, "Job Stopped", f"Stop request sent for:\n{job_name}")
            QTimer.singleShot(2000, self.refresh)

        except Exception as e:
            QMessageBox.warning(self, "Stop Failed", f"Could not stop job:\n\n{e}")

