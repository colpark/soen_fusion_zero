"""S3 Transfer tab for uploading and downloading files."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TransferWorker(QThread):
    """Worker thread for S3 transfers to keep UI responsive."""

    progress = pyqtSignal(str)  # Status message
    finished_signal = pyqtSignal(bool, str)  # (success, message)

    def __init__(
        self,
        operation: str,
        s3_uri: str,
        local_path: str,
        is_directory: bool,
    ) -> None:
        super().__init__()
        self.operation = operation  # "download" or "upload"
        self.s3_uri = s3_uri
        self.local_path = local_path
        self.is_directory = is_directory

    def run(self) -> None:
        """Execute the transfer operation."""
        try:
            import boto3

            # Use boto3 directly for flexibility (any bucket, not just configured one)
            region = os.environ.get("AWS_REGION", "us-east-1")
            s3 = boto3.client("s3", region_name=region)

            if self.operation == "download":
                self._do_download(s3)
            else:
                self._do_upload(s3)

        except Exception as e:
            logger.exception("Transfer failed")
            self.finished_signal.emit(False, str(e))

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and key."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI (must start with s3://): {s3_uri}")

        without_scheme = s3_uri[5:]
        parts = without_scheme.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        if not bucket:
            raise ValueError(f"Invalid S3 URI (missing bucket): {s3_uri}")

        return bucket, key

    def _do_download(self, s3: Any) -> None:
        """Execute download operation."""
        bucket, key = self._parse_s3_uri(self.s3_uri)
        local = Path(self.local_path)

        if self.is_directory:
            # Download all objects under prefix
            local.mkdir(parents=True, exist_ok=True)
            prefix = key.rstrip("/") + "/" if key else ""
            paginator = s3.get_paginator("list_objects_v2")

            count = 0
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    s3_key = obj["Key"]
                    rel_path = s3_key[len(prefix):] if prefix else s3_key

                    if not rel_path:
                        continue

                    local_path = local / rel_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    self.progress.emit(f"Downloading: {rel_path}")
                    s3.download_file(bucket, s3_key, str(local_path))
                    count += 1

            self.finished_signal.emit(True, f"Downloaded {count} files to {self.local_path}")
        else:
            # Single file download
            if not key:
                raise ValueError("S3 URI must include a file path")

            # Determine local file path
            if local.is_dir():
                # User selected a directory, use the S3 filename
                filename = key.split("/")[-1]
                local = local / filename

            local.parent.mkdir(parents=True, exist_ok=True)
            self.progress.emit(f"Downloading: {key}")
            s3.download_file(bucket, key, str(local))
            self.finished_signal.emit(True, f"Downloaded to {local}")

    def _do_upload(self, s3: Any) -> None:
        """Execute upload operation."""
        bucket, prefix = self._parse_s3_uri(self.s3_uri)
        local = Path(self.local_path)

        if self.is_directory:
            # Upload all files in directory
            prefix = prefix.rstrip("/")
            count = 0

            for root, _, files in os.walk(local):
                for filename in files:
                    local_path = Path(root) / filename
                    rel_path = local_path.relative_to(local).as_posix()
                    s3_key = f"{prefix}/{rel_path}" if prefix else rel_path

                    self.progress.emit(f"Uploading: {rel_path}")
                    s3.upload_file(str(local_path), bucket, s3_key)
                    count += 1

            self.finished_signal.emit(True, f"Uploaded {count} files to s3://{bucket}/{prefix}")
        else:
            # Single file upload
            filename = local.name

            # If prefix ends without a filename, append the local filename
            if not prefix or prefix.endswith("/"):
                s3_key = f"{prefix}{filename}"
            else:
                s3_key = prefix

            self.progress.emit(f"Uploading: {filename}")
            s3.upload_file(str(local), bucket, s3_key)
            self.finished_signal.emit(True, f"Uploaded to s3://{bucket}/{s3_key}")


class S3TransferTab(QWidget):
    """Tab for S3 file transfers (upload and download)."""

    def __init__(self) -> None:
        super().__init__()
        self._worker: TransferWorker | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI."""
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Transfer files and folders between S3 and your local filesystem. "
            "Supports any S3 bucket you have access to."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Download Section
        download_group = QGroupBox("Download from S3")
        download_layout = QFormLayout(download_group)

        self.download_s3_input = QLineEdit()
        self.download_s3_input.setPlaceholderText("s3://bucket-name/path/to/file-or-folder")
        download_layout.addRow("S3 Path:", self.download_s3_input)

        local_dest_row = QHBoxLayout()
        self.download_local_input = QLineEdit()
        self.download_local_input.setPlaceholderText("Local destination folder")
        local_dest_row.addWidget(self.download_local_input)

        download_browse_btn = QPushButton("Browse...")
        download_browse_btn.clicked.connect(self._browse_download_destination)
        local_dest_row.addWidget(download_browse_btn)

        download_layout.addRow("Local Destination:", local_dest_row)

        download_btn_row = QHBoxLayout()
        self.download_file_btn = QPushButton("Download File")
        self.download_file_btn.clicked.connect(lambda: self._start_download(is_directory=False))
        download_btn_row.addWidget(self.download_file_btn)

        self.download_folder_btn = QPushButton("Download Folder")
        self.download_folder_btn.clicked.connect(lambda: self._start_download(is_directory=True))
        download_btn_row.addWidget(self.download_folder_btn)

        download_btn_row.addStretch()
        download_layout.addRow("", download_btn_row)

        layout.addWidget(download_group)

        # Upload Section
        upload_group = QGroupBox("Upload to S3")
        upload_layout = QFormLayout(upload_group)

        local_source_row = QHBoxLayout()
        self.upload_local_input = QLineEdit()
        self.upload_local_input.setPlaceholderText("Local file or folder path")
        local_source_row.addWidget(self.upload_local_input)

        upload_browse_file_btn = QPushButton("File...")
        upload_browse_file_btn.clicked.connect(self._browse_upload_file)
        local_source_row.addWidget(upload_browse_file_btn)

        upload_browse_folder_btn = QPushButton("Folder...")
        upload_browse_folder_btn.clicked.connect(self._browse_upload_folder)
        local_source_row.addWidget(upload_browse_folder_btn)

        upload_layout.addRow("Local Source:", local_source_row)

        self.upload_s3_input = QLineEdit()
        self.upload_s3_input.setPlaceholderText("s3://bucket-name/path/to/destination")
        upload_layout.addRow("S3 Destination:", self.upload_s3_input)

        upload_btn_row = QHBoxLayout()
        self.upload_btn = QPushButton("Upload")
        self.upload_btn.clicked.connect(self._start_upload)
        upload_btn_row.addWidget(self.upload_btn)
        upload_btn_row.addStretch()
        upload_layout.addRow("", upload_btn_row)

        layout.addWidget(upload_group)

        # Progress Section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.current_file_label = QLabel("")
        progress_layout.addWidget(self.current_file_label)

        layout.addWidget(progress_group)

        # Status Log
        layout.addWidget(QLabel("Transfer Log:"))
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(150)
        layout.addWidget(self.status_log)

        layout.addStretch()

    def _browse_download_destination(self) -> None:
        """Browse for download destination folder."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Download Destination",
            str(Path.home()),
        )
        if path:
            self.download_local_input.setText(path)

    def _browse_upload_file(self) -> None:
        """Browse for a file to upload."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File to Upload",
            str(Path.home()),
            "All Files (*)",
        )
        if path:
            self.upload_local_input.setText(path)

    def _browse_upload_folder(self) -> None:
        """Browse for a folder to upload."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Upload",
            str(Path.home()),
        )
        if path:
            self.upload_local_input.setText(path)

    def _validate_s3_uri(self, uri: str) -> bool:
        """Validate S3 URI format."""
        if not uri.strip():
            QMessageBox.warning(self, "Missing S3 Path", "Please enter an S3 path.")
            return False

        if not uri.startswith("s3://"):
            QMessageBox.warning(
                self,
                "Invalid S3 Path",
                "S3 path must start with 's3://'\n\n"
                "Example: s3://my-bucket/path/to/files",
            )
            return False

        return True

    def _set_ui_busy(self, busy: bool) -> None:
        """Enable/disable UI during transfers."""
        self.download_file_btn.setEnabled(not busy)
        self.download_folder_btn.setEnabled(not busy)
        self.upload_btn.setEnabled(not busy)
        self.progress_bar.setVisible(busy)

        if not busy:
            self.current_file_label.setText("")

    def _start_download(self, is_directory: bool) -> None:
        """Start a download operation."""
        s3_uri = self.download_s3_input.text().strip()
        local_path = self.download_local_input.text().strip()

        if not self._validate_s3_uri(s3_uri):
            return

        if not local_path:
            QMessageBox.warning(self, "Missing Destination", "Please select a local destination.")
            return

        self._log(f"Starting download: {s3_uri}")
        self._set_ui_busy(True)

        self._worker = TransferWorker(
            operation="download",
            s3_uri=s3_uri,
            local_path=local_path,
            is_directory=is_directory,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_transfer_finished)
        self._worker.start()

    def _start_upload(self) -> None:
        """Start an upload operation."""
        local_path = self.upload_local_input.text().strip()
        s3_uri = self.upload_s3_input.text().strip()

        if not local_path:
            QMessageBox.warning(self, "Missing Source", "Please select a local file or folder.")
            return

        if not Path(local_path).exists():
            QMessageBox.warning(self, "Not Found", f"Local path does not exist:\n{local_path}")
            return

        if not self._validate_s3_uri(s3_uri):
            return

        is_directory = Path(local_path).is_dir()

        self._log(f"Starting upload: {local_path}")
        self._set_ui_busy(True)

        self._worker = TransferWorker(
            operation="upload",
            s3_uri=s3_uri,
            local_path=local_path,
            is_directory=is_directory,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_transfer_finished)
        self._worker.start()

    def _on_progress(self, message: str) -> None:
        """Handle progress update from worker."""
        self.current_file_label.setText(message)

    def _on_transfer_finished(self, success: bool, message: str) -> None:
        """Handle transfer completion."""
        self._set_ui_busy(False)

        if success:
            self._log(f"[OK] {message}")
            QMessageBox.information(self, "Transfer Complete", message)
        else:
            self._log(f"[FAILED] {message}")
            QMessageBox.warning(self, "Transfer Failed", message)

    def _log(self, message: str) -> None:
        """Add a message to the status log."""
        self.status_log.append(message)
        # Auto-scroll to bottom
        scrollbar = self.status_log.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

