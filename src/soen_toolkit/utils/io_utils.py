"""Optimized saving for SOEN checkpoints to reduce IO latency."""

import logging
from pathlib import Path
import queue
import threading

logger = logging.getLogger(__name__)

class AsyncArtifactUploader:
    """Background uploader for MLflow artifacts to avoid blocking training."""

    def __init__(self, mlflow_experiment, run_id: str) -> None:
        self.experiment = mlflow_experiment
        self.run_id = run_id
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True, name="MLflowArtifactUploader")
        self.worker_thread.start()
        logger.info(f"Started AsyncArtifactUploader for run {run_id}")

    def enqueue(self, file_path: Path) -> None:
        """Add a file to the upload queue."""
        self.queue.put(Path(file_path))

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                try:
                    # Get file with timeout to allow checking stop_event
                    file_path = self.queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if file_path is None:
                    continue

                # Check if file still exists (it might have been deleted by top-k pruning)
                if not file_path.exists():
                    logger.debug(f"Skipping upload of deleted file: {file_path}")
                    self.queue.task_done()
                    continue

                logger.debug(f"Uploading artifact to MLflow: {file_path.name}")
                try:
                    # Use log_artifact from the experiment object (MlflowClient or similar)
                    # Note: PL's MLFlowLogger.experiment returns the client
                    self.experiment.log_artifact(self.run_id, str(file_path))
                except Exception as e:
                    # Check for common network errors to avoid spamming error logs
                    error_msg = str(e).lower()
                    if "connection" in error_msg or "timeout" in error_msg:
                        logger.warning(f"Network error uploading {file_path.name}: {e}")
                    else:
                        logger.warning(f"Failed to upload artifact {file_path.name}: {e}")

                self.queue.task_done()

            except Exception as e:
                logger.error(f"Unexpected error in artifact uploader worker: {e}")

    def stop(self) -> None:
        """Stop the worker thread."""
        self.stop_event.set()
        # Do not join here to avoid blocking main thread on exit if queue is full
        # The daemon thread will be killed when process exits

def background_save_checkpoint(
    save_fn,
    *args,
    **kwargs
) -> None:
    """Execute a save function in a background thread to avoid blocking training."""
    def _wrapper():
        try:
            save_fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"Background save failed: {e}", exc_info=True)

    thread = threading.Thread(target=_wrapper, daemon=True)
    thread.start()
    logger.debug("Started background checkpoint save")

