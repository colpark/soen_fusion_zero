#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
from dataclasses import dataclass
from pathlib import Path
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

# Import heavy/optional deps lazily inside functions where possible


# ------------------------------ URL Utils ------------------------------


def _normalize_s3_url(url: str | None) -> str | None:
    """Accept common S3 URL forms and return s3://bucket/prefix or None.
    Supports:
    - s3://bucket/prefix
    - https://bucket.s3.<region>.amazonaws.com/prefix
    - https://s3.<region>.amazonaws.com/bucket/prefix
    - AWS console bucket URL with prefix param.
    """
    if not url:
        return None
    url = str(url).strip()
    if url.startswith("s3://"):
        return url
    try:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
        host = parsed.netloc or ""
        path = parsed.path or ""
        # Console URL: .../s3/buckets/<bucket>?prefix=...
        if "console.aws.amazon.com" in host and "/s3/buckets/" in path:
            parts = path.split("/s3/buckets/")[-1]
            bucket = parts.split("/")[0]
            qs = parse_qs(parsed.query)
            prefix = qs.get("prefix", [""])[0]
            return f"s3://{bucket}/{prefix}" if bucket else None
        # Virtual-hosted style: bucket.s3.region.amazonaws.com
        if ".s3." in host and host.endswith("amazonaws.com"):
            bucket = host.split(".s3.")[0]
            prefix = path.lstrip("/")
            return f"s3://{bucket}/{prefix}"
        # Path-style: s3.region.amazonaws.com/bucket/prefix
        if host.startswith("s3.") and host.endswith("amazonaws.com"):
            segs = path.lstrip("/").split("/", 1)
            bucket = segs[0] if segs else ""
            prefix = segs[1] if len(segs) > 1 else ""
            return f"s3://{bucket}/{prefix}" if bucket else None
    except Exception:
        pass
    return None


# ------------------------------ Data Models ------------------------------


@dataclass
class RemoteFile:
    s3_path: str  # full s3 path without scheme for s3fs (e.g., bucket/prefix/file)
    rel_key: str  # path relative to the provided prefix
    size: int
    etag: str | None


@dataclass
class PlanEntry:
    remote: RemoteFile
    local_path: Path
    action: str  # "download" | "skip"


# ------------------------------ Helpers ------------------------------

_ETAG_SUFFIX = ".etag"


def _parse_bucket_and_prefix(s3_url: str) -> tuple[str, str]:
    assert s3_url.startswith("s3://"), "normalize URL before parsing"
    without = s3_url[len("s3://") :]
    parts = without.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    # Ensure no leading slash in prefix and ensure trailing slash style-neutral
    return bucket, prefix.strip("/")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_local_etag(local_file: Path) -> str | None:
    try:
        etag_file = local_file.with_suffix(local_file.suffix + _ETAG_SUFFIX)
        if not etag_file.exists():
            # Support alternate naming when file may not have a suffix
            etag_alt = Path(str(local_file) + _ETAG_SUFFIX)
            if etag_alt.exists():
                return etag_alt.read_text().strip() or None
            return None
        return etag_file.read_text().strip() or None
    except Exception:
        return None


def _write_local_etag(local_file: Path, etag: str | None) -> None:
    try:
        etag_path = Path(str(local_file) + _ETAG_SUFFIX)
        if etag:
            etag_path.write_text(etag)
        elif etag_path.exists():
            etag_path.unlink(missing_ok=True)
    except Exception:
        pass


def _etag_from_info(info: dict[str, Any]) -> str | None:
    # s3fs may return 'ETag' or 'etag' and sometimes quoted
    etag = info.get("ETag") or info.get("etag") or info.get("ETag")
    if isinstance(etag, str):
        return etag.strip('"')
    return None


def _size_from_info(info: dict[str, Any]) -> int:
    size = info.get("Size") or info.get("size") or info.get("ContentLength")
    try:
        return int(size) if size is not None else 0
    except Exception:
        return 0


# ------------------------------ Local Mirror Helpers ------------------------------


def _should_include_for_what(rel_key: str, what: str) -> bool:
    rel_lc = rel_key.lower()
    if what == "logs":
        if "/logs/" in rel_lc:
            return True
        return bool("tfevents" in rel_lc or rel_lc.endswith(".log"))
    if what == "checkpoints":
        return "/checkpoints/" in rel_lc
    # both
    return True


def _prune_local_extras(dest: Path, remote_rel_keys: Iterable[str], what: str) -> tuple[int, int]:
    """Delete local files under dest that are not present in remote_rel_keys for the selected 'what'.

    Returns (deleted_count, failed_count).
    """
    remote_set = set(remote_rel_keys)
    deleted = failed = 0
    for p in dest.rglob("*"):
        if not p.is_file():
            continue
        # Compute relative key
        try:
            rel_key = str(p.relative_to(dest).as_posix())
        except Exception:
            continue
        # Skip our sidecar etag files here; they will be deleted alongside their main files below
        if rel_key.endswith(_ETAG_SUFFIX):
            continue
        if not _should_include_for_what(rel_key, what):
            continue
        if rel_key not in remote_set:
            # Delete file and its etag sidecar if present
            try:
                p.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                failed += 1
            try:
                etag_sidecar = Path(str(p) + _ETAG_SUFFIX)
                if etag_sidecar.exists():
                    etag_sidecar.unlink(missing_ok=True)
            except Exception:
                pass
    # Optionally, clean up empty directories (best-effort)
    try:
        for d in sorted((x for x in dest.rglob("*") if x.is_dir()), key=lambda x: len(str(x)), reverse=True):
            try:
                # Remove empty dirs only
                next(d.rglob("*"))
            except StopIteration:
                with contextlib.suppress(Exception):
                    d.rmdir()
    except Exception:
        pass
    return deleted, failed


# ------------------------------ S3 Listing/Planning ------------------------------


def _list_remote_files(s3_url: str, what: str) -> tuple[str, str, list[RemoteFile]]:
    """Return (bucket, prefix, files) for the given normalized s3 url.

    Filters by `what` (logs|checkpoints|both) using substring matching on rel path.
    """
    import s3fs

    bucket, prefix = _parse_bucket_and_prefix(s3_url)
    fs = s3fs.S3FileSystem(anon=False)
    base = f"{bucket}/{prefix}" if prefix else bucket

    # find returns list[str] or dict[str, info] with detail=True; normalize to iterable of (name, info)
    entries: list[tuple[str, dict[str, Any]]] = []
    try:
        found = fs.find(base, detail=True)
        if isinstance(found, dict):
            entries = [(name, info) for name, info in found.items()]
        elif isinstance(found, list):
            for name in found:
                try:
                    info = fs.info(name)
                except Exception:
                    continue
                entries.append((name, info))
        else:
            # Fallback: list non-detailed and info each
            for name in fs.find(base):
                try:
                    info = fs.info(name)
                except Exception:
                    continue
                entries.append((name, info))
    except FileNotFoundError:
        entries = []

    files: list[RemoteFile] = []
    for name, info in entries:
        # s3fs returns directories too; skip non-files
        typ = info.get("type") or info.get("Type")
        if typ and str(typ) != "file":
            continue
        size = _size_from_info(info)
        if size is None:
            size = 0
        rel = name[len(f"{bucket}/") :]
        # Ensure we restrict to the prefix
        if prefix and not rel.startswith(prefix.rstrip("/") + "/") and rel != prefix:
            continue
        # Compute relative to the provided prefix root
        rel_key = rel[len(prefix) :].lstrip("/") if prefix else rel

        # Filter by what
        rel_lc = rel_key.lower()
        if what == "logs" and "/logs/" not in rel_lc:
            # also allow common alt naming
            if "tfevents" not in rel_lc and rel_lc.endswith(".log") is False:
                continue
        if what == "checkpoints" and "/checkpoints/" not in rel_lc:
            continue

        files.append(
            RemoteFile(
                s3_path=name,
                rel_key=rel_key,
                size=int(size),
                etag=_etag_from_info(info),
            ),
        )

    return bucket, prefix, files


def _plan_sync(files: list[RemoteFile], dest: Path) -> list[PlanEntry]:
    plan: list[PlanEntry] = []
    for rf in files:
        local_path = dest / rf.rel_key
        action = "download"
        if local_path.exists():
            same = False
            # Prefer ETag match (using sidecar file); fallback to size
            local_etag = _read_local_etag(local_path)
            if rf.etag and local_etag and rf.etag == local_etag:
                same = True
            else:
                try:
                    if local_path.stat().st_size == rf.size and rf.size > 0:
                        same = True
                except FileNotFoundError:
                    same = False
            if same:
                action = "skip"
        plan.append(PlanEntry(remote=rf, local_path=local_path, action=action))
    return plan


# ------------------------------ Download Engine ------------------------------


def _download_with_retries(fs: Any, remote: str, local_path: Path, etag: str | None, max_attempts: int = 3) -> tuple[bool, str | None]:
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            _ensure_dir(local_path.parent)
            fs.get(remote, str(local_path))
            if etag:
                _write_local_etag(local_path, etag)
            return True, None
        except Exception as e:
            err = f"{e}"
            if attempt == max_attempts:
                return False, err
            time.sleep(delay)
            delay = min(delay * 2.0, 15.0)
    return False, "unknown error"


def _execute_plan(plan: list[PlanEntry], s3_url: str, max_workers: int = 6, dry_run: bool = False) -> tuple[int, int, int]:
    import s3fs

    fs = s3fs.S3FileSystem(anon=False)
    downloaded = skipped = failed = 0

    if dry_run:
        for p in plan:
            if p.action == "download":
                pass
            else:
                pass
        return 0, 0, 0

    def task(entry: PlanEntry) -> tuple[str, bool, str | None]:
        if entry.action == "skip":
            return (entry.remote.s3_path, True, None)
        ok, err = _download_with_retries(fs, entry.remote.s3_path, entry.local_path, entry.remote.etag)
        return (entry.remote.s3_path, ok, err)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, p) for p in plan]
        for fut in concurrent.futures.as_completed(futures):
            remote, ok, _err = fut.result()
            if ok:
                # Distinguish skipped vs downloaded by looking back into plan
                pe = next((x for x in plan if x.remote.s3_path == remote), None)
                if pe and pe.action == "skip":
                    skipped += 1
                else:
                    downloaded += 1
            else:
                failed += 1

    return downloaded, skipped, failed


# ------------------------------ Tailing ------------------------------


class _TailThread(threading.Thread):
    def __init__(self, dest: Path, stop_event: threading.Event, poll_sec: float = 3.0) -> None:
        super().__init__(daemon=True)
        self.dest = dest
        self.stop_event = stop_event
        self.poll_sec = poll_sec
        self._last_offsets: dict[Path, int] = {}
        self._tb_last_seen: dict[tuple[str, str], int] = {}  # (run, tag) -> last_step

    def _find_latest_file(self, patterns: tuple[str, ...]) -> Path | None:
        newest: Path | None = None
        newest_mtime = -1.0
        for p in self.dest.rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            if any(tok in name for tok in patterns):
                try:
                    m = p.stat().st_mtime
                except FileNotFoundError:
                    continue
                if m > newest_mtime:
                    newest_mtime = m
                    newest = p
        return newest

    def _tail_text_file(self, path: Path) -> None:
        try:
            last = self._last_offsets.get(path, 0)
            with open(path, encoding="utf-8", errors="ignore") as fp:
                fp.seek(last)
                chunk = fp.read()
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                self._last_offsets[path] = fp.tell()
        except Exception:
            return

    def _tail_tfevents(self, path: Path) -> None:
        # Best-effort: show the latest scalar updates via tensorboard's EventAccumulator
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
        except Exception:
            # tensorboard not installed or unavailable; print a hint once
            # Avoid spamming
            self.stop_event.wait(self.poll_sec)
            return

        try:
            ea = EventAccumulator(str(path))
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                if not events:
                    continue
                last_event = events[-1]
                key = (str(path), tag)
                last_seen = self._tb_last_seen.get(key, -1)
                if last_event.step > last_seen:
                    self._tb_last_seen[key] = last_event.step
        except Exception:
            # Ignore parsing errors, file may be mid-write
            return

    def run(self) -> None:
        printed_hint = False
        while not self.stop_event.is_set():
            # Prefer human-readable logs if present
            latest_log = self._find_latest_file((".log",))
            if latest_log is not None:
                self._tail_text_file(latest_log)
            else:
                # Fall back to TensorBoard event files
                latest_tb = self._find_latest_file(("tfevents",))
                if latest_tb is not None:
                    if not printed_hint:
                        printed_hint = True
                    self._tail_tfevents(latest_tb)
                elif not printed_hint:
                    printed_hint = True
            self.stop_event.wait(self.poll_sec)


# ------------------------------ CLI ------------------------------

_DEF_INTERVAL = 15
_DEF_MAX_WORKERS = 6


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continuously mirror training logs/checkpoints from S3 to a local directory, with optional tail",
    )
    p.add_argument("--s3-url", required=True, help="S3 or console/https URL. Accepts s3://, virtual-hosted, path-style, and console URLs.")
    p.add_argument("--dest", default="./training_progress", help="Local destination directory (default: ./training_progress)")
    p.add_argument("--what", choices=["logs", "checkpoints", "both"], default="both", help="What to sync (default: both)")
    p.add_argument("--interval", type=int, default=_DEF_INTERVAL, help="Sync interval seconds (default: 10–30)")
    p.add_argument("--tail", action="store_true", help="Tail latest logs (best-effort). Uses text logs if present, else tfevents scalars.")
    p.add_argument("--max-workers", type=int, default=_DEF_MAX_WORKERS, help="Max concurrent downloads (default: 6)")
    p.add_argument("--dry-run", action="store_true", help="List planned actions without downloading")
    p.add_argument("--once", action="store_true", help="Run a single sync pass and exit")
    return p.parse_args(argv)


def _do_sync_once(s3_url: str, dest: Path, what: str, max_workers: int, dry_run: bool) -> tuple[int, int, int]:
    _bucket, _prefix, files = _list_remote_files(s3_url, what)
    if not files:
        return 0, 0, 0
    plan = _plan_sync(files, dest)
    sum(1 for p in plan if p.action == "download")
    sum(1 for p in plan if p.action == "skip")
    downloaded, skipped, failed = _execute_plan(plan, s3_url, max_workers=max_workers, dry_run=dry_run)
    # Mirror-delete local extras for the selected scope
    try:
        remote_rel = [rf.rel_key for rf in files]
        pruned, prune_failed = _prune_local_extras(dest, remote_rel, what)
        if pruned or prune_failed:
            pass
    except Exception:
        pass
    return downloaded, skipped, failed


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    norm = _normalize_s3_url(args.s3_url)
    if not norm:
        return 2

    dest = Path(args.dest).expanduser().resolve()
    _ensure_dir(dest)

    stopping = threading.Event()

    def _handle_sigint(signum: Any, frame: Any) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    tail_thread: _TailThread | None = None
    if args.tail:
        tail_thread = _TailThread(dest=dest, stop_event=stopping)
        tail_thread.start()

    try:
        if args.once:
            _do_sync_once(norm, dest, args.what, args.max_workers, args.dry_run)
            return 0
        interval = max(1, int(args.interval))
        while not stopping.is_set():
            with contextlib.suppress(Exception):
                _do_sync_once(norm, dest, args.what, args.max_workers, args.dry_run)
            # Wait for next interval or stop
            for _ in range(interval):
                if stopping.is_set():
                    break
                time.sleep(1)
        return 0
    finally:
        if tail_thread is not None:
            stopping.set()
            tail_thread.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
