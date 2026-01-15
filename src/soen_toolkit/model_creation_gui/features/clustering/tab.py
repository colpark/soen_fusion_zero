# from __future__ import annotations

# import json
# import subprocess
# import sys
# from pathlib import Path
# import shlex
# import os
# import shutil
# from typing import Any, Dict, List, Optional

# from PyQt6.QtCore import Qt
# from PyQt6.QtGui import QTextCursor
# from PyQt6.QtWidgets import (
#     QWidget,
#     QVBoxLayout,
#     QHBoxLayout,
#     QLabel,
#     QLineEdit,
#     QPushButton,
#     QFileDialog,
#     QTableWidget,
#     QTableWidgetItem,
#     QHeaderView,
#     QTextEdit,
#     QMessageBox,
#     QTreeWidget,
#     QTreeWidgetItem,
# )
# from PyQt6.QtWidgets import QAbstractItemView
# from PyQt6.QtCore import QProcess


# class ClusterTab(QWidget):
#     def __init__(self, default_config: Optional[str] = None, parent: Optional[QWidget] = None) -> None:
#         super().__init__(parent)
#         self._default_config = default_config or "src/soen_toolkit/ray_tool/configs/high_level/default.yaml"

#         root = QVBoxLayout(self)

#         # Config picker
#         cfg_row = QHBoxLayout()
#         cfg_row.addWidget(QLabel("High-level config:"))
#         self.cfg_edit = QLineEdit(self._default_config)
#         cfg_row.addWidget(self.cfg_edit, 1)
#         btn_browse = QPushButton("Browse…")
#         btn_browse.clicked.connect(self._browse_config)
#         cfg_row.addWidget(btn_browse)
#         root.addLayout(cfg_row)

#         # Controls row: Up / Down / Status / Nodes
#         ctrls = QHBoxLayout()
#         self.btn_up = QPushButton("Up")
#         self.btn_up.clicked.connect(self._cmd_up)
#         self.btn_down = QPushButton("Down")
#         self.btn_down.clicked.connect(self._cmd_down)
#         self.btn_status = QPushButton("Status")
#         self.btn_status.clicked.connect(self._cmd_status)
#         self.btn_nodes = QPushButton("Refresh Nodes")
#         self.btn_nodes.clicked.connect(self._refresh_nodes)
#         ctrls.addWidget(self.btn_up)
#         ctrls.addWidget(self.btn_down)
#         ctrls.addWidget(self.btn_status)
#         ctrls.addWidget(self.btn_nodes)
#         self.btn_ssh = QPushButton("SSH into Node…")
#         self.btn_ssh.clicked.connect(self._ssh_into_node)
#         # Target label updates when node selection changes
#         self.target_label = QLabel("Target: (head)")
#         ctrls.addStretch(1)
#         ctrls.addWidget(self.btn_ssh)
#         ctrls.addWidget(self.target_label)
#         root.addLayout(ctrls)

#         # Nodes table
#         self.nodes_table = QTableWidget(0, 6)
#         self.nodes_table.setHorizontalHeaderLabels(["NodeID", "IP", "Alive", "IsHead", "CPU/GPU", "Instance Type"])
#         self.nodes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
#         self.nodes_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
#         self.nodes_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
#         self.nodes_table.itemSelectionChanged.connect(self._on_node_selection_changed)
#         root.addWidget(self.nodes_table, 2)

#         # Exec / LS / Tail controls
#         ops = QHBoxLayout()
#         self.cmd_edit = QLineEdit()
#         self.cmd_edit.setPlaceholderText("Command to run on selected node (or head if none selected)")
#         btn_exec = QPushButton("Exec on Node")
#         btn_exec.clicked.connect(self._exec_on_node)
#         ops.addWidget(self.cmd_edit, 1)
#         ops.addWidget(btn_exec)
#         root.addLayout(ops)

#         # File browser
#         fb_ctrls = QHBoxLayout()
#         self.fb_root_edit = QLineEdit("/home/ubuntu")
#         btn_fb_load = QPushButton("Open")
#         btn_fb_load.clicked.connect(self._fb_load_root)
#         btn_tail = QPushButton("Tail Selected")
#         btn_tail.clicked.connect(self._fb_tail_selected)
#         fb_ctrls.addWidget(QLabel("Browse:"))
#         fb_ctrls.addWidget(self.fb_root_edit, 1)
#         fb_ctrls.addWidget(btn_fb_load)
#         fb_ctrls.addWidget(btn_tail)
#         root.addLayout(fb_ctrls)

#         self.fb_tree = QTreeWidget()
#         self.fb_tree.setHeaderLabels(["Name", "Type", "Size", "Modified", "Path"])
#         self.fb_tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
#         self.fb_tree.itemExpanded.connect(self._fb_expand_item)
#         root.addWidget(self.fb_tree, 3)

#         # Output pane
#         self.output = QTextEdit()
#         self.output.setReadOnly(True)
#         root.addWidget(self.output, 3)

#         # Initial load is deferred to avoid blocking UI if no cluster is up

#     # ----- helpers -----
#     def _browse_config(self) -> None:
#         fname, _ = QFileDialog.getOpenFileName(self, "Select high-level Ray config", str(Path.cwd()), "YAML (*.yaml *.yml)")
#         if fname:
#             self.cfg_edit.setText(fname)

#     def _get_config(self) -> str:
#         return self.cfg_edit.text().strip()

#     def _run_cli(self, args: List[str]) -> subprocess.CompletedProcess:
#         cmd = [sys.executable, "-m", "soen_toolkit.ray_tool", "--config", self._get_config(), *args]
#         try:
#             return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
#         except subprocess.TimeoutExpired:
#             proc = subprocess.CompletedProcess(cmd, returncode=124, stdout="", stderr="Command timed out")
#             return proc

#     def _run_cli_json(self, args: List[str]) -> Dict[str, Any]:
#         proc = self._run_cli(args)
#         if proc.returncode != 0:
#             self._append_output(proc.stderr or proc.stdout)
#             raise RuntimeError("Command failed")
#         try:
#             # Tolerate extra wrapper output by extracting the JSON object segment
#             stdout = proc.stdout.strip()
#             start = stdout.find("{")
#             end = stdout.rfind("}")
#             if start != -1 and end != -1 and end > start:
#                 return json.loads(stdout[start : end + 1])
#             # Fallback to direct parse
#             return json.loads(stdout)
#         except Exception:
#             self._append_output(proc.stdout)
#             raise

#     def _append_output(self, text: str) -> None:
#         if not text:
#             return
#         self.output.append(text.rstrip())
#         self.output.moveCursor(QTextCursor.MoveOperation.End)

#     def _selected_node_id(self) -> Optional[str]:
#         idx = self.nodes_table.currentRow()
#         if idx >= 0:
#             item = self.nodes_table.item(idx, 0)
#             return item.text() if item else None
#         # Fallback to head row if no explicit selection
#         for r in range(self.nodes_table.rowCount()):
#             is_head_item = self.nodes_table.item(r, 3)
#             if is_head_item and is_head_item.text().lower() == "true":
#                 item = self.nodes_table.item(r, 0)
#                 return item.text() if item else None
#         return None

#     def _on_node_selection_changed(self) -> None:
#         node_id = self._selected_node_id()
#         if node_id:
#             # Show IP and short NodeID
#             r = self.nodes_table.currentRow()
#             ip = self.nodes_table.item(r, 1).text() if r >= 0 else ""
#             self.target_label.setText(f"Target: {ip} ({node_id[:8]})")
#         else:
#             self.target_label.setText("Target: (head)")

#     # ----- actions -----
#     def _cmd_up(self) -> None:
#         # Use QProcess for long-running up to avoid blocking UI and allow streaming logs
#         self._run_long_cli(["up"])

#     def _cmd_down(self) -> None:
#         reply = QMessageBox.question(self, "Confirm Down", "Tear down the cluster?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
#         if reply != QMessageBox.StandardButton.Yes:
#             return
#         self._run_long_cli(["down", "-y"])  # non-blocking

#     def _cmd_status(self) -> None:
#         proc = self._run_cli(["status"])
#         self._append_output(proc.stdout or proc.stderr)

#     def _ssh_into_node(self) -> None:
#         # Resolve target IP; default to head if none selected
#         idx = self.nodes_table.currentRow()
#         ip = None
#         if idx >= 0:
#             ip = self.nodes_table.item(idx, 1).text()
#         else:
#             # find head row
#             for r in range(self.nodes_table.rowCount()):
#                 if self.nodes_table.item(r, 3) and self.nodes_table.item(r, 3).text().lower() == "true":
#                     ip = self.nodes_table.item(r, 1).text()
#                     break
#         if not ip:
#             QMessageBox.warning(self, "SSH", "No node selected and head not found.")
#             return
#         # Spawn an external terminal and run attach-node with --node-ip=<ip>
#         cmd = [sys.executable, "-m", "soen_toolkit.ray_tool", "--config", self._get_config(), "attach-node", "--node-ip", ip]
#         # Prefer the exact ray CLI path from this process (works even if new terminal has different PATH)
#         ray_path = shutil.which("ray")
#         soen_ray_cli = f"SOEN_RAY_CLI='{ray_path}' " if ray_path else ""
#         # Prepare optional conda activation snippets
#         conda_exe = os.environ.get("CONDA_EXE", "")
#         conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
#         bash_activate = ""
#         if conda_exe and conda_env:
#             bash_activate = f'eval "$({conda_exe} shell.bash hook)"; conda activate {conda_env}; '
#         # macOS: open Terminal.app
#         if sys.platform == "darwin":
#             # Use bash -lc to set env and run in project root; escape for AppleScript
#             repo_root = str(Path(__file__).resolve().parents[4])
#             pyenv = f"PYTHONPATH='{repo_root}/src' "
#             cmd_str = " ".join(shlex.quote(x) for x in cmd)
#             full = f"cd {shlex.quote(repo_root)} && {bash_activate}{pyenv}{soen_ray_cli}{cmd_str}"
#             full_escaped = full.replace("\\", "\\\\").replace("\"", "\\\"")
#             script = f"tell application \"Terminal\" to do script \"{full_escaped}\""
#             try:
#                 subprocess.Popen(["osascript", "-e", script])
#                 return
#             except Exception:
#                 pass
#         # Windows: use start and cmd.exe
#         if sys.platform.startswith("win"):
#             try:
#                 repo_root = str(Path(__file__).resolve().parents[4])
#                 win_cmd = f"cd /d {repo_root} && set PYTHONPATH={repo_root}\\src && "
#                 if ray_path:
#                     win_cmd += f"set SOEN_RAY_CLI={ray_path} && "
#                 # Try explicit conda.bat if available
#                 win_conda = os.path.join(os.path.dirname(conda_exe), 'conda.bat') if conda_exe else ''
#                 if os.path.isfile(win_conda) and conda_env:
#                     win_cmd += f'"{win_conda}" activate {conda_env} && '
#                 elif conda_env:
#                     win_cmd += f'conda activate {conda_env} && '
#                 win_cmd += subprocess.list2cmdline(cmd)
#                 subprocess.Popen(["cmd", "/c", "start", "", "cmd", "/k", win_cmd])
#                 return
#             except Exception:
#                 pass
#         # Linux and others: try gnome-terminal/x-terminal-emulator, fallback to run detached
#         for term in ("gnome-terminal", "konsole", "x-terminal-emulator", "xfce4-terminal", "xterm"):
#             try:
#                 repo_root = str(Path(__file__).resolve().parents[4])
#                 pyenv = f"PYTHONPATH='{repo_root}/src' "
#                 full = f"bash -lc 'cd {repo_root} && {bash_activate}{pyenv}{soen_ray_cli}{' '.join(cmd)}'"
#                 subprocess.Popen([term, "-e", full])
#                 return
#             except Exception:
#                 continue
#         # Fallback: run in background without attaching to GUI
#         try:
#             env = os.environ.copy()
#             if ray_path:
#                 env["SOEN_RAY_CLI"] = ray_path
#             env["PYTHONPATH"] = str(Path(__file__).resolve().parents[4] / "src") + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
#             subprocess.Popen(cmd, env=env)
#         except Exception as e:
#             QMessageBox.critical(self, "SSH", f"Failed to open terminal: {e}")

#     def _refresh_nodes(self) -> None:
#         try:
#             data = self._run_cli_json(["nodes"])  # emitted by cluster_agent
#         except Exception:
#             # On error, clear nodes table and file browser to avoid showing stale entries
#             self.nodes_table.clearContents()
#             self.nodes_table.setRowCount(0)
#             self.fb_tree.clear()
#             self.target_label.setText("Target: (head)")
#             return
#         # Dedupe nodes; prefer NodeID when present, fall back to IP so pending/uninitialized entries show up
#         raw_nodes = data.get("nodes", []) or []
#         nodes: List[Dict[str, Any]] = []
#         seen_ids: set[str] = set()
#         seen_ips: set[str] = set()
#         for n in raw_nodes:
#             nid = (n.get("NodeID") or "").strip()
#             ip = (n.get("NodeManagerAddress") or "").strip()
#             alive = n.get("Alive")
#             # Allow entries without NodeID (pending) to be shown using IP key
#             if nid:
#                 if nid in seen_ids:
#                     continue
#                 seen_ids.add(nid)
#             else:
#                 if not ip or ip in seen_ips:
#                     continue
#                 seen_ips.add(ip)
#             # Keep both alive and not-yet-alive entries so pending nodes are visible
#             nodes.append(n)
#         # Preserve selection by NodeID
#         prev_id = self._selected_node_id()
#         # Clear stale contents to avoid ghost rows
#         self.nodes_table.clearContents()
#         self.nodes_table.setRowCount(len(nodes))
#         if len(nodes) == 0:
#             # Nothing to show; also clear file browser and selection
#             self.nodes_table.clearSelection()
#             self.fb_tree.clear()
#             self.target_label.setText("Target: (head)")
#             return
#         for r, n in enumerate(nodes):
#             cpu_gpu = []
#             res: Dict[str, Any] = n.get("Resources") or {}
#             if "CPU" in res:
#                 cpu_gpu.append(f"CPU:{int(res['CPU']) if isinstance(res['CPU'], (int, float)) else res['CPU']}")
#             if "GPU" in res:
#                 cpu_gpu.append(f"GPU:{int(res['GPU']) if isinstance(res['GPU'], (int, float)) else res['GPU']}")
#             values = [
#                 n.get("NodeID", ""),
#                 n.get("NodeManagerAddress", ""),
#                 str(n.get("Alive", "")),
#                 str(n.get("IsHead", "")),
#                 ", ".join(cpu_gpu),
#                 n.get("InstanceType", ""),
#             ]
#             for c, v in enumerate(values):
#                 self.nodes_table.setItem(r, c, QTableWidgetItem(str(v)))
#         # Restore selection or select head/first
#         target_row = -1
#         if prev_id:
#             for r, n in enumerate(nodes):
#                 if n.get("NodeID") == prev_id:
#                     target_row = r
#                     break
#         if target_row == -1:
#             for r, n in enumerate(nodes):
#                 if n.get("IsHead"):
#                     target_row = r
#                     break
#         if target_row == -1 and nodes:
#             target_row = 0
#         if not nodes:
#             # Clear the file browser and selection when no nodes exist
#             self.fb_tree.clear()
#             self.target_label.setText("Target: (head)")
#         elif target_row != -1:
#             self.nodes_table.selectRow(target_row)
#         else:
#             self.target_label.setText("Target: (head)")
#         self._on_node_selection_changed()

#     def _exec_on_node(self) -> None:
#         cmd = self.cmd_edit.text().strip()
#         if not cmd:
#             return
#         node_id = self._selected_node_id()
#         args = ["exec-node", "--command", cmd]
#         if node_id:
#             args = ["exec-node", "--node-id", node_id, "--command", cmd]
#         proc = self._run_cli(args)
#         self._append_output(proc.stdout or proc.stderr)

#     # ----- file browser -----
#     def _fb_load_root(self) -> None:
#         self.fb_tree.clear()
#         root_path = self.fb_root_edit.text().strip()
#         node_id = self._selected_node_id()
#         args = ["ls", "--path", root_path, "--depth", "1"]
#         if node_id:
#             args = ["ls", "--node-id", node_id, "--path", root_path, "--depth", "1"]
#         else:
#             # If a non-head row is selected but lacks NodeID (pending), target by IP via SSH fallback
#             r = self.nodes_table.currentRow()
#             if r >= 0:
#                 ip = self.nodes_table.item(r, 1).text() if self.nodes_table.item(r, 1) else ""
#                 is_head = (self.nodes_table.item(r, 3).text().lower() == "true") if self.nodes_table.item(r, 3) else False
#                 if ip and not is_head:
#                     args = ["ls", "--node-ip", ip, "--path", root_path, "--depth", "1"]
#         try:
#             out = self._run_cli_json(args)
#         except Exception:
#             # Fallback: if NodeID attempt failed, try by IP via SSH
#             try:
#                 r = self.nodes_table.currentRow()
#                 ip = self.nodes_table.item(r, 1).text() if r >= 0 and self.nodes_table.item(r, 1) else ""
#                 is_head = (self.nodes_table.item(r, 3).text().lower() == "true") if r >= 0 and self.nodes_table.item(r, 3) else False
#                 if ip and not is_head:
#                     out = self._run_cli_json(["ls", "--node-ip", ip, "--path", root_path, "--depth", "1"])
#                 else:
#                     return
#             except Exception:
#                 return
#         root_item = QTreeWidgetItem([root_path, "dir", "", "", root_path])
#         self.fb_tree.addTopLevelItem(root_item)
#         self._fb_populate_children(root_item, out.get("entries") or [])
#         root_item.setExpanded(True)

#     def _fb_populate_children(self, parent: QTreeWidgetItem, entries: List[Dict[str, Any]]) -> None:
#         # Clear existing
#         for i in range(parent.childCount() - 1, -1, -1):
#             parent.removeChild(parent.child(i))
#         for e in entries:
#             name = e.get("name") or Path(e.get("path", "")).name
#             typ = e.get("type", "?")
#             size = e.get("size")
#             mtime = e.get("mtime")
#             path = e.get("path", "")
#             size_str = "" if size in (None, "") else str(size)
#             mtime_str = "" if mtime in (None, "") else str(mtime)
#             item = QTreeWidgetItem([name, typ, size_str, mtime_str, path])
#             parent.addChild(item)
#             # For directories, add a dummy child so it shows expandable; load on demand
#             if typ == "dir":
#                 item.addChild(QTreeWidgetItem(["loading...", "", "", "", ""]))

#     def _fb_expand_item(self, item: QTreeWidgetItem) -> None:
#         # Load children lazily
#         if item.childCount() == 1 and item.child(0).text(0) == "loading...":
#             node_id = self._selected_node_id()
#             path = item.text(4)
#             args = ["ls", "--path", path, "--depth", "1"]
#             if node_id:
#                 args = ["ls", "--node-id", node_id, "--path", path, "--depth", "1"]
#             else:
#                 r = self.nodes_table.currentRow()
#                 if r >= 0:
#                     ip = self.nodes_table.item(r, 1).text() if self.nodes_table.item(r, 1) else ""
#                     is_head = (self.nodes_table.item(r, 3).text().lower() == "true") if self.nodes_table.item(r, 3) else False
#                     if ip and not is_head:
#                         args = ["ls", "--node-ip", ip, "--path", path, "--depth", "1"]
#             try:
#                 out = self._run_cli_json(args)
#             except Exception:
#                 # Fallback: try by IP if available
#                 try:
#                     r = self.nodes_table.currentRow()
#                     ip = self.nodes_table.item(r, 1).text() if r >= 0 and self.nodes_table.item(r, 1) else ""
#                     is_head = (self.nodes_table.item(r, 3).text().lower() == "true") if r >= 0 and self.nodes_table.item(r, 3) else False
#                     if ip and not is_head:
#                         out = self._run_cli_json(["ls", "--node-ip", ip, "--path", path, "--depth", "1"])
#                     else:
#                         return
#                 except Exception:
#                     return
#             # Remove placeholder
#             item.removeChild(item.child(0))
#             self._fb_populate_children(item, out.get("entries") or [])

#     def _fb_tail_selected(self) -> None:
#         item = self.fb_tree.currentItem()
#         if not item:
#             return
#         path = item.text(4)
#         node_id = self._selected_node_id()
#         args = ["tail", "--path", path, "--lines", "200"]
#         if node_id:
#             args = ["tail", "--node-id", node_id, "--path", path, "--lines", "200"]
#         else:
#             r = self.nodes_table.currentRow()
#             if r >= 0:
#                 ip = self.nodes_table.item(r, 1).text() if self.nodes_table.item(r, 1) else ""
#                 is_head = (self.nodes_table.item(r, 3).text().lower() == "true") if self.nodes_table.item(r, 3) else False
#                 if ip and not is_head:
#                     args = ["tail", "--node-ip", ip, "--path", path, "--lines", "200"]
#         try:
#             out = self._run_cli_json(args)
#         except Exception:
#             # Fallback: try by IP via SSH
#             try:
#                 r = self.nodes_table.currentRow()
#                 ip = self.nodes_table.item(r, 1).text() if r >= 0 and self.nodes_table.item(r, 1) else ""
#                 is_head = (self.nodes_table.item(r, 3).text().lower() == "true") if r >= 0 and self.nodes_table.item(r, 3) else False
#                 if ip and not is_head:
#                     out = self._run_cli_json(["tail", "--node-ip", ip, "--path", path, "--lines", "200"])
#                 else:
#                     return
#             except Exception:
#                 return
#         lines = out.get("lines") or []
#         self._append_output("".join(lines))

#     def _tail_file(self) -> None:
#         path = self.tail_path.text().strip()
#         node_id = self._selected_node_id()
#         args = ["tail", "--path", path, "--lines", "200"]
#         if node_id:
#             args = ["tail", "--node-id", node_id, "--path", path, "--lines", "200"]
#         else:
#             r = self.nodes_table.currentRow()
#             if r >= 0:
#                 ip = self.nodes_table.item(r, 1).text() if self.nodes_table.item(r, 1) else ""
#                 is_head = (self.nodes_table.item(r, 3).text().lower() == "true") if self.nodes_table.item(r, 3) else False
#                 if ip and not is_head:
#                     args = ["tail", "--node-ip", ip, "--path", path, "--lines", "200"]
#         try:
#             out = self._run_cli_json(args)
#         except Exception:
#             try:
#                 r = self.nodes_table.currentRow()
#                 ip = self.nodes_table.item(r, 1).text() if r >= 0 and self.nodes_table.item(r, 1) else ""
#                 is_head = (self.nodes_table.item(r, 3).text().lower() == "true") if r >= 0 and self.nodes_table.item(r, 3) else False
#                 if ip and not is_head:
#                     out = self._run_cli_json(["tail", "--node-ip", ip, "--path", path, "--lines", "200"])
#                 else:
#                     return
#             except Exception:
#                 return
#         lines = out.get("lines") or []
#         self._append_output("".join(lines))

#     # ----- long-running process helper -----
#     def _run_long_cli(self, args: List[str]) -> None:
#         cmd = [sys.executable, "-m", "soen_toolkit.ray_tool", "--config", self._get_config(), *args]
#         self._set_controls_enabled(False)
#         proc = QProcess(self)
#         # Capture both stdout and stderr
#         proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
#         proc.readyReadStandardOutput.connect(lambda: self._append_output(bytes(proc.readAllStandardOutput()).decode(errors="ignore")))
#         proc.finished.connect(lambda code, status: self._on_proc_finished(proc, code))
#         proc.start(cmd[0], cmd[1:])
#         # Keep a reference to prevent GC
#         self._current_proc = proc

#     def _on_proc_finished(self, proc: QProcess, code: int) -> None:
#         self._append_output(f"\n[process exited with code {code}]\n")
#         self._set_controls_enabled(True)
#         self._current_proc = None

#     def _set_controls_enabled(self, enabled: bool) -> None:
#         for b in (self.btn_up, self.btn_down, self.btn_status, self.btn_nodes):
#             b.setEnabled(enabled)


