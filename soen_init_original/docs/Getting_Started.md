---
layout: default
title: Getting Started
---

# Getting Started

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Introduction.md" style="margin-right: 2em;">&#8592; Previous: Introduction</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Building_Models.md" style="margin-left: 2em;">Next: Building Models &#8594;</a>
</div>

Install SOEN-Toolkit in 3 simple steps using **uv** - a modern Python package manager that handles everything automatically.

---

## Choose Your Platform

|  |  |
| --- | --- |
| 🍎 <a class="platform-link mac" href="Install_macOS.md">macOS</a><br><small>Apple Silicon & Intel</small> | 🪟 <a class="platform-link win" href="Install_Windows.md">Windows</a><br><small>PowerShell</small> |
| 🐧 <a class="platform-link linux" href="Install_Linux.md">Linux</a><br><small>Ubuntu/Debian</small> | 🪟🐧 <a class="platform-link wsl2" href="Install_WSL2.md">WSL2</a><br><small>Linux on Windows</small> |

---

Click a link above for detailed platform-specific instructions.

---

## Troubleshooting (reset the environment)

If imports or tools act up, reset the venv and re-sync.

macOS/Linux:

```bash
deactivate
rm -rf .venv
uv sync # try just this first!
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
deactivate
Remove-Item -Recurse -Force .venv
uv sync
.venv\Scripts\activate
```

---

## Need Help?

- **Questions** → [GitHub Issues](https://github.com/greatsky-ai/soen-toolkit/issues)

---

<div align="center" style="margin-top: 2em; margin-bottom: 2em;">
  <a href="Introduction.md" style="margin-right: 2em;">&#8592; Previous: Introduction</a>
  &nbsp;|&nbsp;
  <a href="index.md" class="nav-home">Home</a>
  &nbsp;|&nbsp;
  <a href="Building_Models.md" style="margin-left: 2em;">Next: Building Models &#8594;</a>
</div>
