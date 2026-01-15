#!/usr/bin/env python3
"""
Sync documentation from docs/ folder to GitHub Wiki
Automatically converts links and formats for wiki compatibility

Usage:
    python scripts/sync_docs_to_wiki.py

Requirements:
    - Git must be installed
    - You must have created at least one page in the GitHub Wiki manually first
"""

from datetime import datetime
import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)

# Configuration
REPO_URL = "https://github.com/greatsky-ai/soen-toolkit.wiki.git"
WIKI_DIR = Path("wiki_temp")
DOCS_DIR = Path("docs")

# File mapping: source -> wiki page name
# Wiki uses filenames as page titles, so we convert underscores to hyphens
FILE_MAPPING = {
    "index.md": "Home.md",
    "Introduction.md": "01-Introduction.md",
    "Getting_Started.md": "02-Getting-Started.md",
    "Install_macOS.md": "02a-macOS-Install.md",
    "Install_Windows.md": "02b-Windows-Install.md",
    "Install_Linux.md": "02c-Linux-Install.md",
    "Install_WSL2.md": "02d-WSL2-Install.md",
    "Install_WSL_Setup.md": "02e-WSL-Setup.md",
    "Building_Models.md": "04-Building-Models.md",
    "PyTorch_API.md": "04a-PyTorch-API.md",
    "Training_Models.md": "05-Training-Models.md",
    "MLFLOW.md": "05a-MLFLOW.md",
    "DATASETS.md": "05b-DATASETS.md",
    "GUI_Tools.md": "06-GUI-Tools.md",
    "Advanced_Features.md": "07-Advanced-Features.md",
    "Unit_Converter.md": "07a-Unit-Converter.md",
}

def run_command(cmd, cwd=None):
    """Run shell command and return output"""
    result = subprocess.run(
        cmd, check=False, shell=True, cwd=cwd, capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout

def clone_or_pull_wiki():
    """Clone wiki repo or pull if it exists"""
    if WIKI_DIR.exists():
        logger.info("Wiki directory exists, pulling latest...")
        run_command("git pull", cwd=WIKI_DIR)
    else:
        logger.info("Cloning wiki repository...")
        run_command(f"git clone {REPO_URL} {WIKI_DIR}")

def convert_links_for_wiki(content):
    """Convert markdown links to wiki-compatible format"""

    # Remove YAML frontmatter (not supported in wiki)
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

    # Convert ALL filename references with underscores to hyphens in text
    # This prevents GitHub from auto-creating wrong wiki links
    # Example: "01_Introduction.md" -> "01-Introduction.md"
    # Example: "02_Getting_Started.md" -> "02-Getting-Started.md"
    def convert_filename_underscores(match):
        filename = match.group(1)
        # Replace all underscores with hyphens in the filename
        return filename.replace('_', '-')

    # Match any .md filename and convert underscores to hyphens
    content = re.sub(r'(\d\d[a-z]?[-_][A-Za-z_]+\.md)', convert_filename_underscores, content)

    # Convert internal doc links to wiki links
    # Example: [text](01_Introduction.md) -> [text](01-Introduction)
    def replace_link(match):
        text = match.group(1)
        link = match.group(2)

        # Handle special cases
        if link in {"index.md", "../index.md"}:
            return f"[{text}](Home)"

        # Remove leading ../ for relative paths
        link = link.replace("../", "")

        # Remove .md extension and replace underscores with hyphens
        if link.endswith(".md"):
            link = link[:-3]
        link = link.replace("_", "-")

        # Handle DeeperDives subdirectory
        link = link.replace("DeeperDives/", "")

        return f"[{text}]({link})"

    # Match markdown links: [text](link) and preserve anchors
    def md_repl(m):
        text = m.group(1)
        link = m.group(2) or ""
        anchor = m.group(3) or ""
        if link in ("index.md", "../index.md"):
            new_link = "Home"
        else:
            link = link.replace("../", "")
            if link.endswith(".md"):
                link = link[:-3]
            link = link.replace("_", "-")
            link = link.replace("DeeperDives/", "")
            new_link = link
        return f"[{text}]({new_link}{anchor})"

    content = re.sub(r'\[([^\]]+)\]\(([^)#]*\.md)(#[^)]+)?\)', md_repl, content)

    # Also handle href links in HTML (for navigation divs) and preserve anchors
    def href_repl(m):
        before = m.group(1)
        link = m.group(2) or ""
        anchor = m.group(3) or ""
        after = m.group(4)
        if link in ("index.md", "../index.md"):
            new_link = "Home"
        else:
            link = link.replace("../", "")
            if link.endswith(".md"):
                link = link[:-3]
            link = link.replace("_", "-")
            link = link.replace("DeeperDives/", "")
            new_link = link
        return f"{before}{new_link}{anchor}{after}"

    content = re.sub(r'(href=["\'])([^"\']*\.md)(#[^"\']+)?(["\'])', href_repl, content)

    return content

def clean_old_wiki_files():
    """Remove old wiki files with underscores in names"""
    logger.info("Cleaning old wiki files...")

    # List of old filenames with underscores that should be removed
    old_files = [
        "01_Introduction.md",
        "02_Getting_Started.md",
        "03_Core_Concepts.md",
        "04_Building_Models.md",
        "04a_PyTorch_API.md",
        "05_Training_Models.md",
        "05a_MLFLOW.md",
        "05b_DATASETS.md",
        "06_GUI_Tools.md",
        "07_Advanced_Features.md",
        "07a_Unit_Convertor.md",
        "ParaRNN_Solver.md", # It's now in DeeperDives/ so the wiki file name should have changed if it was there before
    ]

    removed_count = 0
    for old_file in old_files:
        old_path = WIKI_DIR / old_file
        if old_path.exists():
            old_path.unlink()
            removed_count += 1
            logger.info(f"  Removed {old_file}")

    if removed_count > 0:
        logger.info(f"  Cleaned {removed_count} old files")
    else:
        logger.info("  No old files to clean")

def copy_and_convert_files():
    """Copy markdown files and convert them for wiki"""
    logger.info("Copying and converting documentation files...")

    for source_name, wiki_name in FILE_MAPPING.items():
        source_path = DOCS_DIR / source_name
        if source_path.exists():
            # Read content
            with open(source_path, encoding='utf-8') as f:
                content = f.read()

            # Convert links
            content = convert_links_for_wiki(content)

            # Write to wiki
            wiki_path = WIKI_DIR / wiki_name
            with open(wiki_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"  {source_name} -> {wiki_name}")
        else:
            logger.warning(f"  {source_name} not found, skipping")

def copy_deeper_dives():
    """Copy DeeperDives content"""
    deeper_dives_dir = DOCS_DIR / "DeeperDives"
    if deeper_dives_dir.exists():
        logger.info("Copying DeeperDives content...")
        for md_file in deeper_dives_dir.glob("*.md"):
            # Read and convert
            with open(md_file, encoding='utf-8') as f:
                content = f.read()
            content = convert_links_for_wiki(content)

            # Write with converted filename
            wiki_name = md_file.stem.replace("_", "-") + ".md"
            wiki_path = WIKI_DIR / wiki_name
            with open(wiki_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"  DeeperDives/{md_file.name} -> {wiki_name}")

def copy_figures():
    """Copy figures folder"""
    figures_dir = DOCS_DIR / "Figures"
    if figures_dir.exists():
        logger.info("Copying figures...")
        wiki_figures = WIKI_DIR / "Figures"
        if wiki_figures.exists():
            shutil.rmtree(wiki_figures)
        shutil.copytree(figures_dir, wiki_figures)
        logger.info("  Figures copied")

def commit_and_push():
    """Commit changes and push to wiki"""
    logger.info("Committing changes...")

    run_command("git add .", cwd=WIKI_DIR)

    # Check if there are changes
    status = run_command("git status --porcelain", cwd=WIKI_DIR)
    if not status.strip():
        logger.info("No changes to commit")
        return False

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_command(
        f'git commit -m "Auto-sync from docs/ folder - {timestamp}"',
        cwd=WIKI_DIR
    )

    logger.info("Pushing to wiki...")
    run_command("git push origin master", cwd=WIKI_DIR)
    logger.info("Wiki sync complete!")
    return True

def main():
    """Main sync process"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger.info("Starting Wiki Sync...\n")

    logger.info("Note: Make sure you've created at least one page in your GitHub Wiki first!")
    logger.info("   Visit: https://github.com/greatsky-ai/soen-toolkit/wiki\n")

    try:
        # Clone or update wiki repo
        clone_or_pull_wiki()

        # Clean old files with underscores
        clean_old_wiki_files()

        # Copy and convert files
        copy_and_convert_files()
        copy_deeper_dives()
        copy_figures()

        # Commit and push
        changed = commit_and_push()

        logger.info("\n" + "="*60)
        if changed:
            logger.info("Done! Your wiki has been updated at:")
        else:
            logger.info("Wiki is already up to date!")
        logger.info("   https://github.com/greatsky-ai/soen-toolkit/wiki")
        logger.info("="*60)
        logger.info("\nTip: Run this script whenever you update your docs")

    except Exception as e:
        logger.error(f"\nError during sync: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

