"""No file the package needs may be hidden by .gitignore.

Version 1.0.0 and 1.1.0 on GitHub lacked src/prithvi_unet/data/ because the rule "data/" matched
that folder too. This test fails if any rule hides a file under the folders below again (data/ included:
its GeoTIFFs and ZIPs are re-included by "!" rules).
Skipped outside a git checkout (for example a copy without .git).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRACKED_FOLDERS = ("src", "scripts", "tests", "configs", "docs", "notebooks", "data")
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(),
                    reason="needs git and a git checkout")
def test_no_project_file_is_gitignored():
    files = [p.relative_to(ROOT).as_posix()
             for folder in TRACKED_FOLDERS for p in (ROOT / folder).rglob("*")
             if p.is_file() and not SKIP_PARTS.intersection(p.parts) and p.suffix != ".pyc"
             and not any(part.endswith(".egg-info") for part in p.parts)    # written by pip install -e
             and not p.name.endswith(".aux.xml")]                          # written by QGIS/GDAL next to rasters
    assert files, "no project files found"
    # --no-index: judge by the .gitignore rules alone, also for files not added yet
    result = subprocess.run(["git", "check-ignore", "--no-index", "--verbose", *files],
                            cwd=ROOT, capture_output=True, text=True)
    # --verbose also lists files matched by a "!" rule (re-included, so not hidden): drop those lines
    hidden = [line for line in result.stdout.splitlines() if line and ":!" not in line.split("\t")[0]]
    assert not hidden, "Hidden by .gitignore (rule, then file):\n" + "\n".join(hidden)
