"""Tidy an executed notebook before committing it (standard library only).

    python scripts/clean_notebook.py notebooks/examples/xuanthuy_example.ipynb            # clean in place
    python scripts/clean_notebook.py notebooks/examples/xuanthuy_example.ipynb --check    # report only

* Progress bars: lines rewritten with carriage returns (tqdm, pip) keep only their final state.
* Terminal colour codes are removed from text outputs and tracebacks.
* Widget state (ipywidgets) is dropped: GitHub cannot render it.
* Fails (exit 1, file unchanged) if an output contains an error, or text that looks like a
  GitHub or Hugging Face token, or if the notebook is larger than --max-mb.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
SECRETS = re.compile(r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gho_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,})")


def tidy_text(text: str) -> str:
    lines = []
    for line in ANSI.sub("", text).split("\n"):
        parts = [p for p in line.split("\r") if p.strip()]
        lines.append(parts[-1] if parts else "")
    return "\n".join(lines)


def as_text(value) -> str:
    return "".join(value) if isinstance(value, list) else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook")
    parser.add_argument("--check", action="store_true", help="report only, do not write")
    parser.add_argument("--max-mb", type=float, default=10.0)
    args = parser.parse_args()
    path = Path(args.notebook)
    nb = json.loads(path.read_text(encoding="utf-8"))

    problems, sizes = [], []
    nb.get("metadata", {}).pop("widgets", None)
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        kept = []
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                out["text"] = tidy_text(as_text(out["text"])).splitlines(keepends=True)
            elif kind == "error":
                out["traceback"] = [ANSI.sub("", line) for line in out.get("traceback", [])]
                problems.append(f"cell {index}: error output {out.get('ename')}: {out.get('evalue')}")
            elif kind in ("display_data", "execute_result"):
                out.get("data", {}).pop("application/vnd.jupyter.widget-view+json", None)
                if "text/plain" in out.get("data", {}):
                    out["data"]["text/plain"] = tidy_text(as_text(out["data"]["text/plain"])).splitlines(keepends=True)
                if not out.get("data"):
                    continue
            kept.append(out)
        cell["outputs"] = kept
        cell.get("metadata", {}).pop("colab", None)
        blob = json.dumps(cell, ensure_ascii=False)
        sizes.append((len(blob.encode("utf-8")), index))
        for match in SECRETS.findall(blob):
            problems.append(f"cell {index}: looks like a token: {match[:8]}...")

    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    megabytes = len(text.encode("utf-8")) / 1e6
    if megabytes > args.max_mb:
        problems.append(f"notebook is {megabytes:.1f} MB (> {args.max_mb} MB): lower the figure dpi or drop figures")
    print(f"{path}: {megabytes:.2f} MB, {len(nb['cells'])} cells; largest code cells:")
    for size, index in sorted(sizes, reverse=True)[:5]:
        print(f"  cell {index:3d}: {size / 1e3:8.1f} kB")
    if problems:
        print("NOT CLEAN:")
        for p in problems:
            print("  -", p)
        return 1
    if not args.check:
        path.write_text(text, encoding="utf-8")
        print("cleaned and written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
