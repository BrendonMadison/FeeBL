#!/usr/bin/env python3
"""Check browser JavaScript syntax and run the deterministic Node.js fixtures."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="node", help="Node.js executable")
    args = parser.parse_args()
    node = shutil.which(args.node)
    if node is None:
        parser.error("Node.js is required; supply --node if it is not on PATH")
    tools = Path(__file__).resolve().parent
    scripts = sorted(tools.glob("collider_*.js"))
    html = (tools / "feebl_gui.html").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="feebl-browser-") as directory:
        inline = Path(directory) / "viewer.js"
        inline.write_text(
            "\n".join(re.findall(r"<script\b[^>]*>(.*?)</script>", html, re.S)), encoding="utf-8"
        )
        for path in [*scripts, inline]:
            subprocess.run([node, "--check", str(path)], check=True, timeout=30)
    tests = sorted(tools.glob("test_collider_*.js"))
    for path in tests:
        subprocess.run([node, str(path)], check=True, timeout=120)
    print(f"Passed syntax checks for {len(scripts) + 1} scripts and {len(tests)} test suites.")


if __name__ == "__main__":
    main()
