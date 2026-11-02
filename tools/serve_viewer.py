#!/usr/bin/env python3
"""Serve the FeeBL beam viewer and one finalized run on the loopback interface."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import webbrowser

TOOLS = Path(__file__).resolve().parent


def resolve_file(request_path: str, run: Path) -> Path | None:
    """Map a URL to a viewer asset or one regular file in the selected run.

    Directory traversal, symlinks, nested paths, and directory listings are
    rejected. This server is a local display utility, not a remote service.
    """
    path = unquote(urlsplit(request_path).path)
    if path.startswith("/data/"):
        directory, name = run, path[len("/data/") :]
    else:
        directory, name = TOOLS, path.removeprefix("/")
        if name != "feebl_gui.html" and not (name.startswith("collider_") and name.endswith(".js")):
            return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        return None
    candidate = directory / name
    if candidate.is_symlink() or not candidate.is_file():
        return None
    if candidate.resolve().parent != directory.resolve():
        return None
    return candidate


def handler_for(run: Path):
    """Create a read-only request handler bound to the selected dataset."""

    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_file(head_only=False)

        def do_HEAD(self):
            self.send_file(head_only=True)

        def send_file(self, head_only):
            if urlsplit(self.path).path == "/":
                self.send_response(302)
                self.send_header("Location", "/feebl_gui.html?run=data")
                self.end_headers()
                return
            path = resolve_file(self.path, run)
            if path is None:
                self.send_error(404)
                return
            self.send_response(200)
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if not head_only:
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        self.wfile.write(block)

    return ViewerHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Finalized dataset directory")
    parser.add_argument(
        "--port", type=int, default=8765, help="Local port; zero selects a free port"
    )
    parser.add_argument("--open", action="store_true", help="Open the URL in the default browser")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    run = args.run.resolve()
    if resolve_file("/data/gui_manifest.json", run) is None:
        parser.error("The run needs gui_manifest.json; run finalize_gui_dataset.py first")
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(run)) as server:
        url = f"http://127.0.0.1:{server.server_port}/feebl_gui.html?run=data"
        print(f"FeeBL viewer: {url}\nDataset: {run}\nPress Ctrl+C to stop.", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
