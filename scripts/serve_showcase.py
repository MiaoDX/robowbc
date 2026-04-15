#!/usr/bin/env python3
"""Serve a generated RoboWBC showcase bundle over HTTP for local debugging."""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import webbrowser
from pathlib import Path


class ShowcaseRequestHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".rrd": "application/octet-stream",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory containing the generated showcase bundle and index.html",
    )
    parser.add_argument("--bind", default="127.0.0.1", help="Address to bind the local server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the local server")
    parser.add_argument("--open", action="store_true", help="Open the served page in the default browser")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"showcase directory not found: {root}")
    if not (root / "index.html").is_file():
        raise SystemExit(f"expected {root / 'index.html'} to exist")

    handler = functools.partial(ShowcaseRequestHandler, directory=str(root))
    with ReusableTCPServer((args.bind, args.port), handler) as httpd:
        url = f"http://{args.bind}:{args.port}/"
        print(f"Serving RoboWBC showcase from {root}")
        print(f"Open {url}")
        print("Press Ctrl-C to stop.")
        if args.open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping showcase server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
