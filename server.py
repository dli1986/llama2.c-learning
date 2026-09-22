#!/usr/bin/env python3
"""Trivial static file server for dist/, reachable from other devices on the LAN.

Usage:
    python server.py [port]

Defaults to port 8000, binds 0.0.0.0 so a phone on the same Wi-Fi/LAN can
reach it at http://<this-PC's-LAN-IP>:<port>/ - find the IP with `ipconfig`.
"""
import http.server
import socketserver
import sys
from pathlib import Path

DIST_DIR = Path(__file__).resolve().parent / "dist"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST_DIR), **kwargs)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    if not DIST_DIR.exists():
        print(f"error: {DIST_DIR} does not exist yet - run build\\build.py first")
        sys.exit(1)
    with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"serving {DIST_DIR} on:")
        print(f"  http://localhost:{port}/")
        print(f"  http://<this-PC's-LAN-IP>:{port}/   (find IP via ipconfig)")
        print("Ctrl+C to stop")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
