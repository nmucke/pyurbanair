"""Read-only loopback serving for registered, completed visualization bundles."""

from __future__ import annotations

import json
import mimetypes
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .data import contained


class BundleAssetServer:
    """Opaque URLs, strict file allowlists, and single-range media delivery.

    The owner should keep this object in the supervisor, outside render workers.
    Idle servers close after ``idle_timeout`` seconds; create a new server when
    retrieving a visualization after that point. No browser is opened here.
    """

    def __init__(self, idle_timeout: float = 1800):
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        self.bundles: dict[str, tuple[Path, set[str]]] = {}
        self.last_access = time.monotonic()
        self.closed = False
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(15)

            def do_HEAD(self) -> None:
                self._serve(False)

            def do_GET(self) -> None:
                self._serve(True)

            def _serve(self, body: bool) -> None:
                if self.headers.get("Host") != owner.authority:
                    self.send_error(403, "Invalid Host")
                    return
                origin = self.headers.get("Origin")
                if origin is not None and origin != f"http://{owner.authority}":
                    self.send_error(403, "Cross-origin access is not allowed")
                    return
                parts = unquote(urlsplit(self.path).path).split("/", 3)
                if len(parts) != 4 or parts[1] != "view":
                    self.send_error(404)
                    return
                with owner._lock:
                    bundle = owner.bundles.get(parts[2])
                if bundle is None:
                    self.send_error(404)
                    return
                root, allowed = bundle
                relative = parts[3] or "index.html"
                if relative not in allowed:
                    self.send_error(404)
                    return
                try:
                    path = contained(root, relative)
                    if not path.is_file():
                        raise ValueError("Missing asset")
                    size = path.stat().st_size
                    if size > 2 * 1024**3:
                        raise ValueError("Asset exceeds serving size limit")
                except (OSError, ValueError):
                    self.send_error(404)
                    return
                start, end = 0, size - 1
                range_header = self.headers.get("Range")
                if range_header:
                    try:
                        if not range_header.startswith("bytes=") or "," in range_header:
                            raise ValueError
                        first, last = range_header[6:].split("-", 1)
                        if first:
                            start = int(first)
                            end = min(int(last), size - 1) if last else size - 1
                        else:
                            length = int(last)
                            if length <= 0:
                                raise ValueError
                            start = max(0, size - length)
                        if start < 0 or start > end or start >= size:
                            raise ValueError
                    except ValueError:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                owner.last_access = time.monotonic()
                self.send_response(206 if range_header else 200)
                self.send_header(
                    "Content-Type",
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )
                self.send_header("Content-Length", str(max(0, end - start + 1)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
                )
                if range_header:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if body:
                    try:
                        with path.open("rb") as stream:
                            stream.seek(start)
                            remaining = end - start + 1
                            while remaining > 0:
                                chunk = stream.read(min(64 * 1024, remaining))
                                if not chunk:
                                    break
                                self.wfile.write(chunk)
                                remaining -= len(chunk)
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.authority = f"127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        def reap() -> None:
            while not self.closed:
                time.sleep(min(idle_timeout / 2, 5))
                if time.monotonic() - self.last_access >= idle_timeout:
                    self.close()

        threading.Thread(target=reap, daemon=True).start()

    def register(self, bundle_root: str | Path) -> str:
        if self.closed:
            raise RuntimeError("Asset server has expired; create a new server")
        root = Path(bundle_root).resolve()
        manifest = json.loads((root / "viewer_manifest.json").read_text())
        if manifest.get("version") != 1 or manifest.get("status") != "complete":
            raise ValueError("Only completed version-1 bundles may be served")
        allowed = {
            "index.html",
            "viewer.css",
            "viewer.js",
            "probe_charts.js",
            "viewer_manifest.json",
            "provenance.json",
            "render_config.resolved.yaml",
            "probes.json",
            "probes.csv",
        }
        for view in manifest["views"]:
            allowed.add(view["poster"])
            if view.get("media"):
                allowed.add(view["media"])
        if (root / "previews" / "probes.png").exists():
            allowed.add("previews/probes.png")
        for relative in allowed:
            if not contained(root, relative).is_file():
                raise ValueError(f"Missing bundle asset {relative}")
        token = secrets.token_urlsafe(24)
        with self._lock:
            self.bundles[token] = (root, allowed)
        self.last_access = time.monotonic()
        return f"http://{self.authority}/view/{token}/"

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.server.shutdown()
            self.server.server_close()

    def __enter__(self) -> BundleAssetServer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
