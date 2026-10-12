"""Loopback map page where the user draws the city area to simulate.

Mirrors `visualization.assets.BundleAssetServer`: a stdlib ThreadingHTTPServer
on 127.0.0.1, Host/Origin checks, opaque per-selection tokens and an idle
shutdown. The page (web/area_selector.html) POSTs the drawn rectangle; the
server validates it and wakes any `wait` on that selection. No solver imports.
"""

from __future__ import annotations

import json
import math
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PAGE = Path(__file__).with_name("web") / "area_selector.html"
EARTH_RADIUS_KM = 6371.0088
MAX_AREA_KM2 = 4.0


def bbox_area_km2(west: float, south: float, east: float, north: float) -> float:
    """Spherical area of a lon/lat rectangle (same formula as the page)."""
    dlon = math.radians(east - west)
    return (
        EARTH_RADIUS_KM**2
        * dlon
        * (math.sin(math.radians(north)) - math.sin(math.radians(south)))
    )


def validate_bbox(bbox: Any, max_area_km2: float = MAX_AREA_KM2) -> dict[str, float]:
    """Check an EPSG:4326 `{west, south, east, north}` mapping (or 4-sequence).

    Returns the bbox as floats plus `area_km2`; raises ValueError with a message
    meant for the user otherwise.
    """
    keys = ("west", "south", "east", "north")
    if isinstance(bbox, dict):
        values: list[Any] = [bbox.get(key) for key in keys]
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        values = list(bbox)
    else:
        raise ValueError("Give the area as {west, south, east, north} in degrees")
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in values
    ):
        raise ValueError("west, south, east and north must be finite numbers")
    west, south, east, north = (float(v) for v in values)
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("Longitudes must lie in [-180, 180]")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError("Latitudes must lie in [-90, 90]")
    if not west < east:
        raise ValueError("west must be smaller than east")
    if not south < north:
        raise ValueError("south must be smaller than north")
    area = bbox_area_km2(west, south, east, north)
    if area > max_area_km2:
        raise ValueError(
            f"Area is {area:.2f} km², the maximum is {max_area_km2:g} km²; "
            "draw a smaller rectangle"
        )
    return {
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "area_km2": area,
    }


class AreaSelector:
    """One loopback server; each `open()` adds a selection with its own token.

    The server closes after `idle_timeout` seconds without page requests or
    `wait` calls; create a new one then (pending selections are lost).
    """

    def __init__(self, idle_timeout: float = 1800):
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        self.selections: dict[str, dict[str, Any]] = {}
        self.last_access = time.monotonic()
        self.closed = False
        self._changed = threading.Condition()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(15)

            def _selection(self, leaf: str) -> dict[str, Any] | None:
                """The selection of /select/<token>/<leaf>, after Host/Origin checks."""
                if self.headers.get("Host") != owner.authority:
                    self.send_error(403, "Invalid Host")
                    return None
                origin = self.headers.get("Origin")
                if origin is not None and origin != f"http://{owner.authority}":
                    self.send_error(403, "Cross-origin access is not allowed")
                    return None
                parts = urlsplit(self.path).path.split("/")
                selection = None
                if len(parts) == 4 and parts[1] == "select" and parts[3] == leaf:
                    with owner._changed:
                        selection = owner.selections.get(parts[2])
                if selection is None:
                    self.send_error(404)
                    return None
                owner.last_access = time.monotonic()
                return selection

            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                # OSM tiles and Nominatim need a Referer; send the origin only,
                # never the token path.
                self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                selection = self._selection("")
                if selection is None:
                    return
                page = PAGE.read_text().replace(
                    "__MAX_AREA_KM2__", f"{selection['max_area_km2']:g}"
                )
                self._send(200, "text/html; charset=utf-8", page.encode())

            def do_POST(self) -> None:
                selection = self._selection("area")
                if selection is None:
                    return
                if self.headers.get_content_type() != "application/json":
                    self.send_error(415, "Send the area as JSON")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("Request body must be 1 B to 4 KiB")
                    payload = json.loads(self.rfile.read(length))
                    bbox = validate_bbox(payload, selection["max_area_km2"])
                except (ValueError, UnicodeDecodeError) as exc:
                    message = str(exc) if isinstance(exc, ValueError) else "Bad JSON"
                    body = json.dumps({"error": message}).encode()
                    self._send(400, "application/json", body)
                    return
                with owner._changed:
                    selection.update(status="selected", bbox=bbox)
                    owner._changed.notify_all()
                body = json.dumps({"status": "selected", "bbox": bbox}).encode()
                self._send(200, "application/json", body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.authority = f"127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def reap() -> None:
            while not self.closed:
                time.sleep(min(idle_timeout / 2, 5))
                if time.monotonic() - self.last_access >= idle_timeout:
                    self.close()

        threading.Thread(target=reap, daemon=True).start()

    def open(self, max_area_km2: float = MAX_AREA_KM2) -> tuple[str, str]:
        """Add a selection; returns its id (the URL token) and the page URL."""
        if self.closed:
            raise RuntimeError("Area selector has expired; create a new one")
        if not (math.isfinite(max_area_km2) and max_area_km2 > 0):
            raise ValueError("max_area_km2 must be a positive number")
        token = secrets.token_urlsafe(24)
        with self._changed:
            self.selections[token] = {
                "status": "pending",
                "max_area_km2": float(max_area_km2),
            }
        self.last_access = time.monotonic()
        return token, f"http://{self.authority}/select/{token}/"

    def wait(self, selection_id: str, timeout: float) -> dict[str, Any]:
        """The selection once submitted, else its pending state after `timeout` s."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._changed:
            if selection_id not in self.selections:
                raise ValueError(
                    "Unknown or expired selection_id; call open_area_selector again"
                )
            selection = self.selections[selection_id]
            while selection["status"] == "pending" and not self.closed:
                self.last_access = time.monotonic()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._changed.wait(min(remaining, 1.0))
            return dict(selection)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.server.shutdown()
            self.server.server_close()
            with self._changed:
                self._changed.notify_all()

    def __enter__(self) -> AreaSelector:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
