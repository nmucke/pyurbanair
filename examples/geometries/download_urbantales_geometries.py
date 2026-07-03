#!/usr/bin/env python3
"""Download all UrbanTALES building geometries (topography height rasters).

UrbanTALES (Nazarian et al. 2025, BAMS) is an open LES dataset of 538 urban
configurations run in PALM. Each case ships a ``*_topo`` file: a 2-D building
height raster in plain-text ASCII (a value grid where 0 = ground / non-building
and positive values are the building height in metres at 1 m resolution). This
is a top-down, 2.5-D extruded height map -- exactly the geometry contract our
CFD backends consume (flat ground at z=0, no overhangs).

This script pulls ONLY the ``*_topo`` geometry files (~350 MB for all 538
cases), not the multi-TB flow fields. Output layout::

    examples/geometries/raw/
        idealized/     224 cases  (aligned/staggered arrays)
        realistic/     314 cases  (real OSM-derived neighbourhoods)
        <set>/metadata.json        per-case morphology (lambda_p, heights, wind, city)

The raw rasters land under ``raw/`` so that a downstream processing step can
write derived artifacts (e.g. STL meshes) to a sibling ``processed/`` tree.

The dataset is CC BY-NC 4.0 -- cite Nazarian et al. (2025) when using it.
Data platform: https://urbantales.vercel.app/   (files served from a Nextcloud
instance; the two /api/metadata_* endpoints are the machine-readable manifests).

Usage:
    python download_urbantales_geometries.py                 # download everything
    python download_urbantales_geometries.py --set idealized # one set only
    python download_urbantales_geometries.py --workers 16    # tune concurrency
    python download_urbantales_geometries.py --force         # re-download existing

Stdlib only -- no pip installs -- so it runs on any machine with Python 3.8+.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://urbantales.vercel.app"
MANIFESTS = {
    "idealized": "/api/metadata_ideal",
    "realistic": "/api/metadata_rea",
}
GEOMETRY_SUFFIX = "_topo"  # the 2-D building-height ASCII raster
USER_AGENT = "urbantales-geometry-fetch/1.0 (research; pyurbanair)"
RETRIES = 4
TIMEOUT = 120


def _get(url: str) -> bytes:
    """GET with retries and exponential backoff."""
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data: bytes = resp.read()
                return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed after {RETRIES} attempts: {url} ({last})")


def fetch_manifest(kind: str) -> list[dict]:
    cases: list[dict] = json.loads(_get(BASE + MANIFESTS[kind]).decode("utf-8"))
    return cases


def topo_entry(case: dict) -> dict | None:
    files: list[dict] = case.get("Files", [])
    for f in files:
        if f["File Name"].endswith(GEOMETRY_SUFFIX):
            return f
    return None


def download_one(case: dict, out_dir: Path, force: bool) -> tuple[str, str]:
    """Return (folder_name, status)."""
    entry = topo_entry(case)
    folder = case.get("Folder Name", "?")
    if entry is None:
        return folder, "no-topo"
    dest = out_dir / entry["File Name"]
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return folder, "skip"
    data = _get(entry["Direct Download Link"])
    if not data:
        return folder, "empty"
    dest.write_bytes(data)
    return folder, "ok"


def write_metadata(cases: list[dict], out_dir: Path) -> None:
    """Persist per-case morphology (everything except the bulky Files table)."""
    meta = []
    for c in cases:
        entry = topo_entry(c)
        row = {k: v for k, v in c.items() if k != "Files"}
        row["topo_file"] = entry["File Name"] if entry else None
        meta.append(row)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def run_set(kind: str, root: Path, workers: int, force: bool) -> None:
    out_dir = root / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{kind}] fetching manifest ...", flush=True)
    cases = fetch_manifest(kind)
    write_metadata(cases, out_dir)
    print(f"[{kind}] {len(cases)} cases -> {out_dir}", flush=True)

    counts = {"ok": 0, "skip": 0, "no-topo": 0, "empty": 0, "error": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_one, c, out_dir, force): c for c in cases}
        for fut in as_completed(futs):
            folder = futs[fut].get("Folder Name", "?")
            try:
                _, status = fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                status = "error"
                print(f"  ! {folder}: {exc}", file=sys.stderr, flush=True)
            counts[status] = counts.get(status, 0) + 1
            done += 1
            if status in ("ok", "error") and done % 20 == 0:
                print(f"  [{kind}] {done}/{len(cases)} ...", flush=True)
    print(f"[{kind}] done: {counts}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--set",
        choices=["idealized", "realistic", "all"],
        default="all",
        help="which layout family to download",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "raw",
        help="output root (default: <script dir>/raw)",
    )
    ap.add_argument(
        "--workers", type=int, default=8, help="concurrent downloads (default: 8)"
    )
    ap.add_argument(
        "--force", action="store_true", help="re-download files that already exist"
    )
    args = ap.parse_args()

    kinds = ["idealized", "realistic"] if args.set == "all" else [args.set]
    for kind in kinds:
        run_set(kind, args.out, args.workers, args.force)
    print("All requested geometries downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
