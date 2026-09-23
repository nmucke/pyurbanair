"""Pure-python view of a render bundle (no bpy): manifest, file lookup, cameras.

Runs inside Blender's Python (numpy + stdlib only), so it must not import
``les_render``. The camera interpolation is a transliteration of
``les_render.cameras.sample_camera`` and must stay in sync with it: smoothstep
easing of the key-segment parameter, a uniform Catmull-Rom spline (end keys
duplicated) for ``location``/``target``, and a linear lerp of the lens
properties on the same eased parameter; hard cuts between shots.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Optional

import numpy as np


class Bundle:
    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root).resolve()
        self.manifest: dict[str, Any] = json.loads(
            (self.root / "manifest.json").read_text()
        )
        if int(self.manifest.get("version", 1)) != 1:
            raise ValueError(
                f"unsupported manifest version {self.manifest.get('version')}"
            )

    # -- basics ---------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return int(self.manifest["timeline"]["n_frames"])

    @property
    def fps(self) -> float:
        return float(self.manifest["timeline"]["fps"])

    @property
    def look(self) -> str:
        return self.manifest.get("render", {}).get("look", "dark")

    @property
    def layers(self) -> list[dict[str, Any]]:
        return list(self.manifest.get("layers") or [])

    @property
    def shots(self) -> list[dict[str, Any]]:
        return list(self.manifest.get("shots") or [])

    def path(self, rel: str) -> pathlib.Path:
        return self.root / rel

    # -- per-frame files ---------------------------------------------------------

    @staticmethod
    def file_index(layer: dict[str, Any], frame: int) -> int:
        """File shown at video ``frame``: ``floor(frame / frame_step)``, clamped."""
        step = max(int(layer.get("frame_step", 1)), 1)
        n = int(layer.get("n_files", 1))
        return int(min(max(frame // step, 0), max(n - 1, 0)))

    def layer_file(self, layer: dict[str, Any], frame: int) -> pathlib.Path:
        return self.root / layer["pattern"].format(frame=self.file_index(layer, frame))

    def hud_file(self, frame: int) -> Optional[pathlib.Path]:
        hud = self.manifest.get("hud")
        if not hud:
            return None
        step = max(int(hud.get("frame_step", 1)), 1)
        return self.root / hud["pattern"].format(frame=frame // step)

    # -- shots -------------------------------------------------------------------

    def shot_at(self, frame: int) -> Optional[dict[str, Any]]:
        shots = self.shots
        if not shots:
            return None
        frame = min(max(frame, shots[0]["start"]), shots[-1]["end"])
        for s in shots:
            if s["start"] <= frame <= s["end"]:
                return s
        return shots[-1]

    def visible_layers(self, frame: int) -> set[str]:
        shot = self.shot_at(frame)
        names = {layer["name"] for layer in self.layers}
        if shot is None or shot.get("layers") is None:
            return names
        return names & set(shot["layers"])


# -- cameras ---------------------------------------------------------------------


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _catmull_rom(p0, p1, p2, p3, u: float) -> np.ndarray:
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    u2, u3 = u * u, u * u * u
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


def sample_camera(shots: list[dict[str, Any]], frame: int) -> dict[str, Any]:
    """Camera state at video ``frame``: location, target, focal_length_mm, fstop, shot."""
    if not shots:
        raise ValueError("no shots to sample")
    frame = int(min(max(frame, shots[0]["start"]), shots[-1]["end"]))
    shot = next((s for s in shots if s["start"] <= frame <= s["end"]), shots[-1])
    keys = sorted(shot["keys"], key=lambda k: k["frame"])

    def pack(loc, tgt, focal, fstop):
        return {
            "location": np.asarray(loc, dtype=np.float64),
            "target": np.asarray(tgt, dtype=np.float64),
            "focal_length_mm": float(focal),
            "fstop": float(fstop),
            "shot": shot["name"],
        }

    def hold(k):
        return pack(
            k["location"], k["target"], k["focal_length_mm"], k.get("fstop", 8.0)
        )

    kf = [k["frame"] for k in keys]
    if len(keys) == 1 or frame <= kf[0]:
        return hold(keys[0])
    if frame >= kf[-1]:
        return hold(keys[-1])
    i = int(np.searchsorted(kf, frame, side="right") - 1)
    i = min(max(i, 0), len(keys) - 2)
    t0, t1 = kf[i], kf[i + 1]
    u = _smoothstep(0.0 if t1 == t0 else (frame - t0) / (t1 - t0))
    im1, ip2 = max(i - 1, 0), min(i + 2, len(keys) - 1)
    loc = _catmull_rom(*(keys[j]["location"] for j in (im1, i, i + 1, ip2)), u)
    tgt = _catmull_rom(*(keys[j]["target"] for j in (im1, i, i + 1, ip2)), u)
    a, b = keys[i], keys[i + 1]
    focal = (1 - u) * a["focal_length_mm"] + u * b["focal_length_mm"]
    fstop = (1 - u) * a.get("fstop", 8.0) + u * b.get("fstop", 8.0)
    return pack(loc, tgt, focal, fstop)


def default_shots(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """A single slow orbit-ish push when the bundle carries no shots."""
    lo = np.array(manifest["domain"]["lower"])
    hi = np.array(manifest["domain"]["upper"])
    c = 0.5 * (lo + hi)
    c[2] = 0.3 * float(
        manifest.get("geometry", {}).get("max_building_height", hi[2] * 0.4)
    )
    span = float(np.linalg.norm(hi[:2] - lo[:2]))
    n = int(manifest["timeline"]["n_frames"])
    loc0 = c + np.array([-0.55 * span, -0.75 * span, 0.45 * span])
    loc1 = c + np.array([-0.35 * span, -0.85 * span, 0.35 * span])
    return [
        {
            "name": "default",
            "start": 0,
            "end": n - 1,
            "keys": [
                {
                    "frame": 0,
                    "location": loc0.tolist(),
                    "target": c.tolist(),
                    "focal_length_mm": 35,
                    "fstop": 11,
                },
                {
                    "frame": n - 1,
                    "location": loc1.tolist(),
                    "target": c.tolist(),
                    "focal_length_mm": 35,
                    "fstop": 11,
                },
            ],
        }
    ]


# -- PLY -------------------------------------------------------------------------

_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_ply(
    path: str | pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Minimal PLY reader: ``(verts (N,3) f4, tris (M,3) i4, vertex props)``.

    Handles ascii and binary (little/big endian) files with a ``vertex``
    element of scalar properties and a ``face`` list element. Faces with more
    than three corners are fan-triangulated.
    """
    raw = pathlib.Path(path).read_bytes()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii", "replace").splitlines()
    fmt = "ascii"
    elements: list[dict[str, Any]] = []
    for line in header:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append({"name": tok[1], "count": int(tok[2]), "props": []})
        elif tok[0] == "property":
            if tok[1] == "list":
                elements[-1]["props"].append(
                    ("list", tok[4], _PLY_TYPES[tok[2]], _PLY_TYPES[tok[3]])
                )
            else:
                elements[-1]["props"].append(("scalar", tok[2], _PLY_TYPES[tok[1]]))
    endian = ">" if fmt == "binary_big_endian" else "<"
    body = raw[end:]
    verts = np.zeros((0, 3), np.float32)
    tris = np.zeros((0, 3), np.int32)
    props: dict[str, np.ndarray] = {}

    if fmt == "ascii":
        lines = body.decode("ascii").split("\n")
        pos = 0
        for el in elements:
            rows = lines[pos : pos + el["count"]]
            pos += el["count"]
            if el["name"] == "vertex":
                arr = np.array([r.split() for r in rows], dtype=np.float64)
                for c, p in enumerate(el["props"]):
                    props[p[1]] = arr[:, c].astype(p[2])
            elif el["name"] == "face":
                faces = [list(map(int, r.split()[1:])) for r in rows]
                tris = _triangulate(faces)
    else:
        off = 0
        for el in elements:
            scalars = [p for p in el["props"] if p[0] == "scalar"]
            lists = [p for p in el["props"] if p[0] == "list"]
            if not lists:
                dt = np.dtype([(p[1], endian + p[2]) for p in scalars])
                arr = np.frombuffer(body, dt, count=el["count"], offset=off)
                off += dt.itemsize * el["count"]
                if el["name"] == "vertex":
                    props = {n: np.array(arr[n]) for n in arr.dtype.names}
                continue
            # list element (faces): fast path for a constant corner count
            if len(el["props"]) == 1 and el["count"] > 0:
                _, _, ct, it = lists[0]
                n0 = int(np.frombuffer(body, endian + ct, count=1, offset=off)[0])
                dt = np.dtype([("n", endian + ct), ("i", endian + it, (n0,))])
                arr = np.frombuffer(body, dt, count=el["count"], offset=off)
                if np.all(arr["n"] == n0):
                    off += dt.itemsize * el["count"]
                    if el["name"] == "face":
                        tris = _triangulate_const(np.array(arr["i"], dtype=np.int32))
                    continue
            # general slow path
            faces = []
            for _ in range(el["count"]):
                row = []
                for p in el["props"]:
                    if p[0] == "scalar":
                        off += np.dtype(p[2]).itemsize
                    else:
                        n = int(
                            np.frombuffer(body, endian + p[2], count=1, offset=off)[0]
                        )
                        off += np.dtype(p[2]).itemsize
                        idx = np.frombuffer(body, endian + p[3], count=n, offset=off)
                        off += np.dtype(p[3]).itemsize * n
                        row = idx.tolist()
                faces.append(row)
            if el["name"] == "face":
                tris = _triangulate(faces)
    if props:
        verts = np.stack([props["x"], props["y"], props["z"]], axis=1).astype(
            np.float32
        )
    return verts, tris, props


def _triangulate_const(idx: np.ndarray) -> np.ndarray:
    if idx.shape[1] == 3:
        return idx
    fan = [idx[:, [0, k, k + 1]] for k in range(1, idx.shape[1] - 1)]
    return np.concatenate(fan).astype(np.int32)


def _triangulate(faces: list[list[int]]) -> np.ndarray:
    out = [[f[0], f[k], f[k + 1]] for f in faces for k in range(1, len(f) - 1)]
    return np.asarray(out, dtype=np.int32).reshape(-1, 3)


# -- particles ---------------------------------------------------------------------


def load_particles(
    path: str | pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``points (L, P, 3) f4, speed (L, P) f4, alpha (L, P) f4`` from a particle npz."""
    with np.load(path) as z:
        pts = np.asarray(z["points"], dtype=np.float32)
        spd = (
            np.asarray(z["speed"], dtype=np.float32)
            if "speed" in z
            else np.zeros(pts.shape[:2], np.float32)
        )
        alpha = (
            np.asarray(z["alpha"], dtype=np.float32)
            if "alpha" in z
            else np.ones(pts.shape[:2], np.float32)
        )
    bad = ~np.isfinite(pts).all(axis=2)
    if bad.any():
        alpha = np.where(bad, 0.0, alpha)
        pts = np.where(bad[..., None], 0.0, pts)
    return pts, np.nan_to_num(spd), np.clip(np.nan_to_num(alpha), 0.0, 1.0)


def visible_segments(alpha: np.ndarray) -> np.ndarray:
    """Edge list (E, 2) over flattened points: segment i->i+1 iff both alphas > 0."""
    L, P = alpha.shape
    ok = (alpha[:, :-1] > 0) & (alpha[:, 1:] > 0)
    li, pi = np.nonzero(ok)
    a = (li * P + pi).astype(np.int32)
    return np.stack([a, a + 1], axis=1)


def visible_runs(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximal chains of consecutive visible points (alpha > 0, length >= 2).

    Returns ``(sizes (R,), flat_point_index (sum(sizes),))`` into the
    flattened ``(n_lines * points_per_line)`` point array.
    """
    L, P = alpha.shape
    vis = alpha > 0
    pad = np.zeros((L, 1), bool)
    v = np.concatenate([pad, vis, pad], axis=1).astype(np.int8)
    d = np.diff(v, axis=1)
    sl, sp = np.nonzero(d == 1)  # run starts (line, point)
    el, ep = np.nonzero(d == -1)  # run ends (exclusive)
    sizes = ep - sp
    keep = sizes >= 2
    sl, sp, sizes = sl[keep], sp[keep], sizes[keep]
    if len(sizes) == 0:
        return sizes.astype(np.int64), np.zeros(0, np.int64)
    start = sl * P + sp
    offs = np.repeat(start - np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes)
    idx = np.arange(int(sizes.sum())) + offs
    return sizes.astype(np.int64), idx.astype(np.int64)


def stretch_fade(
    pts: np.ndarray, sizes: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Per-point opacity factor that fades out stretched segments.

    Far downstream, consecutive streak releases are dispersed by turbulence
    and the chain between them degenerates into long straight zig-zags;
    segments longer than ``lo`` fade linearly to invisible at ``hi``. A point
    takes the factor of its longer adjacent segment.
    """
    n = len(pts)
    f = np.ones(n, np.float32)
    if n < 2:
        return f
    has_next = np.ones(n, bool)
    has_next[np.cumsum(sizes) - 1] = False
    seg_len = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    g = np.clip((hi - seg_len) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    g = np.where(has_next[:-1], g, 1.0)
    f[:-1] = np.minimum(f[:-1], g)
    f[1:] = np.minimum(f[1:], g)
    return f


def chunk_runs(
    pts: np.ndarray, attrs: list[np.ndarray], sizes: np.ndarray, max_points: int
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Split every run into consecutive curves of at most ``max_points``
    points that share their joint point (so the chain stays connected).

    Works around Blender 4.2 EEVEE Next hair drawing, which samples every
    curve at a fixed 8 points (x 2**hair_subdiv) spread along the curve,
    whatever its real point count: a 256-point streakline is drawn as a
    7-segment zig-zag (minimal repro: a 64-point circle renders as a
    heptagon). Curves of <= 8 points are drawn exactly. Cycles is unaffected.
    """
    sizes = np.asarray(sizes, np.int64)
    if max_points < 2 or len(sizes) == 0 or sizes.max() <= max_points:
        return pts, attrs, sizes
    step = max_points - 1
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    n_chunks = np.maximum((sizes - 2) // step + 1, 1)  # ceil((n - 1) / step)
    run = np.repeat(np.arange(len(sizes)), n_chunks)
    k = np.arange(int(n_chunks.sum())) - np.repeat(
        np.cumsum(n_chunks) - n_chunks, n_chunks
    )
    c_start = starts[run] + k * step
    c_size = np.minimum(max_points, starts[run] + sizes[run] - c_start)
    idx = np.repeat(
        c_start - np.concatenate([[0], np.cumsum(c_size)[:-1]]), c_size
    ) + np.arange(int(c_size.sum()))
    return pts[idx], [a[idx] for a in attrs], c_size


def cut_runs(
    pts: np.ndarray, attrs: list[np.ndarray], sizes: np.ndarray, keep_seg: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Split runs at segments with ``keep_seg[j] == False`` (segment j -> j+1
    over the concatenated points; entries at run ends are ignored) and drop
    runs left with fewer than two points."""
    n = len(pts)
    if n == 0:
        return pts, attrs, sizes
    run_end = np.zeros(n, bool)
    run_end[np.cumsum(sizes) - 1] = True
    brk = run_end.copy()
    brk[:-1] |= ~keep_seg[: n - 1]
    run_id = np.concatenate([[0], np.cumsum(brk[:-1])])
    new_sizes = np.bincount(run_id)
    keep_pt = new_sizes[run_id] >= 2
    return pts[keep_pt], [a[keep_pt] for a in attrs], new_sizes[new_sizes >= 2]


def smooth_runs(
    pts: np.ndarray, attrs: list[np.ndarray], sizes: np.ndarray, k: int
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Subdivide every segment of each run into ``k`` pieces along a
    *centripetal* Catmull-Rom spline (Barry-Goldman form). Unlike the uniform
    Catmull-Rom that renderers apply to curve control points, the centripetal
    variant never overshoots or self-intersects on unevenly spaced samples,
    so jagged streaklines become smooth without loops. Point attributes are
    interpolated linearly. ``pts`` (N, 3) are the concatenated runs.
    """
    if k <= 1 or len(pts) < 2:
        return pts, attrs, sizes
    sizes = np.asarray(sizes, np.int64)
    n = len(pts)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    ends = starts + sizes - 1
    run_of = np.repeat(np.arange(len(sizes)), sizes)
    has_next = np.ones(n, bool)
    has_next[ends] = False
    seg = np.nonzero(has_next)[0]  # segment j -> j+1
    p1, p2 = pts[seg], pts[seg + 1]
    is_first = np.zeros(n, bool)
    is_first[starts] = True
    p0 = np.where(is_first[seg][:, None], 2 * p1 - p2, pts[np.maximum(seg - 1, 0)])
    nxt2_ok = has_next[np.minimum(seg + 1, n - 1)]
    p3 = np.where(nxt2_ok[:, None], pts[np.minimum(seg + 2, n - 1)], 2 * p2 - p1)

    def knot(a, b):
        return np.maximum(np.linalg.norm(b - a, axis=1), 1e-4) ** 0.5

    t0 = np.zeros(len(seg))
    t1 = t0 + knot(p0, p1)
    t2 = t1 + knot(p1, p2)
    t3 = t2 + knot(p2, p3)
    u = np.arange(1, k) / k  # interior samples
    T = t1[:, None] + (t2 - t1)[:, None] * u[None, :]  # (S, k-1)

    def lerp(a, b, ta, tb):
        w = ((T - ta[:, None]) / (tb - ta)[:, None])[..., None]
        return a[:, None, :] * (1 - w) + b[:, None, :] * w

    A1 = lerp(p0, p1, t0, t1)
    A2 = lerp(p1, p2, t1, t2)
    A3 = lerp(p2, p3, t2, t3)
    wB1 = ((T - t0[:, None]) / (t2 - t0)[:, None])[..., None]
    B1 = A1 * (1 - wB1) + A2 * wB1
    wB2 = ((T - t1[:, None]) / (t3 - t1)[:, None])[..., None]
    B2 = A2 * (1 - wB2) + A3 * wB2
    wC = u[None, :, None]
    C = B1 * (1 - wC) + B2 * wC  # (S, k-1, 3)

    new_sizes = (sizes - 1) * k + 1
    out_start = np.concatenate([[0], np.cumsum(new_sizes)[:-1]])
    local = np.arange(n) - starts[run_of]
    orig_pos = out_start[run_of] + local * k
    m = int(new_sizes.sum())
    out = np.empty((m, 3), pts.dtype)
    out[orig_pos] = pts
    inner = (orig_pos[seg][:, None] + np.arange(1, k)[None, :]).reshape(-1)
    out[inner] = C.reshape(-1, 3)
    new_attrs = []
    for a in attrs:
        o = np.empty(m, a.dtype)
        o[orig_pos] = a
        o[inner] = (a[seg][:, None] * (1 - u) + a[seg + 1][:, None] * u).reshape(-1)
        new_attrs.append(o)
    return out, new_attrs, new_sizes


def strand_camera_distance(
    pts: np.ndarray, has_next: np.ndarray, cam: np.ndarray
) -> np.ndarray:
    """Per point: min distance from the camera to the point or to either
    adjacent segment. Capping radii with this keeps a long segment that
    sweeps past the lens thin along its whole length (radius interpolates
    linearly between two far-away endpoints otherwise)."""
    c = np.asarray(cam, np.float32)
    d = np.linalg.norm(pts - c, axis=1)
    if len(pts) < 2:
        return d
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(
        np.einsum("ij,ij->i", c - a, ab)
        / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12),
        0.0,
        1.0,
    )
    seg = np.linalg.norm(a + t[:, None] * ab - c, axis=1)
    seg = np.where(has_next[:-1], seg, np.inf)
    d[:-1] = np.minimum(d[:-1], seg)
    d[1:] = np.minimum(d[1:], seg)
    return d


def collapse_hidden(points: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Constant-topology variant for Alembic/Groom: each hidden point is moved
    onto the nearest visible point of its line (ties go to the head side), so
    hidden spans have zero length; together with zero width on hidden points,
    a gap of >= 2 hidden points between two visible runs renders as nothing.
    Fully hidden lines collapse onto their head point."""
    L, P = alpha.shape
    ar = np.arange(P)[None, :]
    vis = alpha > 0
    ffill = np.maximum.accumulate(np.where(vis, ar, -1), axis=1)  # previous visible
    bfill = np.minimum.accumulate(np.where(vis, ar, P)[:, ::-1], axis=1)[
        :, ::-1
    ]  # next visible
    d_prev = np.where(ffill >= 0, ar - ffill, P + 1)
    d_next = np.where(bfill < P, bfill - ar, P + 1)
    src = np.where(d_prev <= d_next, ffill, bfill)
    src = np.where((src < 0) | (src >= P), 0, src)
    return np.take_along_axis(points, src[..., None].repeat(3, axis=2), axis=1)


def camera_quaternion(
    location: np.ndarray, target: np.ndarray
) -> tuple[float, float, float, float]:
    """(w, x, y, z) rotating a Blender camera (looks down -Z, +Y up) to aim at
    ``target`` with world +Z up (no roll)."""
    f = np.asarray(target, float) - np.asarray(location, float)
    f /= max(np.linalg.norm(f), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(f, up))) > 0.9999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, f)
    m = np.stack([right, cam_up, -f], axis=1)  # columns = camera X, Y, Z in world
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    return (float(w), float(x), float(y), float(z))
