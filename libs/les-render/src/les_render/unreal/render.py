# mypy: ignore-errors
# (Unreal Editor script: untyped by design, runs against the dynamic `unreal` module.)
"""Render the LES Level Sequence with Movie Render Queue (MRQ).

Two modes, same file:

Outside Unreal (plain Python, any OS) -- launches a headless MRQ render and
prints the output directory:

    python render.py --bundle <bundle> --project <Project.uproject> [--editor <UnrealEditor-Cmd>] [--dry-run]

It reads ``<bundle>/unreal/ue_scene.json`` (written by ``build_scene.py``) and
runs, per Epic's command-line rendering docs:

    UnrealEditor-Cmd <Project.uproject> <map> -game -LevelSequence=<seq>
        -MoviePipelineConfig=<MRQ config asset> -windowed -ResX=W -ResY=H
        -log -StdOut -allowStdOutLogVerbosity -Unattended -NoLoadingScreen -NoSplash -notexturestreaming

Inside the Unreal Editor (Tools > Execute Python Script, or
``-ExecutePythonScript``) -- queues the saved config in the Movie Render
Queue subsystem and renders it with the in-editor (PIE) executor; the output
directory is printed to the Output Log.

Only ``unreal`` (in-editor), ``json``, ``math``, ``os``, ``pathlib``, ``sys``,
``shlex`` and ``subprocess`` are imported.
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import sys

try:
    import unreal  # type: ignore[import-not-found]
except ImportError:
    unreal = None  # type: ignore[assignment]

# Replaced with the absolute bundle path by ``prepare_unreal``.
DEFAULT_BUNDLE = None

LOG_PREFIX = "[LES render]"

# Keep references so the executor / callbacks are not garbage-collected
# while the render runs in the editor.
_KEEPALIVE = []


def parse_args(argv, env=None, default_bundle=None):
    env = os.environ if env is None else env
    opts = {
        "bundle": None,
        "project": env.get("UE_PROJECT"),
        "editor": env.get("UE_EDITOR_CMD"),
        "dry_run": False,
        "offscreen": False,
        "extra": [],
    }
    args = list(argv)
    i = 0
    while i < len(args):
        a = args[i]
        key = a.split("=", 1)[0]
        takes = key in ("--bundle", "--project", "--editor", "--extra")
        if takes:
            if "=" in a:
                val = a.split("=", 1)[1]
            elif i + 1 < len(args):
                val = args[i + 1]
                i += 1
            else:
                raise SystemExit(f"{key} needs a value")
            if key == "--extra":
                opts["extra"] += shlex.split(val)
            else:
                opts[key[2:]] = val
        elif a == "--dry-run":
            opts["dry_run"] = True
        elif a == "--offscreen":
            opts["offscreen"] = True
        i += 1
    opts["bundle"] = opts["bundle"] or env.get("LES_BUNDLE") or default_bundle
    return opts


def load_scene(bundle):
    path = pathlib.Path(bundle) / "unreal" / "ue_scene.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: run build_scene.py inside Unreal first")
    return json.loads(path.read_text())


def default_editor():
    """Best guess for the UnrealEditor-Cmd binary (override with --editor / UE_EDITOR_CMD)."""
    if sys.platform.startswith("win"):
        return r"C:\Program Files\Epic Games\UE_5.5\Engine\Binaries\Win64\UnrealEditor-Cmd.exe"
    return "UnrealEditor-Cmd"


def build_render_command(scene, project, editor=None, offscreen=False, extra=()):
    """The MRQ command line (list of args) for a scene summary dict."""
    if not project:
        raise SystemExit("pass --project <Project.uproject> (or set UE_PROJECT)")
    w, h = scene["resolution"]
    cmd = [
        editor or default_editor(),
        str(project),
        scene["map"],
        "-game",
        f"-LevelSequence={scene['sequence']}",
        f"-MoviePipelineConfig={scene['mrq_config']}",
        "-windowed",
        f"-ResX={int(w)}",
        f"-ResY={int(h)}",
        "-log",
        "-StdOut",
        "-allowStdOutLogVerbosity",
        "-Unattended",
        "-NoLoadingScreen",
        "-NoSplash",
        "-notexturestreaming",
    ]
    if offscreen:
        cmd.append("-RenderOffscreen")
    cmd += list(extra)
    return cmd


def expected_frames(scene):
    """Output file paths MRQ should write ({sequence_name}.{frame_number}, 4-digit)."""
    seq_name = scene["sequence"].rsplit(".", 1)[-1]
    ext = "exr" if "exr" in str(scene.get("output_format", "png")).lower() else "png"
    out = pathlib.Path(scene["output_dir"])
    return [out / f"{seq_name}.{f:04d}.{ext}" for f in range(int(scene["n_frames"]))]


def run_outside(opts):
    if not opts["bundle"]:
        raise SystemExit("pass --bundle <dir> (or set LES_BUNDLE)")
    scene = load_scene(opts["bundle"])
    cmd = build_render_command(
        scene, opts["project"], opts["editor"], opts["offscreen"], opts["extra"]
    )
    print(f"{LOG_PREFIX} " + " ".join(shlex.quote(c) for c in cmd))
    if opts["dry_run"]:
        print(scene["output_dir"])
        return 0
    pathlib.Path(scene["output_dir"]).mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(cmd)
    frames = expected_frames(scene)
    n_ok = sum(1 for p in frames if p.exists())
    print(f"{LOG_PREFIX} exit code {rc}; {n_ok}/{len(frames)} frames present")
    print(scene["output_dir"])
    return rc


def run_in_editor(opts):
    bundle = opts["bundle"]
    if not bundle:
        unreal.log_error(
            f"{LOG_PREFIX} no bundle: set LES_BUNDLE or run prepare_unreal()"
        )
        return None
    scene = load_scene(bundle)
    q_sys = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    if q_sys.is_rendering():
        unreal.log_warning(f"{LOG_PREFIX} a render is already running")
        return None
    queue = q_sys.get_queue()
    job_name = f"LES_{scene['case']}"
    for job in list(queue.get_jobs()):
        if job.get_editor_property("job_name") == job_name:
            queue.delete_job(job)
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.set_editor_property("job_name", job_name)
    job.set_editor_property("sequence", unreal.SoftObjectPath(scene["sequence"]))
    job.set_editor_property(
        "map", unreal.SoftObjectPath(f"{scene['map']}.{scene['map'].rsplit('/', 1)[1]}")
    )
    preset = unreal.EditorAssetLibrary.load_asset(scene["mrq_config"].split(".", 1)[0])
    if preset is None:
        unreal.log_error(
            f"{LOG_PREFIX} MRQ config {scene['mrq_config']} not found: re-run build_scene.py"
        )
        return None
    job.set_configuration(preset)

    def _done(executor, success):
        unreal.log(
            f"{LOG_PREFIX} finished (success={success}); frames in {scene['output_dir']}"
        )

    executor = q_sys.render_queue_with_executor(unreal.MoviePipelinePIEExecutor)
    if executor is not None:
        try:
            executor.on_executor_finished_delegate.add_callable_unique(_done)
        except Exception:  # noqa: BLE001
            pass
        _KEEPALIVE.append((executor, _done))
    unreal.log(f"{LOG_PREFIX} rendering {scene['sequence']} -> {scene['output_dir']}")
    print(scene["output_dir"])
    return executor


def main(argv=None):
    opts = parse_args(
        sys.argv[1:] if argv is None else argv, default_bundle=DEFAULT_BUNDLE
    )
    if unreal is not None:
        return run_in_editor(opts)
    return run_outside(opts)


if __name__ == "__main__":
    result = main()
    if unreal is None:
        sys.exit(result or 0)
