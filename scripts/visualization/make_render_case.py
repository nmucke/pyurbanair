"""Create a render case folder from a state file.

Discovers the case (state, geometry, inflow params) and links or copies them
into a folder. Useful for staging runs before visualization. Usage:

    python scripts/visualization/make_render_case.py \\
        training_data/pyudales_idealized/state/train/sample_0000.nc \\
        /tmp/render_demo --render-yaml

Links files as absolute symlinks by default (--copy makes them copies).
Writes a commented render.yaml template with --render-yaml.
Refuses to overwrite existing files unless --force.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from les_render.case import discover_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", help="path to state.nc or a case folder")
    parser.add_argument("out_folder", help="output folder to create")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy files instead of symlinking (default: absolute symlinks)",
    )
    parser.add_argument(
        "--render-yaml",
        action="store_true",
        help="write a commented render.yaml template with example overrides",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing files",
    )
    args = parser.parse_args()

    state_path = pathlib.Path(args.state)
    out_dir = pathlib.Path(args.out_folder)

    # Discover the case.
    case = discover_case(state_path)

    # Create output folder.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Link/copy state.nc.
    state_out = out_dir / "state.nc"
    _link_or_copy(case.state_path, state_out, args.copy, args.force)

    # Link/copy geometry.
    if case.geometry_path is not None:
        geom_out = out_dir / case.geometry_path.name
        _link_or_copy(case.geometry_path, geom_out, args.copy, args.force)
        print(f"geometry → {geom_out}")
    else:
        print("geometry → (none; will use blanking)")

    # Link/copy params.
    if case.params_path is not None:
        params_out = out_dir / "params.nc"
        _link_or_copy(case.params_path, params_out, args.copy, args.force)
        print(f"params → {params_out}")
    else:
        print("params → (none)")

    # Write render.yaml template.
    if args.render_yaml:
        render_yaml = out_dir / "render.yaml"
        if render_yaml.exists() and not args.force:
            print(f"refuse to overwrite {render_yaml}; use --force")
            sys.exit(1)
        render_yaml.write_text(
            "# Override render presets for this case\n"
            "# render_preset: {look: daylight}\n"
            "# time: {duration: 20, t_start: 200}\n"
        )
        print(f"render.yaml → {render_yaml}")

    print(f"case: {case.name}")


def _link_or_copy(
    src: pathlib.Path, dst: pathlib.Path, copy: bool, force: bool
) -> None:
    if dst.exists() and not force:
        print(f"refuse to overwrite {dst}; use --force", file=sys.stderr)
        sys.exit(1)
    if dst.exists():
        dst.unlink()
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


if __name__ == "__main__":
    main()
