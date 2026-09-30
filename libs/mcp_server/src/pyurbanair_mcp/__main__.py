"""Start stdio tools with an explicit checkout and optional local storage root."""

from __future__ import annotations

import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, required=True)
    parser.add_argument("--store-root", type=pathlib.Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print read-only capability checks and exit",
    )
    args = parser.parse_args()
    if args.check:
        from pyurbanair_mcp.tools import Tools

        print(
            json.dumps(
                Tools(args.repo_root, args.store_root).get_capabilities(), indent=2
            )
        )
        return
    from pyurbanair_mcp.server import create_server

    create_server(args.repo_root, args.store_root).run(transport="stdio")


if __name__ == "__main__":
    main()
