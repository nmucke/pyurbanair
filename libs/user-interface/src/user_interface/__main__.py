"""Start the local browser app: python -m user_interface [--repo-root .] [--port 0] [--no-browser]."""

from __future__ import annotations

import argparse
import pathlib
import webbrowser

from user_interface.app import App


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    app = App(args.repo_root, port=args.port)
    print(f"pyurbanair user interface: {app.url}  (Ctrl-C stops it)", flush=True)
    if not args.no_browser:
        webbrowser.open(app.url)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.server.server_close()


if __name__ == "__main__":
    main()
