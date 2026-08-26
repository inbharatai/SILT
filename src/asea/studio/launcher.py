"""Cross-platform local launcher for SILT Studio.

Starts the Studio strictly on loopback, reuses an already-running SILT Studio,
and opens the user's browser automatically. This module intentionally does not
support public bind addresses: SILT Studio's default security boundary is the
user's own machine.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import urllib.error
import urllib.request
import webbrowser

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8377


def _health_url(port: int) -> str:
    return f"http://{DEFAULT_HOST}:{port}/api/health"


def _studio_url(port: int) -> str:
    return f"http://{DEFAULT_HOST}:{port}/"


def _silt_is_running(port: int, timeout: float = 0.8) -> bool:
    try:
        with urllib.request.urlopen(_health_url(port), timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return bool(
                payload.get("ok") is True
                and payload.get("service") == "silt-studio"
                and payload.get("mock_free") is True
            )
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
        return False


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((DEFAULT_HOST, port)) == 0


def _open_browser(port: int) -> None:
    webbrowser.open(_studio_url(port), new=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="silt-studio",
        description="Start SILT Studio locally and open it in your browser.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Loopback port to use (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the Studio without opening a browser window.",
    )
    args = parser.parse_args(argv)

    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")

    url = _studio_url(args.port)

    if _silt_is_running(args.port):
        print(f"SILT Studio is already running at {url}")
        if not args.no_browser:
            _open_browser(args.port)
        return 0

    if _port_is_open(args.port):
        print(
            f"Refusing to start: {DEFAULT_HOST}:{args.port} is already in use "
            "by a service that is not SILT Studio.",
            file=sys.stderr,
        )
        return 2

    try:
        import uvicorn
    except ImportError:
        print(
            'SILT Studio dependencies are not installed. Run: '
            'python -m pip install -e ".[studio]"',
            file=sys.stderr,
        )
        return 2

    print("SILT Studio local-first launcher")
    print(f"Binding only to {DEFAULT_HOST}:{args.port}")
    print(f"Studio: {url}")
    print("Models, skill packets, benchmark state and audit data remain local.")

    if not args.no_browser:
        threading.Timer(0.9, _open_browser, args=(args.port,)).start()

    uvicorn.run(
        "asea.studio.server:app",
        host=DEFAULT_HOST,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
