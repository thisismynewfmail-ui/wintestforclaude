#!/usr/bin/env python3
"""
start.py — Entry point for the companion chat system.
Launches the Flask server with concurrent-user support.
"""
import os
import sys
import socket

# Make project root importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app.server import create_app, get_config


def get_lan_ip() -> str:
    """Best-effort LAN IP discovery so the user knows where to point browsers."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def main():
    app = create_app(BASE_DIR)
    cfg = get_config()
    port = int(cfg.get("server", {}).get("port", 5005))
    lan_enabled = bool(cfg.get("server", {}).get("lan_visible", True))
    host = "0.0.0.0" if lan_enabled else "127.0.0.1"

    lan_ip = get_lan_ip()
    print("=" * 60)
    print("  Companion online.")
    print("  Local :  http://127.0.0.1:{}".format(port))
    if lan_enabled:
        print("  LAN   :  http://{}:{}".format(lan_ip, port))
    print("  Admin :  /admincontrols  (must be logged in as admin)")
    print("=" * 60)

    # threaded=True allows concurrent users and concurrent SSE streams.
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
