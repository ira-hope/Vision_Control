"""
Live HTML dashboard for face locking events.

Serves dashboard/index.html and JSON APIs that read history_log.jsonl
and lock_state.json (written by detect.py or faceLockServo.py).

Run (second terminal while detect or faceLockServo is running):
  python -m src.dashboard

Open: http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import DASHBOARD_DIR
from .history_manager import read_events_from_disk
from .lock_state import read_lock_state

DASHBOARD_HTML = DASHBOARD_DIR / "index.html"
DEFAULT_PORT = 8765


def _is_tracked_action(action: str) -> bool:
    return (
        action.startswith("moved ")
        or action in ("smile", "blink")
    )


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the dashboard page and polling APIs."""

    def log_message(self, fmt: str, *args) -> None:
        # Quieter than default request logging
        pass

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404, "Dashboard HTML not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)
        limit = int(qs.get("limit", ["200"])[0])

        if route in ("/", "/index.html"):
            self._send_html(DASHBOARD_HTML)
            return

        if route == "/api/status":
            self._send_json(read_lock_state())
            return

        if route == "/api/events":
            events = read_events_from_disk(limit=limit if limit > 0 else None)
            self._send_json({"events": events})
            return

        if route == "/api/movements":
            events = read_events_from_disk(limit=limit if limit > 0 else None)
            movements = [e for e in events if _is_tracked_action(e.get("action", ""))]
            self._send_json({"movements": movements})
            return

        self.send_error(404, "Not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face locking event dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Dashboard running at {url}")
    print("Start detect.py or faceLockServo.py in another terminal for live events.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
