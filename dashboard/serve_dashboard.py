"""
Serves the live training dashboard on http://localhost:8000.

Usage:
    python dashboard/serve_dashboard.py

Run this in a separate terminal from `python src/server.py` — it just serves
files, it doesn't run training itself. It reads live_metrics.json from
../models (the same --out_dir server.py writes to) and copies it next to
dashboard.html on each request so the browser can fetch it same-origin.
"""
import http.server
import os
import shutil
import socketserver
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "..", "models")
LIVE_METRICS_SRC = os.path.join(MODELS_DIR, "live_metrics.json")
LIVE_METRICS_DST = os.path.join(HERE, "live_metrics.json")
PORT = 8000


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HERE, **kwargs)

    def do_GET(self):
        # Refresh the local copy of live_metrics.json right before serving it,
        # so every poll from the browser gets the latest data written by
        # server.py without needing a symlink or file watcher.
        if self.path.startswith("/live_metrics.json"):
            if os.path.exists(LIVE_METRICS_SRC):
                shutil.copyfile(LIVE_METRICS_SRC, LIVE_METRICS_DST)
            else:
                self.send_response(404)
                self.end_headers()
                return
        return super().do_GET()

    def log_message(self, format, *args):
        pass  # keep the terminal quiet


def main():
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        url = f"http://localhost:{PORT}/dashboard.html"
        print(f"Dashboard running at {url}")
        print("Leave this running, then start training in another terminal:")
        print("    python src/server.py --rounds 20")
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
