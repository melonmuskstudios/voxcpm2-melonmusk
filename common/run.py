"""python -m common.run <model_folder_name>   (run from the repo root)

Starts the model, the 3 endpoints and the tunnel, then prints ONE line to paste into Audico:   <link>|<token>
"""
import importlib
import os
import secrets
import sys
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn

from . import tunnel
from .audico_server import build_app


def main():
    name = sys.argv[1]
    port = int(os.environ.get("AUDICO_PORT", "8000"))
    token = os.environ.get("AUDICO_TOKEN", "").strip() or secrets.token_urlsafe(18)
    work = Path(os.environ.get("AUDICO_WORK", "/kaggle/working/audico_work" if os.path.isdir("/kaggle/working") else "audico_work"))
    work.mkdir(parents=True, exist_ok=True)
    wrapper = importlib.import_module(f"models.{name}.wrapper")
    adapter = wrapper.load(work)            # returns at once; the weights keep loading in the background
    app = build_app(adapter, token, work)

    def announce(url):
        line = f"{url}|{token}"
        (work / "connection.txt").write_text(line, encoding="utf-8")
        print("=" * 70, flush=True)
        print(" AUDICO CONNECTION (paste this one line into Audico):", flush=True)
        print(" " + line, flush=True)
        print("=" * 70, flush=True)

    def after_start():
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/audico_health", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        if os.environ.get("AUDICO_NO_TUNNEL") == "1":      # local testing
            announce(f"http://127.0.0.1:{port}")
            return
        tunnel.start(port, announce, os.environ.get("NGROK_TOKEN", ""), os.environ.get("NGROK_DOMAIN", ""))
    threading.Thread(target=after_start, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
