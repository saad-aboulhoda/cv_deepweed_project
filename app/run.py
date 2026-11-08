"""Lance WeedSight : démarre l'API puis ouvre la page dans le navigateur.

    python run.py
"""
import threading
import time
import urllib.request
import webbrowser

import uvicorn

HOST, PORT = "127.0.0.1", 8000   # 127.0.0.1 plutôt que localhost : évite ~2 s d'attente sous Windows
URL = f"http://{HOST}:{PORT}"


def open_when_ready():
    """Attend que le modèle soit chargé, puis ouvre la page."""
    for _ in range(300):
        try:
            urllib.request.urlopen(URL + "/api/info", timeout=1)
            print(f"\n  WeedSight est prêt : {URL}\n", flush=True)
            webbrowser.open(URL)
            return
        except Exception:
            time.sleep(0.5)


if __name__ == "__main__":
    threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run("backend.main:app", host=HOST, port=PORT, log_level="warning")
