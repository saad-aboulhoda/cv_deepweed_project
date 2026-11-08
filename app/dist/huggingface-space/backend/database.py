"""Historique des analyses, stocké dans une base SQLite (data/weedsight.db)."""
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

_lock = threading.Lock()


@contextmanager
def _connect():
    """Ouvre la base, valide les changements puis ferme la connexion."""
    conn = sqlite3.connect(config.DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    config.DATA_DIR.mkdir(exist_ok=True)
    with _lock, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at    TEXT    NOT NULL,
                filename      TEXT,
                classe        TEXT    NOT NULL,
                confiance     REAL    NOT NULL,
                probabilites  TEXT    NOT NULL,
                temps_ms      REAL    NOT NULL,
                classe_reelle TEXT
            )""")


def add(filename, classe, confiance, probabilites, temps_ms, classe_reelle=None) -> int:
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO analyses (created_at, filename, classe, confiance, probabilites, temps_ms, classe_reelle)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), filename, classe, confiance,
             json.dumps(probabilites), temps_ms, classe_reelle))
        return cur.lastrowid


def recent(limit=5):
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) | {"probabilites": json.loads(r["probabilites"])} for r in rows]


def stats():
    with _lock, _connect() as conn:
        total, conf, temps = conn.execute(
            "SELECT COUNT(*), AVG(confiance), AVG(temps_ms) FROM analyses").fetchone()
        par_classe = dict(conn.execute(
            "SELECT classe, COUNT(*) FROM analyses GROUP BY classe").fetchall())
        verifiees, correctes = conn.execute(
            "SELECT COUNT(*), SUM(classe = classe_reelle) FROM analyses WHERE classe_reelle IS NOT NULL"
        ).fetchone()
    return {
        "total": total,
        "par_classe": {c: par_classe.get(c, 0) for c in config.CLASSES},
        "confiance_moyenne": conf,
        "temps_moyen_ms": temps,
        "exemples_testes": verifiees,
        "exemples_corrects": correctes or 0,
    }


def clear():
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM analyses")
