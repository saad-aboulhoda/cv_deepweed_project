"""API FastAPI de WeedSight.

Lancer depuis la racine du projet :
    python run.py            (ou : uvicorn backend.main:app)
puis ouvrir http://127.0.0.1:8000  (documentation de l'API : /docs)
"""
import json
import shutil
import threading
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, database
from .model_loader import Predictor
from .preprocessing import load_image, normalize, resize, save_thumbnail, to_data_url

app = FastAPI(
    title="WeedSight API",
    description="Identification de mauvaises herbes (Chinee Apple, Lantana, Snake Weed) "
                "avec DINOv2 + Softmax.",
)

# Dossiers de données et base SQLite.
for folder in (config.DATA_DIR, config.THUMBS_DIR, config.SAMPLES_DIR):
    folder.mkdir(parents=True, exist_ok=True)
database.init()


def _read_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


SPECIES = _read_json(config.SPECIES_FILE, {})
SAMPLES = _read_json(config.SAMPLES_FILE, [])
SAMPLE_LABELS = {s["fichier"]: s["classe"] for s in SAMPLES}
EVALUATION = _read_json(config.EVALUATION_FILE, None)

predictor = Predictor()
_predict_lock = threading.Lock()


def thumb_url(analysis_id: int) -> Optional[str]:
    if (config.THUMBS_DIR / f"{analysis_id}.jpg").exists():
        return f"/media/thumbs/{analysis_id}.jpg"
    return None


@app.get("/api/info")
def info():
    """Modèle, scores réels (validation/test), étapes du traitement et fiches espèces."""
    return {
        "modele": predictor.name,
        "pret": predictor.ready,
        "erreur": predictor.error,
        "caracteristiques": getattr(predictor, "n_features", None),
        "evaluation": EVALUATION,
        "etapes": config.PIPELINE,
        "especes": [{"nom": c, **SPECIES.get(c, {})} for c in config.CLASSES],
    }


@app.get("/api/samples")
def samples():
    """Images du jeu de test DeepWeeds, jamais vues pendant l'entraînement."""
    return [{"fichier": s["fichier"], "classe": s["classe"], "url": f"/media/samples/{s['fichier']}"}
            for s in SAMPLES if (config.SAMPLES_DIR / s["fichier"]).exists()]


def _read_upload(file: UploadFile):
    """Contrôles communs : modèle prêt, type, taille, image lisible. Renvoie (image, octets, durée)."""
    if not predictor.ready:
        raise HTTPException(503, "Le modèle n'est pas chargé. Consultez la console du serveur.")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Le fichier envoyé n'est pas une image.")
    t0 = time.perf_counter()
    data = file.file.read()
    if len(data) > config.MAX_UPLOAD:
        raise HTTPException(413, "L'image dépasse la taille maximale de 10 Mo.")
    try:
        img = load_image(data)
    except Exception:
        raise HTTPException(400, "Image illisible ou format non supporté.")
    return img, data, (time.perf_counter() - t0) * 1000


def _fmt(values):
    return " / ".join(f"{v:+.2f}" for v in values)


def run_pipeline(img, data, read_ms, filename, exemple):
    """Exécute le traitement étape par étape et produit un événement à la fin de chacune.

    Événements : {"etape": i, "duree_ms": ..., "detail": ...} puis {"resultat": {...}}.
    """
    durations = [read_ms]

    def step(i, t0, detail, **extra):
        if t0 is not None:
            durations.append((time.perf_counter() - t0) * 1000)
        return {"etape": i, "duree_ms": round(durations[-1], 1), "detail": detail, **extra}

    yield step(0, None, f"{img.width} × {img.height} px · {len(data) / 1024:.0f} Ko")

    t0 = time.perf_counter()
    small = resize(img)
    yield step(1, t0, f"{img.width} × {img.height} → {small.width} × {small.height} px",
               apercu=to_data_url(small))

    t0 = time.perf_counter()
    x = normalize(small)
    yield step(2, t0, f"tenseur {tuple(x.shape)}, moyenne {x.mean().item():+.2f}")

    t0 = time.perf_counter()
    with _predict_lock:
        features = predictor.extract(x)
    yield step(3, t0, f"vecteur x de {features.shape[1]} caractéristiques")

    t0 = time.perf_counter()
    scaled = predictor.standardize(features)
    yield step(4, t0, f"moyenne {scaled.mean():+.2f}, écart-type {scaled.std():.2f}")

    t0 = time.perf_counter()
    logits, probs = predictor.softmax_layer(scaled)
    yield step(5, t0, "z = " + _fmt(logits))

    elapsed_ms = round(sum(durations), 1)
    best = int(np.argmax(probs))
    classe = config.CLASSES[best]
    probabilites = {c: float(p) for c, p in zip(config.CLASSES, probs)}
    classe_reelle = SAMPLE_LABELS.get(exemple) if exemple else None

    analysis_id = database.add(filename, classe, float(probs[best]), probabilites,
                               elapsed_ms, classe_reelle)
    save_thumbnail(img, config.THUMBS_DIR / f"{analysis_id}.jpg")

    yield {"resultat": {
        "id": analysis_id,
        "classe": classe,
        "confiance": float(probs[best]),
        "probabilites": probabilites,
        "scores": {c: float(z) for c, z in zip(config.CLASSES, logits)},
        "classe_reelle": classe_reelle,
        "temps_ms": elapsed_ms,
        "modele": predictor.name,
    }}


@app.post("/api/predict")
def predict(file: UploadFile = File(...), exemple: Optional[str] = Form(None)):
    """Reçoit une image, renvoie l'espèce prédite, les scores z = W·x + b et les probabilités Softmax."""
    img, data, read_ms = _read_upload(file)
    for event in run_pipeline(img, data, read_ms, file.filename, exemple):
        if "resultat" in event:
            return event["resultat"]


@app.post("/api/predict/stream")
def predict_stream(file: UploadFile = File(...), exemple: Optional[str] = Form(None)):
    """Même traitement, mais chaque étape est envoyée dès qu'elle est terminée (une ligne JSON par étape)."""
    img, data, read_ms = _read_upload(file)

    def events():
        try:
            for event in run_pipeline(img, data, read_ms, file.filename, exemple):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"erreur": f"Erreur pendant l'analyse : {e}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/history")
def history(limit: int = 5):
    """Dernières analyses enregistrées."""
    items = database.recent(min(max(limit, 1), 50))
    return [{"id": r["id"], "date": r["created_at"], "classe": r["classe"],
             "confiance": r["confiance"], "classe_reelle": r["classe_reelle"],
             "image": thumb_url(r["id"])} for r in items]


@app.delete("/api/history")
def clear_history():
    """Efface l'historique et les miniatures associées."""
    database.clear()
    shutil.rmtree(config.THUMBS_DIR, ignore_errors=True)
    config.THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    return {"ok": True}


@app.get("/api/stats")
def stats():
    """Statistiques calculées sur toutes les analyses enregistrées."""
    return database.stats()


# Fichiers servis : page web, miniatures, exemples.
app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")
app.mount("/media/thumbs", StaticFiles(directory=config.THUMBS_DIR), name="thumbs")
app.mount("/media/samples", StaticFiles(directory=config.SAMPLES_DIR), name="samples")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")
