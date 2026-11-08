"""Utilitaires du projet : fichiers, tableaux, métriques et sauvegarde des modèles.
Les architectures, l'entraînement et les expériences sont dans le notebook.
"""
from pathlib import Path
import json, hashlib, shutil, time, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from IPython.display import display, Markdown

SHORT = ["CA", "LA", "SW"]
REGISTRY = {}
RECORDS = []
HISTORIES = {}

def initialiser(export, classes):
    global EXPORT, MODELS, CLASSES
    EXPORT = Path(export)
    MODELS = EXPORT / "models"
    CLASSES = list(classes)
    MODELS.mkdir(parents=True, exist_ok=True)
    REGISTRY.clear(); RECORDS.clear(); HISTORIES.clear()

def ecrire_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))

def tableau(rows, decimales=2):
    df = pd.DataFrame(rows)
    formatted = df.copy()
    for col in df.select_dtypes(include="floating"):
        formatted[col] = df[col].map(lambda value: f"{value:.{decimales}f}")
    display(Markdown(formatted.to_markdown(index=False, disable_numparse=True)))
    return df

def empreinte(path, algorithme="sha256"):
    digest = hashlib.new(algorithme)
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def telecharger(url, path, md5=None, sha256=None):
    path = Path(path)
    def valide():
        return (path.exists() and (md5 is None or empreinte(path, "md5") == md5)
                and (sha256 is None or empreinte(path) == sha256))
    if valide():
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DeepWeedsProject/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            temporary.replace(path)
            if not valide():
                raise ValueError("Empreinte du téléchargement incorrecte : " + path.name)
            print("Téléchargé :", path.name)
            return path
        except (OSError, ValueError) as error:
            if attempt == 3:
                raise RuntimeError(f"Téléchargement impossible : {url}") from error
            time.sleep(2 * (attempt + 1))

def mesures(truth, pred):
    p, r, f, support = precision_recall_fscore_support(truth, pred, labels=[0, 1, 2], zero_division=0)
    return {"accuracy": float(accuracy_score(truth, pred)), "precision_macro": float(p.mean()),
            "recall_macro": float(r.mean()), "f1_macro": float(f.mean()),
            "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(),
            "support": support.tolist(), "confusion": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist()}

def matrices(items, colonnes=3):
    fig, axes = plt.subplots((len(items) + colonnes - 1) // colonnes, colonnes,
                             figsize=(3.4 * colonnes, 3 * ((len(items) + colonnes - 1) // colonnes)), squeeze=False)
    for ax, (label, cm) in zip(axes.flat, items):
        cm = np.asarray(cm)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.set(title=label, xlabel="Prédit", ylabel="Réel")
        ax.set_xticks([0, 1, 2], SHORT); ax.set_yticks([0, 1, 2], SHORT)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > .55 * cm.max() else "black")
    for ax in list(axes.flat)[len(items):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()

def dossier_modele(model_id):
    directory = MODELS / model_id
    directory.mkdir(exist_ok=True)
    return directory

def sauvegarder_tf(model_id, model, **config):
    import tensorflow as tf
    directory = dossier_modele(model_id)
    # Copier seulement architecture et poids : export pour l'inférence, sans optimiseur.
    inference_model = tf.keras.models.clone_model(model)
    inference_model.set_weights(model.get_weights())
    inference_model.save(directory / "model.keras")
    del inference_model
    info = {"id": model_id, "classes": CLASSES, "artifact": "model.keras", **config}
    REGISTRY[model_id] = info
    ecrire_json(directory / "metadata.json", info)

def sauvegarder_knn(model_id, model, **config):
    directory = dossier_modele(model_id)
    joblib.dump(model, directory / "classifier.joblib")
    info = {"id": model_id, "classes": CLASSES, "artifact": "classifier.joblib", **config}
    REGISTRY[model_id] = info
    ecrire_json(directory / "metadata.json", info)

def enregistrer(model_id, section, val_true, val_pred, test_true, test_pred, history=None, **extra):
    result = {"id": model_id, "section": section,
              "validation": float(accuracy_score(val_true, val_pred)),
              **mesures(test_true, test_pred), **extra}
    directory = dossier_modele(model_id)
    ecrire_json(directory / "evaluation.json", result)
    np.savez_compressed(directory / "predictions.npz", val_true=val_true, val_pred=val_pred,
                         y_true=test_true, prediction=test_pred)
    if history is not None:
        HISTORIES[model_id] = history
        ecrire_json(directory / "history.json", history)
    RECORDS[:] = [r for r in RECORDS if r["id"] != model_id]
    RECORDS.append(result)
    return result

def resume(rows):
    return tableau([{"Modèle": r["id"], "Validation (%)": 100 * r["validation"],
                     "Test (%)": 100 * r["accuracy"], "P macro (%)": 100 * r["precision_macro"],
                     "R macro (%)": 100 * r["recall_macro"], "F1 macro (%)": 100 * r["f1_macro"]} for r in rows])

