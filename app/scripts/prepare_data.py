"""Prépare et vérifie le modèle DINOv2 + Softmax pour l'application.

Reproduit la sélection et le découpage du notebook
DeepWeeds_Top3_Normalization_Augmentation.ipynb (seed 42, 1 800 / 600 / 600), puis :

  1. recalcule les caractéristiques DINOv2 des 3 000 images (même prétraitement) ;
  2. vérifie qu'elles correspondent au StandardScaler sauvegardé (moyenne du train) ;
  3. extrait W et b de la couche Dense(3, softmax) Keras -> softmax_weights.npz,
     et vérifie que softmax(W·x + b) calculé en NumPy donne les mêmes probabilités que Keras ;
  4. recalcule précision, F1 macro, rapport par classe et matrice de confusion
     (validation et test) -> evaluation.json ;
  5. enregistre des images du jeu de test comme exemples -> samples/.

Lancer depuis la racine du projet :  python scripts/prepare_data.py
(nécessite tensorflow et tensorflow-datasets, uniquement pour ce script)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
SAMPLES_PER_CLASS = 6
BATCH = 32


def load_selection():
    import tensorflow_datasets as tfds

    dataset, info = tfds.load("deep_weeds", split="train", as_supervised=True,
                              with_info=True, shuffle_files=False)
    counts = {0: 0, 1: 0, 7: 0}
    label_map = {0: 0, 1: 1, 7: 2}
    images, labels = [], []
    for image, label in tfds.as_numpy(dataset):
        label = int(label)
        if label in counts and counts[label] < 1000:
            images.append(image)
            labels.append(label_map[label])
            counts[label] += 1
        if min(counts.values()) == 1000:
            break
    return np.array(images), np.array(labels)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def report(y_true, y_pred, classes):
    rep = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=classes,
                                output_dict=True, zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro") * 100, 2),
        "images": int(len(y_true)),
        "par_classe": {c: {"precision": round(rep[c]["precision"] * 100, 2),
                           "rappel": round(rep[c]["recall"] * 100, 2),
                           "f1": round(rep[c]["f1-score"] * 100, 2)} for c in classes},
        "matrice_confusion": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def main():
    import joblib
    import torch
    from backend import config
    from backend.model_loader import load_dino
    from backend.preprocessing import _transform

    meta = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))

    print("1/5 Chargement de DeepWeeds…")
    images, y = load_selection()
    image_hash = hashlib.sha256(memoryview(images)).hexdigest()
    same_images = image_hash == meta["image_hash"]
    print(f"    {len(y)} images, empreinte identique au notebook : {same_images}")
    assert same_images, "Les images ne sont pas celles du notebook."

    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=SEED,
                                          stratify=y[train_val_idx])

    print("2/5 Caractéristiques DINOv2 des 3 000 images (quelques minutes sur CPU)…")
    dino = load_dino()
    feats = []
    with torch.inference_mode():
        for start in range(0, len(images), BATCH):
            batch = torch.stack([_transform(Image.fromarray(im)) for im in images[start:start + BATCH]])
            feats.append(dino(batch).numpy())
            print(f"\r    {min(start + BATCH, len(images))}/{len(images)}", end="", flush=True)
    feats = np.concatenate(feats)
    print()

    scaler = joblib.load(config.SCALER_FILE)
    mean_gap = float(np.abs(feats[train_idx].mean(axis=0) - scaler.mean_).max())
    std_gap = float(np.abs(feats[train_idx].std(axis=0) - scaler.scale_).max())
    print(f"    écart avec le StandardScaler : moyenne {mean_gap:.2e}, écart-type {std_gap:.2e}")
    assert mean_gap < 1e-2 and std_gap < 1e-2, "Prétraitement différent de l'entraînement."

    print("3/5 Extraction de W et b et comparaison avec Keras…")
    import keras
    model = keras.models.load_model(config.KERAS_FILE, compile=False)
    dense = [layer for layer in model.layers if layer.get_weights()]
    assert len(dense) == 1, "Le modèle devrait contenir une seule couche Dense."
    W, b = dense[0].get_weights()                     # W : (384, 3), b : (3,)
    np.savez(config.WEIGHTS_FILE, W=W, b=b, classes=np.array(meta["classes"]))

    x_test = scaler.transform(feats[test_idx])
    p_keras = model.predict(x_test, verbose=0)
    p_numpy = softmax(x_test @ W + b)
    prob_gap = float(np.abs(p_keras - p_numpy).max())
    same_pred = int((p_keras.argmax(1) == p_numpy.argmax(1)).sum())
    print(f"    écart max des probabilités Keras / NumPy : {prob_gap:.2e} ; "
          f"mêmes prédictions : {same_pred}/{len(test_idx)}")
    assert prob_gap < 1e-4 and same_pred == len(test_idx)

    print("4/5 Évaluation sur validation et test…")
    classes = config.CLASSES
    pred = lambda idx: softmax(scaler.transform(feats[idx]) @ W + b).argmax(1)
    evaluation = {
        "modele": config.MODEL_NAME,
        "notebook": "DeepWeeds_Top3_Normalization_Augmentation.ipynb",
        "variante": "softmax_raw_original",
        "validation": report(y[val_idx], pred(val_idx), classes),
        "test": report(y[test_idx], pred(test_idx), classes),
        "verification": {
            "empreinte_images": same_images,
            "ecart_scaler_moyenne": mean_gap,
            "ecart_probabilites_keras_numpy": prob_gap,
        },
    }
    config.EVALUATION_FILE.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    validation : {evaluation['validation']['accuracy']} %  |  "
          f"test : {evaluation['test']['accuracy']} %  (F1 {evaluation['test']['f1_macro']} %)")

    print("5/5 Exemples du jeu de test…")
    samples_dir = config.SAMPLES_DIR
    samples_dir.mkdir(exist_ok=True)
    for old in samples_dir.glob("*.jpg"):
        old.unlink()
    samples = []
    for label, name in enumerate(classes):
        slug = name.lower().replace(" ", "-")
        for n, i in enumerate([i for i in test_idx if y[i] == label][:SAMPLES_PER_CLASS], 1):
            filename = f"{slug}-{n}.jpg"
            Image.fromarray(images[i]).save(samples_dir / filename, quality=92)
            samples.append({"fichier": filename, "classe": name})
    config.SAMPLES_FILE.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Terminé : {len(samples)} exemples, poids et évaluation enregistrés.")


if __name__ == "__main__":
    main()
