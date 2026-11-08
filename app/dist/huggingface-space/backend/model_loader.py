"""Chargement du modèle : DINOv2 (extracteur gelé) + StandardScaler + Softmax.

1. DINOv2 ViT-S/14 transforme l'image en un vecteur x de 384 caractéristiques.
2. Le StandardScaler (appris sur le train) centre et réduit chaque caractéristique.
3. La couche Dense(3) calcule les scores z = W·x + b, puis Softmax les transforme
   en probabilités : p_i = exp(z_i) / somme_j exp(z_j).

W et b sont ceux du modèle Keras entraîné, extraits et vérifiés par
scripts/prepare_data.py (mêmes probabilités que Keras sur les 600 images de test).
Tout est chargé en local depuis models/deepweeds_dino_softmax (pas d'internet).
"""
import os
import sys
import warnings

os.environ.setdefault("XFORMERS_DISABLED", "1")  # attention standard, compatible CPU
warnings.filterwarnings("ignore", message="xFormers is")
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

import joblib
import numpy as np
import torch

from . import config


def load_dino():
    """DINOv2 ViT-S/14 avec les poids pré-entraînés fournis (backbone gelé)."""
    if str(config.DINO_SOURCE) not in sys.path:
        sys.path.insert(0, str(config.DINO_SOURCE))
    from dinov2.hub.backbones import dinov2_vits14

    model = dinov2_vits14(pretrained=False)
    state = torch.load(config.DINO_WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


class Predictor:
    def __init__(self):
        self.name = config.MODEL_NAME
        self.ready = False
        self.error = None
        try:
            self._load()
            self.ready = True
            print(f"[OK] {self.name} prêt ({self.n_features} caractéristiques -> {len(config.CLASSES)} classes).",
                  flush=True)
        except Exception as e:
            self.error = str(e)
            print(f"[ERREUR] Impossible de charger le modèle : {e}", flush=True)

    def _load(self):
        self.dino = load_dino()

        scaler = joblib.load(config.SCALER_FILE)
        self.mean = scaler.mean_.astype(np.float64)
        self.scale = scaler.scale_.astype(np.float64)

        if not config.WEIGHTS_FILE.exists():
            raise FileNotFoundError(
                f"{config.WEIGHTS_FILE.name} absent : lancer d'abord python scripts/prepare_data.py")
        weights = np.load(config.WEIGHTS_FILE)
        self.W = weights["W"].astype(np.float64)   # (384, 3)
        self.b = weights["b"].astype(np.float64)   # (3,)
        self.n_features = self.W.shape[0]

        # Première prédiction « à vide » pour que la 1re vraie requête soit rapide.
        self.predict(torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE))

    @torch.inference_mode()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """DINOv2 : image (1, 3, 224, 224) -> vecteur (1, 384)."""
        return self.dino(x).cpu().numpy().astype(np.float64)

    def standardize(self, features: np.ndarray) -> np.ndarray:
        """StandardScaler : (x - moyenne du train) / écart-type du train."""
        return (features - self.mean) / self.scale

    def softmax_layer(self, x: np.ndarray):
        """Dense(3) + Softmax : renvoie (scores z = W·x + b, probabilités)."""
        z = x @ self.W + self.b
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return z[0], (e / e.sum(axis=1, keepdims=True))[0]

    def predict(self, x: torch.Tensor):
        """Toute la chaîne d'un coup : renvoie (scores z, probabilités)."""
        return self.softmax_layer(self.standardize(self.extract(x)))
