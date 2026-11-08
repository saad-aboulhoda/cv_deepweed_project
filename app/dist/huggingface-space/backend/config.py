"""Réglages de l'application."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
SPECIES_FILE = Path(__file__).resolve().parent / "species.json"

# Modèle exporté depuis DeepWeeds_Top3_Normalization_Augmentation.ipynb
# (variante « softmax_raw_original » : DINOv2 gelé + StandardScaler + Dense(3, softmax)).
MODEL_DIR = ROOT / "models" / "deepweeds_dino_softmax"
DINO_SOURCE = MODEL_DIR / "dinov2_source"
DINO_WEIGHTS = MODEL_DIR / "dinov2_vits14_pretrain.pth"
KERAS_FILE = MODEL_DIR / "softmax_raw_original.keras"
SCALER_FILE = MODEL_DIR / "softmax_raw_original_scaler.joblib"
METADATA_FILE = MODEL_DIR / "metadata.json"
# Générés et vérifiés par scripts/prepare_data.py :
WEIGHTS_FILE = MODEL_DIR / "softmax_weights.npz"   # W (384 x 3) et b (3) de la couche Dense
EVALUATION_FILE = MODEL_DIR / "evaluation.json"    # scores réels sur validation et test

# Données de l'application.
DATA_DIR = ROOT / "data"
THUMBS_DIR = DATA_DIR / "thumbs"            # miniatures des images analysées
DB_FILE = DATA_DIR / "weedsight.db"         # historique des analyses (SQLite)
SAMPLES_DIR = ROOT / "samples"              # images du jeu de test DeepWeeds
SAMPLES_FILE = SAMPLES_DIR / "samples.json"

# Même ordre que les labels de l'entraînement : 0, 1, 2.
CLASSES = ["Chinee Apple", "Lantana", "Snake Weed"]

# Prétraitement DINOv2 (identique au notebook, voir metadata.json).
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MODEL_NAME = "DINOv2 ViT-S/14 + Softmax"

# Étapes affichées pendant l'analyse (une par étape réellement exécutée).
PIPELINE = [
    "Réception de l'image",
    f"Redimensionnement en {IMG_SIZE} × {IMG_SIZE} px",
    "Normalisation des couleurs (ImageNet)",
    "Extraction des caractéristiques (DINOv2)",
    "Standardisation (StandardScaler)",
    "Couche Softmax : z = W·x + b",
]

# Taille maximale d'une image envoyée (octets) et taille des miniatures.
MAX_UPLOAD = 10 * 1024 * 1024
THUMB_SIZE = 160
