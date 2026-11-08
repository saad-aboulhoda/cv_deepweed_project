"""Prétraitement identique à l'entraînement : RGB -> 224x224 -> normalisation ImageNet.

Découpé en étapes pour pouvoir mesurer et afficher chacune en temps réel.
"""
import base64
import io

from PIL import Image, ImageOps
from torchvision import transforms

from . import config

_resize = transforms.Resize((config.IMG_SIZE, config.IMG_SIZE))  # bilinéaire (PIL)
_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
])
# Chaîne complète, identique à inference.py (utilisée par scripts/prepare_data.py).
_transform = transforms.Compose([_resize, _normalize])


def load_image(data: bytes) -> Image.Image:
    """Ouvre les octets reçus en image RGB (lève une erreur si illisible)."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def resize(img: Image.Image) -> Image.Image:
    """Étape 2 : redimensionnement à la taille d'entrée du modèle."""
    return _resize(img)


def normalize(img: Image.Image):
    """Étape 3 : pixels -> tenseur (1, 3, 224, 224) normalisé (moyenne/écart-type ImageNet)."""
    return _normalize(img).unsqueeze(0)


def to_tensor(img: Image.Image):
    """Image RGB -> tenseur prêt pour DINOv2 (les deux étapes d'un coup)."""
    return normalize(resize(img))


def to_data_url(img: Image.Image) -> str:
    """Image -> data URL JPEG, pour montrer au navigateur ce que reçoit le modèle."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def save_thumbnail(img: Image.Image, path):
    """Miniature carrée (recadrée au centre) pour l'historique."""
    ImageOps.fit(img, (config.THUMB_SIZE, config.THUMB_SIZE)).save(path, quality=85)
