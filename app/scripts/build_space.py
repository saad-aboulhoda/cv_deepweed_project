"""Assemble la version à mettre en ligne sur Hugging Face Spaces.

Copie uniquement ce dont l'application a besoin dans dist/huggingface-space/
(environ 90 Mo, au lieu de plus de 1 Go pour le projet complet).

Lancer depuis la racine du projet :  python scripts/build_space.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "huggingface-space"
MODEL = Path("models") / "deepweeds_dino_softmax"

# (source, destination) relatives à la racine du projet / du Space.
COPIES = [
    ("deploy/Dockerfile", "Dockerfile"),
    ("deploy/requirements.txt", "requirements.txt"),
    ("deploy/README.md", "README.md"),
    ("backend", "backend"),
    ("frontend", "frontend"),
    (MODEL / "dinov2_vits14_pretrain.pth", MODEL / "dinov2_vits14_pretrain.pth"),
    (MODEL / "dinov2_source" / "dinov2", MODEL / "dinov2_source" / "dinov2"),
    (MODEL / "dinov2_source" / "LICENSE", MODEL / "dinov2_source" / "LICENSE"),
    (MODEL / "softmax_raw_original_scaler.joblib", MODEL / "softmax_raw_original_scaler.joblib"),
    (MODEL / "softmax_weights.npz", MODEL / "softmax_weights.npz"),
    (MODEL / "evaluation.json", MODEL / "evaluation.json"),
    (MODEL / "metadata.json", MODEL / "metadata.json"),
    ("samples", "samples"),
]

# Indispensables : sans eux l'application ne démarre pas ou perd des fonctions.
REQUIRED = [
    MODEL / "dinov2_vits14_pretrain.pth",
    MODEL / "dinov2_source" / "dinov2" / "hub" / "backbones.py",
    MODEL / "softmax_raw_original_scaler.joblib",
    MODEL / "softmax_weights.npz",
    MODEL / "evaluation.json",
    Path("samples/samples.json"),
]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def size_of(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main():
    missing = [str(p) for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        raise SystemExit("Fichiers manquants (lancer d'abord scripts/prepare_data.py ?) :\n  "
                         + "\n  ".join(missing))

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    for src, dst in COPIES:
        src, dst = ROOT / src, OUT / dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            shutil.copy2(src, dst)

    files = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"Prêt : {OUT}")
    print(f"  {files} fichiers, {size_of(OUT) / 1024 / 1024:.0f} Mo")


if __name__ == "__main__":
    main()
