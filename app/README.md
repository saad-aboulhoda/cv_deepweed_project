# WeedSight : identification des mauvaises herbes

Page web + API qui utilise le meilleur modèle du notebook
`DeepWeeds_Top3_Normalization_Augmentation.ipynb`, **DINOv2 + Softmax**
(variante `softmax_raw_original`), pour reconnaître **Chinee Apple / Lantana / Snake Weed**
sur une photo. Les scores réels (validation, test, par classe, matrice de confusion)
sont recalculés par `scripts/prepare_data.py` et affichés dans l'application.

## Fonctionnement

1. L'image est redimensionnée en 224 × 224 et normalisée (moyenne et écart-type ImageNet).
2. **DINOv2 ViT-S/14** (gelé) la transforme en un vecteur x de 384 caractéristiques.
3. Le **StandardScaler** appris sur les 1 800 images d'entraînement centre et réduit x.
4. La couche **Dense(3)** calcule un score par espèce, **z = W·x + b**.
5. **Softmax** transforme ces scores en probabilités : p_i = exp(z_i) / Σ exp(z_j).

Les probabilités affichées sont celles du modèle Keras entraîné : `W` et `b` en sont extraits
et le script vérifie que l'application donne les mêmes probabilités que Keras sur les
600 images de test.

## Structure

```
Web_project_5_Démonstration/
├── backend/                 ← l'API (Python / FastAPI)
│   ├── main.py              ← routes : /, /api/info, /api/predict, /api/predict/stream…
│   ├── config.py            ← réglages : chemins, classes, taille d'image, étapes
│   ├── model_loader.py      ← charge DINOv2 + StandardScaler + Softmax
│   ├── preprocessing.py     ← image -> tenseur (1, 3, 224, 224)
│   ├── database.py          ← historique des analyses (SQLite)
│   └── species.json         ← fiches des espèces (nom latin, description)
├── frontend/                ← la page web (servie par l'API)
│   ├── index.html           ← la page : en-tête, zone d'analyse, historique
│   ├── img/logo.svg         ← icône 🌿 de l'onglet (favicon)
│   ├── css/style.css        ← apparence
│   ├── js/app.js            ← logique + animations
│   └── libs/gsap.min.js     ← bibliothèque d'animation (en local, marche sans internet)
├── models/
│   ├── DeepWeeds_Top3_Normalization_Augmentation.ipynb  ← notebook du modèle retenu
│   ├── DeepWeeds_Comparison.ipynb                       ← notebook comparatif initial
│   └── deepweeds_dino_softmax/      ← DINOv2, scaler, Softmax Keras, poids W/b, évaluation
├── scripts/
│   ├── prepare_data.py      ← vérifie le modèle, calcule l'évaluation, génère samples/
│   └── build_space.py       ← assemble la version à déployer
├── deploy/                  ← Dockerfile et dépendances pour le déploiement
├── data/                    ← base SQLite et miniatures des images analysées
├── samples/                 ← 18 vraies photos du jeu de test (6 par espèce)
├── run.py                   ← lance l'application (serveur + navigateur)
├── DEPLOIEMENT.md           ← mise en ligne
└── requirements.txt
```

## Installation (une seule fois)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/prepare_data.py      # télécharge DeepWeeds (~470 Mo), vérifie le modèle
```

Le script reproduit la sélection et le découpage du notebook (`seed = 42`), vérifie que les
images ont la même empreinte que dans `metadata.json`, que les caractéristiques DINOv2
correspondent au StandardScaler sauvegardé, et que les probabilités sont identiques à Keras.

## Lancer l'application

```bash
python run.py
```

Le serveur démarre et la page s'ouvre toute seule sur **http://127.0.0.1:8000**
(documentation de l'API : http://127.0.0.1:8000/docs). Utiliser `127.0.0.1` plutôt
que `localhost` : sous Windows, `localhost` ajoute environ 2 secondes à chaque requête.

Au démarrage, la console doit afficher `[OK] DINOv2 ViT-S/14 + Softmax prêt`.
Tout fonctionne hors ligne : le code et les poids de DINOv2 sont dans `models/deepweeds_dino_softmax/`.

## API

| Route | Rôle |
|---|---|
| `GET /api/info` | modèle, évaluation réelle, étapes, fiches espèces |
| `GET /api/samples` | photos du jeu de test avec leur vraie classe |
| `POST /api/predict` | image → espèce, scores z = W·x + b, probabilités Softmax |
| `POST /api/predict/stream` | même traitement, chaque étape envoyée en temps réel (NDJSON) |
| `GET /api/history` | dernières analyses |
| `DELETE /api/history` | efface l'historique |
| `GET /api/stats` | statistiques sur toutes les analyses |
