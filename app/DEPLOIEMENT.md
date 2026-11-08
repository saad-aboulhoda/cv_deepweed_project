# Mettre WeedSight en ligne

L'application est une image Docker (FastAPI + PyTorch CPU + DINOv2 + KNN, environ 1 Go de RAM).
Elle est hébergée sur **Google Cloud Run** : gratuit dans la limite du niveau gratuit,
mise en veille automatique quand personne ne l'utilise.

## 1. Assembler la version à mettre en ligne

```bash
python scripts/build_space.py
```

Le dossier `dist/huggingface-space/` (environ 108 Mo) contient uniquement le nécessaire :
backend, page web, DINOv2 + KNN, miniatures de référence et exemples.

Test facultatif avec Docker :

```bash
cd dist/huggingface-space
docker build -t weedsight .
docker run --rm -p 7860:7860 weedsight      # puis ouvrir http://127.0.0.1:7860
```

## 2. Google Cloud Run (méthode utilisée)

Prérequis, une seule fois :
- un **compte de facturation** Google Cloud : https://console.cloud.google.com/billing
  (carte demandée ; un usage de démo reste dans le niveau gratuit) ;
- l'outil `gcloud` connecté : `gcloud auth login`.

```bash
# Projet dédié, relié au compte de facturation
gcloud projects create weedsight-baskad --name="WeedSight"
gcloud billing projects link weedsight-baskad --billing-account=<ID_DU_COMPTE>
gcloud config set project weedsight-baskad
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# Construction de l'image dans Google Cloud et déploiement
gcloud run deploy weedsight --source dist/huggingface-space --region europe-west1 \
  --allow-unauthenticated --memory 2Gi --cpu 2 --cpu-boost \
  --min-instances 0 --max-instances 2 --timeout 120
```

La commande affiche l'adresse publique : `https://weedsight-xxxxxxxx.europe-west1.run.app`.

- `--max-instances 2` plafonne le nombre de serveurs (et donc les coûts possibles).
- `--min-instances 0` : aucun serveur ne tourne quand personne n'utilise l'application.
  La première visite après une pause prend environ 30 secondes (chargement de DINOv2).

**Mettre à jour** après une modification :

```bash
python scripts/build_space.py
gcloud run deploy weedsight --source dist/huggingface-space --region europe-west1
```

## 3. Autre option : Hugging Face Spaces (abonnement PRO requis)

Les Spaces Docker demandent désormais un abonnement PRO. Avec un compte PRO :

```bash
hf auth login
hf repos create <utilisateur>/weedsight --type space --space-sdk docker --public --exist-ok
hf upload <utilisateur>/weedsight dist/huggingface-space . --type space
```

## Bon à savoir

- **Historique et statistiques** : la base SQLite est locale au serveur ; elle repart à zéro
  quand Cloud Run redémarre ou remplace le serveur.
- **Logs** : console Google Cloud > Cloud Run > weedsight > Journaux.
