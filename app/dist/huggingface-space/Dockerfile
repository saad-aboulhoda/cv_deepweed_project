# Image Docker de WeedSight (CPU) : Google Cloud Run ou Hugging Face Spaces.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2

# Hugging Face exécute les Spaces avec l'utilisateur 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

# PyTorch en version CPU (beaucoup plus léger que la version GPU), puis le reste.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt

COPY --chown=user . .
USER user

# Port : imposé par la plateforme via $PORT (Cloud Run : 8080), sinon 7860 (Hugging Face).
ENV PORT=7860
EXPOSE 7860
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
