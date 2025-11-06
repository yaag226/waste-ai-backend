# ============================================
# WASTE AI BACKEND - DOCKERFILE
# ============================================

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Waste Management AI Team"
LABEL description="Backend IA pour gestion des déchets"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Créer répertoire de travail
WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Installer dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier le code
COPY . .

# Créer dossiers nécessaires
RUN mkdir -p /app/trained_models /app/datasets /app/logs

# Exposer le port
EXPOSE 8000

# Sanity check
RUN python -c "import fastapi; import tensorflow; print('✅ Dépendances OK')"

# Commande de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]