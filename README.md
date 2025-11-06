# 🚀 Waste AI Backend

**Backend Intelligence Artificielle pour la Gestion des Déchets au Burkina Faso**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table des Matières

1. [Fonctionnalités](#-fonctionnalités)
2. [Architecture](#-architecture)
3. [Technologies](#-technologies)
4. [Installation](#-installation)
5. [Configuration](#-configuration)
6. [Utilisation](#-utilisation)
7. [API Endpoints](#-api-endpoints)
8. [Modèles IA](#-modèles-ia)
9. [Tests](#-tests)
10. [Déploiement](#-déploiement)
11. [Contribution](#-contribution)

---

## 🎯 Fonctionnalités

### 1. 🗺️ **Analyse de Risque**
- Détection automatique des zones à risque
- Clustering ML (DBSCAN) sur données géospatiales
- Prédiction de niveaux de risque (Low/Medium/High/Critical)
- Génération de heatmaps
- Identification des zones prioritaires

### 2. 🖼️ **Classification d'Images**
- Classification automatique des types de déchets
- 9 classes : plastique, papier, aluminium, médical, organique, verre, électronique, textile, autre
- CNN basé sur MobileNetV2 (transfer learning)
- Validation qualité d'image
- Confiance par classe

### 3. 📏 **Estimation de Taille**
- Estimation de volume (m³)
- Calcul de superficie (m²)
- Estimation de hauteur (m)
- Calibration avec objets de référence
- Confidence score

---

## 🏗️ Architecture
```
waste-ai-backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── database.py          # Connexion BD Laravel
│   │
│   ├── api/v1/              # Endpoints API
│   │   ├── risk_analysis.py
│   │   ├── classification.py
│   │   └── size_estimation.py
│   │
│   ├── models/              # Modèles IA
│   │   ├── risk_predictor.py
│   │   ├── waste_classifier.py
│   │   └── size_estimator.py
│   │
│   ├── services/            # Business logic
│   │   ├── image_processor.py
│   │   ├── data_fetcher.py
│   │   └── result_saver.py
│   │
│   ├── schemas/             # Pydantic models
│   │   ├── requests.py
│   │   └── responses.py
│   │
│   └── utils/               # Utilitaires
│       ├── image_utils.py
│       ├── geo_utils.py
│       └── validators.py
│
├── trained_models/          # Modèles entraînés
├── datasets/                # Datasets
├── scripts/                 # Scripts d'entraînement
├── tests/                   # Tests unitaires
└── docker/                  # Docker configs
```

---

## 🛠️ Technologies

### Backend & API
- **FastAPI** 0.104 - Framework API moderne
- **Uvicorn** - Serveur ASGI
- **Pydantic** - Validation de données
- **SQLAlchemy** - ORM pour PostgreSQL

### Machine Learning
- **TensorFlow** 2.15 - Deep Learning
- **scikit-learn** - ML classique
- **OpenCV** - Traitement d'images
- **Pillow** - Manipulation d'images

### RAG & LLM (Bonus)
- **ChromaDB** - Base vectorielle
- **sentence-transformers** - Embeddings
- **Ollama** - LLM local

### Base de Données
- **PostgreSQL** 15 - BD Laravel
- **ChromaDB** - Stockage vectoriel

### DevOps
- **Docker** & **Docker Compose**
- **pytest** - Tests
- **GitHub Actions** - CI/CD (à venir)

---

## 📦 Installation

### Prérequis

- Python 3.11+
- PostgreSQL 15+
- Git
- (Optionnel) Docker & Docker Compose

### 1. Cloner le repository
```bash
git clone https://github.com/votre-org/waste-ai-backend.git
cd waste-ai-backend
```

### 2. Créer environnement virtuel
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement
```bash
# Copier le fichier exemple
cp .env.example .env

# Éditer .env avec vos configurations
notepad .env  # Windows
nano .env     # Linux/Mac
```

### 5. Initialiser la base de données
```bash
# Assurez-vous que PostgreSQL est lancé
# La BD Laravel doit déjà exister

# Tester la connexion
python -c "from app.database import engine; engine.connect(); print('✅ BD connectée')"
```

### 6. (Optionnel) Télécharger les modèles pré-entraînés
```bash
# Créer dossier modèles
mkdir -p trained_models

# Télécharger depuis Google Drive / S3
# TODO: Ajouter liens de téléchargement
```

---

## ⚙️ Configuration

### Fichier `.env`
```env
# API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true

# Base de Données Laravel
DB_HOST=localhost
DB_PORT=5432
DB_NAME=waste_management
DB_USER=postgres
DB_PASSWORD=your_password

# Modèles IA
CLASSIFICATION_MODEL_PATH=./trained_models/waste_classifier_v1.h5
RISK_MODEL_PATH=./trained_models/risk_predictor.pkl

# RAG (Optionnel)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VECTOR_DB_PATH=./data/chroma_db
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
```

---

## 🚀 Utilisation

### Démarrer le serveur
```bash
# Mode développement (auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Mode production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Accéder à la documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📡 API Endpoints

### 🗺️ Analyse de Risque

#### `GET /api/v1/risk/analyze`

Analyser les zones à risque.

**Paramètres:**
- `city` (optionnel): Ville à analyser
- `days`: Nombre de jours (1-365, défaut: 30)
- `min_reports`: Min signalements par zone (défaut: 3)

**Réponse:**
```json
{
  "zones": [
    {
      "id": "zone_001",
      "latitude": 12.3714,
      "longitude": -1.5197,
      "risk_level": "high",
      "risk_score": 78.5,
      "report_count": 15,
      "recommendations": ["Intervention urgente"]
    }
  ],
  "heatmap_data": [...],
  "statistics": {...},
  "processing_time": 2.34
}
```

#### `GET /api/v1/risk/priority-zones`

Obtenir les zones prioritaires.

#### `GET /api/v1/risk/statistics`

Statistiques globales.

---

### 🖼️ Classification

#### `POST /api/v1/classification/classify`

Classifier une image de déchet.

**Body (multipart/form-data):**
- `image`: Fichier image (JPEG/PNG)
- `report_id` (optionnel): ID du report
- `save_result` (optionnel): Sauvegarder en BD

**Réponse:**
```json
{
  "waste_type": "plastique",
  "confidence": 89.5,
  "probabilities": {
    "plastique": 0.895,
    "papier": 0.045,
    "aluminium": 0.035
  },
  "is_valid": true,
  "processing_time": 0.45
}
```

#### `GET /api/v1/classification/classes`

Liste des classes supportées.

#### `POST /api/v1/classification/batch-classify`

Classification en batch.

---

### 📏 Estimation de Taille

#### `POST /api/v1/size/estimate`

Estimer la taille d'un tas de déchets.

**Body (multipart/form-data):**
- `image`: Fichier image
- `report_id` (optionnel)
- `reference_height` (optionnel): Hauteur de référence (m)

**Réponse:**
```json
{
  "volume": 2.5,
  "area": 5.0,
  "height": 0.5,
  "confidence": 75.0,
  "unit": "metric",
  "methodology": "depth_estimation",
  "processing_time": 0.67
}
```

---

## 🤖 Modèles IA

### 1. Classificateur de Déchets

- **Architecture**: MobileNetV2 + Transfer Learning
- **Classes**: 9 types de déchets
- **Input**: Images 224x224 RGB
- **Accuracy**: ~85% (validation)
- **Fichier**: `waste_classifier_v1.h5`

**Entraînement:**
```bash
python scripts/train_classifier.py
```

### 2. Prédicteur de Risque

- **Algorithme**: Gradient Boosting Classifier
- **Features**: 7 features spatiotemporelles
- **Classes**: 4 niveaux de risque
- **Accuracy**: ~78%
- **Fichier**: `risk_predictor.pkl`

**Entraînement:**
```bash
python scripts/train_risk_model.py
```

### 3. Estimateur de Taille

- **Méthode**: Computer Vision heuristique
- **Mode**: MOCK (proof of concept)
- **Amélioration**: Depth estimation avec CNN

---

## ✅ Tests

### Lancer tous les tests
```bash
pytest tests/ -v
```

### Tests spécifiques
```bash
# Tests API
pytest tests/test_api.py -v

# Tests modèles
pytest tests/test_models.py -v

# Tests services
pytest tests/test_services.py -v
```

### Coverage
```bash
pytest --cov=app tests/
```

---

## 🐳 Déploiement Docker

### Build & Run
```bash
# Build
docker-compose -f docker/docker-compose.yml build

# Start
docker-compose -f docker/docker-compose.yml up -d

# Logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop
docker-compose -f docker/docker-compose.yml down
```

### Services

- **Backend API**: http://localhost:8000
- **PostgreSQL**: localhost:5432
- **pgAdmin**: http://localhost:5050

---

## 👥 Contribution

### Workflow

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Standards

- Code style: **Black**
- Docstrings: **Google Style**
- Tests: **pytest**
- Type hints: **mypy**

---

## 📄 Licence

Ce projet est sous licence **MIT**. Voir [LICENSE](LICENSE) pour plus de détails.

---

## 👨‍💻 Auteurs

- **Waste Management AI Team**
- Contact: [email@example.com](mailto:email@example.com)

---

## 🙏 Remerciements

- Laravel Backend Team
- Anthropic (Claude)
- HuggingFace
- TensorFlow Community

---

## 📚 Documentation Additionnelle

- [Guide d'Entraînement](docs/TRAINING.md)
- [Guide API](docs/API.md)
- [Architecture Détaillée](docs/ARCHITECTURE.md)
- [FAQ](docs/FAQ.md)

---

**⭐ N'oubliez pas de mettre une étoile si ce projet vous aide !**