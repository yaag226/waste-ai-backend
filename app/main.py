"""
Application FastAPI principale
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from app.config import settings
from app.database import test_connection
from app.api import risk_analysis, classification, size_estimation

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="Waste AI Backend",
    description="Service d'Intelligence Artificielle pour la gestion des déchets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware pour logger les requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware pour logger toutes les requêtes"""
    start_time = time.time()
    
    # Traiter la requête
    response = await call_next(request)
    
    # Calculer le temps de traitement
    process_time = time.time() - start_time
    
    # Logger
    logger.info(
        f"{request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Time: {process_time:.3f}s"
    )
    
    # Ajouter header de temps de traitement
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Event handlers
@app.on_event("startup")
async def startup_event():
    """Actions au démarrage de l'application"""
    logger.info("🚀 Démarrage de Waste AI Backend")
    logger.info(f"📍 Version: {app.version}")
    logger.info(f"🔧 Mode Debug: {settings.API_DEBUG}")
    logger.info(f"🌐 CORS Origins: {settings.ALLOWED_ORIGINS}")
    
    # Tester la connexion DB
    if test_connection():
        logger.info("✅ Application prête")
    else:
        logger.warning("⚠️ Application démarrée mais DB non connectée")


@app.on_event("shutdown")
async def shutdown_event():
    """Actions à l'arrêt de l'application"""
    logger.info("🛑 Arrêt de Waste AI Backend")


# Routes principales
@app.get("/")
async def root():
    """Route racine"""
    return {
        "service": "Waste AI Backend",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "risk_analysis": "/api/v1/risk",
            "classification": "/api/v1/classification",
            "size_estimation": "/api/v1/size"
        }
    }


@app.get("/health")
async def health_check():
    """Vérification de santé du service"""
    db_status = test_connection()
    
    return {
        "status": "healthy" if db_status else "degraded",
        "database": "connected" if db_status else "disconnected",
        "version": "1.0.0"
    }


@app.get("/api/v1")
async def api_info():
    """Information sur l'API v1"""
    return {
        "version": "1.0.0",
        "endpoints": {
            "risk_analysis": {
                "analyze": "GET /api/v1/risk/analyze",
                "description": "Analyse prédictive des zones à risque"
            },
            "classification": {
                "classify": "POST /api/v1/classification/classify",
                "batch": "POST /api/v1/classification/classify-batch",
                "stats": "GET /api/v1/classification/statistics",
                "description": "Classification des types de déchets"
            },
            "size_estimation": {
                "estimate": "POST /api/v1/size/estimate",
                "batch": "POST /api/v1/size/estimate-batch",
                "description": "Estimation de taille des dépotoirs"
            }
        }
    }


# Inclusion des routers
app.include_router(
    risk_analysis.router,
    prefix="/api/v1/risk",
    tags=["Risk Analysis"]
)

app.include_router(
    classification.router,
    prefix="/api/v1/classification",
    tags=["Waste Classification"]
)

app.include_router(
    size_estimation.router,
    prefix="/api/v1/size",
    tags=["Size Estimation"]
)


# Handler d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler pour toutes les exceptions non gérées"""
    logger.error(f"Erreur non gérée: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.API_DEBUG else "Une erreur est survenue",
            "path": str(request.url)
        }
    )


# Point d'entrée pour uvicorn
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level="info"
    )