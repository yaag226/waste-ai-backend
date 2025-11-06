"""
Endpoints pour l'analyse de risque
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
import time
import logging

from app.database import get_db
from app.schemas.requests import RiskAnalysisRequest
from app.schemas.responses import RiskAnalysisResponse
from app.models.risk_predictor import RiskPredictor
from app.services.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)
router = APIRouter()

# Instance du prédicteur de risque
risk_predictor = RiskPredictor()
data_fetcher = DataFetcher()


@router.get("/analyze", response_model=RiskAnalysisResponse)
async def analyze_risk(
    city: Optional[str] = Query(None, description="Ville à analyser"),
    days: int = Query(30, ge=7, le=365, description="Nombre de jours d'historique"),
    min_reports: int = Query(3, ge=1, description="Minimum de signalements pour une zone"),
    db: Session = Depends(get_db)
):
    """
    Analyser les zones à risque d'accumulation de déchets
    
    Cette analyse utilise:
    - L'historique des signalements
    - La densité spatiale (clustering)
    - La récurrence temporelle
    - Les patterns géographiques
    
    Returns:
        RiskAnalysisResponse avec zones, heatmap, priorités et statistiques
    """
    start_time = time.time()
    
    try:
        logger.info(f"Analyse de risque: city={city}, days={days}, min_reports={min_reports}")
        
        # Récupérer les données historiques
        reports_data = await data_fetcher.fetch_reports_for_risk_analysis(
            db=db,
            city=city,
            days=days
        )
        
        if not reports_data:
            return RiskAnalysisResponse(
                zones=[],
                heatmap_data=[],
                priority_zones=[],
                statistics={
                    "critical_zones": 0,
                    "warning_zones": 0,
                    "avg_risk_score": 0,
                    "analyzed_images": 0
                },
                processing_time=time.time() - start_time
            )
        
        # Analyser les risques
        risk_zones = risk_predictor.predict_risk_zones(reports_data, min_reports)
        heatmap_data = risk_predictor.generate_heatmap_data(reports_data)
        priority_zones = risk_predictor.identify_priority_zones(risk_zones)
        
        # Calculer statistiques
        statistics = {
            "critical_zones": len([z for z in risk_zones if z.risk_level == "critical"]),
            "warning_zones": len([z for z in risk_zones if z.risk_level in ["critical", "high"]]),
            "avg_risk_score": sum(z.risk_score for z in risk_zones) / len(risk_zones) if risk_zones else 0,
            "analyzed_images": len(reports_data)
        }
        
        processing_time = time.time() - start_time
        logger.info(f"Analyse terminée en {processing_time:.2f}s - {len(risk_zones)} zones détectées")
        
        return RiskAnalysisResponse(
            zones=risk_zones,
            heatmap_data=heatmap_data,
            priority_zones=priority_zones,
            statistics=statistics,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de risque: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/zones/{zone_id}")
async def get_zone_details(
    zone_id: int,
    db: Session = Depends(get_db)
):
    """
    Obtenir les détails d'une zone à risque spécifique
    """
    # TODO: Implémenter récupération zone par ID
    return {"message": "À implémenter"}


@router.post("/export")
async def export_risk_report():
    """
    Exporter un rapport PDF des zones à risque
    """
    # TODO: Implémenter export PDF
    return {"message": "Export à implémenter"}