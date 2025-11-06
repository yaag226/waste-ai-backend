"""
Endpoints pour l'estimation de taille des dépotoirs
"""
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import Optional, List
import time
import logging
from io import BytesIO
from PIL import Image

from app.database import get_db, Report
from app.schemas.requests import EstimateBatchRequest
from app.schemas.responses import SizeEstimation
from app.models.size_estimator import SizeEstimator
from app.services.image_processor import ImageProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Instances
size_estimator = SizeEstimator()
image_processor = ImageProcessor()


@router.post("/estimate", response_model=SizeEstimation)
async def estimate_size(
    image: UploadFile = File(..., description="Image du dépotoir"),
    location: Optional[str] = Form(None, description="Localisation"),
    notes: Optional[str] = Form(None, description="Notes additionnelles")
):
    """
    Estimer la taille d'un dépotoir depuis une image
    
    Args:
        image: Image du dépotoir
        location: Localisation (optionnel)
        notes: Notes (optionnel)
    
    Returns:
        SizeEstimation avec volume, superficie, hauteur
    """
    start_time = time.time()
    
    try:
        # Lire l'image
        contents = await image.read()
        img = Image.open(BytesIO(contents))
        
        # Prétraiter
        processed_img = image_processor.preprocess_image(img)
        
        # Estimer la taille
        estimation = size_estimator.estimate(processed_img)
        
        processing_time = time.time() - start_time
        estimation['processing_time'] = processing_time
        
        logger.info(
            f"Estimation: volume={estimation['volume']}m³, "
            f"area={estimation['area']}m², "
            f"confidence={estimation['confidence']}% - {processing_time:.3f}s"
        )
        
        # TODO: Sauvegarder dans une table d'analyses
        
        return SizeEstimation(**estimation)
        
    except Exception as e:
        logger.error(f"Erreur estimation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/estimate-batch")
async def estimate_batch(
    request: EstimateBatchRequest,
    db: Session = Depends(get_db)
):
    """
    Estimer la taille pour plusieurs signalements
    
    Args:
        request: Liste des IDs de reports
    
    Returns:
        Liste d'estimations
    """
    start_time = time.time()
    
    try:
        # Récupérer les reports
        reports = db.query(Report).filter(
            Report.id.in_(request.report_ids),
            Report.image_path.isnot(None)
        ).all()
        
        if not reports:
            raise HTTPException(status_code=404, detail="Aucun report trouvé")
        
        analyses = []
        
        for report in reports:
            try:
                # Charger image
                img = image_processor.load_image_from_path(report.image_path)
                
                if img is None:
                    analyses.append({
                        "report_id": report.id,
                        "status": "failed",
                        "error": "Image non trouvée"
                    })
                    continue
                
                # Prétraiter
                processed_img = image_processor.preprocess_image(img)
                
                # Estimer
                estimation = size_estimator.estimate(processed_img)
                
                analyses.append({
                    "report_id": report.id,
                    "status": "success",
                    "image_url": f"/storage/{report.image_path}",
                    "title": report.title,
                    **estimation
                })
                
            except Exception as e:
                logger.error(f"Erreur estimation report {report.id}: {e}")
                analyses.append({
                    "report_id": report.id,
                    "status": "failed",
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        return {
            "total": len(reports),
            "analyses": analyses,
            "processing_time": processing_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur batch estimation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_analysis_history(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Obtenir l'historique des analyses de taille
    
    Args:
        limit: Nombre maximum de résultats
    
    Returns:
        Liste des analyses récentes
    """
    try:
        # TODO: Implémenter récupération depuis table d'analyses
        
        # Mock data
        history = [
            {
                "id": 1,
                "title": "Dépotoir Zone Industrielle",
                "thumbnail": "/images/placeholder.jpg",
                "location": "Zone Industrielle, Ouagadougou",
                "volume": 25.5,
                "area": 48.2,
                "analyzed_at": "2025-10-28T14:30:00"
            },
            {
                "id": 2,
                "title": "Accumulation Marché",
                "thumbnail": "/images/placeholder.jpg",
                "location": "Marché Central",
                "volume": 15.8,
                "area": 32.5,
                "analyzed_at": "2025-10-27T09:15:00"
            }
        ]
        
        return history[:limit]
        
    except Exception as e:
        logger.error(f"Erreur historique: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    