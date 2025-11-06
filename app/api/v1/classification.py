"""
Endpoints pour la classification des déchets
"""
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile
from sqlalchemy.orm import Session
from typing import List
import time
import logging
from io import BytesIO
from PIL import Image

from app.database import get_db, Report
from app.schemas.requests import ClassifyBatchRequest
from app.schemas.responses import ClassificationResult, BatchClassificationResult
from app.models.waste_classifier import WasteClassifier
from app.services.image_processor import ImageProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Instances
classifier = WasteClassifier()
image_processor = ImageProcessor()


@router.post("/classify", response_model=ClassificationResult)
async def classify_waste(
    image: UploadFile = File(..., description="Image à classifier")
):
    """
    Classifier une image de déchet
    
    Args:
        image: Fichier image (JPG, PNG)
    
    Returns:
        ClassificationResult avec type, confiance et probabilités
    """
    start_time = time.time()
    
    try:
        # Valider l'image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        # Lire l'image
        contents = await image.read()
        img = Image.open(BytesIO(contents))
        
        # Prétraiter l'image
        processed_img = image_processor.preprocess_image(img)
        
        # Classifier
        result = classifier.predict(processed_img)
        
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        logger.info(f"Classification: {result['waste_type']} ({result['confidence']:.2f}%) - {processing_time:.3f}s")
        
        return ClassificationResult(**result)
        
    except Exception as e:
        logger.error(f"Erreur classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify-batch", response_model=BatchClassificationResult)
async def classify_batch(
    request: ClassifyBatchRequest,
    db: Session = Depends(get_db)
):
    """
    Classifier plusieurs images depuis la base de données
    
    Args:
        request: Liste des IDs de reports à classifier
    
    Returns:
        BatchClassificationResult avec résultats pour chaque image
    """
    start_time = time.time()
    
    try:
        # Récupérer les reports
        reports = db.query(Report).filter(
            Report.id.in_(request.report_ids),
            Report.image_path.isnot(None)
        ).all()
        
        if not reports:
            raise HTTPException(status_code=404, detail="Aucun report trouvé avec images")
        
        results = []
        classified_count = 0
        failed_count = 0
        
        for report in reports:
            try:
                # Charger l'image depuis le chemin
                img = image_processor.load_image_from_path(report.image_path)
                
                if img is None:
                    failed_count += 1
                    results.append({
                        "report_id": report.id,
                        "status": "failed",
                        "error": "Image non trouvée ou illisible"
                    })
                    continue
                
                # Prétraiter
                processed_img = image_processor.preprocess_image(img)
                
                # Classifier
                classification = classifier.predict(processed_img)
                
                # Sauvegarder dans la BD
                # TODO: Mettre à jour waste_type_id du report
                
                results.append({
                    "report_id": report.id,
                    "status": "success",
                    "classification": classification
                })
                classified_count += 1
                
            except Exception as e:
                logger.error(f"Erreur classification report {report.id}: {e}")
                failed_count += 1
                results.append({
                    "report_id": report.id,
                    "status": "failed",
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        logger.info(f"Batch classification: {classified_count}/{len(reports)} réussis - {processing_time:.2f}s")
        
        return BatchClassificationResult(
            total=len(reports),
            classified=classified_count,
            failed=failed_count,
            results=results,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur batch classification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_classification_statistics(db: Session = Depends(get_db)):
    """
    Obtenir les statistiques de classification
    
    Returns:
        Statistiques par type de déchet, évolution temporelle, etc.
    """
    try:
        # TODO: Implémenter calcul des statistiques depuis la BD
        
        # Mock data pour l'instant
        stats = {
            "total_classified": 150,
            "by_type": {
                "plastique": 45,
                "papier": 30,
                "aluminium": 15,
                "medical": 8,
                "organique": 25,
                "autre": 27
            },
            "by_city": {
                "Ouagadougou": 95,
                "Bobo-Dioulasso": 55
            },
            "confidence_avg": 87.5,
            "last_updated": "2025-11-01T14:30:00"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Erreur statistiques: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-image")
async def validate_image(image: UploadFile = File(...)):
    """
    Valider si une image est lisible et contient des déchets
    
    Returns:
        bool: True si valide, False sinon avec raison
    """
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents))
        
        # Vérifier qualité image
        is_valid, reason = image_processor.validate_image_quality(img)
        
        return {
            "is_valid": is_valid,
            "reason": reason,
            "dimensions": img.size,
            "format": img.format
        }
        
    except Exception as e:
        return {
            "is_valid": False,
            "reason": f"Erreur lecture image: {str(e)}"
        }
    