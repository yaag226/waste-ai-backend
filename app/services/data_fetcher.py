"""
Service pour récupérer des données depuis la BD Laravel
"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from app.database import Report, User

logger = logging.getLogger(__name__)


class DataFetcher:
    """Service pour récupérer des données de la base de données"""
    
    async def fetch_reports_for_risk_analysis(
        self,
        db: Session,
        city: Optional[str] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Récupérer les signalements pour l'analyse de risque
        
        Args:
            db: Session database
            city: Ville à filtrer (optionnel)
            days: Nombre de jours d'historique
        
        Returns:
            Liste de reports transformés en dict
        """
        try:
            # Date de début
            start_date = datetime.now() - timedelta(days=days)
            
            # Query de base
            query = db.query(Report).filter(
                Report.created_at >= start_date,
                Report.latitude.isnot(None),
                Report.longitude.isnot(None)
            )
            
            # Filtre ville
            if city:
                query = query.filter(Report.city == city)
            
            # Exécuter
            reports = query.all()
            
            # Transformer en dict
            results = []
            for report in reports:
                results.append({
                    "id": report.id,
                    "latitude": float(report.latitude),
                    "longitude": float(report.longitude),
                    "city": report.city,
                    "address": report.address,
                    "status": report.status,
                    "is_police_report": report.is_police_report,
                    "created_at": report.created_at.isoformat() if report.created_at else None
                })
            
            logger.info(f"Récupéré {len(results)} reports pour analyse (city={city}, days={days})")
            return results
            
        except Exception as e:
            logger.error(f"Erreur fetch reports: {e}", exc_info=True)
            return []
    
    async def fetch_report_with_image(
        self,
        db: Session,
        report_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Récupérer un report avec son image
        
        Args:
            db: Session
            report_id: ID du report
        
        Returns:
            Dict avec infos report ou None
        """
        try:
            report = db.query(Report).filter(Report.id == report_id).first()
            
            if not report:
                return None
            
            return {
                "id": report.id,
                "title": report.title,
                "image_path": report.image_path,
                "latitude": float(report.latitude) if report.latitude else None,
                "longitude": float(report.longitude) if report.longitude else None,
                "city": report.city,
                "created_at": report.created_at.isoformat() if report.created_at else None
            }
            
        except Exception as e:
            logger.error(f"Erreur fetch report {report_id}: {e}")
            return None
    
    async def fetch_classification_stats(self, db: Session) -> Dict[str, Any]:
        """
        Récupérer statistiques de classification
        
        Args:
            db: Session
        
        Returns:
            Dict avec statistiques
        """
        try:
            # TODO: Implémenter avec vraies queries
            # Pour l'instant, retourner mock data
            
            stats = {
                "total": db.query(Report).count(),
                "classified": db.query(Report).filter(Report.waste_type_id.isnot(None)).count(),
                "pending": db.query(Report).filter(Report.waste_type_id.is_(None)).count(),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur stats classification: {e}")
            return {}
        