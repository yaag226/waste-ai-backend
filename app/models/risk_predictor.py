"""
Modèle de prédiction des zones à risque
"""
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

from app.schemas.responses import RiskZone, HeatmapPoint, RiskLevel

logger = logging.getLogger(__name__)


class RiskPredictor:
    """
    Prédicteur de zones à risque d'accumulation de déchets
    
    Utilise:
    - DBSCAN pour clustering spatial
    - Analyse temporelle pour récurrence
    - Scoring basé sur multiples facteurs
    """
    
    def __init__(self):
        self.eps = 0.01  # ~1km de rayon pour clustering
        self.min_samples = 3
        logger.info("RiskPredictor initialisé")
    
    def predict_risk_zones(
        self, 
        reports_data: List[Dict[str, Any]], 
        min_reports: int = 3
    ) -> List[RiskZone]:
        """
        Identifier les zones à risque depuis les données de signalements
        
        Args:
            reports_data: Liste de signalements avec coordonnées
            min_reports: Minimum de signalements pour considérer une zone
        
        Returns:
            Liste de RiskZone
        """
        if not reports_data or len(reports_data) < min_reports:
            return []
        
        try:
            # Extraire coordonnées
            coordinates = np.array([
                [r['latitude'], r['longitude']] 
                for r in reports_data
            ])
            
            # Clustering DBSCAN
            clustering = DBSCAN(
                eps=self.eps,
                min_samples=min_reports,
                metric='haversine'
            )
            
            # Convertir en radians pour haversine
            coords_rad = np.radians(coordinates)
            labels = clustering.fit_predict(coords_rad)
            
            # Identifier clusters (zones)
            zones = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                if label == -1:  # Bruit, pas un cluster
                    continue
                
                # Points dans ce cluster
                cluster_mask = labels == label
                cluster_reports = [r for i, r in enumerate(reports_data) if cluster_mask[i]]
                
                if len(cluster_reports) < min_reports:
                    continue
                
                # Calculer centre du cluster
                cluster_coords = coordinates[cluster_mask]
                center_lat = float(np.mean(cluster_coords[:, 0]))
                center_lon = float(np.mean(cluster_coords[:, 1]))
                
                # Calculer score de risque
                risk_score = self._calculate_risk_score(cluster_reports)
                risk_level = self._determine_risk_level(risk_score)
                
                # Identifier facteurs de risque
                risk_factors = self._identify_risk_factors(cluster_reports)
                
                # Créer zone
                zone = RiskZone(
                    id=len(zones) + 1,
                    latitude=center_lat,
                    longitude=center_lon,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    report_count=len(cluster_reports),
                    address=cluster_reports[0].get('address', 'Zone détectée'),
                    city=cluster_reports[0]['city'],
                    last_report_date=max(r['created_at'] for r in cluster_reports),
                    risk_factors=risk_factors,
                    recommendations=self._generate_recommendations(risk_level, risk_factors)
                )
                
                zones.append(zone)
            
            # Trier par score de risque décroissant
            zones.sort(key=lambda z: z.risk_score, reverse=True)
            
            logger.info(f"Détecté {len(zones)} zones à risque")
            return zones
            
        except Exception as e:
            logger.error(f"Erreur prédiction zones: {e}", exc_info=True)
            return []
    
    def _calculate_risk_score(self, reports: List[Dict[str, Any]]) -> float:
        """
        Calculer le score de risque d'une zone (0-100)
        
        Facteurs:
        - Nombre de signalements
        - Récurrence temporelle
        - Densité
        - Statut des signalements
        """
        # Nombre de signalements (max 40 points)
        count_score = min(len(reports) / 20 * 40, 40)
        
        # Récurrence temporelle (max 30 points)
        dates = [datetime.fromisoformat(r['created_at'].replace('Z', '+00:00')) for r in reports]
        if len(dates) > 1:
            dates_sorted = sorted(dates)
            time_diffs = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
            avg_interval = np.mean(time_diffs) if time_diffs else 30
            recurrence_score = max(0, 30 - (avg_interval / 2))  # Plus court = plus risqué
        else:
            recurrence_score = 10
        
        # Signalements récents (max 20 points)
        now = datetime.now()
        recent_reports = [r for r in reports if (now - datetime.fromisoformat(r['created_at'].replace('Z', '+00:00'))).days <= 7]
        recent_score = min(len(recent_reports) / len(reports) * 20, 20)
        
        # Statut non traité (max 10 points)
        pending_reports = [r for r in reports if r.get('status') == 'pending']
        pending_score = min(len(pending_reports) / len(reports) * 10, 10)
        
        total_score = count_score + recurrence_score + recent_score + pending_score
        
        return min(round(total_score, 1), 100)
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Déterminer le niveau de risque depuis le score"""
        if score >= 80:
            return RiskLevel.CRITICAL
        elif score >= 60:
            return RiskLevel.HIGH
        elif score >= 40:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _identify_risk_factors(self, reports: List[Dict[str, Any]]) -> List[str]:
        """Identifier les facteurs de risque d'une zone"""
        factors = []
        
        # Accumulation récurrente
        if len(reports) >= 10:
            factors.append("Accumulation récurrente")
        
        # Signalements récents multiples
        now = datetime.now()
        recent = [r for r in reports if (now - datetime.fromisoformat(r['created_at'].replace('Z', '+00:00'))).days <= 7]
        if len(recent) >= 3:
            factors.append("Signalements récents multiples")
        
        # Beaucoup de signalements en attente
        pending = [r for r in reports if r.get('status') == 'pending']
        if len(pending) >= len(reports) * 0.7:
            factors.append("Absence de collecte régulière")
        
        # Police reports
        police = [r for r in reports if r.get('is_police_report')]
        if police:
            factors.append("Signalements Police LAABAL")
        
        return factors if factors else ["Zone nécessitant surveillance"]
    
    def _generate_recommendations(self, risk_level: RiskLevel, factors: List[str]) -> str:
        """Générer recommandations basées sur le niveau de risque"""
        recommendations = {
            RiskLevel.CRITICAL: "⚠️ INTERVENTION URGENTE REQUISE dans les 24-48h. Mobiliser équipe de collecte prioritaire.",
            RiskLevel.HIGH: "Intervention nécessaire sous 72h. Planifier collecte avec équipement adapté.",
            RiskLevel.MEDIUM: "Surveillance rapprochée recommandée. Planifier collecte dans la semaine.",
            RiskLevel.LOW: "Surveillance normale. Inclure dans tournée de collecte régulière."
        }
        
        return recommendations.get(risk_level, "Surveillance recommandée")
    
    def generate_heatmap_data(self, reports_data: List[Dict[str, Any]]) -> List[HeatmapPoint]:
        """
        Générer données pour heatmap
        
        Args:
            reports_data: Signalements
        
        Returns:
            Liste de HeatmapPoint avec intensité
        """
        if not reports_data:
            return []
        
        try:
            heatmap_points = []
            
            # Grouper par proximité pour calculer densité
            coordinates = {}
            for report in reports_data:
                # Arrondir à 3 décimales (~100m de précision)
                lat = round(report['latitude'], 3)
                lon = round(report['longitude'], 3)
                key = (lat, lon)
                
                if key not in coordinates:
                    coordinates[key] = []
                coordinates[key].append(report)
            
            # Créer points avec intensité
            max_count = max(len(reports) for reports in coordinates.values())
            
            for (lat, lon), reports in coordinates.items():
                intensity = len(reports) / max_count
                
                heatmap_points.append(HeatmapPoint(
                    latitude=lat,
                    longitude=lon,
                    intensity=intensity
                ))
            
            return heatmap_points
            
        except Exception as e:
            logger.error(f"Erreur génération heatmap: {e}")
            return []
    
    def identify_priority_zones(self, zones: List[RiskZone], max_zones: int = 5) -> List[RiskZone]:
        """
        Identifier les zones prioritaires nécessitant intervention
        
        Args:
            zones: Toutes les zones
            max_zones: Nombre maximum de zones prioritaires
        
        Returns:
            Zones triées par priorité
        """
        # Filtrer zones critiques et élevées
        priority_zones = [
            z for z in zones 
            if z.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        ]
        
        # Trier par score décroissant
        priority_zones.sort(key=lambda z: z.risk_score, reverse=True)
        
        return priority_zones[:max_zones]