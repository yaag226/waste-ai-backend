"""
Schémas de réponses (outputs)
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class WasteType(str, Enum):
    """Types de déchets"""
    PLASTIQUE = "plastique"
    PAPIER = "papier"
    ALUMINIUM = "aluminium"
    MEDICAL = "medical"
    ORGANIQUE = "organique"
    VERRE = "verre"
    ELECTRONIQUE = "electronique"
    TEXTILE = "textile"
    AUTRE = "autre"


class RiskLevel(str, Enum):
    """Niveaux de risque"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClassificationResult(BaseModel):
    """Résultat de classification d'une image"""
    waste_type: WasteType
    confidence: float = Field(..., ge=0, le=100)
    probabilities: Dict[str, float]
    is_valid: bool
    processing_time: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "waste_type": "plastique",
                "confidence": 92.5,
                "probabilities": {
                    "plastique": 0.925,
                    "papier": 0.045,
                    "aluminium": 0.020,
                    "autre": 0.010
                },
                "is_valid": True,
                "processing_time": 0.234
            }
        }


class BatchClassificationResult(BaseModel):
    """Résultat de classification en batch"""
    total: int
    classified: int
    failed: int
    results: List[Dict[str, Any]]
    processing_time: float


class RiskZone(BaseModel):
    """Zone à risque identifiée"""
    id: int
    latitude: float
    longitude: float
    risk_level: RiskLevel
    risk_score: float = Field(..., ge=0, le=100)
    report_count: int
    address: Optional[str] = None
    city: str
    last_report_date: Optional[str] = None
    risk_factors: Optional[List[str]] = None
    recommendations: Optional[str] = None


class HeatmapPoint(BaseModel):
    """Point pour heatmap"""
    latitude: float
    longitude: float
    intensity: float = Field(..., ge=0, le=1)


class RiskAnalysisResponse(BaseModel):
    """Réponse complète d'analyse de risque"""
    zones: List[RiskZone]
    heatmap_data: List[HeatmapPoint]
    priority_zones: List[RiskZone]
    statistics: Dict[str, Any]
    processing_time: float
    
    class Config:
        schema_extra = {
            "example": {
                "zones": [],
                "heatmap_data": [],
                "priority_zones": [],
                "statistics": {
                    "critical_zones": 2,
                    "warning_zones": 5,
                    "avg_risk_score": 65.3,
                    "analyzed_images": 150
                },
                "processing_time": 1.234
            }
        }


class SizeEstimation(BaseModel):
    """Résultat d'estimation de taille"""
    volume: float = Field(..., description="Volume en m³")
    area: float = Field(..., description="Superficie en m²")
    height: float = Field(..., description="Hauteur moyenne en m")
    confidence: float = Field(..., ge=0, le=100)
    methodology: str
    recommendations: Optional[str] = None
    processing_time: Optional[float] = None


class ErrorResponse(BaseModel):
    """Réponse d'erreur standard"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    