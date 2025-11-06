"""
Schémas de requêtes (inputs)
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class ClassifyImageRequest(BaseModel):
    """Requête pour classifier une image"""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    report_id: Optional[int] = None
    
    @validator('image_url', 'image_base64', 'report_id')
    def at_least_one_field(cls, v, values):
        """Au moins un champ doit être fourni"""
        if not any([values.get('image_url'), values.get('image_base64'), values.get('report_id')]):
            raise ValueError('Au moins un champ (image_url, image_base64, ou report_id) doit être fourni')
        return v


class ClassifyBatchRequest(BaseModel):
    """Requête pour classification en batch"""
    report_ids: List[int] = Field(..., min_items=1, max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "report_ids": [1, 2, 3, 4, 5]
            }
        }


class RiskAnalysisRequest(BaseModel):
    """Requête pour analyse de risque"""
    city: Optional[str] = None
    days: int = Field(default=30, ge=7, le=365)
    min_reports: int = Field(default=3, ge=1)
    
    class Config:
        schema_extra = {
            "example": {
                "city": "Ouagadougou",
                "days": 30,
                "min_reports": 3
            }
        }


class SizeEstimationRequest(BaseModel):
    """Requête pour estimation de taille"""
    report_ids: Optional[List[int]] = None
    image_url: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None


class EstimateBatchRequest(BaseModel):
    """Requête pour estimation en batch"""
    report_ids: List[int] = Field(..., min_items=1, max_items=50)