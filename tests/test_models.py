"""
Tests pour les modèles IA
"""
import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.waste_classifier import WasteClassifier
from app.models.risk_predictor import RiskPredictor
from app.models.size_estimator import SizeEstimator


class TestWasteClassifier:
    """Tests pour le classificateur de déchets"""
    
    @pytest.fixture
    def classifier(self):
        """Fixture du classificateur"""
        return WasteClassifier()
    
    @pytest.fixture
    def test_image(self):
        """Fixture image de test"""
        return Image.new('RGB', (224, 224), color='red')
    
    def test_classifier_initialization(self, classifier):
        """Test initialisation"""
        assert classifier is not None
        assert len(classifier.classes) > 0
        assert classifier.input_size == (224, 224)
    
    def test_predict(self, classifier, test_image):
        """Test prédiction"""
        result = classifier.predict(test_image)
        
        assert "waste_type" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "is_valid" in result
        
        assert result["waste_type"] in classifier.classes
        assert 0 <= result["confidence"] <= 100
        assert isinstance(result["probabilities"], dict)
    
    def test_predict_invalid_image(self, classifier):
        """Test prédiction avec image invalide"""
        with pytest.raises(Exception):
            classifier.predict(None)
    
    def test_get_model_info(self, classifier):
        """Test info modèle"""
        info = classifier.get_model_info()
        
        assert "model_loaded" in info
        assert "classes" in info
        assert "input_size" in info


class TestRiskPredictor:
    """Tests pour le prédicteur de risque"""
    
    @pytest.fixture
    def predictor(self):
        """Fixture du prédicteur"""
        return RiskPredictor()
    
    @pytest.fixture
    def mock_reports(self):
        """Fixture de reports mock"""
        from datetime import datetime, timedelta
        
        reports = []
        base_date = datetime.now()
        
        for i in range(20):
            reports.append({
                "id": i + 1,
                "latitude": 12.37 + np.random.uniform(-0.01, 0.01),
                "longitude": -1.52 + np.random.uniform(-0.01, 0.01),
                "city": "Ouagadougou",
                "address": f"Zone {i // 5}",
                "status": "pending" if i % 3 == 0 else "processed",
                "is_police_report": i % 10 == 0,
                "created_at": (base_date - timedelta(days=i * 2)).isoformat()
            })
        
        return reports
    
    def test_predictor_initialization(self, predictor):
        """Test initialisation"""
        assert predictor is not None
        assert predictor.eps > 0
        assert predictor.min_samples > 0
    
    def test_predict_risk_zones(self, predictor, mock_reports):
        """Test prédiction zones à risque"""
        zones = predictor.predict_risk_zones(mock_reports, min_reports=3)
        
        assert isinstance(zones, list)
        
        if len(zones) > 0:
            zone = zones[0]
            assert hasattr(zone, 'latitude')
            assert hasattr(zone, 'longitude')
            assert hasattr(zone, 'risk_level')
            assert hasattr(zone, 'risk_score')
            assert 0 <= zone.risk_score <= 100
    
    def test_predict_empty_data(self, predictor):
        """Test avec données vides"""
        zones = predictor.predict_risk_zones([])
        assert zones == []
    
    def test_generate_heatmap_data(self, predictor, mock_reports):
        """Test génération heatmap"""
        heatmap = predictor.generate_heatmap_data(mock_reports)
        
        assert isinstance(heatmap, list)
        
        if len(heatmap) > 0:
            point = heatmap[0]
            assert hasattr(point, 'latitude')
            assert hasattr(point, 'longitude')
            assert hasattr(point, 'intensity')
            assert 0 <= point.intensity <= 1
    
    def test_identify_priority_zones(self, predictor, mock_reports):
        """Test identification zones prioritaires"""
        zones = predictor.predict_risk_zones(mock_reports)
        priority = predictor.identify_priority_zones(zones, max_zones=3)
        
        assert isinstance(priority, list)
        assert len(priority) <= 3


class TestSizeEstimator:
    """Tests pour l'estimateur de taille"""
    
    @pytest.fixture
    def estimator(self):
        """Fixture de l'estimateur"""
        return SizeEstimator()
    
    @pytest.fixture
    def test_image(self):
        """Fixture image de test"""
        return Image.new('RGB', (640, 480), color='blue')
    
    def test_estimator_initialization(self, estimator):
        """Test initialisation"""
        assert estimator is not None
        assert estimator.pixel_to_meter_ratio > 0
    
    def test_estimate(self, estimator, test_image):
        """Test estimation"""
        result = estimator.estimate(test_image)
        
        assert "volume" in result
        assert "area" in result
        assert "height" in result
        assert "confidence" in result
        assert "methodology" in result
        
        assert result["volume"] > 0
        assert result["area"] > 0
        assert result["height"] > 0
        assert 0 <= result["confidence"] <= 100
    
    def test_calibrate_scale(self, estimator):
        """Test calibration échelle"""
        initial_ratio = estimator.pixel_to_meter_ratio
        
        estimator.calibrate_scale(reference_object_height=2.0, pixel_height=100)
        
        assert estimator.pixel_to_meter_ratio != initial_ratio
        assert estimator.pixel_to_meter_ratio == 2.0 / 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])