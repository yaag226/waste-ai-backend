"""
Tests pour les endpoints API
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)


class TestRootEndpoints:
    """Tests des endpoints racine"""
    
    def test_read_root(self):
        """Test endpoint racine"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Waste AI Backend"
        assert "version" in data
    
    def test_health_check(self):
        """Test endpoint health"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
    
    def test_api_info(self):
        """Test endpoint API info"""
        response = client.get("/api/v1")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "endpoints" in data


class TestRiskAnalysisAPI:
    """Tests pour l'API d'analyse de risque"""
    
    def test_analyze_risk_default(self):
        """Test analyse de risque avec paramètres par défaut"""
        response = client.get("/api/v1/risk/analyze")
        assert response.status_code == 200
        data = response.json()
        assert "zones" in data
        assert "heatmap_data" in data
        assert "statistics" in data
        assert "processing_time" in data
    
    def test_analyze_risk_with_city(self):
        """Test analyse de risque pour une ville"""
        response = client.get("/api/v1/risk/analyze?city=Ouagadougou&days=30")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["zones"], list)
    
    def test_analyze_risk_invalid_params(self):
        """Test avec paramètres invalides"""
        response = client.get("/api/v1/risk/analyze?days=1000")
        assert response.status_code == 422  # Validation error


class TestClassificationAPI:
    """Tests pour l'API de classification"""
    
    def test_classify_no_image(self):
        """Test classification sans image"""
        response = client.post("/api/v1/classification/classify")
        assert response.status_code == 422  # Validation error
    
    def test_classify_with_mock_image(self):
        """Test classification avec image mock"""
        # Créer image test
        from PIL import Image
        from io import BytesIO
        
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
        response = client.post("/api/v1/classification/classify", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "waste_type" in data
        assert "confidence" in data
        assert "probabilities" in data
    
    def test_get_statistics(self):
        """Test récupération statistiques"""
        response = client.get("/api/v1/classification/statistics")
        assert response.status_code == 200
        data = response.json()
        assert "total_classified" in data or "by_type" in data


class TestSizeEstimationAPI:
    """Tests pour l'API d'estimation de taille"""
    
    def test_estimate_no_image(self):
        """Test estimation sans image"""
        response = client.post("/api/v1/size/estimate")
        assert response.status_code == 422
    
    def test_estimate_with_mock_image(self):
        """Test estimation avec image mock"""
        from PIL import Image
        from io import BytesIO
        
        img = Image.new('RGB', (640, 480), color='blue')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
        response = client.post("/api/v1/size/estimate", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "volume" in data
        assert "area" in data
        assert "height" in data
        assert "confidence" in data
    
    def test_get_history(self):
        """Test récupération historique"""
        response = client.get("/api/v1/size/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestErrorHandling:
    """Tests de gestion d'erreurs"""
    
    def test_not_found(self):
        """Test endpoint inexistant"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test méthode HTTP non autorisée"""
        response = client.post("/")
        assert response.status_code == 405


if __name__ == "__main__":
    pytest.main([__file__, "-v"])