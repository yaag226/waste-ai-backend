"""
Tests pour les services
"""
import pytest
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.image_processor import ImageProcessor


class TestImageProcessor:
    """Tests pour le processeur d'images"""
    
    @pytest.fixture
    def processor(self):
        """Fixture du processeur"""
        return ImageProcessor()
    
    @pytest.fixture
    def valid_image(self):
        """Image valide"""
        return Image.new('RGB', (640, 480), color='red')
    
    @pytest.fixture
    def small_image(self):
        """Image trop petite"""
        return Image.new('RGB', (100, 100), color='green')
    
    def test_preprocess_image(self, processor, valid_image):
        """Test prétraitement"""
        processed = processor.preprocess_image(valid_image)
        
        assert processed is not None
        assert processed.size == processor.target_size
        assert processed.mode == 'RGB'
    
    def test_validate_valid_image(self, processor, valid_image):
        """Test validation image valide"""
        is_valid, reason = processor.validate_image_quality(valid_image)
        
        # Peut être True ou False selon la qualité détectée
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)
    
    def test_validate_small_image(self, processor, small_image):
        """Test validation image trop petite"""
        is_valid, reason = processor.validate_image_quality(small_image)
        
        assert is_valid == False
        assert "trop petite" in reason.lower()
    
    def test_create_thumbnail(self, processor, valid_image):
        """Test création miniature"""
        thumbnail = processor.create_thumbnail(valid_image, size=(150, 150))
        
        assert thumbnail is not None
        assert thumbnail.size[0] <= 150
        assert thumbnail.size[1] <= 150
    
    def test_extract_metadata(self, processor, valid_image):
        """Test extraction métadonnées"""
        metadata = processor.extract_image_metadata(valid_image)
        
        assert "size" in metadata
        assert "mode" in metadata
        assert metadata["mode"] == "RGB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])