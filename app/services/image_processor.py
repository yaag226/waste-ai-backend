"""
Service de traitement d'images
"""
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, Optional
import os
import logging
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Service pour prétraiter et valider les images
    """
    
    def __init__(self):
        self.target_size = (224, 224)
        self.max_file_size = settings.MAX_IMAGE_SIZE
        self.allowed_formats = ['JPEG', 'PNG', 'JPG']
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Prétraiter une image pour les modèles IA
        
        Args:
            image: Image PIL
        
        Returns:
            Image prétraitée
        """
        try:
            # Convertir en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionner
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Améliorer contraste
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Améliorer netteté
            image = image.filter(ImageFilter.SHARPEN)
            
            return image
            
        except Exception as e:
            logger.error(f"Erreur prétraitement image: {e}")
            raise
    
    def validate_image_quality(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Valider la qualité d'une image
        
        Args:
            image: Image PIL
        
        Returns:
            (is_valid, reason)
        """
        try:
            # Vérifier dimensions minimales
            width, height = image.size
            if width < 200 or height < 200:
                return False, "Image trop petite (min 200x200px)"
            
            # Vérifier format
            if image.format not in self.allowed_formats:
                return False, f"Format non supporté: {image.format}"
            
            # Vérifier si l'image est floue (variance de Laplacien)
            if self._is_blurry(image):
                return False, "Image floue ou illisible"
            
            # Vérifier luminosité
            if self._is_too_dark(image) or self._is_too_bright(image):
                return False, "Luminosité inadéquate"
            
            return True, "Image valide"
            
        except Exception as e:
            logger.error(f"Erreur validation image: {e}")
            return False, f"Erreur validation: {str(e)}"
    
    def _is_blurry(self, image: Image.Image, threshold: float = 100.0) -> bool:
        """
        Détecter si une image est floue avec variance de Laplacien
        
        Args:
            image: Image PIL
            threshold: Seuil de détection
        
        Returns:
            True si floue
        """
        try:
            # Convertir en grayscale
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            # Calculer variance de Laplacien
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            
            # Convolution simple
            from scipy import ndimage
            filtered = ndimage.convolve(gray_array, laplacian)
            variance = np.var(filtered)
            
            return variance < threshold
            
        except Exception as e:
            logger.warning(f"Erreur détection flou: {e}")
            return False
    
    def _is_too_dark(self, image: Image.Image, threshold: int = 30) -> bool:
        """Détecter si image trop sombre"""
        try:
            gray = image.convert('L')
            avg_brightness = np.mean(np.array(gray))
            return avg_brightness < threshold
        except:
            return False
    
    def _is_too_bright(self, image: Image.Image, threshold: int = 225) -> bool:
        """Détecter si image trop lumineuse"""
        try:
            gray = image.convert('L')
            avg_brightness = np.mean(np.array(gray))
            return avg_brightness > threshold
        except:
            return False
    
    def load_image_from_path(self, image_path: str) -> Optional[Image.Image]:
        """
        Charger une image depuis un chemin fichier
        
        Args:
            image_path: Chemin relatif ou absolu
        
        Returns:
            Image PIL ou None si erreur
        """
        try:
            # Construire chemin complet
            # Supposer que les images Laravel sont dans storage/app/public
            if not os.path.isabs(image_path):
                # Chemin relatif depuis le projet Laravel
                base_path = Path(__file__).parent.parent.parent.parent / "storage" / "app" / "public"
                full_path = base_path / image_path.lstrip('/')
            else:
                full_path = Path(image_path)
            
            if not full_path.exists():
                logger.warning(f"Image non trouvée: {full_path}")
                return None
            
            # Charger image
            image = Image.open(full_path)
            return image
            
        except Exception as e:
            logger.error(f"Erreur chargement image {image_path}: {e}")
            return None
    
    def save_processed_image(
        self,
        image: Image.Image,
        output_dir: Path,
        filename: str
    ) -> Optional[str]:
        """
        Sauvegarder une image traitée
        
        Args:
            image: Image PIL
            output_dir: Répertoire de sortie
            filename: Nom du fichier
        
        Returns:
            Chemin du fichier sauvegardé ou None
        """
        try:
            # Créer répertoire si n'existe pas
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Chemin complet
            output_path = output_dir / filename
            
            # Sauvegarder
            image.save(output_path, quality=95)
            
            logger.info(f"Image sauvegardée: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde image: {e}")
            return None
    
    def create_thumbnail(
        self,
        image: Image.Image,
        size: Tuple[int, int] = (150, 150)
    ) -> Image.Image:
        """
        Créer une miniature d'image
        
        Args:
            image: Image source
            size: Taille de la miniature
        
        Returns:
            Image miniature
        """
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail
    
    def extract_image_metadata(self, image: Image.Image) -> dict:
        """
        Extraire métadonnées d'une image
        
        Args:
            image: Image PIL
        
        Returns:
            Dict de métadonnées
        """
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height,
            "info": dict(image.info) if hasattr(image, 'info') else {}
        }
    