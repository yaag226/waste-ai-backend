"""
Utilitaires pour traitement d'images
"""
import base64
from io import BytesIO
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def base64_to_image(base64_string: str) -> Optional[Image.Image]:
    """
    Convertir une chaîne base64 en image PIL
    
    Args:
        base64_string: Image encodée en base64
    
    Returns:
        Image PIL ou None
    """
    try:
        # Retirer préfixe data:image si présent
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Décoder
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        
        return image
        
    except Exception as e:
        logger.error(f"Erreur conversion base64->image: {e}")
        return None


def image_to_base64(image: Image.Image, format: str = 'JPEG') -> Optional[str]:
    """
    Convertir une image PIL en base64
    
    Args:
        image: Image PIL
        format: Format de sortie
    
    Returns:
        String base64 ou None
    """
    try:
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/{format.lower()};base64,{img_str}"
        
    except Exception as e:
        logger.error(f"Erreur conversion image->base64: {e}")
        return None


def calculate_image_hash(image: Image.Image) -> str:
    """
    Calculer hash d'une image (pour détecter duplicatas)
    
    Args:
        image: Image PIL
    
    Returns:
        Hash string
    """
    import hashlib
    
    # Convertir en bytes
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    img_bytes = buffered.getvalue()
    
    # Calculer hash SHA256
    hash_obj = hashlib.sha256(img_bytes)
    return hash_obj.hexdigest()


def resize_with_aspect_ratio(
    image: Image.Image,
    target_width: int,
    target_height: int
) -> Image.Image:
    """
    Redimensionner en conservant le ratio
    
    Args:
        image: Image source
        target_width: Largeur cible
        target_height: Hauteur cible
    
    Returns:
        Image redimensionnée
    """
    # Calculer ratio
    ratio = min(target_width / image.width, target_height / image.height)
    
    new_size = (int(image.width * ratio), int(image.height * ratio))
    
    return image.resize(new_size, Image.Resampling.LANCZOS)


def crop_center(image: Image.Image, crop_width: int, crop_height: int) -> Image.Image:
    """
    Rogner au centre
    
    Args:
        image: Image source
        crop_width: Largeur du crop
        crop_height: Hauteur du crop
    
    Returns:
        Image rognée
    """
    img_width, img_height = image.size
    
    left = (img_width - crop_width) // 2
    top = (img_height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    
    return image.crop((left, top, right, bottom))