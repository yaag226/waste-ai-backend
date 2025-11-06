"""
Validateurs personnalisés
"""
from typing import Any, Optional
import re
import logging

logger = logging.getLogger(__name__)


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Valider des coordonnées GPS
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        True si valides
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def validate_image_extension(filename: str, allowed_extensions: list) -> bool:
    """
    Valider l'extension d'un fichier image
    
    Args:
        filename: Nom du fichier
        allowed_extensions: Extensions autorisées
    
    Returns:
        True si valide
    """
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in [e.lower() for e in allowed_extensions]


def validate_phone_number(phone: str, country_code: str = "BF") -> bool:
    """
    Valider un numéro de téléphone
    
    Args:
        phone: Numéro de téléphone
        country_code: Code pays (BF = Burkina Faso)
    
    Returns:
        True si valide
    """
    # Retirer espaces et tirets
    phone_clean = phone.replace(' ', '').replace('-', '')
    
    if country_code == "BF":
        # Burkina Faso: +226 XX XX XX XX ou 00226...
        pattern = r'^(\+226|00226|226)?[0-9]{8}$'
        return bool(re.match(pattern, phone_clean))
    
    # Pattern générique
    pattern = r'^\+?[0-9]{8,15}$'
    return bool(re.match(pattern, phone_clean))


def validate_email(email: str) -> bool:
    """
    Valider une adresse email
    
    Args:
        email: Adresse email
    
    Returns:
        True si valide
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def sanitize_filename(filename: str) -> str:
    """
    Nettoyer un nom de fichier
    
    Args:
        filename: Nom original
    
    Returns:
        Nom nettoyé
    """
    # Retirer caractères dangereux
    filename = re.sub(r'[^\w\s.-]', '', filename)
    
    # Remplacer espaces par underscores
    filename = filename.replace(' ', '_')
    
    return filename.lower()


def validate_confidence_score(score: float) -> bool:
    """
    Valider un score de confiance
    
    Args:
        score: Score (0-100 ou 0-1)
    
    Returns:
        True si valide
    """
    return 0 <= score <= 100 or 0 <= score <= 1