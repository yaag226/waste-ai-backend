"""
Utilitaires géospatiaux
"""
import math
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculer distance entre deux points GPS (formule Haversine)
    
    Args:
        lat1, lon1: Point 1
        lat2, lon2: Point 2
    
    Returns:
        Distance en kilomètres
    """
    # Rayon de la Terre en km
    R = 6371.0
    
    # Convertir en radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Différences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Formule Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    
    return distance


def calculate_bounding_box(
    coordinates: List[Tuple[float, float]],
    padding: float = 0.01
) -> Tuple[float, float, float, float]:
    """
    Calculer bounding box pour une liste de coordonnées
    
    Args:
        coordinates: Liste de (lat, lon)
        padding: Marge en degrés
    
    Returns:
        (min_lat, min_lon, max_lat, max_lon)
    """
    if not coordinates:
        return (0, 0, 0, 0)
    
    lats = [coord[0] for coord in coordinates]
    lons = [coord[1] for coord in coordinates]
    
    min_lat = min(lats) - padding
    max_lat = max(lats) + padding
    min_lon = min(lons) - padding
    max_lon = max(lons) + padding
    
    return (min_lat, min_lon, max_lat, max_lon)


def is_point_in_radius(
    center_lat: float,
    center_lon: float,
    point_lat: float,
    point_lon: float,
    radius_km: float
) -> bool:
    """
    Vérifier si un point est dans un rayon donné
    
    Args:
        center_lat, center_lon: Centre
        point_lat, point_lon: Point à tester
        radius_km: Rayon en km
    
    Returns:
        True si dans le rayon
    """
    distance = haversine_distance(center_lat, center_lon, point_lat, point_lon)
    return distance <= radius_km


def calculate_cluster_center(coordinates: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculer le centre (centroïde) d'un cluster de points
    
    Args:
        coordinates: Liste de (lat, lon)
    
    Returns:
        (center_lat, center_lon)
    """
    if not coordinates:
        return (0, 0)
    
    avg_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
    avg_lon = sum(coord[1] for coord in coordinates) / len(coordinates)
    
    return (avg_lat, avg_lon)


def bearing_between_points(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculer l'azimut (bearing) entre deux points
    
    Args:
        lat1, lon1: Point de départ
        lat2, lon2: Point d'arrivée
    
    Returns:
        Azimut en degrés (0-360)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    
    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normaliser 0-360
    bearing_deg = (bearing_deg + 360) % 360
    
    return bearing_deg


def destination_point(
    lat: float,
    lon: float,
    bearing: float,
    distance_km: float
) -> Tuple[float, float]:
    """
    Calculer point de destination depuis un point, azimut et distance
    
    Args:
        lat, lon: Point de départ
        bearing: Azimut en degrés
        distance_km: Distance en km
    
    Returns:
        (dest_lat, dest_lon)
    """
    R = 6371.0  # Rayon Terre en km
    
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)
    
    dest_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance_km / R) +
        math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad)
    )
    
    dest_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
        math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(dest_lat_rad)
    )
    
    dest_lat = math.degrees(dest_lat_rad)
    dest_lon = math.degrees(dest_lon_rad)
    
    return (dest_lat, dest_lon)
