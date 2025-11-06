"""
Générer des données synthétiques pour le modèle de risque
Utile pour développement/tests avant d'avoir les vraies données
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_risk_data(n_zones: int = 500) -> pd.DataFrame:
    """
    Générer données synthétiques de risque
    
    Args:
        n_zones: Nombre de zones à générer
    
    Returns:
        DataFrame avec features et labels
    """
    logger.info(f"Génération de {n_zones} zones synthétiques...")
    
    # Coordonnées Burkina Faso
    # Ouagadougou: ~12.37° N, -1.52° W
    # Bobo-Dioulasso: ~11.18° N, -4.29° W
    
    cities = ['Ouagadougou', 'Bobo-Dioulasso', 'Koudougou', 'Ouahigouya']
    city_coords = {
        'Ouagadougou': (12.37, -1.52),
        'Bobo-Dioulasso': (11.18, -4.29),
        'Koudougou': (12.25, -2.36),
        'Ouahigouya': (13.58, -2.42)
    }
    
    data = []
    
    for i in range(n_zones):
        city = np.random.choice(cities, p=[0.5, 0.3, 0.1, 0.1])  # Ouaga plus représentée
        base_lat, base_lon = city_coords[city]
        
        # Coordonnées avec variation
        lat = base_lat + np.random.uniform(-0.1, 0.1)
        lon = base_lon + np.random.uniform(-0.1, 0.1)
        
        # Features aléatoires mais corrélées
        report_count = np.random.randint(1, 50)
        
        # Plus de signalements = temps plus court entre eux
        avg_time = max(1, 30 - (report_count * 0.5) + np.random.uniform(-5, 5))
        
        recent_7d = min(report_count, np.random.randint(0, 10))
        recent_30d = min(report_count, recent_7d + np.random.randint(0, 20))
        
        # Zones à risque ont plus de signalements non traités
        pending_ratio = np.random.beta(2, 5) if report_count < 15 else np.random.beta(5, 2)
        
        # Rares cas avec police
        police_ratio = np.random.beta(1, 20)
        
        # Densité corrélée au nombre de signalements
        spatial_density = report_count / np.random.uniform(0.5, 5.0)
        
        # Calculer risk_level basé sur les features
        risk_score = (
            report_count * 2 +
            recent_30d * 3 +
            pending_ratio * 20 +
            police_ratio * 50 +
            spatial_density * 5
        )
        
        if risk_score < 30:
            risk_level = 0  # Low
        elif risk_score < 60:
            risk_level = 1  # Medium
        elif risk_score < 90:
            risk_level = 2  # High
        else:
            risk_level = 3  # Critical
        
        data.append({
            'zone_id': f'zone_{i+1:04d}',
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'city': city,
            'report_count': report_count,
            'avg_time_between_reports': round(avg_time, 2),
            'recent_reports_7d': recent_7d,
            'recent_reports_30d': recent_30d,
            'pending_ratio': round(pending_ratio, 3),
            'police_reports_ratio': round(police_ratio, 3),
            'spatial_density': round(spatial_density, 3),
            'risk_level': risk_level
        })
    
    df = pd.DataFrame(data)
    
    # Afficher distribution
    logger.info("\n" + "="*60)
    logger.info("DISTRIBUTION DES NIVEAUX DE RISQUE")
    logger.info("="*60)
    for level, name in enumerate(['Low', 'Medium', 'High', 'Critical']):
        count = (df['risk_level'] == level).sum()
        percent = count / len(df) * 100
        logger.info(f"{name:10} : {count:4} zones ({percent:.1f}%)")
    logger.info("="*60)
    
    return df


def main():
    """Générer et sauvegarder les données"""
    
    # Créer dossier
    output_dir = Path('datasets/risk_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Générer données complètes
    logger.info("\n🔄 Génération dataset complet...")
    full_data = generate_synthetic_risk_data(n_zones=500)
    
    # Sauvegarder
    output_file = output_dir / 'risk_features.csv'
    full_data.to_csv(output_file, index=False)
    logger.info(f"✅ Dataset sauvegardé: {output_file}")
    logger.info(f"   Total: {len(full_data)} zones")
    
    # Générer split test (20%)
    test_size = int(len(full_data) * 0.2)
    test_data = full_data.sample(n=test_size, random_state=42)
    
    test_file = output_dir / 'test_data.csv'
    test_data.to_csv(test_file, index=False)
    logger.info(f"✅ Test set sauvegardé: {test_file}")
    logger.info(f"   Total: {len(test_data)} zones")
    
    # Statistiques
    logger.info("\n📊 STATISTIQUES DU DATASET")
    logger.info("="*60)
    logger.info(f"Total zones: {len(full_data)}")
    logger.info(f"Villes: {full_data['city'].nunique()}")
    logger.info(f"\nReport count:")
    logger.info(f"  Min: {full_data['report_count'].min()}")
    logger.info(f"  Max: {full_data['report_count'].max()}")
    logger.info(f"  Mean: {full_data['report_count'].mean():.2f}")
    logger.info(f"\nSpatial density:")
    logger.info(f"  Min: {full_data['spatial_density'].min():.2f}")
    logger.info(f"  Max: {full_data['spatial_density'].max():.2f}")
    logger.info(f"  Mean: {full_data['spatial_density'].mean():.2f}")
    logger.info("="*60)
    
    logger.info("\n✅ Génération terminée!")
    logger.info(f"\nFichiers créés:")
    logger.info(f"  1. {output_file}")
    logger.info(f"  2. {test_file}")


if __name__ == "__main__":
    main()