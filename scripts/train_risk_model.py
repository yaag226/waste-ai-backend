"""
Script d'entraînement du modèle de prédiction de risque
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskModelTrainer:
    """
    Entraîneur pour le modèle de prédiction de risque
    
    Features:
    - Nombre de signalements dans la zone
    - Fréquence temporelle
    - Densité spatiale
    - Statut des signalements
    - etc.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'report_count',
            'avg_time_between_reports',
            'recent_reports_7d',
            'recent_reports_30d',
            'pending_ratio',
            'police_reports_ratio',
            'spatial_density'
        ]
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Préparer les features depuis les données brutes
        
        Args:
            data: DataFrame avec colonnes zone_id, report_date, status, etc.
        
        Returns:
            DataFrame avec features
        """
        logger.info("Préparation des features...")
        
        # TODO: Implémenter extraction de features depuis données réelles
        # Pour l'instant, générer des données synthétiques
        
        features = pd.DataFrame({
            'report_count': np.random.randint(1, 50, 1000),
            'avg_time_between_reports': np.random.uniform(1, 30, 1000),
            'recent_reports_7d': np.random.randint(0, 10, 1000),
            'recent_reports_30d': np.random.randint(0, 30, 1000),
            'pending_ratio': np.random.uniform(0, 1, 1000),
            'police_reports_ratio': np.random.uniform(0, 0.3, 1000),
            'spatial_density': np.random.uniform(0, 1, 1000)
        })
        
        # Label: risk_level (0=low, 1=medium, 2=high, 3=critical)
        # Basé sur heuristique
        features['risk_level'] = 0
        features.loc[features['report_count'] > 10, 'risk_level'] = 1
        features.loc[features['report_count'] > 20, 'risk_level'] = 2
        features.loc[features['report_count'] > 35, 'risk_level'] = 3
        
        return features
    
    def train(self, X_train, y_train):
        """
        Entraîner le modèle
        """
        logger.info("Entraînement du modèle de risque...")
        
        # Utiliser Gradient Boosting
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Normaliser features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Entraîner
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        logger.info(f"Cross-validation scores: {scores}")
        logger.info(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Évaluer le modèle
        """
        logger.info("Évaluation du modèle...")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Rapport de classification
        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)
        print(classification_report(
            y_test, y_pred,
            target_names=['Low', 'Medium', 'High', 'Critical']
        ))
        
        # Matrice de confusion
        print("\nCONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        
        # Importance des features
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFEATURE IMPORTANCE")
        print(feature_importance)
        
        return y_pred
    
    def save_model(self, output_path: Path):
        """Sauvegarder le modèle"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Modèle sauvegardé: {output_path}")


def main():
    """Point d'entrée principal"""
    base_dir = Path(__file__).parent.parent
    output_path = base_dir / "trained_models" / "risk_predictor.pkl"
    
    # Créer trainer
    trainer = RiskModelTrainer()
    
    # Préparer données (synthétiques pour l'instant)
    data = trainer.prepare_features(pd.DataFrame())
    
    X = data[trainer.feature_names]
    y = data['risk_level']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Entraîner
    trainer.train(X_train, y_train)
    
    # Évaluer
    trainer.evaluate(X_test, y_test)
    
    # Sauvegarder
    trainer.save_model(output_path)
    
    logger.info("✅ Entraînement terminé!")


if __name__ == "__main__":
    main()
    