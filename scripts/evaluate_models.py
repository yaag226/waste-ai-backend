"""
Script d'évaluation des modèles entraînés
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from PIL import Image
import pickle
import logging
from typing import Dict, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Évaluateur pour tous les modèles IA
    """
    
    def __init__(self, models_path: Path):
        self.models_path = models_path
        self.results = {}
    
    def evaluate_classifier(self, test_images_path: Path) -> Dict:
        """
        Évaluer le classificateur de déchets
        
        Args:
            test_images_path: Dossier avec images de test
        
        Returns:
            Dict avec métriques
        """
        logger.info("🔬 Évaluation du classificateur de déchets...")
        
        try:
            # Charger modèle
            import tensorflow as tf
            model_path = self.models_path / "waste_classifier_v1.h5"
            
            if not model_path.exists():
                logger.warning(f"Modèle non trouvé: {model_path}")
                return {"status": "model_not_found"}
            
            model = tf.keras.models.load_model(str(model_path))
            logger.info(f"✅ Modèle chargé: {model_path}")
            
            # Charger test set
            test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                test_images_path,
                image_size=(224, 224),
                batch_size=32,
                shuffle=False
            )
            
            # Évaluer
            loss, accuracy = model.evaluate(test_ds)
            
            # Prédictions pour matrice de confusion
            predictions = model.predict(test_ds)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Labels réels
            true_classes = np.concatenate([y for x, y in test_ds], axis=0)
            
            # Métriques par classe
            from sklearn.metrics import classification_report, confusion_matrix
            
            class_names = test_ds.class_names
            report = classification_report(
                true_classes,
                predicted_classes,
                target_names=class_names,
                output_dict=True
            )
            
            conf_matrix = confusion_matrix(true_classes, predicted_classes)
            
            results = {
                "status": "success",
                "model": "waste_classifier",
                "test_loss": float(loss),
                "test_accuracy": float(accuracy),
                "classification_report": report,
                "confusion_matrix": conf_matrix.tolist(),
                "num_test_samples": len(true_classes)
            }
            
            logger.info(f"📊 Accuracy: {accuracy:.4f}")
            logger.info(f"📊 Loss: {loss:.4f}")
            
            self.results['classifier'] = results
            return results
            
        except Exception as e:
            logger.error(f"Erreur évaluation classificateur: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def evaluate_risk_predictor(self, test_data_path: Path) -> Dict:
        """
        Évaluer le prédicteur de risque
        
        Args:
            test_data_path: CSV avec données de test
        
        Returns:
            Dict avec métriques
        """
        logger.info("🔬 Évaluation du prédicteur de risque...")
        
        try:
            # Charger modèle
            model_path = self.models_path / "risk_predictor.pkl"
            
            if not model_path.exists():
                logger.warning(f"Modèle non trouvé: {model_path}")
                return {"status": "model_not_found"}
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            
            logger.info(f"✅ Modèle chargé: {model_path}")
            
            # Charger test data
            if test_data_path.exists():
                test_data = pd.read_csv(test_data_path)
            else:
                # Générer données synthétiques pour test
                logger.info("Génération de données de test synthétiques...")
                test_data = pd.DataFrame({
                    'report_count': np.random.randint(1, 50, 200),
                    'avg_time_between_reports': np.random.uniform(1, 30, 200),
                    'recent_reports_7d': np.random.randint(0, 10, 200),
                    'recent_reports_30d': np.random.randint(0, 30, 200),
                    'pending_ratio': np.random.uniform(0, 1, 200),
                    'police_reports_ratio': np.random.uniform(0, 0.3, 200),
                    'spatial_density': np.random.uniform(0, 1, 200)
                })
                
                # Générer labels
                test_data['risk_level'] = 0
                test_data.loc[test_data['report_count'] > 10, 'risk_level'] = 1
                test_data.loc[test_data['report_count'] > 20, 'risk_level'] = 2
                test_data.loc[test_data['report_count'] > 35, 'risk_level'] = 3
            
            X_test = test_data[feature_names]
            y_test = test_data['risk_level']
            
            # Prédire
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Métriques
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred,
                target_names=['Low', 'Medium', 'High', 'Critical'],
                output_dict=True
            )
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results = {
                "status": "success",
                "model": "risk_predictor",
                "test_accuracy": float(accuracy),
                "classification_report": report,
                "confusion_matrix": conf_matrix.tolist(),
                "feature_importance": feature_importance.to_dict('records'),
                "num_test_samples": len(y_test)
            }
            
            logger.info(f"📊 Accuracy: {accuracy:.4f}")
            
            self.results['risk_predictor'] = results
            return results
            
        except Exception as e:
            logger.error(f"Erreur évaluation risk predictor: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def evaluate_size_estimator(self) -> Dict:
        """
        Évaluer l'estimateur de taille
        
        Note: Pour l'instant mock, car pas de modèle réel
        """
        logger.info("🔬 Évaluation de l'estimateur de taille...")
        
        results = {
            "status": "mock",
            "model": "size_estimator",
            "note": "Modèle en mode MOCK - évaluation non disponible"
        }
        
        self.results['size_estimator'] = results
        return results
    
    def generate_report(self, output_path: Path):
        """
        Générer un rapport d'évaluation complet
        
        Args:
            output_path: Chemin du fichier de rapport JSON
        """
        logger.info("📄 Génération du rapport d'évaluation...")
        
        report = {
            "evaluation_date": pd.Timestamp.now().isoformat(),
            "models_evaluated": list(self.results.keys()),
            "results": self.results,
            "summary": {
                "classifier_accuracy": self.results.get('classifier', {}).get('test_accuracy', 'N/A'),
                "risk_predictor_accuracy": self.results.get('risk_predictor', {}).get('test_accuracy', 'N/A'),
            }
        }
        
        # Sauvegarder
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Rapport sauvegardé: {output_path}")
        
        # Afficher résumé
        print("\n" + "="*60)
        print("RÉSUMÉ DE L'ÉVALUATION")
        print("="*60)
        for model_name, results in self.results.items():
            print(f"\n{model_name.upper()}")
            print("-" * 40)
            if results.get('status') == 'success':
                if 'test_accuracy' in results:
                    print(f"  Accuracy: {results['test_accuracy']:.4f}")
                if 'test_loss' in results:
                    print(f"  Loss: {results['test_loss']:.4f}")
            else:
                print(f"  Status: {results.get('status')}")
        print("="*60 + "\n")


def main():
    """Point d'entrée principal"""
    base_dir = Path(__file__).parent.parent
    models_path = base_dir / "trained_models"
    test_data_path = base_dir / "datasets" / "test"
    output_path = base_dir / "evaluation_report.json"
    
    # Créer évaluateur
    evaluator = ModelEvaluator(models_path)
    
    # Évaluer classificateur
    if (test_data_path / "classification").exists():
        evaluator.evaluate_classifier(test_data_path / "classification")
    else:
        logger.warning("⚠️ Pas de données de test pour le classificateur")
    
    # Évaluer risk predictor
    evaluator.evaluate_risk_predictor(test_data_path / "risk_data.csv")
    
    # Évaluer size estimator
    evaluator.evaluate_size_estimator()
    
    # Générer rapport
    evaluator.generate_report(output_path)
    
    logger.info("✅ Évaluation terminée!")


if __name__ == "__main__":
    main()