"""
Classificateur de types de déchets
"""
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any
import os

from app.config import settings
from app.schemas.responses import WasteType

logger = logging.getLogger(__name__)


class WasteClassifier:
    """
    Classificateur de déchets par vision par ordinateur
    
    Pour l'instant: Mock implementation
    En production: Charger modèle TensorFlow/PyTorch entraîné
    """
    
    def __init__(self):
        self.model = None
        self.classes = [wt.value for wt in WasteType]
        self.input_size = (224, 224)
        
        # Essayer de charger le modèle
        self._load_model()
    
    def _load_model(self):
        """Charger le modèle entraîné"""
        model_path = settings.MODEL_PATH / settings.CLASSIFICATION_MODEL
        
        if model_path.exists():
            try:
                # TODO: Charger modèle réel
                # import tensorflow as tf
                # self.model = tf.keras.models.load_model(str(model_path))
                logger.info(f"✅ Modèle chargé: {model_path}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur chargement modèle: {e}")
                logger.info("Mode MOCK activé")
        else:
            logger.warning(f"⚠️ Modèle non trouvé: {model_path}")
            logger.info("Mode MOCK activé - génération de prédictions aléatoires")
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Prédire le type de déchet d'une image
        
        Args:
            image: Image PIL
        
        Returns:
            Dict avec waste_type, confidence, probabilities
        """
        try:
            if self.model is not None:
                # TODO: Prédiction réelle avec modèle
                # preprocessed = self._preprocess(image)
                # predictions = self.model.predict(preprocessed)
                # ...
                pass
            
            # MOCK: Génération aléatoire pour développement
            return self._mock_prediction()
            
        except Exception as e:
            logger.error(f"Erreur prédiction: {e}", exc_info=True)
            raise
    
    def _mock_prediction(self) -> Dict[str, Any]:
        """
        Génération de prédiction mock pour développement
        """
        # Probabilités aléatoires
        probabilities = np.random.dirichlet(np.ones(len(self.classes)))
        
        # Trouver classe avec plus haute probabilité
        max_idx = np.argmax(probabilities)
        predicted_class = self.classes[max_idx]
        confidence = float(probabilities[max_idx] * 100)
        
        # Créer dict de probabilités
        prob_dict = {
            self.classes[i]: float(probabilities[i])
            for i in range(len(self.classes))
        }
        
        return {
            "waste_type": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": prob_dict,
            "is_valid": confidence > settings.CONFIDENCE_THRESHOLD * 100
        }
    
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Prétraiter l'image pour le modèle
        
        Args:
            image: Image PIL
        
        Returns:
            Array numpy prêt pour prédiction
        """
        # Redimensionner
        img_resized = image.resize(self.input_size)
        
        # Convertir en array
        img_array = np.array(img_resized)
        
        # Normaliser [0, 255] -> [0, 1]
        img_normalized = img_array / 255.0
        
        # Ajouter dimension batch
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtenir informations sur le modèle"""
        return {
            "model_loaded": self.model is not None,
            "model_path": str(settings.MODEL_PATH / settings.CLASSIFICATION_MODEL),
            "classes": self.classes,
            "input_size": self.input_size,
            "mode": "production" if self.model else "mock"
        }
    