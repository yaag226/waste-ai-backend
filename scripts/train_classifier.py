"""
Script d'entraînement du classificateur de déchets
"""
import sys
from pathlib import Path

# Ajouter le projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WasteClassifierTrainer:
    """
    Entraîneur pour le classificateur de déchets
    
    Architecture: CNN (MobileNetV2 transfer learning)
    """
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.classes = [
            'plastique', 'papier', 'aluminium', 'medical',
            'organique', 'verre', 'electronique', 'textile', 'autre'
        ]
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        
        self.model = None
    
    def load_dataset(self):
        """
        Charger le dataset depuis le dossier
        
        Structure attendue:
        datasets/classification/
            plastique/
                img1.jpg
                img2.jpg
            papier/
                img1.jpg
            ...
        """
        logger.info("Chargement du dataset...")
        
        # Utiliser tf.keras.preprocessing.image_dataset_from_directory
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.dataset_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size
        )
        
        # Optimisation performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def build_model(self):
        """
        Construire le modèle CNN avec transfer learning
        """
        logger.info("Construction du modèle...")
        
        # Base: MobileNetV2 pré-entraîné
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Geler les couches de base
        base_model.trainable = False
        
        # Ajouter couches de classification
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        
        # Normalisation
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Pooling et classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(len(self.classes), activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compiler
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Modèle créé: {self.model.count_params()} paramètres")
        
        return self.model
    
    def train(self, train_ds, val_ds):
        """
        Entraîner le modèle
        """
        logger.info("Début de l'entraînement...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                '../trained_models/checkpoints/waste_classifier_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Entraînement
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks
        )
        
        logger.info("Entraînement terminé!")
        
        return history
    
    def fine_tune(self, train_ds, val_ds):
        """
        Fine-tuning: dégeler une partie du modèle de base
        """
        logger.info("Début du fine-tuning...")
        
        # Dégeler les dernières couches du base model
        base_model = self.model.layers[4]  # Le MobileNetV2
        base_model.trainable = True
        
        # Geler les 100 premières couches
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # Recompiler avec learning rate plus faible
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continuer l'entraînement
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=20,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        logger.info("Fine-tuning terminé!")
        
        return history
    
    def save_model(self, output_path: Path):
        """Sauvegarder le modèle"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(output_path)
        logger.info(f"✅ Modèle sauvegardé: {output_path}")
    
    def evaluate(self, test_ds):
        """Évaluer le modèle"""
        loss, accuracy = self.model.evaluate(test_ds)
        logger.info(f"📊 Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return loss, accuracy


def main():
    """Point d'entrée principal"""
    # Chemins
    base_dir = Path(__file__).parent.parent
    dataset_path = base_dir / "datasets" / "classification"
    output_path = base_dir / "trained_models" / "waste_classifier_v1.h5"
    
    # Vérifier dataset
    if not dataset_path.exists():
        logger.error(f"❌ Dataset non trouvé: {dataset_path}")
        logger.info("Créez la structure: datasets/classification/[classe]/images.jpg")
        return
    
    # Créer trainer
    trainer = WasteClassifierTrainer(dataset_path)
    
    # Charger données
    train_ds, val_ds = trainer.load_dataset()
    
    # Construire modèle
    trainer.build_model()
    
    # Entraîner
    history1 = trainer.train(train_ds, val_ds)
    
    # Fine-tuning
    history2 = trainer.fine_tune(train_ds, val_ds)
    
    # Évaluer
    trainer.evaluate(val_ds)
    
    # Sauvegarder
    trainer.save_model(output_path)
    
    logger.info("✅ Entraînement terminé avec succès!")


if __name__ == "__main__":
    main()