"""
Télécharger des images d'exemple pour le dataset de classification
Utilise des APIs d'images libres de droits
"""
import requests
from pathlib import Path
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Classes de déchets
WASTE_CLASSES = [
    'plastique', 'papier', 'aluminium', 'medical',
    'organique', 'verre', 'electronique', 'textile', 'autre'
]

# Mots-clés de recherche (en anglais pour meilleurs résultats)
SEARCH_KEYWORDS = {
    'plastique': ['plastic waste', 'plastic bottle', 'plastic bag'],
    'papier': ['paper waste', 'cardboard', 'newspaper waste'],
    'aluminium': ['aluminum can', 'metal waste', 'tin can'],
    'medical': ['medical waste', 'hospital waste'],
    'organique': ['organic waste', 'food waste', 'compost'],
    'verre': ['glass bottle waste', 'broken glass'],
    'electronique': ['electronic waste', 'e-waste', 'old electronics'],
    'textile': ['textile waste', 'old clothes', 'fabric waste'],
    'autre': ['mixed waste', 'garbage', 'trash']
}


def download_from_unsplash(query: str, save_path: Path, count: int = 10):
    """
    Télécharger images depuis Unsplash
    
    Note: Nécessite une clé API Unsplash (gratuit)
    Inscription: https://unsplash.com/developers
    """
    API_KEY = "VOTRE_CLE_API_UNSPLASH"  # À remplacer
    
    if API_KEY == "VOTRE_CLE_API_UNSPLASH":
        logger.warning("⚠️ Clé API Unsplash non configurée")
        logger.info("Inscrivez-vous sur: https://unsplash.com/developers")
        return 0
    
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {API_KEY}"}
    
    downloaded = 0
    
    try:
        response = requests.get(
            url,
            headers=headers,
            params={"query": query, "per_page": count}
        )
        response.raise_for_status()
        
        results = response.json().get('results', [])
        
        for i, photo in enumerate(results):
            try:
                # URL de l'image régulière
                img_url = photo['urls']['regular']
                
                # Télécharger
                img_response = requests.get(img_url, timeout=10)
                img_response.raise_for_status()
                
                # Sauvegarder
                filename = f"{query.replace(' ', '_')}_{i+1:03d}.jpg"
                file_path = save_path / filename
                
                with open(file_path, 'wb') as f:
                    f.write(img_response.content)
                
                downloaded += 1
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Erreur téléchargement image {i}: {e}")
        
        return downloaded
        
    except Exception as e:
        logger.error(f"Erreur API Unsplash: {e}")
        return 0


def create_placeholder_images(output_dir: Path, images_per_class: int = 10):
    """
    Créer des images placeholder pour développement
    """
    from PIL import Image, ImageDraw, ImageFont
    
    logger.info("🎨 Création d'images placeholder...")
    
    total_created = 0
    
    for waste_class in WASTE_CLASSES:
        class_dir = output_dir / 'train' / waste_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(images_per_class):
            # Créer image 300x300
            img = Image.new('RGB', (300, 300), color=(200, 200, 200))
            draw = ImageDraw.Draw(img)
            
            # Texte
            text = f"{waste_class.upper()}\nImage {i+1}"
            
            # Dessiner texte au centre
            bbox = draw.textbbox((0, 0), text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            position = ((300 - text_width) // 2, (300 - text_height) // 2)
            draw.text(position, text, fill=(50, 50, 50))
            
            # Sauvegarder
            filename = f"{waste_class}_{i+1:03d}.jpg"
            img.save(class_dir / filename, quality=85)
            
            total_created += 1
    
    logger.info(f"✅ {total_created} images placeholder créées")
    return total_created


def main():
    """Point d'entrée principal"""
    
    base_dir = Path('datasets/classification')
    
    logger.info("="*60)
    logger.info("TÉLÉCHARGEMENT D'IMAGES POUR CLASSIFICATION")
    logger.info("="*60)
    
    # Créer structure
    for split in ['train', 'val', 'test']:
        for waste_class in WASTE_CLASSES:
            class_dir = base_dir / split / waste_class
            class_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n📥 Options de téléchargement:")
    logger.info("  1. API Unsplash (nécessite clé API)")
    logger.info("  2. Images placeholder (pour développement)")
    logger.info("  3. Manuel (instructions)")
    
    choice = input("\nChoisissez une option (1/2/3): ").strip()
    
    if choice == "1":
        logger.info("\n📥 Téléchargement depuis Unsplash...")
        logger.info("⚠️ Configurez votre clé API dans le script")
        
        total = 0
        for waste_class in WASTE_CLASSES:
            keywords = SEARCH_KEYWORDS[waste_class]
            class_dir = base_dir / 'train' / waste_class
            
            for keyword in keywords[:1]:  # 1 mot-clé par classe
                downloaded = download_from_unsplash(
                    query=keyword,
                    save_path=class_dir,
                    count=5  # 5 images par mot-clé
                )
                total += downloaded
                time.sleep(1)
        
        logger.info(f"\n✅ Total téléchargé: {total} images")
        
    elif choice == "2":
        logger.info("\n🎨 Création d'images placeholder...")
        create_placeholder_images(base_dir, images_per_class=10)
        
    else:
        logger.info("\n📖 INSTRUCTIONS MANUELLES")
        logger.info("="*60)
        logger.info("\n1. Sources d'images gratuites:")
        logger.info("   - Unsplash: https://unsplash.com/")
        logger.info("   - Pexels: https://www.pexels.com/")
        logger.info("   - Pixabay: https://pixabay.com/")
        logger.info("\n2. Datasets publics:")
        logger.info("   - TrashNet: https://github.com/garythung/trashnet")
        logger.info("   - TACO: http://tacodataset.org/")
        logger.info("   - Kaggle Waste Classification")
        logger.info("\n3. Organisation:")
        logger.info("   - Placez images dans: datasets/classification/train/[classe]/")
        logger.info("   - Minimum 50 images par classe")
        logger.info("   - Format: JPEG ou PNG")
        logger.info("="*60)
    
    logger.info("\n✅ Terminé!")


if __name__ == "__main__":
    main()