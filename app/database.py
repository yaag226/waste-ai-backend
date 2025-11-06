"""
Connexion à la base de données Laravel
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Engine SQLAlchemy
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour les modèles
Base = declarative_base()


# Modèles SQLAlchemy (mapping tables Laravel)
class Report(Base):
    """Modèle pour la table reports"""
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    waste_type_id = Column(Integer, nullable=True)
    title = Column(String(255))
    description = Column(Text, nullable=True)
    image_path = Column(String(255), nullable=True)
    latitude = Column(Float)
    longitude = Column(Float)
    address = Column(String(255), nullable=True)
    city = Column(String(100))
    reporter_name = Column(String(255), nullable=True)
    reporter_phone = Column(String(20), nullable=True)
    reporter_email = Column(String(255), nullable=True)
    status = Column(String(50), default="pending")
    is_police_report = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)
    processed_by = Column(Integer, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class User(Base):
    """Modèle pour la table users"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    phone = Column(String(20), unique=True)
    email = Column(String(255), nullable=True)
    city = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class WasteType(Base):
    """Modèle pour la table waste_types"""
    __tablename__ = "waste_types"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    description = Column(Text, nullable=True)
    icon = Column(String(50), nullable=True)
    color = Column(String(20), nullable=True)


# Dependency pour obtenir une session DB
def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour obtenir une session de base de données
    
    Usage:
        @app.get("/reports")
        def get_reports(db: Session = Depends(get_db)):
            reports = db.query(Report).all()
            return reports
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection() -> bool:
    """Tester la connexion à la base de données"""
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        logger.info("✅ Connexion à la base de données réussie")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion à la base de données: {e}")
        return False