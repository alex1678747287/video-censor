from sqlalchemy import create_engine, Column, String, Float, Text, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
from . import config

engine = create_engine(config.DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    status = Column(String(20), default="pending")  # pending/processing/done/failed
    progress = Column(Float, default=0.0)
    duration = Column(Float, nullable=True)  # video duration in seconds
    violations = Column(Text, default="[]")  # JSON: detected violations
    highlights = Column(Text, default="[]")  # JSON: highlight segments
    output_path = Column(String(500), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(engine)
