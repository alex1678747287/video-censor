from sqlalchemy import create_engine, Column, String, Float, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, timezone
from . import config

engine = create_engine(config.DB_URL, connect_args={"check_same_thread": False, "timeout": 30})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    status = Column(String(20), default="pending")  # pending/processing/done/failed
    progress = Column(Float, default=0.0)
    stage = Column(String(50), default="")
    run_token = Column(String(36), nullable=True)
    retry_count = Column(Integer, default=0)
    execution_profile = Column(String(32), default=config.CLOUD_EXECUTION_PROFILE)
    duration = Column(Float, nullable=True)  # video duration in seconds
    violations = Column(Text, default="[]")  # JSON: detected violations
    highlights = Column(Text, default="[]")  # JSON: highlight segments
    cloud_usage = Column(Text, default="{}")  # JSON: heuristic cloud usage/cost
    output_path = Column(String(500), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(engine)


class Drama(Base):
    __tablename__ = "dramas"

    id = Column(String(36), primary_key=True)
    title = Column(String(255), nullable=False)
    status = Column(String(20), default="pending")
    progress = Column(Float, default=0.0)
    stage = Column(String(50), default="")
    run_token = Column(String(36), nullable=True)
    retry_count = Column(Integer, default=0)
    execution_profile = Column(String(32), default=config.CLOUD_EXECUTION_PROFILE)
    episode_start = Column(Integer, default=1)
    episode_end = Column(Integer, default=1)
    speed_factor = Column(Float, default=config.DRAMA_SPEED_DEFAULT)
    max_duration = Column(Integer, default=900)
    source_duration = Column(Float, nullable=True)
    cloud_usage = Column(Text, default="{}")
    disclaimer = Column(String(500), default="热门漫剧 影视效果 无任何不良引导 请勿模仿")
    output_path = Column(String(500), nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    episodes = relationship("DramaEpisode", back_populates="drama",
                            order_by="DramaEpisode.episode_num")


class DramaEpisode(Base):
    __tablename__ = "drama_episodes"

    id = Column(String(36), primary_key=True)
    drama_id = Column(String(36), ForeignKey("dramas.id"), nullable=False)
    episode_num = Column(Integer, nullable=False)
    task_id = Column(String(36), nullable=True)
    filename = Column(String(255), default="")
    status = Column(String(20), default="pending")
    upload_path = Column(String(500), nullable=True)
    censored_path = Column(String(500), nullable=True)
    highlight_clip_path = Column(String(500), nullable=True)
    highlights = Column(Text, default="[]")  # JSON: highlight segments
    violations = Column(Text, default="[]")  # JSON: per-episode violations
    violations_count = Column(Integer, default=0)

    drama = relationship("Drama", back_populates="episodes")


Drama.__table__.create(engine, checkfirst=True)
DramaEpisode.__table__.create(engine, checkfirst=True)


def _ensure_sqlite_column(table_name: str, column_name: str, column_sql: str):
    """Lightweight SQLite migration for newly added columns."""
    if not config.DB_URL.startswith("sqlite"):
        return
    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table_name})")}
        if column_name not in cols:
            conn.exec_driver_sql(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
            )


_ensure_sqlite_column("drama_episodes", "violations", "TEXT DEFAULT '[]'")
_ensure_sqlite_column("tasks", "stage", "TEXT DEFAULT ''")
_ensure_sqlite_column("tasks", "run_token", "TEXT")
_ensure_sqlite_column("tasks", "retry_count", "INTEGER DEFAULT 0")
_ensure_sqlite_column("tasks", "execution_profile", f"TEXT DEFAULT '{config.CLOUD_EXECUTION_PROFILE}'")
_ensure_sqlite_column("tasks", "started_at", "DATETIME")
_ensure_sqlite_column("tasks", "completed_at", "DATETIME")
_ensure_sqlite_column("tasks", "cloud_usage", "TEXT DEFAULT '{}'")
_ensure_sqlite_column("dramas", "run_token", "TEXT")
_ensure_sqlite_column("dramas", "retry_count", "INTEGER DEFAULT 0")
_ensure_sqlite_column("dramas", "execution_profile", f"TEXT DEFAULT '{config.CLOUD_EXECUTION_PROFILE}'")
_ensure_sqlite_column("dramas", "source_duration", "FLOAT")
_ensure_sqlite_column("dramas", "started_at", "DATETIME")
_ensure_sqlite_column("dramas", "completed_at", "DATETIME")
_ensure_sqlite_column("dramas", "cloud_usage", "TEXT DEFAULT '{}'")
