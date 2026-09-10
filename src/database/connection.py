import os
from contextlib import contextmanager
from typing import Generator
from pathlib import Path
import logging
from sqlalchemy import create_engine, event, exc, pool, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
logger = logging.getLogger(__name__)


class DatabaseConnection:

    def __init__(self):
        self.database_url = self._build_database_url()
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine, expire_on_commit=False
        )
        logger.info("Database connection initialized successfully")

    def _build_database_url(self) -> str:
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "mlops_registry")
        user = os.getenv("DB_USER", "mlops_user")
        password = os.getenv("DB_PASSWORD", "mlops_password")
        if os.getenv("ENVIRONMENT") == "production" and password == "mlops_password":
            logger.warning(
                "Using default password in production environment. Please set DB_PASSWORD environment variable."
            )
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        logger.info(
            f"Database URL configured: postgresql://{user}@{host}:{port}/{database}"
        )
        return url

    def _create_engine(self):
        pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
        max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            future=True,
        )
        self._setup_event_listeners(engine)
        logger.info(
            f"Database engine created with pool_size={pool_size}, max_overflow={max_overflow}"
        )
        return engine

    def _setup_event_listeners(self, engine):

        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            logger.debug("New database connection established")

        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")

        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            logger.debug("Connection returned to pool")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
            logger.debug("Session committed successfully")
        except exc.SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Session rollback due to error: {str(e)}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error, session rolled back: {str(e)}")
            raise
        finally:
            session.close()
            logger.debug("Session closed")

    def health_check(self) -> bool:
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            logger.info("Database health check passed")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

    def close(self):
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


_db_connection: DatabaseConnection | None = None


def get_db_connection() -> DatabaseConnection:
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection


def get_session() -> Generator[Session, None, None]:
    db = get_db_connection()
    with db.get_session() as session:
        yield session
