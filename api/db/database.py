"""
Async database plumbing: engine, session factory, declarative Base, and the
FastAPI dependency.

Everything here is built on SQLAlchemy 2.0's asyncio support. The single
`engine` owns a connection pool shared across the process. Each request gets its
own short-lived `AsyncSession` via the `get_db` dependency — sessions are never
shared across requests (see CLAUDE.md "Architecture Rules").
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from api.config import settings

# ─────────────────────────────────────────────────────────────
# Engine + connection pool
# ─────────────────────────────────────────────────────────────
# pool_size/max_overflow bound concurrent connections; pool_pre_ping issues a
# lightweight liveness check before handing out a pooled connection so we don't
# hand back a stale one after a DB restart.
engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)

# ─────────────────────────────────────────────────────────────
# Session factory
# ─────────────────────────────────────────────────────────────
# expire_on_commit=False keeps ORM attributes accessible after commit() without
# triggering a fresh (async, and therefore awkward) lazy load.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# ─────────────────────────────────────────────────────────────
# Declarative base for ORM models (api/db/models.py)
# ─────────────────────────────────────────────────────────────
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a request-scoped async session.

    Usage:

        @router.get("/thing")
        async def handler(db: AsyncSession = Depends(get_db)):
            ...

    The `async with` block guarantees the session is closed (and any open
    transaction rolled back) when the request finishes.
    """
    async with AsyncSessionLocal() as session:
        yield session
