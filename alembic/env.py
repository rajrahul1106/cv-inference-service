"""
Alembic environment, wired for async SQLAlchemy (asyncpg).

Key points:
- The DB URL comes from api.config.settings, not alembic.ini, so migrations
  always target the same database as the app.
- `target_metadata` is our declarative `Base.metadata`. Importing api.db.models
  is what registers the Job/Result tables onto that metadata, so autogenerate
  can see them.
- Online migrations run through an async engine; Alembic's migration context is
  synchronous, so we drive it via `connection.run_sync(...)`.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from api.config import settings
from api.db.database import Base
import api.db.models  # noqa: F401  -- registers models on Base.metadata

# Alembic Config object (provides access to values in alembic.ini).
config = context.config

# Inject the application's database URL.
config.set_main_option("sqlalchemy.url", settings.database_url)

# Configure Python logging from the ini file, if present.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata autogenerate diffs the live database against.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL, no DBAPI connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure the context on a live (sync-facing) connection and migrate."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations within it."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
