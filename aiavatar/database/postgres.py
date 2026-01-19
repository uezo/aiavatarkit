import asyncio
from typing import Optional

import asyncpg

from .base import PoolProvider


class PostgreSQLPoolProvider(PoolProvider):
    """
    Centralized PostgreSQL connection pool provider.

    Use this to share a single connection pool across multiple components,
    enabling better control over total database connections.

    Example:
        pool_provider = PostgreSQLPoolProvider(
            connection_str="postgresql://user:pass@localhost:5432/db",
            min_size=5,
            max_size=30,
        )

        context_manager = PostgreSQLContextManager(get_pool=pool_provider.get_pool)
        session_manager = PostgreSQLSessionStateManager(get_pool=pool_provider.get_pool)
    """

    def __init__(
        self,
        connection_str: str = None,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = None,
        user: str = None,
        password: str = None,
        min_size: int = 5,
        max_size: int = 20,
    ):
        self.connection_str = connection_str
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.min_size = min_size
        self.max_size = max_size

        self._pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()

    @property
    def db_type(self) -> str:
        return "postgresql"

    async def get_pool(self) -> asyncpg.Pool:
        """
        Get the shared connection pool, creating it on first access.

        This method is thread-safe and will only create one pool
        even if called concurrently from multiple coroutines.
        """
        if self._pool is not None:
            return self._pool

        async with self._pool_lock:
            if self._pool is not None:
                return self._pool

            if self.connection_str:
                self._pool = await asyncpg.create_pool(
                    dsn=self.connection_str,
                    min_size=self.min_size,
                    max_size=self.max_size,
                )
            else:
                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.dbname,
                    user=self.user,
                    password=self.password,
                    min_size=self.min_size,
                    max_size=self.max_size,
                )

            return self._pool

    async def close(self) -> None:
        """Close the connection pool if it exists."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def get_stats(self) -> dict:
        """
        Get current pool statistics.

        Returns:
            dict: Pool statistics including:
                - initialized: Whether the pool has been created
                - size: Total connections in the pool
                - idle: Idle (available) connections
                - in_use: Currently used connections
                - min_size: Configured minimum pool size
                - max_size: Configured maximum pool size
        """
        if self._pool is None:
            return {"initialized": False}

        return {
            "initialized": True,
            "size": self._pool.get_size(),
            "idle": self._pool.get_idle_size(),
            "in_use": self._pool.get_size() - self._pool.get_idle_size(),
            "min_size": self._pool.get_min_size(),
            "max_size": self._pool.get_max_size(),
        }
