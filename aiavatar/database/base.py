from abc import ABC, abstractmethod
from typing import Any


class PoolProvider(ABC):
    """Abstract base class for database connection pool providers."""

    @property
    @abstractmethod
    def db_type(self) -> str:
        """Database type identifier (e.g., 'postgresql', 'mysql')."""
        pass

    @abstractmethod
    async def get_pool(self) -> Any:
        """Get the database connection pool."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection pool."""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get pool statistics."""
        pass
