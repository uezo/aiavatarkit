import base64
import binascii
import inspect
import secrets
from typing import Any, Awaitable, Callable, Protocol, Union

from fastapi import HTTPException, Request, status


AuthResult = Union[Any, Awaitable[Any]]


class AdminAuthenticator(Protocol):
    """Replaceable authentication boundary for every Admin route."""

    def __call__(self, request: Request) -> AuthResult:
        ...


class BasicAdminAuthenticator:
    def __init__(self, username: str, password: str, *, realm: str = "Admin"):
        self.username = username
        self.password = password
        self.realm = realm

    async def __call__(self, request: Request) -> str:
        unauthorized = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing credentials",
            headers={"WWW-Authenticate": f'Basic realm="{self.realm}"'},
        )
        header = request.headers.get("Authorization", "")
        scheme, _, encoded = header.partition(" ")
        if scheme.lower() != "basic" or not encoded:
            raise unauthorized
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (binascii.Error, ValueError, UnicodeDecodeError):
            raise unauthorized
        if not (
            secrets.compare_digest(username, self.username)
            and secrets.compare_digest(password, self.password)
        ):
            raise unauthorized
        return username


def create_auth_dependency(authenticator: AdminAuthenticator = None) -> Callable:
    async def authenticate(request: Request):
        if authenticator is None:
            return None
        result = authenticator(request)
        if inspect.isawaitable(result):
            result = await result
        if result is False:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
            )
        return result

    return authenticate
