from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer_scheme = HTTPBearer(auto_error=False)


def create_api_key_dependency(api_key: str):
    async def _auth(credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme)):
        if not credentials or credentials.scheme.lower() != "bearer" or credentials.credentials != api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key",
            )
    return _auth
