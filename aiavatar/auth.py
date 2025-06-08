from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class BearerAuth:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.bearer_scheme = HTTPBearer(auto_error=False)
    
    async def __call__(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))
    ) -> Optional[str]:
        # No API key configured = no authentication required
        if not self.api_key:
            return None
        
        # API key configured but no credentials provided
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify the token
        if credentials.credentials != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return credentials.credentials
