"""Authentication module"""

from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import setting
from app.core.logger import logger


security = HTTPBearer(auto_error=False)


class AuthManager:
    """Authentication manager"""

    @staticmethod
    def verify(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
        """Verify authentication token"""
        api_key = setting.grok_config.get("api_key")

        if not api_key:
            logger.debug("[Auth] API_KEY not set, skipping verification.")
            return credentials.credentials if credentials else None

        if not credentials:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Missing authentication token",
                        "type": "authentication_error",
                        "code": "missing_token"
                    }
                }
            )

        if credentials.credentials != api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": f"Invalid token, length: {len(credentials.credentials)}",
                        "type": "authentication_error",
                        "code": "invalid_token"
                    }
                }
            )

        logger.debug("[Auth] Token authentication successful")
        return credentials.credentials


auth_manager = AuthManager()