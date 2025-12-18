"""Authentication module"""

from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import setting
from app.core.logger import logger


# Bearer security scheme
security = HTTPBearer(auto_error=False)


def _build_error(message: str, code: str = "invalid_token") -> dict:
    """Build authentication error"""
    return {
        "error": {
            "message": message,
            "type": "authentication_error",
            "code": code
        }
    }


class AuthManager:
    """Authentication manager"""

    @staticmethod
    def verify(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
        """Verify authentication token"""
        api_key = setting.grok_config.get("api_key")

        # Skip if not set
        if not api_key:
            logger.debug("[Auth] API_KEY not set, skipping verification.")
            return credentials.credentials if credentials else None

        # Check token
        if not credentials:
            raise HTTPException(
                status_code=401,
                detail=_build_error("Missing authentication token", "missing_token")
            )

        # Verify token
        if credentials.credentials != api_key:
            raise HTTPException(
                status_code=401,
                detail=_build_error(f"Invalid token, length: {len(credentials.credentials)}", "invalid_token")
            )

        logger.debug("[Auth] Token authentication successful")
        return credentials.credentials


# Global instance
auth_manager = AuthManager()