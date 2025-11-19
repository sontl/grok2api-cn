"""Grok Token Manager Module"""

import json
import time
import asyncio
import aiofiles
from pathlib import Path
from curl_cffi.requests import AsyncSession
from typing import Dict, Any, Optional, Tuple

from app.models.grok_models import TokenType, Models
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.core.config import setting
from app.services.grok.statsig import get_dynamic_headers

# Constant definitions
RATE_LIMIT_ENDPOINT = "https://grok.com/rest/rate-limits"
REQUEST_TIMEOUT = 30
IMPERSONATE_BROWSER = "chrome133a"
MAX_FAILURE_COUNT = 3
TOKEN_INVALID_CODE = 401  # SSO Token expired
STATSIG_INVALID_CODE = 403  # x-statsig-id expired


class GrokTokenManager:
    """
    Grok Token Manager

    Singleton Token manager, responsible for:
    - Token file read/write operations
    - Token load balancing
    - Token status management
    - Supporting normal Token and Super Token
    """
    
    _instance: Optional['GrokTokenManager'] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> 'GrokTokenManager':
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Token manager"""
        if hasattr(self, '_initialized'):
            return

        self.token_file = Path(__file__).parents[3] / "data" / "token.json"
        self._file_lock = asyncio.Lock()
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self._storage = None

        # Synchronously load initial data
        self._load_data()
        self._initialized = True

        logger.debug(f"[Token] Manager initialization completed, file: {self.token_file}")

    def set_storage(self, storage) -> None:
        """Set storage instance"""
        self._storage = storage

    def _load_data(self) -> None:
        """Synchronously load Token data (for initialization only)"""
        default_data = {
            TokenType.NORMAL.value: {},
            TokenType.SUPER.value: {}
        }

        try:
            if self.token_file.exists():
                with open(self.token_file, "r", encoding="utf-8") as f:
                    self.token_data = json.load(f)
            else:
                self.token_data = default_data
                logger.debug("[Token] Creating new Token data file")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"[Token] Failed to load Token data: {str(e)}")
            self.token_data = default_data

    async def _save_data(self) -> None:
        """Asynchronously save Token data to storage"""
        try:
            if not self._storage:
                # If storage is not set, use traditional file saving method (backward compatibility)
                async with self._file_lock:
                    async with aiofiles.open(self.token_file, "w", encoding="utf-8") as f:
                        await f.write(json.dumps(self.token_data, indent=2, ensure_ascii=False))
            else:
                # Use storage abstraction layer
                await self._storage.save_tokens(self.token_data)
        except IOError as e:
            logger.error(f"[Token] Failed to save Token data: {str(e)}")
            raise GrokApiException(
                f"Token data save failed: {str(e)}",
                "TOKEN_SAVE_ERROR",
                {"file_path": str(self.token_file)}
            )

    @staticmethod
    def _extract_sso(auth_token: str) -> Optional[str]:
        """Extract SSO value from authentication token"""
        if "sso=" in auth_token:
            return auth_token.split("sso=")[1].split(";")[0]
        logger.warning("[Token] Unable to extract SSO value from authentication token")
        return None

    def _find_token(self, sso_value: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Find Token data, return (token_type, token_data)"""
        for token_type in [TokenType.NORMAL.value, TokenType.SUPER.value]:
            if sso_value in self.token_data[token_type]:
                return token_type, self.token_data[token_type][sso_value]
        return None, None

    async def add_token(self, tokens: list[str], token_type: TokenType) -> None:
        """Add Token to manager"""
        if not tokens:
            logger.debug("[Token] Attempting to add empty Token list")
            return

        added_count = 0
        for token in tokens:
            if not token or not token.strip():
                logger.debug("[Token] Skipping empty Token")
                continue

            self.token_data[token_type.value][token] = {
                "createdTime": int(time.time() * 1000),
                "remainingQueries": -1,
                "heavyremainingQueries": -1,
                "status": "active",
                "failedCount": 0,
                "lastFailureTime": None,
                "lastFailureReason": None,
                "priority": 0,
                "tags": [],
                "note": ""
            }
            added_count += 1

        await self._save_data()
        logger.info(f"[Token] Successfully added {added_count} {token_type.value} Tokens")

    async def delete_token(self, tokens: list[str], token_type: TokenType) -> None:
        """Delete specified Tokens"""
        if not tokens:
            logger.debug("[Token] Attempting to delete empty Token list")
            return

        deleted_count = 0
        for token in tokens:
            if token in self.token_data[token_type.value]:
                del self.token_data[token_type.value][token]
                deleted_count += 1
            else:
                logger.debug(f"[Token] Token does not exist: {token[:10]}...")

        await self._save_data()
        logger.info(f"[Token] Successfully deleted {deleted_count} {token_type.value} Tokens")

    async def update_token_tags(self, token: str, token_type: TokenType, tags: list[str]) -> None:
        """Update Token tags"""
        if token not in self.token_data[token_type.value]:
            logger.warning(f"[Token] Token does not exist: {token[:10]}...")
            raise GrokApiException(
                "Token does not exist",
                "TOKEN_NOT_FOUND",
                {"token": token[:10]}
            )

        # Ensure tags is a list and doesn't contain empty strings
        cleaned_tags = [tag.strip() for tag in tags if tag and tag.strip()]
        self.token_data[token_type.value][token]["tags"] = cleaned_tags

        await self._save_data()
        logger.info(f"[Token] Successfully updated Token {token[:10]}... tags: {cleaned_tags}")

    async def update_token_note(self, token: str, token_type: TokenType, note: str) -> None:
        """Update Token note"""
        if token not in self.token_data[token_type.value]:
            logger.warning(f"[Token] Token does not exist: {token[:10]}...")
            raise GrokApiException(
                "Token does not exist",
                "TOKEN_NOT_FOUND",
                {"token": token[:10]}
            )

        self.token_data[token_type.value][token]["note"] = note.strip()

        await self._save_data()
        logger.info(f"[Token] Successfully updated Token {token[:10]}... note")

    def get_tokens(self) -> Dict[str, Any]:
        """Get all Token data"""
        return self.token_data.copy()

    def get_token(self, model: str, exclude_tokens: list[str] = None) -> str:
        """Get Token for specified model"""
        jwt_token = self.select_token(model, exclude_tokens)
        return f"sso-rw={jwt_token};sso={jwt_token}"
    
    def get_next_token(self, model: str, exclude_tokens: list[str] = None) -> Optional[str]:
        """Get next available token, excluding specified tokens
        
        Args:
            model: Model name
            exclude_tokens: List of SSO tokens to exclude (already tried)
            
        Returns:
            SSO token string or None if no tokens available
        """
        try:
            jwt_token = self.select_token(model, exclude_tokens)
            return f"sso-rw={jwt_token};sso={jwt_token}"
        except GrokApiException as e:
            if e.error_code == "NO_AVAILABLE_TOKEN":
                return None
            raise
    
    def select_token(self, model: str, exclude_tokens: list[str] = None) -> str:
        """Select the best Token based on model type and remaining count
        
        Args:
            model: Model name
            exclude_tokens: List of SSO tokens to exclude (already tried)
        """
        exclude_tokens = exclude_tokens or []
        
        def select_best_token(tokens_dict: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
            """Select the best token from the token dictionary"""
            unused_tokens = []  # tokens with remaining = -1
            used_tokens = []    # tokens with remaining > 0

            for token_key, token_data in tokens_dict.items():
                # Skip excluded tokens
                if token_key in exclude_tokens:
                    continue
                    
                # Skip expired tokens
                if token_data.get("status") == "expired":
                    continue

                remaining = int(token_data.get(remaining_field, -1))
                priority = int(token_data.get("priority", 0))

                # Skip rate-limited tokens
                if remaining == 0:
                    continue

                # Classify and store with priority
                if remaining == -1:
                    unused_tokens.append((token_key, priority))
                elif remaining > 0:
                    used_tokens.append((token_key, remaining, priority))

            # Sort unused tokens by priority (descending)
            if unused_tokens:
                unused_tokens.sort(key=lambda x: x[1], reverse=True)
                return unused_tokens[0][0], -1

            # Sort used tokens by priority first, then by remaining count
            if used_tokens:
                used_tokens.sort(key=lambda x: (x[2], x[1]), reverse=True)
                return used_tokens[0][0], used_tokens[0][1]

            return None, None

        max_token_key = None
        max_remaining = None

        # Deep copy
        token_data_snapshot = {
            TokenType.NORMAL.value: self.token_data[TokenType.NORMAL.value].copy(),
            TokenType.SUPER.value: self.token_data[TokenType.SUPER.value].copy()
        }

        if model == "grok-4-heavy":
            # grok-4-heavy can only use Super Token + heavy remaining queries
            remaining_field = "heavyremainingQueries"
            max_token_key, max_remaining = select_best_token(token_data_snapshot[TokenType.SUPER.value])
        else:
            # Other models use remaining Queries
            remaining_field = "remainingQueries"

            # Prioritize using normal tokens
            max_token_key, max_remaining = select_best_token(token_data_snapshot[TokenType.NORMAL.value])

            # If no normal tokens are available, try using Super Token
            if max_token_key is None:
                max_token_key, max_remaining = select_best_token(token_data_snapshot[TokenType.SUPER.value])

        if max_token_key is None:
            excluded_info = f" (excluding {len(exclude_tokens)} tried tokens)" if exclude_tokens else ""
            raise GrokApiException(
                f"No available Token for model {model}{excluded_info}",
                "NO_AVAILABLE_TOKEN",
                {
                    "model": model,
                    "normal_count": len(token_data_snapshot[TokenType.NORMAL.value]),
                    "super_count": len(token_data_snapshot[TokenType.SUPER.value]),
                    "excluded_count": len(exclude_tokens)
                }
            )

        status_text = "Unused" if max_remaining == -1 else f"{max_remaining} remaining"
        excluded_info = f" (excluding {len(exclude_tokens)} tried tokens)" if exclude_tokens else ""
        logger.debug(f"[Token] Allocating Token for model {model} ({status_text}){excluded_info}")
        return max_token_key

    async def check_limits(self, auth_token: str, model: str) -> Optional[Dict[str, Any]]:
        """Check and update model rate limits"""
        try:
            rate_limit_model_name = Models.to_rate_limit(model)

            # Prepare request
            payload = {"requestKind": "DEFAULT", "modelName": rate_limit_model_name}
            cf_clearance = setting.grok_config.get("cf_clearance", "")
            cookie = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token

            headers = get_dynamic_headers("/rest/rate-limits")
            headers["Cookie"] = cookie

            # Get proxy configuration
            proxy_url = setting.grok_config.get("proxy_url", "")
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

            # Send async request
            async with AsyncSession() as session:
                response = await session.post(
                    RATE_LIMIT_ENDPOINT,
                    headers=headers,
                    json=payload,
                    impersonate=IMPERSONATE_BROWSER,
                    timeout=REQUEST_TIMEOUT,
                    proxies=proxies
                )

                if response.status_code == 200:
                    rate_limit_data = response.json()

                    # Save rate limit information
                    sso_value = self._extract_sso(auth_token)
                    if sso_value:
                        if model == "grok-4-heavy":
                            await self.update_limits(sso_value, normal=None, heavy=rate_limit_data.get("remainingQueries", -1))
                            logger.info(f"[Token] Updated Token limit: sso={sso_value[:10]}..., heavy={rate_limit_data.get('remainingQueries', -1)}")
                        else:
                            await self.update_limits(sso_value, normal=rate_limit_data.get("remainingTokens", -1), heavy=None)
                            logger.info(f"[Token] Updated Token limit: sso={sso_value[:10]}..., basic={rate_limit_data.get('remainingTokens', -1)}")

                    return rate_limit_data
                else:
                    logger.warning(f"[Token] Failed to get rate limits, status code: {response.status_code}")

                    # Handle different errors based on HTTP status code
                    sso_value = self._extract_sso(auth_token)
                    if sso_value:
                        if response.status_code == 401:
                            # Token expired
                            await self.record_failure(auth_token, 401, "Token authentication failed")
                        elif response.status_code == 403:
                            # Server blocked, don't change token status, but record failure information
                            await self.record_failure(auth_token, 403, "Server blocked (Cloudflare)")
                        else:
                            # Other errors, set to rate-limited state
                            await self.record_failure(auth_token, response.status_code, f"Rate limit or other error: {response.status_code}")

                    return None

        except Exception as e:
            logger.error(f"[Token] Error occurred while checking rate limits: {str(e)}")
            return None

    async def update_limits(self, sso_value: str, normal: Optional[int] = None, heavy: Optional[int] = None) -> None:
        """Update Token limit information"""
        try:
            for token_type in [TokenType.NORMAL.value, TokenType.SUPER.value]:
                if sso_value in self.token_data[token_type]:
                    if normal is not None:
                        self.token_data[token_type][sso_value]["remainingQueries"] = normal
                    if heavy is not None:
                        self.token_data[token_type][sso_value]["heavyremainingQueries"] = heavy

                    await self._save_data()
                    logger.info(f"[Token] Updated limit information for Token {sso_value[:10]}...")
                    return

            logger.warning(f"[Token] Token with SSO value {sso_value[:10]}... not found")

        except Exception as e:
            logger.error(f"[Token] Error occurred while updating Token limits: {str(e)}")
    
    async def record_failure(self, auth_token: str, status_code: int, error_message: str) -> None:
        """Record Token failure information

        Error code description:
        - 401: SSO Token expired, will mark Token as expired
        - 403: Server IP blocked, does not affect Token status

        Args:
            auth_token: Complete authentication token (format: sso-rw=xxx;sso=xxx)
            status_code: HTTP status code
            error_message: Error message
        """
        try:
            # 403 error is server IP blocked, not a Token issue
            if status_code == STATSIG_INVALID_CODE:
                logger.warning(
                    f"[Token] Server IP blocked (403), please 1. Change server IP 2. Use proxy IP "
                    f"3. Log in to Grok.com on the server, after passing the shield find the CF value in F12 and enter it in the backend settings"
                )
                return

            sso_value = self._extract_sso(auth_token)
            if not sso_value:
                return

            _, token_data = self._find_token(sso_value)
            if not token_data:
                logger.warning(f"[Token] Token with SSO value {sso_value[:10]}... not found")
                return

            # Update failure count
            token_data["failedCount"] = token_data.get("failedCount", 0) + 1
            token_data["lastFailureTime"] = int(time.time() * 1000)
            token_data["lastFailureReason"] = f"{status_code}: {error_message}"

            logger.warning(
                f"[Token] Token {sso_value[:10]}... failed (status code: {status_code}), "
                f"failure count: {token_data['failedCount']}/{MAX_FAILURE_COUNT}, "
                f"reason: {error_message}"
            )

            # Only mark as expired when 401 error (SSO Token expired) and failure count reaches limit
            if status_code == TOKEN_INVALID_CODE and token_data["failedCount"] >= MAX_FAILURE_COUNT:
                token_data["status"] = "expired"
                logger.error(
                    f"[Token] SSO Token {sso_value[:10]}... has been marked as expired "
                    f"(consecutive 401 errors {token_data['failedCount']} times)"
                )

            await self._save_data()

        except Exception as e:
            logger.error(f"[Token] Error occurred while recording Token failure information: {str(e)}")

    async def reset_failure(self, auth_token: str) -> None:
        """Reset Token failure count

        Call this method when Token successfully completes a request to clear failure records.

        Args:
            auth_token: Complete authentication token (format: sso-rw=xxx;sso=xxx)
        """
        try:
            sso_value = self._extract_sso(auth_token)
            if not sso_value:
                return

            _, token_data = self._find_token(sso_value)
            if not token_data:
                logger.warning(f"[Token] Token with SSO value {sso_value[:10]}... not found")
                return

            # Only reset and save when there are failure records
            if token_data.get("failedCount", 0) > 0:
                token_data["failedCount"] = 0
                token_data["lastFailureTime"] = None
                token_data["lastFailureReason"] = None

                await self._save_data()
                logger.info(f"[Token] Token {sso_value[:10]}... failure count has been reset")

        except Exception as e:
            logger.error(f"[Token] Error occurred while resetting Token failure count: {str(e)}")
    
    async def mark_token_priority(self, auth_token: str) -> None:
        """Increase priority of a working token
        
        Call this method when a token successfully completes a request to increase its priority.
        Higher priority tokens will be selected first in future requests.
        
        Args:
            auth_token: Complete authentication token (format: sso-rw=xxx;sso=xxx)
        """
        try:
            sso_value = self._extract_sso(auth_token)
            if not sso_value:
                return

            token_type, token_data = self._find_token(sso_value)
            if not token_data:
                logger.warning(f"[Token] Token with SSO value {sso_value[:10]}... not found")
                return

            # Increment priority
            current_priority = token_data.get("priority", 0)
            token_data["priority"] = current_priority + 1

            await self._save_data()
            logger.info(f"[Token] Token {sso_value[:10]}... priority increased to {token_data['priority']}")

        except Exception as e:
            logger.error(f"[Token] Error occurred while marking token priority: {str(e)}")


# Global Token manager instance
token_manager = GrokTokenManager()
