"""Grok Token Manager Module"""

import orjson
import time
import asyncio
import portalocker
import json
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
TOKEN_INVALID_CODE = 401
STATSIG_INVALID_CODE = 403

class GrokTokenManager:
    """
    Grok Token Manager (Singleton)
    Responsible for Token file read/write, load balancing, and status management.
    """
    
    _instance: Optional['GrokTokenManager'] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> 'GrokTokenManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self.token_file = Path(__file__).parents[3] / "data" / "token.json"
        self._file_lock = asyncio.Lock()
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self._storage = None

        self.token_data = None  # Lazy load
        
        # Batch save queue
        self._save_pending = False
        self._save_task = None
        self._shutdown = False
        
        self._initialized = True
        logger.debug(f"[Token] Manager initialization completed, file: {self.token_file}")

    def set_storage(self, storage) -> None:
        """Set storage instance"""
        self._storage = storage

    async def _load_data(self) -> None:
        """Asynchronously load Token data"""
        default_data = {
            TokenType.NORMAL.value: {},
            TokenType.SUPER.value: {}
        }
        
        try:
            if self.token_file.exists():
                async with self._file_lock:
                    with open(self.token_file, "r", encoding="utf-8") as f:
                        portalocker.lock(f, portalocker.LOCK_SH)
                        try:
                            content = f.read()
                            self.token_data = orjson.loads(content)
                        finally:
                            portalocker.unlock(f)
            else:
                self.token_data = default_data
                logger.debug("[Token] Creating new Token data file")
        except Exception as e:
            logger.error(f"[Token] Failed to load Token data: {str(e)}")
            self.token_data = default_data

    async def _save_data(self) -> None:
        """Asynchronously save Token data"""
        try:
            if not self._storage:
                async with self._file_lock:
                    with open(self.token_file, "w", encoding="utf-8") as f:
                        portalocker.lock(f, portalocker.LOCK_EX)
                        try:
                            content = orjson.dumps(self.token_data, option=orjson.OPT_INDENT_2).decode()
                            f.write(content)
                            f.flush()
                        finally:
                            portalocker.unlock(f)
            else:
                await self._storage.save_tokens(self.token_data)
        except Exception as e:
            logger.error(f"[Token] Failed to save Token data: {str(e)}")
            raise GrokApiException(
                f"Token data save failed: {str(e)}",
                "TOKEN_SAVE_ERROR",
                {"file_path": str(self.token_file)}
            )

    def _mark_dirty(self) -> None:
        """Mark data as pending save"""
        self._save_pending = True

    async def _batch_save_worker(self) -> None:
        """Batch save background task"""
        # import inside to avoid circular dependency if needed
        from app.core.config import setting
        
        interval = setting.global_config.get("batch_save_interval", 1.0)
        logger.info(f"[Token] Batch save task started, interval: {interval}s")
        
        while not self._shutdown:
            await asyncio.sleep(interval)
            
            if self._save_pending and not self._shutdown:
                try:
                    await self._save_data()
                    self._save_pending = False
                    logger.debug("[Token] Batch save completed")
                except Exception as e:
                    logger.error(f"[Token] Batch save failed: {e}")

    async def start_batch_save(self) -> None:
        """Start batch save task"""
        if self._save_task is None:
            self._save_task = asyncio.create_task(self._batch_save_worker())
            logger.info("[Token] Batch save task created")

    async def shutdown(self) -> None:
        """Shutdown and flush pending data"""
        self._shutdown = True
        
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        if self._save_pending:
            await self._save_data()
            logger.info("[Token] Flush completed on shutdown")

    @staticmethod
    def _extract_sso(auth_token: str) -> Optional[str]:
        if not auth_token:
            return None
        sso_value = None
        parts = auth_token.split(';')
        for part in parts:
            part = part.strip()
            if part.startswith('sso='):
                sso_value = part[4:].strip()
                break
        if not sso_value and "sso=" in auth_token and "sso-rw=" not in auth_token:
             try:
                 sso_value = auth_token.split("sso=")[1].split(";")[0].strip()
             except IndexError:
                 pass
        if sso_value:
            if '%' in sso_value:
                try:
                    from urllib.parse import unquote
                    decoded = unquote(sso_value)
                    if decoded != sso_value:
                         sso_value = decoded
                except Exception:
                    pass
            return sso_value
        logger.warning(f"[Token] Unable to extract SSO value. Auth token: {auth_token[:50]}...")
        return None

    def _find_token(self, sso_value: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        keys_to_check = [TokenType.NORMAL.value, TokenType.SUPER.value]
        
        # Direct match
        for key in keys_to_check:
            if key in self.token_data and sso_value in self.token_data[key]:
                return key, self.token_data[key][sso_value]
        
        # Fallback search
        for key in keys_to_check:
            if key in self.token_data:
                for token_key, token_data in self.token_data[key].items():
                    if sso_value in token_key:
                        return key, token_data
                    if "sso=" in token_key:
                        extracted = self._extract_sso(token_key)
                        if extracted == sso_value:
                            return key, token_data
        return None, None

    async def add_token(self, tokens: list[str], token_type: TokenType) -> None:
        if not tokens:
            return
        count = 0
        for token in tokens:
            if not token or not token.strip():
                continue
            clean_token = token.strip()
            self.token_data[token_type.value][clean_token] = {
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
            count += 1
        self._mark_dirty()
        logger.info(f"[Token] Successfully added {count} {token_type.value} Tokens")

    async def delete_token(self, tokens: list[str], token_type: TokenType) -> None:
        if not tokens:
            return
        count = 0
        for token in tokens:
            if token in self.token_data[token_type.value]:
                del self.token_data[token_type.value][token]
                count += 1
        self._mark_dirty()
        logger.info(f"[Token] Successfully deleted {count} {token_type.value} Tokens")

    async def update_token_tags(self, token: str, token_type: TokenType, tags: list[str]) -> None:
        if token not in self.token_data[token_type.value]:
            raise GrokApiException("Token does not exist", "TOKEN_NOT_FOUND", {"token": token[:10]})
        cleaned_tags = [tag.strip() for tag in tags if tag and tag.strip()]
        self.token_data[token_type.value][token]["tags"] = cleaned_tags
        self._mark_dirty()
        logger.info(f"[Token] Successfully updated Token {token[:10]}... tags")

    async def update_token_note(self, token: str, token_type: TokenType, note: str) -> None:
        if token not in self.token_data[token_type.value]:
            raise GrokApiException("Token does not exist", "TOKEN_NOT_FOUND", {"token": token[:10]})
        self.token_data[token_type.value][token]["note"] = note.strip()
        self._mark_dirty()
        logger.info(f"[Token] Successfully updated Token {token[:10]}... note")

    def get_tokens(self) -> Dict[str, Any]:
        return self.token_data.copy()

    def _reload_if_needed(self) -> None:
        """Reload data if needed (for multi-process mode)"""
        if self._storage:
            return
        try:
            if self.token_file.exists():
                with open(self.token_file, "r", encoding="utf-8") as f:
                    portalocker.lock(f, portalocker.LOCK_SH)
                    try:
                        content = f.read()
                        self.token_data = orjson.loads(content)
                    finally:
                        portalocker.unlock(f)
        except Exception as e:
            logger.warning(f"[Token] Reload failed: {e}")

    def get_token(self, model: str, exclude_tokens: list[str] = None) -> str:
        jwt_token = self.select_token(model, exclude_tokens)
        if "sso=" in jwt_token and "sso-rw=" in jwt_token:
            return jwt_token
        return f"sso-rw={jwt_token};sso={jwt_token}"

    def get_next_token(self, model: str, exclude_tokens: list[str] = None) -> Optional[str]:
        try:
            jwt_token = self.select_token(model, exclude_tokens)
            return f"sso-rw={jwt_token};sso={jwt_token}"
        except GrokApiException as e:
            if e.error_code == "NO_AVAILABLE_TOKEN":
                return None
            raise

    def select_token(self, model: str, exclude_tokens: list[str] = None) -> str:
        self._reload_if_needed()
        exclude_tokens = exclude_tokens or []
        
        def select_best_token(tokens_dict: Dict[str, Any], field: str) -> Tuple[Optional[str], Optional[int]]:
            unused_tokens = []
            used_tokens = []
            
            for token_key, token_data in tokens_dict.items():
                if token_key in exclude_tokens:
                    continue
                # Detailed exclusion check
                is_excluded = False
                for excluded in exclude_tokens:
                    if excluded in token_key or token_key in excluded:
                        is_excluded = True
                        break
                if is_excluded:
                    continue
                
                if token_data.get("status") == "expired":
                    continue
                if token_data.get("failedCount", 0) >= MAX_FAILURE_COUNT:
                    continue
                
                remaining = int(token_data.get(field, -1))
                priority = int(token_data.get("priority", 0))
                
                if remaining == 0:
                    continue
                    
                if remaining == -1:
                    unused_tokens.append((token_key, priority))
                elif remaining > 0:
                    used_tokens.append((token_key, remaining, priority))
            
            if unused_tokens:
                # Sort by priority desc
                unused_tokens.sort(key=lambda x: x[1], reverse=True)
                return unused_tokens[0][0], -1
            
            if used_tokens:
                # Sort by priority desc, then remaining desc
                used_tokens.sort(key=lambda x: (x[2], x[1]), reverse=True)
                return used_tokens[0][0], used_tokens[0][1]
            
            return None, None

        token_data_snapshot = {
            TokenType.NORMAL.value: self.token_data[TokenType.NORMAL.value].copy(),
            TokenType.SUPER.value: self.token_data[TokenType.SUPER.value].copy()
        }
        
        # Decide field
        remaining_field = "remainingQueries"
        if model == "grok-4-heavy":
            remaining_field = "heavyremainingQueries"
            max_token_key, max_remaining = select_best_token(token_data_snapshot[TokenType.SUPER.value], remaining_field)
        else:
            # Try normal first
            max_token_key, max_remaining = select_best_token(token_data_snapshot[TokenType.NORMAL.value], remaining_field)
            if max_token_key is None:
                max_token_key, max_remaining = select_best_token(token_data_snapshot[TokenType.SUPER.value], remaining_field)
        
        if max_token_key is None:
             excluded_info = f" (excluding {len(exclude_tokens)} tried tokens)" if exclude_tokens else ""
             raise GrokApiException(
                f"No available Token for model {model}{excluded_info}",
                "NO_AVAILABLE_TOKEN",
                {"model": model}
             )
        
        status_text = "Unused" if max_remaining == -1 else f"{max_remaining} remaining"
        logger.debug(f"[Token] Allocating Token for model {model} ({status_text})")
        return max_token_key

    async def check_limits(self, auth_token: str, model: str) -> Optional[Dict[str, Any]]:
        try:
             rate_limit_model_name = Models.to_rate_limit(model)
             payload = {"requestKind": "DEFAULT", "modelName": rate_limit_model_name}
             cf_clearance = setting.grok_config.get("cf_clearance", "")
             
             headers = get_dynamic_headers("/rest/rate-limits")
             headers["Cookie"] = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token
             
             proxy_url = setting.grok_config.get("proxy_url", "")
             proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
             
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
                    data = response.json()
                    sso_value = self._extract_sso(auth_token)
                    if sso_value:
                        if model == "grok-4-heavy":
                            await self.update_limits(sso_value, normal=None, heavy=data.get("remainingQueries", -1))
                            logger.info(f"[Token] Updated limits: sso={sso_value[:10]}..., heavy={data.get('remainingQueries', -1)}")
                        else:
                            await self.update_limits(sso_value, normal=data.get("remainingTokens", -1), heavy=None)
                            logger.info(f"[Token] Updated limits: sso={sso_value[:10]}..., basic={data.get('remainingTokens', -1)}")
                    return data
                else:
                    logger.warning(f"[Token] Failed to get rate limits: {response.status_code}")
                    sso_value = self._extract_sso(auth_token)
                    if sso_value:
                        if response.status_code == 401:
                             await self.record_failure(auth_token, 401, "Token expired")
                        elif response.status_code == 403:
                             await self.record_failure(auth_token, 403, "Server blocked (Cloudflare)")
                        else:
                             await self.record_failure(auth_token, response.status_code, f"Error: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"[Token] Rate limit check error: {str(e)}")
            return None

    async def update_limits(self, sso: str, normal: Optional[int] = None, heavy: Optional[int] = None) -> None:
        try:
            found = False
            for token_type in [TokenType.NORMAL.value, TokenType.SUPER.value]:
                if sso in self.token_data[token_type]:
                    if normal is not None:
                        self.token_data[token_type][sso]["remainingQueries"] = normal
                    if heavy is not None:
                        self.token_data[token_type][sso]["heavyremainingQueries"] = heavy
                    found = True
                    break
            
            if found:
                self._mark_dirty()
            else:
                logger.warning(f"[Token] Token {sso[:10]}... not found for limit update")
        except Exception as e:
            logger.error(f"[Token] Update limits error: {str(e)}")

    async def record_failure(self, auth_token: str, status_code: int, error_message: str) -> None:
        try:
            if status_code == STATSIG_INVALID_CODE:
                 logger.warning("[Token] IP blocked (403), please check proxy or IP")
                 return
            
            sso_value = self._extract_sso(auth_token)
            if not sso_value:
                return
            
            token_type, token_data = self._find_token(sso_value)
            if not token_data:
                return

            token_data["failedCount"] = token_data.get("failedCount", 0) + 1
            token_data["lastFailureTime"] = int(time.time() * 1000)
            token_data["lastFailureReason"] = f"{status_code}: {error_message}"

            logger.warning(f"[Token] Failure: {sso_value[:10]}... ({status_code}), Count: {token_data['failedCount']}/{MAX_FAILURE_COUNT}")

            if status_code == TOKEN_INVALID_CODE and token_data["failedCount"] >= MAX_FAILURE_COUNT:
                token_data["status"] = "expired"
                logger.error(f"[Token] Marked expired: {sso_value[:10]}...")
            
            self._mark_dirty()
        except Exception as e:
            logger.error(f"[Token] Record failure error: {str(e)}")

    async def reset_failure(self, auth_token: str) -> None:
        try:
            sso_value = self._extract_sso(auth_token)
            if not sso_value:
                return
            
            token_type, token_data = self._find_token(sso_value)
            if not token_data:
                return

            if token_data.get("failedCount", 0) > 0:
                token_data["failedCount"] = 0
                token_data["lastFailureTime"] = None
                token_data["lastFailureReason"] = None
                self._mark_dirty()
                logger.info(f"[Token] Reset failure count: {sso_value[:10]}...")
        except Exception as e:
            logger.error(f"[Token] Reset failure error: {str(e)}")

    async def mark_token_priority(self, auth_token: str) -> None:
        try:
            sso_value = self._extract_sso(auth_token)
            if not sso_value:
                return

            token_type, token_data = self._find_token(sso_value)
            if not token_data:
                return

            max_priority = 0
            for t_type in [TokenType.NORMAL.value, TokenType.SUPER.value]:
                if t_type in self.token_data:
                    for t_data in self.token_data[t_type].values():
                        p = int(t_data.get("priority", 0))
                        if p > max_priority:
                            max_priority = p

            token_data["priority"] = max_priority + 1
            self._mark_dirty()
            logger.info(f"[Token] Priority increased: {sso_value[:10]}... -> {token_data['priority']}")
        except Exception as e:
            logger.error(f"[Token] Mark priority error: {str(e)}")

token_manager = GrokTokenManager()
