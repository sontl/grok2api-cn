"""Post creation manager - for session creation before video generation"""

import asyncio
import orjson
from typing import Dict, Any, Optional
from curl_cffi.requests import AsyncSession

from app.services.grok.statsig import get_dynamic_headers
from app.core.exception import GrokApiException
from app.core.config import setting
from app.core.logger import logger


# Constants
ENDPOINT = "https://grok.com/rest/media/post/create"
TIMEOUT = 120
BROWSER = "chrome133a"


class PostCreateManager:
    """Grok session creation manager"""

    @staticmethod
    async def create(file_id: str, file_uri: str, auth_token: str) -> Optional[Dict[str, Any]]:
        """Create session record"""
        if not file_id or not file_uri:
            raise GrokApiException("File ID or URI missing", "INVALID_PARAMS")
        
        if not auth_token:
            raise GrokApiException("Authentication token missing", "NO_AUTH_TOKEN")

        try:
            data = {
                "media_url": f"https://assets.grok.com/{file_uri}",
                "media_type": "MEDIA_POST_TYPE_IMAGE"
            }

            cf_clearance = setting.grok_config.get("cf_clearance", "")
            headers = {
                **get_dynamic_headers("/rest/media/post/create"),
                "Cookie": f"{auth_token};{cf_clearance}" if cf_clearance else auth_token
            }
            
            retry_codes = setting.grok_config.get("retry_status_codes", [401, 403, 429])
            MAX_OUTER_RETRY = 1
            
            for outer_retry in range(MAX_OUTER_RETRY + 1):
                max_403_retries = 2
                retry_403_count = 0
                
                while retry_403_count <= max_403_retries:
                    from app.core.proxy_pool import proxy_pool
                    
                    if retry_403_count > 0 and proxy_pool._enabled:
                        logger.info(f"[PostCreate] 403 retry {retry_403_count}/{max_403_retries}, refreshing proxy...")
                        proxy = await proxy_pool.force_refresh()
                    else:
                        proxy = await setting.get_proxy_async("service")
                    
                    proxies = {"http": proxy, "https": proxy} if proxy else None

                    async with AsyncSession() as session:
                        response = await session.post(
                            ENDPOINT, headers=headers, json=data,
                            impersonate=BROWSER, timeout=TIMEOUT, proxies=proxies
                        )

                        if response.status_code == 403 and proxy_pool._enabled:
                            retry_403_count += 1
                            if retry_403_count <= max_403_retries:
                                logger.warning(f"[PostCreate] 403 error, retrying ({retry_403_count}/{max_403_retries})...")
                                await asyncio.sleep(0.5)
                                continue
                            logger.error(f"[PostCreate] 403 error, retried {retry_403_count-1} times, giving up")
                        
                        if response.status_code in retry_codes:
                            if outer_retry < MAX_OUTER_RETRY:
                                delay = (outer_retry + 1) * 0.1
                                logger.warning(f"[PostCreate] {response.status_code} error, outer retry ({outer_retry+1}/{MAX_OUTER_RETRY})")
                                await asyncio.sleep(delay)
                                break
                            else:
                                logger.error(f"[PostCreate] {response.status_code} error, retried {outer_retry} times")
                                raise GrokApiException(f"Creation failed: {response.status_code}", "CREATE_ERROR")

                        if response.status_code == 200:
                            result = response.json()
                            post_id = result.get("post", {}).get("id", "")
                            if outer_retry > 0 or retry_403_count > 0:
                                logger.info(f"[PostCreate] Retry successful!")
                            logger.debug(f"[PostCreate] Session created, ID: {post_id}")
                            return {"post_id": post_id, "file_id": file_id, "file_uri": file_uri, "success": True, "data": result}
                        
                        try:
                            error = response.json()
                            msg = f"Status: {response.status_code}, Details: {error}"
                        except:
                            msg = f"Status: {response.status_code}, Details: {response.text[:200]}"
                        logger.error(f"[PostCreate] Failed: {msg}")
                        raise GrokApiException(f"Creation failed: {msg}", "CREATE_ERROR")

        except GrokApiException:
            raise
        except Exception as e:
            logger.error(f"[PostCreate] Exception: {e}")
            raise GrokApiException(f"Session creation exception: {e}", "CREATE_ERROR") from e
