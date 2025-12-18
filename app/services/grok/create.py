"""Post creation manager - for session creation before video generation"""

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
        """
        Create session record

        Args:
            file_id: File ID after upload
            file_uri: File URI after upload
            auth_token: Authentication token

        Returns:
            Created session information, including post_id, etc.
        """
        if not file_id or not file_uri:
            raise GrokApiException("File ID or URI missing", "INVALID_PARAMS")
        
        if not auth_token:
            raise GrokApiException("Authentication token missing", "NO_AUTH_TOKEN")

        try:
            # Build request
            data = {
                "media_url": f"https://assets.grok.com/{file_uri}",
                "media_type": "MEDIA_POST_TYPE_IMAGE"
            }

            cf_clearance = setting.grok_config.get("cf_clearance", "")
            headers = {
                **get_dynamic_headers("/rest/media/post/create"),
                "Cookie": f"{auth_token};{cf_clearance}" if cf_clearance else auth_token
            }

            proxy_url = setting.grok_config.get("proxy_url", "")
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

            # Send async request
            async with AsyncSession() as session:
                response = await session.post(
                    ENDPOINT,
                    headers=headers,
                    json=data,
                    impersonate=BROWSER,
                    timeout=TIMEOUT,
                    proxies=proxies
                )

                if response.status_code == 200:
                    result = response.json()
                    post_id = result.get("post", {}).get("id", "")
                    logger.debug(f"[PostCreate] Session creation successful, session ID: {post_id}")
                    return {
                        "post_id": post_id,
                        "file_id": file_id,
                        "file_uri": file_uri,
                        "success": True,
                        "data": result
                    }
                else:
                    error_msg = f"Status code: {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg = f"{error_msg}, Details: {error_data}"
                    except:
                        error_msg = f"{error_msg}, Details: {response.text[:200]}"

                    logger.error(f"[PostCreate] Session creation failed: {error_msg}")
                    raise GrokApiException(f"Session creation failed: {error_msg}", "CREATE_ERROR")

        except GrokApiException:
            raise
        except Exception as e:
            logger.error(f"[PostCreate] Session creation exception: {e}")
            raise GrokApiException(f"Session creation exception: {e}", "CREATE_ERROR") from e
