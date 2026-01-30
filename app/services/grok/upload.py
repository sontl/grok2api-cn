"""Image upload manager"""

import asyncio
import base64
import re
import asyncio
from typing import Tuple, Optional
from urllib.parse import urlparse
from curl_cffi.requests import AsyncSession

from app.services.grok.statsig import get_dynamic_headers
from app.core.exception import GrokApiException
from app.core.config import setting
from app.core.logger import logger


# Constant definitions
UPLOAD_ENDPOINT = "https://grok.com/rest/app-chat/upload-file"
REQUEST_TIMEOUT = 120
IMPERSONATE_BROWSER = "chrome133a"
DEFAULT_MIME_TYPE = "image/jpeg"
DEFAULT_EXTENSION = "jpg"


class ImageUploadManager:
    """
    Grok image upload manager

    Provides image upload functionality, supporting:
    - Base64 format image upload
    - URL image download and upload
    - Multiple image format support
    """

    @staticmethod
    async def upload(image_input: str, auth_token: str) -> Tuple[str, str, bool]:
        """Upload image to Grok, supporting Base64 or URL
        
        Returns:
            Tuple of (file_id, file_uri, rate_limited)
            - file_id: The uploaded file ID, empty string on failure
            - file_uri: The uploaded file URI, empty string on failure
            - rate_limited: True if upload failed due to 403 rate limiting (token should be rotated)
        """
        if ImageUploadManager._is_url(image_input):
            # Download URL image
            image_buffer, mime_type = await ImageUploadManager._download(image_input)

            # Get image info
            file_name, _ = ImageUploadManager._get_info("", mime_type)

        else:
            # Process base64 data
            image_buffer = image_input.split(",")[1] if "data:image" in image_input else image_input
            # Get image info
            file_name, mime_type = ImageUploadManager._get_info(image_input)

        # Build upload data
        upload_data = {
            "fileName": file_name,
            "fileMimeType": mime_type,
            "content": image_buffer,
        }

        if not auth_token:
            raise GrokApiException("Authentication token is missing", "NO_AUTH_TOKEN")

        try:
            # Outer retry: configurable status codes (401/429 etc.)
            retry_codes = setting.grok_config.get("retry_status_codes", [401, 403, 429])
            MAX_OUTER_RETRY = 1
            
            for outer_retry in range(MAX_OUTER_RETRY + 1):  # +1 to ensure 3 actual retries
                try:
                    # Inner retry: 403 proxy pool retry
                    max_403_retries = 2
                    retry_403_count = 0
                    
                    while retry_403_count <= max_403_retries:
                        # Request configuration
                        cf = setting.grok_config.get("cf_clearance", "")
                        headers = {
                            **get_dynamic_headers("/rest/app-chat/upload-file"),
                            "Cookie": f"{auth_token};{cf}" if cf else auth_token,
                        }
                        
                        # Async get proxy (supports proxy pool)
                        from app.core.proxy_pool import proxy_pool
                        
                        # If 403 retry and using proxy pool, force refresh proxy
                        if retry_403_count > 0 and proxy_pool._enabled:
                            logger.info(f"[Upload] 403 retry {retry_403_count}/{max_403_retries}, refreshing proxy...")
                            proxy = await proxy_pool.force_refresh()
                        else:
                            proxy = await setting.get_proxy_async("service")
                        
                        proxies = {"http": proxy, "https": proxy} if proxy else None

                        # Upload
                        async with AsyncSession() as session:
                            response = await session.post(
                                UPLOAD_ENDPOINT,
                                headers=headers,
                                json=upload_data,
                                impersonate=IMPERSONATE_BROWSER,
                                timeout=REQUEST_TIMEOUT,
                                proxies=proxies,
                            )

                            # Handle 403 error - this is rate limiting, not just proxy issue
                            if response.status_code == 403:
                                if proxy_pool._enabled:
                                    # With proxy pool, try different proxies
                                    retry_403_count += 1
                                    
                                    if retry_403_count <= max_403_retries:
                                        logger.warning(f"[Upload] Encountered 403 error, retrying with new proxy ({retry_403_count}/{max_403_retries})...")
                                        await asyncio.sleep(0.5)
                                        continue
                                    
                                    # All proxy retries failed - rate limited
                                    logger.error(f"[Upload] 403 error, retried {retry_403_count-1} times with proxies, giving up (rate limited)")
                                    return "", "", True
                                else:
                                    # No proxy pool - 403 means token is rate limited
                                    logger.error(f"[Upload] 403 error (rate limited), no proxy pool to retry")
                                    return "", "", True
                            
                            # Handle 429 error - always rate limiting
                            if response.status_code == 429:
                                logger.error(f"[Upload] 429 error (rate limited)")
                                return "", "", True
                            
                            # Handle 401 error - token expired, do outer retry with same token once
                            if response.status_code == 401:
                                if outer_retry < MAX_OUTER_RETRY:
                                    delay = (outer_retry + 1) * 0.1
                                    logger.warning(f"[Upload] Encountered 401 error, outer retry ({outer_retry+1}/{MAX_OUTER_RETRY}), waiting {delay}s...")
                                    await asyncio.sleep(delay)
                                    break  # Break out of inner loop, enter outer retry
                                else:
                                    logger.error(f"[Upload] 401 error, retried {outer_retry} times, giving up")
                                    return "", "", False  # 401 is token issue, not rate limit
                            
                            if response.status_code == 200:
                                result = response.json()
                                file_id = result.get("fileMetadataId", "")
                                file_uri = result.get("fileUri", "")
                                
                                if outer_retry > 0 or retry_403_count > 0:
                                    logger.info(f"[Upload] Retry successful!")
                                
                                logger.debug(f"[Upload] Image upload successful, file ID: {file_id}")
                                return file_id, file_uri, False
                            
                            # Other errors return directly (not rate limited)
                            logger.error(f"[Upload] Failed, status code: {response.status_code}")
                            return "", "", False
                    
                    # Inner loop ended normally (not break), means 403 retry all failed - rate limited
                    return "", "", True
                
                except Exception as e:
                    if outer_retry < MAX_OUTER_RETRY - 1:
                        logger.warning(f"[Upload] Exception: {e}, outer retry ({outer_retry+1}/{MAX_OUTER_RETRY})...")
                        await asyncio.sleep(0.5)
                        continue
                    
                    logger.warning(f"[Upload] Failed: {e}")
                    return "", "", False
            
            return "", "", False

        except Exception as e:
            logger.warning(f"[Upload] Failed: {e}")
            return "", "", False


    @staticmethod
    def _is_url(image_input: str) -> bool:
        """Check if input is a valid URL"""
        try:
            result = urlparse(image_input)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception as e:
            logger.warning(f"[Upload] URL parsing failed: {e}")
            return False

    @staticmethod
    async def _download(url: str) -> Tuple[str, str]:
        """Download image and convert to Base64"""
        max_retries = 3
        retry_delay = 2
        timeout = 30

        for attempt in range(max_retries):
            try:
                async with AsyncSession() as session:
                    response = await session.get(url, timeout=timeout)
                    response.raise_for_status()

                    # Get content type
                    content_type = response.headers.get('content-type', DEFAULT_MIME_TYPE)
                    if not content_type.startswith('image/'):
                        content_type = DEFAULT_MIME_TYPE

                    # Convert to Base64
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
                    return image_base64, content_type
            except Exception as e:
                logger.warning(f"[Upload] Image download attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)

        logger.error(f"[Upload] All {max_retries} image download attempts failed for url: {url}")
        return "", ""

    @staticmethod
    def _get_info(image_data: str, mime_type: Optional[str] = None) -> Tuple[str, str]:
        """Get image filename and MIME type"""
        if mime_type:
            ext = mime_type.split("/")[1] if "/" in mime_type else DEFAULT_EXTENSION
            return f"image.{ext}", mime_type

        # mime_type has no value, use default
        mime_type = DEFAULT_MIME_TYPE
        extension = DEFAULT_EXTENSION

        # Extract MIME type from Base64 data
        if "data:image" in image_data:
            if match := re.search(r"data:([a-zA-Z0-9]+/[a-zA-Z0-9-.+]+);base64,", image_data):
                mime_type = match.group(1)
                extension = mime_type.split("/")[1]

        return f"image.{extension}", mime_type