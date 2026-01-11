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
    async def upload(image_input: str, auth_token: str) -> Tuple[str, str]:
        """Upload image to Grok, supporting Base64 or URL"""
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
            retry_codes = setting.grok_config.get("retry_status_codes", [401, 429])
            MAX_OUTER_RETRY = 3
            
            for outer_retry in range(MAX_OUTER_RETRY + 1):  # +1 to ensure 3 actual retries
                try:
                    # Inner retry: 403 proxy pool retry
                    max_403_retries = 5
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

                            # Inner 403 retry: only trigger when proxy pool exists
                            if response.status_code == 403 and proxy_pool._enabled:
                                retry_403_count += 1
                                
                                if retry_403_count <= max_403_retries:
                                    logger.warning(f"[Upload] Encountered 403 error, retrying ({retry_403_count}/{max_403_retries})...")
                                    await asyncio.sleep(0.5)
                                    continue
                                
                                # Inner retry all failed
                                logger.error(f"[Upload] 403 error, retried {retry_403_count-1} times, giving up")
                            
                            # Check configurable status code errors - outer retry
                            if response.status_code in retry_codes:
                                if outer_retry < MAX_OUTER_RETRY:
                                    delay = (outer_retry + 1) * 0.1  # Progressive delay: 0.1s, 0.2s, 0.3s
                                    logger.warning(f"[Upload] Encountered {response.status_code} error, outer retry ({outer_retry+1}/{MAX_OUTER_RETRY}), waiting {delay}s...")
                                    await asyncio.sleep(delay)
                                    break  # Break out of inner loop, enter outer retry
                                else:
                                    logger.error(f"[Upload] {response.status_code} error, retried {outer_retry} times, giving up")
                                    return "", ""
                            
                            if response.status_code == 200:
                                result = response.json()
                                file_id = result.get("fileMetadataId", "")
                                file_uri = result.get("fileUri", "")
                                
                                if outer_retry > 0 or retry_403_count > 0:
                                    logger.info(f"[Upload] Retry successful!")
                                
                                logger.debug(f"[Upload] Image upload successful, file ID: {file_id}")
                                return file_id, file_uri
                            
                            # Other errors return directly
                            logger.error(f"[Upload] Failed, status code: {response.status_code}")
                            return "", ""
                    
                    # Inner loop ended normally (not break), means 403 retry all failed
                    return "", ""
                
                except Exception as e:
                    if outer_retry < MAX_OUTER_RETRY - 1:
                        logger.warning(f"[Upload] Exception: {e}, outer retry ({outer_retry+1}/{MAX_OUTER_RETRY})...")
                        await asyncio.sleep(0.5)
                        continue
                    
                    logger.warning(f"[Upload] Failed: {e}")
                    return "", ""
            
            return "", ""

        except Exception as e:
            logger.warning(f"[Upload] Failed: {e}")
            return "", ""


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