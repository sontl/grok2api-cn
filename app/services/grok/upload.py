"""Image upload manager"""

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
REQUEST_TIMEOUT = 30
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
    async def upload(image_input: str, auth_token: str) -> str:
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

        # Get authentication token
        if not auth_token:
            raise GrokApiException("Authentication token is missing or empty", "NO_AUTH_TOKEN")

        cf_clearance = setting.grok_config.get("cf_clearance", "")
        cookie = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token

        proxy_url = setting.grok_config.get("proxy_url", "")
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

        # Retry configuration
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Send async request
                async with AsyncSession() as session:
                    response = await session.post(
                        UPLOAD_ENDPOINT,
                        headers={
                            **get_dynamic_headers("/rest/app-chat/upload-file"),
                            "Cookie": cookie,
                        },
                        json=upload_data,
                        impersonate=IMPERSONATE_BROWSER,
                        timeout=REQUEST_TIMEOUT,
                        proxies=proxies,
                    )

                    # Check response
                    if response.status_code == 200:
                        result = response.json()
                        # print the result
                        print(result)
                        file_id = result.get("fileMetadataId", "")
                        file_uri = result.get("fileUri", "")
                        logger.debug(f"[Upload] Image upload successful, file ID: {file_id}")
                        return file_id, file_uri
                    
                    logger.warning(f"[Upload] Upload attempt {attempt + 1}/{max_retries} failed with status: {response.status_code}")

            except Exception as e:
                logger.warning(f"[Upload] Upload attempt {attempt + 1}/{max_retries} failed with error: {e}")
            
            # Wait before retrying, but not after the last attempt
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        logger.error(f"[Upload] All {max_retries} upload attempts failed")
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
        try:
            async with AsyncSession() as session:
                response = await session.get(url, timeout=5)
                response.raise_for_status()

                # Get content type
                content_type = response.headers.get('content-type', DEFAULT_MIME_TYPE)
                if not content_type.startswith('image/'):
                    content_type = DEFAULT_MIME_TYPE

                # Convert to Base64
                image_base64 = base64.b64encode(response.content).decode('utf-8')
                return image_base64, content_type
        except Exception as e:
            logger.warning(f"[Upload] Image download failed: {e}")
            return "", ""

    @staticmethod
    def _get_info(image_data: str, mime_type: Optional[str] = None) -> Tuple[str, str]:
        """Get image filename and MIME type"""
        # mime_type has value, use directly
        if mime_type:
            extension = mime_type.split("/")[1] if "/" in mime_type else DEFAULT_EXTENSION
            file_name = f"image.{extension}"
            return file_name, mime_type

        # mime_type has no value, use default
        mime_type = DEFAULT_MIME_TYPE
        extension = DEFAULT_EXTENSION

        # Extract MIME type from Base64 data
        if "data:image" in image_data:
            match = re.search(r"data:([a-zA-Z0-9]+/[a-zA-Z0-9-.+]+);base64,", image_data)
            if match:
                mime_type = match.group(1)
                extension = mime_type.split("/")[1]

        file_name = f"image.{extension}"
        return file_name, mime_type