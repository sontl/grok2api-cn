"""Grok API Client Module"""

import asyncio
import orjson
import json
from typing import Dict, List, Tuple, Any, Optional
from curl_cffi import requests as curl_requests

from app.core.config import setting
from app.core.logger import logger
from app.models.grok_models import Models
from app.services.grok.processer import GrokResponseProcessor
from app.services.grok.statsig import get_dynamic_headers
from app.services.grok.token import token_manager
from app.services.grok.upload import ImageUploadManager
from app.services.grok.create import PostCreateManager
from app.services.grok.upscale import VideoUpscaleManager
from app.core.exception import GrokApiException


# Constant definitions
API_ENDPOINT = "https://grok.com/rest/app-chat/conversations/new"
REQUEST_TIMEOUT = 120
IMPERSONATE_BROWSER = "chrome133a"
MAX_RETRY = 10
MAX_UPLOADS = 20


class GrokClient:
    """Grok API Client"""

    _upload_sem = None  # Lazy initialization

    @staticmethod
    def _get_upload_semaphore():
        """Get upload semaphore (dynamic configuration)"""
        if GrokClient._upload_sem is None:
            max_concurrency = setting.global_config.get("max_upload_concurrency", MAX_UPLOADS)
            GrokClient._upload_sem = asyncio.Semaphore(max_concurrency)
            logger.debug(f"[Client] Initialized upload concurrency limit: {max_concurrency}")
        return GrokClient._upload_sem

    @staticmethod
    async def openai_to_grok(openai_request: dict):
        """Convert OpenAI request to Grok request and process response"""
        model = openai_request["model"]
        messages = openai_request["messages"]
        stream = openai_request.get("stream", False)
        auto_upscale = openai_request.get("auto_upscale")

        # Extract message content and image URLs
        content, image_urls = GrokClient._extract_content(messages)
        model_name, model_mode = Models.to_grok(model)
        is_video_model = Models.get_model_info(model).get("is_video_model", False)

        # Video model special handling
        if is_video_model:
            if len(image_urls) > 1:
                logger.warning(f"[Client] Video model only allows one image, currently has {len(image_urls)} images, using only the first one")
                image_urls = image_urls[:1]

        # Retry logic
        return await GrokClient._try(model, content, image_urls, model_name, model_mode, is_video_model, stream, auto_upscale)

    @staticmethod
    async def upscale_video(video_id: str, model: str = "grok-3"):
        """Upscale video to HD"""
        # Get available token
        auth_token = token_manager.get_token(model)
        if not auth_token:
            raise GrokApiException("No available token found", "NO_AVAILABLE_TOKEN")
            
        return await VideoUpscaleManager.upscale(video_id, auth_token)

    @staticmethod
    async def _try(model: str, content: str, image_urls: List[str], model_name: str, model_mode: str, is_video: bool, stream: bool, auto_upscale: bool = None):
        """Request execution with retry and token rotation"""
        last_err = None
        tried_tokens = []  # Track SSO tokens that have been tried
        
        # Get all available tokens count for better retry logic
        all_tokens = token_manager.get_tokens()
        total_available = len(all_tokens.get("normal", {})) + len(all_tokens.get("super", {}))
        max_attempts = min(MAX_RETRY, total_available) if total_available > 0 else MAX_RETRY

        for i in range(max_attempts):
            auth_token = None
            try:
                # Get token, excluding already-tried ones
                if i == 0:
                    # First attempt - use normal token selection
                    auth_token = token_manager.get_token(model)
                else:
                    # Subsequent attempts - exclude tried tokens
                    auth_token = token_manager.get_next_token(model, tried_tokens)
                    if not auth_token:
                        logger.error(f"[Client] All available tokens exhausted ({len(tried_tokens)} tokens tried)")
                        break
                
                # Track this token
                sso_value = token_manager._extract_sso(auth_token)
                if sso_value and sso_value not in tried_tokens:
                    tried_tokens.append(sso_value)
                logger.debug(f"[Client] Using token {auth_token[:250]}...")
                
                # Upload images
                imgs, uris = await GrokClient._upload_imgs(image_urls, auth_token)

                # Video model - Create session
                post_id = None
                if is_video and imgs and uris:
                    try:
                        create_result = await PostCreateManager.create(imgs[0], uris[0], auth_token)
                        if create_result and create_result.get("success"):
                            post_id = create_result.get("post_id")
                        else:
                            logger.warning(f"[Client] Session creation failed, continuing with original flow")
                    except Exception as e:
                        logger.warning(f"[Client] Session creation exception: {e}, continuing with original flow")

                # Build and send request
                payload = GrokClient._build_payload(content, model_name, model_mode, imgs, uris, is_video, post_id)
                result = await GrokClient._send_request(payload, auth_token, model, stream, post_id, auto_upscale)
                
                # Success! Mark this token as prioritized
                asyncio.create_task(token_manager.mark_token_priority(auth_token))
                logger.info(f"[Client] Request succeeded with token (attempt {i+1}/{max_attempts})")
                
                return result

            except GrokApiException as e:
                last_err = e
                # Check if it's retryable
                if e.error_code not in ["HTTP_ERROR", "NO_AVAILABLE_TOKEN"]:
                    raise

                status = e.context.get("status") if e.context else None
                retry_codes = setting.grok_config.get("retry_status_codes", [401, 403, 429])
                
                if status not in retry_codes:
                    raise

        raise last_err if last_err else GrokApiException("Request failed after trying all available tokens", "REQUEST_ERROR")

    @staticmethod
    def _extract_content(messages: List[Dict]) -> Tuple[str, List[str]]:
        """Extract message content and image URLs"""
        content_parts = []
        image_urls = []

        for msg in messages:
            msg_content = msg.get("content", "")

            # Handle complex message format
            if isinstance(msg_content, list):
                for item in msg_content:
                    item_type = item.get("type")
                    if item_type == "text":
                        content_parts.append(item.get("text", ""))
                    elif item_type == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url:
                            image_urls.append(url)
            # Handle plain text messages
            else:
                content_parts.append(msg_content)
        
        return "".join(content_parts), image_urls

    @staticmethod
    async def _upload_imgs(image_urls: List[str], auth_token: str) -> Tuple[List[str], List[str]]:
        """Upload images and return attachment ID list"""
        if not image_urls:
            return [], []

        async def upload_limited(url):
            async with GrokClient._get_upload_semaphore():
                return await ImageUploadManager.upload(url, auth_token)

        # Upload all images concurrently with semaphore
        results = await asyncio.gather(*[upload_limited(url) for url in image_urls], return_exceptions=True)
        
        image_attachments = []
        image_uris = []

        for url, result in zip(image_urls, results):
            if isinstance(result, Exception):
                logger.warning(f"[Client] Image upload failed: {url}, Error: {result}")
            elif isinstance(result, tuple) and len(result) == 2:
                file_id, file_uri = result
                if file_id:
                    image_attachments.append(file_id)
                    image_uris.append(file_uri)

        return image_attachments, image_uris

    @staticmethod
    def _build_payload(content: str, model_name: str, model_mode: str, image_attachments: List[str], image_uris: List[str], is_video_model: bool = False, post_id: str = None) -> Dict[str, Any]:
        """Build Grok API request payload"""
        payload = {
            "temporary": setting.grok_config.get("temporary", True),
            "modelName": model_name,
            "message": content,
            "fileAttachments": image_attachments,
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "sendFinalMetadata": True,
            "isReasoning": False,
            "webpageUrls": [],
            "disableTextFollowUps": True,
            "responseMetadata": {"requestModelDetails": {"modelId": model_name}},
            "disableMemory": False,  # Note: upstream says False, Head said False.
            "forceSideBySide": False,
            "modelMode": model_mode,
            "isAsyncChat": False
        }

        # Video model configuration
        if is_video_model and image_uris:
            image_url = image_uris[0]

            # Build URL message
            if post_id:
                image_message = f"https://grok.com/imagine/{post_id}  {content} --mode=custom"
            else:
                image_message = f"https://assets.grok.com/post/{image_url}  {content} --mode=custom"

            payload = {
                "temporary": True,
                "modelName": "grok-3",
                "message": image_message,
                "fileAttachments": image_attachments,
                "toolOverrides": {"videoGen": True}
            }

        return payload

    @staticmethod
    async def _send_request(payload: dict, auth_token: str, model: str, stream: bool, post_id: str = None, auto_upscale: bool = None):
        """Send HTTP request to Grok API
        
        Note: This function only retries 403 errors (proxy issues).
        For 429/401 errors (token rate limit/auth), it immediately fails
        so the parent _try() function can select a different token.
        """
        # Validate authentication token
        if not auth_token:
            raise GrokApiException("Authentication token missing", "NO_AUTH_TOKEN")

        # Only retry 403 errors (proxy issues) - for 429/401, let _try() pick a different token
        max_403_retries = 5
        retry_403_count = 0
        
        while retry_403_count <= max_403_retries:
            try:
                # Build request
                headers = GrokClient._build_headers(auth_token)
                if model == "grok-imagine-0.9":
                    file_attachments = payload.get("fileAttachments", [])
                    ref_id = post_id or (file_attachments[0] if file_attachments else "")
                    if ref_id:
                        headers["Referer"] = f"https://grok.com/imagine/{ref_id}"
                
                # Async get proxy
                from app.core.proxy_pool import proxy_pool
                
                # If 403 retry and using proxy pool, force refresh proxy
                if retry_403_count > 0 and proxy_pool._enabled:
                    logger.info(f"[Client] 403 retry {retry_403_count}/{max_403_retries}, refreshing proxy...")
                    proxy = await proxy_pool.force_refresh()
                else:
                    proxy = await setting.get_proxy_async("service")
                
                proxies = {"http": proxy, "https": proxy} if proxy else None
                
                # Execute request
                response = await asyncio.to_thread(
                    curl_requests.post,
                    API_ENDPOINT,
                    headers=headers,
                    data=orjson.dumps(payload),
                    impersonate=IMPERSONATE_BROWSER,
                    timeout=REQUEST_TIMEOUT,
                    stream=True,
                    proxies=proxies
                )
                
                # 403 retry: only trigger when proxy pool exists
                if response.status_code == 403 and proxy_pool._enabled:
                    retry_403_count += 1
                    
                    if retry_403_count <= max_403_retries:
                        logger.warning(f"[Client] Encountered 403 error, retrying with new proxy ({retry_403_count}/{max_403_retries})...")
                        await asyncio.sleep(0.5)
                        continue
                    
                    # 403 retry all failed
                    logger.error(f"[Client] 403 error, retried {retry_403_count-1} times with different proxies, giving up")
                
                # For 429/401 and other errors, immediately fail - let _try() pick a different token
                if response.status_code != 200:
                    GrokClient._handle_error(response, auth_token)
                
                # Success - reset failure count
                asyncio.create_task(token_manager.reset_failure(auth_token))
                
                # If 403 retry succeeded, log it
                if retry_403_count > 0:
                    logger.info(f"[Client] 403 retry successful with new proxy!")
                
                # Process response
                result = (GrokResponseProcessor.process_stream(response, auth_token, auto_upscale) if stream 
                         else await GrokResponseProcessor.process_normal(response, auth_token, model, auto_upscale))
                
                asyncio.create_task(GrokClient._update_rate_limits(auth_token, model))
                return result
                
            except curl_requests.RequestsError as e:
                logger.error(f"[Client] Network error: {e}")
                raise GrokApiException(f"Network error: {e}", "NETWORK_ERROR") from e
            except GrokApiException:
                # Re-raise GrokApiException (let _try() handle token rotation)
                raise
            except Exception as e:
                logger.error(f"[Client] Request error: {e}")
                raise GrokApiException(f"Request error: {e}", "REQUEST_ERROR") from e
        
        # 403 retries exhausted
        raise GrokApiException("Request failed: 403 error after exhausting proxy retries", "PROXY_ERROR")

    @staticmethod
    def _build_headers(auth_token: str) -> Dict[str, str]:
        """Build request headers"""
        headers = get_dynamic_headers("/rest/app-chat/conversations/new")

        # Build Cookie
        cf_clearance = setting.grok_config.get("cf_clearance", "")
        headers["Cookie"] = f"{auth_token};{cf_clearance}" if cf_clearance else auth_token

        return headers

    @staticmethod
    def _handle_error(response, auth_token: str):
        """Handle error response"""
        # Handle 403 error
        if response.status_code == 403:
            error_message = "Server IP is blocked, please try 1. Change server IP 2. Use proxy IP 3. Log in to Grok.com on the server, find CF value in F12 after passing shield, and enter it in backend settings"
            error_data = {"cf_blocked": True, "status": 403}
            logger.warning(f"[Client] {error_message}")
        else:
            # Try to parse JSON for other errors
            try:
                error_data = response.json()
                error_message = str(error_data)
            except Exception as e:
                error_data = response.text
                error_message = error_data[:200] if error_data else str(e)

        # Record token failure
        asyncio.create_task(token_manager.record_failure(auth_token, response.status_code, error_message))

        raise GrokApiException(
            f"Request failed: {response.status_code} - {error_message}",
            "HTTP_ERROR",
            {"status": response.status_code, "data": error_data}
        )

    @staticmethod
    async def _update_rate_limits(auth_token: str, model: str):
        """Asynchronously update rate limit information"""
        try:
            await token_manager.check_limits(auth_token, model)
        except Exception as e:
            logger.error(f"[Client] Failed to update rate limits: {e}")
