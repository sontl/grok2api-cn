"""Grok API Client Module"""

import asyncio
import json
from typing import Dict, List, Tuple, Any

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
GROK_API_ENDPOINT = "https://grok.com/rest/app-chat/conversations/new"
REQUEST_TIMEOUT = 120
IMPERSONATE_BROWSER = "chrome133a"
MAX_RETRY = 10  # Maximum retry attempts


class GrokClient:
    """Grok API Client"""

    @staticmethod
    async def openai_to_grok(openai_request: dict):
        """Convert OpenAI request to Grok request and process response"""
        model = openai_request["model"]
        messages = openai_request["messages"]
        stream = openai_request.get("stream", False)

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
        return await GrokClient._try(model, content, image_urls, model_name, model_mode, is_video_model, stream)

    @staticmethod
    async def upscale_video(video_id: str, model: str = "grok-3"):
        """Upscale video to HD"""
        # Get available token
        auth_token = token_manager.get_token(model)
        if not auth_token:
            raise GrokApiException("No available token found", "NO_AVAILABLE_TOKEN")
            
        return await VideoUpscaleManager.upscale(video_id, auth_token)

    @staticmethod
    async def _try(model: str, content: str, image_urls: List[str], model_name: str, model_mode: str, is_video: bool, stream: bool):
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
                result = await GrokClient._send_request(payload, auth_token, model, stream, post_id)
                
                # Success! Mark this token as prioritized
                asyncio.create_task(token_manager.mark_token_priority(auth_token))
                logger.info(f"[Client] Request succeeded with token (attempt {i+1}/{max_attempts})")
                
                return result

            except GrokApiException as e:
                last_err = e
                
                # Check if it's a retryable error
                if e.error_code not in ["HTTP_ERROR", "NO_AVAILABLE_TOKEN"]:
                    raise

                # Check if it's a retryable status code
                status = e.details.get("status") if e.details else None
                
                # For 429 errors, try next token
                if status == 429:
                    if i < max_attempts - 1:
                        logger.warning(f"[Client] Token rate limited (429), trying next token (attempt {i+1}/{max_attempts})")
                        await asyncio.sleep(0.5)  # Brief delay
                        continue
                    else:
                        logger.error(f"[Client] All tokens rate limited after {max_attempts} attempts")
                        break
                
                # For 401 errors, try next token
                elif status == 401:
                    if i < max_attempts - 1:
                        logger.warning(f"[Client] Token authentication failed (401), trying next token (attempt {i+1}/{max_attempts})")
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        logger.error(f"[Client] All tokens failed authentication after {max_attempts} attempts")
                        break
                
                # Other errors are not retryable
                else:
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
        image_attachments = []
        image_uris = []
        # Upload all images concurrently
        tasks = [ImageUploadManager.upload(url, auth_token) for url in image_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, (file_id, file_uri) in zip(image_urls, results):
            if isinstance(file_id, Exception):
                logger.warning(f"[Client] Image upload failed: {url}, Error: {file_id}")
            elif file_id:
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
            "disableMemory": False,
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
    async def _send_request(payload: dict, auth_token: str, model: str, stream: bool, post_id: str = None):
        """Send HTTP request to Grok API"""
        # Validate authentication token
        if not auth_token:
            raise GrokApiException("Authentication token missing", "NO_AUTH_TOKEN")

        try:
            # Build request headers
            headers = GrokClient._build_headers(auth_token)
            if model == "grok-imagine-0.9":
                # Pass in session ID
                file_attachments = payload.get("fileAttachments", [])
                referer_id = post_id if post_id else (file_attachments[0] if file_attachments else "")
                if referer_id:
                    headers["Referer"] = f"https://grok.com/imagine/{referer_id}"

            # Use service proxy
            proxy_url = setting.get_service_proxy()
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None

            # Build request parameters
            request_kwargs = {
                "headers": headers,
                "data": json.dumps(payload),
                "impersonate": IMPERSONATE_BROWSER,
                "timeout": REQUEST_TIMEOUT,
                "stream": True,
                "proxies": proxies
            }

            # Execute synchronous HTTP request in thread pool to avoid blocking event loop
            response = await asyncio.to_thread(
                curl_requests.post,
                GROK_API_ENDPOINT,
                **request_kwargs
            )

            # Handle non-success response
            if response.status_code != 200:
                GrokClient._handle_error(response, auth_token)

            # Request successful, reset failure count
            asyncio.create_task(token_manager.reset_failure(auth_token))

            # Process and return response
            return await GrokClient._process_response(response, auth_token, model, stream)

        except curl_requests.RequestsError as e:
            logger.error(f"[Client] Network request error: {e}")
            raise GrokApiException(f"Network error: {e}", "NETWORK_ERROR") from e
        except json.JSONDecodeError as e:
            logger.error(f"[Client] JSON parsing error: {e}")
            raise GrokApiException(f"JSON parsing error: {e}", "JSON_ERROR") from e
        except GrokApiException:
            raise
        except Exception as e:
            # Handle potential class mismatch for GrokApiException
            if type(e).__name__ == 'GrokApiException':
                logger.warning(f"[Client] Caught GrokApiException via generic handler. Re-raising. Type: {type(e)}")
                raise GrokApiException(
                    message=getattr(e, 'message', str(e)),
                    error_code=getattr(e, 'error_code', 'UNKNOWN'),
                    details=getattr(e, 'details', {})
                )
            
            logger.error(f"[Client] Unknown request error: {type(e).__name__}: {e}")
            raise GrokApiException(f"Request processing error: {e}", "REQUEST_ERROR") from e

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
    async def _process_response(response, auth_token: str, model: str, stream: bool):
        """Process API response"""
        if stream:
            result = GrokResponseProcessor.process_stream(response, auth_token)
            asyncio.create_task(GrokClient._update_rate_limits(auth_token, model))
        else:
            result = await GrokResponseProcessor.process_normal(response, auth_token, model)
            asyncio.create_task(GrokClient._update_rate_limits(auth_token, model))

        return result

    @staticmethod
    async def _update_rate_limits(auth_token: str, model: str):
        """Asynchronously update rate limit information"""
        try:
            await token_manager.check_limits(auth_token, model)
        except Exception as e:
            logger.error(f"[Client] Failed to update rate limits: {e}")