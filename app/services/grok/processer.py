"""Grok API Response Processor Module"""

import orjson
import uuid
import time
import os
import asyncio
from typing import AsyncGenerator, Tuple, Optional

from app.services.grok.upscale import VideoUpscaleManager

from app.core.config import setting
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.models.openai_schema import (
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionMessage,
    OpenAIChatCompletionChunkResponse,
    OpenAIChatCompletionChunkChoice,
    OpenAIChatCompletionChunkMessage
)
from app.services.grok.cache import image_cache_service, video_cache_service


class StreamTimeoutManager:
    """Streaming response timeout manager"""
    
    def __init__(self, chunk_timeout: int = 120, first_timeout: int = 30, total_timeout: int = 600):
        self.chunk_timeout = chunk_timeout
        self.first_timeout = first_timeout
        self.total_timeout = total_timeout
        self.start_time = asyncio.get_event_loop().time()
        self.last_chunk_time = self.start_time
        self.first_received = False
    
    def check_timeout(self) -> Tuple[bool, str]:
        """Check timeout"""
        now = asyncio.get_event_loop().time()
        
        if not self.first_received and now - self.start_time > self.first_timeout:
            return True, f"First response timeout ({self.first_timeout}s)"
        
        if self.total_timeout > 0 and now - self.start_time > self.total_timeout:
            return True, f"Total timeout ({self.total_timeout}s)"
        
        if self.first_received and now - self.last_chunk_time > self.chunk_timeout:
            return True, f"Chunk interval timeout ({self.chunk_timeout}s)"
        
        return False, ""
    
    def mark_received(self):
        """Mark chunk received"""
        self.last_chunk_time = asyncio.get_event_loop().time()
        self.first_received = True
    
    def duration(self) -> float:
        """Get total duration"""
        return asyncio.get_event_loop().time() - self.start_time


class GrokResponseProcessor:
    """Grok API Response Processor"""

    @staticmethod
    async def process_normal(response, auth_token: str, model: str = None, auto_upscale: bool = None) -> OpenAIChatCompletionResponse:
        """Process non-streaming response"""
        response_closed = False
        try:
            for chunk in response.iter_lines():
                if not chunk:
                    continue

                data = orjson.loads(chunk)

                # Error check
                if error := data.get("error"):
                    raise GrokApiException(
                        f"API error: {error.get('message', 'Unknown error')}",
                        "API_ERROR",
                        {"code": error.get("code")}
                    )

                grok_resp = data.get("result", {}).get("response", {})
                
                # Video response
                if video_resp := grok_resp.get("streamingVideoGenerationResponse"):
                    if video_url := video_resp.get("videoUrl"):
                        video_id = video_resp.get("videoId")
                        content = await GrokResponseProcessor._build_video_content(video_url, video_id, auth_token, auto_upscale)
                        result = GrokResponseProcessor._build_response(content, model or "grok-imagine-0.9")
                        response_closed = True
                        response.close()
                        return result

                # Model response
                model_response = grok_resp.get("modelResponse")
                if not model_response:
                    continue

                if error_msg := model_response.get("error"):
                    raise GrokApiException(f"Model error: {error_msg}", "MODEL_ERROR")

                # Build content
                content = model_response.get("message", "")
                model_name = model_response.get("model")

                # Handle images
                if images := model_response.get("generatedImageUrls"):
                    content = await GrokResponseProcessor._append_images(content, images, auth_token)

                result = GrokResponseProcessor._build_response(content, model_name)
                response_closed = True
                response.close()
                return result

            raise GrokApiException("No response data", "NO_RESPONSE")

        except orjson.JSONDecodeError as e:
            logger.error(f"[Processor] JSON parsing failed: {e}")
            raise GrokApiException(f"JSON parsing failed: {e}", "JSON_ERROR") from e
        except Exception as e:
            logger.error(f"[Processor] Processing error: {type(e).__name__}: {e}")
            raise GrokApiException(f"Response processing error: {e}", "PROCESS_ERROR") from e
        finally:
            if not response_closed and hasattr(response, 'close'):
                try:
                    response.close()
                except Exception as e:
                    logger.warning(f"[Processor] Error closing response object: {e}")

    @staticmethod
    async def process_stream(response, auth_token: str, auto_upscale: bool = None) -> AsyncGenerator[str, None]:
        """Process streaming response"""
        # State variables
        is_image = False
        is_thinking = False
        thinking_finished = False
        model = None
        filtered_tags = setting.grok_config.get("filtered_tags", "").split(",")
        video_progress_started = False
        last_video_progress = -1
        response_closed = False
        show_thinking = setting.grok_config.get("show_thinking", True)

        # Timeout management
        timeout_mgr = StreamTimeoutManager(
            chunk_timeout=setting.grok_config.get("stream_chunk_timeout", 120),
            first_timeout=setting.grok_config.get("stream_first_response_timeout", 30),
            total_timeout=setting.grok_config.get("stream_total_timeout", 600)
        )

        def make_chunk(content: str, finish: str = None):
            """Generate response chunk"""
            chunk_data = OpenAIChatCompletionChunkResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=model or "grok-4-mini-thinking-tahoe",
                choices=[OpenAIChatCompletionChunkChoice(
                    index=0,
                    delta=OpenAIChatCompletionChunkMessage(
                        role="assistant",
                        content=content
                    ) if content else {},
                    finish_reason=finish
                )]
            )
            return f"data: {chunk_data.model_dump_json()}\n\n"

        try:
            for chunk in response.iter_lines():
                # Timeout check
                is_timeout, timeout_msg = timeout_mgr.check_timeout()
                if is_timeout:
                    logger.warning(f"[Processor] {timeout_msg}")
                    yield make_chunk("", "stop")
                    yield "data: [DONE]\n\n"
                    return

                logger.debug(f"[Processor] Received chunk: {len(chunk)} bytes")
                if not chunk:
                    continue

                try:
                    data = orjson.loads(chunk)

                    # Error check
                    if error := data.get("error"):
                        error_msg = error.get('message', 'Unknown error')
                        logger.error(f"[Processor] API Error: {error_msg}")
                        yield make_chunk(f"Error: {error_msg}", "stop")
                        yield "data: [DONE]\n\n"
                        return

                    grok_resp = data.get("result", {}).get("response", {})
                    if not grok_resp:
                        continue
                    
                    timeout_mgr.mark_received()

                    # Update model
                    if user_resp := grok_resp.get("userResponse"):
                        if m := user_resp.get("model"):
                            model = m

                    # Video processing
                    if video_resp := grok_resp.get("streamingVideoGenerationResponse"):
                        progress = video_resp.get("progress", 0)
                        v_url = video_resp.get("videoUrl")
                        
                        # Progress update
                        if progress > last_video_progress:
                            last_video_progress = progress
                            if show_thinking:
                                if not video_progress_started:
                                    content = f"<think>Video generated {progress}%\n"
                                    video_progress_started = True
                                elif progress < 100:
                                    content = f"Video generated {progress}%\n"
                                else:
                                    content = f"Video generated {progress}%</think>\n"
                                yield make_chunk(content)
                        
                        # Video URL
                        if v_url:
                            logger.debug("[Processor] Video generation completed")
                            video_id = video_resp.get("videoId")
                            video_content = await GrokResponseProcessor._build_video_content(v_url, video_id, auth_token, auto_upscale)
                            yield make_chunk(video_content)
                        
                        continue

                    # Image mode
                    if grok_resp.get("imageAttachmentInfo"):
                        is_image = True

                    token = grok_resp.get("token", "")

                    # Image processing
                    if is_image:
                        if model_resp := grok_resp.get("modelResponse"):
                            image_mode = setting.global_config.get("image_mode", "url")
                            content = ""

                            for img in model_resp.get("generatedImageUrls", []):
                                try:
                                    if image_mode == "base64":
                                        # Base64 mode - send in chunks
                                        base64_str = await image_cache_service.download_base64(f"/{img}", auth_token)
                                        if base64_str:
                                            # Chunk large data
                                            if not base64_str.startswith("data:"):
                                                parts = base64_str.split(",", 1)
                                                if len(parts) == 2:
                                                    yield make_chunk(f"![Generated Image](data:{parts[0]},")
                                                    # 8KB chunks
                                                    for i in range(0, len(parts[1]), 8192):
                                                        yield make_chunk(parts[1][i:i+8192])
                                                    yield make_chunk(")\n")
                                                else:
                                                    yield make_chunk(f"![Generated Image]({base64_str})\n")
                                            else:
                                                yield make_chunk(f"![Generated Image]({base64_str})\n")
                                        else:
                                            yield make_chunk(f"![Generated Image](https://assets.grok.com/{img})\n")
                                    else:
                                        # URL mode
                                        await image_cache_service.download_image(f"/{img}", auth_token)
                                        img_path = img.replace('/', '-')
                                        base_url = setting.global_config.get("base_url", "")
                                        img_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                                        content += f"![Generated Image]({img_url})\n"
                                except Exception as e:
                                    logger.warning(f"[Processor] Failed to process image: {e}")
                                    content += f"![Generated Image](https://assets.grok.com/{img})\n"

                            yield make_chunk(content.strip(), "stop")
                            return
                        elif token:
                            yield make_chunk(token)

                    # Conversation processing
                    else:
                        if isinstance(token, list):
                            continue

                        if any(tag in token for tag in filtered_tags if token):
                            continue

                        current_is_thinking = grok_resp.get("isThinking", False)
                        message_tag = grok_resp.get("messageTag")

                        if thinking_finished and current_is_thinking:
                            continue

                        # Web search results processing
                        if grok_resp.get("toolUsageCardId"):
                            if web_search := grok_resp.get("webSearchResults"):
                                if current_is_thinking:
                                    if show_thinking:
                                        for result in web_search.get("results", []):
                                            title = result.get("title", "")
                                            url = result.get("url", "")
                                            preview = result.get("preview", "")
                                            preview_clean = preview.replace("\n", "") if isinstance(preview, str) else ""
                                            token += f'\n- [{title}]({url} "{preview_clean}")'
                                        token += "\n"
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                continue

                        if token:
                            content = token

                            if message_tag == "header":
                                content = f"\n\n{token}\n\n"

                            # Thinking state switching
                            should_skip = False
                            if not is_thinking and current_is_thinking:
                                if show_thinking:
                                    content = f"<think>\n{content}"
                                else:
                                    should_skip = True
                            elif is_thinking and not current_is_thinking:
                                if show_thinking:
                                    content = f"\n</think>\n{content}"
                                thinking_finished = True
                            elif current_is_thinking:
                                if not show_thinking:
                                    should_skip = True

                            if not should_skip:
                                yield make_chunk(content)
                            
                            is_thinking = current_is_thinking

                except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"[Processor] Parsing failed: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"[Processor] Error processing chunk: {e}")
                    continue

            yield make_chunk("", "stop")
            yield "data: [DONE]\n\n"
            logger.info(f"[Processor] Stream completed, duration: {timeout_mgr.duration():.2f}s")

        except Exception as e:
            logger.error(f"[Processor] Critical error: {e}")
            yield make_chunk(f"Processing error: {e}", "error")
            yield "data: [DONE]\n\n"
        finally:
            if not response_closed and hasattr(response, 'close'):
                try:
                    response.close()
                    logger.debug("[Processor] Response closed")
                except Exception as e:
                    logger.warning(f"[Processor] Failed to close response: {e}")

    @staticmethod
    async def _build_video_content(video_url: str, video_id: str, auth_token: str, auto_upscale: bool = None) -> str:
        """Build video content response"""
        logger.debug(f"[Processor] Video detected: {video_url}")
        
        # Handle User's custom auto-upscale feature
        is_upscaled = False
        if auto_upscale and video_id:
            try:
                upscale_result = await VideoUpscaleManager.upscale(video_id, auth_token)
                if upscale_result and upscale_result.get("success"):
                    video_url = upscale_result.get("hd_media_url")
                    is_upscaled = True
            except Exception as e:
                logger.warning(f"[Processor] Auto upscale failed: {e}")

        # If upscaled, use the detailed response directly? Or simply wrap in video tag.
        # HEAD code used full_url mostly.
        
        full_url = f"https://assets.grok.com/{video_url}"
        
        if is_upscaled:
             # Upscaled URLs are often simpler or directly accessible? 
             # Head code: f'<video src="{video_url}" controls="controls" width="500" height="300"></video>\n'
             return f'<video src="{video_url}" controls="controls" width="500" height="300"></video>\n'

        try:
            cache_path = await video_cache_service.download_video(f"/{video_url}", auth_token)
            if cache_path:
                video_path = video_url.replace('/', '-')
                base_url = setting.global_config.get("base_url", "")
                local_url = f"{base_url}/images/{video_path}" if base_url else f"/images/{video_path}"
                return f'<video src="{local_url}" controls="controls" width="500" height="300"></video>\n'
        except Exception as e:
            logger.warning(f"[Processor] Failed to cache video: {e}")
        
        return f'<video src="{full_url}" controls="controls" width="500" height="300"></video>\n'

    @staticmethod
    async def _append_images(content: str, images: list, auth_token: str) -> str:
        """Append images to content"""
        image_mode = setting.global_config.get("image_mode", "url")
        
        for img in images:
            try:
                if image_mode == "base64":
                    base64_str = await image_cache_service.download_base64(f"/{img}", auth_token)
                    if base64_str:
                        content += f"\n![Generated Image]({base64_str})"
                    else:
                        content += f"\n![Generated Image](https://assets.grok.com/{img})"
                else:
                    cache_path = await image_cache_service.download_image(f"/{img}", auth_token)
                    if cache_path:
                        img_path = img.replace('/', '-')
                        base_url = setting.global_config.get("base_url", "")
                        img_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                        content += f"\n![Generated Image]({img_url})"
                    else:
                        content += f"\n![Generated Image](https://assets.grok.com/{img})"
            except Exception as e:
                logger.warning(f"[Processor] Failed to process image: {e}")
                content += f"\n![Generated Image](https://assets.grok.com/{img})"
        
        return content

    @staticmethod
    def _build_response(content: str, model: str) -> OpenAIChatCompletionResponse:
        """Build response object"""
        return OpenAIChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[OpenAIChatCompletionChoice(
                index=0,
                message=OpenAIChatCompletionMessage(
                    role="assistant",
                    content=content
                ),
                finish_reason="stop"
            )],
            usage=None
        )
