"""Grok API Response Processor Module"""

import json
import uuid
import time
import os
import asyncio
from typing import AsyncGenerator

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

    def __init__(self, chunk_timeout: int = 120, first_response_timeout: int = 30, total_timeout: int = 600):
        """Initialize timeout manager

        Args:
            chunk_timeout: Chunk interval timeout (seconds)
            first_response_timeout: First response timeout (seconds)
            total_timeout: Total timeout limit (seconds, 0 means no limit)
        """
        self.chunk_timeout = chunk_timeout
        self.first_response_timeout = first_response_timeout
        self.total_timeout = total_timeout

        self.start_time = asyncio.get_event_loop().time()
        self.last_chunk_time = self.start_time
        self.first_chunk_received = False

    def check_timeout(self) -> tuple[bool, str]:
        """Check if timeout has occurred

        Returns:
            (is_timeout, timeout_message): Whether timeout occurred and timeout message
        """
        current_time = asyncio.get_event_loop().time()

        # Check first response timeout
        if not self.first_chunk_received:
            if current_time - self.start_time > self.first_response_timeout:
                return True, f"First response timeout ({self.first_response_timeout}s without receiving first chunk)"

        # Check total timeout
        if self.total_timeout > 0:
            if current_time - self.start_time > self.total_timeout:
                return True, f"Stream response total timeout ({self.total_timeout}s)"

        # Check chunk interval timeout
        if self.first_chunk_received:
            if current_time - self.last_chunk_time > self.chunk_timeout:
                return True, f"Chunk interval timeout ({self.chunk_timeout}s without new data)"

        return False, ""

    def mark_chunk_received(self):
        """Mark chunk received"""
        self.last_chunk_time = asyncio.get_event_loop().time()
        self.first_chunk_received = True

    def get_total_duration(self) -> float:
        """Get total duration (seconds)"""
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

                data = json.loads(chunk.decode("utf-8"))

                # Error check
                if error := data.get("error"):
                    raise GrokApiException(
                        f"API error: {error.get('message', 'Unknown error')}",
                        "API_ERROR",
                        {"code": error.get("code")}
                    )

                # Extract response data
                grok_resp = data.get("result", {}).get("response", {})

                # Extract video data
                if video_resp := grok_resp.get("streamingVideoGenerationResponse"):
                    video_url = video_resp.get("videoUrl")
                    
                    if video_url:
                        is_upscaled = False
                        # Auto upscale - only when we have the video URL
                        if auto_upscale:
                            video_id = video_resp.get("videoId")
                            if video_id:
                                try:
                                    upscale_result = await VideoUpscaleManager.upscale(video_id, auth_token)
                                    if upscale_result and upscale_result.get("success"):
                                        video_url = upscale_result.get("hd_media_url")
                                        is_upscaled = True
                                except Exception as e:
                                    logger.warning(f"[Processor] Auto upscale failed: {e}")
                        logger.debug(f"[Processor] Detected video generation: {video_url}")
                        
                        if is_upscaled:
                            content = f'<video src="{video_url}" controls="controls" width="500" height="300"></video>\n'
                        else:
                            full_video_url = f"https://assets.grok.com/{video_url}"

                            # Download and cache video
                            try:
                                cache_path = await video_cache_service.download_video(f"/{video_url}", auth_token)
                                if cache_path:
                                    video_path = video_url.replace('/', '-')
                                    base_url = setting.global_config.get("base_url", "")
                                    local_video_url = f"{base_url}/images/{video_path}" if base_url else f"/images/{video_path}"
                                    content = f'<video src="{local_video_url}" controls="controls" width="500" height="300"></video>\n'
                                else:
                                    content = f'<video src="{full_video_url}" controls="controls" width="500" height="300"></video>\n'
                            except Exception as e:
                                logger.warning(f"[Processor] Failed to cache video: {e}")
                                content = f'<video src="{full_video_url}" controls="controls" width="500" height="300"></video>\n'

                        # Return video response
                        result = OpenAIChatCompletionResponse(
                            id=f"chatcmpl-{uuid.uuid4()}",
                            object="chat.completion",
                            created=int(time.time()),
                            model=model or "grok-imagine-0.9",
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
                        response_closed = True
                        response.close()
                        return result

                # Extract model response
                model_response = grok_resp.get("modelResponse")
                if not model_response:
                    continue

                # Check error in modelResponse
                if error_msg := model_response.get("error"):
                    raise GrokApiException(
                        f"Model response error: {error_msg}",
                        "MODEL_ERROR"
                    )

                # Build response content
                model_name = model_response.get("model")
                content = model_response.get("message", "")

                # Extract image data
                if images := model_response.get("generatedImageUrls"):
                    # Get image return mode
                    image_mode = setting.global_config.get("image_mode", "url")

                    for img in images:
                        try:
                            if image_mode == "base64":
                                # base64 mode: download and convert to base64
                                base64_str = await image_cache_service.download_base64(f"/{img}", auth_token)
                                if base64_str:
                                    content += f"\n![Generated Image]({base64_str})"
                                else:
                                    content += f"\n![Generated Image](https://assets.grok.com/{img})"
                            else:
                                # url mode: cache and return link
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

                # Return OpenAI response format
                result = OpenAIChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=model_name,
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
                response_closed = True
                response.close()
                return result

            raise GrokApiException("No response data", "NO_RESPONSE")

        except json.JSONDecodeError as e:
            logger.error(f"[Processor] JSON parsing failed: {e}")
            raise GrokApiException(f"JSON parsing failed: {e}", "JSON_ERROR") from e
        except Exception as e:
            logger.error(f"[Processor] Unknown error occurred while processing response: {type(e).__name__}: {e}")
            raise GrokApiException(f"Response processing error: {e}", "PROCESS_ERROR") from e
        finally:
            # Ensure response object is closed to prevent double release
            if not response_closed and hasattr(response, 'close'):
                try:
                    response.close()
                except Exception as e:
                    logger.warning(f"[Processor] Error closing response object: {e}")

    @staticmethod
    async def process_stream(response, auth_token: str, auto_upscale: bool = None) -> AsyncGenerator[str, None]:
        """Process streaming response"""
        # 流式生成状态
        is_image = False
        is_thinking = False
        thinking_finished = False
        chunk_index = 0
        model = None
        filtered_tags = setting.grok_config.get("filtered_tags", "").split(",")
        video_progress_started = False
        last_video_progress = -1
        response_closed = False
        show_thinking = setting.grok_config.get("show_thinking", True)

        # 初始化超时管理器
        timeout_manager = StreamTimeoutManager(
            chunk_timeout=setting.grok_config.get("stream_chunk_timeout", 120),
            first_response_timeout=setting.grok_config.get("stream_first_response_timeout", 30),
            total_timeout=setting.grok_config.get("stream_total_timeout", 600)
        )

        def make_chunk(chunk_content: str, finish: str = None):
            """生成OpenAI格式的响应块"""
            chunk_data = OpenAIChatCompletionChunkResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=model or "grok-4-mini-thinking-tahoe",
                choices=[OpenAIChatCompletionChunkChoice(
                    index=chunk_index,
                    delta=OpenAIChatCompletionChunkMessage(
                        role="assistant",
                        content=chunk_content
                    ) if chunk_content else {},
                    finish_reason=finish
                )]
            ).model_dump()
            # SSE 格式返回
            return f"data: {json.dumps(chunk_data)}\n\n"

        try:
            for chunk in response.iter_lines():
                # 超时检查
                is_timeout, timeout_msg = timeout_manager.check_timeout()
                if is_timeout:
                    logger.warning(f"[Processor] {timeout_msg}")
                    yield make_chunk("", "stop")
                    yield "data: [DONE]\n\n"
                    return

                logger.debug(f"[Processor] 接收到数据块, 长度: {len(chunk)} bytes")
                if not chunk:
                    continue

                try:
                    data = json.loads(chunk.decode("utf-8"))

                    # 错误检查
                    if error := data.get("error"):
                        error_msg = error.get('message', '未知错误')
                        logger.error(f"[Processor] Grok API返回错误: {error_msg}")
                        yield make_chunk(f"Error: {error_msg}", "stop")
                        yield "data: [DONE]\n\n"
                        return

                    # 提取响应数据
                    grok_resp = data.get("result", {}).get("response", {})
                    logger.debug(f"[Processor] 解析响应数据, 长度: {len(grok_resp)} bytes")
                    if not grok_resp:
                        continue
                    
                    # 更新超时计时器
                    timeout_manager.mark_chunk_received()

                    # 更新模型名称
                    if user_resp := grok_resp.get("userResponse"):
                        if m := user_resp.get("model"):
                            model = m

                    # 提取视频数据
                    if video_resp := grok_resp.get("streamingVideoGenerationResponse"):
                        progress = video_resp.get("progress", 0)
                        v_url = video_resp.get("videoUrl")
                        
                        # 处理进度更新
                        if progress > last_video_progress:
                            last_video_progress = progress
                            
                            # 思考状态
                            if show_thinking:
                                # 添加 <think> 标签
                                if not video_progress_started:
                                    content = f"<think>视频已生成{progress}%\n"
                                    video_progress_started = True
                                elif progress < 100:
                                    content = f"视频已生成{progress}%\n"
                                else:
                                    # 关闭 <think> 标签
                                    content = f"视频已生成{progress}%</think>\n"

                                yield make_chunk(content)
                                chunk_index += 1
                        
                        # 处理视频URL - only when video generation is complete (has videoUrl)
                        if v_url:
                            logger.debug(f"[Processor] 视频生成完成")
                            is_upscaled = False
                            # Auto upscale - only when we have the video URL
                            if auto_upscale:
                                video_id = video_resp.get("videoId")
                                if video_id:
                                    try:
                                        upscale_result = await VideoUpscaleManager.upscale(video_id, auth_token)
                                        if upscale_result and upscale_result.get("success"):
                                            v_url = upscale_result.get("hd_media_url")
                                            is_upscaled = True
                                    except Exception as e:
                                        logger.warning(f"[Processor] Auto upscale failed: {e}")

                            if is_upscaled:
                                video_content = f'<video src="{v_url}" controls="controls"></video>\n'
                            else:
                                full_video_url = f"https://assets.grok.com/{v_url}"
                                
                                try:
                                    cache_path = await video_cache_service.download_video(f"/{v_url}", auth_token)
                                    if cache_path:
                                        video_path = v_url.replace('/', '-')
                                        base_url = setting.global_config.get("base_url", "")
                                        local_video_url = f"{base_url}/images/{video_path}" if base_url else f"/images/{video_path}"
                                        video_content = f'<video src="{local_video_url}" controls="controls"></video>\n'
                                    else:
                                        video_content = f'<video src="{full_video_url}" controls="controls"></video>\n'
                                except Exception as e:
                                    logger.warning(f"[Processor] 缓存视频失败: {e}")
                                    video_content = f'<video src="{full_video_url}" controls="controls"></video>\n'
                            
                            yield make_chunk(video_content)
                            chunk_index += 1
                        
                        continue

                    # 检查生成模式
                    if grok_resp.get("imageAttachmentInfo"):
                        is_image = True

                    # 获取token
                    token = grok_resp.get("token", "")

                    # 提取图片数据
                    if is_image:
                        if model_resp := grok_resp.get("modelResponse"):
                            # 获取图片返回模式
                            image_mode = setting.global_config.get("image_mode", "url")

                            # 初始化内容变量
                            content = ""

                            # 生成图片链接并缓存
                            for img in model_resp.get("generatedImageUrls", []):
                                try:
                                    if image_mode == "base64":
                                        # base64 模式：下载并转换为 base64
                                        base64_str = await image_cache_service.download_base64(f"/{img}", auth_token)
                                        if base64_str:
                                            # 分块发送 base64 数据，每 8KB 一个 chunk
                                            markdown_prefix = "![Generated Image](data:"
                                            markdown_suffix = ")\n"

                                            # 提取 data URL 的 mime 和 base64 部分
                                            if not base64_str.startswith("data:"):
                                                parts = base64_str.split(",", 1)
                                                if len(parts) == 2:
                                                    mime_part = parts[0] + ","
                                                    b64_data = parts[1]

                                                    # 发送前缀
                                                    yield make_chunk(markdown_prefix + mime_part)
                                                    chunk_index += 1

                                                    # 分块发送 base64 数据
                                                    chunk_size = 8192
                                                    for i in range(0, len(b64_data), chunk_size):
                                                        chunk_data = b64_data[i:i + chunk_size]
                                                        yield make_chunk(chunk_data)
                                                        chunk_index += 1

                                                    # 发送后缀
                                                    yield make_chunk(markdown_suffix)
                                                    chunk_index += 1
                                                else:
                                                    yield make_chunk(f"![Generated Image]({base64_str})\n")
                                                    chunk_index += 1
                                            else:
                                                yield make_chunk(f"![Generated Image]({base64_str})\n")
                                                chunk_index += 1
                                        else:
                                            yield make_chunk(f"![Generated Image](https://assets.grok.com/{img})\n")
                                            chunk_index += 1
                                    else:
                                        # url 模式：缓存并返回链接
                                        await image_cache_service.download_image(f"/{img}", auth_token)
                                        # 本地图片路径
                                        img_path = img.replace('/', '-')
                                        base_url = setting.global_config.get("base_url", "")
                                        img_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                                        content += f"![Generated Image]({img_url})\n"
                                except Exception as e:
                                    logger.warning(f"[Processor] 处理图片失败: {e}")
                                    content += f"![Generated Image](https://assets.grok.com/{img})\n"

                            # 发送内容
                            yield make_chunk(content.strip(), "stop")
                            return
                        elif token:
                            yield make_chunk(token)
                            chunk_index += 1

                    # 提取对话数据
                    else:
                        # 过滤 list 格式的 token
                        if isinstance(token, list):
                            continue

                        # 过滤特定标签
                        if any(tag in token for tag in filtered_tags if token):
                            continue

                        # 获取当前状态
                        current_is_thinking = grok_resp.get("isThinking", False)
                        message_tag = grok_resp.get("messageTag")

                        # 跳过后续的 <think> 标签
                        if thinking_finished and current_is_thinking:
                            continue

                        # 检查 toolUsageCardId
                        if grok_resp.get("toolUsageCardId"):
                            if web_search := grok_resp.get("webSearchResults"):
                                if current_is_thinking:
                                    if show_thinking:
                                        # 封装搜索结果
                                        for result in web_search.get("results", []):
                                            title = result.get("title", "")
                                            url = result.get("url", "")
                                            preview = result.get("preview", "")
                                            preview_clean = preview.replace("\n", "") if isinstance(preview, str) else ""
                                            token += f'\n- [{title}]({url} "{preview_clean}")'
                                        token += "\n"
                                    else:
                                        # show_thinking=false 时跳过 thinking 状态下的搜索结果
                                        continue
                                else:
                                    # 有 webSearchResults 但 isThinking 为 false
                                    continue
                            else:
                                # 没有 webSearchResults
                                continue

                        if token:
                            content = token

                            # header 在 token 后换行
                            if message_tag == "header":
                                content = f"\n\n{token}\n\n"

                            # is_thinking 状态切换
                            should_skip = False
                            if not is_thinking and current_is_thinking:
                                # 进入 thinking 状态
                                if show_thinking:
                                    content = f"<think>\n{content}"
                                else:
                                    should_skip = True
                            elif is_thinking and not current_is_thinking:
                                # 退出 thinking 状态
                                if show_thinking:
                                    content = f"\n</think>\n{content}"
                                thinking_finished = True
                            elif current_is_thinking:
                                # 处于 thinking 状态中
                                if not show_thinking:
                                    should_skip = True

                            # 只在不需要跳过时才发送
                            if not should_skip:
                                yield make_chunk(content)
                                chunk_index += 1
                            
                            is_thinking = current_is_thinking

                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"[Processor] 解析chunk失败: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"[Processor] 处理chunk出错: {e}")
                    continue

            # 发送结束块
            yield make_chunk("", "stop")

            # 发送流结束标记
            yield "data: [DONE]\n\n"
            
            # 记录流式响应统计
            logger.info(f"[Processor] 流式响应完成，总耗时: {timeout_manager.get_total_duration():.2f}秒")

        except Exception as e:
            logger.error(f"[Processor] 流式处理发生严重错误: {e}")
            yield make_chunk(f"处理错误: {e}", "error")
            # 发送流结束标记
            yield "data: [DONE]\n\n"
        finally:
            # 确保响应对象被关闭
            if not response_closed and hasattr(response, 'close'):
                try:
                    response.close()
                    logger.debug("[Processor] 流式响应对象已关闭")
                except Exception as e:
                    logger.warning(f"[Processor] 关闭流式响应对象时出错: {e}")