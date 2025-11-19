# -*- coding: utf-8 -*-
"""MCP Tools - Grok AI Conversation Tool"""

import json
from typing import Optional
from app.services.grok.client import GrokClient
from app.core.logger import logger
from app.core.exception import GrokApiException


async def ask_grok_impl(
    query: str,
    model: str = "grok-3-fast",
    system_prompt: Optional[str] = None
) -> str:
    """
    Internal implementation: Call Grok API and collect complete response

    Args:
        query: User question
        model: Model name
        system_prompt: System prompt

    Returns:
        str: Complete Grok response content
    """
    try:
        # Build message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        # Build request
        request_data = {
            "model": model,
            "messages": messages,
            "stream": True
        }

        logger.info(f"[MCP] ask_grok call, model: {model}")

        # Call Grok client (streaming)
        response_iterator = await GrokClient.openai_to_grok(request_data)

        # Collect all streaming response chunks
        content_parts = []
        async for chunk in response_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8')

            # Parse SSE format
            if chunk.startswith("data: "):
                data_str = chunk[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if content := delta.get("content"):
                            content_parts.append(content)
                except json.JSONDecodeError:
                    continue

        result = "".join(content_parts)
        logger.info(f"[MCP] ask_grok completed, response length: {len(result)}")
        return result

    except GrokApiException as e:
        logger.error(f"[MCP] Grok API error: {str(e)}")
        raise Exception(f"Grok API call failed: {str(e)}")
    except Exception as e:
        logger.error(f"[MCP] ask_grok exception: {str(e)}", exc_info=True)
        raise Exception(f"Error processing request: {str(e)}")
