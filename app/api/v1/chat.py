# -*- coding: utf-8 -*-
"""
Chat API routing module

Provides an OpenAI-compatible chat API endpoint, supporting interaction with Grok models.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from fastapi.responses import StreamingResponse

from app.core.auth import auth_manager
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.services.grok.client import GrokClient
from app.models.openai_schema import OpenAIChatRequest

# Chat router
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/completions", response_model=None)
async def chat_completions(
    request: OpenAIChatRequest,
    _: Optional[str] = Depends(auth_manager.verify)
):
    """
    Create chat completion.

    Provides an OpenAI-compatible chat endpoint supporting both streaming and non‑streaming responses.

    Args:
        request: OpenAI‑format chat request.
        _: Authentication dependency (auto‑validated).

    Returns:
        OpenAIChatCompletionResponse: Non‑streaming response.
        StreamingResponse: Streaming response.

    Raises:
        HTTPException: When request processing fails.
    """
    try:
        logger.info(f"[Chat] Received chat request")

        # Call Grok client to process request
        result = await GrokClient.openai_to_grok(request.model_dump())
        
        # If streaming response, GrokClient already returns an iterator; wrap it in StreamingResponse
        if request.stream:
            return StreamingResponse(
                content=result,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        # Non‑streaming response: return directly
        return result
        
    except GrokApiException as e:
        logger.error(f"[Chat] Grok API error: {str(e)} - details: {e.details}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": e.error_code or "grok_api_error",
                    "code": e.error_code or "unknown"
                }
            }
        )
    except Exception as e:
        logger.error(f"[Chat] Chat request processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                    "code": "internal_server_error"
                }
            }
        )
