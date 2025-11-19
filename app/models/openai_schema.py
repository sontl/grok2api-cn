"""OpenAI Request-Response Models"""

from fastapi import HTTPException
from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator


class OpenAIChatRequest(BaseModel):
    """OpenAI Chat Request Model"""

    model: str = Field(..., description="Model name", min_length=1)
    messages: List[Dict[str, Any]] = Field(..., description="Message list", min_length=1)
    stream: bool = Field(False, description="Enable streaming response")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=100000, description="Maximum token count")
    top_p: Optional[float] = Field(1.0, ge=0, le=1, description="Sampling parameter")

    @classmethod
    @field_validator('messages')
    def validate_messages(cls, v):
        """Validate message format"""
        if not v:
            raise HTTPException(
                status_code=400,
                detail="Message list cannot be empty"
            )

        for msg in v:
            if not isinstance(msg, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Each message must be a dictionary"
                )
            if 'role' not in msg:
                raise HTTPException(
                    status_code=400,
                    detail="Message is missing required field 'role'"
                )
            if 'content' not in msg:
                raise HTTPException(
                    status_code=400,
                    detail="Message is missing required field 'content'"
                )
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid role '{msg['role']}', must be 'system', 'user', or 'assistant'"
                )

        return v

    @classmethod
    @field_validator('model')
    def validate_model(cls, v):
        """Validate model name"""
        allowed_models = [
            'grok-3-fast', 'grok-4-fast', 'grok-4-fast-expert',
            'grok-4-expert', 'grok-4-heavy', 'grok-imagine-0.9'
        ]
        if v not in allowed_models:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model '{v}', supported models: {', '.join(allowed_models)}"
            )
        return v

class OpenAIChatCompletionMessage(BaseModel):
    """Chat completion message"""
    role: str = Field(..., description="Role")
    content: str = Field(..., description="Message content")
    reference_id: Optional[str] = Field(default=None, description="Reference ID")
    annotations: Optional[List[str]] = Field(default=None, description="Annotations")


class OpenAIChatCompletionChoice(BaseModel):
    """Chat completion choice"""
    index: int = Field(..., description="Choice index")
    message: OpenAIChatCompletionMessage = Field(..., description="Response message")
    logprobs: Optional[float] = Field(default=None, description="Log probabilities")
    finish_reason: str = Field(default="stop", description="Finish reason")


class OpenAIChatCompletionResponse(BaseModel):
    """Chat completion response"""
    id: str = Field(..., description="Response ID")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[OpenAIChatCompletionChoice] = Field(..., description="Response choices")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage")


class OpenAIChatCompletionChunkMessage(BaseModel):
    """Streaming response message chunk"""
    role: str = Field(..., description="Role")
    content: str = Field(..., description="Message content")


class OpenAIChatCompletionChunkChoice(BaseModel):
    """Streaming response choice"""
    index: int = Field(..., description="Choice index")
    delta: Optional[Union[Dict[str, Any], OpenAIChatCompletionChunkMessage]] = Field(
        None, description="Delta data"
    )
    finish_reason: Optional[str] = Field(None, description="Finish reason")


class OpenAIChatCompletionChunkResponse(BaseModel):
    """Streaming chat completion response"""
    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    system_fingerprint: Optional[str] = Field(default=None, description="System fingerprint")
    choices: List[OpenAIChatCompletionChunkChoice] = Field(..., description="Response choices")