"""
Model API module

Provides an OpenAI-compatible /v1/models endpoint, returning the list of all models supported by the system.
"""

import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends

from app.models.grok_models import Models
from app.core.auth import auth_manager
from app.core.logger import logger


# Create router
router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models(_: Optional[str] = Depends(auth_manager.verify)) -> Dict[str, Any]:
    """
    Retrieve available model list

    Returns an OpenAI-compatible list of models, including detailed information for all Grok models supported by the system.

    Args:
        _: Authentication dependency (auto-validated)

    Returns:
        Dict[str, Any]: Response data containing the model list
    """
    try:
        logger.debug("[Models] Requesting model list")

        # Get current timestamp
        current_timestamp = int(time.time())
        
        # Build model data list
        model_data: List[Dict[str, Any]] = []
        
        for model in Models:
            model_id = model.value
            config = Models.get_model_info(model_id)
            
            # Base information
            model_info = {
                "id": model_id,
                "object": "model", 
                "created": current_timestamp,
                "owned_by": "x-ai",
                "display_name": config.get("display_name", model_id),
                "description": config.get("description", ""),
                "raw_model_path": config.get("raw_model_path", f"xai/{model_id}"),
                "default_temperature": config.get("default_temperature", 1.0),
                "default_max_output_tokens": config.get("default_max_output_tokens", 8192),
                "supported_max_output_tokens": config.get("supported_max_output_tokens", 131072),
                "default_top_p": config.get("default_top_p", 0.95)
            }
            
            model_data.append(model_info)
        
        # Build response
        response = {
            "object": "list",
            "data": model_data
        }

        logger.debug(f"[Models] Successfully returned {len(model_data)} models")
        return response
        
    except Exception as e:
        logger.error(f"[Models] Error retrieving model list: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Failed to retrieve models: {e}",
                    "type": "internal_error",
                    "code": "model_list_error"
                }
            }
        )


@router.get("/models/{model_id}")
async def get_model(model_id: str, _: Optional[str] = Depends(auth_manager.verify)) -> Dict[str, Any]:
    """
    Retrieve specific model information

    Args:
        model_id (str): Model ID
        _: Authentication dependency (auto-validated)

    Returns:
        Dict[str, Any]: Detailed model information
    """
    try:
        logger.debug(f"[Models] Requesting model info: {model_id}")

        # Validate model existence
        if not Models.is_valid_model(model_id):
            logger.warning(f"[Models] Requested model does not exist: {model_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"Model '{model_id}' not found",
                        "type": "invalid_request_error", 
                        "code": "model_not_found"
                    }
                }
            )
        
        # Get current timestamp
        current_timestamp = int(time.time())
        
        # Get model configuration
        config = Models.get_model_info(model_id)
        
        # Build model information
        model_info = {
            "id": model_id,
            "object": "model",
            "created": current_timestamp,
            "owned_by": "x-ai",
            "display_name": config.get("display_name", model_id),
            "description": config.get("description", ""),
            "raw_model_path": config.get("raw_model_path", f"xai/{model_id}"),
            "default_temperature": config.get("default_temperature", 1.0),
            "default_max_output_tokens": config.get("default_max_output_tokens", 8192),
            "supported_max_output_tokens": config.get("supported_max_output_tokens", 131072),
            "default_top_p": config.get("default_top_p", 0.95)
        }

        logger.debug(f"[Models] Successfully returned model info: {model_id}")
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Models] Error retrieving model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Failed to retrieve model: {e}",
                    "type": "internal_error",
                    "code": "model_retrieve_error"
                }
            }
        )
