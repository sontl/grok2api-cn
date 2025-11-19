"""Image Service API routing module

Provides endpoints for retrieving cached images and videos."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.logger import logger
from app.services.grok.cache import image_cache_service, video_cache_service


# Image router
router = APIRouter()


@router.get("/images/{img_path:path}")
async def get_image(img_path: str):
    """Get cached image or video

    Args:
        img_path: File path, format like users-xxx-generated-xxx-image.jpg or users-xxx-generated-xxx-video.mp4

    Returns:
        File response
    """
    try:
        # Convert the path back to its original format (replace hyphens with slashes)
        original_path = "/" + img_path.replace('-', '/')

        # Determine if it's an image or video
        is_video = any(original_path.lower().endswith(ext) for ext in ['.mp4', '.webm', '.mov', '.avi'])

        if is_video:
            # Check video cache
            cache_path = video_cache_service.get_cached(original_path)
            media_type = "video/mp4"
        else:
            # Check image cache
            cache_path = image_cache_service.get_cached(original_path)
            media_type = "image/jpeg"

        if cache_path and cache_path.exists():
            logger.debug(f"[MediaAPI] Returning cached file: {cache_path}")
            return FileResponse(
                path=str(cache_path),
                media_type=media_type,
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Access-Control-Allow-Origin": "*"
                }
            )

        # File does not exist
        logger.warning(f"[MediaAPI] File not found: {original_path}")
        raise HTTPException(status_code=404, detail="File not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MediaAPI] Failed to get file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
