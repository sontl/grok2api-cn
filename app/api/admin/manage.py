"""
Admin interface module

Provides Token management functions, including login authentication, Token add/delete/query operations.
"""

import secrets
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.config import setting
from app.core.logger import logger
from app.services.grok.token import token_manager
from app.models.grok_models import TokenType


# Create router
router = APIRouter(tags=["Admin"])

# Constant definitions
STATIC_DIR = Path(__file__).parents[2] / "template"
TEMP_DIR = Path(__file__).parents[3] / "data" / "temp"
IMAGE_CACHE_DIR = TEMP_DIR / "image"
VIDEO_CACHE_DIR = TEMP_DIR / "video"
SESSION_EXPIRE_HOURS = 24
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024

# Simple session storage
_sessions: Dict[str, datetime] = {}


# === Request/Response Models ===

class LoginRequest(BaseModel):
    """Login request"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response"""
    success: bool
    token: Optional[str] = None
    message: str


class AddTokensRequest(BaseModel):
    """Batch add token request"""
    tokens: List[str]
    token_type: str  # "sso" or "ssoSuper"


class DeleteTokensRequest(BaseModel):
    """Batch delete token request"""
    tokens: List[str]
    token_type: str  # "sso" or "ssoSuper"


class TokenInfo(BaseModel):
    """Token information"""
    token: str
    token_type: str
    created_time: Optional[int] = None
    remaining_queries: int
    heavy_remaining_queries: int
    status: str  # "Unused", "Rate-limited", "Expired", "Active"
    tags: List[str] = []  # Tag list
    note: str = ""  # Note


class TokenListResponse(BaseModel):
    """Token list response"""
    success: bool
    data: List[TokenInfo]
    total: int


# === Helper Functions ===

def validate_token_type(token_type_str: str) -> TokenType:
    """Validate and convert Token type string to enum"""
    if token_type_str not in ["sso", "ssoSuper"]:
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid token type, must be 'sso' or 'ssoSuper'", "code": "INVALID_TYPE"}
        )
    return TokenType.NORMAL if token_type_str == "sso" else TokenType.SUPER


def parse_created_time(created_time) -> Optional[int]:
    """Parse creation time, unify different formats"""
    if isinstance(created_time, str):
        return int(created_time) if created_time else None
    elif isinstance(created_time, int):
        return created_time
    return None


def calculate_token_stats(tokens: Dict[str, Any], token_type: str) -> Dict[str, int]:
    """Calculate token statistics"""
    total = len(tokens)
    expired = sum(1 for t in tokens.values() if t.get("status") == "expired")

    if token_type == "normal":
        unused = sum(1 for t in tokens.values()
                    if t.get("status") != "expired" and t.get("remainingQueries", -1) == -1)
        limited = sum(1 for t in tokens.values()
                     if t.get("status") != "expired" and t.get("remainingQueries", -1) == 0)
        active = sum(1 for t in tokens.values()
                    if t.get("status") != "expired" and t.get("remainingQueries", -1) > 0)
    else:  # super token
        unused = sum(1 for t in tokens.values()
                    if t.get("status") != "expired" and
                    t.get("remainingQueries", -1) == -1 and t.get("heavyremainingQueries", -1) == -1)
        limited = sum(1 for t in tokens.values()
                     if t.get("status") != "expired" and
                     (t.get("remainingQueries", -1) == 0 or t.get("heavyremainingQueries", -1) == 0))
        active = sum(1 for t in tokens.values()
                    if t.get("status") != "expired" and
                    (t.get("remainingQueries", -1) > 0 or t.get("heavyremainingQueries", -1) > 0))

    return {
        "total": total,
        "unused": unused,
        "limited": limited,
        "expired": expired,
        "active": active
    }


def verify_admin_session(authorization: Optional[str] = Header(None)) -> bool:
    """Verify admin session"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized access", "code": "UNAUTHORIZED"}
        )

    token = authorization[7:]  # Remove "Bearer " prefix

    # Check if token exists and has not expired
    if token not in _sessions:
        raise HTTPException(
            status_code=401,
            detail={"error": "Session expired or invalid", "code": "SESSION_INVALID"}
        )

    # Check if session has expired (24 hours)
    if datetime.now() > _sessions[token]:
        del _sessions[token]
        raise HTTPException(
            status_code=401,
            detail={"error": "Session expired", "code": "SESSION_EXPIRED"}
        )

    return True


def get_token_status(token_data: Dict[str, Any], token_type: str) -> str:
    """Get Token status"""
    # First check if expired (from status field in token.json)
    if token_data.get("status") == "expired":
        return "Expired"

    # Get remaining count
    remaining_queries = token_data.get("remainingQueries", -1)
    heavy_remaining = token_data.get("heavyremainingQueries", -1)

    # Select correct field based on token type
    if token_type == "ssoSuper":
        # Super token may use heavy model
        relevant_remaining = max(remaining_queries, heavy_remaining)
    else:
        # Normal token mainly checks remaining_queries
        relevant_remaining = remaining_queries

    if relevant_remaining == -1:
        return "Unused"
    elif relevant_remaining == 0:
        return "Rate-limited"
    else:
        return "Active"


# === Page Routes ===

@router.get("/login", response_class=HTMLResponse)
async def login_page():
    """Login page"""
    login_html = STATIC_DIR / "login.html"
    if login_html.exists():
        return login_html.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="Login page does not exist")


@router.get("/manage", response_class=HTMLResponse)
async def manage_page():
    """Management page"""
    admin_html = STATIC_DIR / "admin.html"
    if admin_html.exists():
        return admin_html.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="Management page does not exist")


# === API Endpoints ===

@router.post("/api/login", response_model=LoginResponse)
async def admin_login(request: LoginRequest) -> LoginResponse:
    """
    Admin login

    Verify username and password, return session token upon success.
    """
    try:
        logger.debug(f"[Admin] Admin login attempt - Username: {request.username}")

        # Verify username and password
        expected_username = setting.global_config.get("admin_username", "")
        expected_password = setting.global_config.get("admin_password", "")

        if request.username != expected_username or request.password != expected_password:
            logger.warning(f"[Admin] Login failed: Invalid username or password - username: {request.username}")
            return LoginResponse(
                success=False,
                message="Invalid username or password"
            )

        # Generate session token
        session_token = secrets.token_urlsafe(32)

        # Set session expiration time
        expire_time = datetime.now() + timedelta(hours=SESSION_EXPIRE_HOURS)
        _sessions[session_token] = expire_time

        logger.debug(f"[Admin] Admin login successful - username: {request.username}")

        return LoginResponse(
            success=True,
            token=session_token,
            message="Login successful"
        )

    except Exception as e:
        logger.error(f"[Admin] Login processing exception - username: {request.username}, error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Login failed: {str(e)}", "code": "LOGIN_ERROR"}
        )


@router.post("/api/logout")
async def admin_logout(_: bool = Depends(verify_admin_session),
                       authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Admin logout

    Clear session token.
    """
    try:
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            if token in _sessions:
                del _sessions[token]
                logger.debug("[Admin] Admin logout successful")
                return {"success": True, "message": "Logout successful"}

        logger.warning("[Admin] Logout failed: Invalid or missing session token")
        return {"success": False, "message": "Invalid session"}

    except Exception as e:
        logger.error(f"[Admin] Logout processing exception - error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Logout failed: {str(e)}", "code": "LOGOUT_ERROR"}
        )


@router.get("/api/tokens", response_model=TokenListResponse)
async def list_tokens(_: bool = Depends(verify_admin_session)) -> TokenListResponse:
    """
    Get all Token list

    Returns all Tokens in the system and their status information.
    """
    try:
        logger.debug("[Admin] Starting to retrieve token list")

        all_tokens_data = token_manager.get_tokens()
        token_list: List[TokenInfo] = []

        # Process normal tokens
        normal_tokens = all_tokens_data.get(TokenType.NORMAL.value, {})
        for token, data in normal_tokens.items():
            token_list.append(TokenInfo(
                token=token,
                token_type="sso",
                created_time=parse_created_time(data.get("createdTime")),
                remaining_queries=data.get("remainingQueries", -1),
                heavy_remaining_queries=data.get("heavyremainingQueries", -1),
                status=get_token_status(data, "sso"),
                tags=data.get("tags", []),  # Backward compatibility, return empty list if no tags field
                note=data.get("note", "")  # Backward compatibility, return empty string if no note field
            ))

        # Process Super Token
        super_tokens = all_tokens_data.get(TokenType.SUPER.value, {})
        for token, data in super_tokens.items():
            token_list.append(TokenInfo(
                token=token,
                token_type="ssoSuper",
                created_time=parse_created_time(data.get("createdTime")),
                remaining_queries=data.get("remainingQueries", -1),
                heavy_remaining_queries=data.get("heavyremainingQueries", -1),
                status=get_token_status(data, "ssoSuper"),
                tags=data.get("tags", []),  # Backward compatibility, return empty list if no tags field
                note=data.get("note", "")  # Backward compatibility, return empty string if no note field
            ))

        normal_count = len(normal_tokens)
        super_count = len(super_tokens)
        total_count = len(token_list)

        logger.debug(f"[Admin] Token list retrieval successful - Normal tokens: {normal_count}, Super tokens: {super_count}, Total: {total_count}")

        return TokenListResponse(
            success=True,
            data=token_list,
            total=total_count
        )

    except Exception as e:
        logger.error(f"[Admin] Token list retrieval exception - error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to retrieve token list: {str(e)}", "code": "LIST_ERROR"}
        )


@router.post("/api/tokens/add")
async def add_tokens(request: AddTokensRequest,
                    _: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """
    Batch add tokens

    Supports adding normal tokens (sso) and Super tokens (ssoSuper).
    """
    try:
        logger.debug(f"[Admin] Batch adding tokens - Type: {request.token_type}, Count: {len(request.tokens)}")

        # Validate and convert token type
        token_type = validate_token_type(request.token_type)

        # Add tokens
        await token_manager.add_token(request.tokens, token_type)

        logger.debug(f"[Admin] Token addition successful - Type: {request.token_type}, Count: {len(request.tokens)}")

        return {
            "success": True,
            "message": f"Successfully added {len(request.tokens)} tokens",
            "count": len(request.tokens)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin] Token addition exception - Type: {request.token_type}, Count: {len(request.tokens)}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to add tokens: {str(e)}", "code": "ADD_ERROR"}
        )


@router.post("/api/tokens/delete")
async def delete_tokens(request: DeleteTokensRequest,
                       _: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """
    Batch delete tokens

    Supports deleting normal tokens (sso) and Super tokens (ssoSuper).
    """
    try:
        logger.debug(f"[Admin] Batch deleting tokens - Type: {request.token_type}, Count: {len(request.tokens)}")

        # Validate and convert token type
        token_type = validate_token_type(request.token_type)

        # Delete tokens
        await token_manager.delete_token(request.tokens, token_type)

        logger.debug(f"[Admin] Token deletion successful - Type: {request.token_type}, Count: {len(request.tokens)}")

        return {
            "success": True,
            "message": f"Successfully deleted {len(request.tokens)} tokens",
            "count": len(request.tokens)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin] Token deletion exception - Type: {request.token_type}, Count: {len(request.tokens)}, Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to delete tokens: {str(e)}", "code": "DELETE_ERROR"}
        )


@router.get("/api/settings")
async def get_settings(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Get global configuration"""
    try:
        logger.debug("[Admin] Getting global configuration")
        return {
            "success": True,
            "data": {
                "global": setting.global_config,
                "grok": setting.grok_config
            }
        }
    except Exception as e:
        logger.error(f"[Admin] Failed to get configuration: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"Failed to get configuration: {str(e)}", "code": "GET_SETTINGS_ERROR"})


class UpdateSettingsRequest(BaseModel):
    """Update configuration request"""
    global_config: Optional[Dict[str, Any]] = None
    grok_config: Optional[Dict[str, Any]] = None


class StreamTimeoutSettings(BaseModel):
    """Streaming timeout configuration"""
    stream_chunk_timeout: int = 120
    stream_first_response_timeout: int = 30
    stream_total_timeout: int = 600


@router.post("/api/settings")
async def update_settings(request: UpdateSettingsRequest, _: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Update global configuration"""
    try:
        logger.debug("[Admin] Updating global configuration")

        # Use ConfigManager's save method (supports storage abstraction layer)
        await setting.save(
            global_config=request.global_config,
            grok_config=request.grok_config
        )

        logger.debug("[Admin] Configuration update successful")
        return {"success": True, "message": "Configuration updated successfully"}
    except Exception as e:
        logger.error(f"[Admin] Failed to update configuration: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": f"Failed to update configuration: {str(e)}", "code": "UPDATE_SETTINGS_ERROR"})


def _calculate_dir_size(directory: Path) -> int:
    """Calculate the size of all files in the directory (bytes)"""
    total_size = 0
    for file_path in directory.iterdir():
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except Exception as e:
                logger.warning(f"[Admin] Unable to get file size: {file_path.name}, Error: {str(e)}")
    return total_size


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string"""
    size_mb = size_bytes / BYTES_PER_MB
    if size_mb < 1:
        size_kb = size_bytes / BYTES_PER_KB
        return f"{size_kb:.1f} KB"
    return f"{size_mb:.1f} MB"


@router.get("/api/cache/size")
async def get_cache_size(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Get cache size"""
    try:
        logger.debug("[Admin] Starting to get cache size")

        # Calculate image cache size
        image_size = 0
        if IMAGE_CACHE_DIR.exists():
            image_size = _calculate_dir_size(IMAGE_CACHE_DIR)

        # Calculate video cache size
        video_size = 0
        if VIDEO_CACHE_DIR.exists():
            video_size = _calculate_dir_size(VIDEO_CACHE_DIR)

        # Total size
        total_size = image_size + video_size

        logger.debug(f"[Admin] Cache size retrieval completed - Images: {_format_size(image_size)}, Videos: {_format_size(video_size)}, Total: {_format_size(total_size)}")

        return {
            "success": True,
            "data": {
                "image_size": _format_size(image_size),
                "video_size": _format_size(video_size),
                "total_size": _format_size(total_size),
                "image_size_bytes": image_size,
                "video_size_bytes": video_size,
                "total_size_bytes": total_size
            }
        }

    except Exception as e:
        logger.error(f"[Admin] Cache size retrieval exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get cache size: {str(e)}", "code": "CACHE_SIZE_ERROR"}
        )


@router.post("/api/cache/clear")
async def clear_cache(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Clear cache

    Delete all temporary files"""
    try:
        logger.debug("[Admin] Starting cache clearing")

        deleted_count = 0
        image_count = 0
        video_count = 0

        # Clear image cache
        if IMAGE_CACHE_DIR.exists():
            for file_path in IMAGE_CACHE_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        image_count += 1
                        logger.debug(f"[Admin] Deleted image cache: {file_path.name}")
                    except Exception as e:
                        logger.error(f"[Admin] Failed to delete image cache: {file_path.name}, Error: {str(e)}")

        # Clear video cache
        if VIDEO_CACHE_DIR.exists():
            for file_path in VIDEO_CACHE_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        video_count += 1
                        logger.debug(f"[Admin] Deleted video cache: {file_path.name}")
                    except Exception as e:
                        logger.error(f"[Admin] Failed to delete video cache: {file_path.name}, Error: {str(e)}")

        deleted_count = image_count + video_count
        logger.debug(f"[Admin] Cache clearing completed - Images: {image_count}, Videos: {video_count}, Total: {deleted_count}")

        return {
            "success": True,
            "message": f"Successfully cleared cache, deleted {image_count} images, {video_count} videos, total {deleted_count} files",
            "data": {
                "deleted_count": deleted_count,
                "image_count": image_count,
                "video_count": video_count
            }
        }

    except Exception as e:
        logger.error(f"[Admin] Cache clearing exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to clear cache: {str(e)}", "code": "CACHE_CLEAR_ERROR"}
        )


@router.post("/api/cache/clear/images")
async def clear_image_cache(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Clear image cache

    Delete image cache files only"""
    try:
        logger.debug("[Admin] Starting image cache clearing")

        deleted_count = 0

        # Clear image cache
        if IMAGE_CACHE_DIR.exists():
            for file_path in IMAGE_CACHE_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"[Admin] Deleted image cache: {file_path.name}")
                    except Exception as e:
                        logger.error(f"[Admin] Failed to delete image cache: {file_path.name}, Error: {str(e)}")

        logger.debug(f"[Admin] Image cache clearing completed - Deleted {deleted_count} files")

        return {
            "success": True,
            "message": f"Successfully cleared image cache, deleted {deleted_count} files",
            "data": {
                "deleted_count": deleted_count,
                "type": "images"
            }
        }

    except Exception as e:
        logger.error(f"[Admin] Image cache clearing exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to clear image cache: {str(e)}", "code": "IMAGE_CACHE_CLEAR_ERROR"}
        )


@router.post("/api/cache/clear/videos")
async def clear_video_cache(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """Clear video cache

    Delete video cache files only"""
    try:
        logger.debug("[Admin] Starting video cache clearing")

        deleted_count = 0

        # Clear video cache
        if VIDEO_CACHE_DIR.exists():
            for file_path in VIDEO_CACHE_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"[Admin] Deleted video cache: {file_path.name}")
                    except Exception as e:
                        logger.error(f"[Admin] Failed to delete video cache: {file_path.name}, Error: {str(e)}")

        logger.debug(f"[Admin] Video cache clearing completed - Deleted {deleted_count} files")

        return {
            "success": True,
            "message": f"Successfully cleared video cache, deleted {deleted_count} files",
            "data": {
                "deleted_count": deleted_count,
                "type": "videos"
            }
        }

    except Exception as e:
        logger.error(f"[Admin] Video cache clearing exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to clear video cache: {str(e)}", "code": "VIDEO_CACHE_CLEAR_ERROR"}
        )


@router.get("/api/stats")
async def get_stats(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """
    Get statistics

    Return token statistics.
    """
    try:
        logger.debug("[Admin] Starting to get statistics")

        all_tokens_data = token_manager.get_tokens()

        # Statistics for normal tokens
        normal_tokens = all_tokens_data.get(TokenType.NORMAL.value, {})
        normal_stats = calculate_token_stats(normal_tokens, "normal")

        # Statistics for Super Token
        super_tokens = all_tokens_data.get(TokenType.SUPER.value, {})
        super_stats = calculate_token_stats(super_tokens, "super")

        total_count = normal_stats["total"] + super_stats["total"]

        stats = {
            "success": True,
            "data": {
                "normal": normal_stats,
                "super": super_stats,
                "total": total_count
            }
        }

        logger.debug(f"[Admin] Statistics retrieval successful - Normal tokens: {normal_stats['total']}, Super Token: {super_stats['total']}, Total: {total_count}")
        return stats

    except Exception as e:
        logger.error(f"[Admin] Statistics retrieval exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get statistics: {str(e)}", "code": "STATS_ERROR"}
        )


@router.get("/api/storage/mode")
async def get_storage_mode(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """
    Get current storage mode

    Return current storage mode (file/mysql/redis).
    """
    try:
        logger.debug("[Admin] Getting storage mode")

        import os
        storage_mode = os.getenv("STORAGE_MODE", "file").upper()

        return {
            "success": True,
            "data": {
                "mode": storage_mode
            }
        }

    except Exception as e:
        logger.error(f"[Admin] Storage mode retrieval exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get storage mode: {str(e)}", "code": "STORAGE_MODE_ERROR"}
        )


class UpdateTokenTagsRequest(BaseModel):
    """Update token tags request"""
    token: str
    token_type: str
    tags: List[str]


@router.post("/api/tokens/tags")
async def update_token_tags(
    request: UpdateTokenTagsRequest,
    _: bool = Depends(verify_admin_session)
) -> Dict[str, Any]:
    """
    Update token tags

    Update tags for the specified token.
    """
    try:
        logger.debug(f"[Admin] Update token tags - Token: {request.token[:10]}..., Tags: {request.tags}")

        # Validate and convert token type
        token_type = validate_token_type(request.token_type)

        # Update tags
        await token_manager.update_token_tags(request.token, token_type, request.tags)

        logger.debug(f"[Admin] Token tags updated successfully - Token: {request.token[:10]}..., Tags: {request.tags}")

        return {
            "success": True,
            "message": "Tags updated successfully",
            "tags": request.tags
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin] Token tags update exception - Token: {request.token[:10]}..., Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to update tags: {str(e)}", "code": "UPDATE_TAGS_ERROR"}
        )


@router.get("/api/tokens/tags/all")
async def get_all_tags(_: bool = Depends(verify_admin_session)) -> Dict[str, Any]:
    """
    Get all tags

    Return a list of all tags used by tokens in the system (deduplicated).
    """
    try:
        logger.debug("[Admin] Getting all tags")

        all_tokens_data = token_manager.get_tokens()
        tags_set = set()

        # Collect all tags
        for token_type_data in all_tokens_data.values():
            for token_data in token_type_data.values():
                tags = token_data.get("tags", [])
                if isinstance(tags, list):
                    tags_set.update(tags)

        tags_list = sorted(list(tags_set))
        logger.debug(f"[Admin] Tags retrieval successful - Total {len(tags_list)} tags")

        return {
            "success": True,
            "data": tags_list
        }

    except Exception as e:
        logger.error(f"[Admin] Tags retrieval exception - Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to get tags: {str(e)}", "code": "GET_TAGS_ERROR"}
        )


class UpdateTokenNoteRequest(BaseModel):
    """Update token note request"""
    token: str
    token_type: str
    note: str


@router.post("/api/tokens/note")
async def update_token_note(
    request: UpdateTokenNoteRequest,
    _: bool = Depends(verify_admin_session)
) -> Dict[str, Any]:
    """
    Update token note

    Add or modify note information for the specified token.
    """
    try:
        logger.debug(f"[Admin] Update token note - Token: {request.token[:10]}...")

        # Validate and convert token type
        token_type = validate_token_type(request.token_type)

        # Update note
        await token_manager.update_token_note(request.token, token_type, request.note)

        logger.debug(f"[Admin] Token note updated successfully - Token: {request.token[:10]}...")

        return {
            "success": True,
            "message": "Note updated successfully",
            "note": request.note
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin] Token note update exception - Token: {request.token[:10]}..., Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to update note: {str(e)}", "code": "UPDATE_NOTE_ERROR"}
        )


class TestTokenRequest(BaseModel):
    """Test token request"""
    token: str
    token_type: str


@router.post("/api/tokens/test")
async def test_token(
    request: TestTokenRequest,
    _: bool = Depends(verify_admin_session)
) -> Dict[str, Any]:
    """
    Test Token availability

    Verify if the Token is valid by sending rate limit check request.
    Handle according to different HTTP status codes:
    - 401: Token expired
    - 403: Server blocked, do not change Token status
    - Other errors: Set to rate-limited status
    """
    try:
        logger.debug(f"[Admin] Testing Token - Token: {request.token[:10]}...")

        # Validate and convert token type
        token_type = validate_token_type(request.token_type)

        # Construct full auth token
        auth_token = f"sso-rw={request.token};sso={request.token}"

        # Use check_limits method to test token
        result = await token_manager.check_limits(auth_token, "grok-4-fast")

        if result:
            logger.debug(f"[Admin] Token test successful - Token: {request.token[:10]}...")
            return {
                "success": True,
                "message": "Token is valid",
                "data": {
                    "valid": True,
                    "remaining_queries": result.get("remainingTokens", -1),
                    "limit": result.get("limit", -1)
                }
            }
        else:
            # Test failed, check_limits method has already called record_failure to handle error
            # Now check token status to determine error type
            logger.warning(f"[Admin] Token test failed - Token: {request.token[:10]}...")

            # Check current token status
            all_tokens = token_manager.get_tokens()
            token_data = all_tokens.get(token_type.value, {}).get(request.token)

            if token_data:
                if token_data.get("status") == "expired":
                    # Token is marked as expired (401 error)
                    return {
                        "success": False,
                        "message": "Token has expired",
                        "data": {
                            "valid": False,
                            "error_type": "expired",
                            "error_code": 401
                        }
                    }
                elif token_data.get("remainingQueries") == 0:
                    # Token is set to rate-limited status (other errors)
                    return {
                        "success": False,
                        "message": "Token is rate-limited",
                        "data": {
                            "valid": False,
                            "error_type": "limited",
                            "error_code": "other"
                        }
                    }
                else:
                    # Could be 403 error or other network issues, token status unchanged
                    return {
                        "success": False,
                        "message": "Server blocked or network error",
                        "data": {
                            "valid": False,
                            "error_type": "blocked",
                            "error_code": 403
                        }
                    }
            else:
                # Can't find token data
                return {
                    "success": False,
                    "message": "Token data exception",
                    "data": {
                        "valid": False,
                        "error_type": "unknown",
                        "error_code": "data_error"
                    }
                }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Admin] Token test exception - Token: {request.token[:10]}..., Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to test Token: {str(e)}", "code": "TEST_TOKEN_ERROR"}
        )
