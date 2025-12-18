"""Cache service module"""

import asyncio
import base64
from pathlib import Path
from typing import Optional, Tuple
from curl_cffi.requests import AsyncSession

from app.core.config import setting
from app.core.logger import logger
from app.services.grok.statsig import get_dynamic_headers


# Constants
MIME_TYPES = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
    '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp',
}
DEFAULT_MIME = 'image/jpeg'
ASSETS_URL = "https://assets.grok.com"


class CacheService:
    """Base cache service class"""

    def __init__(self, cache_type: str, timeout: float = 30.0):
        self.cache_type = cache_type
        self.cache_dir = Path(f"data/temp/{cache_type}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._cleanup_lock = asyncio.Lock()

    def _get_path(self, file_path: str) -> Path:
        """Convert file path to cache path"""
        return self.cache_dir / file_path.lstrip('/').replace('/', '-')

    def _log(self, level: str, msg: str):
        """Unified log output"""
        getattr(logger, level)(f"[{self.cache_type.upper()}Cache] {msg}")

    def _build_headers(self, file_path: str, auth_token: str) -> dict:
        """Build request headers"""
        cf = setting.grok_config.get("cf_clearance", "")
        return {
            **get_dynamic_headers(pathname=file_path),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-site",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://grok.com/",
            "Cookie": f"{auth_token};{cf}" if cf else auth_token
        }

    async def download(self, file_path: str, auth_token: str, timeout: Optional[float] = None) -> Optional[Path]:
        """Download and cache file"""
        cache_path = self._get_path(file_path)
        if cache_path.exists():
            self._log("debug", "File already cached")
            return cache_path

        try:
            proxy = setting.get_proxy("cache") # Using unified get_proxy for cache
            proxies = {"http": proxy, "https": proxy} if proxy else {}
            
            if proxy:
                self._log("debug", f"Using proxy: {proxy.split('@')[-1] if '@' in proxy else proxy}")

            async with AsyncSession() as session:
                url = f"{ASSETS_URL}{file_path}"
                self._log("debug", f"Downloading: {url}")
                
                response = await session.get(
                    url,
                    headers=self._build_headers(file_path, auth_token),
                    proxies=proxies,
                    timeout=timeout or self.timeout,
                    allow_redirects=True,
                    impersonate="chrome133a"
                )
                response.raise_for_status()
                
                cache_path.write_bytes(response.content)
                self._log("debug", "Cached successfully")
                
                # Async cleanup (safe)
                asyncio.create_task(self._safe_cleanup())
                return cache_path
                
        except Exception as e:
            self._log("error", f"Download failed: {e}")
            return None

    def get_cached(self, file_path: str) -> Optional[Path]:
        """Get cached file path"""
        path = self._get_path(file_path)
        return path if path.exists() else None

    async def _safe_cleanup(self):
        """Safe cleanup (captures exceptions)"""
        try:
            await self.cleanup()
        except Exception as e:
            self._log("error", f"Background cleanup failed: {e}")

    async def cleanup(self):
        """Clean up cache directory"""
        if self._cleanup_lock.locked():
            return
        
        async with self._cleanup_lock:
            try:
                max_mb = setting.global_config.get(f"{self.cache_type}_cache_max_size_mb", 500)
                max_bytes = max_mb * 1024 * 1024

                # Get file info (path, size, mtime)
                files = [(f, (s := f.stat()).st_size, s.st_mtime) 
                        for f in self.cache_dir.glob("*") if f.is_file()]
                total = sum(size for _, size, _ in files)

                if total <= max_bytes:
                    return

                self._log("info", f"Cleaning cache {total/1024/1024:.1f}MB -> {max_mb}MB")
                
                # Delete oldest files
                for path, size, _ in sorted(files, key=lambda x: x[2]):
                    if total <= max_bytes:
                        break
                    path.unlink()
                    total -= size
                
                self._log("info", f"Cleanup completed: {total/1024/1024:.1f}MB")
            except Exception as e:
                self._log("error", f"Cleanup failed: {e}")


class ImageCache(CacheService):
    """Image cache service"""

    def __init__(self):
        super().__init__("image", timeout=30.0)

    async def download_image(self, path: str, token: str) -> Optional[Path]:
        """Download image"""
        # Using extended timeout from HEAD
        return await self.download(path, token, timeout=130.0)

    @staticmethod
    def to_base64(image_path: Path) -> Optional[str]:
        """Convert image to base64"""
        try:
            if not image_path.exists():
                logger.error(f"[ImageCache] File does not exist: {image_path}")
                return None

            data = base64.b64encode(image_path.read_bytes()).decode()
            mime = MIME_TYPES.get(image_path.suffix.lower(), DEFAULT_MIME)
            return f"data:{mime};base64,{data}"
        except Exception as e:
            logger.error(f"[ImageCache] Conversion failed: {e}")
            return None

    async def download_base64(self, path: str, token: str) -> Optional[str]:
        """Download and convert to base64 (auto delete temp file)"""
        try:
            # Using extended timeout from HEAD
            cache_path = await self.download(path, token, timeout=130.0)
            if not cache_path:
                return None

            result = self.to_base64(cache_path)
            
            # Clean up temp file
            try:
                cache_path.unlink()
            except Exception as e:
                logger.warning(f"[ImageCache] Failed to delete temp file: {e}")

            return result
        except Exception as e:
            logger.error(f"[ImageCache] Download base64 failed: {e}")
            return None


class VideoCache(CacheService):
    """Video cache service"""

    def __init__(self):
        super().__init__("video", timeout=60.0)

    async def download_video(self, path: str, token: str) -> Optional[Path]:
        """Download video"""
        # Using extended timeout from HEAD
        return await self.download(path, token, timeout=160.0)


# Global instances (using names compatible with existing code)
image_cache_service = ImageCache()
video_cache_service = VideoCache()
