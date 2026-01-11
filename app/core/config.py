"""Configuration Manager"""

import toml
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import aiofiles

# Default Configuration
DEFAULT_GROK = {
    "api_key": "",
    "proxy_url": "",
    "proxy_pool_url": "",
    "proxy_pool_interval": 300,
    "cache_proxy_url": "",
    "cf_clearance": "",
    "x_statsig_id": "",
    "dynamic_statsig": True,
    "filtered_tags": "xaiartifact,xai:tool_usage_card",
    "show_thinking": True,
    "temporary": False,
    "max_upload_concurrency": 20,
    "max_request_concurrency": 100,
    "stream_first_response_timeout": 30,
    "stream_chunk_timeout": 120,
    "stream_total_timeout": 600,
    "retry_status_codes": [401, 429],  # Retryable HTTP status codes
}

DEFAULT_GLOBAL = {
    "base_url": "http://localhost:8000",
    "log_level": "INFO",
    "image_mode": "url",
    "admin_password": "admin",
    "admin_username": "admin",
    "image_cache_max_size_mb": 512,
    "video_cache_max_size_mb": 1024,
    "max_upload_concurrency": 20,
    "max_request_concurrency": 50,
    "batch_save_interval": 1.0,  # Batch save interval (seconds)
    "batch_save_threshold": 10   # Threshold for triggering batch save
}


class ConfigManager:
    """Configuration Manager"""

    def __init__(self) -> None:
        """Initialize configuration"""
        self.config_path: Path = Path(__file__).parents[2] / "data" / "setting.toml"
        self._storage: Optional[Any] = None
        self._ensure_exists()
        self.global_config: Dict[str, Any] = self.load("global")
        self.grok_config: Dict[str, Any] = self.load("grok")
    
    def _ensure_exists(self) -> None:
        """Ensure configuration file exists"""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._create_default()
    
    def _create_default(self) -> None:
        """Create default configuration"""
        default = {"grok": DEFAULT_GROK.copy(), "global": DEFAULT_GLOBAL.copy()}
        with open(self.config_path, "w", encoding="utf-8") as f:
            toml.dump(default, f)
    
    def _normalize_proxy(self, proxy: str) -> str:
        """Normalize proxy URL (sock5/socks5 → socks5h://)"""
        if not proxy:
            return proxy

        proxy = proxy.strip()
        if proxy.startswith("sock5h://"):
            proxy = proxy.replace("sock5h://", "socks5h://", 1)
        if proxy.startswith("sock5://"):
            proxy = proxy.replace("sock5://", "socks5://", 1)
        if proxy.startswith("socks5://"):
            return proxy.replace("socks5://", "socks5h://", 1)
        return proxy
    
    def _normalize_cf(self, cf: str) -> str:
        """Normalize CF Clearance (add prefix automatically)"""
        if cf and not cf.startswith("cf_clearance="):
            return f"cf_clearance={cf}"
        return cf

    def set_storage(self, storage: Any) -> None:
        """Set storage instance"""
        self._storage = storage

    def load(self, section: Literal["global", "grok"]) -> Dict[str, Any]:
        """Load configuration section"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = toml.load(f)[section]

            # Normalize Grok configuration
            if section == "grok":
                if "proxy_url" in config:
                    config["proxy_url"] = self._normalize_proxy(config["proxy_url"])
                if "cache_proxy_url" in config:
                    config["cache_proxy_url"] = self._normalize_proxy(config["cache_proxy_url"])
                if "cf_clearance" in config:
                    config["cf_clearance"] = self._normalize_cf(config["cf_clearance"])

            return config
        except Exception as e:
            raise Exception(f"[Setting] Configuration loading failed: {e}") from e
    
    async def reload(self) -> None:
        """Reload configuration"""
        self.global_config = self.load("global")
        self.grok_config = self.load("grok")
    
    async def _save_file(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """Save to file"""
        async with aiofiles.open(self.config_path, "r", encoding="utf-8") as f:
            content = await f.read()
            config = toml.loads(content)
        
        for section, data in updates.items():
            if section in config:
                config[section].update(data)
        
        async with aiofiles.open(self.config_path, "w", encoding="utf-8") as f:
            await f.write(toml.dumps(config))
    
    async def _save_storage(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """Save to storage"""
        config = await self._storage.load_config()
        
        for section, data in updates.items():
            if section in config:
                config[section].update(data)
        
        await self._storage.save_config(config)
    
    def _prepare_grok(self, grok: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare Grok configuration (remove prefix)"""
        processed = grok.copy()
        if "cf_clearance" in processed:
            cf = processed["cf_clearance"]
            if cf and cf.startswith("cf_clearance="):
                processed["cf_clearance"] = cf.replace("cf_clearance=", "", 1)
        return processed

    async def save(self, global_config: Optional[Dict[str, Any]] = None, grok_config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration"""
        updates = {}
        
        if global_config:
            updates["global"] = global_config
        if grok_config:
            updates["grok"] = self._prepare_grok(grok_config)
        
        # Choose storage method
        if self._storage:
            await self._save_storage(updates)
        else:
            await self._save_file(updates)
        
        await self.reload()
    
    async def get_proxy_async(self, proxy_type: Literal["service", "cache"] = "service") -> str:
        """Async get proxy URL (supports proxy pool)
        
        Args:
            proxy_type: Proxy type
                - service: Service proxy (client/upload)
                - cache: Cache proxy (cache)
        """
        from app.core.proxy_pool import proxy_pool
        
        if proxy_type == "cache":
            cache_proxy = self.grok_config.get("cache_proxy_url", "")
            if cache_proxy:
                return cache_proxy
        
        # Get from proxy pool
        return await proxy_pool.get_proxy() or ""
    
    def get_proxy(self, proxy_type: Literal["service", "cache"] = "service") -> str:
        """Get proxy URL (synchronous method, for backward compatibility)
        
        Args:
            proxy_type: Proxy type
                - service: Service proxy (client/upload)
                - cache: Cache proxy (cache)
        """
        from app.core.proxy_pool import proxy_pool
        
        if proxy_type == "cache":
            cache_proxy = self.grok_config.get("cache_proxy_url", "")
            if cache_proxy:
                return cache_proxy
        
        # Return current proxy (if using proxy pool, return the last fetched one)
        return proxy_pool.get_current_proxy() or self.grok_config.get("proxy_url", "")

    def get_service_proxy(self) -> str:
        """Get service proxy URL (backward compatibility)"""
        return self.get_proxy("service")

    def get_cache_proxy(self) -> str:
        """Get cache proxy URL (backward compatibility)"""
        return self.get_proxy("cache")


# Global instance
setting = ConfigManager()
