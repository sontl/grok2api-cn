"""Configuration Manager"""

import toml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """Configuration Manager"""

    def __init__(self) -> None:
        """Initialize"""

        # Load environment variables
        self.config_path: Path = Path(__file__).parents[2] / "data" / "setting.toml"
        self.global_config: Dict[str, Any] = self.load("global")
        self.grok_config: Dict[str, Any] = self.load("grok")
        self._storage = None

    def set_storage(self, storage) -> None:
        """Set storage instance"""
        self._storage = storage

    def load(self, section: str) -> Dict[str, Any]:
        """Configuration loader"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = toml.load(f)[section]

                # Automatically convert SOCKS5 to SOCKS5H
                if section == "grok" and "proxy_url" in config:
                    proxy_url = config["proxy_url"]
                    if proxy_url and proxy_url.startswith("socks5://"):
                        config["proxy_url"] = proxy_url.replace("socks5://", "socks5h://", 1)

                # Automatically add prefix to CF Clearance
                if section == "grok" and "cf_clearance" in config:
                    cf_clearance = config["cf_clearance"]
                    if cf_clearance and not cf_clearance.startswith("cf_clearance="):
                        config["cf_clearance"] = f"cf_clearance={cf_clearance}"

                return config
        except Exception as e:
            raise Exception(f"[Setting] Configuration loading failed: {e}")

    async def reload(self) -> None:
        """Reload configuration (for syncing from storage)"""
        self.global_config = self.load("global")
        self.grok_config = self.load("grok")

    async def save(self, global_config: Dict[str, Any] = None, grok_config: Dict[str, Any] = None) -> None:
        """Save configuration to storage"""
        if not self._storage:
            # If no storage is set, use traditional file saving method
            import aiofiles
            async with aiofiles.open(self.config_path, "r", encoding="utf-8") as f:
                content = await f.read()
                config = toml.loads(content)

            if global_config:
                config["global"].update(global_config)
            if grok_config:
                # Process cf_clearance, remove prefix when saving
                processed_grok_config = grok_config.copy()
                if "cf_clearance" in processed_grok_config:
                    cf_clearance = processed_grok_config["cf_clearance"]
                    if cf_clearance and cf_clearance.startswith("cf_clearance="):
                        processed_grok_config["cf_clearance"] = cf_clearance.replace("cf_clearance=", "", 1)
                config["grok"].update(processed_grok_config)

            async with aiofiles.open(self.config_path, "w", encoding="utf-8") as f:
                await f.write(toml.dumps(config))
        else:
            # Use storage abstraction layer
            config_data = await self._storage.load_config()

            if global_config:
                config_data["global"].update(global_config)
            if grok_config:
                # Process cf_clearance, remove prefix when saving
                processed_grok_config = grok_config.copy()
                if "cf_clearance" in processed_grok_config:
                    cf_clearance = processed_grok_config["cf_clearance"]
                    if cf_clearance and cf_clearance.startswith("cf_clearance="):
                        processed_grok_config["cf_clearance"] = cf_clearance.replace("cf_clearance=", "", 1)
                config_data["grok"].update(processed_grok_config)

            await self._storage.save_config(config_data)

        # Reload configuration
        await self.reload()

    def get_service_proxy(self) -> str:
        """Get service proxy URL (for client and upload)"""
        return self.grok_config.get("proxy_url", "")

    def get_cache_proxy(self) -> str:
        """Get cache proxy URL (for cache)

        Logic:
        - If only proxy_url is set, both cache and service use proxy_url
        - If both proxy_url and cache_proxy_url are set, cache uses cache_proxy_url
        """
        cache_proxy = self.grok_config.get("cache_proxy_url", "")
        service_proxy = self.grok_config.get("proxy_url", "")

        # If cache_proxy_url is set, use it preferentially
        if cache_proxy:
            return cache_proxy

        # Otherwise use proxy_url (service proxy)
        return service_proxy

# Global settings
setting = ConfigManager()

if __name__ == "__main__":
    print(setting.global_config)
    print(setting.grok_config)