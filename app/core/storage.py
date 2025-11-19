"""Storage abstraction layer - supports file, MySQL and Redis storage"""

import os
import json
import toml
import asyncio
import warnings
import aiofiles
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from abc import ABC, abstractmethod
from urllib.parse import urlparse, unquote

from app.core.logger import logger


StorageMode = Literal["file", "mysql", "redis"]


class BaseStorage(ABC):
    """Storage base class"""

    @abstractmethod
    async def init_db(self) -> None:
        """Initialize database"""
        pass

    @abstractmethod
    async def load_tokens(self) -> Dict[str, Any]:
        """Load token data"""
        pass

    @abstractmethod
    async def save_tokens(self, data: Dict[str, Any]) -> None:
        """Save token data"""
        pass

    @abstractmethod
    async def load_config(self) -> Dict[str, Any]:
        """Load configuration data"""
        pass

    @abstractmethod
    async def save_config(self, data: Dict[str, Any]) -> None:
        """Save configuration data"""
        pass


class FileStorage(BaseStorage):
    """File storage implementation"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.token_file = data_dir / "token.json"
        self.config_file = data_dir / "setting.toml"
        self._token_lock = asyncio.Lock()
        self._config_lock = asyncio.Lock()

    async def init_db(self) -> None:
        """Initialize file storage"""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize token file
        if not self.token_file.exists():
            await self._write_file(self.token_file, json.dumps({"sso": {}, "ssoSuper": {}}, indent=2, ensure_ascii=False))
            logger.info("[Storage] Creating new token file")

        # Initialize configuration file
        if not self.config_file.exists():
            default_config = {
                "global": {"api_keys": [], "admin_username": "admin", "admin_password": "admin"},
                "grok": {"proxy_url": "", "cf_clearance": "", "x_statsig_id": ""}
            }
            await self._write_file(self.config_file, toml.dumps(default_config))
            logger.info("[Storage] Creating new configuration file")

    async def _read_file(self, file_path: Path) -> str:
        """Read file content"""
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            return await f.read()

    async def _write_file(self, file_path: Path, content: str) -> None:
        """Write file content"""
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)

    async def _load_json(self, file_path: Path, default: Dict[str, Any], lock: asyncio.Lock) -> Dict[str, Any]:
        """Load JSON file"""
        try:
            async with lock:
                if not file_path.exists():
                    return default
                return json.loads(await self._read_file(file_path))
        except Exception as e:
            logger.error(f"[Storage] Failed to load {file_path.name}: {e}")
            return default

    async def _save_json(self, file_path: Path, data: Dict[str, Any], lock: asyncio.Lock) -> None:
        """Save JSON file"""
        try:
            async with lock:
                await self._write_file(file_path, json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"[Storage] Failed to save {file_path.name}: {e}")
            raise

    async def _load_toml(self, file_path: Path, default: Dict[str, Any], lock: asyncio.Lock) -> Dict[str, Any]:
        """Load TOML file"""
        try:
            async with lock:
                if not file_path.exists():
                    return default
                return toml.loads(await self._read_file(file_path))
        except Exception as e:
            logger.error(f"[Storage] Failed to load {file_path.name}: {e}")
            return default

    async def _save_toml(self, file_path: Path, data: Dict[str, Any], lock: asyncio.Lock) -> None:
        """Save TOML file"""
        try:
            async with lock:
                await self._write_file(file_path, toml.dumps(data))
        except Exception as e:
            logger.error(f"[Storage] Failed to save {file_path.name}: {e}")
            raise

    async def load_tokens(self) -> Dict[str, Any]:
        """Load token data"""
        return await self._load_json(self.token_file, {"sso": {}, "ssoSuper": {}}, self._token_lock)

    async def save_tokens(self, data: Dict[str, Any]) -> None:
        """Save token data"""
        await self._save_json(self.token_file, data, self._token_lock)

    async def load_config(self) -> Dict[str, Any]:
        """Load configuration data"""
        return await self._load_toml(self.config_file, {"global": {}, "grok": {}}, self._config_lock)

    async def save_config(self, data: Dict[str, Any]) -> None:
        """Save configuration data"""
        await self._save_toml(self.config_file, data, self._config_lock)


class MysqlStorage(BaseStorage):
    """MySQL storage implementation"""

    def __init__(self, database_url: str, data_dir: Path):
        self.database_url = database_url
        self.data_dir = data_dir
        self._pool = None
        self._file = FileStorage(data_dir)

    async def init_db(self) -> None:
        """Initialize MySQL"""
        try:
            import aiomysql
            parsed = self._parse_url(self.database_url)
            logger.info(f"[Storage] Parsing database connection: {parsed['user']}@{parsed['host']}:{parsed['port']}/{parsed['db']}")

            # Create database
            await self._create_db(parsed)

            # Create connection pool
            self._pool = await aiomysql.create_pool(
                host=parsed['host'], port=parsed['port'], user=parsed['user'],
                password=parsed['password'], db=parsed['db'], charset="utf8mb4",
                autocommit=True, maxsize=10
            )

            # Create tables
            await self._create_tables()

            # Initialize file storage and synchronize data
            await self._file.init_db()
            await self._sync_data()

        except ImportError:
            raise Exception("aiomysql not installed")
        except Exception as e:
            logger.error(f"[Storage] MySQL initialization failed: {e}")
            raise

    def _parse_url(self, url: str) -> Dict[str, Any]:
        """Parse database URL"""
        parsed = urlparse(url)
        return {
            'user': unquote(parsed.username) if parsed.username else "",
            'password': unquote(parsed.password) if parsed.password else "",
            'host': parsed.hostname,
            'port': parsed.port or 3306,
            'db': parsed.path[1:] if parsed.path else "grok2api"
        }

    async def _create_db(self, parsed: Dict[str, Any]) -> None:
        """Create database"""
        import aiomysql
        temp_pool = await aiomysql.create_pool(
            host=parsed['host'], port=parsed['port'], user=parsed['user'],
            password=parsed['password'], charset="utf8mb4", autocommit=True, maxsize=1
        )

        try:
            async with temp_pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*database exists')
                        await cursor.execute(
                            f"CREATE DATABASE IF NOT EXISTS `{parsed['db']}` "
                            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                        )
                    logger.info(f"[Storage] MySQL database '{parsed['db']}' is ready")
        finally:
            temp_pool.close()
            await temp_pool.wait_closed()

    async def _create_tables(self) -> None:
        """Create tables"""
        tables = {
            "grok_tokens": """
                CREATE TABLE IF NOT EXISTS grok_tokens (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    data JSON NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            "grok_settings": """
                CREATE TABLE IF NOT EXISTS grok_settings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    data JSON NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        }

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*already exists')
                    for sql in tables.values():
                        await cursor.execute(sql)
                logger.info("[Storage] MySQL tables created/verified successfully")

    async def _sync_data(self) -> None:
        """Synchronize data"""
        try:
            for table, key in [("grok_tokens", "sso"), ("grok_settings", "global")]:
                data = await self._load_db(table)
                if data:
                    if table == "grok_tokens":
                        await self._file.save_tokens(data)
                    else:
                        await self._file.save_config(data)
                    logger.info(f"[Storage] {table.split('_')[1]} data synchronized from database to file")
                else:
                    if table == "grok_tokens":
                        file_data = await self._file.load_tokens()
                        if file_data.get(key) or file_data.get("ssoSuper"):
                            await self._save_db(table, file_data)
                            logger.info("[Storage] Token data initialized from file to database")
                    else:
                        file_data = await self._file.load_config()
                        if file_data.get(key) or file_data.get("grok"):
                            await self._save_db(table, file_data)
                            logger.info("[Storage] Configuration data initialized from file to database")
        except Exception as e:
            logger.warning(f"[Storage] Data synchronization failed: {e}")

    async def _load_db(self, table: str) -> Optional[Dict[str, Any]]:
        """Load data from database"""
        try:
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(f"SELECT data FROM {table} ORDER BY id DESC LIMIT 1")
                    result = await cursor.fetchone()
                    return json.loads(result[0]) if result else None
        except Exception as e:
            logger.error(f"[Storage] Failed to load {table} from database: {e}")
            return None

    async def _save_db(self, table: str, data: Dict[str, Any]) -> None:
        """Save data to database"""
        try:
            async with self._pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    json_data = json.dumps(data, ensure_ascii=False)
                    await cursor.execute(f"SELECT id FROM {table} ORDER BY id DESC LIMIT 1")
                    result = await cursor.fetchone()

                    if result:
                        await cursor.execute(f"UPDATE {table} SET data = %s WHERE id = %s", (json_data, result[0]))
                    else:
                        await cursor.execute(f"INSERT INTO {table} (data) VALUES (%s)", (json_data,))
        except Exception as e:
            logger.error(f"[Storage] Failed to save data to {table}: {e}")
            raise

    async def load_tokens(self) -> Dict[str, Any]:
        """Load token data"""
        return await self._file.load_tokens()

    async def save_tokens(self, data: Dict[str, Any]) -> None:
        """Save token data"""
        await self._file.save_tokens(data)
        await self._save_db("grok_tokens", data)

    async def load_config(self) -> Dict[str, Any]:
        """Load configuration data"""
        return await self._file.load_config()

    async def save_config(self, data: Dict[str, Any]) -> None:
        """Save configuration data"""
        await self._file.save_config(data)
        await self._save_db("grok_settings", data)

    async def close(self) -> None:
        """Close connection"""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            logger.info("[Storage] MySQL connection pool closed")


class RedisStorage(BaseStorage):
    """Redis storage implementation"""

    def __init__(self, redis_url: str, data_dir: Path):
        self.redis_url = redis_url
        self.data_dir = data_dir
        self._redis = None
        self._file = FileStorage(data_dir)

    async def init_db(self) -> None:
        """Initialize Redis"""
        try:
            import redis.asyncio as redis
            parsed = self._parse_url(self.redis_url)
            logger.info(f"[Storage] Parsing Redis URL: host={parsed['host']}, port={parsed['port']}, db={parsed.get('db', 0)}, username={parsed.get('username')}, password={'***' if parsed.get('password') else None}")

            # Create Redis connection
            self._redis = redis.Redis(
                host=parsed['host'], port=parsed['port'], password=parsed.get('password'),
                username=parsed.get('username'), db=parsed.get('db', 0),
                encoding="utf-8", decode_responses=True
            )

            # Test connection
            await self._redis.ping()
            logger.info(f"[Storage] Redis connection successful: {parsed['host']}:{parsed['port']}/{parsed['db']}")

            # Initialize file storage and synchronize data
            await self._file.init_db()
            await self._sync_data()

        except ImportError:
            raise Exception("redis not installed")
        except Exception as e:
            logger.error(f"[Storage] Redis initialization failed: {e}")
            raise

    def _parse_url(self, url: str) -> Dict[str, Any]:
        """Parse Redis URL"""
        if url.startswith('redis://'):
            url = url[8:]
        parsed = urlparse(f'//{url}')

        result = {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 6379,
            'db': int(parsed.path.lstrip('/')) if parsed.path and parsed.path != '/' else 0,
            'username': unquote(parsed.username) if parsed.username else None,
            'password': unquote(parsed.password) if parsed.password else None
        }

        # Redis 6+ default username
        if result['password'] and not result['username']:
            result['username'] = 'default'

        return result

    async def _sync_data(self) -> None:
        """Synchronize data"""
        try:
            for key, file_func, key_name in [
                ("grok:tokens", self._file.load_tokens, "sso"),
                ("grok:settings", self._file.load_config, "global")
            ]:
                data = await self._redis.get(key)
                if data:
                    parsed = json.loads(data)
                    if key == "grok:tokens":
                        await self._file.save_tokens(parsed)
                    else:
                        await self._file.save_config(parsed)
                    logger.info(f"[Storage] {key.split(':')[1]} data synchronized from Redis to file")
                else:
                    file_data = await file_func()
                    if file_data.get(key_name) or (key == "grok:tokens" and file_data.get("ssoSuper")):
                        json_data = json.dumps(file_data, ensure_ascii=False)
                        await self._redis.set(key, json_data)
                        logger.info(f"[Storage] {key.split(':')[1]} data initialized from file to Redis")
        except Exception as e:
            logger.warning(f"[Storage] Data synchronization failed: {e}")

    async def _save_redis(self, key: str, data: Dict[str, Any]) -> None:
        """Save to Redis"""
        try:
            await self._redis.set(key, json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.error(f"[Storage] Failed to save to Redis: {e}")
            raise

    async def load_tokens(self) -> Dict[str, Any]:
        """Load token data"""
        return await self._file.load_tokens()

    async def save_tokens(self, data: Dict[str, Any]) -> None:
        """Save token data"""
        await self._file.save_tokens(data)
        await self._save_redis("grok:tokens", data)

    async def load_config(self) -> Dict[str, Any]:
        """Load configuration data"""
        return await self._file.load_config()

    async def save_config(self, data: Dict[str, Any]) -> None:
        """Save configuration data"""
        await self._file.save_config(data)
        await self._save_redis("grok:settings", data)

    async def close(self) -> None:
        """Close connection"""
        if self._redis:
            await self._redis.close()
            logger.info("[Storage] Redis connection closed")


class StorageManager:
    """Storage manager"""

    _instance: Optional['StorageManager'] = None
    _storage: Optional[BaseStorage] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def init(self) -> None:
        """Initialize storage"""
        if self._initialized:
            return

        mode = os.getenv("STORAGE_MODE", "file").lower()
        url = os.getenv("DATABASE_URL", "")
        data_dir = Path(__file__).parents[2] / "data"

        storage_classes = {
            "mysql": MysqlStorage,
            "redis": RedisStorage,
            "file": FileStorage
        }

        if mode in ("mysql", "redis") and not url:
            raise ValueError(f"{mode.upper()} mode requires DATABASE_URL environment variable")

        storage_class = storage_classes.get(mode, FileStorage)
        self._storage = storage_class(url, data_dir) if mode != "file" else storage_class(data_dir)

        await self._storage.init_db()
        self._initialized = True
        logger.info(f"[Storage] Using {mode} storage mode")
        logger.info("[Storage] Storage manager initialization completed")

    def get_storage(self) -> BaseStorage:
        """Get storage instance"""
        if not self._initialized or not self._storage:
            raise RuntimeError("StorageManager not initialized")
        return self._storage

    async def close(self) -> None:
        """Close storage"""
        if self._storage and hasattr(self._storage, 'close'):
            await self._storage.close()


# Global storage manager instance
storage_manager = StorageManager()
