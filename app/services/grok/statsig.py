"""Grok Request Headers Management Module"""

import base64
import random
import string
import uuid
from typing import Dict

from app.core.logger import logger
from app.core.config import setting


def _generate_random_string(length: int, use_letters: bool = True) -> str:
    """Generate random string

    Args:
        length: String length
        use_letters: Whether to use letters (True) or numbers+letters (False)

    Returns:
        Random string
    """
    if use_letters:
        # Generate random letters (lowercase)
        return ''.join(random.choices(string.ascii_lowercase, k=length))
    else:
        # Generate random numbers and letters combination (lowercase)
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _generate_statsig_id() -> str:
    """Dynamically generate x-statsig-id

    Randomly selects one of two formats:
    1. e:TypeError: Cannot read properties of null (reading 'children['xxxxx']')
       where xxxxx is a 5-character random string
    2. e:TypeError: Cannot read properties of undefined (reading 'xxxxxxxxxx')
       where xxxxxxxxxx is a 10-character random letter string

    Returns:
        base64 encoded string
    """
    # Randomly select one format
    format_type = random.choice([1, 2])

    if format_type == 1:
        # Format 1: children['xxxxx']
        random_str = _generate_random_string(5, use_letters=False)
        error_msg = f"e:TypeError: Cannot read properties of null (reading 'children['{random_str}']')"
    else:
        # Format 2: 'xxxxxxxxxx'
        random_str = _generate_random_string(10, use_letters=True)
        error_msg = f"e:TypeError: Cannot read properties of undefined (reading '{random_str}')"

    # base64 encoding
    encoded = base64.b64encode(error_msg.encode('utf-8')).decode('utf-8')
    return encoded


def get_dynamic_headers(pathname: str = "/rest/app-chat/conversations/new") -> Dict[str, str]:
    """Get request headers

    Args:
        pathname: Request path

    Returns:
        Headers dictionary
    """
    # Check if dynamic generation is enabled
    dynamic_statsig = setting.grok_config.get("dynamic_statsig", False)

    if dynamic_statsig:
        # Dynamically generate x-statsig-id
        statsig_id = _generate_statsig_id()
        logger.debug(f"[Statsig] Dynamically generated value {statsig_id}")
    else:
        # Use fixed value from config file
        statsig_id = setting.grok_config.get("x_statsig_id")
        logger.debug(f"[Statsig] Using fixed value {statsig_id}")
        if not statsig_id:
            raise ValueError("x_statsig_id not set in config file")

    # Build basic request headers
    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": "application/json" if "upload-file" not in pathname else "text/plain;charset=UTF-8",
        "Connection": "keep-alive",
        "Origin": "https://grok.com",
        "Priority": "u=1, i",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "Sec-Ch-Ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Baggage": "sentry-environment=production,sentry-public_key=b311e0f2690c81f25e2c4cf6d4f7ce1c",
        "x-statsig-id": statsig_id,
        "x-xai-request-id": str(uuid.uuid4())
    }

    return headers