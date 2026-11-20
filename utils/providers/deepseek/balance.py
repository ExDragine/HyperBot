"""DeepSeek account balance endpoint."""

from urllib.parse import urlparse, urlunparse

from utils.http_client import get_http_client

from .constants import DEFAULT_BASE_URL

DEFAULT_BALANCE_URL = f"{DEFAULT_BASE_URL}/user/balance"

httpx_client = get_http_client("deepseek_balance")


def _balance_url(base_url: str) -> str:
    raw_base = base_url.strip()
    if not raw_base:
        return DEFAULT_BALANCE_URL

    parsed = urlparse(raw_base)
    if not parsed.scheme or not parsed.netloc:
        return DEFAULT_BALANCE_URL
    return urlunparse((parsed.scheme, parsed.netloc, "/user/balance", "", "", ""))


async def fetch_balance(base_url: str, api_key: str) -> object:
    response = await httpx_client.get(_balance_url(base_url), headers={"Authorization": f"Bearer {api_key}"})
    response.raise_for_status()
    return response.json()
