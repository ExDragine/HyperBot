from langchain_core.tools import tool
from nonebot import logger

from utils.http_client import get_http_client

httpx_client = get_http_client("weather")

NASA_WEATHER_URL = "https://mars.nasa.gov/rss/api/?feed=weather&category=msl&feedtype=json"


@tool(response_format="content")
async def mars_weather() -> str:
    """
    获取火星天气
    Returns:
        火星天气
    """
    try:
        resp = await httpx_client.get(NASA_WEATHER_URL)
        resp.raise_for_status()
        data = resp.json()
        return f"火星天气: {data['descriptions']}"
    except Exception as e:
        logger.error("Mars weather error", exc_info=e)
        return f"❌ 火星天气失败: {e}"
