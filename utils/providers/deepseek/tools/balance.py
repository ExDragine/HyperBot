from langchain_core.tools import tool
from nonebot import logger

from utils.configs import get_provider_profile

from ..balance import fetch_balance


def _format_balance(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("响应不是 JSON 对象")

    available = "可用" if data.get("is_available") else "不可用"
    balance_infos = data.get("balance_infos")
    if not isinstance(balance_infos, list):
        raise ValueError("响应缺少 balance_infos 数组")
    if not balance_infos:
        return f"DeepSeek API 余额：{available}\n余额明细：无"

    lines = [f"DeepSeek API 余额：{available}"]
    for item in balance_infos:
        if not isinstance(item, dict):
            raise ValueError("balance_infos 包含非对象条目")
        currency = item.get("currency", "UNKNOWN")
        total = item.get("total_balance", "0")
        granted = item.get("granted_balance", "0")
        topped_up = item.get("topped_up_balance", "0")
        lines.append(f"- {currency} 总余额 {total}，赠金 {granted}，充值 {topped_up}")
    return "\n".join(lines)


@tool(response_format="content")
async def get_deepseek_api_balance() -> str:
    """查询 DeepSeek API 账户余额。"""
    profile = get_provider_profile("deepseek")
    api_key = str(profile.get("api_key", "")).strip()
    if not api_key:
        return "未配置 DeepSeek API Key：请在 env.toml 的 [providers.deepseek].api_key 中填写。"

    try:
        return _format_balance(await fetch_balance(str(profile.get("base_url", "")), api_key))
    except Exception as e:
        logger.error("DeepSeek balance query error", exc_info=e)
        return f"获取 DeepSeek API 余额失败: {e}"
