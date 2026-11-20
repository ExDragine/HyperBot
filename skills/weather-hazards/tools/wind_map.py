from langchain_core.tools import tool

from utils.alconna import UniMessage


@tool(response_format="content_and_artifact")
async def get_wind_map(name: str) -> tuple[str, UniMessage | None]:
    """
    获取风切变或涡旋图像

    Args:
        name (str): 图像名称，字符串类型，只有风切图和涡旋图两种选择

    Returns:
        tuple[str, UniMessage | None]: 返回构建好的消息串
    """
    url = {
        "风切图": "https://tropic.ssec.wisc.edu/real-time/westpac/winds/wgmssht.GIF",
        "wind_shear": "https://tropic.ssec.wisc.edu/real-time/westpac/winds/wgmssht.GIF",
        "涡旋图": "https://tropic.ssec.wisc.edu/real-time/westpac/winds/wgmsvor.GIF",
        "vorticity": "https://tropic.ssec.wisc.edu/real-time/westpac/winds/wgmsvor.GIF",
    }
    if name not in url:
        return "❌ 图像名称无效", None
    return "获取成功", UniMessage.image(url=url[name])
