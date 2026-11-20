# ruff: noqa: S101

import pytest


def test_http_client_uses_httpx2_backend_and_exports_compatibility_types():
    """HTTP 后端切到 httpx2，但业务侧只依赖本地兼容导出。"""
    from utils import http_client as registry

    assert registry.HTTP_BACKEND_NAME == "httpx2"
    assert registry.AsyncClient.__module__.startswith("httpx2")
    assert registry.AsyncHTTPTransport.__module__.startswith("httpx2")
    assert registry.HTTPError.__module__.startswith("httpx2")
    assert registry.ConnectError.__module__.startswith("httpx2")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("skill", "module_name", "client_name", "client_attr"),
    [
        ("astronomy", "heavens_above", "heavens_above", "httpx_client"),
        ("satellite-imagery", "satellite", "satellite", "httpx_client"),
        ("space-weather", "space_weather", "space_weather", "httpx_client"),
        ("space-weather", "mars_weather", "weather", "httpx_client"),
        ("astronomy", "rocket", "rocket", "http_client"),
    ],
)
async def test_tool_module_http_clients_use_registry(load_tool_module, skill, module_name, client_name, client_attr):
    """验证工具模块通过 http_client 注册表获取客户端而非各自创建。"""
    from utils import http_client as registry

    mod = load_tool_module(module_name, skill=skill)
    client = getattr(mod, client_attr, None)

    assert client is not None, f"{client_name}.{client_attr} is None"
    assert registry.get_http_client(client_name) is client


def test_deepseek_balance_client_uses_registry():
    from utils.http_client import get_http_client
    from utils.providers.deepseek import balance

    assert balance.httpx_client is get_http_client("deepseek_balance")


@pytest.mark.asyncio
async def test_http_client_aclose_all_unifies_lifecycle():
    """验证 aclose_all() 统一关闭所有注册客户端。"""
    from utils import http_client as registry

    registry._clients.clear()
    registry._aclose_all_called = False

    client_a = registry.get_http_client("test_a")
    client_b = registry.get_http_client("test_b")
    assert client_a is registry.get_http_client("test_a")
    assert client_b is registry.get_http_client("test_b")

    closed = await registry.aclose_all()
    assert "test_a" in closed
    assert "test_b" in closed
    assert len(registry._clients) == 0

    assert await registry.aclose_all() == []
