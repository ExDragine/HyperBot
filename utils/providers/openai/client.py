"""First-party OpenAI SDK clients scoped to a configured provider profile."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from openai.resources.realtime.realtime import AsyncRealtimeConnection

from utils.providers.base import clean

from . import is_official_openai_base, resolve_openai_base_url


class OpenAIProfileError(ValueError):
    pass


@asynccontextmanager
async def openai_client(profile_name: str = "openai", *, timeout: float = 60.0) -> AsyncIterator[AsyncOpenAI]:
    from utils.configs import get_provider_profile

    profile = get_provider_profile(profile_name)
    base_url = resolve_openai_base_url(profile.get("base_url"))
    if profile.get("type") != "openai" or not is_official_openai_base(base_url):
        raise OpenAIProfileError("OpenAI 原生接口需要官方 OpenAI provider profile")
    if not (api_key := clean(profile.get("api_key"))):
        raise OpenAIProfileError("OpenAI provider profile 缺少 API key")
    async with AsyncOpenAI(
        api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0,
        http_client=DefaultAsyncHttpxClient(follow_redirects=False),
    ) as client:
        yield client


@asynccontextmanager
async def realtime_session(profile_name: str, *, model: str) -> AsyncIterator[AsyncRealtimeConnection]:
    async with openai_client(profile_name) as client:
        async with client.realtime.connect(model=model, max_retries=0) as connection:
            yield connection
