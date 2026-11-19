# ruff: noqa: S101
"""Run dependency contracts in fresh interpreters, outside unit-test module stubs."""

import subprocess
import sys
from pathlib import Path


def test_quoted_images_reach_deepseek_as_supported_encoded_images(tmp_path):
    script = r'''
import asyncio
import base64
import sys
from io import BytesIO
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])

import nonebot
nonebot.init(driver="nonebot.drivers.fastapi:Driver", log_level="WARNING")
nonebot.require("nonebot_plugin_alconna")
from PIL import Image
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from plugins.agent import reply_context
from plugins.agent.chat_context import build_chat_context
from utils.agents.inputs import filter_messages_for_model_capabilities
from utils.configs import EnvConfig
from utils.database import MessageDatabase
from plugins.agent.message_normalizer import NORMALIZED_VERSION, segments_to_raw_json

model_name = "deepseek-v4-flash-vision-exp"
EnvConfig.ADVAN_MODEL = model_name
EnvConfig.ADVAN_MODEL_CAPABILITIES = ["vision"]
models = [
    ChatDeepSeek(model=model_name, api_key="contract-key"),
    ChatOpenAI(model=model_name, api_key="contract-key", base_url="https://api.deepseek.com",
               use_responses_api=True),
]
database = MessageDatabase()
event = SimpleNamespace(self_id="999", data=SimpleNamespace(message_scene="group", peer_id=123))

async def run():
    sequence = 0
    for image_format in ("BMP", "TIFF", "AVIF", "PNG", "JPEG", "GIF", "WEBP"):
        data = BytesIO()
        Image.new("RGB", (4, 4), "red").save(data, format=image_format)
        original = data.getvalue()
        for source in ("cache", "remote", "forward"):
            sequence += 1
            segment = {"type": "image", "data": {"temp_url": "https://example.test/quoted"}}
            segments = [segment] if source != "forward" else [
                {"type": "forward", "data": {"forward_id": "forward-1"}}
            ]
            fetched = []
            async def get(url):
                fetched.append(url)
                return SimpleNamespace(content=original)
            async def get_message(**kwargs):
                return SimpleNamespace(segments=segments)
            async def get_forwarded_messages(**kwargs):
                return [SimpleNamespace(segments=[segment], sender_name="Alice")]
            reply_context._httpx_client = SimpleNamespace(get=get)
            bot = SimpleNamespace(get_message=get_message, get_forwarded_messages=get_forwarded_messages)
            stored = await database.insert(
                time=sequence, msg_id=sequence, user_id=111, group_id=123, user_name="Alice",
                role="user", content="[图片]", normalized_version=NORMALIZED_VERSION,
                normalized_status="complete",
                raw_segments_json=segments_to_raw_json(segments) if source != "forward" else None,
            )
            if source == "cache":
                await database.insert_images(sequence, 111, 123, [original], message_id=stored.message_id)
            payload, images = await reply_context.build_reply_context(bot, event, sequence, 123, database)
            assert images == [original], (image_format, source)
            assert bool(fetched) == (source != "cache")
            messages = build_chat_context(
                payload={"content": "解释引用图片", "reply_to": payload}, history=[],
                images=[], audio=[], videos=[], quoted_images=images, recent_images=[],
                max_bytes=1024 * 1024, max_images=4,
            )
            filtered = filter_messages_for_model_capabilities(messages, model_name, role="advanced")
            for model in models:
                request = model._get_request_payload(filtered)
                outgoing = request.get("messages", request.get("input"))[-1]
                assert outgoing["role"] == "user"
                parts = outgoing["content"]
                image = next(part for part in parts if part["type"] in {"image_url", "input_image"})
                url = image["image_url"]
                if isinstance(url, dict):
                    url = url["url"]
                header, encoded = url.split(",", 1)
                assert header in {
                    "data:image/jpeg;base64", "data:image/png;base64",
                    "data:image/gif;base64", "data:image/webp;base64",
                }, (image_format, source, header)
                decoded = base64.b64decode(encoded)
                with Image.open(BytesIO(decoded)) as actual:
                    assert actual.format in {"JPEG", "PNG", "GIF", "WEBP"}
                    actual.load()
                if image_format in {"PNG", "JPEG", "GIF", "WEBP"}:
                    assert decoded == original

asyncio.run(run())
'''
    result = subprocess.run(  # noqa: S603 - fixed script and repository path
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_deep_agent_stream_and_media_artifact(tmp_path):
    script = r'''
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
import nonebot
nonebot.init(driver="nonebot.drivers.fastapi:Driver", log_level="WARNING")
nonebot.require("nonebot_plugin_alconna")

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from nonebot_plugin_alconna import UniMessage
from pydantic import Field
from utils.agents import cognitive
from utils.agents.runtime import conversation_workspace_key
from utils.configs import EnvConfig

class ContractModel(FakeMessagesListChatModel):
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen_messages.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

@tool(response_format="content_and_artifact")
def illustration() -> tuple[str, UniMessage]:
    """Return a locally generated image for the dependency contract."""
    return "image ready", UniMessage.image(raw=b"contract-image")

model = ContractModel(responses=[
    AIMessage(content="", tool_calls=[{"name": "illustration", "args": {}, "id": "image-1"}]),
    AIMessage(content="contract completed"),
])
cognitive.create_llm = lambda **kwargs: model
cognitive.agent_tools = SimpleNamespace(restricted_tools=[])
EnvConfig.ADVAN_MODEL = "contract-model"
EnvConfig.AGENT_JOB_TIMEOUT_SECONDS = 30
agent = object.__new__(cognitive.FrontierCognitive)
agent.tools = [illustration]
agent.document_subagent = None
agent.working_dir = str(Path.cwd() / "sandbox")
agent.load_system_prompt = lambda group_id: "Answer the current request."

result = asyncio.run(agent.chat_agent(
    [{"role": "user", "content": "Create an illustration."}],
    user_id="contract", user_name="Contract", enable_acp_subagents=False,
))
assert not result.get("error"), result
assert result["status"] == "success", result
assert result["response"]["messages"][-1].content == "contract completed", result
assert result["artifacts"][0].kind == "image", result
assert result["artifacts"][0].data == b"contract-image", result
assert len(model.seen_messages) == 2, model.seen_messages
skill_body = (Path(sys.argv[1]) / "skills" / "rich-markdown" / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[2].strip()
for messages in model.seen_messages:
    pinned = [message for message in messages if message.additional_kwargs.get("lc_source") == "pinned_skill"]
    assert len(pinned) == 1, messages
    assert pinned[0].additional_kwargs["skill"]["name"] == "rich-markdown", pinned[0]
    assert pinned[0].additional_kwargs["skill"]["path"] == "/skills/rich-markdown/SKILL.md", pinned[0]
    assert skill_body in pinned[0].content.replace("\r\n", "\n"), pinned[0]
assert not any(call["name"] == "read_file" for messages in model.seen_messages for message in messages
               if isinstance(message, AIMessage) for call in message.tool_calls)
workspace_key = conversation_workspace_key("contract", None)
assert (Path(agent.working_dir) / "memory" / workspace_key / "SOUL.md").is_file()
'''
    result = subprocess.run(  # noqa: S603 - fixed script and repository path
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_news_agent_uses_json_contract_for_research_and_revision(tmp_path):
    script = r'''
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from plugins.news import editor
from plugins.news.schemas import NewsPayload
from utils.agents import cognitive
from utils.agents.runtime_gateway import FrontierAgentRuntime
from utils.configs import EnvConfig

EnvConfig.ADVAN_MODEL = "contract-model"
EnvConfig.SYSTEM_PROMPT = "QQ-only persona for {name}"
cognitive.model_supports_native_web_search = lambda *_: True
cognitive.native_web_search_tools = lambda *_: ("deepseek_web_search", "deepseek_read_url")
agent = object.__new__(cognitive.FrontierCognitive)
agent.working_dir = str(Path.cwd() / "sandbox")
editor.FrontierAgentRuntime = lambda: FrontierAgentRuntime(agent)

class Model(FakeMessagesListChatModel):
    research: bool

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        system = messages[0].text
        assert "符合请求中 schema 的一个 JSON 对象" in system
        assert "完全缺少合格证据时返回 `{}`" in system
        assert system.count("不要在输出中使用 Markdown 链接。") == 1
        assert "QQ-only persona" not in system and "直接返回聊天正文" not in system
        assert "<workspace_soul>" not in system
        assert ("本轮可用 DeepSeek 服务端联网" in system) is self.research
        task, pinned = messages[1:]
        data = json.loads(task.text)
        assert data["mode"] == ("research" if self.research else "revision")
        assert ("articles" in data) is not self.research
        assert data["schema"] == NewsPayload.model_json_schema()
        assert pinned.additional_kwargs["skill"]["name"] == "daily-news"
        assert "`mode=revision` 时，只使用给定的 `articles` 修订" in pinned.text
        return super()._generate(messages, *args, **kwargs)

now = datetime.now(UTC)
edition = SimpleNamespace(start=now - timedelta(days=1), scheduled_at=now, report_id="test", namespace="test")
news = editor.NewsEditor(SimpleNamespace(target_stories=1, min_stories=1))
article_id = "https://example.com/evidence"
quote = "The city opened its new public library."
expected = NewsPayload(top_stories=[{
    "title": "Library opened", "category": "Local", "summary": quote,
    "evidence": [{"article_id": article_id, "quote": quote}],
}], worth_reading=[])
for research in (True, False):
    model = Model(research=research, responses=[AIMessage(expected.model_dump_json())])
    cognitive.create_llm = lambda **_: model
    data = {"queries": ["test"]} if research else {
        "articles": [{"article_id": article_id, "text": quote}], "validation_error": None,
    }
    messages = asyncio.run(news._run_skill(edition, data, research=research))
    assert NewsPayload.model_validate_json(messages[-1].text) == expected
'''
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_milky_13_adapter_and_tool_schemas(tmp_path):
    script = r'''
import asyncio
import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])

from nonebot.adapters.milky.event import EVENT_CLASSES, GroupDisbandEvent, MessageEvent
from nonebot.adapters.milky.message import Message
from tools import milky_message, milky_file

event = MessageEvent.model_validate({
    "time": 123, "self_id": 999,
    "data": {"message_scene": "friend", "peer_id": 456, "message_seq": 1,
             "sender_id": 456, "time": 123,
             "segments": [{"type": "markdown", "data": {"content": "# hello"}}]},
})
assert event.data.segments[0]["data"]["content"] == "# hello"
assert event.message[0].type == "markdown"
assert EVENT_CLASSES["group_disband"] is GroupDisbandEvent
assert "runtime" not in milky_file.persist_group_file.tool_call_schema.model_json_schema()["properties"]
assert "is_self_send" in milky_file.get_private_file_download_url.args
assert set(milky_message.send_plain_text.tool_call_schema.model_json_schema()["properties"]) == {"text"}
assert not hasattr(milky_message, "send_forwarded_message")
assert milky_message.get_forwarded_messages.name == "get_forwarded_messages"

calls = []
async def send(**kwargs):
    elements = Message(kwargs["message"]).to_elements()
    assert [segment["type"] for segment in elements] == ["text"]
    calls.append((kwargs["group_id"], elements[0]["data"]["text"]))
    return SimpleNamespace(message_seq=len(calls), time=456)

milky_message.get_bot = lambda: SimpleNamespace(send_group_message=send)
async def run():
    article = "长文\n" * 500
    result = await milky_message.send_plain_text.ainvoke(
        {"text": article}, config={"configurable": {"group_id": 789}},
    )
    assert "message_seq=2" in result
    assert calls == [(789, article[:1000]), (789, article[1000:])]
asyncio.run(run())
'''
    result = subprocess.run(  # noqa: S603 - fixed script and repository path
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
