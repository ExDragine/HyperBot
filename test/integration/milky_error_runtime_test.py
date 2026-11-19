# ruff: noqa: S101
"""A real v3 tool failure must log useful diagnostics once without resending."""

import subprocess
import sys
from pathlib import Path


def test_real_milky_failure_logs_and_projection_deduplication(tmp_path):
    script = r'''
import asyncio
import logging
import sys
sys.path.insert(0, sys.argv[1])
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from nonebot.adapters.milky.exception import ActionFailed
from utils.agents.progress import collect_progress, finish_progress_collection
from utils.agents.tool_errors import ToolExecutionUncertainError, tool_error_middleware

records, calls = [], []
class Capture(logging.Handler):
    def emit(self, record):
        records.append(record)
for name in ("utils.agents.tool_errors", "utils.agents.progress"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    logger.addHandler(Capture())

class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

@tool
async def send_plain_text(text: str) -> str:
    """Send a synthetic QQ message."""
    calls.append(text)
    raise ActionFailed(status="failed", retcode=1200,
                       message="消息过长: " + text + " token=private-token",
                       data={"user_id": 123456789, "secret": "private-data"})

async def main():
    model = Model(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "send_plain_text", "args": {"text": "private outgoing body"}, "id": "send-1",
        }]),
        AIMessage(content="must not continue"),
    ])
    graph = create_agent(model, tools=[send_plain_text], middleware=[tool_error_middleware()])
    stream = await graph.astream_events({"messages": [{"role": "user", "content": "send"}]}, version="v3")
    progress = asyncio.create_task(collect_progress(stream, None))
    try:
        await stream.output()
        raise AssertionError("mutating tool failure must stop the graph")
    except ToolExecutionUncertainError as exc:
        assert str(exc) == "send_plain_text"
    finally:
        await stream.abort()
        await finish_progress_collection(progress)
    assert calls == ["private outgoing body"]
    failures = [r.getMessage() for r in records if r.name == "utils.agents.tool_errors"]
    assert len(failures) == 1 and "milky_retcode=1200" in failures[0] and "消息过长" in failures[0]
    warnings = [r.getMessage() for r in records if r.name == "utils.agents.progress"]
    assert len(warnings) == 1, warnings
    assert set(warnings[0].split("projections=", 1)[1].split(",")) == {"messages", "subagents", "tool_calls"}
    for value in ("private outgoing body", "private-token", "private-data", "123456789"):
        assert value not in "\n".join(failures + warnings)

asyncio.run(main())
'''
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        cwd=tmp_path, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_forward_rejection_can_finish_turn_as_plain_text(tmp_path):
    script = r'''
import asyncio
import sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from nonebot.adapters.milky.exception import ActionFailed
from tools import milky_message
from utils.agents.tool_errors import tool_error_middleware

class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

calls = []
async def send(group_id, message):
    assert group_id == 123
    segment, = message
    calls.append(segment)
    if segment.type == "forward":
        raise ActionFailed(status="failed", retcode=500, message=(
            "Internal error: PMHQ send failed: cmd=trpc.group.long_msg_interface.MsgService.SsoSendLongMsg "
            "code=-2 build scratch failed pb=<empty>"))
    return SimpleNamespace(message_seq=len(calls), time=123)

text = "文章正文\n" * 260
milky_message.get_bot = lambda: SimpleNamespace(send_group_message=send)
model = Model(responses=[
    AIMessage("", tool_calls=[{"name": "send_forwarded_message", "id": "forward", "args": {
        "message_scene": "group", "messages": [{"user_id": 999, "sender_name": "Bot", "text": text}],
    }}]),
    AIMessage("", tool_calls=[{"name": "send_plain_text", "id": "plain", "args": {"text": text, "fold_long_text": False}}]),
    AIMessage("已改发文字。"),
])
graph = create_agent(model, tools=[milky_message.send_forwarded_message, milky_message.send_plain_text],
                     middleware=[tool_error_middleware()])
result = asyncio.run(graph.ainvoke({"messages": [{"role": "user", "content": "把文章发到这里"}]},
                                   {"configurable": {"group_id": 123, "user_id": 456}}))
forward, plain = [message for message in result["messages"] if isinstance(message, ToolMessage)]
assert forward.status == "error" and "fold_long_text=false" in forward.content
assert plain.status == "success" and "message_seq=3" in plain.content
assert [s.type for s in calls] == ["forward", "text", "text"]
assert "".join(s.data["text"] for s in calls[1:]) == text
'''
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        cwd=tmp_path, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
