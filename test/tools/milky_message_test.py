# ruff: noqa: S101
import json
import types

import pytest


def _config(group_id=123, user_id="456"):
    return {"configurable": {"group_id": group_id, "user_id": user_id}}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name, group_id, method, target", [
    ("send_plain_text", 123, "send_group_message", {"group_id": 123}),
    ("send_plain_text", None, "send_private_message", {"user_id": 456}),
    ("send_plain_text", 0, "send_private_message", {"user_id": 456}),
    ("send_group_message", 123, "send_group_message", {"group_id": 123}),
    ("send_private_message", None, "send_private_message", {"user_id": 456}),
])
async def test_tool_sent_long_text_uploads_a_utf8_file_to_the_same_target(
    load_tool_module, install_milky_bot, monkeypatch, tool_name, group_id, method, target,
):
    from utils import message as delivery

    async def fail_render(*_args, **_kwargs):
        pytest.fail("plain text must never enter Markdown rendering")

    monkeypatch.setattr(delivery, "markdown_to_image", fail_render)
    monkeypatch.setattr(delivery, "markdown_to_text", fail_render)
    module = load_tool_module("milky_message")
    upload = method.replace("send_", "upload_").replace("_message", "_file")
    bot = install_milky_bot(module, {upload: "txt-id"})
    text = "  ```python\r\nprint('hello')\n```\n" + "第一段📖\n  " * 400
    result = await getattr(module, tool_name)(
        text, config=_config(group_id=group_id), file_title="测试记录",
    )

    assert "测试记录.txt" in result and "file_id=txt-id" in result
    assert bot.calls == [(upload, {**target, "raw": text.encode("utf-8"), "file_name": "测试记录.txt"})]


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [1000, 1001])
async def test_plain_text_uses_file_only_above_character_limit(load_tool_module, install_milky_bot, length):
    module = load_tool_module("milky_message")
    bot = install_milky_bot(module, {
        "send_group_message": types.SimpleNamespace(message_seq=7, time=123),
        "upload_group_file": "txt-id",
    })
    text = "字" * length
    result = await module.send_plain_text(text, config=_config(), file_title=" .txt ")

    if length == 1000:
        assert "message_seq=7" in result
        assert bot.calls == [("send_group_message", {"group_id": 123, "message": text})]
    else:
        assert "file_id=txt-id" in result
        assert bot.calls == [
            ("upload_group_file", {"group_id": 123, "raw": text.encode("utf-8"), "file_name": "长文本.txt"}),
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("title, file_name", [
    (" 项目记录.TXT ", "项目记录.txt"),
    ("../会议:纪要\n", "会议 纪要.txt"),
    ("内容" * 40, "内容" * 30 + ".txt"),
])
async def test_text_attachment_normalizes_title(load_tool_module, install_milky_bot, title, file_name):
    module = load_tool_module("milky_message")
    bot = install_milky_bot(module, {"upload_group_file": "txt-id"})
    await module.send_group_message("字" * 1001, config=_config(), file_title=title)
    assert len(bot.calls) == 1 and bot.calls[0][1]["file_name"] == file_name


@pytest.mark.asyncio
async def test_plain_text_upload_failure_does_not_resend_or_switch_transport(load_tool_module, install_milky_bot):
    module = load_tool_module("milky_message")

    def fail(_kwargs):
        raise TimeoutError

    bot = install_milky_bot(module, {"upload_group_file": fail})
    with pytest.raises(TimeoutError):
        await module.send_plain_text("字" * 1001, config=_config())
    assert [name for name, _ in bot.calls] == ["upload_group_file"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text, config", [
    (" \n", _config()),
    ("hello", {}),
    ("hello", _config(group_id=-1)),
    ("hello", _config(group_id="invalid")),
    ("hello", _config(group_id=None, user_id=0)),
])
async def test_plain_text_rejects_empty_text_or_invalid_context(load_tool_module, install_milky_bot, text, config):
    message = load_tool_module("milky_message")
    bot = install_milky_bot(message)
    result = await message.send_plain_text(text, config=config)
    assert result
    assert not bot.calls


@pytest.mark.asyncio
async def test_plain_text_applies_output_safety_and_does_not_retry_on_failure(
    load_tool_module, monkeypatch,
):
    from utils import message as delivery

    async def sanitize(text):
        assert text == "original"
        return "blocked"

    calls = []

    class Bot:
        async def send_group_message(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("platform rejected")

    message = load_tool_module("milky_message")
    monkeypatch.setattr(delivery, "sanitize_outgoing_text", sanitize)
    monkeypatch.setattr(message, "get_bot", lambda: Bot())
    with pytest.raises(RuntimeError, match="platform rejected"):
        await message.send_plain_text("original", config=_config())
    assert len(calls) == 1
    assert calls[0]["message"] == "blocked"


def _echoed_message(kwargs):
    return types.SimpleNamespace(
        message_scene=kwargs["message_scene"],
        peer_id=kwargs["peer_id"],
        message_seq=kwargs["message_seq"],
        sender_id=456,
        time=1714521600,
        segments=[{"type": "text", "data": {"text": "hello"}}],
    )


@pytest.mark.asyncio
async def test_message_send_and_recall_tools_call_milky(load_tool_module, install_milky_bot):
    message = load_tool_module("milky_message")
    bot = install_milky_bot(
        message,
        {
            "send_private_message": types.SimpleNamespace(message_seq=11, time=1714521600),
            "send_group_message": types.SimpleNamespace(message_seq=22, time=1714521601),
        },
    )

    private_sent = await message.send_private_message(user_id=456, message_text="你好")
    group_sent = await message.send_group_message(message_text="大家好", config=_config())
    private_recalled = await message.recall_private_message(user_id=456, message_seq=11)
    group_recalled = await message.recall_group_message(message_seq=22, config=_config())

    assert "message_seq=11" in private_sent
    assert "message_seq=22" in group_sent
    assert private_recalled == "已撤回私聊 456 的消息 11"
    assert group_recalled == "已撤回群 123 的消息 22"
    assert bot.calls == [
        ("send_private_message", {"user_id": 456, "message": "你好"}),
        ("send_group_message", {"group_id": 123, "message": "大家好"}),
        ("recall_private_message", {"user_id": 456, "message_seq": 11}),
        ("recall_group_message", {"group_id": 123, "message_seq": 22}),
    ]


@pytest.mark.asyncio
async def test_message_query_tools_use_context_and_format_results(load_tool_module, install_milky_bot):
    message = load_tool_module("milky_message")
    bot = install_milky_bot(
        message,
        {
            "get_message": _echoed_message,
            "get_resource_temp_url": "https://example.com/resource",
            "get_forwarded_messages": [
                types.SimpleNamespace(
                    message_seq=33,
                    sender_name="Alice",
                    avatar_url="https://example.com/avatar",
                    time=1714521600,
                    segments=[{"type": "text", "data": {"text": "forward"}}],
                )
            ],
        },
    )

    one = await message.get_message(message_scene="group", message_seq=88, config=_config())
    url = await message.get_resource_temp_url(resource_id="res-1")
    forwarded = await message.get_forwarded_messages(forward_id="fwd-1")
    marked = await message.mark_message_as_read(message_scene="group", message_seq=88, config=_config())

    assert "message_seq=88" in one
    assert "hello" in one
    assert url == "https://example.com/resource"
    assert "forward" in forwarded
    assert marked == "已将 group 会话 123 中消息 88 及之前消息标为已读"
    assert bot.calls == [
        ("get_message", {"message_scene": "group", "peer_id": 123, "message_seq": 88}),
        ("get_resource_temp_url", {"resource_id": "res-1"}),
        ("get_forwarded_messages", {"forward_id": "fwd-1"}),
        ("mark_message_as_read", {"message_scene": "group", "peer_id": 123, "message_seq": 88}),
    ]


@pytest.mark.asyncio
async def test_history_attachment_read_uses_trusted_scope_and_mounted_path(load_tool_module, monkeypatch, tmp_path):
    from plugins.agent import historical_attachments
    from utils.agent_context import FrontierRuntimeContext
    from utils.agent_protocol import ConversationRef

    mod = load_tool_module("milky_message")
    called = []
    bot = types.SimpleNamespace(self_id="999")
    monkeypatch.setattr(mod, "get_bot", lambda account: bot if account == "999" else pytest.fail("wrong account"))

    async def load(actual_bot, source, seq, db, **kwargs):
        called.append((actual_bot, source, seq, kwargs))
        return {"sender_user_id": 111, "attachments": [{"path": "/memory/group-123/files/pic.png"}]}

    monkeypatch.setattr(historical_attachments, "load_historical_attachments", load)
    runtime = types.SimpleNamespace(context=FrontierRuntimeContext(
        user_id="456", group_id=123, group_member_role=None, workspace_dir=str(tmp_path),
        conversation=ConversationRef("qq", "999", "group", "123"),
    ))
    config = {"configurable": {
        "group_id": 888, "user_id": "777", "virtual_roots": {"/memory/group-123/": str(tmp_path)},
    }}
    result = await mod.get_message(
        "group", 77, include_attachments=True, attachment_index=2, config=config, runtime=runtime,
    )
    assert json.loads(result)["sender_user_id"] == 111
    assert called[0][1].peer_id == 123 and called[0][2] == 77
    assert called[0][3] == {"workspace_key": "group-123", "memory_dir": str(tmp_path), "attachment_index": 2}
    for scene, peer in [("group", 888), ("friend", 456), ("temp", 456)]:
        rejected = await mod.get_message(scene, 77, peer_id=peer, include_attachments=True, config=config, runtime=runtime)
        assert "当前会话" in rejected
    assert len(called) == 1


@pytest.mark.asyncio
async def test_history_attachment_requires_scope_and_mount_before_platform_access(load_tool_module, install_milky_bot):
    mod = load_tool_module("milky_message")
    bot = install_milky_bot(mod)
    for config in [{}, _config()]:
        result = await mod.get_message("group", 77, include_attachments=True, config=config)
        assert result and not bot.calls
    result = await mod.get_message("group", 77, attachment_index=1, config=_config())
    assert "include_attachments=true" in result and not bot.calls
