# ruff: noqa: S101
"""Real graph coverage for reading, calling, and forgetting tool-bearing skills."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.agent_eval
@pytest.mark.usefixtures("offline_agent_eval")
def test_domain_tools_require_a_skill_read_and_reset_at_the_next_turn(tmp_path):
    script = r'''
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

import yaml
from agentevals.trajectory.match import create_trajectory_match_evaluator
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field
from tools import ModuleTools
from utils.agents.skill_tools import TurnSkillsMiddleware
from utils.agents.workspace import SKILLS_BACKEND_PATH, build_agent_backend
from utils.providers.deepseek.tools import balance

TARGET = "get_china_earthquake"
SKILL = "weather-hazards"
SKILL_FILE = f"{SKILLS_BACKEND_PATH}/{SKILL}/SKILL.md"
DIRECT_TOOLS = {"get_deepseek_api_balance", "deepseek_ocr"}

def tool_call(name, id_, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])

class RecordingModel(FakeMessagesListChatModel):
    visible: list[set[str]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self.bind(visible_tools={item.name for item in tools})

    def _generate(self, messages, *args, **kwargs):
        self.visible.append(kwargs.pop("visible_tools"))
        return super()._generate(messages, *args, **kwargs)

calls = []

@tool
def get_china_earthquake() -> str:
    "Return offline earthquake data for this contract test."
    calls.append(TARGET)
    return "earthquake-data"

registry = ModuleTools()
backend = build_agent_backend(str(Path.cwd() / "sandbox"), "skill-test")
manifests = {}
skill_dirs = backend.ls(SKILLS_BACKEND_PATH)
assert skill_dirs.error is None
paths = [entry["path"].rstrip("/") + "/SKILL.md" for entry in skill_dirs.entries if entry["is_dir"]]
for manifest in backend.download_files(paths):
    assert manifest.error is None, manifest
    frontmatter = yaml.safe_load(manifest.content.decode().split("---", 2)[1])
    names = (frontmatter.get("metadata") or {}).get("include_tools", "").split()
    if names:
        manifests[frontmatter["name"]] = set(names)

registered = {item.name for item in registry.subagent_tools["skills"]}
declared = set().union(*manifests.values())
assert declared == registered, (declared - registered, registered - declared)
assert sum(len(names) for names in manifests.values()) == len(registered)
assert {
    "get_static_china_radar", "get_fy4b_satellite_image", "get_fy4b_cloud_map",
    "get_fy4b_geos_cloud_map", "get_himawari_satellite_image",
} <= manifests[SKILL]
assert all(registry.tool_metadata[name]["skill"] == skill for skill, names in manifests.items() for name in names)
assert not declared.intersection(tool.name for tool in registry.direct_tools)
assert not declared.intersection(tool.name for tool in registry.restricted_tools)
assert "ens_professional" not in {tool.name for tool in registry.skill_tools}

balance.get_provider_profile = lambda _name: {"api_key": ""}
model = RecordingModel(responses=[
    AIMessage(content="", tool_calls=[
        {"name": TARGET, "args": {}, "id": "before"},
        {"name": "get_deepseek_api_balance", "args": {}, "id": "balance"},
        {"name": "deepseek_ocr", "args": {"file_path": "/missing.pdf"}, "id": "ocr"},
    ]),
    tool_call("read_file", "read-skill", file_path=SKILL_FILE),
    tool_call(TARGET, "after"),
    AIMessage("First answer"),
    tool_call(TARGET, "next-turn"),
    AIMessage("Unrelated answer"),
    AIMessage("Pinned answer"),
    AIMessage("Unrelated after pin"),
])
tools = [item if item.name != TARGET else get_china_earthquake for item in registry.skill_tools]
agent = create_deep_agent(
    model=model, backend=backend, subagents=[], skills=[SKILLS_BACKEND_PATH],
    tools=[item for item in registry.direct_tools if item.name in DIRECT_TOOLS],
    middleware=[TurnSkillsMiddleware(backend=backend, sources=[SKILLS_BACKEND_PATH], tools=tools)],
    checkpointer=InMemorySaver(),
)
config = {"configurable": {"thread_id": "skill-test"}}

async def main():
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "查地震"}]}, config)
    assert calls == [TARGET]
    outputs = {message.tool_call_id: message for message in result["messages"] if isinstance(message, ToolMessage)}
    assert outputs["before"].status == "error"
    assert outputs["balance"].status == "success"
    assert "未配置 DeepSeek API Key" in outputs["balance"].content
    assert json.loads(outputs["ocr"].content)["code"] == "file_not_found"
    assert all(DIRECT_TOOLS <= visible for visible in model.visible)
    assert outputs["read-skill"].status == "success"
    assert outputs["after"].content == "earthquake-data"
    next_turn = await agent.ainvoke({"messages": [{"role": "user", "content": "聊点别的"}]}, config)
    next_outputs = {
        message.tool_call_id: message for message in next_turn["messages"] if isinstance(message, ToolMessage)
    }
    assert next_outputs["next-turn"].status == "error"
    assert calls == [TARGET]
    successful_skill_calls = [
        AIMessage(content="", tool_calls=[call])
        for message in next_turn["messages"] if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] in {"read_file", TARGET} and next_outputs[call["id"]].status == "success"
    ]
    match = create_trajectory_match_evaluator(trajectory_match_mode="strict")
    evaluation = match(
        outputs=successful_skill_calls,
        reference_outputs=[
            tool_call("read_file", "reference-read", file_path=SKILL_FILE),
            tool_call(TARGET, "reference-weather"),
        ],
    )
    assert evaluation["score"], evaluation
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "查雷达"}], "pinned_skills": [SKILL]}, config,
    )
    await agent.ainvoke({"messages": [{"role": "user", "content": "继续闲聊"}]}, config)
    assert [visible & declared for visible in model.visible] == [
        set(), set(), manifests[SKILL], manifests[SKILL], set(), set(), manifests[SKILL], set(),
    ]

asyncio.run(main())
'''
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[2])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
