import importlib
import pkgutil
from collections.abc import Iterator
from pathlib import Path

from langchain_core.tools import BaseTool

from utils.ens_gate import _ens_prefix
from utils.providers import provider_tools
from utils.providers.deepseek.web import TOOL_NAMES as DEEPSEEK_SEARCH_TOOLS
from utils.providers.deepseek.web import deepseek_search_available

from .mcp_client import mcp_get_tools_async

_EXCLUDED_MODULES = {"__init__", "mcp_client"}

_ALL_TOOL_GROUPS = ("main", "skills", "external", "restricted")

_RESTRICTED_TOOL_MODULES = frozenset({"webpage_screenshot", "webpage_recording"})

_READ_ONLY_MODULES = {
    "comet", "deepseek_search", "earthquake", "iching", "radar",
    "mars_weather", "rocket", "space_weather", "tarot",
}
_READ_ONLY_PREFIXES = {
    "milky_file": ("get_",),
    "milky_friend": ("get_",),
    "milky_group": ("get_",),
    "milky_message": ("get_",),
    "milky_system": ("get_",),
    "scheduled_task": ("list_",),
}

_MODULE_REQUIRED_CAPABILITIES: dict[str, frozenset[str]] = {
    "adapter": frozenset({"qq:message"}),
    "milky_file": frozenset({"qq:file"}),
    "milky_friend": frozenset({"qq:friend"}),
    "milky_group": frozenset({"qq:group"}),
    "milky_message": frozenset({"qq:message"}),
    "milky_system": frozenset({"qq:system"}),
}
_BROAD_QQ_CAPABILITIES = frozenset({"qq", "platform:qq", "qq:tools"})


def _module_tools(name: str, package: str | None = None) -> Iterator[BaseTool]:
    module = importlib.import_module(name, package=package)
    return (item for item in vars(module).values() if isinstance(item, BaseTool))


def _discover_skill_tools(skills_dir: Path) -> Iterator[tuple[str, str, BaseTool]]:
    for tools_dir in sorted(skills_dir.glob("*/tools")):
        if not tools_dir.with_name("SKILL.md").is_file():
            continue
        skill = tools_dir.parent.name
        for module in pkgutil.iter_modules([str(tools_dir)]):
            if module.name.startswith("_"):
                continue
            for tool in _module_tools(f"skills.{skill}.tools.{module.name}"):
                yield skill, module.name, tool


def _discover_tools() -> tuple[
    dict[str, list[BaseTool]],
    dict[str, dict[str, object]],
]:
    """收集普通工具、领域技能工具和 Provider 工具。"""
    tools_dir = Path(__file__).parent
    grouped_tools: dict[str, list[BaseTool]] = {group: [] for group in _ALL_TOOL_GROUPS}
    tool_metadata: dict[str, dict[str, object]] = {}

    for mod_info in pkgutil.iter_modules([str(tools_dir)]):
        if mod_info.name in _EXCLUDED_MODULES:
            continue
        group = "restricted" if mod_info.name in _RESTRICTED_TOOL_MODULES else "main"
        for tool in _module_tools(f".{mod_info.name}", __package__):
            grouped_tools[group].append(tool)
            tool_metadata[tool.name] = {
                "module": mod_info.name,
                "group": group,
                "platform": "qq" if mod_info.name in _MODULE_REQUIRED_CAPABILITIES else None,
                "required_capabilities": _MODULE_REQUIRED_CAPABILITIES.get(mod_info.name, frozenset()),
            }

    for skill, module, tool in _discover_skill_tools(tools_dir.parent / "skills"):
        grouped_tools["skills"].append(tool)
        tool_metadata[tool.name] = {"group": "skills", "module": module, "skill": skill}

    for tool, read_only in provider_tools():
        grouped_tools["main"].append(tool)
        tool_metadata[tool.name] = {"group": "main", "read_only": read_only}

    return grouped_tools, tool_metadata


class ModuleTools:
    def __init__(self):
        (
            self.subagent_tools,
            self.tool_metadata,
        ) = _discover_tools()

        self._registered_tool_ids = {
            id(tool) for group in ("main", "skills") for tool in self.subagent_tools[group]
        }

    @property
    def mcp_tools(self):
        return self.subagent_tools["external"]

    async def initialize(self):
        self.subagent_tools["external"] = await mcp_get_tools_async()

    @property
    def restricted_tools(self):
        return self.subagent_tools.get("restricted", [])

    @property
    def skill_tools(self) -> list[BaseTool]:
        """Return domain tools eligible for disclosure in the current turn."""
        professional_ens = _ens_prefix.get() == "vep"
        return [
            tool for tool in self.subagent_tools["skills"]
            if tool.name != "ens_professional" or professional_ens
        ]

    def _capability_allows(self, tool: BaseTool, capabilities: frozenset[str]) -> bool:
        """Return whether a tool is available to an explicit adapter context.

        Common tools have no required capabilities and remain available to all
        platforms.  Platform tools declare a capability at module discovery
        time.  Broad QQ capabilities are useful for the legacy QQ adapter,
        while a scoped capability (for example ``qq:message``) exposes only
        that tool family.
        """

        if not capabilities:
            return True
        metadata = self.tool_metadata.get(getattr(tool, "name", ""), {})
        required = metadata.get("required_capabilities") or frozenset()
        if not required:
            return True
        if _BROAD_QQ_CAPABILITIES & capabilities and metadata.get("platform") == "qq":
            return True
        return bool(required & capabilities)

    def _filtered_main_tools(self, capabilities: frozenset[str]) -> list[BaseTool]:
        search_available = deepseek_search_available()
        return [
            tool for tool in self.subagent_tools["main"]
            if self._capability_allows(tool, capabilities)
            and (tool.name not in DEEPSEEK_SEARCH_TOOLS or search_available)
        ]

    def direct_tools_for(self, capabilities: frozenset[str]) -> list[BaseTool]:
        """Return direct tools visible in an explicit capability context."""

        return self._filtered_main_tools(capabilities) + self.mcp_tools

    @property
    def direct_tools(self):
        """Return regular Agent tools, including network search and page reading."""
        return [*self._filtered_main_tools(frozenset()), *self.mcp_tools]

    def is_read_only_tool(self, tool: BaseTool) -> bool:
        """Classify registered query objects without trusting name prefixes alone."""
        if id(tool) not in self._registered_tool_ids:
            return False
        if getattr(tool, "response_format", None) != "content":
            return False
        metadata = self.tool_metadata.get(tool.name, {})
        module = metadata.get("module", "")
        return (
            metadata.get("read_only") is True
            or module in _READ_ONLY_MODULES
            or tool.name.startswith(_READ_ONLY_PREFIXES.get(module, ()))
        )


_AGENT_TOOLS = None


def __getattr__(name):
    if name == "agent_tools":
        global _AGENT_TOOLS
        if _AGENT_TOOLS is None:
            _AGENT_TOOLS = ModuleTools()
        return _AGENT_TOOLS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
