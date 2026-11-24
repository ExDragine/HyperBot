"""Operator entry point for explicit OpenAI platform requests."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import get_args

from openai import APIStatusError, OpenAIError

from utils.providers.openai.platform import BatchEndpoint, CompletionWindow, openai_platform


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="openai", help="Official provider profile from env.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("conversation-create")
    for name in ("conversation-get", "conversation-delete", "conversation-items"):
        command = commands.add_parser(name)
        command.add_argument("conversation_id")
        if name == "conversation-items":
            command.add_argument("--after")
    for name in ("respond", "background-start", "embed", "moderate"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=name in {"respond", "background-start"})
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--text")
        source.add_argument("--json", type=Path, help="Read an API input value from this JSON file")
        if name == "respond":
            command.add_argument("--conversation-id")
        if name == "background-start":
            command.add_argument("--completion-window", choices=get_args(CompletionWindow), default="soonest")
        if name == "embed":
            command.add_argument("--dimensions", type=int)
    for name, argument in (
        ("response-get", "response_id"), ("response-cancel", "response_id"), ("batch-upload", "path"),
        ("batch-create", "file_id"), ("batch-get", "batch_id"), ("batch-cancel", "batch_id"),
        ("batch-download", "file_id"),
    ):
        command = commands.add_parser(name)
        command.add_argument(argument, type=Path if argument == "path" else str)
        if name == "batch-create":
            command.add_argument("--endpoint", choices=get_args(BatchEndpoint), default="/v1/responses")
    return parser


_METHODS = {
    "conversation-create": "create_conversation",
    "conversation-get": "retrieve_conversation",
    "conversation-delete": "delete_conversation",
    "conversation-items": "conversation_items",
    "respond": "respond",
    "background-start": "start_background_response",
    "response-get": "retrieve_response",
    "response-cancel": "cancel_response",
    "batch-upload": "upload_batch_file",
    "batch-create": "create_batch",
    "batch-get": "retrieve_batch",
    "batch-cancel": "cancel_batch",
    "embed": "embed",
    "moderate": "moderate",
}


async def _run(args: argparse.Namespace) -> None:
    kwargs = vars(args).copy()
    profile, command = kwargs.pop("profile"), kwargs.pop("command")
    input_path = kwargs.pop("json", None)
    input_text = kwargs.pop("text", None)
    if input_path is not None:
        input_data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(input_data, (str, list)):
            raise ValueError("JSON 输入必须是字符串或数组")
        kwargs["input_data"] = input_data
    elif input_text is not None:
        kwargs["input_data"] = input_text
    async with openai_platform(profile) as platform:
        if command == "batch-download":
            async for chunk in platform.batch_file_content(kwargs["file_id"]):
                sys.stdout.buffer.write(chunk)
            return
        invoke = getattr(platform, _METHODS[command])
        result = await invoke(**{key: value for key, value in kwargs.items() if value is not None})
    print(result.model_dump_json(indent=2))


def main() -> None:
    args = _parser().parse_args()
    try:
        asyncio.run(_run(args))
    except APIStatusError as exc:
        sys.exit(f"OpenAI HTTP {exc.status_code}")
    except json.JSONDecodeError as exc:
        sys.exit(f"JSON 格式错误，位置 {exc.lineno}:{exc.colno}")
    except (OpenAIError, OSError, ValueError) as exc:
        sys.exit(f"OpenAI 请求未完成（{type(exc).__name__}）")


if __name__ == "__main__":
    main()
