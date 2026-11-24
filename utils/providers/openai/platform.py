"""Explicit first-party operations for OpenAI platform workflows."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI, omit
from openai.types import Batch, CreateEmbeddingResponse, ModerationCreateResponse
from openai.types.conversations import Conversation, ConversationDeletedResource
from openai.types.file_object import FileObject
from openai.types.moderation_multi_modal_input_param import ModerationMultiModalInputParam
from openai.types.responses import Response, ResponseInputItemParam, ResponseInputParam

from .client import openai_client

BatchEndpoint = Literal[
    "/v1/responses", "/v1/chat/completions", "/v1/embeddings", "/v1/moderations",
    "/v1/images/generations", "/v1/images/edits",
]
CompletionWindow = Literal["soonest", "2h", "8h"]


class OpenAIPlatform:
    """Execute platform operations using a client owned by openai_platform()."""

    def __init__(self, client: AsyncOpenAI):
        self._client = client

    async def create_conversation(self, *, items: list[ResponseInputItemParam] | None = None) -> Conversation:
        return await self._client.conversations.create(items=items if items is not None else omit)

    async def retrieve_conversation(self, conversation_id: str) -> Conversation:
        return await self._client.conversations.retrieve(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> ConversationDeletedResource:
        return await self._client.conversations.delete(conversation_id)

    async def conversation_items(self, conversation_id: str, *, limit: int = 20, after: str | None = None):
        return await self._client.conversations.items.list(conversation_id, limit=limit, after=after if after is not None else omit)

    async def respond(
        self, model: str, input_data: ResponseInputParam | str, *, conversation_id: str | None = None,
    ) -> Response:
        return await self._client.responses.create(
            model=model, input=input_data, conversation=conversation_id or omit,
            store=False, include=["reasoning.encrypted_content"],
        )

    async def start_background_response(
        self, model: str, input_data: ResponseInputParam | str, *, completion_window: CompletionWindow = "soonest",
    ) -> Response:
        if completion_window == "soonest":
            return await self._client.responses.create(model=model, input=input_data, background=True, store=False)
        return await self._client.responses.create(
            model=model, input=input_data, background=True, store=False, service_tier="default",
            extra_body={"background_options": {"completion_window": completion_window}},
        )

    async def retrieve_response(self, response_id: str) -> Response:
        return await self._client.responses.retrieve(response_id)

    async def cancel_response(self, response_id: str) -> Response:
        return await self._client.responses.cancel(response_id)

    async def upload_batch_file(self, path: Path) -> FileObject:
        if path.suffix.lower() != ".jsonl":
            raise ValueError("Batch 输入文件必须使用 .jsonl 扩展名")
        with path.open("rb") as upload:
            return await self._client.files.create(file=(path.name, upload, "application/jsonl"), purpose="batch")

    async def create_batch(self, file_id: str, endpoint: BatchEndpoint) -> Batch:
        return await self._client.batches.create(
            input_file_id=file_id, endpoint=endpoint, completion_window="24h",
        )

    async def retrieve_batch(self, batch_id: str) -> Batch:
        return await self._client.batches.retrieve(batch_id)

    async def cancel_batch(self, batch_id: str) -> Batch:
        return await self._client.batches.cancel(batch_id)

    async def batch_file_content(self, file_id: str) -> AsyncIterator[bytes]:
        async with self._client.files.with_streaming_response.content(file_id) as response:
            async for chunk in response.iter_bytes():
                yield chunk

    async def embed(
        self, input_data: str | list[str], *, model: str = "text-embedding-3-small", dimensions: int | None = None,
    ) -> CreateEmbeddingResponse:
        if dimensions is None:
            return await self._client.embeddings.create(model=model, input=input_data, encoding_format="float")
        return await self._client.embeddings.create(
            model=model, input=input_data, dimensions=dimensions, encoding_format="float",
        )

    async def moderate(
        self, input_data: str | list[str] | list[ModerationMultiModalInputParam], *,
        model: str = "omni-moderation-latest",
    ) -> ModerationCreateResponse:
        return await self._client.moderations.create(model=model, input=input_data)


@asynccontextmanager
async def openai_platform(profile_name: str = "openai") -> AsyncIterator[OpenAIPlatform]:
    async with openai_client(profile_name) as client:
        yield OpenAIPlatform(client)
