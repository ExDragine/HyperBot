# ruff: noqa: S101

from types import SimpleNamespace

import pytest

from plugins.agent import document_parsing
from utils.configs import DocumentParsingConfig


@pytest.mark.asyncio
async def test_parse_staged_files_injects_bounded_page_text(monkeypatch, tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"pdf")
    staged = SimpleNamespace(local_path=path, file_name="report.pdf", file_size=3)

    async def fake_ocr(data, file_name, *, provider, purpose):
        assert data == b"pdf"
        assert file_name == "report.pdf"
        assert provider == "deepseek"
        assert purpose == "text_recognition"
        return {"page_count": 2, "pages": [{"index": 0, "text": "第一页"}, {"index": 1, "text": "第二页"}]}

    monkeypatch.setattr(document_parsing, "ocr_file", fake_ocr)
    text, parsed = await document_parsing.parse_staged_files(
        [staged], DocumentParsingConfig(enabled=True, max_text_chars=5)
    )

    assert "report.pdf" in text
    assert "第一页" in text
    assert "第二页" not in text
    assert parsed[0].page_count == 2


@pytest.mark.asyncio
async def test_parse_staged_files_skips_unsupported_files(monkeypatch, tmp_path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"zip")
    staged = SimpleNamespace(local_path=path, file_name="archive.zip", file_size=3)

    async def fail(*args, **kwargs):
        raise AssertionError("unsupported file must not reach OCR")

    monkeypatch.setattr(document_parsing, "ocr_file", fail)
    text, parsed = await document_parsing.parse_staged_files(
        [staged], DocumentParsingConfig(enabled=True)
    )

    assert text == ""
    assert parsed == ()
