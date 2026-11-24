"""QQ SILK and mono PCM WAV conversions shared by speech providers."""

import asyncio
import io
import os
import subprocess
import sys
import wave
from contextlib import suppress
from pathlib import Path

SILK_HEADERS = (b"\x02#!SILK_V3", b"#!SILK_V3")
DECODE_WORKER = Path(__file__).with_name("speech_decode.py")


class AudioCodecError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def pcm_to_wav(pcm: bytes | bytearray, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)
    return output.getvalue()


async def decode_silk(data: bytes, target: Path, max_seconds: int) -> None:
    source = target.with_suffix(".silk")
    source.write_bytes(data)
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, os.fspath(DECODE_WORKER),
            os.fspath(source), os.fspath(target), str(max_seconds),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            _, stderr = await process.communicate()
        except asyncio.CancelledError:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
            raise
    except OSError as exc:
        raise AudioCodecError("audio_decode_failed") from exc
    if process.returncode:
        raise AudioCodecError("audio_duration_limit" if b"audio_duration_limit" in stderr else "audio_decode_failed")


def wav_duration(wav: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav), "rb") as stream:
            return stream.getnframes() / stream.getframerate()
    except (EOFError, OSError, ValueError, wave.Error, ZeroDivisionError) as exc:
        raise AudioCodecError("invalid_audio") from exc


def wav_to_silk(wav: bytes) -> bytes:
    try:
        import pysilk

        input_stream = io.BytesIO()
        with wave.open(io.BytesIO(wav), "rb") as source:
            pcm = source.readframes(source.getnframes())
            rate = source.getframerate()
            channels = source.getnchannels()
        if channels != 1:
            raise AudioCodecError("invalid_audio")
        input_stream.write(pcm)
        input_stream.seek(0)
        output = io.BytesIO()
        pysilk.encode(input_stream, output, rate, rate, tencent=True)
        return output.getvalue()
    except AudioCodecError:
        raise
    except Exception as exc:
        raise AudioCodecError("silk_encode_failed") from exc
