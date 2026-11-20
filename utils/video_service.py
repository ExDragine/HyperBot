"""Shared video generation entry point."""

from utils.providers.openai.videos import MediaReference, VideoGenerationResult, generate_video

__all__ = ["MediaReference", "VideoGenerationResult", "generate_video"]
