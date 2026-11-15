"""DeepSeek provider: chat adapter, vendor semantics and hosted services.

``media`` and ``web`` are imported explicitly by their callers; they read
application configuration and are not needed to build a chat model.
"""

from .chat import DeepSeekAdapter, DeepSeekVendor

__all__ = ["DeepSeekAdapter", "DeepSeekVendor"]
