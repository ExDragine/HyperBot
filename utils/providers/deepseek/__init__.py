"""DeepSeek provider: chat adapter, vendor semantics and hosted services.

``media`` is imported explicitly by callers that need the hosted media APIs.
"""

from .chat import DeepSeekAdapter, DeepSeekVendor

__all__ = ["DeepSeekAdapter", "DeepSeekVendor"]
