"""Platform adapters for the Agent boundary.

Importing this package only loads pure adapter facades.  It does not register
NoneBot matchers or import the QQ event handlers.
"""

from .qq import (
    QqDelivery,
    QqHistoryStore,
    QqMessageAdapter,
    QqReplyPolicy,
    QqToolProvider,
    qq_artifact_messages,
    send_qq_artifacts,
)

__all__ = [
    "QqDelivery",
    "QqHistoryStore",
    "QqMessageAdapter",
    "QqReplyPolicy",
    "QqToolProvider",
    "qq_artifact_messages",
    "send_qq_artifacts",
]
