"""Python <-> Unity integration (Developer 4).

* :mod:`python_backend.integration.protocol` — the JSON wire contract.
* :mod:`python_backend.integration.session`  — the live simulation session.
* :mod:`python_backend.integration.server`   — localhost TCP server (Python side).
* :mod:`python_backend.integration.client`   — reference client + headless viewer.

Protocol documentation: ``docs/communication_protocol.md``.
"""

from python_backend.integration.client import GridMindClient
from python_backend.integration.protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode,
    encode,
)
from python_backend.integration.server import GridMindServer
from python_backend.integration.session import LiveSession, SessionConfig

__all__ = [
    "PROTOCOL_VERSION",
    "ProtocolError",
    "decode",
    "encode",
    "GridMindClient",
    "GridMindServer",
    "LiveSession",
    "SessionConfig",
]
