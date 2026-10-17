# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Length-prefixed pickle frames over a Unix socket, between a vLLM frontend and the host."""

from __future__ import annotations

import pickle
import socket
import struct
import threading
import traceback
from typing import Any

_HEADER = struct.Struct("!Q")


class RemoteError(RuntimeError):
    """An exception raised in the device host, re-raised in the frontend with its traceback."""


def send(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=5)
    sock.sendall(_HEADER.pack(len(data)) + data)


def _recvn(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("device host closed the connection")
        buf += chunk
    return bytes(buf)


def recv(sock: socket.socket) -> Any:
    (n,) = _HEADER.unpack(_recvn(sock, _HEADER.size))
    return pickle.loads(_recvn(sock, n))


def ok(value: Any) -> dict:
    return {"ok": True, "value": value}


def error(exc: BaseException) -> dict:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


class Client:
    """One request at a time over one connection; the vLLM engine thread is its only caller."""

    def __init__(self, path: str, *, timeout_s: float = 600.0):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout_s)
        self._sock.connect(path)
        self._sock.settimeout(None)
        self._lock = threading.Lock()

    def call(self, op: str, **kw: Any) -> Any:
        with self._lock:
            send(self._sock, {"op": op, **kw})
            reply = recv(self._sock)
        if not reply["ok"]:
            raise RemoteError(f"{reply['error']}\n--- device host traceback ---\n{reply['traceback']}")
        return reply["value"]

    def close(self) -> None:
        self._sock.close()
