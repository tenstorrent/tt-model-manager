# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""One device thread, owned by one member at a time.

Every ttnn call of every member runs on the same thread, so the mesh's command queue sees
one ordered stream. A member owns the device for a whole step (vLLM's ``execute_model``
through its ``sample_tokens``). Another member's trace replay can therefore never land
between a step's forward and the enqueue of its readback. Owners are served first come,
first served, so no member starves.

In the device host that thread is the main thread (``Device(own_thread=False)`` + ``loop()``):
tt-metal tears the chips down on the main thread at exit, and UMD's per-chip lock can only be
released by the thread that took it.
"""

from __future__ import annotations

import collections
import gc
import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional


class Device:
    def __init__(self, *, own_thread: bool = True) -> None:
        self._cv = threading.Condition()
        self._owner: Optional[str] = None
        self._owner_thread: Optional[int] = None
        self._depth = 0
        self._waiting: collections.deque = collections.deque()
        self._jobs: "queue.Queue[tuple[Optional[Callable[[], Any]], Optional[Future]]]" = queue.Queue()
        if own_thread:
            self._thread = threading.Thread(target=self.loop, name="tt-cohost-device", daemon=True)
            self._thread.start()
        else:
            self._thread = threading.current_thread()   # which then calls loop()

    def loop(self) -> None:
        """Run device jobs on this thread until ``stop()``."""
        while True:
            fn, fut = self._jobs.get()
            if fn is None or fut is None:
                return
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as e:  # noqa: BLE001 - handed back to the submitting member
                fut.set_exception(e)

    def stop(self) -> None:
        self._jobs.put((None, None))

    @property
    def owner(self) -> Optional[str]:
        return self._owner

    def acquire(self, who: str) -> None:
        """Take ownership for ``who``: re-entrant on the thread that holds it, FIFO for everyone else
        (a member's second concurrent request too, or it could keep the device from the others)."""
        me = (who, threading.get_ident())
        with self._cv:
            if (self._owner, self._owner_thread) == me:
                self._depth += 1
                return
            self._waiting.append(me)
            while self._owner is not None or self._waiting[0] != me:
                self._cv.wait()
            self._waiting.popleft()
            (self._owner, self._owner_thread), self._depth = me, 1

    def release(self, who: str) -> None:
        with self._cv:
            if self._owner != who:
                raise RuntimeError(f"{who} released the device but {self._owner} owns it")
            self._depth -= 1
            if self._depth == 0:
                self._owner = self._owner_thread = None
                self._cv.notify_all()

    def run(self, who: str, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on the device thread while ``who`` owns the device.

        Called from the device thread itself (a job that needs more device work), it runs
        inline under the current owner rather than queueing behind itself.
        """
        if threading.current_thread() is self._thread:
            return fn()
        self.acquire(who)
        try:
            fut: Future = Future()
            self._jobs.put((fn, fut))
            return fut.result()
        finally:
            self.release(who)


class NoGC:
    """Keep the cycle collector from freeing device tensors in the middle of a trace capture."""

    def __enter__(self):
        self._was = gc.isenabled()
        gc.disable()
        return self

    def __exit__(self, *exc):
        if self._was:
            gc.enable()
        return False
