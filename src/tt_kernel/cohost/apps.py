# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""App members: models with their own HTTP API (speech, music, ...), served in the device host.

An app member is ``factory(host, member) -> AppMember``, named by ``Member.app``
("module:factory"). The host drives every app through the same boot:

1. ``load(mesh)`` for every member (weights, caches, model objects);
2. ``warmup()`` once per member, in config order (this compiles every program and creates
   every lazily allocated buffer, and may capture traces);
3. ``release_traces()`` for every member;
4. ``warmup()`` again, in order (this recaptures).

The second round compiles and allocates nothing new, so no long-lived buffer of any member
can sit in another member's trace scratch. The app's HTTP server then runs in this process
on the member's port. Its request handlers must run their device work through
``host.step(member.name, fn)``, which makes each piece of device work exclusive and counts
any program it compiled after capture.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import threading
from typing import Any, Callable, List

from .config import Member

log = logging.getLogger("tt_cohost")


class AppMember:
    """What an app adapter implements. Every method runs on the device thread."""

    def load(self, mesh) -> None:
        raise NotImplementedError

    def warmup(self) -> None:
        raise NotImplementedError

    def release_traces(self) -> None:
        raise NotImplementedError

    def asgi_app(self) -> Any:
        """The ASGI app (FastAPI etc.) that serves this member's API."""
        raise NotImplementedError


def load_factory(spec: str) -> Callable:
    module, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module), attr)


def serve_asgi(app: Any, port: int, name: str) -> threading.Thread:
    """Run ``app`` under uvicorn on ``port`` in a daemon thread with its own event loop."""
    import uvicorn

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", lifespan="on")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # signals belong to the launcher

    def run():
        asyncio.run(server.serve())
    t = threading.Thread(target=run, name=f"tt-cohost-app-{name}", daemon=True)
    t.start()
    return t


def boot_apps(host, members: List[Member]) -> List[AppMember]:
    """Load, warm, release and re-warm every app member (see the module docstring), then serve."""
    apps = []
    for m in members:
        app = load_factory(m.app)(host, m)
        apps.append((m, app))
    from .host import _member_env

    mesh = host.open_mesh_direct()
    for m, app in apps:
        before = host.device.run(m.name, host.dram_gb_per_chip)
        with _member_env(m):
            host.device.run(m.name, lambda: app.load(mesh))
        host.device.run(m.name, lambda: host.account(m.name, "load", before))
        log.info("cohost: %s load_model: loaded", m.name)
    for rnd in ("compile", "capture"):
        for m, app in apps:
            before = host.device.run(m.name, host.dram_gb_per_chip)
            with _member_env(m):
                host.device.run(m.name, app.warmup)
            host.device.run(m.name, lambda: host.account(m.name, f"warmup-{rnd}", before))
            log.info("cohost: %s %s", m.name, "compiled" if rnd == "compile" else "traces captured")
        if rnd == "compile":
            for m, app in apps:
                host.device.run(m.name, app.release_traces)
    host.device.run("host", host.after_all_captured)
    for m, app in apps:
        serve_asgi(app.asgi_app(), m.port, m.name)
    return [app for _, app in apps]
