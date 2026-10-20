# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Co-host container entrypoint: ``python -m tt_kernel.cohost.launch --config <json>``.

It starts the device host in this process, then one stock vLLM OpenAI server per vLLM
member, each on its own port and wired to the host by ``RemoteExecutor``. It prints
``cohost: all members ready`` once every member answers ``/v1/models``. If any member dies,
the whole group goes down: the members share the device, so a half-alive group is not served.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, List

from .config import CohostConfig, Member, load, loads

READY_LINE = "cohost: all members ready"
log = logging.getLogger("tt_cohost")


def frontend_argv(m: Member) -> List[str]:
    """``vllm serve`` for one member, with its worker in the device host."""
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", m.model, *m.args,
            "--host", "0.0.0.0", "--port", str(m.port),
            "--distributed-executor-backend", "tt_kernel.cohost.vllm_remote.RemoteExecutor"]


def frontend_env(cfg: CohostConfig, m: Member) -> Dict[str, str]:
    pkg_root = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, **m.env, "TT_COHOST_SOCKET": cfg.socket_path(m.name),
           "TT_COHOST_MEMBER": m.name}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (pkg_root, env.get("PYTHONPATH", "")) if p)
    return env


def _pump(name: str, stream) -> None:
    for line in iter(stream.readline, ""):
        sys.stdout.write(f"[{name}] {line}")
        sys.stdout.flush()


def _answers(port: int, kind: str = "vllm") -> bool:
    """A vLLM member is ready when it lists its model; an app when its port takes connections."""
    if kind != "vllm":
        try:
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
            return True
        except OSError:
            return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tt-cohost")
    ap.add_argument("--config", help="config JSON file (default: the TT_COHOST_CONFIG env, inline JSON)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    cfg = load(args.config) if args.config else loads(os.environ["TT_COHOST_CONFIG"])

    from . import host as hostmod  # imports vllm/ttnn lazily, after the env above is final
    from .device import Device
    host = hostmod.build(cfg, Device(own_thread=False))   # this (main) thread runs the device
    stopping = threading.Event()

    def _stop(*_):
        stopping.set()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    result = {"rc": 1}

    def _supervisor():
        try:
            result["rc"] = _supervise(cfg, host, stopping)
        finally:
            host.device.stop()
    sup = threading.Thread(target=_supervisor, name="tt-cohost-supervisor", daemon=True)
    sup.start()
    host.device.loop()
    sup.join()
    return result["rc"]


def _supervise(cfg, host, stopping: threading.Event) -> int:
    """Start the frontends and app members, announce readiness, and stop the group on a failure."""
    procs: Dict[str, subprocess.Popen] = {}
    for m in cfg.members:
        if m.kind != "vllm":
            continue
        p = subprocess.Popen(frontend_argv(m), env=frontend_env(cfg, m), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        procs[m.name] = p
        threading.Thread(target=_pump, args=(m.name, p.stdout), daemon=True).start()
        log.info("cohost: started %s (pid %d) on port %d", m.name, p.pid, m.port)

    apps = [m for m in cfg.members if m.kind == "app"]
    servers: Dict[str, threading.Thread] = {}   # app member -> its HTTP server thread, once booted
    failed = threading.Event()
    if apps:
        from .apps import boot_apps

        def _boot():
            try:
                servers.update(boot_apps(host, apps))
            except BaseException:  # noqa: BLE001 - the group goes down with the failure logged
                log.exception("cohost: app boot failed")
                failed.set()
                stopping.set()
        threading.Thread(target=_boot, name="tt-cohost-apps", daemon=True).start()

    rc = 0
    announced = False
    while not stopping.is_set():
        dead = [n for n, p in procs.items() if p.poll() is not None]
        dead += [n for n, t in list(servers.items()) if not t.is_alive()]
        if dead:
            log.error("cohost: member(s) %s exited; stopping the group", dead)
            rc = 1
            break
        for m in cfg.members:
            if m.name not in host.ready and _answers(m.port, m.kind):
                host.ready.add(m.name)
                log.info("cohost: %s ready on port %d", m.name, m.port)
        if not announced and len(host.ready) == len(cfg.members):
            print(READY_LINE, flush=True)
            print("Application startup complete.", flush=True)
            announced = True
        time.sleep(2)

    if failed.is_set():
        rc = 1
    for p in procs.values():
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + 60
    for p in procs.values():
        try:
            p.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
    host.close()
    return rc




if __name__ == "__main__":
    sys.exit(main())
