# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The device host: one process, one mesh, every member's device work.

Boot is phased across ALL members, because a buffer or program binary allocated after any
trace capture can sit in that trace's scratch and be overwritten when it replays:

1. load weights and allocate KV caches;
2. compile-warm with tracing off;
3. capture traces, one member at a time, in config order.

After phase 3 nothing new is compiled or allocated for the life of the group. A
program-cache count that grows after that is reported as ``compiles_after_capture`` on the
status endpoint.
"""

from __future__ import annotations

import http.server
import itertools
import json
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

from . import rpc
from .config import CohostConfig, Member
from .device import Device, NoGC

log = logging.getLogger("tt_cohost")

BARRIER_TIMEOUT_S = float(os.environ.get("TT_COHOST_BARRIER_TIMEOUT_S", "7200"))

#: Mesh shapes by MESH_DEVICE name, for groups that open the mesh without the vLLM plugin.
MESH_SHAPES = {"P150": (1, 1), "P100": (1, 1), "N150": (1, 1), "P300": (1, 2), "P150x2": (1, 2),
               "N300": (1, 2), "P150x4": (1, 4), "P300x2": (1, 4), "QB2": (1, 4), "P150x8": (1, 8)}


class Barrier:
    """Phase gate: ``wait`` returns once every expected member has arrived, or raises."""

    def __init__(self, name: str, expected: List[str]):
        self.name, self.expected = name, set(expected)
        self._arrived: set = set()
        self._failed: Optional[str] = None
        self._cv = threading.Condition()

    def arrive(self, who: str) -> None:
        with self._cv:
            self._arrived.add(who)
            self._cv.notify_all()

    def fail(self, why: str) -> None:
        with self._cv:
            self._failed = why
            self._cv.notify_all()

    def wait(self, who: str) -> None:
        self.arrive(who)
        deadline = time.monotonic() + BARRIER_TIMEOUT_S
        with self._cv:
            while self._arrived < self.expected and not self._failed:
                left = deadline - time.monotonic()
                if left <= 0:
                    missing = sorted(self.expected - self._arrived)
                    raise TimeoutError(f"barrier {self.name}: {missing} never arrived")
                self._cv.wait(left)
            if self._failed:
                raise RuntimeError(f"barrier {self.name} failed: {self._failed}")


class Host:
    def __init__(self, cfg: CohostConfig, device: Optional[Device] = None):
        self.cfg = cfg
        self.device = device or Device()
        self.mesh = None
        self.members: Dict[str, "VllmMember"] = {}
        names = [m.name for m in cfg.members]
        self.loaded = Barrier("loaded", names)        # weights + KV allocated
        self.compiled = Barrier("compiled", names)    # phase 2 done everywhere
        self._capture_turn = 0
        self._capture_cv = threading.Condition()
        self.capture_order = names
        self.cache_entries_at_capture: Optional[int] = None
        self.compiles_after_capture: Dict[str, int] = {}
        self.ready: set = set()
        self._open_mesh = self._close_mesh = None
        self.dram: Dict[str, Dict[str, float]] = {}
        self.l1: Dict[str, Optional[float]] = {}
        self.dram_total: Optional[float] = None

    # ------------------------------------------------------------------ mesh
    def open_mesh(self):
        """Open the group's one mesh on the device thread (first caller wins)."""
        def _open():
            if self.mesh is None:
                os.environ["MESH_DEVICE"] = self.cfg.mesh_device
                self.mesh = self._open_mesh(self.cfg.tt_config, "all")
                log.info("cohost mesh %s open: %d devices", self.cfg.mesh_device,
                         self.mesh.get_num_devices())
            return self.mesh
        return self.device.run("host", _open)

    def open_mesh_direct(self):
        """Open the group's mesh with ttnn alone (an app-only group has no vLLM plugin)."""
        def _open():
            if self.mesh is None:
                import ttnn
                rows, cols = MESH_SHAPES[self.cfg.mesh_device]
                tc = self.cfg.tt_config
                fabric = tc.get("fabric_config")
                if fabric and rows * cols > 1:
                    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric))
                self.mesh = ttnn.open_mesh_device(
                    ttnn.MeshShape(rows, cols), l1_small_size=int(tc.get("l1_small_size", 32768)),
                    trace_region_size=int(tc.get("trace_region_size", 0)),
                    num_command_queues=int(tc.get("num_command_queues", 1)))

                def close(mesh, _cfg):
                    ttnn.close_mesh_device(mesh)
                    if fabric and rows * cols > 1:
                        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                self._close_mesh = close
                log.info("cohost mesh %s open: %d devices", self.cfg.mesh_device, self.mesh.get_num_devices())
            return self.mesh
        return self.device.run("host", _open)

    def install_mesh_hooks(self) -> None:
        """Every TTWorker gets the shared mesh, and none of them may close it."""
        from vllm_tt_plugin import worker as ttw
        self._open_mesh, self._close_mesh = ttw.open_mesh_device, ttw.close_mesh_device
        ttw.open_mesh_device = lambda *a, **k: self.open_mesh()
        ttw.close_mesh_device = lambda *a, **k: None
        blocks = ttw.get_num_available_blocks_tt
        caps = {m.name: m.kv_tokens for m in self.cfg.members if m.kv_tokens}

        def capped_blocks(vllm_config, num_devices=1):
            """The plugin's KV block count, capped for the member whose init_device is running."""
            n = blocks(vllm_config, num_devices)
            cap = caps.get(self.device.owner)
            if cap:
                n = min(n, max(1, cap // vllm_config.cache_config.block_size))
            return n
        ttw.get_num_available_blocks_tt = capped_blocks

    def close(self) -> None:
        """Close the shared mesh exactly once (a mesh left open wedges the ethernet cores)."""
        def _close():
            if self.mesh is not None and self._close_mesh is not None:
                self._close_mesh(self.mesh, self.cfg.tt_config)
                self.mesh = None
        self.device.run("host", _close)

    # ------------------------------------------------------------- captures
    @contextmanager
    def capture_turn(self, who: str):
        """Capture in config order, one member at a time, with the cycle collector off."""
        idx = self.capture_order.index(who)
        with self._capture_cv:
            while self._capture_turn != idx:
                self._capture_cv.wait()
        try:
            with NoGC():
                yield
        finally:
            with self._capture_cv:
                self._capture_turn += 1
                self._capture_cv.notify_all()

    def after_all_captured(self) -> None:
        if self.mesh is not None and hasattr(self.mesh, "num_program_cache_entries"):
            self.cache_entries_at_capture = self.mesh.num_program_cache_entries()

    def step(self, who: str, fn: Callable[[], Any]) -> Any:
        """Run one request's device work as ``who``, then count what it compiled after capture."""
        def go():
            out = fn()
            self.check_cache(who)
            return out
        return self.device.run(who, go)

    def check_cache(self, who: str, step: str = "") -> None:
        """Count (and log) programs ``who`` compiled after capture: each can corrupt a parked trace.
        Device thread only."""
        if self.cache_entries_at_capture is None or self.mesh is None:
            return
        now = self.mesh.num_program_cache_entries()
        if now > self.cache_entries_at_capture:
            n = now - self.cache_entries_at_capture
            self.compiles_after_capture[who] = self.compiles_after_capture.get(who, 0) + n
            log.warning("cohost: %s compiled %d program(s) after trace capture (%d so far%s); a parked "
                        "trace may be corrupted", who, n, self.compiles_after_capture[who],
                        f"; step: {step}" if step else "")
            self.cache_entries_at_capture = now

    def dram_gb_per_chip(self) -> Optional[float]:
        """DRAM allocated on each chip right now (the mesh allocator is the same on every chip)."""
        if self.mesh is None:
            return None
        import ttnn
        mv = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM)
        return round(mv.total_bytes_allocated_per_bank * mv.num_banks / 2**30, 3)

    def l1_kib_per_core(self) -> Optional[float]:
        """Worker L1 held by live buffers, per core (KiB). Persistent L1 in one member shrinks
        the circular-buffer room every other member's programs were compiled to expect."""
        if self.mesh is None:
            return None
        import ttnn
        mv = ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1)
        return round(mv.total_bytes_allocated_per_bank / 1024, 1)

    def account(self, who: str, phase: str, before: Optional[float]) -> None:
        """Record what ``phase`` of member ``who`` added to per-chip DRAM (GiB), and the L1 held after it."""
        after = self.dram_gb_per_chip()
        if before is not None and after is not None:
            l1 = self.l1_kib_per_core()
            self.dram_total = after
            self.dram.setdefault(who, {})[phase] = round(after - before, 3)
            self.l1[who] = l1
            log.info("cohost: %s %s: +%.2f GiB/chip DRAM (total %.2f); L1 held %s KiB/core",
                     who, phase, after - before, after, l1)
            if os.environ.get("TT_COHOST_L1_REPORT"):
                self.log_l1_buffers(f"{who} {phase}")

    def log_l1_buffers(self, when: str) -> None:
        """Every live L1 buffer on the first chip: what holds L1 that other members' programs need."""
        import ttnn
        rows = sorted({(b.address, b.max_size_per_bank, str(b.buffer_layout))
                       for b in ttnn._ttnn.reports.get_buffers(self.mesh)
                       if str(b.buffer_type).endswith("L1") and b.device_id == 0})
        log.info("cohost: L1 buffers after %s (%d): %s", when, len(rows),
                 ", ".join(f"{a}+{n}({lay.split('.')[-1]})" for a, n, lay in rows) or "none")

    # --------------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "mesh_device": self.cfg.mesh_device,
            "members": {m.name: {"kind": m.kind, "port": m.port, "ready": m.name in self.ready}
                        for m in self.cfg.members},
            "device_owner": self.device.owner,
            "compiles_after_capture": self.compiles_after_capture,
            # cached from the device thread: ttnn is only ever called there
            "dram_gib_per_chip": {"members": self.dram, "total": self.dram_total},
            "l1_kib_per_core_after_phase": self.l1,
        }

    def serve_status(self) -> None:
        if not self.cfg.status_port:
            return
        host = self

        class _H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - http.server API
                body = json.dumps(host.status()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        srv = http.server.ThreadingHTTPServer(("0.0.0.0", self.cfg.status_port), _H)
        threading.Thread(target=srv.serve_forever, daemon=True, name="tt-cohost-status").start()


@contextmanager
def _member_env(member: Member):
    """Apply the member's env for a load-time call; restore afterwards."""
    saved = {k: os.environ.get(k) for k in member.env}
    os.environ.update(member.env)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class VllmMember:
    """The device-host side of one vLLM member: a real TTWorker driven over a Unix socket."""

    LOAD_TIME = {"init_worker", "init_device", "load_model", "initialize_from_config",
                 "determine_available_memory", "get_kv_cache_spec", "compile_or_warm_up_model"}
    #: Load-time calls whose DRAM growth is recorded per member (weights; KV cache).
    ACCOUNTED = {"load_model", "initialize_from_config"}

    def __init__(self, host: Host, member: Member):
        self.host, self.member, self.name = host, member, member.name
        self.wrapper = None
        self.vllm_config = None
        self._refs: Dict[int, Any] = {}
        self._ids = itertools.count(1)
        self._sample_pending = False
        self._step = ""                       # what the current step runs, for the late-compile log

    # ------------------------------------------------------------- serving
    def serve(self) -> None:
        srv = rpc.listen(self.host.cfg.socket_path(self.name))
        threading.Thread(target=self._accept, args=(srv,), daemon=True,
                         name=f"tt-cohost-{self.name}").start()

    def _accept(self, srv: socket.socket) -> None:
        conn = rpc.accept(srv)
        log.info("cohost: %s frontend connected", self.name)
        while True:
            try:
                req = rpc.recv(conn)
            except ConnectionError:
                log.warning("cohost: %s frontend disconnected", self.name)
                return
            try:
                reply = rpc.ok(self.handle(req))
            except BaseException as e:  # noqa: BLE001 - every failure goes back to the frontend
                log.exception("cohost: %s %s failed", self.name, req.get("method", req["op"]))
                if req.get("method") in self.LOAD_TIME:
                    self.host.loaded.fail(f"{self.name}: {e}")
                    self.host.compiled.fail(f"{self.name}: {e}")
                reply = rpc.error(e)
            rpc.send(conn, reply)

    # ------------------------------------------------------------ dispatch
    def handle(self, req: dict) -> Any:
        op = req["op"]
        if op == "ping":
            return "pong"
        if op == "init_worker":
            return self._init_worker(req["kwargs"])
        if op == "get_output":
            out = self._refs.pop(req["ref"])
            return self._on_device(out.get_output)
        if op != "run":
            raise ValueError(f"unknown op {op!r}")
        method, args, kwargs = req["method"], req.get("args", ()), req.get("kwargs") or {}
        if method == "compile_or_warm_up_model":
            return self._phased_warmup()
        if method == "execute_model":
            return self._execute(args, kwargs)
        if method == "sample_tokens":
            return self._sample(args, kwargs)
        if method == "initialize_from_config":
            result = self._call(method, args, kwargs)
            return result
        return self._call(method, args, kwargs)

    def _on_device(self, fn):
        return self.host.device.run(self.name, fn)

    def _call(self, method, args, kwargs):
        from vllm.v1.serial_utils import run_method

        def go():
            if method in self.LOAD_TIME:
                before = self.host.dram_gb_per_chip() if method in self.ACCOUNTED else None
                with _member_env(self.member):
                    result = run_method(self.wrapper, method, args, kwargs)
                self.host.account(self.name, method, before)
                return result
            return run_method(self.wrapper, method, args, kwargs)
        return self._wrap_async(self._on_device(go))

    def _wrap_async(self, result):
        from vllm.v1.outputs import AsyncModelRunnerOutput
        if isinstance(result, AsyncModelRunnerOutput):
            ref = next(self._ids)
            self._refs[ref] = result
            return {"__cohost_async_ref__": ref}
        return result

    def _init_worker(self, kwargs: dict) -> None:
        from threading import Lock

        from vllm.v1.worker.worker_base import WorkerWrapperBase
        self.vllm_config = kwargs["vllm_config"]
        self.wrapper = WorkerWrapperBase(rpc_rank=0)
        kwargs = {**kwargs, "shared_worker_lock": Lock()}
        with _member_env(self.member):
            if self.member.preload:
                from .apps import load_factory
                load_factory(self.member.preload)()
            self.wrapper.init_worker(all_kwargs=[kwargs])

    # ----------------------------------------------- step-long ownership
    def _execute(self, args, kwargs):
        """Own the device from execute_model until its sample_tokens (when one is pending)."""
        so = args[0] if args else None
        self._step = (f"{len(getattr(so, 'scheduled_new_reqs', []))} new + "
                      f"{len(getattr(getattr(so, 'scheduled_cached_reqs', None), 'req_ids', []))} running "
                      f"requests, {getattr(so, 'total_num_scheduled_tokens', '?')} tokens")
        self.host.device.acquire(self.name)
        try:
            result = self._call("execute_model", args, kwargs)
        except BaseException:
            self.host.device.release(self.name)
            raise
        if result is None:
            self._sample_pending = True        # released by the matching sample_tokens
        else:
            self.host.device.release(self.name)
        return result

    def _sample(self, args, kwargs):
        try:
            out = self._call("sample_tokens", args, kwargs)
            self._on_device(lambda: self.host.check_cache(self.name, self._step))
            return out
        finally:
            if self._sample_pending:
                self._sample_pending = False
                self.host.device.release(self.name)

    # --------------------------------------------------- phased warmup
    def _phased_warmup(self):
        from vllm.v1.worker.worker_base import CompilationTimes
        from vllm_tt_plugin.platform import TTPlatform

        worker = self.wrapper.worker
        runner = worker.model_runner
        start = time.perf_counter()
        self.host.loaded.wait(self.name)
        if not getattr(worker, "enable_model_warmup", True):
            self.host.compiled.wait(self.name)
            return CompilationTimes(language_model=0.0, encoder=0.0)

        def kwargs_for():
            TTPlatform.check_and_update_config(self.vllm_config)  # this member's sampling mode
            sod = getattr(TTPlatform, "sample_on_device_mode", None)
            prefill = dict(kv_cache=runner.kv_caches, can_sample_on_device=sod == "all")
            decode = dict(kv_cache=runner.kv_caches, max_batch_size=runner.tt_max_batch_size,
                          num_blocks=runner.max_num_blocks_per_req,
                          can_sample_on_device=sod in ("all", "decode_only"))
            return prefill, decode

        def compile_pass():
            with _member_env(self.member):
                prefill, decode = kwargs_for()
                runner.model.warmup_model_prefill(enable_trace=False, **prefill)
                runner.model.warmup_model_decode(enable_trace=False, **decode)
                if hasattr(runner.model, "already_warmed_up_prefill"):
                    runner.model.already_warmed_up_prefill = False

        def capture_pass():
            with _member_env(self.member):
                prefill, decode = kwargs_for()
                if runner.trace_mode == "all":
                    runner.model.warmup_model_prefill(enable_trace=True, **prefill)
                if runner.trace_mode in ("all", "decode_only"):
                    runner.model.warmup_model_decode(enable_trace=True, **decode)

        before = self._on_device(self.host.dram_gb_per_chip)
        self._on_device(compile_pass)
        self._on_device(lambda: self.host.account(self.name, "compile", before))
        log.info("cohost: %s compiled", self.name)
        self.host.compiled.wait(self.name)
        with self.host.capture_turn(self.name):
            before = self._on_device(self.host.dram_gb_per_chip)
            self._on_device(capture_pass)
            self._on_device(lambda: self.host.account(self.name, "capture", before))
            log.info("cohost: %s traces captured", self.name)
        if self.name == self.host.capture_order[-1]:
            self._on_device(self.host.after_all_captured)
        return CompilationTimes(language_model=time.perf_counter() - start, encoder=0.0)


def build(cfg: CohostConfig, device: Optional[Device] = None) -> Host:
    host = Host(cfg, device)
    if any(m.kind == "vllm" for m in cfg.members):
        host.install_mesh_hooks()
    for m in cfg.members:
        if m.kind == "vllm":
            vm = VllmMember(host, m)
            host.members[m.name] = vm
            vm.serve()
    host.serve_status()
    return host
