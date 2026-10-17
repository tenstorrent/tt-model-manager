# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The co-host runtime's device-free core: config, device ownership, barriers, RPC, lock window."""

import json
import socket
import sys
import threading
import time
import types

import pytest

from tt_kernel.cohost import config as cfgmod
from tt_kernel.cohost import host as hostmod
from tt_kernel.cohost import launch, rpc
from tt_kernel.cohost.device import Device


def _write(tmp_path, **over):
    raw = {"cohost_config_version": 1, "mesh_device": "P150x4",
           "members": [{"name": "a", "kind": "vllm", "port": 21000, "model": "m/a"},
                       {"name": "b", "kind": "vllm", "port": 21001, "model": "m/b"}]}
    raw.update(over)
    p = tmp_path / "cohost.json"
    p.write_text(json.dumps(raw))
    return str(p)


# ------------------------------------------------------------------------------- config

def test_config_loads_members_in_order(tmp_path):
    cfg = cfgmod.load(_write(tmp_path))
    assert [m.name for m in cfg.members] == ["a", "b"]
    assert cfg.socket_path("a").endswith("/a.sock")


@pytest.mark.parametrize("over, match", [
    ({"cohost_config_version": 2}, "version"),
    ({"members": [{"name": "a", "kind": "vllm", "port": 1}, {"name": "a", "kind": "vllm", "port": 2}]}, "duplicate"),
    ({"members": [{"name": "a", "kind": "vllm", "port": 1}, {"name": "b", "kind": "vllm", "port": 1}]}, "distinct ports"),
    ({"members": [{"name": "a", "kind": "llm", "port": 1}]}, "kind"),
    ({"members": [{"name": "a", "kind": "app", "port": 1}]}, "module:factory"),
])
def test_config_refuses_what_cannot_run(tmp_path, over, match):
    with pytest.raises(ValueError, match=match):
        cfgmod.load(_write(tmp_path, **over))


# ------------------------------------------------------------------------------- device

def test_device_runs_every_job_on_one_thread():
    dev = Device()
    seen = {dev.run("a", lambda: threading.current_thread().name),
            dev.run("b", lambda: threading.current_thread().name)}
    assert seen == {"tt-cohost-device"}


def test_a_device_without_its_own_thread_runs_jobs_on_the_thread_that_loops_it():
    dev = Device(own_thread=False)
    out = []
    threading.Thread(target=lambda: (out.append(dev.run("a", threading.current_thread)), dev.stop())).start()
    dev.loop()
    assert out == [threading.current_thread()]


def test_launch_runs_the_device_on_the_main_thread_and_closes_the_mesh_there(tmp_path, monkeypatch):
    """tt-metal tears the chips down on the main thread at exit; UMD's per-chip lock must be
    released by the thread that took it, so the mesh is opened and closed on the main thread."""
    from tt_kernel.cohost import apps

    closed_on = []

    class FakeHost:
        def __init__(self, cfg, device):
            self.cfg, self.device, self.ready = cfg, device, set()

        def close(self):
            self.device.run("host", lambda: closed_on.append(threading.current_thread()))

    monkeypatch.setattr(hostmod, "build", lambda cfg, device: FakeHost(cfg, device))
    monkeypatch.setattr(apps, "boot_apps", lambda host, members: 1 / 0)   # the group goes down
    members = [{"name": "tts", "kind": "app", "port": 1, "app": "x:y"}]
    assert launch.main(["--config", str(_write(tmp_path, members=members))]) == 1
    assert closed_on == [threading.main_thread()]


def test_device_ownership_is_reentrant_and_blocks_other_owners():
    dev = Device()
    dev.acquire("a")
    dev.acquire("a")
    got = threading.Event()
    t = threading.Thread(target=lambda: (dev.acquire("b"), got.set()))
    t.start()
    dev.release("a")
    assert not got.wait(0.2), "b took the device while a still held it"
    dev.release("a")
    assert got.wait(2)
    assert dev.owner == "b"
    dev.release("b")
    t.join()


def test_device_hands_ownership_out_first_come_first_served():
    dev = Device()
    dev.acquire("x")
    order = []
    threads = []
    for who in ("a", "b", "c"):
        th = threading.Thread(target=lambda w=who: (dev.acquire(w), order.append(w), dev.release(w)))
        th.start()
        threads.append(th)
        time.sleep(0.05)
    dev.release("x")
    for th in threads:
        th.join(2)
    assert order == ["a", "b", "c"]


def test_device_work_nested_inside_a_job_runs_inline_instead_of_deadlocking():
    """A member's init_device job opens the shared mesh, which is itself device work."""
    dev = Device()
    done = threading.Event()
    t = threading.Thread(target=lambda: (dev.run("member", lambda: dev.run("host", lambda: 42)), done.set()))
    t.start()
    assert done.wait(2), "nested device work deadlocked"


# ------------------------------------------------------------------------------ barrier

def test_barrier_releases_only_when_everyone_arrived():
    b = hostmod.Barrier("loaded", ["a", "b"])
    done = []
    t = threading.Thread(target=lambda: (b.wait("a"), done.append("a")))
    t.start()
    time.sleep(0.1)
    assert not done
    b.wait("b")
    t.join(2)
    assert done == ["a"]


def test_a_failed_member_fails_the_barrier_for_everyone():
    b = hostmod.Barrier("loaded", ["a", "b"])
    b.fail("b: out of DRAM")
    with pytest.raises(RuntimeError, match="out of DRAM"):
        b.wait("a")


# ---------------------------------------------------------------------------------- rpc

def test_rpc_round_trip_and_remote_errors_carry_the_host_traceback(tmp_path):
    path = str(tmp_path / "s.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        for _ in range(2):
            req = rpc.recv(conn)
            try:
                if req["op"] == "boom":
                    raise KeyError("no such ref")
                rpc.send(conn, rpc.ok({"echo": req["x"]}))
            except KeyError as e:
                rpc.send(conn, rpc.error(e))
    threading.Thread(target=serve, daemon=True).start()
    c = rpc.Client(path)
    assert c.call("echo", x=b"\x00" * 100_000) == {"echo": b"\x00" * 100_000}
    with pytest.raises(rpc.RemoteError, match="no such ref") as err:
        c.call("boom")
    assert "device host traceback" in str(err.value)


# ------------------------------------------------------------- step-long device ownership

@pytest.fixture
def fake_vllm(monkeypatch):
    """Just the vLLM names the host imports, so the lock window is testable without vLLM."""
    class AsyncModelRunnerOutput:
        def get_output(self):
            return "resolved"

    calls = []

    def run_method(obj, method, args, kwargs):
        calls.append(method)
        return obj.results[method]

    mods = {
        "vllm": types.ModuleType("vllm"), "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.outputs": types.SimpleNamespace(AsyncModelRunnerOutput=AsyncModelRunnerOutput),
        "vllm.v1.serial_utils": types.SimpleNamespace(run_method=run_method),
    }
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return types.SimpleNamespace(calls=calls, Async=AsyncModelRunnerOutput)


def _member(tmp_path, results):
    cfg = cfgmod.load(_write(tmp_path))
    host = hostmod.Host(cfg)
    vm = hostmod.VllmMember(host, cfg.members[0])
    vm.wrapper = types.SimpleNamespace(results=results)
    return host, vm


def test_a_pending_sample_keeps_the_device_until_sample_tokens(tmp_path, fake_vllm):
    host, vm = _member(tmp_path, {"execute_model": None, "sample_tokens": "out"})
    assert vm.handle({"op": "run", "method": "execute_model", "args": ("so",)}) is None
    assert host.device.owner == "a", "another member's replay could land before the readback"
    assert vm.handle({"op": "run", "method": "sample_tokens", "args": (None,)}) == "out"
    assert host.device.owner is None


def test_a_step_with_nothing_to_sample_releases_at_once(tmp_path, fake_vllm):
    host, vm = _member(tmp_path, {"execute_model": "EMPTY"})
    assert vm.handle({"op": "run", "method": "execute_model"}) == "EMPTY"
    assert host.device.owner is None


def test_deferred_outputs_come_back_as_a_ref_and_resolve_on_request(tmp_path, fake_vllm):
    host, vm = _member(tmp_path, {"sample_tokens": fake_vllm.Async()})
    ref = vm.handle({"op": "run", "method": "sample_tokens"})
    assert set(ref) == {"__cohost_async_ref__"}
    assert vm.handle({"op": "get_output", "ref": ref["__cohost_async_ref__"]}) == "resolved"


# ------------------------------------------------------------------------------ launch

def test_each_vllm_member_is_a_stock_server_wired_to_the_host(tmp_path):
    cfg = cfgmod.load(_write(tmp_path))
    argv = launch.frontend_argv(cfg.members[1])
    assert argv[1:4] == ["-m", "vllm.entrypoints.openai.api_server", "--model"]
    assert argv[argv.index("--port") + 1] == "21001"
    assert argv[argv.index("--distributed-executor-backend") + 1] == \
        "tt_kernel.cohost.vllm_remote.RemoteExecutor"
    env = launch.frontend_env(cfg, cfg.members[1])
    assert env["TT_COHOST_SOCKET"] == cfg.socket_path("b") and env["TT_COHOST_MEMBER"] == "b"


def test_the_shared_mesh_is_opened_once_with_the_real_opener(tmp_path, monkeypatch):
    """The plugin's opener is patched to hand out the shared mesh; the host must still call
    the original underneath, or every TTWorker recurses into itself."""
    opened = []
    mesh = types.SimpleNamespace(get_num_devices=lambda: 4)
    ttw = types.SimpleNamespace(open_mesh_device=lambda cfg, mode: opened.append(mode) or mesh,
                                close_mesh_device=lambda *a: None,
                                get_num_available_blocks_tt=lambda cfg, n=1: 1000)
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin", types.SimpleNamespace(worker=ttw))
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin.worker", ttw)
    host = hostmod.Host(cfgmod.load(_write(tmp_path)))
    host.install_mesh_hooks()
    assert ttw.open_mesh_device({}, "all", 0) is mesh
    assert ttw.open_mesh_device({}, "all", 0) is mesh
    assert opened == ["all"]


def test_a_member_kv_cap_limits_only_that_members_pool(tmp_path, monkeypatch):
    ttw = types.SimpleNamespace(open_mesh_device=None, close_mesh_device=None,
                                get_num_available_blocks_tt=lambda cfg, n=1: 1000)
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin", types.SimpleNamespace(worker=ttw))
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin.worker", ttw)
    members = [{"name": "a", "kind": "vllm", "port": 1, "kv_tokens": 6400},
               {"name": "b", "kind": "vllm", "port": 2}]
    host = hostmod.Host(cfgmod.load(_write(tmp_path, members=members)))
    host.install_mesh_hooks()
    vcfg = types.SimpleNamespace(cache_config=types.SimpleNamespace(block_size=64))
    assert host.device.run("a", lambda: ttw.get_num_available_blocks_tt(vcfg, 4)) == 100
    assert host.device.run("b", lambda: ttw.get_num_available_blocks_tt(vcfg, 4)) == 1000


# ---------------------------------------------------------------------------- app members

def test_app_members_boot_load_all_warm_all_release_all_warm_all_then_serve(tmp_path, monkeypatch):
    """The second warm round compiles and allocates nothing new, so no member's long-lived buffer
    can sit in another member's trace scratch."""
    from tt_kernel.cohost import apps

    events = []

    class Fake(apps.AppMember):
        def __init__(self, host, member):
            self.n = member.name

        def load(self, mesh):
            events.append(("load", self.n))

        def warmup(self):
            events.append(("warmup", self.n))

        def release_traces(self):
            events.append(("release", self.n))

        def asgi_app(self):
            return self.n

    members = [{"name": n, "kind": "app", "port": p, "app": "x:y"} for n, p in (("stt", 1), ("tts", 2))]
    host = hostmod.Host(cfgmod.load(_write(tmp_path, members=members)))
    monkeypatch.setattr(host, "open_mesh_direct", lambda: "MESH")
    monkeypatch.setattr(apps, "load_factory", lambda spec: Fake)
    served = []
    monkeypatch.setattr(apps, "serve_asgi", lambda app, port, name: served.append((app, port)))
    apps.boot_apps(host, host.cfg.members)
    assert events == [("load", "stt"), ("load", "tts"), ("warmup", "stt"), ("warmup", "tts"),
                      ("release", "stt"), ("release", "tts"), ("warmup", "stt"), ("warmup", "tts")]
    assert served == [("stt", 1), ("tts", 2)]


def test_a_group_mixing_vllm_and_app_members_is_refused(tmp_path):
    members = [{"name": "a", "kind": "vllm", "port": 1, "model": "m"}, {"name": "b", "kind": "app", "port": 2, "app": "x:y"}]
    with pytest.raises(ValueError, match="mixed"):
        cfgmod.load(_write(tmp_path, members=members))


def test_a_late_compile_is_counted_for_the_member_that_compiled_it_on_the_device_thread(tmp_path, fake_vllm):
    host, vm = _member(tmp_path, {"execute_model": None, "sample_tokens": "out"})
    seen = []

    class Mesh:
        entries = 10

        def num_program_cache_entries(self):
            seen.append((threading.current_thread() is host.device._thread, host.device.owner))
            return self.entries

    host.mesh, host.cache_entries_at_capture = Mesh(), 10
    host.mesh.entries = 13                 # this step compiled three programs
    vm.handle({"op": "run", "method": "execute_model"})
    vm.handle({"op": "run", "method": "sample_tokens"})
    assert host.compiles_after_capture == {"a": 3}
    assert seen == [(True, "a")], "ttnn is only called on the device thread, inside the step"


def test_an_app_request_step_counts_its_late_compiles_on_the_device_thread(tmp_path):
    members = [{"name": "tts", "kind": "app", "port": 1, "app": "x:y"}]
    host = hostmod.Host(cfgmod.load(_write(tmp_path, members=members)))
    seen = []

    class Mesh:
        entries = 5

        def num_program_cache_entries(self):
            seen.append(threading.current_thread() is host.device._thread)
            return self.entries

    host.mesh = Mesh()
    host.device.run("host", host.after_all_captured)

    def request():
        host.mesh.entries = 7          # this request compiled two programs
        return "audio"
    assert host.step("tts", request) == "audio"
    assert host.compiles_after_capture == {"tts": 2}
    assert seen == [True, True]
