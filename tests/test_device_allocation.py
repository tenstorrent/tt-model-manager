# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The free-chip scan/picker: what makes ``serve`` share a board safely.

This is the safety-critical half of the device-scoping fix -- whether a foreign
container (tt-studio, tt-inference-server, a bare ``docker run``) reads as claiming the
right chips is what stands between "boots immediately" and "hangs forever on the UMD
lock" for a concurrent serve. Golden-string tests on the scan's inputs and outputs, not
on any real docker daemon.
"""

import json
from pathlib import Path

import pytest

from tt_kernel import container
from tt_kernel.container_manifest import ContainerManifest
from tt_kernel.launchers import launcher_for

from test_container_manifest import BASE

# The autouse fixture in conftest.py stubs `pick_free_devices` for every test in the suite
# (so serve_container tests never touch real docker/hardware) -- captured here, before any
# test runs, so the tests below that exercise the real picker can restore it explicitly.
_real_pick_free_devices = container.pick_free_devices


class _R:
    def __init__(self, stdout="", rc=0):
        self.stdout, self.returncode = stdout, rc


def _fake_docker(monkeypatch, *, ps_ids=(), inspected=(), ps_rc=0, inspect_rc=0,
                 inspect_stdout=None):
    """Stub ``container._run`` for exactly the two calls ``_claimed_devices`` makes:
    ``docker ps -q`` and ``docker inspect <ids...>`` (never ``--format``, which is every
    OTHER docker call in this module -- distinguished so this fake can't accidentally
    swallow an unrelated call)."""
    def fake(argv, **kw):
        if argv[:3] == ["docker", "ps", "-q"]:
            return _R("\n".join(ps_ids), ps_rc)
        if argv[:2] == ["docker", "inspect"] and "--format" not in argv:
            stdout = inspect_stdout if inspect_stdout is not None else json.dumps(list(inspected))
            return _R(stdout, inspect_rc)
        raise AssertionError(f"unexpected docker call: {argv}")
    monkeypatch.setattr(container, "_run", fake)


def _dev_root(tmp_path, ids=(0, 1, 2, 3)):
    root = tmp_path / "tenstorrent"
    root.mkdir()
    for i in ids:
        (root / str(i)).touch()
    return root


def _container(*, labels=None, devices=None, mounts=None, ipc="host", privileged=False):
    return {
        "Config": {"Labels": labels or {}},
        "HostConfig": {
            "IpcMode": ipc,
            "Privileged": privileged,
            "Devices": devices or [],
        },
        "Mounts": mounts or [],
    }


# --------------------------------------------------------------- _claimed_from_container


def test_own_label_is_read_back_exactly_regardless_of_mounts():
    info = _container(labels={container.DEVICES_LABEL: "2,3"}, ipc="private")
    assert container._claimed_from_container(info, all_ids=[0, 1, 2, 3]) == {2, 3}


def test_a_foreign_container_with_specific_device_nodes_claims_just_those():
    info = _container(devices=[
        {"PathOnHost": "/dev/tenstorrent/1", "PathInContainer": "/dev/tenstorrent/1"},
    ])
    assert container._claimed_from_container(info, all_ids=[0, 1, 2, 3]) == {1}


def test_a_foreign_bind_mount_of_a_node_is_caught_too():
    """Some tools bind-mount a node with --mount/--volume instead of --device."""
    info = _container(mounts=[{"Source": "/dev/tenstorrent/2", "Destination": "/dev/tenstorrent/2"}])
    assert container._claimed_from_container(info, all_ids=[0, 1, 2, 3]) == {2}


def test_a_foreign_whole_directory_device_mount_claims_every_id():
    info = _container(devices=[{"PathOnHost": "/dev/tenstorrent"}])
    assert container._claimed_from_container(info, all_ids=[0, 1, 2, 3]) == {0, 1, 2, 3}


def test_a_foreign_whole_directory_bind_mount_claims_every_id():
    info = _container(mounts=[{"Source": "/dev/tenstorrent"}])
    assert container._claimed_from_container(info, all_ids=[0, 1, 2, 3]) == {0, 1, 2, 3}


def test_a_private_ipc_container_claims_nothing_even_with_the_whole_directory_bound():
    """The tt-studio-backend case: broad device access for telemetry, but it cannot share
    the UMD lock without --ipc host, so it must not make every chip look busy."""
    info = _container(devices=[{"PathOnHost": "/dev/tenstorrent"}], ipc="private")
    assert container._claimed_from_container(info, all_ids=[0, 1, 2, 3]) == set()


def test_a_privileged_host_ipc_container_claims_everything_even_with_no_devices_listed():
    """--privileged reaches every node via the disabled device cgroup without any of it
    appearing in Devices/Mounts, so it must be treated as claiming the whole board."""
    info = _container(privileged=True, devices=[], mounts=[])
    assert container._claimed_from_container(info, all_ids=[0, 1, 2, 3]) == {0, 1, 2, 3}


def test_a_privileged_but_private_ipc_container_still_claims_nothing():
    info = _container(privileged=True, ipc="private")
    assert container._claimed_from_container(info, all_ids=[0, 1, 2, 3]) == set()


# ------------------------------------------------------------------------ _claimed_devices


def test_claimed_devices_aggregates_across_every_running_container(monkeypatch):
    _fake_docker(monkeypatch, ps_ids=["a", "b"], inspected=[
        _container(labels={container.DEVICES_LABEL: "0"}),
        _container(devices=[{"PathOnHost": "/dev/tenstorrent/2"}]),
    ])
    assert container._claimed_devices([0, 1, 2, 3]) == {0, 2}


def test_claimed_devices_is_empty_when_nothing_is_running(monkeypatch):
    _fake_docker(monkeypatch, ps_ids=[])
    assert container._claimed_devices([0, 1, 2, 3]) == set()


def test_claimed_devices_returns_none_when_docker_ps_fails(monkeypatch):
    _fake_docker(monkeypatch, ps_ids=["a"], ps_rc=1)
    assert container._claimed_devices([0, 1, 2, 3]) is None


def test_claimed_devices_returns_none_when_inspect_fails(monkeypatch):
    _fake_docker(monkeypatch, ps_ids=["a"], inspect_rc=1)
    assert container._claimed_devices([0, 1, 2, 3]) is None


def test_claimed_devices_returns_none_on_unparsable_inspect_output(monkeypatch):
    _fake_docker(monkeypatch, ps_ids=["a"], inspect_stdout="not json")
    assert container._claimed_devices([0, 1, 2, 3]) is None


# ------------------------------------------------------------------------ pick_free_devices


def test_pick_free_devices_returns_the_lowest_free_ids_ascending(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "pick_free_devices", _real_pick_free_devices)
    _fake_docker(monkeypatch, ps_ids=["a"], inspected=[
        _container(devices=[{"PathOnHost": "/dev/tenstorrent/0"}]),
    ])
    assert container.pick_free_devices(2, dev_root=_dev_root(tmp_path)) == [1, 2]


def test_pick_free_devices_raises_a_capacity_error_naming_whats_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "pick_free_devices", _real_pick_free_devices)
    _fake_docker(monkeypatch, ps_ids=["a"], inspected=[
        _container(devices=[{"PathOnHost": "/dev/tenstorrent"}]),
    ])
    with pytest.raises(container.ContainerError, match="only 0 of 4"):
        container.pick_free_devices(1, dev_root=_dev_root(tmp_path))


def test_pick_free_devices_raises_scan_unavailable_when_the_host_cant_be_read(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "pick_free_devices", _real_pick_free_devices)
    _fake_docker(monkeypatch, ps_ids=["a"], ps_rc=1)
    with pytest.raises(container.DeviceScanUnavailable):
        container.pick_free_devices(1, dev_root=_dev_root(tmp_path))


# ------------------------------------------------------------------------ ensure_devices_free


def test_ensure_devices_free_passes_when_nothing_claims_the_requested_ids(monkeypatch):
    _fake_docker(monkeypatch, ps_ids=[])
    container.ensure_devices_free([0, 1])  # must not raise


def test_ensure_devices_free_refuses_an_already_claimed_pin(monkeypatch):
    _fake_docker(monkeypatch, ps_ids=["a"], inspected=[
        _container(devices=[{"PathOnHost": "/dev/tenstorrent/1"}]),
    ])
    with pytest.raises(container.ContainerError, match="chip.*1.*already in use"):
        container.ensure_devices_free([0, 1])


def test_ensure_devices_free_raises_scan_unavailable_on_a_broken_scan(tmp_path, monkeypatch):
    _fake_docker(monkeypatch, ps_ids=["a"], inspect_rc=1)
    with pytest.raises(container.DeviceScanUnavailable):
        container.ensure_devices_free([0], dev_root=_dev_root(tmp_path))


def test_ensure_devices_free_refuses_a_chip_this_host_does_not_have(tmp_path, monkeypatch):
    """`--device-id 99` on a four-chip box otherwise passes every earlier check and fails
    minutes later inside docker -- the late failure the flag exists to prevent."""
    _fake_docker(monkeypatch, ps_ids=[])
    with pytest.raises(container.ContainerError, match="does not have"):
        container.ensure_devices_free([99], dev_root=_dev_root(tmp_path))


# ------------------------------------------------ scan failures the preview must tolerate


def test_an_absent_device_root_is_scan_unavailable_not_a_capacity_error(tmp_path):
    """`serve --print` on a machine with no card falls back to the whole-directory preview,
    which it can only do if this is classified as not-knowing rather than a refusal."""
    with pytest.raises(container.DeviceScanUnavailable):
        container.all_device_ids(tmp_path / "nope")


def test_a_missing_docker_binary_reads_as_a_scan_failure(monkeypatch):
    """With no docker at all `_run` raises instead of returning non-zero; `--print` has to
    keep working there, so it must normalise to the same None as an unreachable daemon."""
    def boom(argv, **kw):
        raise FileNotFoundError("docker")
    monkeypatch.setattr(container, "_run", boom)
    assert container._claimed_devices([0, 1]) is None


# ------------------------------------------------------- scoped docker run / reset composition


def _wire(**over):
    raw = json.loads(json.dumps(BASE))
    raw.update(over)
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    return m.to_wire(image_tag="tt-model/my-model:abc123", tt_metal_version="0.72.1",
                     tt_kernel_version="0.1.0", hostname="h",
                     created_at="2026-01-01T00:00:00+00:00")


_SINGLE_CHIP = [{"name": "p150", "hardware": "p150", "mesh_device": "P150",
                 "max_num_seqs": 32, "max_model_len": 131072}]


def _run_argv(m, **kw):
    profile = m.container.resolve_profile()
    launcher = launcher_for(m.container.kind)
    return container.compose_run(
        m, profile, launcher.serve_argv(m, profile), launcher.serve_env(m, profile),
        hf_home_dir=Path("/home/u/.cache/huggingface"),
        cache_dir=Path("/home/u/c"), weight_cache_dir=Path("/home/u/w"),
        tensor_cache_dir=Path("/home/u/t"), include_hf_token=False, **kw,
    )


def _device_flags(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--device"]


def test_scoped_ids_become_one_device_flag_per_chip():
    argv = _run_argv(_wire(), device_ids=[1, 2])
    assert _device_flags(argv) == [
        "/dev/tenstorrent/1:/dev/tenstorrent/1",
        "/dev/tenstorrent/2:/dev/tenstorrent/2",
    ]
    assert "/dev/tenstorrent" not in argv  # never the whole directory alongside them


def test_scoped_ids_are_recorded_as_a_label_for_stop_to_read_back():
    argv = _run_argv(_wire(), device_ids=[1, 2])
    assert f"{container.DEVICES_LABEL}=1,2" in argv


def test_no_ids_keeps_the_whole_directory_and_writes_no_devices_label():
    argv = _run_argv(_wire())
    assert _device_flags(argv) == ["/dev/tenstorrent"]
    assert not [a for a in argv if a.startswith(f"{container.DEVICES_LABEL}=")]


def test_a_single_chip_scope_adds_the_generic_mesh_graph_descriptor():
    """One ASIC of a fused board (half a P300) reports its real board type, which tt-metal
    cannot match to a preset -- without a descriptor it refuses to open the mesh at all."""
    argv = _run_argv(_wire(serve_profiles=_SINGLE_CHIP), device_ids=[3])
    assert ("TT_MESH_GRAPH_DESC_PATH=/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/"
            "p150_mesh_graph_descriptor.textproto") in argv


def test_a_multi_chip_scope_does_not_override_the_real_fabric_topology():
    argv = _run_argv(_wire(), device_ids=[0, 1, 2, 3])
    assert not [a for a in argv if a.startswith("TT_MESH_GRAPH_DESC_PATH=")]


def test_an_author_set_mesh_graph_descriptor_is_never_overridden():
    profiles = json.loads(json.dumps(_SINGLE_CHIP))
    profiles[0]["env"] = {"TT_MESH_GRAPH_DESC_PATH": "/custom/mine.textproto"}
    argv = _run_argv(_wire(serve_profiles=profiles), device_ids=[0])
    assert "TT_MESH_GRAPH_DESC_PATH=/custom/mine.textproto" in argv
    assert len([a for a in argv if a.startswith("TT_MESH_GRAPH_DESC_PATH=")]) == 1


def test_the_reset_container_is_scoped_to_the_same_ids():
    argv = container.compose_reset_mesh("img", device_ids=[2, 3])
    assert _device_flags(argv) == [
        "/dev/tenstorrent/2:/dev/tenstorrent/2",
        "/dev/tenstorrent/3:/dev/tenstorrent/3",
    ]
