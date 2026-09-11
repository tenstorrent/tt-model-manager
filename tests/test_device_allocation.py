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

import pytest

from tt_kernel import container

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


def test_ensure_devices_free_raises_scan_unavailable_on_a_broken_scan(monkeypatch):
    _fake_docker(monkeypatch, ps_ids=["a"], inspect_rc=1)
    with pytest.raises(container.DeviceScanUnavailable):
        container.ensure_devices_free([0])
