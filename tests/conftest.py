# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Shared fixtures for the whole suite."""

import pytest

from tt_kernel import container


@pytest.fixture(autouse=True)
def _no_forced_color(monkeypatch):
    """Keep the offline suite's output deterministic regardless of the runner.

    Rich/click force ANSI colour when they detect a CI environment (``GITHUB_ACTIONS`` /
    ``CI`` / ``FORCE_COLOR``), even for output that is piped, not a TTY. That breaks the
    ``test_cli_output`` "no escape codes when piped" assertions on a GitHub runner while they
    pass on a developer box where those vars are unset. Remove the colour-forcing signals so
    piped output is judged the way a plain pipe would be — the behaviour the tests assert.
    """
    for var in ("GITHUB_ACTIONS", "CI", "FORCE_COLOR"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _hugepages_reachable(monkeypatch):
    """Treat the real ``/dev/hugepages-1G`` as reachable so device tests stay hermetic.

    The rootless preflight checks writability of ``HUGEPAGES_MOUNT`` with a real
    ``os.access`` (``_reachable_in_userns``), which passes on a provisioned TT host but fails
    on a runner that has no such mount — flagging ``hugepages`` in tests that only mean to
    exercise the device checks. Stub only the real default path; a test that exercises
    hugepages reachability points ``HUGEPAGES_MOUNT`` at its own tmp fake and is unaffected.
    """
    real = container._reachable_in_userns

    def _stub(path, *, mode):
        if str(path) == "/dev/hugepages-1G":
            return True
        return real(path, mode=mode)

    monkeypatch.setattr(container, "_reachable_in_userns", _stub)


#: Captured at import, before any fixture can stub it, so a test that wants the genuine
#: picker (against a faked docker + a temporary device root) can ask for it by fixture.
_REAL_PICK_FREE_DEVICES = container.pick_free_devices


@pytest.fixture
def real_picker(monkeypatch):
    """Undo ``_no_real_device_scan`` for a test that exercises the picker itself."""
    monkeypatch.setattr(container, "pick_free_devices", _REAL_PICK_FREE_DEVICES)
    return _REAL_PICK_FREE_DEVICES


@pytest.fixture(autouse=True)
def _no_real_device_scan(monkeypatch):
    """Stub the free-chip picker so no test shells out to docker or reads /dev/tenstorrent.

    ``container.pick_free_devices`` is the one thing ``serve_container``'s real-launch path
    calls that per-test setup doesn't already mock, and it genuinely inspects the host
    (``docker ps``/``inspect``, ``/dev/tenstorrent``) — exactly what a unit test must not
    depend on. Ascending indices match the picker's own real contract, so tests asserting on
    device flags see the same low numbers an uncontended host would produce. A test that
    exercises the picker itself re-patches this within its own body.
    """
    monkeypatch.setattr(container, "pick_free_devices",
                        lambda count, dev_root=None: list(range(count)))
