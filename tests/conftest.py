# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Shared fixtures for the whole suite."""

import os

import pytest

from tt_kernel import container


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


@pytest.fixture(autouse=True)
def _hermetic_alloc_lock(monkeypatch, tmp_path_factory):
    """Point the allocation flock at a scratch file instead of the real device directory.

    ``alloc_lock`` flocks ``/dev/tenstorrent`` itself (see ``_open_alloc_lock``), which a
    test must not touch — but the lock's own logic (contention timeout, release, close) is
    worth keeping under test, so only the descriptor is redirected, not the mechanism.
    """
    lock = tmp_path_factory.mktemp("alloc-lock") / "lock"
    lock.touch()
    monkeypatch.setattr(container, "_open_alloc_lock",
                        lambda dev_root=None: os.open(str(lock), os.O_RDONLY))
