# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Shared fixtures for the whole suite."""

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
