# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Shared isolation: every test runs with its own XDG/HOME state so nothing reads or
writes the developer's real caches. (The pattern the old suite proved out.)"""

from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)


@pytest.fixture()
def laguna():
    from tt_model.manifest import load_manifest

    return load_manifest(EXAMPLES / "laguna-xs-2.1.yaml")


@pytest.fixture()
def ornith():
    from tt_model.manifest import load_manifest

    return load_manifest(EXAMPLES / "ornith-1.0-35b.yaml")
