# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""A v6 bundle's tt_metal_version is the ttnn it pins, not whatever the packaging host has."""

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, metal, packaging
from tt_kernel.manifest import Manifest

_runner = CliRunner()


@pytest.fixture
def stale_host(monkeypatch):
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.65.1rc17.dev6200")


def _package(tmp_path, requirements, *extra):
    model_py = tmp_path / "model.py"
    model_py.write_text("class C: pass\n")
    args = ["package-thin", "--model-py", str(model_py), "--arch", "blackhole",
            "--arch-name", "A", "--main-class", "model:C", "--out", str(tmp_path / "out"), *extra]
    if requirements is not None:
        req = tmp_path / "requirements.txt"
        req.write_text(requirements)
        args += ["--requirements", str(req)]
    res = _runner.invoke(cli.app, args)
    assert res.exit_code == 0, res.output
    return Manifest.from_json((tmp_path / "out" / "tt_kernel_manifest.json").read_text())


def test_version_comes_from_the_ttnn_pin(tmp_path, stale_host):
    m = _package(tmp_path, "torch\nttnn==0.77.0  # engine\n")
    assert m.tt_metal_version == "0.77.0"


def test_override_flag_wins(tmp_path, stale_host):
    m = _package(tmp_path, "ttnn==0.77.0\n", "--tt-metal-version", "0.78.0")
    assert m.tt_metal_version == "0.78.0"


def test_host_version_only_without_an_exact_pin(tmp_path, stale_host):
    assert _package(tmp_path, "ttnn>=0.77\n").tt_metal_version == "0.65.1rc17.dev6200"


@pytest.mark.parametrize("text,want", [
    ("ttnn==0.78.0\n", "0.78.0"),
    ("TTNN == 0.78.0 ; python_version >= '3.10'\n", "0.78.0"),
    ("tt-metal-models==0.79.0\n", "0.79.0"),
    ("tt_metal_models==0.79.0\nttnn==0.78.0\n", "0.78.0"),
    ("# ttnn==0.1.0\nttnn-extras==2\n", None),
    ("ttnn>=0.77\n", None),
])
def test_pinned_ttnn_version(text, want):
    assert packaging.pinned_ttnn_version(text) == want
