# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`serve --profile` / `--device-id` on a v5/v6 bundle: ignored, but not silently."""

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, localdb

_runner = CliRunner()
_ID = "org/thin"


@pytest.fixture
def installed(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    inst = tmp_path / "models" / "org" / "thin"
    inst.mkdir(parents=True)
    run_sh = inst / "run.sh"
    run_sh.write_text("exit 0\n")
    localdb.record(_ID, {"repo_id": _ID, "self_contained": True, "install_dir": str(inst),
                         "bundle_path": str(inst), "run_script": str(run_sh)})


@pytest.mark.parametrize("flags, named", [
    (["--profile", "batch32"], ["--profile"]),
    (["--device-id", "0,1"], ["--device-id"]),
    (["--profile", "batch32", "--device-id", "0,1"], ["--profile", "--device-id"]),
])
def test_container_only_flags_on_a_bundle_are_reported(installed, flags, named):
    res = _runner.invoke(cli.app, ["serve", _ID, "--local-only", *flags])
    assert res.exit_code == 0, res.output
    out = " ".join(res.output.split())
    one = len(named) == 1
    assert ("applies only to container packages" if one else "apply only to container packages") in out
    assert ("so it was ignored" if one else "so they were ignored") in out
    for f in named:
        assert f in out


def test_no_note_without_those_flags(installed):
    res = _runner.invoke(cli.app, ["serve", _ID, "--local-only"])
    assert res.exit_code == 0, res.output
    assert "only to container packages" not in res.output
