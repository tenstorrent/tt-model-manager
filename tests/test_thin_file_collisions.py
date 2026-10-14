# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""package-thin must not silently replace files it does not own.

The runner is copied to the bundle root under its own name, and a push writes every staged
top-level file into the repo. Both used to clobber a same-named file without a word.
"""

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, hub, packaging

_runner = CliRunner()


def _runner_file(tmp_path, name):
    p = tmp_path / "src" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("class C: pass\n")
    return p


@pytest.mark.parametrize("name", ["requirements.txt", "run.sh", "install.sh",
                                  "tt_kernel_manifest.json", "vllm-overrides.txt"])
def test_runner_named_like_a_bundle_file_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match=name):
        packaging.stage_thin_package(
            tmp_path / "b", name="m", arch="blackhole", model_py=_runner_file(tmp_path, name),
            tt_kernel_version="0", vllm_metadata={"arch": "A", "main_class": "model:C"},
        )


def test_cli_reports_a_runner_collision_cleanly(tmp_path):
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(_runner_file(tmp_path, "run.sh")), "--arch", "blackhole",
        "--arch-name", "A", "--main-class", "model:C", "--out", str(tmp_path / "out"),
    ])
    assert res.exit_code == 1
    assert "run.sh" in res.output and "Traceback" not in res.output


class _Api:
    def __init__(self, remote):
        self.remote, self.uploaded = remote, False

    def list_repo_files(self, **kw):
        return list(self.remote)

    def upload_folder(self, **kw):
        self.uploaded = True


def _staged(tmp_path):
    d = tmp_path / "staged"
    d.mkdir()
    for f in ("app.py", "requirements.txt", "run.sh"):
        (d / f).write_text("x")
    return d


def test_push_refuses_to_overwrite_files_of_a_repo_that_is_not_a_bundle(monkeypatch, tmp_path):
    api = _Api(["app.py", "requirements.txt", "README.md", ".gitattributes"])
    monkeypatch.setattr(hub, "_api", lambda: api)
    with pytest.raises(hub.ForeignFilesError) as exc:
        hub.push_folder("me/m", _staged(tmp_path), "msg", refuse_foreign=True)
    assert exc.value.paths == ["app.py", "requirements.txt"]
    assert not api.uploaded


@pytest.mark.parametrize("remote", [
    [],                                                  # new, empty repo
    ["README.md", ".gitattributes"],                     # nothing in common
    ["app.py", "requirements.txt", "tt_kernel_manifest.json"],  # already a bundle: an update
])
def test_push_proceeds_when_nothing_foreign_is_overwritten(monkeypatch, tmp_path, remote):
    api = _Api(remote)
    monkeypatch.setattr(hub, "_api", lambda: api)
    hub.push_folder("me/m", _staged(tmp_path), "msg", refuse_foreign=True)
    assert api.uploaded


def test_package_thin_push_stops_before_clobbering(monkeypatch, tmp_path):
    api = _Api(["app.py", "README.md"])
    monkeypatch.setattr(hub, "_api", lambda: api)
    monkeypatch.setattr(cli, "_ensure_repo", lambda *a, **k: None)
    res = _runner.invoke(cli.app, [
        "package-thin", "me/m", "--model-py", str(_runner_file(tmp_path, "app.py")),
        "--kind", "tt-dit-server", "--app", "server.app:app", "--arch", "blackhole",
    ])
    assert res.exit_code == 1, res.output
    assert "app.py" in res.output and "Traceback" not in res.output
    assert not api.uploaded
