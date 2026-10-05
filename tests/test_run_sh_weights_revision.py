# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""run.sh must serve the weights revision the manifest pins, not whatever the repo's tip is."""

import shlex

from tt_kernel import packaging
from tt_kernel.manifest import WeightsRef

SHA = "0123456789abcdef0123456789abcdef01234567"


def _stage(tmp_path, *, kind="vllm", revision=SHA):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_py = tmp_path / "model.py"
    model_py.write_text("class M:\n    pass\n")
    kw = dict(vllm_metadata={"arch": "M", "main_class": "model:M"})
    if kind != "vllm":
        kw = dict(app="server:app", with_vllm=False)
    packaging.stage_thin_package(
        tmp_path / "b", name="m", arch="blackhole", model_py=model_py, tt_kernel_version="0",
        kind=kind, weights=WeightsRef(repo="org/weights", revision=revision), **kw,
    )
    return (tmp_path / "b" / "run.sh").read_text()


def _cmd_argv(run_sh: str):
    line = next(ln for ln in run_sh.splitlines() if ln.startswith("CMD=("))
    return shlex.split(line[len("CMD=("):-1])


def test_vllm_run_sh_passes_the_pinned_revision_to_vllm(tmp_path):
    argv = _cmd_argv(_stage(tmp_path))
    assert argv[argv.index("--revision") + 1] == SHA
    assert argv[argv.index("--tokenizer-revision") + 1] == SHA
    # Before "$@", so a revision typed on `tt-model serve` still wins (argparse last-wins).
    assert argv.index("--revision") < argv.index("$@")


def test_run_sh_exports_the_pinned_revision(tmp_path):
    for kind in ("vllm", "tt-dit-server"):
        run = _stage(tmp_path / kind, kind=kind)
        assert f'export TT_MODEL_WEIGHTS_REVISION="${{TT_MODEL_WEIGHTS_REVISION:-{SHA}}}"' in run


def test_unpinned_weights_render_no_revision(tmp_path):
    run = _stage(tmp_path, revision=None)
    assert "--revision" not in _cmd_argv(run)
    assert "TT_MODEL_WEIGHTS_REVISION" not in run
