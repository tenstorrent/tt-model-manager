# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""v6 thin bundles can ship a tree of hand-written pure-Python model ops (issue #127).

The tree travels in the bundle under extra_code/, goes on PYTHONPATH ahead of the runner so
`import <pkg>` resolves, and a `verify` hook runs install-time sanity checks (import the tree /
assert the ttnn version it was validated against). Additive and v6-only; nothing about how ops are
installed changes.
"""

import shlex

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, packaging
from tt_kernel.manifest import Manifest, Mesh, Resources, WeightsRef

_runner = CliRunner()


def _ops_tree(tmp_path, pkg="myops"):
    """A directory whose top-level package is `pkg` (what model.py would import)."""
    d = tmp_path / "ops_src"
    (d / pkg).mkdir(parents=True)
    (d / pkg / "__init__.py").write_text("def fast_attn(): ...\n")
    (d / pkg / "kernels.py").write_text("x = 1\n")
    (d / pkg / "__pycache__").mkdir()
    (d / pkg / "__pycache__" / "junk.pyc").write_text("junk")
    return d


def _stage(tmp_path, *, extra_code=None, verify=None, model_src="import myops\nclass M: pass\n"):
    model_py = tmp_path / "model.py"
    model_py.write_text(model_src)
    staged = tmp_path / "thin"
    m = packaging.stage_thin_package(
        staged, name="m", arch="blackhole", model_py=model_py,
        vllm_metadata={"arch": "M", "main_class": "model:M"}, tt_kernel_version="0.0.0",
        extra_code=extra_code, verify=verify,
        weights=WeightsRef(repo="org/w"), mesh=Mesh(devices=1, topology="P150"),
        resources=Resources(max_num_seqs=8, block_size=64),
    )
    return staged, m


# -- shipping the ops tree ------------------------------------------------------------

def test_extra_code_tree_travels_in_the_bundle_and_is_on_pythonpath(tmp_path):
    staged, m = _stage(tmp_path, extra_code=_ops_tree(tmp_path))
    # staged under extra_code/, preserving the author's package layout
    assert (staged / "extra_code" / "myops" / "__init__.py").is_file()
    assert (staged / "extra_code" / "myops" / "kernels.py").is_file()
    assert not (staged / "extra_code" / "myops" / "__pycache__").exists()   # junk ignored
    # manifest records it
    assert m.deps.extra_code_dir == "extra_code"
    # run.sh puts it on PYTHONPATH ahead of the bundle root (so `import myops` resolves)
    run = (staged / "run.sh").read_text()
    pp = next(ln for ln in run.splitlines() if ln.startswith("export PYTHONPATH="))
    assert '"$HERE/extra_code:$HERE:' in pp
    # round-trips through the wire manifest
    m2 = Manifest.from_json((staged / "tt_kernel_manifest.json").read_text())
    assert m2.deps.extra_code_dir == "extra_code"


def test_no_extra_code_leaves_the_bundle_and_pythonpath_unchanged(tmp_path):
    staged, m = _stage(tmp_path, model_src="class M: pass\n")
    assert m.deps.extra_code_dir is None
    assert not (staged / "extra_code").exists()
    run = (staged / "run.sh").read_text()
    pp = next(ln for ln in run.splitlines() if ln.startswith("export PYTHONPATH="))
    assert '"$HERE:' in pp and "extra_code" not in pp


def test_a_missing_extra_code_dir_is_an_error_not_a_silent_skip(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        _stage(tmp_path, extra_code=tmp_path / "does-not-exist")


# -- the verify hook ------------------------------------------------------------------

def test_verify_statements_run_last_in_install_sh(tmp_path):
    verify = ["import myops; assert hasattr(myops, 'fast_attn')",
              "import ttnn; assert ttnn.__version__.startswith('0.77')"]
    staged, m = _stage(tmp_path, extra_code=_ops_tree(tmp_path), verify=verify)
    assert m.deps.verify == verify
    inst = (staged / "install.sh").read_text()
    for stmt in verify:
        # each statement is shell-quoted (safe against the embedded quotes) after `python -c`
        assert f'"$VENV/bin/python" -c {shlex.quote(stmt)}' in inst
    # runs AFTER the deps/wheels install (it must see the built venv)
    assert inst.index("import myops") > inst.rindex("uv pip install")


def test_no_verify_adds_no_python_c_lines(tmp_path):
    staged, _ = _stage(tmp_path, extra_code=_ops_tree(tmp_path))
    assert '"$VENV/bin/python" -c' not in (staged / "install.sh").read_text()


# -- CLI --------------------------------------------------------------------------------

def test_cli_package_thin_ships_extra_code_and_verify(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    (tmp_path / "model.py").write_text("import myops\nclass M: pass\n")
    ops = _ops_tree(tmp_path)
    out = tmp_path / "out"
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(tmp_path / "model.py"), "--arch", "blackhole",
        "--arch-name", "M", "--main-class", "model:M", "--weights", "org/w", "--no-vllm",
        "--extra-code", str(ops), "--verify", "import myops", "--out", str(out)])
    assert res.exit_code == 0, res.output
    m = Manifest.from_json((out / "tt_kernel_manifest.json").read_text())
    assert m.deps.extra_code_dir == "extra_code" and m.deps.verify == ["import myops"]
    assert (out / "extra_code" / "myops" / "__init__.py").is_file()
