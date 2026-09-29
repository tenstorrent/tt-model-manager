# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""v6 thin bundles auto-record which shipped generic_op wheels model.py imports.

`Deps.custom_ops` is metadata only: the wheels still ship + install via `Deps.wheels` exactly as
before. The recorded set is derived from what model.py actually imports, matched against what each
`--ops-wheel` wheel provides — so an author never hand-maintains the list, and a stray or
string-dispatched op never breaks a bundle (it just isn't recorded).
"""

import zipfile

from typer.testing import CliRunner

from tt_kernel import cli, packaging
from tt_kernel.manifest import Mesh, Resources, WeightsRef

_runner = CliRunner()


def _wheel(path, *, top_level=None, files=()):
    """A real (readable-as-zip) wheel. If ``top_level`` is given, write a top_level.txt naming it;
    otherwise the import name must be derivable from the archive's package dirs (``files``)."""
    dist = "pkg-0.1.dist-info"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{dist}/METADATA", "Metadata-Version: 2.1\nName: pkg\nVersion: 0.1\n")
        if top_level is not None:
            z.writestr(f"{dist}/top_level.txt", "\n".join(top_level) + "\n")
        for f in files:
            z.writestr(f, "")
    return path


def _stage(tmp_path, *, model_src, plugin_wheel=None, extra_wheels=None):
    model_py = tmp_path / "model.py"
    model_py.write_text(model_src)
    staged = tmp_path / "thin"
    m = packaging.stage_thin_package(
        staged, name="m", arch="blackhole", model_py=model_py,
        vllm_metadata={"arch": "M", "main_class": "model:M"}, tt_kernel_version="0.0.0",
        plugin_wheel=plugin_wheel, extra_wheels=extra_wheels,
        weights=WeightsRef(repo="org/w"), mesh=Mesh(devices=1, topology="P150"),
        resources=Resources(max_num_seqs=8, block_size=64),
    )
    return staged, m


# -- the field ------------------------------------------------------------------------

def test_imported_ops_wheel_is_recorded_and_plugin_is_not(tmp_path):
    pw = _wheel(tmp_path / "vllm_tt_plugin-0.1-py3-none-any.whl", top_level=["vllm_tt_plugin"])
    ow = _wheel(tmp_path / "my_ops-0.1-py3-none-any.whl", top_level=["myops"])
    staged, m = _stage(tmp_path, model_src="import myops\nclass M: pass\n",
                       plugin_wheel=pw, extra_wheels=[ow])
    # the model imports the ops package -> the ops wheel is recorded
    assert m.deps.custom_ops == [f"wheels/{ow.name}"]
    # the plugin is infrastructure, never a "custom op", even though model.py could import it
    assert f"wheels/{pw.name}" not in m.deps.custom_ops
    # the install contract is untouched: both wheels still ship + install by path, plugin first
    assert m.deps.wheels == [f"wheels/{pw.name}", f"wheels/{ow.name}"]
    assert (staged / "wheels" / ow.name).is_file()


def test_shipped_ops_wheel_not_imported_is_omitted(tmp_path):
    """model.py imports nothing from the wheel -> not recorded, but still shipped + installed."""
    ow = _wheel(tmp_path / "my_ops-0.1-py3-none-any.whl", top_level=["myops"])
    staged, m = _stage(tmp_path, model_src="class M: pass\n", extra_wheels=[ow])
    assert m.deps.custom_ops == []
    assert m.deps.wheels == [f"wheels/{ow.name}"]          # shipping/install unaffected


def test_wheel_without_top_level_txt_is_matched_by_derived_name(tmp_path):
    """A wheel that omits top_level.txt: the import name comes from its package dir."""
    ow = _wheel(tmp_path / "my_ops-0.1-py3-none-any.whl", files=["myops/__init__.py"])
    _staged, m = _stage(tmp_path, model_src="from myops import fast_op\nclass M: pass\n",
                        extra_wheels=[ow])
    assert m.deps.custom_ops == [f"wheels/{ow.name}"]


def test_string_dispatched_op_with_no_import_is_not_detected(tmp_path):
    """ttnn.generic_op("name", ...) with no Python import of the wheel is a documented blind spot:
    not recorded, but the wheel still ships + installs so serving is unaffected."""
    ow = _wheel(tmp_path / "my_ops-0.1-py3-none-any.whl", top_level=["myops"])
    _staged, m = _stage(tmp_path,
                        model_src="import ttnn\nclass M:\n    def f(self): ttnn.generic_op('myop')\n",
                        extra_wheels=[ow])
    assert m.deps.custom_ops == []
    assert m.deps.wheels == [f"wheels/{ow.name}"]


def test_cli_warns_when_a_shipped_ops_wheel_is_not_imported(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    (tmp_path / "model.py").write_text("class M: pass\n")   # imports nothing
    ow = _wheel(tmp_path / "my_ops-0.1-py3-none-any.whl", top_level=["myops"])
    out = tmp_path / "out"
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(tmp_path / "model.py"), "--arch", "blackhole",
        "--arch-name", "M", "--main-class", "model:M", "--weights", "org/w",
        "--ops-wheel", str(ow), "--no-vllm", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "not imported" in res.output and ow.name in res.output


# -- the pure helpers -----------------------------------------------------------------

def test_wheel_top_level_packages_prefers_top_level_txt(tmp_path):
    w = _wheel(tmp_path / "a-1-py3-none-any.whl", top_level=["myops", "_myops_c"],
               files=["myops/__init__.py"])
    assert packaging.wheel_top_level_packages(w) == {"myops", "_myops_c"}


def test_wheel_top_level_packages_derives_without_top_level_txt(tmp_path):
    w = _wheel(tmp_path / "a-1-py3-none-any.whl",
               files=["myops/__init__.py", "myops/kernels.py"])
    assert packaging.wheel_top_level_packages(w) == {"myops"}


def test_wheel_top_level_packages_single_module_strips_py(tmp_path):
    w = _wheel(tmp_path / "a-1-py3-none-any.whl", files=["solo.py"])
    assert packaging.wheel_top_level_packages(w) == {"solo"}


def test_wheel_top_level_packages_on_a_non_zip_is_empty(tmp_path):
    junk = tmp_path / "b.whl"; junk.write_bytes(b"not a zip")
    assert packaging.wheel_top_level_packages(junk) == set()


def test_imported_top_level_collects_absolute_imports_only(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("import a\nimport b.c\nfrom d import e\nfrom . import f\nfrom .g import h\n")
    assert packaging.imported_top_level(p) == {"a", "b", "d"}   # relative imports name nothing


def test_imported_top_level_on_a_syntax_error_is_empty(tmp_path):
    p = tmp_path / "m.py"; p.write_text("def (:\n")
    assert packaging.imported_top_level(p) == set()
