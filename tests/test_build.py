# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Staging + provenance, against a fabricated tt-metal tree. No docker."""

from pathlib import Path

import pytest
import yaml

from tt_model import build
from tt_model.manifest import load_manifest

from conftest import EXAMPLES


def _fake_metal(root: Path) -> Path:
    """The minimum tree stage() consults: a tt_metal/ marker and the code paths."""
    metal = root / "tt-metal"
    (metal / "tt_metal").mkdir(parents=True)
    (metal / "models" / "common").mkdir(parents=True)
    (metal / "models" / "common" / "lightweightmodule.py").write_text("x = 1\n")
    mdl = metal / "models" / "autoports" / "poolside_laguna_xs_2_1"
    (mdl / "tt").mkdir(parents=True)
    (mdl / "tt" / "generator_vllm.py").write_text("class L: pass\n")
    (mdl / "vllm_ext" / "extra_models" / "laguna").mkdir(parents=True)
    (mdl / "vllm_ext" / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n")
    (mdl / "doc" / "datatype_sweep").mkdir(parents=True)
    (mdl / "doc" / "datatype_sweep" / "selected_precision_config.json").write_text("{}")
    # junk that the staging ignore list must NOT ship
    (mdl / "tt" / "__pycache__").mkdir()
    (mdl / "tt" / "__pycache__" / "generator_vllm.cpython-312.pyc").write_text("junk")
    (mdl / "tt" / "weights.safetensors").write_text("not really")
    return metal


def _manifest_for(metal: Path, tmp_path: Path) -> Path:
    raw = yaml.safe_load((EXAMPLES / "laguna-xs-2.1.yaml").read_text())
    raw["source"]["tt_metal"] = str(metal)
    p = tmp_path / "tt-model.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    return p


def test_stage_code_ships_the_allowlist_and_only_that(tmp_path):
    metal = _fake_metal(tmp_path)
    m = load_manifest(_manifest_for(metal, tmp_path))
    dest = tmp_path / "code"
    tree = build.stage_code(m, metal, dest)
    assert (dest / "models/common/lightweightmodule.py").exists()
    assert (dest / "models/autoports/poolside_laguna_xs_2_1/tt/generator_vllm.py").exists()
    assert (dest / "models/autoports/poolside_laguna_xs_2_1/doc/datatype_sweep/"
                   "selected_precision_config.json").exists()   # the doc/ trap
    # junk filtered
    assert not list(dest.rglob("__pycache__"))
    assert not list(dest.rglob("*.safetensors"))
    assert any(e.startswith("models/common") for e in tree)


def test_missing_allowlist_entry_raises_not_skips(tmp_path):
    """The silent miss is the failure mode where the package ImportErrors on a
    consumer long after the push looked fine."""
    metal = _fake_metal(tmp_path)
    mpath = _manifest_for(metal, tmp_path)
    raw = yaml.safe_load(mpath.read_text())
    raw["source"]["code"].append("models/autoports/typo_dir")
    mpath.write_text(yaml.safe_dump(raw, sort_keys=False))
    m = load_manifest(mpath)
    with pytest.raises(build.BuildError, match="typo_dir"):
        build.stage_code(m, metal, tmp_path / "code")


def test_stage_writes_ctx_and_pins_manifest(tmp_path, monkeypatch):
    metal = _fake_metal(tmp_path)
    mpath = _manifest_for(metal, tmp_path)
    monkeypatch.setattr(build, "resolve_git_ref", lambda repo, ref: "f" * 40)

    staged = build.stage(mpath, out_root=tmp_path / "out")

    ctx = staged.ctx
    assert (ctx / "Dockerfile").exists()
    assert (ctx / "entrypoint.sh").exists()
    assert (ctx / "install_engine.sh").exists()
    assert (ctx / "verify.sh").exists()
    assert (ctx / "code" / "models" / "common").is_dir()

    m = staged.manifest
    assert m.built["image"] == staged.image
    assert m.built["tt_metal"]["mode"] == "local"
    assert m.runtime["plugin"]["sha"] == "f" * 40           # ref pinned
    assert m.built["code_sha256"]
    # install script carries the pinned sha, not the branch name
    assert "f" * 40 in (ctx / "install_engine.sh").read_text()
    # build args wire the type + extension through
    assert staged.build_args["TT_MODEL_TYPE"] == "vllm"
    assert staged.build_args["EXTRA_MODELS_DIR"].endswith("vllm_ext/extra_models")
    assert staged.build_args["PYTHON_VERSION"] == "3.12"
    argv = build.build_argv(staged)
    assert "--progress=plain" in argv
    assert f"metalsrc={metal}" in " ".join(argv)


def test_scm_version_scheme(tmp_path, monkeypatch):
    cases = {
        "v0.74.0-0-gabc1234": "0.74.0",                    # exact tag
        "v0.74.0-219-gabc1234": "0.74.1.dev219",           # guess_next_dev
        "v0.74.0-219-gabc1234-dirty": "0.74.1.dev219+gabc1234",
        None: "0.0.0.dev0",                                # no tags at all
    }
    for desc, expect in cases.items():
        monkeypatch.setattr(build, "_git", lambda repo, *a, _d=desc: _d)
        assert build.scm_version(Path(".")) == expect, desc


def test_lock_is_copied_when_named(tmp_path, monkeypatch):
    metal = _fake_metal(tmp_path)
    mpath = _manifest_for(metal, tmp_path)
    raw = yaml.safe_load(mpath.read_text())
    raw["runtime"]["lock"] = "requirements.lock"
    mpath.write_text(yaml.safe_dump(raw, sort_keys=False))
    (tmp_path / "requirements.lock").write_text("vllm==0.24.0\n")
    monkeypatch.setattr(build, "resolve_git_ref", lambda repo, ref: "f" * 40)

    staged = build.stage(mpath, out_root=tmp_path / "out")
    assert (staged.ctx / "requirements.lock").read_text() == "vllm==0.24.0\n"
    # with a lock, the engine install must NOT resolve live
    script = (staged.ctx / "install_engine.sh").read_text()
    assert "requirements.lock" in script and "--no-deps" in script


def test_missing_lock_raises(tmp_path, monkeypatch):
    metal = _fake_metal(tmp_path)
    mpath = _manifest_for(metal, tmp_path)
    raw = yaml.safe_load(mpath.read_text())
    raw["runtime"]["lock"] = "nope.lock"
    mpath.write_text(yaml.safe_dump(raw, sort_keys=False))
    monkeypatch.setattr(build, "resolve_git_ref", lambda repo, ref: "f" * 40)
    with pytest.raises(build.BuildError, match="nope.lock"):
        build.stage(mpath, out_root=tmp_path / "out")
