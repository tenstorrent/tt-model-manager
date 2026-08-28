# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Unit tests for the runtime helpers (models dir resolution + weights download).

All network side effects are monkeypatched — nothing is actually downloaded.
"""

from pathlib import Path

from tt_kernel import runtime
from tt_kernel.manifest import WeightsRef


# --------------------------------------------------------------------------- models dir

def test_resolve_models_dir_flag_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(runtime.ENV_MODELS_DIR, str(tmp_path / "env"))
    out = runtime.resolve_models_dir(str(tmp_path / "flag"), "org/model")
    assert out == tmp_path / "flag" / "org" / "model"


def test_resolve_models_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv(runtime.ENV_MODELS_DIR, str(tmp_path / "env"))
    out = runtime.resolve_models_dir(None, "org/model")
    assert out == tmp_path / "env" / "org" / "model"


def test_resolve_models_dir_default(monkeypatch):
    monkeypatch.delenv(runtime.ENV_MODELS_DIR, raising=False)
    monkeypatch.setenv("HOME", "/home/someone")
    out = runtime.resolve_models_dir(None, "org/model")
    assert out == Path("/home/someone/.cache/tt-model/models/org/model")


def test_resolve_models_dir_no_org():
    out = runtime.resolve_models_dir("/tmp/x", "just-a-name")
    assert out == Path("/tmp/x/just-a-name")


# --------------------------------------------------------------------------- weights

def test_download_weights_forwards_args(monkeypatch, tmp_path):
    seen = {}

    def fake_snapshot(**kwargs):
        seen.update(kwargs)
        return str(tmp_path / "dl")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    w = WeightsRef(repo_id="org/m", revision="abc", allow_patterns=["*.safetensors"])
    out = runtime.download_weights(w, tmp_path / "dest")
    assert seen["repo_id"] == "org/m"
    assert seen["revision"] == "abc"
    assert seen["allow_patterns"] == ["*.safetensors"]
    assert seen["repo_type"] == "model"
    assert out == tmp_path / "dl"
