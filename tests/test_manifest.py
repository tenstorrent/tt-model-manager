# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The manifest validation matrix — every rejection is a boot failure caught early."""

import copy

import pytest
import yaml

from tt_model.manifest import (
    Manifest,
    ManifestError,
    hardware_chip_count,
    load_manifest,
    parse_mesh_device,
)

from conftest import EXAMPLES


def _raw(name):
    return yaml.safe_load((EXAMPLES / name).read_text())


def _validate(raw):
    m = Manifest.model_validate(raw)
    m.validate_semantics()
    return m


# ---------------------------------------------------------------- the examples
def test_both_examples_load_and_roundtrip():
    for name in ("laguna-xs-2.1.yaml", "ornith-1.0-35b.yaml", "muse-glimmer-30b.yaml"):
        m = load_manifest(EXAMPLES / name)
        again = Manifest.model_validate(yaml.safe_load(m.to_yaml()))
        again.validate_semantics()
        assert again.name == m.name


def test_laguna_profiles_are_device_targets(laguna):
    assert laguna.profile_names() == ["p150x4", "p150x2"]
    assert laguna.resolved_default() == "p150x4"
    p2 = laguna.resolve_profile("p150x2")
    assert p2.max_model_len == 65536                      # profile override
    assert p2.additional_config["tt"]["fabric_config"] == "FABRIC_1D"
    assert p2.env["TT_LAGUNA_PIPE_CHUNK"] == "2048"       # inherited from serve:


def test_ornith_profiles_are_deployment_shapes(ornith):
    lat = ornith.resolve_profile("p150x4-latency")
    cap = ornith.resolve_profile("p150x4-capacity")
    assert (lat.max_num_seqs, cap.max_num_seqs) == (1, 32)
    # everything else is shared through serve:
    assert lat.max_model_len == cap.max_model_len == 262144
    assert lat.hardware == cap.hardware == "p150x4"


# ------------------------------------------------------------------ rejections
def test_unknown_type_lists_the_registry():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["type"] = "diffusers"
    with pytest.raises(ManifestError, match="vllm, vllm-legacy"):
        _validate(raw)


def test_illegal_mesh_device_fails_at_load_not_at_boot():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["serve_profiles"][0]["mesh_device"] = "p300c"
    with pytest.raises(ManifestError, match="invalid mesh_device"):
        _validate(raw)


def test_mesh_chip_count_must_match_hardware():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["serve_profiles"][0]["mesh_device"] = "P150x2"  # (1,2) vs hardware p150x4
    with pytest.raises(ManifestError, match="implies 4"):
        _validate(raw)


def test_multiple_profiles_require_a_default():
    raw = _raw("laguna-xs-2.1.yaml")
    del raw["default_profile"]
    with pytest.raises(ManifestError, match="default_profile"):
        _validate(raw)


def test_default_must_name_a_profile():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["default_profile"] = "nope"
    with pytest.raises(ManifestError, match="names no profile"):
        _validate(raw)


def test_duplicate_profile_names_rejected():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["serve_profiles"][1]["name"] = raw["serve_profiles"][0]["name"]
    raw["default_profile"] = raw["serve_profiles"][0]["name"]
    with pytest.raises(ManifestError, match="duplicate"):
        _validate(raw)


def test_empty_profiles_rejected():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["serve_profiles"] = []
    with pytest.raises(Exception):
        _validate(raw)


@pytest.mark.parametrize("field", ["max_num_seqs", "block_size", "mesh_device", "hardware"])
def test_required_merged_fields(field):
    """max_num_seqs / block_size: the TT backend rejects vLLM's own defaults, so a
    profile without them would boot straight into a failure."""
    raw = _raw("laguna-xs-2.1.yaml")
    for prof in raw["serve_profiles"]:
        prof.pop(field, None)
    raw.get("serve", {}).pop(field, None)
    with pytest.raises(ManifestError, match=field):
        _validate(raw)


def test_vllm_plugin_accepts_pypi_release():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["runtime"]["plugin"] = {"version": "0.2.0"}
    _validate(raw)  # a release that already registers the model is legal


def test_vllm_plugin_rejects_both_forms_at_once():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["runtime"]["plugin"]["version"] = "0.2.0"
    with pytest.raises(ManifestError, match="not both"):
        _validate(raw)


def test_vllm_plugin_requires_some_pin():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["runtime"]["plugin"] = {}
    with pytest.raises(ManifestError, match="repo, ref.*version"):
        _validate(raw)


def test_vllm_type_requires_version_not_fork():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["runtime"]["vllm"] = {"repo": "https://github.com/tenstorrent/vllm", "ref": "dev"}
    with pytest.raises(ManifestError, match="vllm-legacy"):
        _validate(raw)


def test_legacy_type_requires_model_dir():
    raw = _raw("ornith-1.0-35b.yaml")
    del raw["runtime"]["model_dir"]
    with pytest.raises(ManifestError, match="model_dir"):
        _validate(raw)


def test_legacy_model_dir_must_be_shipped():
    raw = _raw("ornith-1.0-35b.yaml")
    raw["runtime"]["model_dir"] = "models/autoports/somewhere_else"
    with pytest.raises(ManifestError, match="not covered by source.code"):
        _validate(raw)


def test_absolute_code_paths_rejected():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["source"]["code"].append("/etc/passwd")
    with pytest.raises(Exception, match="relative"):
        _validate(raw)


# -------------------------------------------------------------------- helpers
def test_deep_merge_profile_overrides_win():
    raw = _raw("laguna-xs-2.1.yaml")
    raw["serve_profiles"][0]["port"] = 9999
    raw["serve_profiles"][0]["args"] = ["--only-this"]
    m = _validate(raw)
    p = m.resolve_profile("p150x4")
    assert p.port == 9999
    assert p.flat_args() == ["--only-this"]           # override wholesale, not append
    assert p.env["TT_LAGUNA_PIPE_CHUNK"] == "2048"    # untouched keys inherited


def test_unknown_profile_error_lists_available(laguna):
    with pytest.raises(ManifestError, match="p150x4, p150x2"):
        laguna.resolve_profile("nope")


@pytest.mark.parametrize("hw,chips", [
    ("p150", 1), ("p150x2", 2), ("p150x4", 4), ("p100", 1),
    ("p300", 2), ("p300x2", 4), ("n300", 2), ("n300x4", 8),
    ("p300c", 2), ("mystery9000", None),
])
def test_hardware_chip_count(hw, chips):
    assert hardware_chip_count(hw) == chips


@pytest.mark.parametrize("value,grid", [
    ("P150x4", (1, 4)), ("P150", (1, 1)), ("T3K", (1, 8)),
    ("(1, 4)", (1, 4)), ("(2,4)", (2, 4)),
])
def test_parse_mesh_device(value, grid):
    assert parse_mesh_device(value) == grid
