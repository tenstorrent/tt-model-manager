# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The v5.1 authoring manifest: what it accepts, what it refuses, and what it renders.

Every check here is one an author would otherwise discover ten minutes into a boot (or,
worse, never — as a silent fallback at serve time). No hardware, no docker, no network.
"""

import json

import pytest
import yaml

from tt_kernel.container_manifest import (
    ContainerManifest,
    ContainerManifestError,
    hardware_chip_count,
    load_container_manifest,
    parse_mesh_device,
)
from tt_kernel.manifest import Manifest

BASE = {
    "schema": "5.1",
    "repo": "you/my-model",
    "name": "my-model",
    "weights": "org/Weights-7B",
    "kind": "vllm-plugin",
    "arch": "blackhole",
    "source": {
        "tt_metal": "/tmp/tt-metal",
        "code": ["models/common"],
        "ubuntu": "22.04",
        "python": "3.12",
    },
    "runtime": {
        "vllm": {"version": "0.24.0"},
        "plugin": {"repo": "https://github.com/tenstorrent/vllm-tt-plugin", "ref": "bc4af2d5"},
    },
    "serve": {"port": 8000, "block_size": 64},
    "serve_profiles": [
        {
            "name": "p150x4",
            "hardware": "p150x4",
            "mesh_device": "P150x4",
            "max_num_seqs": 32,
            "max_model_len": 131072,
        }
    ],
}


# The other kind, as an override: `_mani(**FORK)`.
FORK = {
    "kind": "vllm-fork",
    "runtime": {
        "vllm": {"repo": "https://github.com/tenstorrent/vllm", "ref": "bf98d556"},
        "model_dir": "models/common",
    },
}


def _mani(**over):
    raw = json.loads(json.dumps(BASE))
    raw.update(over)
    return ContainerManifest.model_validate(raw)


def _write(tmp_path, raw):
    p = tmp_path / "tt-model.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


# --------------------------------------------------------------- happy path


def test_a_minimal_manifest_loads_and_validates(tmp_path):
    m = load_container_manifest(_write(tmp_path, BASE))
    assert m.name == "my-model"
    assert m.resolved_default() == "p150x4"


def test_profile_inherits_serve_defaults():
    """`serve:` is the shared block; a profile that omits a field inherits it."""
    merged = _mani().resolve_profile("p150x4")
    assert merged.block_size == 64  # from serve:
    assert merged.port == 8000  # from serve:
    assert merged.max_num_seqs == 32  # from the profile


def test_profile_overrides_serve_defaults():
    m = _mani(
        serve={"port": 8000, "block_size": 64},
        serve_profiles=[
            {
                "name": "p150x4",
                "hardware": "p150x4",
                "mesh_device": "P150x4",
                "max_num_seqs": 32,
                "block_size": 32,
            }
        ],
    )
    assert m.resolve_profile("p150x4").block_size == 32


def test_serve_args_are_inherited_not_blanked_by_an_omitting_profile():
    m = _mani(serve={"port": 8000, "block_size": 64, "args": ["--trust-remote-code"]})
    assert m.resolve_profile("p150x4").flat_args() == ["--trust-remote-code"]


def test_flat_args_flattens_pairs():
    m = _mani(serve={"block_size": 64, "args": ["--flag", ["--opt", "v"]]})
    assert m.resolve_profile("p150x4").flat_args() == ["--flag", "--opt", "v"]


# --------------------------------------------------------------- refusals


def test_unknown_arch_is_refused():
    with pytest.raises(ContainerManifestError, match="arch must be one of"):
        _mani(arch="grayskull").validate_semantics()


def test_unknown_kind_is_refused():
    with pytest.raises(ContainerManifestError, match="kind must be one of"):
        _mani(kind="tensorrt").validate_semantics()


def test_a_fork_runtime_under_the_plugin_kind_names_the_right_kind():
    """The two stacks are easy to mix up; each diagnosis must name the other kind."""
    with pytest.raises(ContainerManifestError, match="that is kind vllm-fork"):
        _mani(runtime={"vllm": {"repo": "https://x/y", "ref": "main"}}).validate_semantics()


def test_a_stock_runtime_under_the_fork_kind_names_the_right_kind():
    with pytest.raises(ContainerManifestError, match="that is kind vllm-plugin"):
        _mani(kind="vllm-fork", runtime={"vllm": {"version": "0.24.0"},
                                         "model_dir": "models/common"}).validate_semantics()


def test_unnamespaced_repo_is_refused():
    with pytest.raises(ContainerManifestError, match="namespaced HF id"):
        _mani(repo="my-model").validate_semantics()


def test_unnamespaced_weights_is_refused():
    with pytest.raises(ContainerManifestError, match="namespaced HF id"):
        _mani(weights="Weights-7B").validate_semantics()


def test_mesh_device_outside_the_plugin_table_is_refused():
    """The plugin raises at boot on an unknown MESH_DEVICE; refuse it ~10 min earlier."""
    with pytest.raises(ContainerManifestError, match="invalid mesh_device"):
        _mani(
            serve_profiles=[
                {
                    "name": "p",
                    "hardware": "p150x4",
                    "mesh_device": "P150x9",
                    "max_num_seqs": 8,
                    "block_size": 64,
                }
            ]
        ).validate_semantics()


def test_mesh_device_that_contradicts_hardware_is_refused():
    with pytest.raises(ContainerManifestError, match="implies 4"):
        _mani(
            serve_profiles=[
                {
                    "name": "p",
                    "hardware": "p150x4",
                    "mesh_device": "P150x2",
                    "max_num_seqs": 8,
                    "block_size": 64,
                }
            ]
        ).validate_semantics()


def _with_hardware(label, mesh="P150x4"):
    return _mani(serve_profiles=[{"name": "p", "hardware": label, "mesh_device": mesh,
                                  "max_num_seqs": 8, "block_size": 64}])


@pytest.mark.parametrize("label", ["wibble", "t3k", "galaxy", "x4", "p150b7"])
def test_unrecognised_hardware_is_refused(label):
    """An unreadable label is not harmless: to_wire publishes `device_count: 1` for it and
    the mesh cross-check is skipped, so a 4-chip model ships claiming one chip and the one
    assertion tt-model makes about the mesh never runs. Both failures are silent."""
    with pytest.raises(ContainerManifestError, match="not a recognised board label"):
        _with_hardware(label).validate_semantics()


def test_a_mesh_sku_in_the_hardware_field_is_diagnosed():
    """The easy slip: `QB2` is a real thing to type, just not in this field. Naming it as a
    SKU is the difference between a fixable message and a puzzling one."""
    with pytest.raises(ContainerManifestError, match="is a mesh_device SKU"):
        _with_hardware("QB2", mesh="QB2").validate_semantics()


def test_a_zero_board_multiplier_is_refused():
    """`hardware_chip_count("p150x0")` is 0, which is falsy — so `... or 1` would rewrite it
    to one chip exactly like an unknown label. Guarding only on None leaves this open."""
    with pytest.raises(ContainerManifestError, match="not a recognised board label"):
        _with_hardware("p150x0").validate_semantics()


@pytest.mark.parametrize("label", ["p150", "p150x4", "p300x2", "n300x2", "P150X4", "p150a"])
def test_recognised_hardware_labels_still_load(label):
    """The refusal must not narrow what a correct manifest may say: revision letters,
    case and the bare board are all still valid."""
    mesh = {1: "P150", 2: "P150x2", 4: "P150x4"}[hardware_chip_count(label)]
    _with_hardware(label, mesh=mesh).validate_semantics()


@pytest.mark.parametrize("field", ["max_num_seqs", "block_size", "mesh_device", "hardware"])
def test_required_serve_fields_are_required_after_merge(field):
    profile = {
        "name": "p",
        "hardware": "p150x4",
        "mesh_device": "P150x4",
        "max_num_seqs": 8,
        "block_size": 64,
    }
    del profile[field]
    with pytest.raises(ContainerManifestError, match=f"{field} is required"):
        _mani(serve={}, serve_profiles=[profile]).validate_semantics()


def test_multiple_profiles_require_an_explicit_default():
    """The author decides the default, not the consumer's luck."""
    with pytest.raises(ContainerManifestError, match="must name a default_profile"):
        _mani(
            serve_profiles=[
                {"name": "a", "hardware": "p150x2", "mesh_device": "P150x2", "max_num_seqs": 8},
                {"name": "b", "hardware": "p150x4", "mesh_device": "P150x4", "max_num_seqs": 8},
            ]
        ).validate_semantics()


def test_default_profile_naming_no_profile_is_refused():
    with pytest.raises(ContainerManifestError, match="names no profile"):
        _mani(default_profile="nope").validate_semantics()


def test_duplicate_profile_names_are_refused():
    p = {"name": "a", "hardware": "p150x4", "mesh_device": "P150x4", "max_num_seqs": 8}
    with pytest.raises(ContainerManifestError, match="duplicate serve profile"):
        _mani(serve_profiles=[p, dict(p)], default_profile="a").validate_semantics()


def test_absolute_and_escaping_code_paths_are_refused():
    for bad in ("/etc/passwd", "../../secrets"):
        with pytest.raises(Exception, match="relative to the tt-metal tree"):
            _mani(source={**BASE["source"], "code": [bad]})


def test_unknown_top_level_key_is_refused():
    """extra="forbid": a typo'd key must not be silently ignored."""
    with pytest.raises(Exception, match="serve_profile"):
        _mani(serve_profile=[])


def test_missing_code_paths_are_an_error_not_a_silent_skip(tmp_path):
    (tmp_path / "models").mkdir()
    m = _mani(
        source={
            "tt_metal": str(tmp_path),
            "code": ["models", "models/gone"],
            "ubuntu": "22.04",
            "python": "3.12",
        }
    )
    with pytest.raises(ContainerManifestError, match="models/gone"):
        m.validate_sources_exist()


def test_bad_yaml_and_non_mapping_are_diagnosed(tmp_path):
    p = tmp_path / "tt-model.yaml"
    p.write_text("just a string")
    with pytest.raises(ContainerManifestError, match="does not contain a YAML mapping"):
        load_container_manifest(p)
    p.write_text("a: [1,\n")
    with pytest.raises(ContainerManifestError, match="not valid YAML"):
        load_container_manifest(p)


def test_missing_file_is_diagnosed(tmp_path):
    with pytest.raises(ContainerManifestError, match="manifest not found"):
        load_container_manifest(tmp_path / "nope.yaml")


# --------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    "label,chips",
    [("p150", 1), ("p150x4", 4), ("p300", 2), ("p300x2", 4), ("n300", 2), ("wibble", None)],
)
def test_hardware_chip_count(label, chips):
    assert hardware_chip_count(label) == chips


def test_parse_mesh_device_accepts_presets_and_tuples():
    assert parse_mesh_device("P150x4") == (1, 4)
    assert parse_mesh_device("(2, 4)") == (2, 4)


# --------------------------------------------------------------- rendering to the wire


def test_to_wire_renders_a_v5_1_manifest_that_round_trips_as_json():
    wire = _mani().to_wire(
        image_tag="tt-model/my-model:abc123",
        tt_metal_version="0.72.1",
        tt_kernel_version="0.1.0",
        built={"tt_metal": {"sha": "abc123"}},
    )
    assert wire.schema_version == "5.1"
    assert wire.is_container and not wire.is_self_contained

    # the published document is JSON, and must survive the schema gate
    back = Manifest.from_json(wire.to_json())
    assert back.container is not None
    assert back.container.image.tag == "tt-model/my-model:abc123"
    assert back.container.resolve_profile().name == "p150x4"
    assert back.weights.repo_id == "org/Weights-7B"


def test_to_wire_derives_device_count_from_the_default_profile():
    assert _mani().to_wire(
        image_tag="t:1", tt_metal_version="v", tt_kernel_version="0.1.0"
    ).device_count == 4


def test_hf_is_the_default_registry_and_has_no_pull_ref():
    img = _mani().to_wire(
        image_tag="tt-model/my-model:abc", tt_metal_version="v", tt_kernel_version="0.1.0"
    ).container.image
    assert img.is_hub_hosted and img.pull_ref is None


def test_a_real_registry_produces_a_docker_pull_ref():
    img = _mani(image={"registry": "ghcr.io/tenstorrent"}).to_wire(
        image_tag="tt-model/my-model:abc", tt_metal_version="v", tt_kernel_version="0.1.0"
    ).container.image
    assert not img.is_hub_hosted
    assert img.pull_ref == "ghcr.io/tenstorrent/my-model:abc"


def test_a_registry_digest_pins_the_pull_ref():
    m = _mani(image={"registry": "ghcr.io/tenstorrent", "repository": "laguna"})
    img = m.to_wire(
        image_tag="t:abc",
        tt_metal_version="v",
        tt_kernel_version="0.1.0",
        digest="sha256:" + "ab" * 32,
    ).container.image
    assert img.pull_ref == "ghcr.io/tenstorrent/laguna@sha256:" + "ab" * 32


def test_the_shipped_example_manifest_is_valid(tmp_path):
    """examples/container-example.yaml is documentation; keep it loadable."""
    from pathlib import Path

    m = load_container_manifest(Path(__file__).parent.parent / "examples" / "container-example.yaml")
    assert m.profile_names() == ["p150x2", "p150x4"]
    assert m.resolve_profile().max_num_seqs == 32


# --------------------------------------------------------------- capabilities


def test_capabilities_declared_under_serve_reach_every_profile():
    """Tool/reasoning parsers are model facts, so they are normally declared once under
    `serve:` and inherited — that is the answer to "where do tool-call flags go?"."""
    m = _mani(
        serve={
            "block_size": 64,
            "capabilities": {"tool_parser": "hermes", "reasoning_parser": "deepseek_r1"},
        }
    )
    caps = m.resolve_profile("p150x4").capabilities
    assert caps.tool_parser == "hermes"
    assert caps.reasoning_parser == "deepseek_r1"


def test_a_profile_can_override_one_capability_without_dropping_the_other():
    """capabilities is a dict, so _deep_merge recurses into it rather than replacing it."""
    m = _mani(
        serve={
            "block_size": 64,
            "capabilities": {"tool_parser": "hermes", "reasoning_parser": "deepseek_r1"},
        },
        serve_profiles=[
            {
                "name": "p150x4",
                "hardware": "p150x4",
                "mesh_device": "P150x4",
                "max_num_seqs": 8,
                "capabilities": {"tool_parser": "qwen3_coder"},
            }
        ],
    )
    caps = m.resolve_profile("p150x4").capabilities
    assert caps.tool_parser == "qwen3_coder"
    assert caps.reasoning_parser == "deepseek_r1"


def test_capabilities_survive_the_json_round_trip():
    m = _mani(serve={"block_size": 64, "capabilities": {"tool_parser": "hermes"}})
    wire = m.to_wire(image_tag="t:1", tt_metal_version="v", tt_kernel_version="0.1.0")
    back = Manifest.from_json(wire.to_json())
    assert back.container.resolve_profile().capabilities.tool_parser == "hermes"


def test_an_unknown_capability_key_is_refused():
    with pytest.raises(Exception):
        _mani(serve={"block_size": 64, "capabilities": {"tool_praser": "hermes"}})


# --------------------------------------------------------------- schema version


def test_the_authored_schema_matches_the_published_one():
    """One version number for one format — not an authoring counter plus a wire counter."""
    from tt_kernel.container_manifest import SCHEMA_VERSION
    from tt_kernel.manifest import CONTAINER_SCHEMA

    assert SCHEMA_VERSION == CONTAINER_SCHEMA == "5.1"


def test_a_wrong_schema_version_is_refused():
    with pytest.raises(ContainerManifestError, match="unsupported manifest schema"):
        _mani(schema="1").validate_semantics()


def test_v5_1_is_accepted_by_the_wire_schema_gate():
    """v5 (fat), v5.1 (container) and v6 (thin) are all readable; anything else is refused
    outright rather than half-read. "6" is a REAL schema — the thin bundle — which is why
    the container path took 5.1 and left the whole number free."""
    from tt_kernel.manifest import SUPPORTED_SCHEMAS

    assert {"5", "5.1", "6"} <= SUPPORTED_SCHEMAS

    wire = _mani().to_wire(image_tag="t:1", tt_metal_version="v", tt_kernel_version="0.1.0")
    raw = json.loads(wire.to_json())
    raw["schema_version"] = "7"
    with pytest.raises(ValueError, match="Unsupported bundle schema_version"):
        Manifest.from_json(json.dumps(raw))


# ------------------------------------------------- the v5 idiom: no profiles at all
#
# v4/v5 had no profiles: one bundle was one configuration, and authors encoded the target
# in the repo NAME ("you/mymodel-blackholex1"). That idiom must remain first-class here —
# an author who does not want profiles should not have to learn what one is.

FLAT = {
    **{k: v for k, v in BASE.items() if k != "serve_profiles"},
    "repo": "you/my-model-blackholex4",
    "serve": {
        "port": 8000,
        "block_size": 64,
        "hardware": "p150x4",
        "mesh_device": "P150x4",
        "max_num_seqs": 32,
    },
}


def test_a_manifest_with_no_serve_profiles_is_valid(tmp_path):
    m = load_container_manifest(_write(tmp_path, FLAT))
    assert m.serve_profiles == []           # the author declared none...
    assert m.profile_names() == ["default"]  # ...and gets exactly one, synthesized


def test_the_synthesized_profile_carries_the_whole_serve_block():
    m = ContainerManifest.model_validate(FLAT)
    p = m.resolve_profile()
    assert (p.name, p.hardware, p.mesh_device) == ("default", "p150x4", "P150x4")
    assert (p.max_num_seqs, p.block_size, p.port) == (32, 64, 8000)


def test_a_flat_manifest_still_enforces_the_required_launch_fields():
    """Omitting profiles must not become a way to skip validation."""
    raw = json.loads(json.dumps(FLAT))
    del raw["serve"]["max_num_seqs"]
    with pytest.raises(ContainerManifestError, match="max_num_seqs is required"):
        ContainerManifest.model_validate(raw).validate_semantics()


def test_a_flat_manifest_still_cross_checks_mesh_against_hardware():
    raw = json.loads(json.dumps(FLAT))
    raw["serve"]["mesh_device"] = "P150x2"
    with pytest.raises(ContainerManifestError, match="implies 4"):
        ContainerManifest.model_validate(raw).validate_semantics()


def test_a_flat_manifest_needs_no_default_profile():
    ContainerManifest.model_validate(FLAT).validate_semantics()  # does not raise


def test_flat_and_single_profile_manifests_produce_the_SAME_wire_document():
    """The two authoring styles are one format. A consumer cannot tell them apart."""
    explicit = json.loads(json.dumps(FLAT))
    explicit["serve"] = {"port": 8000, "block_size": 64}
    explicit["serve_profiles"] = [
        {
            "name": "default",
            "hardware": "p150x4",
            "mesh_device": "P150x4",
            "max_num_seqs": 32,
        }
    ]

    kw = dict(image_tag="t:1", tt_metal_version="v", tt_kernel_version="0.1.0",
              hostname="h", created_at="2026-01-01T00:00:00+00:00")
    a = ContainerManifest.model_validate(FLAT).to_wire(**kw)
    b = ContainerManifest.model_validate(explicit).to_wire(**kw)
    assert a.container.resolve_profile().model_dump() == b.container.resolve_profile().model_dump()


def test_a_flat_manifest_publishes_a_profile_so_consumers_see_one_shape():
    """`list` and `--profile` must work identically no matter how it was authored."""
    wire = ContainerManifest.model_validate(FLAT).to_wire(
        image_tag="t:1", tt_metal_version="v", tt_kernel_version="0.1.0"
    )
    back = Manifest.from_json(wire.to_json())
    assert back.container.profile_names() == ["default"]
    assert back.container.resolve_profile().max_num_seqs == 32
    assert back.device_count == 4  # still derived from the hardware label


def test_a_hand_edited_package_with_no_profiles_fails_loudly_not_with_an_IndexError():
    from tt_kernel.manifest import ContainerSpec, ImageRef

    spec = ContainerSpec(image=ImageRef(tag="t:1"))
    with pytest.raises(ValueError, match="declares no serve profiles"):
        spec.resolve_profile()


# ------------------------------------------------------------------ the example is the docs
#
# examples/container-example.yaml is the ONLY reference for what a manifest may contain.
# It had silently fallen behind — omitting serve.env and serve.additional_config, both of
# which a real model needs — so a reader would have concluded they do not exist.


def _example_text() -> str:
    from pathlib import Path

    return (Path(__file__).parent.parent / "examples" / "container-example.yaml").read_text()


def test_the_example_mentions_every_manifest_field():
    from tt_kernel.container_manifest import ContainerManifest, ImageSettings, Source
    from tt_kernel.manifest import ServeProfile, ServeSettings

    text = _example_text()
    fields = set()
    for model in (ContainerManifest, Source, ImageSettings, ServeSettings, ServeProfile):
        fields |= set(model.model_fields)
    # aliased / internal names that never appear verbatim in a manifest
    fields -= {"schema_version"}
    missing = sorted(f for f in fields if f not in text)
    assert not missing, f"example does not mention: {missing}"


def test_the_example_mentions_every_runtime_key():
    from tt_kernel.launchers import KINDS

    text = _example_text()
    keys = {k for launcher in KINDS.values() for k in launcher.RUNTIME_KEYS}
    missing = sorted(k for k in keys if k not in text)
    assert not missing, f"example does not mention runtime keys: {missing}"


def test_the_example_mentions_every_kind():
    from tt_kernel.launchers import KINDS

    text = _example_text()
    assert not [k for k in KINDS if k not in text]
