# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``kind: cohost``: several members, one container, one port per member."""

import json

import pytest

from tt_kernel import container
from tt_kernel.cohost import config as cohost_config
from tt_kernel.launchers import CohostLauncher, LauncherError, launcher_for
from tt_kernel.manifest import Manifest

from test_container_cli import _wire

MEMBERS = [
    {"name": "gptoss", "kind": "vllm", "model": "openai/gpt-oss-20b", "port_offset": 0,
     "env": {"GPT_OSS_20B_KV_POOL_BLOCKS": "4128"}},
    {"name": "llama8b", "kind": "vllm", "model": "meta-llama/Llama-3.1-8B-Instruct", "port_offset": 1,
     "kv_tokens": 65536, "args": ["--max-model-len", "8192"], "preload": "models.cohost_llm.fit:distributed_prefill_norm"},
    {"name": "qwen32b", "kind": "vllm", "model": "Qwen/Qwen3-32B", "port_offset": 2},
]


@pytest.fixture
def manifest(tmp_path):
    raw = json.loads(_wire(tmp_path).read_text())
    raw["container"]["kind"] = "cohost"
    raw["container"]["runtime"] = {
        "members": MEMBERS,
        "profiles": {"gptoss-llama": ["gptoss", "llama8b"], "gptoss-qwen": ["gptoss", "qwen32b"]},
        "tt_config": {"fabric_config": "FABRIC_1D_RING"},
    }
    raw["container"]["serve_profiles"] = [
        {"name": n, "hardware": "p300x2", "mesh_device": "P150x4", "port": 21000}
        for n in ("gptoss-llama", "gptoss-qwen", "all")]
    raw["container"]["default_profile"] = "gptoss-llama"
    return Manifest.from_json(json.dumps(raw))


def test_cohost_is_a_registered_kind():
    assert isinstance(launcher_for("cohost"), CohostLauncher)


def test_a_profile_serves_its_members_each_on_its_own_port(manifest):
    profile = manifest.container.resolve_profile("gptoss-llama")
    assert container.published_ports(manifest, profile) == [21000, 21001, 21002]  # + status


def test_the_config_reaches_the_runtime_inline_and_parses(manifest):
    profile = manifest.container.resolve_profile("gptoss-qwen")
    env = launcher_for("cohost").serve_env(manifest, profile)
    cfg = cohost_config.loads(env["TT_COHOST_CONFIG"])
    assert [(m.name, m.port) for m in cfg.members] == [("gptoss", 21000), ("qwen32b", 21002)]
    assert cfg.mesh_device == "P150x4" and cfg.status_port == 21003
    assert cfg.tt_config == {"fabric_config": "FABRIC_1D_RING"}
    assert cfg.members[0].env == {"GPT_OSS_20B_KV_POOL_BLOCKS": "4128"}


def test_a_member_preload_hook_reaches_the_runtime(manifest):
    profile = manifest.container.resolve_profile("gptoss-llama")
    cfg = cohost_config.loads(launcher_for("cohost").serve_env(manifest, profile)["TT_COHOST_CONFIG"])
    assert [m.preload for m in cfg.members] == ["", "models.cohost_llm.fit:distributed_prefill_norm"]


def test_a_profile_not_in_the_map_serves_every_member(manifest):
    profile = manifest.container.resolve_profile("all")
    assert len(launcher_for("cohost").members(manifest.container.runtime, profile.name)) == 3


def test_compose_run_publishes_every_member_port(manifest, tmp_path):
    profile = manifest.container.resolve_profile("gptoss-llama")
    argv = container.compose_run(manifest, profile, ["python", "-m", "tt_kernel.cohost.launch"], {},
                                 device_ids=[0, 1, 2, 3], hf_home_dir=tmp_path, cache_dir=tmp_path,
                                 weight_cache_dir=tmp_path, tensor_cache_dir=tmp_path)
    published = [argv[i + 1] for i, a in enumerate(argv) if a == "--publish"]
    assert published == ["21000:21000", "21001:21001", "21002:21002"]


@pytest.mark.parametrize("runtime, match", [
    ({}, "non-empty"),
    ({"members": [{"name": "a", "kind": "llm"}]}, "kind vllm"),
    ({"members": [{"name": "a", "kind": "vllm"}]}, "needs model"),
    ({"members": [{"name": "a", "kind": "vllm", "model": "m"},
                  {"name": "b", "kind": "vllm", "model": "m"}]}, "distinct"),
    ({"members": [{"name": "a", "kind": "vllm", "model": "m", "port_offset": -1}]}, ">= 0"),
    ({"members": [{"name": "a", "kind": "vllm", "model": "m"}], "gpus": 1}, "not valid"),
    ({"members": [{"name": "a", "kind": "vllm", "model": "m"}], "profiles": {"p": ["zz"]}}, "unknown member"),
])
def test_bad_member_lists_are_refused(runtime, match):
    with pytest.raises(LauncherError, match=match):
        CohostLauncher().members(runtime, "p")


def test_the_free_port_walk_finds_a_whole_range(monkeypatch):
    busy = {20001}
    monkeypatch.setattr(container, "port_is_free", lambda p: p not in busy)
    assert container.pick_free_port(20000, span=3) == 20002
    assert container.pick_free_port(20000) == 20000


def test_the_port_walk_covers_a_profile_whose_member_ports_leave_a_gap(manifest, monkeypatch):
    """gptoss-qwen publishes members 0 and 2 and the status port 3: four ports from the base, not three."""
    profile = manifest.container.resolve_profile("gptoss-qwen")
    assert container.port_span(manifest, profile) == 4
    busy = {20003}
    monkeypatch.setattr(container, "port_is_free", lambda p: p not in busy)
    assert container.pick_free_port(20000, span=container.port_span(manifest, profile)) == 20004
