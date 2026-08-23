# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The golden-string tests — the highest-value pair in the suite.

Each model type's serve_argv must reproduce, from the example manifest alone, the
launch command that actually served that model on real hardware:

- laguna:  models/autoports/poolside_laguna_xs_2_1/serve_vllm.sh   (type: vllm)
- ornith:  the Ornith-1.0-35B p150x4 quickstart                    (type: vllm-legacy)

These are what stop the recipes drifting. If one of these fails, either the manifest
example or the type's renderer no longer matches a validated deployment — figure out
which BEFORE "fixing" the test.
"""

from tt_model.types import TYPES

# serve_vllm.sh, verbatim modulo argv ordering (canonical order: model, core limits,
# additional-config, author args, port):
LAGUNA_GOLDEN = [
    "vllm", "serve", "poolside/Laguna-XS-2.1",
    "--max-model-len", "131072",
    "--max-num-seqs", "8",
    "--block-size", "64",
    "--additional-config",
    '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 1500000000,'
    ' "fabric_config": "FABRIC_1D_RING"}}',
    "--trust-remote-code",
    "--enable-prefix-caching",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "poolside_v1",
    "--reasoning-parser", "poolside_v1",
    "--port", "8000",
]

ORNITH_GOLDEN_LATENCY = [
    "python", "-m", "models.common.readiness_check.run_vllm_server",
    "--stages", "serve",
    "--model-dir", "models/autoports/ornith_ai_ornith_1_0_35b",
    "--hf-model", "ornith-ai/Ornith-1.0-35B",
    "--mesh-device", "(1, 4)",
    "--max-num-seqs", "1",
    "--max-model-len", "262144",
    "--block-size", "64",
    "--server-timeout", "2400",
    "--port", "8100",
    "--tt-config",
    '{"trace_region_size": 200000000, "l1_small_size": 24576,'
    ' "fabric_config": "FABRIC_1D_RING", "fabric_router_max_packet_bytes": 8192}',
    "--additional-server-args",
    "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml",
]


def test_laguna_serve_argv_is_the_validated_recipe(laguna):
    t = TYPES[laguna.type]
    assert t.serve_argv(laguna, laguna.resolve_profile()) == LAGUNA_GOLDEN


def test_laguna_serve_env_is_the_validated_env(laguna):
    t = TYPES[laguna.type]
    env = t.serve_env(laguna, laguna.resolve_profile())
    assert env["MESH_DEVICE"] == "P150x4"
    assert env["HF_MODEL"] == "poolside/Laguna-XS-2.1"     # adapter reads env, not --model
    assert env["TT_LAGUNA_PIPE_CHUNK"] == "2048"
    assert "VLLM_PLUGINS" not in env                       # allow-list; must stay unset


def test_ornith_latency_profile_is_the_quickstart_command(ornith):
    t = TYPES[ornith.type]
    argv = t.serve_argv(ornith, ornith.resolve_profile("p150x4-latency"))
    assert argv == ORNITH_GOLDEN_LATENCY


def test_ornith_capacity_profile_differs_by_exactly_one_flag(ornith):
    """--max-num-seqs is the deployment decision; nothing else may drift with it."""
    t = TYPES[ornith.type]
    lat = t.serve_argv(ornith, ornith.resolve_profile("p150x4-latency"))
    cap = t.serve_argv(ornith, ornith.resolve_profile("p150x4-capacity"))
    diff = [(a, b) for a, b in zip(lat, cap) if a != b]
    assert diff == [("1", "32")]
    assert len(lat) == len(cap)


def test_laguna_p150x2_profile_changes_only_what_it_declares(laguna):
    t = TYPES[laguna.type]
    p4 = t.serve_argv(laguna, laguna.resolve_profile("p150x4"))
    p2 = t.serve_argv(laguna, laguna.resolve_profile("p150x2"))
    # context shrinks, trace region + fabric change; args/port stay
    assert "65536" in p2 and "131072" in p4
    assert '"FABRIC_1D"' in " ".join(p2)
    assert p2[-2:] == p4[-2:] == ["--port", "8000"]
