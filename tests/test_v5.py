# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the v5 self-contained ("fat") packaging model: the ``bundled`` block that ships the
author's ttnn/vLLM/plugin wheels inside the bundle, schema round-trip, and the compare() rule that
a self-contained bundle gates only on ``arch`` (fatal) + ``device_count`` (forceable) — it ships
its own platform, so no host version check applies. No hardware, no network.
"""

from tt_kernel import metal
from tt_kernel.manifest import (
    BundledPlatform,
    Entrypoint,
    Manifest,
    Producer,
    WeightsRef,
    WheelArtifact,
    compare,
)


def _wheel(**over):
    base = dict(
        path="wheels/ttnn-0.75.0-cp312-cp312-linux_x86_64.whl",
        sha256="deadbeef",
        size=193_000_000,
        python_tag="cp312",
        abi_tag="cp312",
        platform_tag="linux_x86_64",
    )
    base.update(over)
    return WheelArtifact(**base)


def _v5_manifest(**over):
    base = dict(
        schema_version="5",
        name="llama-3.2-3b-tt",
        tt_metal_version="0.75.0",
        arch="blackhole",
        device_count=1,
        producer=Producer(tt_kernel_version="0", created_at="t"),
        entrypoint=Entrypoint(**{"class": "generator_vllm:LlamaForCausalLM", "arch_name": "LlamaForCausalLM"}),
        weights=WeightsRef(repo="unsloth/Llama-3.2-3B-Instruct"),
        bundled=BundledPlatform(
            ttnn_wheel=_wheel(),
            plugin_wheel=_wheel(path="wheels/vllm_tt_plugin-0.3.0-py3-none-any.whl", platform_tag="any"),
            metal_dir="metal",
            install_script="install.sh",
            run_script="run.sh",
            requirements="requirements.txt",
        ),
    )
    base.update(over)
    return Manifest(**base)


def _env(**over):
    base = dict(arch="blackhole", device_count=1)
    base.update(over)
    return metal.LocalEnv(**base)


def test_v5_schema_roundtrip():
    m = _v5_manifest()
    assert m.is_self_contained is True
    m2 = Manifest.from_json(m.to_json())
    assert m2.schema_version == "5"
    assert m2.is_self_contained is True
    assert [w.path for w in m2.bundled.wheels] == [
        "wheels/ttnn-0.75.0-cp312-cp312-linux_x86_64.whl",
        "wheels/vllm_tt_plugin-0.3.0-py3-none-any.whl",
    ]


def test_self_contained_compatible_on_matching_env():
    # The host has NO tt-metal/vLLM — a self-contained bundle must not care (it ships its own).
    m = _v5_manifest()
    report = compare(m, _env())
    assert report.compatible is True
    assert report.issues == []


def test_self_contained_arch_still_fatal():
    m = _v5_manifest()
    report = compare(m, _env(arch="wormhole_b0"))
    assert report.has_fatal is True
    assert any(i.field == "arch" and i.fatal for i in report.issues)


def test_self_contained_device_count_still_forceable():
    # device_count still matters (it's the mesh the model needs) — forceable, not fatal.
    m = _v5_manifest(device_count=4)
    report = compare(m, _env(device_count=1))
    assert report.compatible is False
    assert report.forceable is True
    assert any(i.field == "device_count" and not i.fatal for i in report.issues)
