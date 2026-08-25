# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the v5 self-contained ("fat") packaging model: the ``bundled`` block that ships the
author's ttnn/vLLM/plugin wheels inside the bundle, schema round-trip, and the compare() rule that
a self-contained bundle skips host version-range gates (it ships its own platform) while ``arch``
stays fatal. No hardware, no network.
"""

from tt_kernel import metal
from tt_kernel.manifest import (
    BundledPlatform,
    Entrypoint,
    Manifest,
    Platform,
    Producer,
    Runtime,
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


def test_self_contained_skips_host_range_gates():
    # Host has NO ttnn/vLLM and a mismatched platform range — a self-contained bundle must not
    # gate on it, because it ships its own wheels.
    m = _v5_manifest(platform=Platform(ttnn=">=0.99"), runtime=Runtime(kind="vllm", version=">=99"))
    report = compare(m, _env(tt_metal_version=None, vllm_version=None))
    assert report.compatible is True
    assert report.issues == []


def test_self_contained_arch_still_fatal():
    m = _v5_manifest()
    report = compare(m, _env(arch="wormhole_b0"))
    assert report.has_fatal is True
    assert any(i.field == "arch" and i.fatal for i in report.issues)


def test_self_contained_ignores_host_tt_metal_version():
    # The host has tt-metal installed at a DIFFERENT version than the bundle was built against.
    # A self-contained bundle ships its own engine, so this must NOT surface as an issue — even
    # when the bundle carries a kernel build_key (which for a host-provisioned bundle would gate).
    m = _v5_manifest(tt_metal_version="0.75.0", build_key=12345)
    report = compare(m, _env(tt_metal_version="0.99.0-different", build_key=None))
    assert report.compatible is True
    assert not any(i.field in ("tt_metal_version", "build_key") for i in report.issues)


def test_self_contained_device_count_still_forceable():
    # device_count still matters (it's the mesh the model needs) — forceable, not fatal.
    m = _v5_manifest(device_count=4)
    report = compare(m, _env(device_count=1))
    assert report.compatible is False
    assert report.forceable is True
    assert any(i.field == "device_count" and not i.fatal for i in report.issues)


def test_self_contained_no_runner_version_advisory():
    from tt_kernel.manifest import runner_version_advisory

    # A v5 bundle has a weights pointer, so the advisory would otherwise fire on a host with a
    # mismatched tt-metal. Self-contained => no host tt-metal advisory at all.
    m = _v5_manifest(tt_metal_version="0.75.0")
    assert runner_version_advisory(m, _env(tt_metal_version="0.99.0-different")) is None


def test_v4_without_bundled_is_not_self_contained():
    m = _v5_manifest(schema_version="4", bundled=None)
    assert m.is_self_contained is False
    assert m.is_v4 is True
