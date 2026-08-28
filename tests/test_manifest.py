# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Compatibility comparison matrix — the correctness core.

Every bundle is self-contained (v5/v6): ``compare`` gates only on ``arch`` (fatal) and
``device_count`` (forceable); nothing about the host's tt-metal/vLLM is consulted.
"""

import pytest

from tt_kernel.manifest import Manifest, Producer, WeightsRef, compare
from tt_kernel.metal import LocalEnv


def _manifest(**overrides) -> Manifest:
    base = dict(
        name="m",
        tt_metal_version="v1.0.0-abc",
        arch="blackhole",
        device_count=1,
        producer=Producer(tt_kernel_version="0.1.0", created_at="now"),
    )
    base.update(overrides)
    return Manifest(**base)


def _env(**overrides) -> LocalEnv:
    base = dict(arch="blackhole", device_count=1)
    base.update(overrides)
    return LocalEnv(**base)


def test_fully_compatible():
    report = compare(_manifest(), _env())
    assert report.compatible
    assert not report.issues


def test_arch_mismatch_is_fatal():
    report = compare(_manifest(arch="blackhole"), _env(arch="wormhole_b0"))
    assert not report.compatible
    assert report.has_fatal
    assert not report.forceable
    assert report.issues[0].field == "arch"


def test_device_count_mismatch_is_forceable():
    report = compare(_manifest(device_count=1), _env(device_count=2))
    assert report.forceable
    assert any(i.field == "device_count" for i in report.issues)


def test_unknown_local_fields_do_not_block():
    # When detection fails (arch unknown / no device), we don't fabricate mismatches.
    report = compare(_manifest(), _env(arch=None, device_count=0))
    assert report.compatible


def test_manifest_json_roundtrip():
    m = _manifest(weights=WeightsRef(repo_id="org/model", revision="main"))
    assert Manifest.from_json(m.to_json()) == m


# --------------------------------------------------------------------------- schema gate

_LEGACY_V4_JSON = """
{
  "schema_version": "4",
  "name": "legacy",
  "tt_metal_version": "v1.0.0-abc",
  "arch": "blackhole",
  "device_count": 1,
  "producer": {"tt_kernel_version": "0.1.0", "created_at": "now"}
}
"""


@pytest.mark.parametrize("schema", ["1", "3", "4"])
def test_legacy_schema_is_rejected(schema):
    # Only v5/v6 are supported now; older bundles are refused, not silently half-read.
    js = _LEGACY_V4_JSON.replace('"schema_version": "4"', f'"schema_version": "{schema}"')
    with pytest.raises(ValueError, match="schema_version"):
        Manifest.from_json(js)
