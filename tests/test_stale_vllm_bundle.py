# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Issue #13: a cached v4 vLLM bundle pins its launch command, so a republished model must not be
served with stale parameters. `_ensure_vllm_pulled` re-pulls when the source revision diverged (or
when --refresh is passed), reuses when up to date / pinned / offline. No hardware, no network —
the HF download and the install are stubbed.
"""

from pathlib import Path

from tt_kernel import cli, localdb
from tt_kernel.manifest import Manifest, Producer, RunnerPayload, WeightsRef


def _v4_vllm_manifest_json():
    m = Manifest(
        schema_version="4", name="m", tt_metal_version="0.75.0", arch="blackhole",
        producer=Producer(tt_kernel_version="0", created_at="t"),
        runner=RunnerPayload(backend="vllm", bundle_dir="vllm_bundle"),
        weights=WeightsRef(repo="org/model"),
    )
    return m.to_json()


def _seed_installed(tmp_path, *, revision, pinned=False):
    bp = tmp_path / "bundles" / "acme__laguna"
    bp.mkdir(parents=True, exist_ok=True)
    localdb.record("acme/laguna", {
        "repo_id": "acme/laguna", "bundle_path": str(bp),
        "revision": revision, "pinned": pinned,
    })
    return bp


def _wire(monkeypatch, tmp_path, *, latest):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))  # isolate localdb
    calls = {"downloads": 0, "installs": 0, "last_download_rev": None}

    def _dl(repo_id, revision, dest):
        d = Path(dest) / "snap"
        d.mkdir(parents=True)
        (d / cli.MANIFEST_NAME).write_text(_v4_vllm_manifest_json())
        calls["downloads"] += 1
        calls["last_download_rev"] = revision
        return d

    def _install(repo_id, snapshot, manifest, **kw):
        calls["installs"] += 1
        bp = tmp_path / "bundles" / repo_id.replace("/", "__")
        bp.mkdir(parents=True, exist_ok=True)
        localdb.record(repo_id, {
            "repo_id": repo_id, "bundle_path": str(bp),
            "revision": kw.get("resolved_revision"), "pinned": kw.get("revision") is not None,
        })

    monkeypatch.setattr(cli.hub, "download_bundle", _dl)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: latest)
    monkeypatch.setattr(cli, "_install_vllm_bundle", _install)
    return calls


def test_up_to_date_bundle_is_reused(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path, latest="rev1")
    _seed_installed(tmp_path, revision="rev1")
    cli._ensure_vllm_pulled("acme/laguna", None, arch=None, bundles_dir=None)
    assert calls["installs"] == 0 and calls["downloads"] == 0  # reused, no re-pull


def test_diverged_bundle_is_repulled(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path, latest="rev2")
    _seed_installed(tmp_path, revision="rev1")
    cli._ensure_vllm_pulled("acme/laguna", None, arch=None, bundles_dir=None)
    assert calls["installs"] == 1 and calls["downloads"] == 1  # re-pulled the new revision
    assert calls["last_download_rev"] == "rev2"                 # fetched the resolved sha
    assert localdb.get("acme/laguna")["revision"] == "rev2"


def test_refresh_forces_repull_even_when_up_to_date(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path, latest="rev1")
    _seed_installed(tmp_path, revision="rev1")
    cli._ensure_vllm_pulled("acme/laguna", None, arch=None, bundles_dir=None, refresh=True)
    assert calls["installs"] == 1  # --refresh re-pulls regardless


def test_pinned_bundle_is_not_repulled_on_divergence(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path, latest="rev2")
    _seed_installed(tmp_path, revision="rev1", pinned=True)
    cli._ensure_vllm_pulled("acme/laguna", None, arch=None, bundles_dir=None)
    assert calls["installs"] == 0  # a deliberately pinned @revision is left alone


def test_unreachable_hub_reuses_the_cache(monkeypatch, tmp_path):
    calls = _wire(monkeypatch, tmp_path, latest=None)  # latest_revision -> None (offline/timeout)
    _seed_installed(tmp_path, revision="rev1")
    cli._ensure_vllm_pulled("acme/laguna", None, arch=None, bundles_dir=None)
    assert calls["installs"] == 0  # can't tell -> never wipe a working install
