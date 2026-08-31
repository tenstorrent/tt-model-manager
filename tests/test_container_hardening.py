# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Hardening for the v5.1 container path — the should-fix items from PR #37's review:

1. manifest ``name`` is a filesystem path + container name → must be a safe slug (it drives
   an rmtree), enforced at authoring AND defensively where the path is derived.
2. serve-field typos must fail at load (``ServeSettings`` forbids unknown keys).
3. manifest-sourced values interpolated into the generated build shell must be shlex-quoted.
4. ``wait_ready`` must honour its timeout even while the container is silent (no hang).

...and from PR #50's review, which added the ``tt-dit-server`` kind:

5. ``model_weight_cache_dir`` must apply the same ``_safe_name`` guard as its sibling — it is
   mkdir'd and bind-mounted, and both caches must share one parent so ``rm`` catches both.
"""

import json
import time

import pytest
from pydantic import ValidationError

from tt_kernel import container
from tt_kernel.container import ContainerError
from tt_kernel.container_manifest import ContainerManifest, ContainerManifestError
from tt_kernel.launchers import launcher_for
from tt_kernel.manifest import Manifest, Producer

_BASE = {
    "schema": "5.1", "repo": "you/my-model", "name": "my-model",
    "weights": "org/Weights-7B", "kind": "vllm-plugin", "arch": "blackhole",
    "source": {"tt_metal": "/tmp/tt-metal", "code": ["models/common"],
               "ubuntu": "22.04", "python": "3.12"},
    "runtime": {"vllm": {"version": "0.24.0"},
                "plugin": {"repo": "https://github.com/tenstorrent/vllm-tt-plugin", "ref": "bc4af2d5"}},
    "serve": {"port": 8000, "block_size": 64},
    "serve_profiles": [{"name": "p150x4", "hardware": "p150x4", "mesh_device": "P150x4",
                        "max_num_seqs": 32, "max_model_len": 131072}],
}


def _mani(**over):
    raw = json.loads(json.dumps(_BASE))
    raw.update(over)
    return ContainerManifest.model_validate(raw)


# ---- 1. name is a safe slug -------------------------------------------------------------

@pytest.mark.parametrize("good", ["my-model", "qwen3-coder-30b-a3b", "m", "a.b_c-1"])
def test_valid_names_load(good):
    assert _mani(name=good).name == good


@pytest.mark.parametrize("bad", ["../../etc", "a/b", "..", "/abs", "Up", "", " x", "a b"])
def test_hostile_names_are_rejected_at_authoring(bad):
    with pytest.raises((ContainerManifestError, ValidationError)) as e:
        _mani(name=bad)
    assert "slug" in str(e.value).lower() or "name" in str(e.value).lower()


def _wire_manifest(name):
    # A wire manifest reaches Manifest directly (a hand-crafted pulled JSON bypasses the
    # authoring validator), so the path-deriving code must guard itself.
    return Manifest(schema_version="5.1", name=name, tt_metal_version="x", arch="blackhole",
                    producer=Producer(tt_kernel_version="0", created_at="t"))


def test_model_cache_dir_refuses_a_traversal_name():
    # This is what remove_container rmtree's — it must never escape the cache root.
    with pytest.raises(ContainerError):
        container.model_cache_dir(_wire_manifest("../../../../tmp/pwned"))


def test_model_cache_dir_accepts_a_safe_name():
    p = container.model_cache_dir(_wire_manifest("my-model"))
    assert p.name == "cache" and p.parent.name == "my-model"


def test_model_weight_cache_dir_refuses_a_traversal_name():
    # Parity with its sibling above. This one is never rmtree'd directly, but
    # ensure_mount_sources mkdirs it with parents=True and compose_run bind-mounts it, so an
    # unguarded name creates and mounts an arbitrary host directory instead.
    with pytest.raises(ContainerError):
        container.model_weight_cache_dir(_wire_manifest("../../../../tmp/pwned"))


def test_model_weight_cache_dir_accepts_a_safe_name():
    p = container.model_weight_cache_dir(_wire_manifest("my-model"))
    assert p.name == "weights" and p.parent.name == "my-model"


def test_both_caches_share_one_guarded_parent():
    """``remove_container`` drops the parent to remove both caches at once, so they must
    agree on it — a divergence there is how the 105 GB weight tree gets orphaned."""
    m = _wire_manifest("my-model")
    assert container.model_cache_dir(m).parent == container.model_weight_cache_dir(m).parent


# ---- 2. serve-field typos fail at load --------------------------------------------------

def test_serve_typo_is_rejected():
    with pytest.raises((ContainerManifestError, ValidationError)):
        _mani(serve={"port": 8000, "block_size": 64, "server_timeoutt": 99})


def test_known_serve_fields_still_accepted():
    m = _mani(serve={"port": 8000, "block_size": 64, "server_timeout": 99, "max_num_seqs": 32})
    assert m.serve.server_timeout == 99


# ---- 3. manifest values are shlex-quoted in the generated build shell -------------------

def test_vllm_version_is_shlex_quoted_in_install_lines():
    evil = "0.24.0; touch /tmp/pwned"
    m = _mani(runtime={"vllm": {"version": evil},
                       "plugin": {"version": "0.1.0"}})
    joined = "\n".join(launcher_for(m.kind).install_lines(m))
    assert f"vllm=={evil}" not in joined            # not injected raw
    assert "'0.24.0; touch /tmp/pwned'" in joined   # present, quoted


def test_plugin_version_and_extension_are_shlex_quoted():
    m = _mani(runtime={"vllm": {"version": "0.24.0"},
                       "plugin": {"version": "1.0; id"},
                       "extension": "ext; rm -rf /"})
    joined = "\n".join(launcher_for(m.kind).install_lines(m))
    assert "vllm-tt-plugin==1.0; id" not in joined
    assert "'1.0; id'" in joined
    assert "/opt/tt-metal/ext; rm -rf /" not in joined
    assert "'ext; rm -rf /'" in joined


# ---- 4. wait_ready is time-bounded even when the container is silent --------------------

class _BlockingStdout:
    """An iterator that never yields — models a booted-but-hung container (no output)."""
    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(3600)  # pragma: no cover - only the reader thread touches this


class _FakeProc:
    def __init__(self):
        self.stdout = _BlockingStdout()
        self.terminated = False

    def terminate(self):
        self.terminated = True


def test_wait_ready_honours_timeout_on_a_silent_container(monkeypatch):
    monkeypatch.setattr(container.subprocess, "Popen", lambda *a, **k: _FakeProc())
    t0 = time.monotonic()
    r = container.wait_ready("tt-model-x", "APP READY", timeout_s=1, echo=lambda *_: None)
    elapsed = time.monotonic() - t0
    assert r.ready is False
    # A silent container is still RUNNING — the distinction the caller needs to tell
    # "slow boot" from "crashed".
    assert r.exited is False
    assert elapsed < 5, f"wait_ready hung past its timeout ({elapsed:.1f}s)"
