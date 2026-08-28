# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The ``tt-dit-server`` kind: a diffusion model behind its own HTTP app, offline.

A diffusion pipeline has no tokens, no KV cache and no continuous batching, so this kind
exists to prove the container path is not vLLM-shaped. The assertions that matter here
are the ones a wrong answer makes expensive: a serve profile that demands engine settings
the kind has no use for, an ASGI target the code allowlist never ships, and a mesh that
silently gives FLUX.2 a parallel factor of 1.
"""

import json

import pytest

from tt_kernel.container_manifest import ContainerManifest, ContainerManifestError
from tt_kernel.launchers import KINDS, launcher_for, required_serve_fields
from tt_kernel.manifest import Manifest

DIT_BASE = {
    "schema": "5.1",
    "repo": "you/my-diffusion-model",
    "name": "my-diffusion-model",
    "weights": "org/Weights",
    "kind": "tt-dit-server",
    "arch": "blackhole",
    "source": {
        "tt_metal": "/path/to/tt-metal",
        "code": ["models/common", "models/tt_dit"],
        "ubuntu": "24.04",
        "python": "3.12",
    },
    "runtime": {"app": "models.tt_dit.server.flux2.app:app"},
    "serve": {"hardware": "p300x2", "mesh_device": "QB2", "port": 8000},
}


def _manifest(**over) -> ContainerManifest:
    raw = json.loads(json.dumps(DIT_BASE))
    raw.update(over)
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    return m


def _wire(m: ContainerManifest) -> Manifest:
    return m.to_wire(
        image_tag="tt-model/my-diffusion-model:abc123",
        tt_metal_version="0.72.1",
        tt_kernel_version="0.1.0",
        hostname="h",
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_the_kind_is_registered():
    assert "tt-dit-server" in KINDS


def test_it_needs_no_engine_settings():
    """max_num_seqs and block_size configure a continuous-batching engine. This kind has
    none, so demanding them would force the author to invent meaningless numbers."""
    assert required_serve_fields("tt-dit-server") == ("hardware", "mesh_device")
    assert "max_num_seqs" in required_serve_fields("vllm-plugin")
    # ...and a manifest without them validates.
    assert _manifest().serve.max_num_seqs is None


def test_an_unknown_kind_still_reports_the_historical_fields():
    """A bad kind is reported by kind validation; it must not be masked by a confusing
    complaint about missing engine settings."""
    assert required_serve_fields("no-such-kind") == (
        "hardware",
        "mesh_device",
        "max_num_seqs",
        "block_size",
    )


def test_runtime_app_is_required():
    with pytest.raises(ContainerManifestError, match="requires runtime.app"):
        _manifest(runtime={})


def test_runtime_app_must_be_an_asgi_target():
    with pytest.raises(ContainerManifestError, match="requires runtime.app"):
        _manifest(runtime={"app": "models.tt_dit.server.flux2.app"})


def test_an_app_the_allowlist_does_not_ship_is_refused():
    """source.code promises EXACTLY what ships. An app outside it would import fine on the
    author's machine and vanish inside the image."""
    with pytest.raises(ContainerManifestError, match="no source.code entry"):
        _manifest(
            runtime={"app": "somewhere.else.app:app"},
            source={
                "tt_metal": "/path/to/tt-metal",
                "code": ["models/tt_dit"],
                "ubuntu": "24.04",
                "python": "3.12",
            },
        )


def test_unknown_runtime_keys_are_refused():
    with pytest.raises(ContainerManifestError, match="not valid for kind"):
        _manifest(runtime={"app": "models.tt_dit.server.flux2.app:app", "vllm": {}})


def test_serve_argv_launches_the_declared_app():
    m = _manifest()
    wire = _wire(m)
    argv = launcher_for(m.kind).serve_argv(wire, wire.container.resolve_profile())
    assert argv == [
        "python",
        "-m",
        "uvicorn",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--lifespan",
        "on",
        "models.tt_dit.server.flux2.app:app",
    ]


def test_serve_env_resolves_the_mesh_sku_to_a_shape():
    """The manifest names a SKU; the servers take a shape. Resolving it here is what keeps
    the two from drifting apart."""
    m = _manifest()
    wire = _wire(m)
    env = launcher_for(m.kind).serve_env(wire, wire.container.resolve_profile())
    assert env["MESH_DEVICE"] == "QB2"
    assert env["FLUX2_MESH_SHAPE"] == "2x2"
    assert env["HF_MODEL"] == "org/Weights"


def test_qb2_is_a_square_not_a_line():
    """FLUX.2 needs sequence AND tensor parallel factors above 1. P300x2 is the same four
    chips as a (1, 4) line, on which one factor is 1 and attention fails deep inside."""
    from tt_kernel.container_manifest import MESH_DEVICE_PRESETS, parse_mesh_device

    assert parse_mesh_device("QB2") == (2, 2)
    assert MESH_DEVICE_PRESETS["P300x2"] == (1, 4)


def test_qb2_agrees_with_the_hardware_chip_count():
    """A mesh whose chip count disagrees with the hardware label is refused, so the new
    preset has to line up with p300x2's four chips or every profile using it breaks."""
    from tt_kernel.container_manifest import hardware_chip_count, parse_mesh_device

    rows, cols = parse_mesh_device("QB2")
    assert rows * cols == hardware_chip_count("p300x2") == 4


def test_verify_lines_resolve_the_app_itself():
    """Importing the module is not enough: registration is lazy, so the attribute uvicorn
    will look up has to be resolved at build time, on the author's machine."""
    lines = launcher_for("tt-dit-server").verify_lines(_manifest())
    # the lines are shell-quoted for the Dockerfile, so match on the payload, not quoting
    joined = "\n".join(lines).replace("'\"'\"'", "'")
    assert "importlib.import_module('models.tt_dit.server.flux2.app')" in joined
    assert "hasattr(mod, 'app')" in joined


def test_install_lines_do_not_install_vllm():
    lines = "\n".join(launcher_for("tt-dit-server").install_lines(_manifest()))
    assert "vllm" not in lines
    assert "fastapi" in lines and "uvicorn" in lines


def test_a_lock_replaces_resolution_entirely():
    lines = launcher_for("tt-dit-server").install_lines(_manifest(runtime={"app": "models.tt_dit.x:app", "lock": True}))
    assert len(lines) == 1
    assert "requirements.lock" in lines[0]
    assert "fastapi" not in lines[0]


def test_torch_is_installed_explicitly(monkeypatch):
    """Nothing in an HTTP stack depends on torch, and diffusers/transformers declare it
    optional — so unlike the vLLM kinds nothing pulls it in. The image built without it
    and failed verify, so the launcher adds it at tt-metal's own pin."""
    from tt_kernel import launchers

    monkeypatch.setattr(launchers, "metal_torch_pin", lambda _tree: "2.11.0")
    line = launcher_for("tt-dit-server").install_lines(_manifest())[0]
    assert "torch==2.11.0" in line


def test_an_author_supplied_torch_wins(monkeypatch):
    from tt_kernel import launchers

    monkeypatch.setattr(launchers, "metal_torch_pin", lambda _tree: "2.11.0")
    line = launcher_for("tt-dit-server").install_lines(
        _manifest(runtime={"app": "models.tt_dit.x:app", "packages": ["torch==2.9.0", "fastapi"]})
    )[0]
    assert "torch==2.9.0" in line
    assert "torch==2.11.0" not in line


def test_torch_is_unpinned_when_the_metal_tree_is_not_local(monkeypatch):
    """A git source has no tree to read a pin from; install torch anyway rather than
    shipping an image whose verify step is guaranteed to fail."""
    from tt_kernel import launchers

    monkeypatch.setattr(launchers, "metal_torch_pin", lambda _tree: None)
    line = launcher_for("tt-dit-server").install_lines(_manifest())[0]
    assert " torch " in line
