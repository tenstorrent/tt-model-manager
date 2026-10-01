# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`tt-model serve` stamps tt.model.* descriptor labels on a container so a UI (TT Studio) can
read what the package is instead of reverse-engineering it from names/routes (issue #140).

Only facts the manifest already encodes are emitted — no invented `task` taxonomy. That is enough
to fix the two reported misclassifications: a tool-calling model is marked `tool_calling=true`, and
an image-gen (`tt-dit-server`) package is marked `openai_compatible=false` so it can't register as
chat.
"""

import json

from tt_kernel import container
from tt_kernel.container_manifest import ContainerManifest

from test_container_manifest import BASE, FORK

_NS = container.DESCRIPTOR_LABEL_NS  # "tt.model"


def _wire(**over):
    raw = json.loads(json.dumps(BASE))
    raw.update(over)
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    return m.to_wire(image_tag="tt-model/m:abc", tt_metal_version="0.72.1",
                     tt_kernel_version="0.1.0", hostname="h",
                     created_at="2026-01-01T00:00:00+00:00")


def _dit_wire():
    raw = json.loads(json.dumps(BASE))
    raw["kind"] = "tt-dit-server"
    raw["runtime"] = {"app": "models.tt_dit.server.flux2.app:app"}
    raw.pop("serve_profiles", None)
    raw.pop("default_profile", None)
    raw["serve"] = {"hardware": "p150x4", "mesh_device": "P150x4", "port": 8000}
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    return m.to_wire(image_tag="tt-model/m:abc", tt_metal_version="0.72.1",
                     tt_kernel_version="0.1.0", hostname="h",
                     created_at="2026-01-01T00:00:00+00:00")


def _labels(m, port=20000):
    return container.descriptor_labels(m, m.container.resolve_profile(), port)


# -- the derived label set ------------------------------------------------------------

def test_a_vllm_plugin_package_is_described_as_an_openai_chat_stack():
    labels = _labels(_wire())
    assert labels[f"{_NS}.kind"] == "vllm-plugin"
    assert labels[f"{_NS}.openai_compatible"] == "true"
    assert labels[f"{_NS}.tool_calling"] == "false"      # BASE declares no capabilities
    assert labels[f"{_NS}.reasoning"] == "false"
    assert labels[f"{_NS}.port"] == "20000"
    assert labels[f"{_NS}.manifest_schema"] == "5.1"


def test_tool_and_reasoning_capabilities_flip_the_labels():
    """The gpt-oss miss: the model was launched with a tool-call parser but the UI recorded it as
    non-tool-calling. The label reads straight off the manifest's capabilities."""
    m = _wire(serve={**BASE["serve"],
                     "capabilities": {"tool_parser": "openai", "reasoning_parser": "deepseek_r1"}})
    labels = _labels(m)
    assert labels[f"{_NS}.tool_calling"] == "true"
    assert labels[f"{_NS}.reasoning"] == "true"


def test_a_fork_stack_is_still_openai_compatible():
    labels = _labels(_wire(**FORK))
    assert labels[f"{_NS}.kind"] == "vllm-fork"
    assert labels[f"{_NS}.openai_compatible"] == "true"


def test_a_dit_server_is_not_openai_compatible():
    """The qwen-image miss: a POST /predict server registered as chat. It is now plainly marked
    not-OpenAI, so a consumer won't route it to the chat page."""
    labels = _labels(_dit_wire())
    assert labels[f"{_NS}.kind"] == "tt-dit-server"
    assert labels[f"{_NS}.openai_compatible"] == "false"
    assert labels[f"{_NS}.tool_calling"] == "false"


def test_port_label_follows_the_port_serve_uses():
    assert _labels(_wire(), port=8001)[f"{_NS}.port"] == "8001"


# -- they actually land on the docker run argv ----------------------------------------

def test_compose_run_stamps_the_descriptor_labels_on_the_container():
    m = _wire(serve={**BASE["serve"], "capabilities": {"tool_parser": "openai"}})
    profile = m.container.resolve_profile()
    argv = container.compose_run(m, profile, ["vllm", "serve"], {}, include_hf_token=False)
    port = profile.port or 20000
    for k, v in container.descriptor_labels(m, profile, port).items():
        assert f"{k}={v}" in argv, f"missing --label {k}={v}"
    # the operational labels are still there too (we added to, not replaced, them)
    assert f"{container.LABEL}={m.name}" in argv
