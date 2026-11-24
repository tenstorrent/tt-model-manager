# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``kind: cohost``: several members, one container, one port per member, run by tt-cohost."""

import json
import shlex
import subprocess
import sys

import pytest
import yaml

from tt_kernel import build, container
from tt_kernel.container_manifest import ContainerManifestError, load_container_manifest
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
COHOST_SHA = "0123456789abcdef0123456789abcdef01234567"

# An authored co-host manifest: FROM an existing package image, tt-cohost pinned by ref.
AUTHORED = {
    "schema": "5.1",
    "repo": "you/cohost-llm",
    "name": "cohost-llm",
    "weights": "openai/gpt-oss-20b",
    "auxiliary_weights": ["meta-llama/Llama-3.1-8B-Instruct"],
    "kind": "cohost",
    "arch": "blackhole",
    "runtime": {
        "base_image": "tt-model/gpt-oss-20b:abc123",
        "tt_cohost": {"repo": "https://github.com/example/tt-cohost", "ref": "main"},
        "packages": ["torchvision==0.26.0"],
        "members": MEMBERS[:2],
        "tt_config": {"fabric_config": "FABRIC_1D_RING"},
    },
    "serve": {"port": 8000},
    "serve_profiles": [{"name": "p150x4", "hardware": "p150x4", "mesh_device": "P150x4"}],
    "verify": ["import models.cohost_llm"],
}


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


def _authored(tmp_path, **runtime):
    raw = json.loads(json.dumps(AUTHORED))
    raw["runtime"].update(runtime)
    p = tmp_path / "tt-model.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


# ------------------------------------------------------------------------------ serve


def test_cohost_is_a_registered_kind():
    assert isinstance(launcher_for("cohost"), CohostLauncher)


def test_the_container_runs_the_tt_cohost_launcher(manifest):
    profile = manifest.container.resolve_profile("gptoss-llama")
    assert launcher_for("cohost").serve_argv(manifest, profile) == ["python", "-m", "tt_cohost.launch"]


def test_a_profile_serves_its_members_each_on_its_own_port(manifest):
    profile = manifest.container.resolve_profile("gptoss-llama")
    assert container.published_ports(manifest, profile) == [21000, 21001, 21002]  # + status


def test_the_group_config_reaches_the_runtime_inline(manifest):
    profile = manifest.container.resolve_profile("gptoss-qwen")
    env = launcher_for("cohost").serve_env(manifest, profile)
    cfg = json.loads(env["TT_COHOST_CONFIG"])
    assert cfg["cohost_config_version"] == 1
    assert [(m["name"], m["port"]) for m in cfg["members"]] == [("gptoss", 21000), ("qwen32b", 21002)]
    assert cfg["mesh_device"] == "P150x4" and cfg["status_port"] == 21003
    assert cfg["tt_config"] == {"fabric_config": "FABRIC_1D_RING"}
    assert cfg["members"][0]["env"] == {"GPT_OSS_20B_KV_POOL_BLOCKS": "4128"}
    assert env["MESH_DEVICE"] == "P150x4"


def test_a_member_preload_hook_reaches_the_runtime(manifest):
    profile = manifest.container.resolve_profile("gptoss-llama")
    cfg = json.loads(launcher_for("cohost").serve_env(manifest, profile)["TT_COHOST_CONFIG"])
    assert [m["preload"] for m in cfg["members"]] == ["", "models.cohost_llm.fit:distributed_prefill_norm"]


def test_a_profile_not_in_the_map_serves_every_member(manifest):
    profile = manifest.container.resolve_profile("all")
    assert len(launcher_for("cohost").members(manifest.container.runtime, profile.name)) == 3


def test_compose_run_publishes_every_member_port(manifest, tmp_path):
    profile = manifest.container.resolve_profile("gptoss-llama")
    argv = container.compose_run(manifest, profile, ["python", "-m", "tt_cohost.launch"], {},
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
    ({"members": [{"name": "a", "kind": "vllm", "model": "m"}], "profiles": {"p": []}}, "no members"),
    ({"members": [{"name": "a", "kind": "app"}]}, "needs app"),
    ({"members": [{"name": "b", "kind": "app", "app": "x:y"},
                  {"name": "a", "kind": "vllm", "model": "m", "port_offset": 1}]}, "vllm members before"),
])
def test_bad_member_lists_are_refused(runtime, match):
    with pytest.raises(LauncherError, match=match):
        CohostLauncher().members(runtime, "p")


def test_a_mixed_group_with_its_apps_last_is_accepted():
    runtime = {"members": [{"name": "a", "kind": "vllm", "model": "m"},
                           {"name": "b", "kind": "app", "app": "x:y", "port_offset": 1}]}
    assert [m["name"] for m in CohostLauncher().members(runtime, None)] == ["a", "b"]


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


def test_a_runtime_profile_that_names_no_serve_profile_is_refused(manifest):
    """A typo there would serve every member under the profile the author meant."""
    manifest.container.runtime["profiles"]["gptoss-qwem"] = manifest.container.runtime["profiles"].pop("gptoss-qwen")
    with pytest.raises(ContainerManifestError, match="name no serve profile"):
        CohostLauncher().validate(manifest.container)


# ----------------------------------------------------------------------- authoring


def test_an_authored_cohost_manifest_needs_no_source(tmp_path):
    m = load_container_manifest(_authored(tmp_path))
    assert m.source is None and m.kind == "cohost"


def test_a_cohost_manifest_refuses_a_source_block(tmp_path):
    raw = json.loads(json.dumps(AUTHORED))
    raw["source"] = {"tt_metal": "/tmp/tt-metal", "code": ["models"], "ubuntu": "22.04", "python": "3.10"}
    p = tmp_path / "tt-model.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContainerManifestError, match="takes no source"):
        load_container_manifest(p)


@pytest.mark.parametrize("runtime, match", [
    ({"base_image": ""}, "runtime.base_image"),
    ({"tt_cohost": None}, "runtime.tt_cohost"),
    ({"tt_cohost": {"repo": "https://github.com/example/tt-cohost"}}, "runtime.tt_cohost"),
    ({"packages": "torchvision"}, "runtime.packages"),
    ({"code": [{"root": "/x"}]}, "runtime.code"),
])
def test_cohost_build_inputs_are_checked_at_load(tmp_path, runtime, match):
    with pytest.raises(ContainerManifestError, match=match):
        load_container_manifest(_authored(tmp_path, **runtime))


def test_a_kind_that_builds_tt_metal_still_needs_source(tmp_path):
    from test_container_manifest import BASE
    raw = {k: v for k, v in BASE.items() if k != "source"}
    p = tmp_path / "tt-model.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ContainerManifestError, match="needs a source: block"):
        load_container_manifest(p)


# ---------------------------------------------------------------------------- build


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A co-host package staged against a fake base image, with the tt-cohost ref pinned."""
    code_root = tmp_path / "llm"
    (code_root / "models" / "cohost_llm").mkdir(parents=True)
    (code_root / "models" / "cohost_llm" / "fit.py").write_text("def tt_transformers():\n    pass\n")
    monkeypatch.setattr(build, "resolve_git_ref", lambda repo, ref: COHOST_SHA)
    monkeypatch.setattr(build, "_local_image", lambda ref: {
        "id": "sha256:" + "b" * 64, "user": "1000:1000",
        "labels": {"org.tenstorrent.tt-model.tt-metal": "ee5703ce" + "0" * 32}})
    path = _authored(tmp_path, code=[{"root": str(code_root), "paths": ["models/cohost_llm"]}])
    return build.stage(path, tmp_path / "out")


def test_the_rendered_dockerfile_installs_the_pinned_tt_cohost(staged):
    dockerfile = (staged.ctx / "Dockerfile").read_text()
    assert "FROM tt-model/gpt-oss-20b:abc123\n" in dockerfile
    assert f"ADD https://github.com/example/tt-cohost.git#{COHOST_SHA} /\n" in dockerfile
    install = dockerfile[dockerfile.index("RUN --mount=from=ghcr.io/astral-sh/uv"):]
    install = install[:install.index("\nCOPY")]
    assert "--no-deps" in install and "/tmp/tt-cohost torchvision==0.26.0" in install
    assert "COPY code/ /opt/tt-metal/" in dockerfile
    # only what shipped: a chmod of the base's whole models/ would copy it into a new layer
    assert "RUN chmod -R a+rX /opt/tt-metal/models/cohost_llm \\\n" in dockerfile
    # COPY resets an existing parent dir to the context's mode, which locks the serve uid out
    assert "    && chmod a+rX /opt/tt-metal/models \\\n" in dockerfile
    assert dockerfile.index("USER 1000:1000") < dockerfile.index("RUN bash /ctx/verify.sh")
    assert f'org.tenstorrent.tt-model.tt-cohost="{COHOST_SHA}"' in dockerfile
    assert dockerfile.rstrip().endswith('CMD ["/usr/local/bin/serve-default.sh"]')


def test_staging_records_the_base_image_and_the_tt_cohost_commit(staged):
    assert staged.metal is None and staged.code_tree == ["models/cohost_llm/"]
    assert staged.built["base_image"] == {"ref": "tt-model/gpt-oss-20b:abc123", "id": "sha256:" + "b" * 64}
    assert staged.built["tt_cohost"] == {"repo": "https://github.com/example/tt-cohost", "sha": COHOST_SHA}
    assert staged.built["tt_metal"]["sha"].startswith("ee5703ce")
    assert staged.manifest.runtime["tt_cohost"]["sha"] == COHOST_SHA
    # code/ ships in the repo; the author's local root is not published
    assert "code" not in staged.manifest.runtime
    assert (staged.ctx / "code" / "models" / "cohost_llm" / "fit.py").is_file()


def test_the_image_verifies_the_launcher_and_every_preload_hook(staged):
    verify = (staged.ctx / "verify.sh").read_text().splitlines()
    checks = [shlex.split(line)[-1] for line in verify if "python" in line]
    assert checks == ["import tt_cohost.launch; import models.cohost_llm.fit", "import models.cohost_llm"]


APP_MEMBERS = [
    {"name": "whisper", "kind": "app", "model": "openai/whisper-large-v3", "port_offset": 0,
     "app": "models.cohost_speech.whisper:member"},
    {"name": "s2pro", "kind": "app", "model": "fishaudio/s2-pro", "port_offset": 1,
     "app": "models.cohost_speech.s2pro:member"},
]


def _first_check(m):
    return shlex.split(CohostLauncher().verify_lines(m)[0])[-1]


def test_the_image_resolves_every_app_factory_after_the_launcher(tmp_path):
    m = load_container_manifest(_authored(tmp_path, members=APP_MEMBERS))
    assert _first_check(m) == (
        "import tt_cohost.launch; import importlib; "
        "getattr(importlib.import_module('models.cohost_speech.whisper'), 'member'); "
        "getattr(importlib.import_module('models.cohost_speech.s2pro'), 'member')")


@pytest.mark.parametrize("whisper, ok", [
    ("from tt_cohost.apps import AppMember\ndef member(host, m):\n    pass\n", True),
    # an import the image cannot satisfy, e.g. a runtime it no longer ships: refused at build
    ("from old_runtime.apps import AppMember\ndef member(host, m):\n    pass\n", False),
    ("from tt_cohost.apps import AppMember\n", False),   # no factory
])
def test_the_app_check_fails_a_member_that_cannot_load(tmp_path, whisper, ok):
    files = {"tt_cohost/__init__.py": "", "tt_cohost/launch.py": "",
             "tt_cohost/apps.py": "class AppMember:\n    pass\n",
             "models/__init__.py": "", "models/cohost_speech/__init__.py": "",
             "models/cohost_speech/whisper.py": whisper,
             "models/cohost_speech/s2pro.py": "def member(host, m):\n    pass\n"}
    for rel, text in files.items():
        (tmp_path / "tree" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "tree" / rel).write_text(text)
    check = _first_check(load_container_manifest(_authored(tmp_path, members=APP_MEMBERS)))
    run = subprocess.run([sys.executable, "-c", check], env={"PYTHONPATH": str(tmp_path / "tree")},
                         capture_output=True, text=True)
    assert (run.returncode == 0) is ok, run.stderr


def test_the_image_default_command_serves_the_default_profile(staged):
    serve = (staged.ctx / "serve-default.sh").read_text()
    assert "exec python -m tt_cohost.launch" in serve and "TT_COHOST_CONFIG=" in serve


def test_a_from_image_build_passes_no_metal_context(staged):
    argv = build.build_argv(staged)
    assert "--build-context" not in argv and argv[-1] == str(staged.ctx)


def test_the_card_names_the_base_image_and_tt_cohost(staged):
    card = build.render_model_card(staged.manifest, staged.built)
    assert "| base image | `tt-model/gpt-oss-20b:abc123`" in card
    assert f"| tt-cohost | [`{COHOST_SHA[:9]}" in card


def test_a_missing_base_image_is_named_before_the_build(tmp_path, monkeypatch):
    class _Done:
        returncode, stdout, stderr = 1, "", "No such image"
    monkeypatch.setattr(build.subprocess, "run", lambda *a, **k: _Done())
    with pytest.raises(build.BuildError, match="tt-model/gpt-oss-20b:abc123 is not on this machine"):
        build.stage(_authored(tmp_path), tmp_path / "out")


def test_a_short_tt_cohost_sha_is_refused_before_the_build(tmp_path, monkeypatch):
    """BuildKit fetches the tt-cohost commit by name, and a short sha is not one it can fetch."""
    monkeypatch.setattr(build, "_local_image", lambda ref: {"id": "sha256:1", "user": "", "labels": {}})
    monkeypatch.setattr(build, "resolve_git_ref", lambda repo, ref: ref)
    path = _authored(tmp_path, tt_cohost={"repo": "https://github.com/example/tt-cohost", "ref": "cd9feea"})
    with pytest.raises(build.BuildError, match="full 40-character commit sha"):
        build.stage(path, tmp_path / "out")


def test_deploy_reserves_a_cohost_instance_whole_port_range(manifest, tmp_path, monkeypatch):
    """gptoss-qwen publishes members at base and base+2 and the status port at base+3."""
    from tt_kernel import cli

    small = manifest.model_copy(deep=True)
    small.container.serve_profiles = [small.container.serve_profiles[0].model_copy(
        update={"name": "one", "hardware": "p150", "mesh_device": "P150", "port": None})]
    small.container.default_profile = "one"
    (tmp_path / "d.yaml").write_text("instances:\n  - {name: group, model: org/group, profile: gptoss-qwen}\n"
                                     "  - {name: next, model: org/next}\n")
    monkeypatch.setattr(cli, "_require_container", lambda model: manifest if model == "org/group" else small)
    monkeypatch.setattr(container, "all_device_ids", lambda: [0, 1, 2, 3, 4])
    monkeypatch.setattr(container, "_claimed_devices", lambda inventory: set())
    monkeypatch.setattr(container, "chip_boards", lambda: {0: "A", 1: "A", 2: "B", 3: "B", 4: "C"})
    monkeypatch.setattr(container, "port_is_free", lambda port: True)
    monkeypatch.setattr(container, "free_hugepages_1g", lambda: 16)
    _, plan = cli._deploy_plan(str(tmp_path / "d.yaml"))
    group, nxt = plan
    assert group.port == 20000 and nxt.port == 20004  # members at 20000 and 20002, status port 20003
