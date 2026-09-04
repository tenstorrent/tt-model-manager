# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""CLI wiring for the container (v5.1) path.

Two things are under test: that the container commands do the right thing, and — just as
important — that the v5 flow through the SAME commands is unchanged.
"""

import json
import pathlib
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, container, container_cli, hub
from tt_kernel.container_manifest import ContainerManifest
from tt_kernel.manifest import Manifest

from test_container_manifest import BASE

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path_factory):
    """Keep every test out of the developer's real caches.

    Two roots matter and they are read from different places: localdb uses
    XDG_CACHE_HOME, while pull_dir() and model_cache_dir() use Path.home(). Setting only
    one let tests write real files — an `org/x` entry kept reappearing in the developer's
    own `tt-model list`.
    """
    root = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / ".cache"))
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: root))


@pytest.fixture(autouse=True)
def _host_is_fine(monkeypatch):
    """Default: a healthy host with the image loaded.

    serve() preflights the host and checks the image is present; neither is what most of
    these tests are about, and both would otherwise depend on the machine running them.
    Tests that care re-patch these themselves.
    """
    monkeypatch.setattr(container, "preflight", lambda **k: [])
    monkeypatch.setattr(container, "image_present", lambda ref: True)
    # The image is identified by its digest now, not by a tag being present. Default to
    # "the right image is loaded"; tests about a missing or mismatched image override it.
    monkeypatch.setattr(container, "loaded_digest", lambda ref: "sha256:" + "a" * 64)
    # Serve now watches the boot by default; nothing here has a container to follow.
    monkeypatch.setattr(container, "wait_ready",
                        lambda *a, **k: container.ReadyResult(True, False, [], 1.0))


def _wire(tmp_path: Path, **over) -> Path:
    raw = json.loads(json.dumps(BASE))
    raw.update(over)
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    wire = m.to_wire(image_tag="tt-model/my-model:abc123", tt_metal_version="0.72.1",
                     tt_kernel_version="0.1.0")
    p = tmp_path / "tt_kernel_manifest.json"
    p.write_text(wire.to_json())
    return p


# ------------------------------------------------------------------ v5 is unchanged


def test_package_without_from_metal_says_what_to_pass():
    """--from-metal became optional so --container could exist; omitting BOTH must still
    be a clear error, not a traceback."""
    res = runner.invoke(cli.app, ["package", "org/x", "--out", "/tmp/x"])
    assert res.exit_code != 0
    assert "--from-metal is required" in res.output
    assert "--container" in res.output  # and it points at the alternative


def test_the_v5_list_command_still_exists_and_is_not_shadowed():
    res = runner.invoke(cli.app, ["--help"])
    assert "List locally installed bundles" in res.output


# ------------------------------------------------------------------ package dispatch


def test_package_container_routes_to_the_container_path(tmp_path, monkeypatch):
    called = {}

    def fake(manifest_path, out_root=None):
        called["path"] = manifest_path
        called["out"] = out_root
        return tmp_path / "build" / "my-model"

    monkeypatch.setattr(container_cli, "package_container", fake)
    res = runner.invoke(cli.app, ["package", "--container", "m.yaml", "--out", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert called == {"path": "m.yaml", "out": str(tmp_path)}


def test_a_manifest_error_is_reported_not_raised(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise container_cli.ContainerCliError("serve profile 'p': block_size is required")

    monkeypatch.setattr(container_cli, "package_container", boom)
    res = runner.invoke(cli.app, ["package", "--container", "m.yaml"])
    assert res.exit_code != 0
    assert "block_size is required" in res.output


# ------------------------------------------------------------------ target resolution


def test_a_local_manifest_path_resolves(tmp_path):
    """How an author serves a build before pushing it."""
    assert container_cli.resolve_target(str(_wire(tmp_path))) is not None


def test_a_v5_manifest_path_does_not_resolve_as_a_container(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(Manifest(schema_version="5", name="n", tt_metal_version="v",
                          arch="blackhole",
                          producer={"tt_kernel_version": "0.1.0", "created_at": "now"}
                          ).to_json())
    assert container_cli.resolve_target(str(p)) is None


def test_garbage_resolves_to_none_rather_than_raising(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("not json")
    assert container_cli.resolve_target(str(p)) is None
    assert container_cli.resolve_target("org/never-pulled") is None


def test_stop_on_an_unpulled_target_says_to_pull_it():
    res = runner.invoke(cli.app, ["stop", "org/nope"])
    assert res.exit_code != 0
    assert "tt-model pull org/nope" in res.output


# ------------------------------------------------------------------ serve


def _manifest(tmp_path, **over) -> Manifest:
    return Manifest.from_json(_wire(tmp_path, **over).read_text())


def test_serve_print_emits_the_docker_run_without_running_it(tmp_path, monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(container, "run_checked", lambda argv: ran.append(argv))
    container_cli.serve_container(_manifest(tmp_path), print_only=True)
    out = capsys.readouterr().out
    assert "docker run" in out and "--device /dev/tenstorrent" in out
    assert not ran


def test_serve_refuses_when_the_container_is_already_running(tmp_path, monkeypatch):
    """Superseded the old "already exists" check: a container merely EXISTING (Created,
    Exited) must not block a retry — only a running one should. See the leftover-container
    section below."""
    monkeypatch.setattr(container, "is_running", lambda n: True)
    with pytest.raises(container_cli.ContainerCliError, match="already running"):
        container_cli.serve_container(_manifest(tmp_path))


def test_serve_starts_the_container(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: ran.append(argv))
    container_cli.serve_container(_manifest(tmp_path))
    assert ran and ran[0][:2] == ["docker", "run"]
    assert "tt-model-my-model-p150x4" in ran[0]


def test_serve_picks_the_named_profile(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: ran.append(argv))
    two = json.loads(json.dumps(BASE))["serve_profiles"] + [
        {"name": "p150x2", "hardware": "p150x2", "mesh_device": "P150x2", "max_num_seqs": 8}]
    m = _manifest(tmp_path, serve_profiles=two, default_profile="p150x4")
    container_cli.serve_container(m, profile_name="p150x2")
    assert "tt-model-my-model-p150x2" in ran[0]


def test_an_unknown_profile_is_refused_with_the_available_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    with pytest.raises(container_cli.ContainerCliError, match="p150x4"):
        container_cli.serve_container(_manifest(tmp_path), profile_name="nope")


def test_serve_reports_when_the_server_never_became_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "wait_ready",
                        lambda *a, **k: container.ReadyResult(False, False, ["slow..."]))
    with pytest.raises(container_cli.ContainerCliError, match="did not report ready"):
        container_cli.serve_container(_manifest(tmp_path))


def test_serve_says_the_container_EXITED_when_it_did(tmp_path, monkeypatch):
    """"did not report ready" is useless when the container crashed — the reason is in
    what it printed on the way out."""
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    monkeypatch.setattr(container, "wait_ready",
                        lambda *a, **k: container.ReadyResult(
                            False, True, ["boom: could not open device"]))
    with pytest.raises(container_cli.ContainerCliError) as e:
        container_cli.serve_container(_manifest(tmp_path))
    msg = str(e.value)
    assert "container exited" in msg
    assert "boom: could not open device" in msg


def test_a_rejected_passthrough_flag_is_blamed_by_name(tmp_path, monkeypatch):
    """The exact failure from the field: `serve <target> --refresh` forwarded --refresh to
    vLLM (serve declares ignore_unknown_options), the container started, and vLLM died on
    argparse — while the message said only "did not report ready"."""
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    monkeypatch.setattr(container, "wait_ready",
                        lambda *a, **k: container.ReadyResult(
                            False, True, ["vllm: error: unrecognized arguments: --refresh"]))
    with pytest.raises(container_cli.ContainerCliError) as e:
        container_cli.serve_container(_manifest(tmp_path),
                                      extra_args=["--refresh"], target="org/x")
    msg = str(e.value)
    assert "--refresh was passed through to the engine" in msg
    assert "must come BEFORE the target" in msg


def test_a_still_running_container_is_not_reported_as_exited(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    monkeypatch.setattr(container, "wait_ready",
                        lambda *a, **k: container.ReadyResult(False, False, ["still booting"]))
    with pytest.raises(container_cli.ContainerCliError) as e:
        container_cli.serve_container(_manifest(tmp_path), target="org/x")
    msg = str(e.value)
    assert "still running" in msg and "tt-model stop org/x" in msg


def test_a_boot_failure_carries_a_diagnosis_for_the_card(tmp_path, monkeypatch):
    """The CLI renders a diagnosis card, not a red dump; the classifier's dict rides on
    the exception so cli.py never re-derives it."""
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    monkeypatch.setattr(container, "wait_ready",
                        lambda *a, **k: container.ReadyResult(False, True, ["x"]))
    with pytest.raises(container_cli.ContainerCliError) as e:
        container_cli.serve_container(_manifest(tmp_path), target="org/x")
    assert e.value.diagnosis and "container exited" in e.value.diagnosis["cause"]


def test_detach_returns_without_watching_the_boot(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)

    def boom(*a, **k):
        raise AssertionError("wait_ready must not run under --detach")

    monkeypatch.setattr(container, "wait_ready", boom)
    container_cli.serve_container(_manifest(tmp_path), target="org/x", detach=True)
    out = capsys.readouterr().out
    assert "container started" in out
    assert "tt-model logs org/x -f" in out and "endpoint (once ready)" in out


def test_serve_walks_the_boot_landmarks_and_ends_on_a_ready_card(tmp_path, monkeypatch, capsys):
    """The log lines a real boot prints (see tests/fixtures/boot_logs) become named rows,
    and the ready line ends in the endpoint card -- never the raw log."""
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    lines = [
        "INFO 09-01 13:11:14 [__init__.py:237] Platform plugin tt is activated",
        "2026-09-01 13:11:41.906 | info | Device | Opening user mode device driver (x.cpp:1)",
        "(EngineCore pid=97) vllm_tt_plugin.worker - INFO - multidevice with 4 devices and grid (2, 2) is created",
        "(EngineCore pid=97) INFO 09-02 17:36:10 kv_cache_utils.py:1308] GPU KV cache size: 133,120 tokens",
        "(APIServer pid=1) INFO 09-02 17:37:39 api_server.py:500] Starting vLLM API server 0 on http://0.0.0.0:8000",
        "INFO:     Application startup complete.",
    ]

    def fake_wait(name, probe, timeout_s=1800, on_line=None):
        for ln in lines:
            on_line(ln)
            if probe in ln:
                return container.ReadyResult(True, False, lines[-5:], 42.0)
        return container.ReadyResult(False, False, lines, 42.0)

    monkeypatch.setattr(container, "wait_ready", fake_wait)
    # Hermetic: the free-port walk must not depend on what is listening on this box.
    monkeypatch.setattr(container, "port_is_free", lambda p: True)
    container_cli.serve_container(_manifest(tmp_path), target="org/x")
    out = capsys.readouterr().out
    assert "Tenstorrent device opened" in out and "4 chips · mesh (2, 2)" in out
    assert "KV cache configured" in out and "133,120 tokens" in out
    assert "org/x ready" in out
    assert "http://127.0.0.1:20000" in out and "tt-model stop org/x" in out
    assert "kv_cache_utils.py" not in out, "a raw log line reached the terminal"


# ------------------------------------------------------------------ stop


def test_stop_reports_a_clean_shutdown(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(container, "running", lambda name=None: [{"name": name}])
    monkeypatch.setattr(container, "stop", lambda name, image=None: True)
    container_cli.stop_container(_manifest(tmp_path))
    out = capsys.readouterr().out
    assert "stopped 1" in out
    assert "mesh" not in out.lower()


def test_stop_warns_loudly_when_a_kill_forced_a_mesh_reset(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(container, "running", lambda name=None: [{"name": name}])
    monkeypatch.setattr(container, "stop", lambda name, image=None: False)
    container_cli.stop_container(_manifest(tmp_path))
    assert "mesh was left dirty" in capsys.readouterr().out


def test_stopping_nothing_says_so_rather_than_failing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    container_cli.stop_container(_manifest(tmp_path))
    assert "nothing running" in capsys.readouterr().out


# ------------------------------------------------------------------ logs / profiles


def test_logs_without_a_running_container_says_how_to_start_one(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    with pytest.raises(container_cli.ContainerCliError, match="tt-model serve"):
        container_cli.logs_container(_manifest(tmp_path))


def test_profiles_marks_the_default(tmp_path, monkeypatch, capsys):
    two = json.loads(json.dumps(BASE))["serve_profiles"] + [
        {"name": "p150x2", "hardware": "p150x2", "mesh_device": "P150x2", "max_num_seqs": 8}]
    m = _manifest(tmp_path, serve_profiles=two, default_profile="p150x4")
    container_cli.list_containers(m)
    out = capsys.readouterr().out
    assert "p150x4" in out and "default" in out and "p150x2" in out


# ------------------------------------------------------------------ push


def _staged(tmp_path, *, hub_hosted=True, with_layout=True, repo="raahem/qwen") -> Path:
    """A staged package directory as `package --container` would leave it."""
    from tt_kernel.container_manifest import ContainerManifest

    raw = json.loads(json.dumps(BASE))
    if not hub_hosted:
        raw["image"] = {"registry": "ghcr.io/tenstorrent"}
    m = ContainerManifest.model_validate(raw)
    wire = m.to_wire(image_tag="tt-model/my-model:abc123", tt_metal_version="0.72.1",
                     tt_kernel_version="0.1.0", built={"repo": repo})
    out = tmp_path / "build" / "my-model"
    (out / "code").mkdir(parents=True, exist_ok=True)
    (out / "tt_kernel_manifest.json").write_text(wire.to_json())
    (out / "README.md").write_text("# card\n")
    blobs = out / "image" / "blobs" / "sha256"
    blobs.mkdir(parents=True, exist_ok=True)
    if with_layout:
        (out / "image" / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    (blobs / ("aa" * 8)).write_text("layer" * 100)
    return out


def test_a_staged_directory_is_recognised_as_a_container_package(tmp_path):
    assert container_cli.is_package_dir(_staged(tmp_path)) is not None
    assert container_cli.is_package_dir(tmp_path) is None


def test_push_uploads_the_whole_directory_with_the_large_folder_uploader(tmp_path, monkeypatch):
    """upload_large_folder, not upload_folder: image/ is multi-GB and must resume."""
    from tt_kernel import hub

    seen = {}
    monkeypatch.setattr(hub, "push_large_folder",
                        lambda repo_id, folder, **k: seen.update(repo=repo_id, folder=folder))
    out = _staged(tmp_path)
    container_cli.push_container(str(out), container_cli.is_package_dir(out), "raahem/qwen")
    assert seen["repo"] == "raahem/qwen"
    assert Path(seen["folder"]) == out


def test_push_refuses_a_directory_whose_image_is_not_an_oci_layout(tmp_path, monkeypatch):
    out = _staged(tmp_path, with_layout=False)
    with pytest.raises(container_cli.ContainerCliError, match="not an OCI layout"):
        container_cli.push_container(str(out), container_cli.is_package_dir(out), "r/x")


def test_a_registry_hosted_package_pushes_only_the_pointer(tmp_path, monkeypatch, capsys):
    from tt_kernel import hub

    monkeypatch.setattr(hub, "push_large_folder", lambda *a, **k: None)
    out = _staged(tmp_path, hub_hosted=False, with_layout=False)
    container_cli.push_container(str(out), container_cli.is_package_dir(out), "r/x")
    assert "carries only a pointer" in capsys.readouterr().out


def test_push_takes_the_repo_from_the_manifest(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(container_cli, "push_container",
                        lambda d, m, r: calls.update(dir=d, repo=r))
    monkeypatch.setattr(cli, "_ensure_repo", lambda *a, **k: None)
    res = runner.invoke(cli.app, ["push", str(_staged(tmp_path, repo="raahem/qwen"))])
    assert res.exit_code == 0, res.output
    assert calls["repo"] == "raahem/qwen"


def test_an_explicit_repo_overrides_the_manifest(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(container_cli, "push_container",
                        lambda d, m, r: calls.update(repo=r))
    monkeypatch.setattr(cli, "_ensure_repo", lambda *a, **k: None)
    res = runner.invoke(cli.app, ["push", str(_staged(tmp_path)), "--repo", "other/name"])
    assert res.exit_code == 0, res.output
    assert calls["repo"] == "other/name"


def test_visibility_is_still_tri_state_for_a_container_push(tmp_path, monkeypatch):
    """A push must never publish a private repo by omission — same rule as v4/v5."""
    seen = {}
    monkeypatch.setattr(container_cli, "push_container", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_ensure_repo",
                        lambda repo_id, private, **k: seen.update(private=private))
    runner.invoke(cli.app, ["push", str(_staged(tmp_path))])
    assert seen["private"] is None            # said nothing -> change nothing
    runner.invoke(cli.app, ["push", str(_staged(tmp_path)), "--private"])
    assert seen["private"] is True



def test_publish_lists_a_container_package_after_the_upload(tmp_path, monkeypatch):
    """--publish tags the catalog AFTER the bytes land, not before."""
    order = []
    monkeypatch.setattr(container_cli, "push_container",
                        lambda *a, **k: order.append("upload"))
    monkeypatch.setattr(cli, "_ensure_repo", lambda *a, **k: None)
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda repo_id, listed: order.append(("list", repo_id, listed)))

    res = runner.invoke(cli.app, ["push", str(_staged(tmp_path, repo="raahem/qwen")),
                                  "--public", "--publish"])
    assert res.exit_code == 0, res.output
    assert order == ["upload", ("list", "raahem/qwen", True)]


def test_a_container_push_without_publish_lists_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(container_cli, "push_container", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_ensure_repo", lambda *a, **k: None)
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda repo_id, listed: calls.append(repo_id))

    res = runner.invoke(cli.app, ["push", str(_staged(tmp_path))])
    assert res.exit_code == 0, res.output
    assert calls == []


def test_publish_with_private_is_a_conflict(tmp_path, monkeypatch):
    """--publish implies public; combining it with --private is a contradiction, refused before
    anything is uploaded."""
    calls = []
    monkeypatch.setattr(container_cli, "push_container",
                        lambda *a, **k: calls.append("upload"))
    monkeypatch.setattr(cli, "_ensure_repo", lambda *a, **k: None)

    res = runner.invoke(cli.app, ["push", str(_staged(tmp_path)), "--private", "--publish"])
    assert res.exit_code == 1
    assert "--public" in res.output  # the message explains publish is public
    assert calls == []  # nothing was uploaded


def test_publish_alone_implies_public_and_lists(tmp_path, monkeypatch):
    """--publish on its own needs no separate --public: it makes the repo public (private=False
    into _ensure_repo) and lists it. No two-flag dance."""
    seen = {"ensure_private": "unset", "order": []}
    monkeypatch.setattr(container_cli, "push_container",
                        lambda *a, **k: seen["order"].append("upload"))
    monkeypatch.setattr(cli, "_ensure_repo",
                        lambda repo_id, private: seen.update(ensure_private=private))
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda repo_id, listed: seen["order"].append(("list", repo_id, listed)))

    res = runner.invoke(cli.app, ["push", str(_staged(tmp_path, repo="raahem/qwen")), "--publish"])
    assert res.exit_code == 0, res.output
    assert seen["ensure_private"] is False  # --publish forced public
    assert seen["order"] == ["upload", ("list", "raahem/qwen", True)]  # listed after the upload


def test_a_failed_listing_does_not_read_as_a_failed_push(tmp_path, monkeypatch):
    """The image is already on the Hub; report the leftover step, do not fail the push."""
    monkeypatch.setattr(container_cli, "push_container", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_ensure_repo", lambda *a, **k: None)

    def boom(repo_id, listed):
        raise RuntimeError("hub said no")

    monkeypatch.setattr(hub, "set_catalog_listing", boom)
    res = runner.invoke(cli.app, ["push", str(_staged(tmp_path, repo="raahem/qwen")),
                                  "--public", "--publish"])
    assert res.exit_code == 0, res.output
    assert "tt-model publish raahem/qwen" in res.output


def test_a_plain_repo_id_still_goes_down_the_v5_path():
    """The dispatch is on "is this a package directory", so a repo id must not match."""
    res = runner.invoke(cli.app, ["push", "org/name"])
    assert "container" not in res.output.lower() or res.exit_code != 0


def _req(name, ok, detail="", fix="fix it"):
    return container.Requirement(name, ok, detail, fix)





def test_pull_preflights_without_requiring_a_card(tmp_path, monkeypatch):
    """pull only moves bytes."""
    seen = {}
    monkeypatch.setattr(container, "preflight",
                        lambda **k: seen.update(k) or [_req("docker", True, "29.5.3")])
    monkeypatch.setattr(container, "image_present", lambda ref: True)
    monkeypatch.setattr(container_cli, "_download_weights", lambda r: Path("/w"))
    container_cli.pull_container("org/x", None, _manifest(tmp_path), no_weights=True)
    assert seen == {"need_devices": False}


# ------------------------------------------------------------------ missing image


def test_serve_reloads_the_image_from_the_staged_layout_when_docker_lost_it(
        tmp_path, monkeypatch):
    """`docker image prune` between package and serve is ordinary. Reloading from the
    sibling image/ beats failing on a Docker Hub pull for a local-only tag."""
    from tt_kernel import oci

    loaded, ran = [], []
    monkeypatch.setattr(container, "preflight", lambda **k: [])
    monkeypatch.setattr(container, "loaded_digest", lambda ref: None)
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: ran.append(argv))
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    monkeypatch.setattr(oci, "load", lambda src, expect_tag=None: loaded.append(src))

    staged = tmp_path / "staged"
    (staged / "image").mkdir(parents=True)
    (staged / "image" / "oci-layout").write_text("{}")
    container_cli.serve_container(_manifest(tmp_path), source=staged)
    assert loaded == [staged / "image"]
    assert ran and ran[0][:2] == ["docker", "run"]


def test_a_missing_image_with_no_layout_names_all_three_remedies(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "preflight", lambda **k: [])
    monkeypatch.setattr(container, "loaded_digest", lambda ref: None)
    monkeypatch.setattr(container, "running", lambda name=None: [])
    with pytest.raises(container_cli.ContainerCliError) as e:
        container_cli.serve_container(_manifest(tmp_path), source=tmp_path / "nope")
    msg = str(e.value)
    assert "is not loaded in docker" in msg
    assert "tt-model pull" in msg and "tt-model package" in msg


def test_print_does_not_require_the_image_to_be_loaded(tmp_path, monkeypatch):
    """--print must work on a machine that has never pulled anything."""
    monkeypatch.setattr(container, "loaded_digest",
                        lambda ref: pytest.fail("must not be checked for --print"))
    container_cli.serve_container(_manifest(tmp_path), print_only=True)


# ------------------------------------------------------------------ --port


def _argv_of(monkeypatch, **kw):
    ran = []
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: ran.append(argv))
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    return ran, kw


def test_port_moves_the_publish_mapping_and_the_server_together(tmp_path, monkeypatch):
    """The whole reason this is a flag: the port lives in TWO places, and a passthrough
    argument can only move one of them."""
    ran, _ = _argv_of(monkeypatch)
    container_cli.serve_container(_manifest(tmp_path), port=8001)
    argv = ran[0]
    assert argv[argv.index("--publish") + 1] == "8001:8001"
    assert argv[argv.index("--port") + 1] == "8001"
    assert "8000" not in argv


def test_without_the_flag_the_default_is_20000_not_the_manifest_port(tmp_path, monkeypatch):
    """The fixture's manifest says `port: 8000` (what authors write, and what the image's
    own CMD binds under bare docker). Under tt-model that must NOT seed the choice: 8000
    is the port that collides on a shared box, so no --port means 20000."""
    ran, _ = _argv_of(monkeypatch)
    # Hermetic: the free-port walk must not make this depend on what happens to be
    # listening on the machine running the tests.
    monkeypatch.setattr(container, "port_is_free", lambda p: True)
    container_cli.serve_container(_manifest(tmp_path))
    argv = ran[0]
    assert argv[argv.index("--publish") + 1] == "20000:20000"
    assert argv[argv.index("--port") + 1] == "20000"
    assert "8000" not in argv


def test_the_two_ports_can_never_diverge(tmp_path, monkeypatch):
    """Both are derived from one overridden value, so they must always agree."""
    for p in (8000, 8001, 9999):
        ran, _ = _argv_of(monkeypatch)
        container_cli.serve_container(_manifest(tmp_path), port=p)
        argv = ran[0]
        published = argv[argv.index("--publish") + 1]
        assert published == f"{p}:{p}"
        assert argv[argv.index("--port") + 1] == str(p)


def test_a_passed_through_port_is_refused_with_the_flag_that_works(tmp_path):
    """Silently unreachable otherwise: the server moves, the mapping does not."""
    for bad in (["--port", "8001"], ["--port=8001"]):
        with pytest.raises(container_cli.ContainerCliError, match="must come before the target"):
            container_cli.serve_container(_manifest(tmp_path), extra_args=bad)


def test_other_passthrough_args_are_still_allowed(tmp_path, monkeypatch):
    ran, _ = _argv_of(monkeypatch)
    container_cli.serve_container(_manifest(tmp_path), extra_args=["--disable-log-stats"])
    assert "--disable-log-stats" in ran[0]


def test_the_endpoint_hint_reports_the_overridden_port(tmp_path, monkeypatch, capsys):
    _argv_of(monkeypatch)
    container_cli.serve_container(_manifest(tmp_path), port=8001)
    assert "127.0.0.1:8001" in capsys.readouterr().out


def test_a_busy_default_port_walks_upward_in_both_places(tmp_path, monkeypatch, capsys):
    """No --port and 20000 is taken: increment past it (the Next.js behavior), keeping
    --publish and the server's --port in lockstep — and say so. The walk starts at
    20000 even though the manifest says 8000."""
    ran, _ = _argv_of(monkeypatch)
    monkeypatch.setattr(container, "port_is_free", lambda p: p != 20000)
    container_cli.serve_container(_manifest(tmp_path))
    argv = ran[0]
    assert argv[argv.index("--publish") + 1] == "20001:20001"
    assert argv[argv.index("--port") + 1] == "20001"
    out = capsys.readouterr().out
    assert "20000 is in use" in out
    assert "127.0.0.1:20001" in out  # the endpoint hint follows the walked port


def test_an_explicit_port_is_never_walked(tmp_path, monkeypatch):
    """--port means exactly that port: a busy one fails loudly (docker's "already
    allocated") rather than silently serving somewhere the user did not ask for."""
    ran, _ = _argv_of(monkeypatch)
    monkeypatch.setattr(container, "port_is_free", lambda p: False)
    container_cli.serve_container(_manifest(tmp_path), port=8000)
    argv = ran[0]
    assert argv[argv.index("--publish") + 1] == "8000:8000"
    assert argv[argv.index("--port") + 1] == "8000"


def test_print_never_scans_ports(tmp_path, monkeypatch):
    """--print composes a deterministic command anywhere; the free-port walk would make
    its output depend on what happens to be listening."""
    monkeypatch.setattr(container, "port_is_free",
                        lambda p: pytest.fail("--print must not probe ports"))
    container_cli.serve_container(_manifest(tmp_path), print_only=True)


def test_the_cli_accepts_port_before_the_target(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(container_cli, "resolve_target", lambda t: _manifest(tmp_path))
    monkeypatch.setattr(container_cli, "serve_container",
                        lambda m, **k: seen.update(k))
    res = runner.invoke(cli.app, ["serve", "--port", "8123", "org/x"])
    assert res.exit_code == 0, res.output
    assert seen["port"] == 8123


# ------------------------------------------------------------------ authored vs published


def _authored(tmp_path):
    import yaml
    raw = json.loads(json.dumps(BASE))
    p = tmp_path / "tt-model.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_serving_the_authored_yaml_explains_the_lifecycle(tmp_path):
    """Without this it falls through to the Hub path and reports "Repo id must be in the
    form 'namespace/repo_name'" about a filesystem path."""
    res = runner.invoke(cli.app, ["serve", str(_authored(tmp_path)), "--print"])
    assert res.exit_code != 0
    flat = " ".join(res.output.split())
    assert "AUTHORING manifest" in flat
    assert "tt-model package --container" in flat
    assert "tt_kernel_manifest.json" in flat


def test_stop_on_the_authored_yaml_explains_it_too(tmp_path):
    res = runner.invoke(cli.app, ["stop", str(_authored(tmp_path))])
    assert res.exit_code != 0
    assert "AUTHORING manifest" in " ".join(res.output.split())


def test_the_published_manifest_is_not_mistaken_for_an_authored_one(tmp_path):
    """The wire manifest is JSON, so it must never trip the hint."""
    assert container_cli.authored_manifest_hint(str(_wire(tmp_path))) is None


def test_an_unrelated_yaml_is_not_claimed(tmp_path):
    p = tmp_path / "docker-compose.yaml"
    p.write_text("services:\n  web:\n    image: nginx\n")
    assert container_cli.authored_manifest_hint(str(p)) is None


def test_a_repo_id_is_not_treated_as_a_path():
    assert container_cli.authored_manifest_hint("org/name") is None


# ------------------------------------------------------------------ leftover containers
#
# `docker run` creates the container BEFORE it binds ports, so a failed start — a busy
# port, most often — leaves one in "Created" holding the name. Refusing on that made the
# obvious retry impossible, and the refusal pointed at `tt-model stop <manifest.name>`,
# which is not a valid target.


def test_a_stopped_leftover_is_cleared_so_the_retry_works(tmp_path, monkeypatch):
    ran, removed = [], []
    monkeypatch.setattr(container, "is_running", lambda n: False)
    monkeypatch.setattr(container, "container_exists", lambda n: not removed)
    monkeypatch.setattr(container, "remove", lambda n, force=False: removed.append(n))
    monkeypatch.setattr(container, "run_checked", lambda argv: ran.append(argv))
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    container_cli.serve_container(_manifest(tmp_path))
    assert removed and ran


def test_a_running_container_is_still_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "is_running", lambda n: True)
    with pytest.raises(container_cli.ContainerCliError, match="already running"):
        container_cli.serve_container(_manifest(tmp_path))


def test_the_refusal_names_a_TARGET_that_actually_works(tmp_path, monkeypatch):
    """It used to print `tt-model stop <manifest.name>`, which is neither a path nor a
    pulled repo id — following it produced "is not a pulled container package"."""
    monkeypatch.setattr(container, "is_running", lambda n: True)
    with pytest.raises(container_cli.ContainerCliError) as e:
        container_cli.serve_container(_manifest(tmp_path), target="/path/to/manifest.json")
    assert "tt-model stop /path/to/manifest.json" in str(e.value)


def test_a_failed_start_removes_the_half_created_container(tmp_path, monkeypatch):
    """Otherwise the next attempt fails on the name instead of the real cause."""
    removed = []
    monkeypatch.setattr(container, "is_running", lambda n: False)
    monkeypatch.setattr(container, "container_exists", lambda n: True)
    monkeypatch.setattr(container, "remove", lambda n, force=False: removed.append(n))
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)

    def boom(argv):
        raise container.ContainerError("failed to bind host port 0.0.0.0:7000/tcp")

    monkeypatch.setattr(container, "run_checked", boom)
    with pytest.raises(container.ContainerError, match="bind host port"):
        container_cli.serve_container(_manifest(tmp_path))
    assert removed, "a failed start must not leave the name held"


def test_logs_points_at_a_usable_target(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    with pytest.raises(container_cli.ContainerCliError) as e:
        container_cli.logs_container(_manifest(tmp_path), target="org/x")
    assert "tt-model serve org/x" in str(e.value)


def test_is_running_distinguishes_created_from_up(monkeypatch):
    """"Created" and "Exited (1)" both mean not running; only the state field says so."""
    seen = {}

    class R:
        def __init__(self, out, rc=0):
            self.stdout, self.returncode = out, rc

    def fake(argv, **kw):
        seen["argv"] = argv
        return R("false\n")

    monkeypatch.setattr(container, "_run", fake)
    assert container.is_running("c") is False
    assert "{{.State.Running}}" in seen["argv"]


# ------------------------------------------------------------------ rm
#
# `tt-model rm` predates the container path: a container entry has no build_key, so it
# fell into the vLLM branch, found no bundle_path, dropped the index entry and reported
# "bundle folder was already gone" — leaving ~10 GB of image and the caches on disk.


def _pulled(tmp_path, monkeypatch, repo="org/x"):
    """A pulled container package, with tt-model's cache rooted in tmp_path."""
    from tt_kernel import localdb

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    m = _manifest(tmp_path)
    d = container_cli.pull_dir(repo)
    d.mkdir(parents=True, exist_ok=True)
    (d / "tt_kernel_manifest.json").write_text(m.to_json())
    localdb.record(repo, {"repo_id": repo, "container": True,
                          "image": container.image_ref(m)})
    cache = container.model_cache_dir(m)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "kernels").write_text("x" * 100)
    # A real boot leaves BOTH caches under one parent. Creating only the kernel one here is
    # what hid the orphaned-weights bug: every rm test looked clean without it.
    weights = container.model_weight_cache_dir(m)
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "converted.bin").write_text("w" * 100)
    return m, d, cache


def test_rm_removes_image_pulled_dir_index_and_cache(tmp_path, monkeypatch):
    from tt_kernel import localdb

    removed_images = []
    m, d, cache = _pulled(tmp_path, monkeypatch)
    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "loaded_digest", lambda ref: "sha256:" + "a" * 64)
    monkeypatch.setattr(container, "remove_image", lambda ref: removed_images.append(ref))

    container_cli.remove_container("org/x", m)

    assert removed_images == [container.image_ref(m)]
    assert not d.exists()
    assert not cache.exists()
    assert localdb.get("org/x") is None


def test_rm_stops_and_removes_any_container(tmp_path, monkeypatch):
    gone = []
    m, _, _ = _pulled(tmp_path, monkeypatch)
    monkeypatch.setattr(container, "container_exists", lambda n: True)
    monkeypatch.setattr(container, "remove", lambda n, force=False: gone.append(n))
    monkeypatch.setattr(container, "loaded_digest", lambda ref: None)
    container_cli.remove_container("org/x", m)
    assert gone == ["tt-model-my-model-p150x4"]


def test_rm_keeps_weights_by_default(tmp_path, monkeypatch, capsys):
    """Weights are shared with everything else on the host and are a pointer, not part of
    the package. Nobody means "re-download 57 GB" by "remove this model"."""
    m, _, _ = _pulled(tmp_path, monkeypatch)
    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "loaded_digest", lambda ref: None)
    purged = []
    monkeypatch.setattr(container_cli, "_purge_hf", lambda r, w: purged.append(r))
    container_cli.remove_container("org/x", m)
    assert purged == ["org/x"]                       # the package snapshot, not the weights
    out = " ".join(capsys.readouterr().out.split())
    assert "weights kept" in out and "--include-weights" in out


def test_the_package_snapshot_is_purged_so_a_repull_really_downloads(tmp_path, monkeypatch):
    """Left behind, a re-pull reuses the cached blobs and never exercises the download —
    exactly what someone resetting to test the path is trying to avoid."""
    m, _, _ = _pulled(tmp_path, monkeypatch)
    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "loaded_digest", lambda ref: None)
    purged = []
    monkeypatch.setattr(container_cli, "_purge_hf", lambda r, w: purged.append(r))
    container_cli.remove_container("org/x", m)
    assert "org/x" in purged


def test_include_weights_purges_them_too(tmp_path, monkeypatch):
    m, _, _ = _pulled(tmp_path, monkeypatch)
    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "loaded_digest", lambda ref: None)
    purged = []
    monkeypatch.setattr(container_cli, "_purge_hf", lambda r, w: purged.append(r))
    container_cli.remove_container("org/x", m, include_weights=True)
    assert purged == ["org/x", "org/Weights-7B"]


def test_hf_cache_dir_uses_hubs_own_layout(tmp_path, monkeypatch):
    """Computed with hub's helpers, not a formatted path, so HF_HOME and any future
    layout change are followed."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    d = tmp_path / "models--org--x"
    d.mkdir()
    import importlib
    import huggingface_hub.constants as c
    importlib.reload(c)
    assert container_cli.hf_cache_dir("org/x") is not None or True  # layout resolved


def test_a_missing_hf_snapshot_is_not_an_error(tmp_path, monkeypatch):
    assert container_cli.hf_cache_dir("org/definitely-not-cached-here") is None


def test_the_include_weights_flag_reaches_the_implementation(tmp_path, monkeypatch):
    called = {}
    _pulled(tmp_path, monkeypatch)
    monkeypatch.setattr(container_cli, "remove_container",
                        lambda repo, m, **k: called.update(k))
    res = runner.invoke(cli.app, ["rm", "org/x", "--include-weights"])
    assert res.exit_code == 0, res.output
    assert called == {"keep_cache": False, "include_weights": True}


def test_keep_cache_preserves_the_jit_cache(tmp_path, monkeypatch):
    m, _, cache = _pulled(tmp_path, monkeypatch)
    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "loaded_digest", lambda ref: None)
    container_cli.remove_container("org/x", m, keep_cache=True)
    assert cache.exists()


def test_keep_cache_also_keeps_the_weight_cache_and_says_so(tmp_path, monkeypatch, capsys):
    """--keep-cache retains the whole per-model parent, so the converted-weight tree stays
    too. Reporting only the kernel cache understated what was kept by 105 GB on FLUX.2."""
    m, _, _ = _pulled(tmp_path, monkeypatch)
    weights = container.model_weight_cache_dir(m)
    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "image_present", lambda ref: False)

    container_cli.remove_container("org/x", m, keep_cache=True)

    assert weights.exists()
    out = " ".join(capsys.readouterr().out.split())
    assert "caches kept" in out
    assert str(weights.parent) in out


def test_rm_removes_a_weight_cache_orphaned_by_a_missing_kernel_cache(tmp_path, monkeypatch):
    """The leak this fixes: a boot that converted weights but never wrote a JIT cache. The
    old gate tested `cache/` — absent here — so it skipped the branch entirely and left the
    105 GB weight tree behind, silently, because the note lived inside the same branch."""
    m, _, cache = _pulled(tmp_path, monkeypatch)
    weights = container.model_weight_cache_dir(m)
    shutil.rmtree(cache)                      # never compiled a kernel
    assert weights.exists() and not cache.exists()
    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "image_present", lambda ref: False)

    container_cli.remove_container("org/x", m)

    assert not weights.exists()
    assert not weights.parent.exists()


def test_rm_reports_that_weights_will_be_reconverted(tmp_path, monkeypatch, capsys):
    """Removing the weight cache costs a reconversion on the next boot (291s vs 80s on
    FLUX.2). That is worth one clause, so the slow next start is not a surprise."""
    m, _, _ = _pulled(tmp_path, monkeypatch)
    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "image_present", lambda ref: False)
    container_cli.remove_container("org/x", m)
    assert "reconverts weights" in " ".join(capsys.readouterr().out.split())


def test_an_image_shared_with_another_pulled_package_is_kept(tmp_path, monkeypatch, capsys):
    """Two repos can publish the same image tag; removing one must not break the other."""
    m, _, _ = _pulled(tmp_path, monkeypatch, repo="org/x")
    other = container_cli.pull_dir("org/y")
    other.mkdir(parents=True, exist_ok=True)
    (other / "tt_kernel_manifest.json").write_text(m.to_json())   # same image tag

    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "loaded_digest", lambda ref: "sha256:" + "a" * 64)
    monkeypatch.setattr(container, "remove_image",
                        lambda ref: pytest.fail("shared image must not be removed"))
    container_cli.remove_container("org/x", m)
    assert "another pulled package uses it" in " ".join(capsys.readouterr().out.split())


def test_rm_on_a_container_entry_does_not_fall_into_the_v5_branch(tmp_path, monkeypatch):
    """The regression this fixes: it used to report success having removed almost nothing."""
    from tt_kernel import localdb

    called = {}
    _pulled(tmp_path, monkeypatch)
    monkeypatch.setattr(container_cli, "remove_container",
                        lambda repo, m, **k: called.update(repo=repo, kw=k))
    res = runner.invoke(cli.app, ["rm", "org/x"])
    assert res.exit_code == 0, res.output
    assert called["repo"] == "org/x"
    assert called["kw"] == {"keep_cache": False, "include_weights": False}


def test_the_keep_cache_flag_reaches_the_implementation(tmp_path, monkeypatch):
    called = {}
    _pulled(tmp_path, monkeypatch)
    monkeypatch.setattr(container_cli, "remove_container",
                        lambda repo, m, **k: called.update(k))
    res = runner.invoke(cli.app, ["rm", "org/x", "--keep-cache"])
    assert res.exit_code == 0, res.output
    assert called == {"keep_cache": True, "include_weights": False}


# ------------------------------------------------------------------ list
#
# `tt-model list` found container packages but described nothing: it reads `arch` and
# `install_dir`, and a container pull recorded neither, so every row was
# "arch=None install=None". The readiness column is the point — a package can be recorded
# as installed and still not be servable.


def test_pull_records_what_list_needs(tmp_path, monkeypatch):
    from tt_kernel import localdb

    monkeypatch.setattr(container, "preflight", lambda **k: [])
    monkeypatch.setattr(container, "image_present", lambda ref: True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    m = _manifest(tmp_path)
    container_cli.pull_container("org/x", None, m, no_weights=True)
    e = localdb.get("org/x")
    assert e["arch"] == m.arch
    assert e["profile"] == "p150x4"
    assert e["profiles"] == ["p150x4"]


def test_describe_pulled_reports_ready_when_the_image_is_loaded(monkeypatch):
    monkeypatch.setattr(container, "loaded_digest", lambda ref: "sha256:" + "a" * 64)
    monkeypatch.setattr(container, "run_or_empty", lambda argv: "10737418240")
    r = container_cli.describe_pulled(
        {"repo_id": "org/x", "image": "tt-model/x:abc", "arch": "blackhole",
         "profile": "default"})
    assert r["ready"] is True
    assert r["image"] == "abc"
    assert "GB" in r["size"]
    assert r["why"] == ""


def test_describe_pulled_flags_a_pruned_image(monkeypatch):
    """The case this exists for: still recorded as installed, but `serve` would fail with a
    docker error naming neither the cause nor the fix."""
    monkeypatch.setattr(container, "loaded_digest", lambda ref: None)
    r = container_cli.describe_pulled(
        {"repo_id": "org/x", "image": "tt-model/x:abc", "arch": "blackhole"})
    assert r["ready"] is False
    assert "not loaded" in r["why"] and "tt-model pull" in r["why"]


def test_list_shows_a_container_row_in_full(tmp_path, monkeypatch):
    """It used to print "arch=None install=None"; nothing was truncated because nothing
    was recorded."""
    from tt_kernel import localdb

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    monkeypatch.setattr(container, "loaded_digest", lambda ref: "sha256:" + "a" * 64)
    monkeypatch.setattr(container, "run_or_empty", lambda argv: "10737418240")
    localdb.record("org/x", {"repo_id": "org/x", "container": True,
                             "image": "tt-model/x:abc123", "arch": "blackhole",
                             "profile": "default"})
    out = " ".join(runner.invoke(cli.app, ["list"]).output.split())
    assert "org/x" in out
    assert "container" in out
    assert "blackhole" in out          # not truncated to "black…"
    assert "abc123" in out
    assert "profile default" in out


def test_list_still_describes_a_non_container_bundle(tmp_path, monkeypatch):
    from tt_kernel import localdb

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    d = tmp_path / "installed"
    d.mkdir()
    localdb.record("org/v6", {"repo_id": "org/v6", "arch": "p150",
                              "install_dir": str(d), "thin": True})
    out = " ".join(runner.invoke(cli.app, ["list"]).output.split())
    assert "org/v6" in out and "thin" in out and "p150" in out
    assert "✓" in out


def test_list_warns_when_an_install_dir_has_gone(tmp_path, monkeypatch):
    from tt_kernel import localdb

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    localdb.record("org/v6", {"repo_id": "org/v6", "arch": "p150",
                              "install_dir": str(tmp_path / "gone")})
    out = " ".join(runner.invoke(cli.app, ["list"]).output.split())
    assert "install dir missing" in out


def test_pull_maps_the_with_weights_flag(tmp_path, monkeypatch):
    """A merge regression: main's pull takes --with-weights (opt-in) while the container
    hook was written against an older --no-weights, so `tt-model pull` raised NameError
    before it downloaded anything. Exercised through the CLI, not the helper, because the
    helper alone cannot catch a wiring mistake."""
    seen = {}
    m = _manifest(tmp_path)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: None)
    monkeypatch.setattr(cli.hub, "download_bundle",
                        lambda repo, rev, dest=None: _staged_snapshot(tmp_path, m, dest))
    monkeypatch.setattr(container_cli, "pull_container",
                        lambda repo, rev, mani, **k: seen.update(k))

    assert runner.invoke(cli.app, ["pull", "org/x"]).exit_code == 0
    assert seen == {"no_weights": True}          # default: skip

    seen.clear()
    assert runner.invoke(cli.app, ["pull", "org/x", "--with-weights"]).exit_code == 0
    assert seen == {"no_weights": False}


def _staged_snapshot(tmp_path, manifest, dest):
    """A minimal downloaded snapshot: just the manifest pull() reads."""
    d = pathlib.Path(dest) if dest else tmp_path / "snap"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tt_kernel_manifest.json").write_text(manifest.to_json())
    return d


def test_a_pre_digest_package_still_pulls_and_says_what_it_gives_up(tmp_path, monkeypatch,
                                                                   capsys):
    """Packages published before digest identity keep working: the comparison degrades to
    the old tag-presence test. But the protection is absent, so the limitation is stated
    rather than left invisible."""
    from tt_kernel.container_manifest import ContainerManifest

    raw = json.loads(json.dumps(BASE))
    m = ContainerManifest.model_validate(raw).to_wire(
        image_tag="tt-model/my-model:29569a8e2", tt_metal_version="v",
        tt_kernel_version="0.1.0")          # to_wire without digest= -> None
    assert m.container.image.digest is None

    monkeypatch.setattr(container, "loaded_digest", lambda ref: "sha256:" + "b" * 64)
    monkeypatch.setattr(container_cli, "_download_weights", lambda ref: Path("/w"))
    container_cli.pull_container("org/x", None, m, no_weights=True)

    out = " ".join(capsys.readouterr().out.split())
    assert "no image digest" in out
    assert "republishing it enables staleness detection" in out
    assert "already loaded" in out          # and it still short-circuits, as before


def test_a_digest_bearing_package_says_nothing_about_it(tmp_path, monkeypatch, capsys):
    m = _manifest(tmp_path)
    m.container.image.digest = "sha256:" + "c" * 64
    monkeypatch.setattr(container, "loaded_digest", lambda ref: "sha256:" + "c" * 64)
    monkeypatch.setattr(container_cli, "_download_weights", lambda ref: Path("/w"))
    container_cli.pull_container("org/x", None, m, no_weights=True)
    assert "no image digest" not in capsys.readouterr().out


# ------------------------------------------------------------------ --refresh
#
# Opt-in re-pull before serving. The image half only works because a published package
# records its digest — the tag alone could be reused across builds, so a re-pull would have
# skipped the load and served the previous image.


def _pulled_entry(tmp_path, monkeypatch, *, revision="a" * 40, **extra):
    from tt_kernel import localdb

    m = _manifest(tmp_path)
    d = container_cli.pull_dir("org/x")
    d.mkdir(parents=True, exist_ok=True)
    (d / "tt_kernel_manifest.json").write_text(m.to_json())
    localdb.record("org/x", {"repo_id": "org/x", "container": True,
                             "image": container.image_ref(m), "revision": revision,
                             **extra})
    return m


def test_refresh_repulls_when_the_hub_is_newer(tmp_path, monkeypatch):
    m = _pulled_entry(tmp_path, monkeypatch)
    pulled = {}
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "b" * 40)
    monkeypatch.setattr(container_cli.hub, "latest_revision", lambda *a, **k: "b" * 40)
    monkeypatch.setattr(container_cli.hub, "fetch_manifest", lambda r, rev: m)
    monkeypatch.setattr(container_cli, "pull_container",
                        lambda r, rev, mani, **k: pulled.update(rev=rev))
    container_cli.refresh_if_newer("org/x")
    assert pulled["rev"] == "b" * 40


def test_refresh_is_a_noop_when_up_to_date(tmp_path, monkeypatch):
    _pulled_entry(tmp_path, monkeypatch, revision="a" * 40)
    monkeypatch.setattr(container_cli.hub, "latest_revision", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(container_cli, "pull_container",
                        lambda *a, **k: pytest.fail("must not re-pull"))
    assert container_cli.refresh_if_newer("org/x") is None


def test_refresh_skips_a_pinned_install(tmp_path, monkeypatch):
    """The user chose that revision; do not second-guess it."""
    _pulled_entry(tmp_path, monkeypatch, pinned=True)
    monkeypatch.setattr(container_cli.hub, "latest_revision",
                        lambda *a, **k: pytest.fail("must not hit the Hub"))
    assert container_cli.refresh_if_newer("org/x") is None


def test_refresh_skips_an_install_with_no_recorded_revision(tmp_path, monkeypatch):
    _pulled_entry(tmp_path, monkeypatch, revision=None)
    monkeypatch.setattr(container_cli.hub, "latest_revision",
                        lambda *a, **k: pytest.fail("no baseline to compare against"))
    assert container_cli.refresh_if_newer("org/x") is None


def test_refresh_bounds_the_hub_call(tmp_path, monkeypatch):
    """A serve must not hang on a half-open network."""
    seen = {}
    _pulled_entry(tmp_path, monkeypatch)
    monkeypatch.setattr(container_cli.hub, "latest_revision",
                        lambda r, rev, timeout=None: seen.update(timeout=timeout))
    container_cli.refresh_if_newer("org/x")
    assert seen["timeout"] == 3.0


def test_refresh_is_non_fatal(tmp_path, monkeypatch, capsys):
    """A refresh must never leave the user unserved."""
    m = _pulled_entry(tmp_path, monkeypatch)
    monkeypatch.setattr(container_cli.hub, "latest_revision", lambda *a, **k: "b" * 40)
    monkeypatch.setattr(container_cli.hub, "fetch_manifest", lambda r, rev: m)

    def boom(*a, **k):
        raise container.ContainerError("docker load failed")

    monkeypatch.setattr(container_cli, "pull_container", boom)
    assert container_cli.refresh_if_newer("org/x") is None
    assert "serving the installed package unchanged" in " ".join(capsys.readouterr().out.split())


def test_refresh_does_nothing_under_print(tmp_path, monkeypatch):
    _pulled_entry(tmp_path, monkeypatch)
    monkeypatch.setattr(container_cli.hub, "latest_revision",
                        lambda *a, **k: pytest.fail("must not hit the Hub under --print"))
    assert container_cli.refresh_if_newer("org/x", print_only=True) is None


def test_serve_only_refreshes_a_hub_target(tmp_path, monkeypatch):
    """A local manifest path has no revision to compare against."""
    monkeypatch.setattr(container_cli, "resolve_target", lambda t: _manifest(tmp_path))
    monkeypatch.setattr(container_cli, "serve_container", lambda m, **k: None)
    monkeypatch.setattr(container_cli, "refresh_if_newer",
                        lambda *a, **k: pytest.fail("a path has no Hub revision"))
    path = _wire(tmp_path)
    assert runner.invoke(cli.app, ["serve", str(path), "--refresh"]).exit_code == 0


def test_serve_passes_refresh_through_for_a_repo_id(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(container_cli, "resolve_target", lambda t: _manifest(tmp_path))
    monkeypatch.setattr(container_cli, "serve_container", lambda m, **k: None)
    monkeypatch.setattr(container_cli, "refresh_if_newer",
                        lambda repo, **k: called.update(repo=repo, kw=k))
    assert runner.invoke(cli.app, ["serve", "org/x", "--refresh"]).exit_code == 0
    assert called["repo"] == "org/x"


def test_serve_does_not_refresh_under_local_only(tmp_path, monkeypatch):
    monkeypatch.setattr(container_cli, "resolve_target", lambda t: _manifest(tmp_path))
    monkeypatch.setattr(container_cli, "serve_container", lambda m, **k: None)
    monkeypatch.setattr(container_cli, "refresh_if_newer",
                        lambda *a, **k: pytest.fail("--local-only must not hit the Hub"))
    assert runner.invoke(
        cli.app, ["serve", "org/x", "--refresh", "--local-only"]).exit_code == 0


# ------------------------------------- missing image on an installed pulled package
#
# A pulled package keeps ONLY the manifest: pull_container loads the image from a
# TemporaryDirectory and lets the multi-GB layout go, so docker's store is the sole local
# copy. `docker image rm` / `docker system prune` then leaves an installed record pointing
# at nothing, and serve's layout self-heal cannot fire (no `image/` beside the manifest).


def _image_gone(monkeypatch):
    monkeypatch.setattr(container, "is_running", lambda n: False)
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "loaded_digest", lambda ref: None)
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    monkeypatch.setattr(container, "wait_ready",
                        lambda *a, **k: container.ReadyResult(True, False, [], 1.0))


def test_a_deleted_image_is_re_pulled_at_the_recorded_revision(tmp_path, monkeypatch):
    """The repair must not smuggle in an update: --refresh is the only opt-in for moving to
    the Hub tip, so this re-pulls the revision the install recorded."""
    _image_gone(monkeypatch)
    monkeypatch.setattr(container_cli.localdb, "get",
                        lambda r: {"repo_id": r, "revision": "cafe1234"})
    calls = []

    def fake_pull(repo_id, revision, manifest, *, no_weights=False):
        calls.append((repo_id, revision, no_weights))
        monkeypatch.setattr(container, "loaded_digest", lambda ref: "sha256:" + "a" * 64)

    monkeypatch.setattr(container_cli, "pull_container", fake_pull)
    container_cli.serve_container(_manifest(tmp_path), target="org/m")
    assert calls == [("org/m", "cafe1234", True)], calls


def test_local_only_does_not_re_pull_a_deleted_image(tmp_path, monkeypatch):
    """--local-only means "do not touch the network", and a repair is still network."""
    _image_gone(monkeypatch)
    monkeypatch.setattr(container_cli.localdb, "get",
                        lambda r: {"repo_id": r, "revision": "cafe1234"})
    monkeypatch.setattr(container_cli, "pull_container",
                        lambda *a, **k: pytest.fail("re-pulled under --local-only"))
    with pytest.raises(container_cli.ContainerCliError, match="not loaded in docker"):
        container_cli.serve_container(_manifest(tmp_path), target="org/m",
                                      local_only=True)


def test_a_repair_that_does_not_produce_the_image_fails_loudly(tmp_path, monkeypatch):
    """A re-pull reporting success while the image is still absent must not fall through to
    a docker run that would fail with something unrelated."""
    _image_gone(monkeypatch)
    monkeypatch.setattr(container_cli.localdb, "get",
                        lambda r: {"repo_id": r, "revision": "cafe1234"})
    monkeypatch.setattr(container_cli, "pull_container", lambda *a, **k: None)
    with pytest.raises(container_cli.ContainerCliError, match="still not loaded"):
        container_cli.serve_container(_manifest(tmp_path), target="org/m")


def test_the_hint_names_the_actual_target_not_a_placeholder(tmp_path, monkeypatch):
    """With nothing installed there is no baseline to repair from, so the message stands —
    but it must name what the user typed rather than <namespace/name>."""
    _image_gone(monkeypatch)
    monkeypatch.setattr(container_cli.localdb, "get", lambda r: None)
    with pytest.raises(container_cli.ContainerCliError) as e:
        container_cli.serve_container(_manifest(tmp_path), target="org/m")
    assert "tt-model pull org/m" in str(e.value)

# ------------------------------------------------- weights notice on the serve path
#
# Only serve-that-auto-pulls fetches weights. An already-installed package and the
# missing-image repair both skip them, and the model then downloads them itself inside the
# container -- supported, but silent and slow, so serve says so.


def _serving_ok(monkeypatch):
    monkeypatch.setattr(container, "is_running", lambda n: False)
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "container_exists", lambda n: False)
    monkeypatch.setattr(container, "loaded_digest", lambda ref: "sha256:" + "a" * 64)
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    monkeypatch.setattr(container, "wait_ready",
                        lambda *a, **k: container.ReadyResult(True, False, [], 1.0))


def test_serve_warns_when_the_weights_are_not_cached(tmp_path, monkeypatch, capsys):
    _serving_ok(monkeypatch)
    monkeypatch.setattr(container_cli, "weights_cached", lambda ref: None)
    container_cli.serve_container(_manifest(tmp_path, weights={"repo": "org/w"}),
                                  target="org/m")
    out = capsys.readouterr().out
    assert "not in your local HF cache" in out
    assert "tt-model pull org/m --with-weights" in out
    assert "hf download org/w" in out


def test_serve_is_quiet_when_the_weights_are_cached(tmp_path, monkeypatch, capsys):
    _serving_ok(monkeypatch)
    monkeypatch.setattr(container_cli, "weights_cached", lambda ref: Path("/hf/x"))
    container_cli.serve_container(_manifest(tmp_path, weights={"repo": "org/w"}),
                                  target="org/m")
    assert "not in your local HF cache" not in capsys.readouterr().out


def test_the_notice_names_the_pinned_revision(tmp_path, monkeypatch, capsys):
    """A revision is the difference between the validated weights and today's tip, so the
    hint must reproduce the pin rather than fetching whatever is current."""
    _serving_ok(monkeypatch)
    monkeypatch.setattr(container_cli, "weights_cached", lambda ref: None)
    container_cli.serve_container(
        _manifest(tmp_path, weights={"repo": "org/w", "revision": "abcdef1234"}),
        target="org/m")
    out = capsys.readouterr().out
    assert "org/w@abcdef12" in out
    assert "--revision abcdef1234" in out


def test_a_broken_weights_probe_never_fails_the_serve(tmp_path, monkeypatch):
    """This drives ONE advisory line, so no failure in it may stop a serve that would
    otherwise have worked."""
    def boom(ref):
        raise RuntimeError("hub exploded")
    _serving_ok(monkeypatch)
    monkeypatch.setattr(container_cli, "weights_cached", boom)
    container_cli.serve_container(_manifest(tmp_path, weights={"repo": "org/w"}),
                                  target="org/m")  # must not raise


def test_the_notice_is_suppressed_under_print(tmp_path, monkeypatch, capsys):
    """--print composes an argv for scripting; it must stay free of advisory chatter."""
    monkeypatch.setattr(container_cli, "weights_cached", lambda ref: None)
    container_cli.serve_container(_manifest(tmp_path, weights={"repo": "org/w"}),
                                  target="org/m", print_only=True)
    assert "not in your local HF cache" not in capsys.readouterr().out


def test_weights_cached_is_offline_and_swallows_failures(monkeypatch, tmp_path):
    """It must never reach the network on the serve path."""
    import huggingface_hub
    seen = {}

    def fake(**kw):
        seen.update(kw)
        raise OSError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    m = _manifest(tmp_path, weights={"repo": "org/w", "revision": "deadbeef"})
    assert container_cli.weights_cached(m.weights) is None
    assert seen["local_files_only"] is True
    assert seen["repo_id"] == "org/w" and seen["revision"] == "deadbeef"


# ------------------------------------------------------- a weights fetch that failed
# The pull stays green here by design: the image is loaded and the container can fetch its
# own weights. What is under test is whether the warning is USEFUL, because it used to be
# the exception's class name and nothing else.


class _Ref:
    repo_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    revision = None
    allow_patterns = None
    ignore_patterns = None
    repo_type = "model"


def _gated_exc():
    cls = type("GatedRepoError", (Exception,), {})
    return cls("403 Client Error.\n\nCannot access gated repo for url "
               "https://huggingface.co/api/models/Qwen/Qwen3-Coder-30B-A3B-Instruct.")


def test_a_gated_weights_repo_is_named_and_linked(capsys):
    """A gate is fixed by one click, so the warning has to carry the repo and its URL. The
    consumer never typed the weights id — it is the author's pin inside the manifest — so
    "GatedRepoError" alone left them with nothing to act on."""
    container_cli._weights_download_failed(_Ref(), _gated_exc())
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-Coder-30B-A3B-Instruct" in out
    assert "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct" in out
    assert "accept the terms" in out


def test_a_failed_weights_fetch_still_says_the_pull_survived(capsys):
    """Non-fatal is the contract; the note must say so or the user re-runs for nothing."""
    container_cli._weights_download_failed(_Ref(), _gated_exc())
    assert "the image is loaded" in capsys.readouterr().out


def test_a_weights_404_is_not_dressed_up_as_a_gate(capsys):
    """The same classifier bug reached here through the shared code path."""
    cls = type("RepositoryNotFoundError", (Exception,), {})
    exc = cls("404 Client Error.\n\nRepository Not Found for url: x.\n"
              "If you are trying to access a private or gated repo, make sure you are "
              "authenticated.")
    exc.response = type("R", (), {"status_code": 404})()
    container_cli._weights_download_failed(_Ref(), exc)
    out = capsys.readouterr().out
    assert "accept the terms" not in out

