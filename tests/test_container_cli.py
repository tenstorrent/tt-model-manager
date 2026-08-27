# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""CLI wiring for the container (v5.1) path.

Two things are under test: that the container commands do the right thing, and — just as
important — that the v5 flow through the SAME commands is unchanged.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, container, container_cli
from tt_kernel.container_manifest import ContainerManifest
from tt_kernel.manifest import Manifest

from test_container_manifest import BASE

runner = CliRunner()


@pytest.fixture(autouse=True)
def _host_is_fine(monkeypatch):
    """Default: a healthy host with the image loaded.

    serve() preflights the host and checks the image is present; neither is what most of
    these tests are about, and both would otherwise depend on the machine running them.
    Tests that care re-patch these themselves.
    """
    monkeypatch.setattr(container, "preflight", lambda **k: [])
    monkeypatch.setattr(container, "image_present", lambda ref: True)


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


def test_serve_refuses_when_the_container_already_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running",
                        lambda name=None: [{"name": name, "status": "Up 3 minutes"}])
    with pytest.raises(container_cli.ContainerCliError, match="already exists"):
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


def test_follow_reports_when_the_server_never_became_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "wait_ready", lambda *a, **k: False)
    with pytest.raises(container_cli.ContainerCliError, match="did not report ready"):
        container_cli.serve_container(_manifest(tmp_path), follow=True)


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


def test_publish_is_refused_for_a_container_package(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_ensure_repo", lambda *a, **k: None)
    res = runner.invoke(cli.app, ["push", str(_staged(tmp_path)), "--publish", "--public"])
    assert res.exit_code != 0
    assert "not listed there yet" in res.output


def test_a_plain_repo_id_still_goes_down_the_v5_path():
    """The dispatch is on "is this a package directory", so a repo id must not match."""
    res = runner.invoke(cli.app, ["push", "org/name"])
    assert "container" not in res.output.lower() or res.exit_code != 0


# ------------------------------------------------------------------ doctor


def _req(name, ok, detail="", fix="fix it"):
    return container.Requirement(name, ok, detail, fix)


def test_doctor_reports_the_container_path_separately(monkeypatch):
    """A consumer whose box can serve must not be told to install a host toolchain the
    container path exists to avoid."""
    monkeypatch.setattr(container, "preflight",
                        lambda **k: [_req("docker", True, "29.5.3"),
                                     _req("tt devices", True, "/dev/tenstorrent"),
                                     _req("hugepages", True, "/dev/hugepages-1G")])
    res = runner.invoke(cli.app, ["doctor"])
    assert "Container path (v5.1)" in res.output
    assert "no host tt-metal, vLLM or venv" in res.output


def test_doctor_says_which_paths_work_when_the_host_toolchain_is_inadequate(monkeypatch):
    monkeypatch.setattr(container, "preflight",
                        lambda **k: [_req("docker", True, "29.5.3"),
                                     _req("tt devices", True, "/dev/tenstorrent"),
                                     _req("hugepages", True, "/dev/hugepages-1G")])
    res = runner.invoke(cli.app, ["doctor"])
    if res.exit_code != 0:            # a box with no host vLLM exits 1
        # rich hard-wraps at the terminal width, so compare on collapsed whitespace
        flat = " ".join(res.output.split())
        assert "host toolchain inadequate" in flat
        assert "container (v5.1) packages can" in flat


def test_doctor_surfaces_a_container_blocker_with_its_fix(monkeypatch):
    monkeypatch.setattr(container, "preflight",
                        lambda **k: [_req("docker", True, "29.5.3"),
                                     _req("hugepages", False, "not mounted",
                                          "mount 1G hugepages at exactly /dev/hugepages-1G")])
    res = runner.invoke(cli.app, ["doctor"])
    assert "mount 1G hugepages at exactly" in res.output


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
    monkeypatch.setattr(container, "image_present", lambda ref: False)
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
    monkeypatch.setattr(container, "image_present", lambda ref: False)
    monkeypatch.setattr(container, "running", lambda name=None: [])
    with pytest.raises(container_cli.ContainerCliError) as e:
        container_cli.serve_container(_manifest(tmp_path), source=tmp_path / "nope")
    msg = str(e.value)
    assert "is not loaded in docker" in msg
    assert "tt-model pull" in msg and "tt-model package" in msg


def test_print_does_not_require_the_image_to_be_loaded(tmp_path, monkeypatch):
    """--print must work on a machine that has never pulled anything."""
    monkeypatch.setattr(container, "image_present",
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


def test_without_the_flag_the_manifest_port_is_used(tmp_path, monkeypatch):
    ran, _ = _argv_of(monkeypatch)
    container_cli.serve_container(_manifest(tmp_path))
    argv = ran[0]
    assert argv[argv.index("--publish") + 1] == "8000:8000"
    assert argv[argv.index("--port") + 1] == "8000"


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
