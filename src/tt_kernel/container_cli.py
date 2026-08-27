# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The container (v5.1) path's command implementations.

These live outside ``cli.py`` on purpose. ``cli.py`` is 2600 lines of v3/v4/v5 flow and
the constraint on this whole path is that none of it changes behaviour; keeping the
container implementations here means ``cli.py`` gains only small dispatch hooks that can
be read in one screen.

Everything user-facing is rendered through :mod:`tt_kernel.console` so the container path
looks like the rest of the tool. The modules underneath (``build``, ``container``,
``oci``, ``launchers``) stay presentation-free.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

from . import MANIFEST_NAME, console, container, hub, localdb, oci
from .build import BuildError, build_log_path, finalize, run_build, stage
from .container_manifest import ContainerManifestError
from .launchers import launcher_for
from .manifest import Manifest

PHASES = ["Resolve", "Stage", "Build", "Export"]


class ContainerCliError(RuntimeError):
    """User-facing failure; the caller turns this into an exit code."""


def require_host(*, need_devices: bool) -> None:
    """Fail fast on a host that cannot possibly run this, naming the fix.

    Called before anything slow. Each of these checks exists because the unchecked
    failure is late and unrecognisable — most of all the hugepages mount, which surfaces
    as a device-open error ten minutes into a boot.
    """
    reqs = container.preflight(need_devices=need_devices)
    bad = container.preflight_failures(reqs)
    if not bad:
        return
    lines = []
    for r in bad:
        lines.append(f"{r.name}: {r.detail}")
        lines.append(f"  → {r.fix}")
    raise ContainerCliError("\n".join(lines))


# --------------------------------------------------------------------------- package


def package_container(manifest_path: str, *, out_root: Optional[str] = None) -> Path:
    """``tt-model package --container <manifest.yaml>``.

    A cold build is 2.5-4 hours, so the log's ``tail -f`` command is printed BEFORE the
    wait starts and the build streams under an interrupt guard.
    """
    console.register_phases(PHASES)

    console.phase("Resolve")
    try:
        with console.step("reading the manifest") as st:
            staged = stage(Path(manifest_path), Path(out_root) if out_root else None)
            st.detail(f"{staged.manifest.name} · {staged.manifest.kind}")
    except (ContainerManifestError, BuildError) as e:
        raise ContainerCliError(str(e)) from None

    metal = staged.metal
    console.note(
        f"tt-metal {(metal.sha or '?')[:9]}"
        + (f" ({metal.branch})" if metal.branch else "")
        + (" (dirty tree — packaged as-is)" if metal.dirty else "")
        + f" · {metal.mode} source",
        marker="•",
    )
    if metal.mode == "local" and metal.pushed is False:
        # Informational, NOT a warning. The target user is a community developer on a
        # local branch or fork; the image carries the whole tree, so nothing ever has
        # to resolve this sha. Saying "push the branch to make provenance verifiable"
        # implied a requirement that does not exist and made a normal situation read
        # like a defect.
        console.note(
            "this commit is on no remote — that is fine, and expected on a local "
            "branch or fork: the image ships the tree itself, so nothing needs to "
            "fetch it. The sha is recorded for reference only.",
            marker="○", style="muted",
        )
    for key in ("vllm", "plugin"):
        pinned = staged.built.get(key)
        if isinstance(pinned, dict):
            console.note(f"{key} pinned to {str(pinned.get('sha'))[:9]}", marker="•")

    console.phase("Stage")
    console.note(f"{len(staged.code_tree)} code path(s) → code/", marker="•")
    if staged.code_skipped:
        # Never silent: a file that vanishes on the way into the image otherwise shows up
        # as a ModuleNotFoundError from verify.sh with nothing explaining why.
        shown = ", ".join(staged.code_skipped[:5])
        more = f" (+{len(staged.code_skipped) - 5} more)" if len(staged.code_skipped) > 5 else ""
        console.note(f"skipped by the ignore list: {shown}{more}", marker="○")

    console.phase("Build")
    log_path = build_log_path(staged.manifest.name)
    console.note(f"watch from another terminal:  tail -f {log_path}", marker="→")
    try:
        run_build(staged, echo=console.raw if console.is_verbose() else None)
    except KeyboardInterrupt as e:
        raise ContainerCliError(str(e)) from None
    except BuildError as e:
        raise ContainerCliError(str(e)) from None

    console.phase("Export")
    with console.step("exporting the image as an OCI layout"):
        out = finalize(staged)
    size = sum(p.stat().st_size for p in (out / "image").rglob("*") if p.is_file())
    console.note(f"image/ {console.fmt_bytes(size)} as content-addressed blobs", marker="•")

    console.milestone(f"packaged {staged.manifest.name} → {out}")
    console.note(f"serve it locally:  tt-model serve {out / MANIFEST_NAME}", marker="→")
    console.note(f"publish it:        tt-model push {out}", marker="→")
    return out


# ------------------------------------------------------------------------------ push


def is_package_dir(path: Path) -> Optional[Manifest]:
    """The container manifest in a staged package directory, or None if it is not one."""
    mpath = Path(path) / MANIFEST_NAME
    if not mpath.is_file():
        return None
    try:
        m = Manifest.from_json(mpath.read_text())
    except ValueError:
        return None
    return m if m.is_container else None


def push_container(staged_dir: str, manifest: Manifest, repo_id: str) -> None:
    """Upload a staged container package directory to the Hub.

    The caller owns repo creation and visibility (``_ensure_repo`` in the CLI), so this
    only moves bytes. The model card is already written into the directory by ``package``
    and carries its own tags, so nothing here rewrites it — ``tag_repo`` would clobber it.
    """
    out = Path(staged_dir)
    image_dir = out / "image"
    spec = manifest.container
    assert spec is not None

    if spec.image.is_hub_hosted:
        if not (image_dir / "oci-layout").is_file():
            raise ContainerCliError(
                f"{image_dir} is not an OCI layout — re-run `tt-model package --container`. "
                "(The manifest says the image travels in this repo.)"
            )
        blobs = [p for p in (image_dir / "blobs").rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in blobs)
        console.note(
            f"image/ {console.fmt_bytes(total)} in {len(blobs)} content-addressed blobs — "
            "layers shared with another model on the same tt-metal commit upload once",
            marker="•",
        )
    else:
        console.note(
            f"image lives in {spec.image.registry}; this repo carries only a pointer "
            f"({spec.image.pull_ref}). Push the image there separately.",
            marker="•",
        )

    with console.step(f"uploading to {repo_id}"):
        hub.push_large_folder(repo_id, out)

    console.milestone(f"pushed {repo_id}")
    console.note(f"consumers:  tt-model serve {repo_id}", marker="→")


# ------------------------------------------------------------------------------ pull


def pull_dir(repo_id: str) -> Path:
    """Where a pulled package's manifest is kept, so serve/stop can find it by id."""
    return Path.home() / ".cache" / "tt-model" / "pulled" / repo_id.replace("/", "__")


def pull_container(repo_id: str, revision: Optional[str], manifest: Manifest, *,
                   no_weights: bool = False) -> None:
    """Snapshot the repo, load the image into docker, and put weights in the HOST cache."""
    spec = manifest.container
    assert spec is not None
    # pull only moves bytes: it works on a machine with no card attached
    require_host(need_devices=False)

    ref = container.image_ref(manifest)
    if container.image_present(ref):
        console.note(f"image {ref} already loaded", marker="•")
    elif spec.image.is_hub_hosted:
        with console.step(f"docker load {ref}"):
            with tempfile.TemporaryDirectory() as td:
                snapshot = hub.download_bundle(repo_id, revision, dest=td)
                oci.load(snapshot / "image", expect_tag=ref)
    else:
        # The image lives in a real registry; the repo carried only a pointer.
        argv = container.compose_pull(manifest)
        with console.step(f"docker pull {ref}"):
            container.run_checked(argv)

    dest = pull_dir(repo_id)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / MANIFEST_NAME).write_text(manifest.to_json())
    localdb.record(repo_id, {
        "repo_id": repo_id, "container": True, "image": ref,
        "revision": revision, "manifest": str(dest / MANIFEST_NAME),
    })

    if not no_weights and manifest.weights:
        try:
            with console.step("weights → host HF cache") as st:
                path = _download_weights(manifest.weights)
                st.detail(str(path))
        except Exception as e:  # noqa: BLE001
            console.note(
                f"weights not downloaded ({e.__class__.__name__}); the image is loaded and "
                "the model will fetch them at first boot (slower, inside the container)",
                marker="⚠", style="warning",
            )

    console.milestone(f"pulled {repo_id}")
    console.note(f"next:  tt-model serve {repo_id}", marker="→")


def _download_weights(ref) -> Path:
    """Fetch the weights into the HOST HF cache, honouring whatever the author pinned.

    A revision is the difference between "the weights that were validated" and "whatever
    the default branch points at today", so it has to reach snapshot_download rather than
    just being recorded in the manifest.
    """
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=ref.repo_id,
        revision=ref.revision,
        allow_patterns=ref.allow_patterns,
        ignore_patterns=ref.ignore_patterns,
    ))


# ----------------------------------------------------------------------------- serve


def load_pulled(repo_id: str) -> Optional[Manifest]:
    """The manifest of an already-pulled container package, or None."""
    path = pull_dir(repo_id) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        m = Manifest.from_json(path.read_text())
    except ValueError:
        return None
    return m if m.is_container else None


def authored_manifest_hint(target: str) -> Optional[str]:
    """If ``target`` is an AUTHORED tt-model.yaml, the message explaining what to do.

    ``serve`` takes the PUBLISHED manifest (``tt_kernel_manifest.json``) or an ``org/name``
    — never the YAML the author wrote, because an image has to be built from it first. But
    the YAML is what an author thinks of as "the manifest", so pointing serve at it is the
    obvious mistake, and without this it falls through to the Hub path and reports
    "Repo id must be in the form 'namespace/repo_name'" about a filesystem path.
    """
    p = Path(target)
    if not p.is_file() or p.suffix.lower() not in (".yaml", ".yml"):
        return None
    try:
        from .container_manifest import load_container_manifest

        m = load_container_manifest(p)
    except Exception:  # noqa: BLE001 — not an authored manifest; let the caller proceed
        return None
    return (
        f"{p} is the AUTHORING manifest, which describes how to BUILD the image — serve "
        f"needs the built package.\n"
        f"  → build it:  tt-model package --container {p}\n"
        f"  → then:      tt-model serve <out-dir>/{m.name}/{MANIFEST_NAME}\n"
        f"  → or, once published:  tt-model serve {m.repo}"
    )


def resolve_target(target: str) -> Optional[Manifest]:
    """A command target is either a local manifest path or a pulled ``org/name``."""
    p = Path(target)
    if p.is_file():
        try:
            m = Manifest.from_json(p.read_text())
        except ValueError:
            return None
        return m if m.is_container else None
    return load_pulled(target)


def serve_container(manifest: Manifest, *, profile_name: Optional[str] = None,
                    print_only: bool = False, follow: bool = False,
                    extra_args: Optional[List[str]] = None,
                    source: Optional[Path] = None,
                    port: Optional[int] = None,
                    target: Optional[str] = None) -> None:
    """Run one serve profile.

    ``source`` is the staged/pulled dir the manifest came from, used to reload the image
    from ``image/`` if docker no longer has it. ``port`` overrides the profile's port.
    ``target`` is the argument the user actually typed, so hints can be copy-pasted —
    ``manifest.name`` is NOT a valid target and telling someone to use it sends them in a
    circle.
    """
    spec = manifest.container
    assert spec is not None
    if not print_only:
        # --print composes a command without running it, so it must work anywhere.
        require_host(need_devices=True)

    # The port has to move in TWO places at once — docker's --publish mapping and the
    # server's own --port — so it cannot be a passthrough argument. Passing --port after
    # the target reaches vLLM only: the server moves, the published mapping does not, and
    # the endpoint becomes silently unreachable with no error anywhere. Refuse it and
    # name the flag that works.
    for a in (extra_args or []):
        if a == "--port" or a.startswith("--port="):
            raise ContainerCliError(
                "--port must come before the target so tt-model can publish it too: "
                "`tt-model serve --port <n> <target>`. Passed after the target it reaches "
                "only the server inside the container, while docker still publishes the "
                "manifest's port — the endpoint would be unreachable."
            )

    try:
        profile = spec.resolve_profile(profile_name)
    except ValueError as e:
        raise ContainerCliError(str(e)) from None
    if port is not None:
        # Override BEFORE composition, so --publish and the launcher's --port are both
        # derived from the same value and cannot diverge.
        profile = profile.model_copy(update={"port": port})
    if profile_name is None and len(spec.serve_profiles) > 1:
        console.note(
            f"profile {profile.name!r} (the author's default; "
            f"others: {', '.join(n for n in spec.profile_names() if n != profile.name)})",
            marker="•",
        )

    # An image can go missing between package and serve — `docker image prune`, or a
    # manifest edit that moved the tag. Without this, docker tries to PULL
    # "tt-model/<name>:<sha>" from Docker Hub and reports "pull access denied", which
    # names neither the cause nor the fix.
    if not print_only:
        ref = container.image_ref(manifest)
        if not container.image_present(ref):
            layout = (Path(source) / "image") if source else None
            if layout and (layout / "oci-layout").is_file():
                with console.step(f"docker load {ref} (image was not loaded)"):
                    oci.load(layout, expect_tag=ref)
            else:
                raise ContainerCliError(
                    f"image {ref} is not loaded in docker.\n"
                    f"  → if you have the staged package dir:  tt-model serve "
                    f"<dir>/{MANIFEST_NAME}\n"
                    f"  → if it was published:                 tt-model pull {manifest.name}\n"
                    f"  → otherwise rebuild it:                tt-model package --container "
                    f"<manifest.yaml>"
                )

    launcher = launcher_for(spec.kind)
    argv = launcher.serve_argv(manifest, profile) + list(extra_args or [])
    env = launcher.serve_env(manifest, profile)
    run_argv = container.compose_run(manifest, profile, argv, env, detach=not print_only)

    if print_only:
        console.raw(" ".join(run_argv))
        return

    name = container.container_name(manifest, profile)
    what = target or manifest.name
    if container.is_running(name):
        raise ContainerCliError(
            f"{name} is already running. Stop it first:  tt-model stop {what}"
        )
    if container.container_exists(name):
        # Not running, but holding the name — `docker run` creates the container before
        # it binds ports, so a failed start (a busy port, usually) leaves one in
        # "Created". Refusing here would make the obvious retry impossible.
        with console.step(f"removing a stopped {name}"):
            container.remove(name, force=True)

    # As the host user, so the daemon does not create them as root: see
    # container.ensure_mount_sources.
    container.ensure_mount_sources(manifest)

    try:
        with console.step(f"starting {name}"):
            container.run_checked(run_argv)
    except container.ContainerError:
        # Leave no half-created container behind to block the next attempt.
        if container.container_exists(name):
            container.remove(name, force=True)
        raise

    port = profile.port or 8000
    console.milestone(f"{name} started")
    if follow:
        console.note("waiting for the server to become ready (a cold boot JIT-compiles "
                     "kernels; ~10 min the first time)", marker="○")
        ready = container.wait_ready(name, launcher.ready_probe(manifest),
                                     echo=console.raw)
        if not ready:
            raise ContainerCliError(
                f"the server did not report ready. Logs:  tt-model logs {what} -f"
            )
        console.milestone(f"ready at http://127.0.0.1:{port}")
    else:
        console.note(f"endpoint (once ready):  http://127.0.0.1:{port}", marker="→")
        console.note(f"follow the boot:        tt-model logs {what} -f", marker="→")


# ------------------------------------------------------------------------ stop / logs


def stop_container(manifest: Manifest, *, profile_name: Optional[str] = None) -> None:
    spec = manifest.container
    assert spec is not None
    names = ([container.container_name(manifest, spec.resolve_profile(profile_name))]
             if profile_name else
             [container.container_name(manifest, spec.resolve_profile(n))
              for n in spec.profile_names()])

    stopped = 0
    for name in names:
        if not container.running(name):
            continue
        stopped += 1
        with console.step(f"stopping {name}") as st:
            clean = container.stop(name, image=container.image_ref(manifest))
            st.detail("clean shutdown" if clean else "killed — mesh reset")
        if not clean:
            console.note(
                "the server did not exit on SIGTERM, so the mesh was left dirty and has "
                "been reset with tt-smi; the next boot is safe",
                marker="⚠", style="warning",
            )
    if not stopped:
        console.note("nothing running", marker="○")
    else:
        console.milestone(f"stopped {stopped} container(s)")


def logs_container(manifest: Manifest, *, profile_name: Optional[str] = None,
                   follow: bool = False, target: Optional[str] = None) -> int:
    spec = manifest.container
    assert spec is not None
    for n in ([profile_name] if profile_name else spec.profile_names()):
        name = container.container_name(manifest, spec.resolve_profile(n))
        if container.running(name):
            return container.logs(name, follow=follow)
    what = target or manifest.name
    raise ContainerCliError(
        f"no running container for {manifest.name}. Start it:  tt-model serve {what}"
    )


# ------------------------------------------------------------------------------ list


def list_containers(manifest: Optional[Manifest] = None) -> None:
    """With a manifest: that model's profiles. Without: what is on this host."""
    if manifest is not None:
        spec = manifest.container
        assert spec is not None
        default = spec.resolved_default()
        table = console.check_table()
        for n in spec.profile_names():
            p = spec.resolve_profile(n)
            console.check_row(
                table, "•", n + (" (default)" if n == default else ""),
                found=f"{p.hardware or '?'} · {p.mesh_device or '?'}",
                need=f"seqs {p.max_num_seqs} · len {p.max_model_len or '-'}",
            )
        console.print_table(table)
        return

    images = container.images()
    running = container.running()
    if not images and not running:
        console.note("no container packages on this host", marker="○")
        return
    table = console.check_table()
    for row in images:
        console.check_row(table, "•", row["image"], found=row["size"], need=row["created"])
    for row in running:
        console.check_row(table, "▶", row["name"], found=row["status"], need=row["ports"])
    console.print_table(table)
