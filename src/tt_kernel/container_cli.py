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
        + (" (dirty tree — packaged as-is)" if metal.dirty else "")
        + f" · {metal.mode} source",
        marker="•",
    )
    for key in ("vllm", "plugin"):
        pinned = staged.built.get(key)
        if isinstance(pinned, dict):
            console.note(f"{key} pinned to {str(pinned.get('sha'))[:9]}", marker="•")

    console.phase("Stage")
    console.note(f"{len(staged.code_tree)} code path(s) → code/", marker="•")

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


# ------------------------------------------------------------------------------ pull


def pull_dir(repo_id: str) -> Path:
    """Where a pulled package's manifest is kept, so serve/stop can find it by id."""
    return Path.home() / ".cache" / "tt-model" / "pulled" / repo_id.replace("/", "__")


def pull_container(repo_id: str, revision: Optional[str], manifest: Manifest, *,
                   no_weights: bool = False) -> None:
    """Snapshot the repo, load the image into docker, and put weights in the HOST cache."""
    spec = manifest.container
    assert spec is not None

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
                path = _download_weights(manifest.weights.repo_id)
                st.detail(str(path))
        except Exception as e:  # noqa: BLE001
            console.note(
                f"weights not downloaded ({e.__class__.__name__}); the image is loaded and "
                "the model will fetch them at first boot (slower, inside the container)",
                marker="⚠", style="warning",
            )

    console.milestone(f"pulled {repo_id}")
    console.note(f"next:  tt-model serve {repo_id}", marker="→")


def _download_weights(weights_repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=weights_repo))


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
                    extra_args: Optional[List[str]] = None) -> None:
    spec = manifest.container
    assert spec is not None

    try:
        profile = spec.resolve_profile(profile_name)
    except ValueError as e:
        raise ContainerCliError(str(e)) from None
    if profile_name is None and len(spec.serve_profiles) > 1:
        console.note(
            f"profile {profile.name!r} (the author's default; "
            f"others: {', '.join(n for n in spec.profile_names() if n != profile.name)})",
            marker="•",
        )

    launcher = launcher_for(spec.kind)
    argv = launcher.serve_argv(manifest, profile) + list(extra_args or [])
    env = launcher.serve_env(manifest, profile)
    run_argv = container.compose_run(manifest, profile, argv, env, detach=not print_only)

    if print_only:
        console.raw(" ".join(run_argv))
        return

    name = container.container_name(manifest, profile)
    existing = container.running(name)
    if existing:
        raise ContainerCliError(
            f"{name} already exists ({existing[0]['status']}). "
            f"Stop it first:  tt-model stop {manifest.name}"
        )

    with console.step(f"starting {name}"):
        container.run_checked(run_argv)

    port = profile.port or 8000
    console.milestone(f"{name} started")
    if follow:
        console.note("waiting for the server to become ready (a cold boot JIT-compiles "
                     "kernels; ~10 min the first time)", marker="○")
        ready = container.wait_ready(name, launcher.ready_probe(manifest),
                                     echo=console.raw)
        if not ready:
            raise ContainerCliError(
                f"the server did not report ready. Logs:  tt-model logs {manifest.name} -f"
            )
        console.milestone(f"ready at http://127.0.0.1:{port}")
    else:
        console.note(f"endpoint (once ready):  http://127.0.0.1:{port}", marker="→")
        console.note(f"follow the boot:        tt-model logs {manifest.name} -f", marker="→")


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
                   follow: bool = False) -> int:
    spec = manifest.container
    assert spec is not None
    for n in ([profile_name] if profile_name else spec.profile_names()):
        name = container.container_name(manifest, spec.resolve_profile(n))
        if container.running(name):
            return container.logs(name, follow=follow)
    raise ContainerCliError(
        f"no running container for {manifest.name}. Start it:  tt-model serve {manifest.name}"
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
