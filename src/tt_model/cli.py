# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The tt-model CLI: seven commands, no business logic.

package  build a model into a self-contained Docker image (one YAML in, image out)
push     upload image + manifest + code to a Hugging Face repo
pull     download from HF: image -> docker, weights -> the host HF cache
serve    docker run with the right devices, mounts, and serve profile
stop     SIGTERM-first stop; resets the mesh only if a hard kill was needed
logs     the running server's logs
list     local tt-model images/containers, and a manifest's serve profiles
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

from . import MANIFEST_NAME, __version__, console
from .manifest import Manifest, ManifestError, load_manifest

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Package Tenstorrent models as Docker images and publish them on Hugging Face.",
)


def _err(msg: str) -> "typer.Exit":
    console.note(msg, marker="✗", style="error")
    return typer.Exit(code=1)


def _fail_card(name: str, diagnosis: dict) -> None:
    console._body_print(console.failure_card(name, diagnosis))


def _version_cb(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full detail."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable styled output."),
    version: bool = typer.Option(False, "--version", callback=_version_cb, is_eager=True),
):
    console.set_verbose(verbose)
    if no_color:
        console.set_no_color(True)


# ------------------------------------------------------------------- local state
def _pull_dir(repo_id: str) -> Path:
    return Path.home() / ".cache" / "tt-model" / "pulled" / repo_id.replace("/", "__")


def _load_pulled_manifest(repo_id: str) -> Optional[Manifest]:
    p = _pull_dir(repo_id) / MANIFEST_NAME
    if p.exists():
        return load_manifest(p)
    return None


def _resolve_manifest(target: str) -> Manifest:
    """A command target is either a local manifest path or a pulled org/name."""
    if target.endswith((".yaml", ".yml")) or Path(target).exists():
        return load_manifest(target)
    m = _load_pulled_manifest(target)
    if m is None:
        raise _err(
            f"{target} is neither a manifest file nor a pulled model — run "
            f"`tt-model pull {target}` first"
        )
    return m


# ----------------------------------------------------------------------- package
@app.command(rich_help_panel="Publish models")
def package(
    manifest_path: Path = typer.Argument(..., help="The model's tt-model.yaml. The only input."),
    out: Optional[Path] = typer.Option(None, "--out", help="Staging root (default ./build)."),
    quiet: bool = typer.Option(False, "--quiet", help="Collapse build output (CI)."),
):
    """Build the model into a Docker image + a staged HF repo directory.

    Output streams here live and tees to a log (the tail command prints up front).
    A first Ctrl-C on a TTY only warns; a second within 10s cancels and cleans up,
    keeping the caches that make a re-run cheap.
    """
    from . import build

    try:
        staged = build.stage(manifest_path, out_root=out)
    except (ManifestError, build.BuildError) as e:
        raise _err(str(e))

    ms = staged.metal
    console.note(f"tt-metal: {ms.mode} @ {(ms.sha or 'unknown')[:12]}"
                 + (" (dirty)" if ms.dirty else ""), marker="•")
    if ms.dirty:
        console.note("the checkout has uncommitted changes — they WILL be packaged "
                     "(hermetic by design); provenance records dirty=true", marker="⚠", style="warning")

    try:
        build.run_build(staged, quiet=quiet)
        out_dir = build.finalize(staged)
    except KeyboardInterrupt as e:
        # partial staging is removed; caches and log are kept
        import shutil
        shutil.rmtree(staged.out, ignore_errors=True)
        console.note(str(e), marker="✗", style="error")
        console.note(f"resume:  tt-model package {manifest_path}", marker="→")
        raise typer.Exit(code=130)
    except build.BuildError as e:
        raise _err(str(e))

    console.milestone(f"built {staged.image}")
    console.milestone(f"staged {out_dir}")
    console.note(f"next:  tt-model push {out_dir}", marker="→")
    console.note(f"or serve the local image:  tt-model serve {out_dir / MANIFEST_NAME}", marker="→")


# -------------------------------------------------------------------------- push
@app.command(rich_help_panel="Publish models")
def push(
    staged: Path = typer.Argument(..., help="A directory produced by `tt-model package`."),
    private: Optional[bool] = typer.Option(
        None, "--private/--public",
        help="Repo visibility. Omitted: a NEW repo is created private; an existing "
             "repo's visibility is never touched.",
    ),
):
    """Upload the staged package to its Hugging Face repo (from the manifest)."""
    from . import hub

    mpath = staged / MANIFEST_NAME
    if not mpath.exists():
        raise _err(f"{staged} has no {MANIFEST_NAME} — is this a `tt-model package` output?")
    m = load_manifest(mpath)

    try:
        url = hub.ensure_repo(m.repo, private)
        console.note(f"repo: {url}", marker="•")
        hub.push_package(m.repo, staged)
        tags = sorted({m.arch, m.type, *(p.hardware for p in
                       (m.resolve_profile(n) for n in m.profile_names()) if p.hardware)})
        hub.tag_repo(m.repo, ["tt-model", *tags])
    except Exception as e:  # noqa: BLE001 — hub errors get one diagnosis card
        _fail_card("push", hub.classify_hub_error(e, m.repo))
        raise typer.Exit(code=1)

    console.milestone(f"pushed {m.repo}")
    console.note(f"consumers:  tt-model pull {m.repo}", marker="→")


# -------------------------------------------------------------------------- pull
@app.command(rich_help_panel="Run models")
def pull(
    repo_id: str = typer.Argument(..., help="HF repo id, e.g. org/name."),
    no_weights: bool = typer.Option(False, "--no-weights", help="Skip the weights download."),
):
    """Download a model: image -> docker load, weights -> the host HF cache.

    The image blobs land in the HF cache too, so a re-pull is a no-op and blobs shared
    between models are stored once.
    """
    from . import hub, oci

    try:
        snapshot = hub.download_package(repo_id)
    except Exception as e:  # noqa: BLE001
        _fail_card("pull", hub.classify_hub_error(e, repo_id))
        raise typer.Exit(code=1)

    m = load_manifest(snapshot / MANIFEST_NAME)

    # keep a stable pointer for serve/list
    dest = _pull_dir(repo_id)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / MANIFEST_NAME).write_text((snapshot / MANIFEST_NAME).read_text())

    from .container import image_tag
    tag = image_tag(m)
    have = subprocess.run(["docker", "image", "inspect", tag],
                          capture_output=True).returncode == 0
    if have:
        console.note(f"image {tag} already loaded", marker="•")
    else:
        with console.step(f"docker load {tag}"):
            oci.load(snapshot / "image", expect_tag=tag)

    if not no_weights:
        try:
            hub.download_weights(m.weights)
        except Exception as e:  # noqa: BLE001
            _fail_card("weights", hub.classify_hub_error(e, m.weights))
            console.note("the image is loaded; weights can also download at first boot "
                         "(slower, inside the container)", marker="⚠", style="warning")

    console.milestone(f"pulled {repo_id}")
    console.note(f"next:  tt-model serve {repo_id}", marker="→")


# ------------------------------------------------------------------------- serve
@app.command(rich_help_panel="Run models")
def serve(
    target: str = typer.Argument(..., help="org/name (pulled) or a manifest path."),
    profile: Optional[str] = typer.Option(None, "--profile", help="Serve profile (default: the manifest's default_profile)."),
    print_only: bool = typer.Option(False, "--print", help="Print the docker run command instead of running."),
    follow: bool = typer.Option(False, "--follow", help="Stream logs until the server is ready."),
    force: bool = typer.Option(False, "--force", help="Proceed despite hardware/exclusivity warnings."),
):
    """Launch the model's container with the right devices, mounts, and profile."""
    from . import container, hardware
    from .types import TYPES

    m = _resolve_manifest(target)

    try:
        prof = m.resolve_profile(profile)
    except ManifestError as e:
        raise _err(str(e))
    if profile is None:
        console.note(f"profile: {prof.name} (default)", marker="•")
    else:
        console.note(f"profile: {prof.name}", marker="•")

    # hardware sanity: warn + suggest, never silently substitute
    host = hardware.detect()
    if host is not None and not hardware.profile_fits(prof, host, m.arch):
        fits = hardware.fitting_profiles(m, host)
        hint = f"; profile {fits[0]!r} matches — rerun with --profile {fits[0]}" if fits else ""
        msg = (f"detected {host.chips} {host.arch or 'unknown-arch'} chip(s), but profile "
               f"{prof.name!r} targets {prof.hardware}{hint}")
        if not force:
            raise _err(msg + "  (--force to launch anyway)")
        console.note(msg, marker="⚠", style="warning")

    mtype = TYPES[m.type]
    argv = container.compose_run(m, prof, mtype.serve_argv(m, prof), mtype.serve_env(m, prof))

    if print_only:
        typer.echo(" ".join(shlex.quote(a) for a in argv))
        raise typer.Exit()

    # /dev/tenstorrent is exclusive: refuse while any tt-model container runs
    live = [r for r in container.running() if "Up" in r["status"]]
    if live and not force:
        names = ", ".join(r["name"] for r in live)
        raise _err(f"a tt-model container is already running ({names}) — the devices are "
                   f"exclusive. `tt-model stop` it first, or --force if you know better.")

    container.model_cache_dir(m).mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "rm", container.container_name(m, prof)],
                   capture_output=True)  # a stopped leftover with the same name
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise _err(f"docker run failed:\n{r.stderr.strip()}")

    name = container.container_name(m, prof)
    port = prof.port or 8000
    console.milestone(f"started {name}")
    console.note(f"endpoint (once ready):  http://localhost:{port}/v1", marker="→")
    console.note(f"watch the boot (~10 min):  tt-model logs {m.repo} --follow", marker="→")
    console.note(f"  or:  docker logs -f {name}", marker=" ")

    if follow:
        ok = container.wait_ready(name, mtype.ready_probe(m))
        if ok:
            console.milestone(f"ready: http://localhost:{port}/v1")
        else:
            raise _err("the server did not report ready — see the logs above")


# -------------------------------------------------------------------------- stop
@app.command(rich_help_panel="Run models")
def stop(
    target: str = typer.Argument(..., help="org/name, a manifest path, or a container name."),
):
    """Stop the model's container. SIGTERM-first with a generous grace period; the
    mesh is reset (tt-smi -r all, in a throwaway container) only if a hard kill was
    unavoidable — a clean shutdown closes the mesh itself."""
    from . import container

    rows = container.running(name_filter=None)
    if not rows:
        raise _err("no tt-model containers found")

    # match by container name, or by the model the target resolves to
    wanted = [r for r in rows if r["name"] == target]
    if not wanted:
        try:
            m = _resolve_manifest(target)
            prefix = f"tt-model-{m.name}-"
            wanted = [r for r in rows if r["name"].startswith(prefix)]
        except SystemExit:
            raise
        except Exception:
            wanted = []
    if not wanted:
        names = ", ".join(r["name"] for r in rows)
        raise _err(f"nothing matching {target!r}; containers: {names}")

    for r in wanted:
        with console.step(f"stopping {r['name']} (SIGTERM, up to {container.STOP_TIMEOUT_S}s)"):
            clean = container.stop(r["name"], image=r["image"])
        if clean:
            console.milestone(f"{r['name']}: clean shutdown — mesh closed by the server")
        else:
            console.note(f"{r['name']}: had to be killed; mesh was reset (tt-smi -r all)",
                         marker="⚠", style="warning")


# -------------------------------------------------------------------------- logs
@app.command(rich_help_panel="Run models")
def logs(
    target: str = typer.Argument(..., help="org/name, a manifest path, or a container name."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream."),
):
    """Logs of the model's (running or last) container."""
    from . import container

    rows = container.running()
    match = [r for r in rows if r["name"] == target]
    if not match:
        try:
            m = _resolve_manifest(target)
            prefix = f"tt-model-{m.name}-"
            match = [r for r in rows if r["name"].startswith(prefix)]
        except SystemExit:
            raise
        except Exception:
            match = []
    if not match:
        raise _err(f"no container found for {target!r}")
    raise typer.Exit(code=container.logs(match[0]["name"], follow=follow))


# -------------------------------------------------------------------------- list
@app.command("list", rich_help_panel="Run models")
def list_(
    target: Optional[str] = typer.Argument(None, help="Optional org/name or manifest path: show its serve profiles."),
):
    """Local tt-model images and containers; with a target, its serve profiles."""
    from . import container, hardware

    if target:
        m = _resolve_manifest(target)
        host = hardware.detect()
        default = m.resolved_default()
        typer.echo(f"{m.name}  ({m.repo})  type={m.type}  arch={m.arch}")
        for name in m.profile_names():
            p = m.resolve_profile(name)
            fits = ""
            if host is not None:
                fits = "  [fits this machine]" if hardware.profile_fits(p, host, m.arch) else "  [does NOT fit this machine]"
            star = "*" if name == default else " "
            typer.echo(f"  {star} {name:24s} {p.hardware or '':10s} mesh={p.mesh_device}"
                       f" seqs={p.max_num_seqs}{fits}")
            if p.description:
                typer.echo(f"      {p.description}")
        typer.echo("  (* = default profile)")
        return

    images = container.images()
    rows = container.running()
    if not images and not rows:
        console.note("no tt-model images or containers on this machine", marker="○")
        console.note("tt-model pull <org/name>   # or: tt-model package <manifest.yaml>",
                     marker="→")
        return
    if images:
        console.note("images:", marker="•", style="accent")
        for i in images:
            typer.echo(f"  {i['image']:48s} {i['size']:>10s}  {i['created']}")
    if rows:
        console.note("containers:", marker="•", style="accent")
        for r in rows:
            typer.echo(f"  {r['name']:48s} {r['status']}  {r['ports']}")


def main():
    app()


if __name__ == "__main__":
    main()
