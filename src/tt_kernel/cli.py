# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``tt-model`` command-line interface."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import click
import typer
from typer.core import TyperGroup

from . import console
from . import MANIFEST_NAME, TT_MODEL_CATALOG_TAG, TT_MODEL_TAG, __version__
from . import (
    auth, compat, container, hub, localdb, metal,
    packaging, runtime,
)
from .manifest import (
    CompatibilityReport,
    Manifest,
    Mesh,
    Resources,
    WeightsRef,
    compare,
)

def _did_you_mean(argv: List[str]) -> Optional[str]:
    """Suggest a corrected command line for a slipped word.

    `tt-model pull serve <id>` produced a bare "Got unexpected extra argument(s)". Both
    tokens are real command names and the trailing one looks like a repo id, so the intent
    is recoverable — and a CLI that can name the fix should.

    Pure (argv in, string or None out) so the matrix is testable without invoking Click.
    """
    commands = {c.name for c in app.registered_commands if c.name} | {
        (c.name or (c.callback.__name__ if c.callback else None))
        for c in app.registered_commands
    }
    commands = {c.replace("_", "-") for c in commands if c}
    # Click passes us the args after the program name.
    words = [a for a in argv if not a.startswith("-")]
    if len(words) < 3:
        return None
    first, second = words[0], words[1]
    if first in commands and second in commands and first != second:
        # Two command names in a row: the user typed one too many. The later one usually
        # carries the intent (`pull serve X` reads as "serve X"), and the rest follows it.
        rest = " ".join(words[2:])
        return f"tt-model {second} {rest}".strip()
    return None


def _usage_error_classes():
    """Every UsageError class that can reach us.

    Typer (>=0.16) vendors its own click fork, so a usage error raised while parsing a
    subcommand is `typer._click.exceptions.UsageError` and is NOT a subclass of
    `click.UsageError`. Catching only the latter silently matched nothing. Collect both,
    tolerating either being absent so a Typer/click upgrade degrades to "no hint" rather
    than an import error.
    """
    classes = [click.UsageError]
    try:  # pragma: no cover - depends on the installed Typer layout
        from typer._click.exceptions import UsageError as TyperUsageError

        classes.append(TyperUsageError)
    except Exception:  # noqa: BLE001
        pass
    return tuple(dict.fromkeys(classes))


_USAGE_ERRORS = _usage_error_classes()


class _SuggestingGroup(TyperGroup):
    """Adds a "Did you mean" line to Click's usage errors where we can infer one.

    Both hooks are needed: an unknown command fails in resolve_command, while
    `pull serve <id>` resolves `pull` fine and then fails on extra arguments inside
    make_context, which happens under invoke.
    """

    def _augment(self, exc, args):
        hint = _did_you_mean([a for a in args])
        if hint and "Did you mean" not in (exc.message or ""):
            exc.message = f"{exc.message}\n\nDid you mean:  {hint}"
        return exc

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except _USAGE_ERRORS as exc:
            raise self._augment(exc, list(args))

    def invoke(self, ctx):
        # Snapshot the argv BEFORE delegating: TyperGroup.invoke does `ctx.args = []` and
        # `ctx._protected_args = []` before the make_context call that raises, so reading
        # them from the except block yields nothing.
        argv = [*getattr(ctx, "_protected_args", []), *ctx.args]
        try:
            return super().invoke(ctx)
        except _USAGE_ERRORS as exc:
            raise self._augment(exc, argv)


app = typer.Typer(
    name="tt-model",
    help="Publish and pull self-contained tt-metal model bundles over Hugging Face Hub.",
    no_args_is_help=True,
    add_completion=False,
    # Typer enables these by default, which turned every unhandled exception into a
    # ~60-line syntax-highlighted stack with source frames from httpx and
    # huggingface_hub. A traceback is not a user-facing error message; the handlers
    # below render a diagnosis card instead. `--verbose` puts the traceback back.
    pretty_exceptions_enable=False,
    cls=_SuggestingGroup,
)


def _version_cb(value: bool) -> None:
    if value:
        console.raw(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show full per-step output instead of the collapsed summary (and restore tracebacks).",
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colour and styling."),
    version: bool = typer.Option(
        False, "--version", callback=_version_cb, is_eager=True,
        help="Print the tt-model version and exit.",
    ),
) -> None:
    """Publish, pull, and serve precompiled tt-metal model bundles."""
    console.set_verbose(verbose)
    if no_color:
        console.set_no_color(True)
        # Click styles independently of our console module, and every not-yet-converted
        # `typer.secho` goes through it. ctx.color is the documented way to force it off
        # on a TTY (NO_COLOR alone does not reach click's resolve_color_default).
        ctx.color = False
        os.environ.setdefault("NO_COLOR", "1")


def _err(msg: str) -> "typer.Exit":
    typer.secho(msg, fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _fail_card(name: str, diagnosis: dict, *, consequence: Optional[str] = None) -> "typer.Exit":
    """Render a diagnosis dict as a card and return the Exit to raise.

    Mirrors ``_err``'s contract (print, return an Exit) so call sites read the same, but
    surfaces cause/evidence/consequence/next-steps instead of one red line. The diagnosis
    itself is built by a pure classifier (e.g. ``hub.classify_hub_error``) so the wording
    matrix is testable without a network.
    """
    console.console.print(
        console.failure_card(name, diagnosis, consequence=consequence), style=None
    )
    return typer.Exit(code=1)


def _hub(op, repo_id: str, *, what: str, consequence: Optional[str] = None):
    """Run a Hub call, turning any failure into a diagnosis card instead of a traceback.

    Every Hub entry point used to be unguarded, so a 404 escaped as a Rich stack — from
    ``pull`` (download_bundle) and ``info`` (fetch_manifest) alike. Wrapping at the call
    site rather than inside ``hub`` keeps that module free of CLI concerns and lets each
    caller name what it was doing and whether the run can continue.
    """
    try:
        return op()
    except typer.Exit:
        raise
    except BaseException as exc:  # noqa: BLE001 — classified and re-raised as an Exit
        if console.is_verbose():
            raise
        raise _fail_card(what, hub.classify_hub_error(exc, repo_id), consequence=consequence)


def _routine(msg: str, fg: str = typer.colors.GREEN) -> None:
    """A confirmation worth printing on its own, but not inside a phase.

    Within a phase the collapsed `✓ Phase k/N` line IS the confirmation, so these fold;
    `--verbose` brings them back, and a standalone `pull`/`info` (no phase) still prints
    them. Failures and actionable warnings must never route through here.
    """
    if console.show_detail():
        typer.secho(msg, fg=fg)


def _print_report(report: CompatibilityReport) -> None:
    if report.compatible:
        _routine("✓ compatible with the local environment")
        return
    # Never folded: an incompatibility is why the user is watching, and the next line the
    # install prints is a refusal that only makes sense next to it.
    console.note("compatibility issues", marker="!", style="warning")
    for i in report.issues:
        style = "error" if i.fatal else "warning"
        tag = "FATAL" if i.fatal else "warn"
        console.note(f"[{tag}] {i.field}: bundle={i.expected!r} local={i.detected!r}",
                     marker=" ", style=style)


def _ensure_repo(repo_id: str, private: Optional[bool], *, publish: bool = False) -> None:
    """Make sure ``repo_id`` exists, without ever changing visibility as a side effect.

    ``private`` is deliberately tri-state:

    - ``None``  — the user said nothing. A push is a *content* operation, so an existing
      repo keeps whatever visibility it has. (Previously the flag was a plain ``bool``
      defaulting to ``False`` and ``set_visibility`` ran on every push, so a bare
      ``tt-kernel push you/private-model`` silently published a private repo.)
    - ``True`` / ``False`` — the user passed ``--private`` / ``--public`` and means it. We
      honour it and print what changed, so a visibility flip is never invisible.

    At *creation* time there is no prior state to preserve, so the flag (or the documented
    public default) simply becomes the new repo's visibility.
    """
    if not hub.repo_exists(repo_id):
        typer.echo(f"Creating repo {repo_id} ({'private' if private else 'public'})")
        hub.create_repo(repo_id, private=bool(private))
        return

    # The repo already exists and belongs to whoever set its visibility.
    if private is None:
        # A catalog listing is public by definition. Rather than quietly flipping the repo
        # (the exact bug this function exists to prevent), make the user say --public.
        if publish and _is_private_or_unknown(repo_id):
            raise _err(
                f"{repo_id} already exists and is private, but --publish lists it in the "
                "public community catalog. Re-run with --public to make it public (tt-kernel "
                "will not change visibility unless you ask), or drop --publish to push privately."
            )
        typer.echo(f"Repo {repo_id} exists; leaving its visibility unchanged")
        return

    want = "private" if private else "public"
    if hub.is_private_safe(repo_id) is private:
        typer.echo(f"Repo {repo_id} exists and is already {want}")
        return
    hub.set_visibility(repo_id, private=private)
    typer.secho(f"! Changed visibility of {repo_id} to {want} (as requested)",
                fg=typer.colors.YELLOW)


def _is_private_or_unknown(repo_id: str) -> bool:
    """True unless we can positively confirm the repo is public.

    Used only to gate ``--publish``: if the Hub will not tell us, err toward asking the user
    instead of assuming a repo is safe to list.
    """
    return hub.is_private_safe(repo_id) is not False


# -------------------------------------------------------------------------- start
def _ensure_portable_wheel(wheel: Path, *, repair: bool, plat: Optional[str] = None) -> Path:
    """Make the ttnn wheel self-contained with auditwheel: vendor its external libs (libtracy/
    libmpi/libhwloc/libnuma/...) and rewrite RPATH to $ORIGIN. Without this the shipped .so's load
    from the author's build tree / host and fail on another machine (the #1 cross-machine bug).
    Skips an already-portable (manylinux-tagged) wheel; --no-repair ships the raw wheel with a loud
    warning.

    ``plat`` optionally forces an auditwheel target policy (e.g. ``manylinux_2_28_x86_64``).
    auditwheel already auto-picks the LOWEST glibc the wheel's symbols permit, so this can only
    ASSERT a floor, not lower one: if the wheel was compiled against a newer glibc than ``plat``
    allows, auditwheel errors — which is the point (fail at package time, on the author's box,
    instead of at every consumer). To actually broaden compatibility, build the ttnn wheel on the
    OLDEST target OS (e.g. Ubuntu 22.04, glibc 2.35); then it repairs to manylinux_2_35 and runs
    on both 22.04 and 24.04."""
    if "manylinux" in wheel.name:
        return wheel  # already repaired/portable
    if not repair:
        typer.secho(
            "  ! --no-repair: shipping the ttnn wheel as-is. Its .so RPATH likely points at your "
            "build tree and external libs aren't vendored, so it will probably fail on another "
            "machine. Only use this if you know the wheel is already portable.",
            fg=typer.colors.YELLOW,
        )
        return wheel
    if importlib.util.find_spec("auditwheel") is None:
        raise _err("auditwheel is required to make the ttnn wheel portable (vendors libtracy/"
                   "libmpi/libnuma/libhwloc and rewrites RPATH to $ORIGIN). Install it with "
                   "`pip install auditwheel patchelf`, or re-run with --no-repair to ship as-is.")
    outdir = Path(tempfile.mkdtemp(prefix="tt-model-repair-"))
    plat_note = f" targeting {plat}" if plat else ""
    typer.echo(f"Repairing ttnn wheel for portability (auditwheel: vendor libs + $ORIGIN RPATH){plat_note} ...")
    cmd = [sys.executable, "-m", "auditwheel", "repair", str(wheel), "-w", str(outdir)]
    if plat:
        cmd += ["--plat", plat]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        hint = ""
        if plat:
            hint = (f" If it reports the wheel requires a newer glibc than {plat} allows, the ttnn "
                    "wheel was built on too-new an OS — rebuild it on the oldest target (e.g. "
                    "Ubuntu 22.04, glibc 2.35) to reach that floor.")
        raise _err(f"auditwheel repair failed (exit {exc.returncode}). The build tree the wheel was "
                   "built from must be present so auditwheel can find the external libs to vendor; "
                   f"or re-run with --no-repair.{hint} See the output above.")
    repaired = sorted(outdir.glob("ttnn-*.whl"))
    if not repaired:
        raise _err("auditwheel produced no wheel; re-run with --no-repair or repair manually.")
    typer.secho(f"  ✓ portable ttnn wheel: {repaired[0].name}", fg=typer.colors.GREEN)
    return repaired[0]


def _vendor_dependencies(bundle_dir: Path, manifest: Manifest) -> None:
    """Download the full dependency closure as wheels into the bundle for offline, reproducible
    install. Runs on the author's box (cp312/linux == the bundle's pinned target); the consumer
    then installs with ``--no-index`` so there is no PyPI/resolver drift or network at install."""
    req = bundle_dir / (manifest.bundled.requirements or "requirements.txt")
    wheels = bundle_dir / packaging.WHEELS_DIR
    if not req.is_file():
        typer.secho("  (no requirements.txt to vendor)", fg=typer.colors.YELLOW)
        return
    typer.echo("Vendoring dependency wheels for offline install (torch/transformers/...) ...")
    # uv has no `pip download` and uv-created venvs ship no pip, so use an ephemeral SEEDED venv
    # (uv venv --seed includes pip), pinned to the bundle's target Python so the downloaded wheels
    # match the consumer's interpreter/platform.
    pyver = (manifest.bundled.python if manifest.bundled and manifest.bundled.python else None)
    dl_env = Path(tempfile.mkdtemp(prefix="tt-model-dl-"))
    try:
        venv_cmd = ["uv", "venv", "--seed"]
        if pyver:
            venv_cmd += ["--python", pyver]
        venv_cmd.append(str(dl_env))
        subprocess.run(venv_cmd, check=True)
        pip = dl_env / "bin" / "pip"
        # Resolve the deps TOGETHER with the shipped platform wheels (ttnn/vLLM/plugin) so the
        # vendored closure is consistent with what actually gets installed — this pulls vLLM's own
        # runtime deps AND resolves version conflicts (e.g. vLLM's pydantic floor) up front, instead
        # of exploding at the consumer's offline install.
        platform_wheels = sorted(str(w) for w in wheels.glob("*.whl"))
        subprocess.run(
            [str(pip), "download", *platform_wheels, "-r", str(req), "-d", str(wheels),
             "--only-binary=:all:", "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise _err(f"dependency vendoring failed (exit {exc.returncode}). "
                   "Re-run with --no-vendor-deps to install deps from the index instead.")
    finally:
        shutil.rmtree(dl_env, ignore_errors=True)


def _classify_wheels(wheels_dir: Path) -> dict:
    """Auto-classify the .whl files in a dir by distribution name prefix.

    "Package what's on your box": the author collects their built wheels in one dir and tt-model
    sorts them — ``ttnn-*`` -> the engine, ``vllm-*`` (not the plugin) -> base vLLM,
    ``vllm_tt_plugin-*`` -> the plugin, everything else -> extras.
    """
    found: dict = {"ttnn": None, "vllm": None, "plugin": None, "extra": []}
    for whl in sorted(wheels_dir.glob("*.whl")):
        low = whl.name.lower()
        if low.startswith("ttnn-"):
            found["ttnn"] = whl
        elif low.startswith(("vllm_tt_plugin-", "vllm-tt-plugin-")):
            found["plugin"] = whl
        elif low.startswith("vllm-"):
            found["vllm"] = whl
        else:
            found["extra"].append(whl)
    return found


@app.command(rich_help_panel="Publish models")
def package(
    repo_id: Optional[str] = typer.Argument(
        None, help="Target HF repo namespace/name to push to. Omit (with --out) to only stage locally."
    ),
    container_manifest: Optional[str] = typer.Option(
        None, "--container", help="Package as a CONTAINER (v5.1): path to a tt-model.yaml. "
        "The whole authoring interface is that one file, so every other flag here is "
        "ignored. The consumer then needs only Docker + a TT card."
    ),
    from_metal: Optional[str] = typer.Option(
        None, "--from-metal", help="Path to your modified tt-metal-community tree (embedded as metal/)."
    ),
    ttnn_wheel: Optional[str] = typer.Option(
        None, "--ttnn-wheel", help="Your built ttnn wheel (custom kernels compiled in). "
        "Required unless --wheels-dir supplies one."
    ),
    vllm_wheel: Optional[str] = typer.Option(None, "--vllm-wheel", help="The empty-target base vLLM wheel."),
    plugin_wheel: Optional[str] = typer.Option(None, "--plugin-wheel", help="The vllm_tt_plugin wheel."),
    extra_wheel: Optional[List[str]] = typer.Option(
        None, "--extra-wheel", help="Additional wheel to ship (repeatable)."
    ),
    wheels_dir: Optional[str] = typer.Option(
        None, "--wheels-dir", help="Dir of .whl files, auto-classified by filename (ttnn-*, vllm-*, "
        "vllm_tt_plugin-*). Explicit --*-wheel flags override the auto-picked one."
    ),
    metadata: Optional[str] = typer.Option(
        None, "--metadata", help="Path to a vllm_metadata.json (arch + main_class) to ship. "
        "Or pass --arch-name and --main-class."
    ),
    arch_name: Optional[str] = typer.Option(
        None, "--arch-name", help="HF architecture name for vllm_metadata (e.g. LlamaForCausalLM)."
    ),
    main_class: Optional[str] = typer.Option(
        None, "--main-class", help="Adapter as module:Class for vllm_metadata "
        "(e.g. generator_vllm:LlamaForCausalLM)."
    ),
    arch: Optional[str] = typer.Option(None, "--arch", help="TT arch (blackhole|wormhole_b0). Detected if omitted."),
    name: Optional[str] = typer.Option(None, "--name", help="Bundle name (defaults to the repo/metal-dir name)."),
    weights: Optional[str] = typer.Option(
        None, "--weights", help="HF model repo id for the weights (a POINTER — weights are not embedded)."
    ),
    weights_revision: Optional[str] = typer.Option(None, "--weights-revision"),
    mesh_topology: Optional[str] = typer.Option(
        None, "--mesh", help="Device topology / MESH_DEVICE (e.g. P150, N300, 1x4)."
    ),
    device_count: int = typer.Option(1, "--device-count", help="Number of devices the model uses."),
    max_num_seqs: Optional[int] = typer.Option(
        None, "--max-num-seqs", help="vLLM max concurrent sequences (TT backend needs a supported "
        "batch; defaults to 32 in the launcher if unset)."
    ),
    block_size: Optional[int] = typer.Option(
        None, "--block-size", help="vLLM paged-attention block size (TT backend requires it; "
        "defaults to 64 in the launcher if unset)."
    ),
    max_model_len: Optional[int] = typer.Option(
        None, "--max-model-len", help="vLLM max context length (bounds KV-cache allocation)."
    ),
    env: Optional[List[str]] = typer.Option(
        None, "--env", help="KEY=VALUE serving env, overlaid at run time (repeatable)."
    ),
    firmware_min: Optional[str] = typer.Option(None, "--firmware-min", help="Minimum card firmware/driver version."),
    vendor_deps: bool = typer.Option(
        True, "--vendor-deps/--no-vendor-deps", help="Vendor the full dependency closure "
        "(torch/transformers/... as wheels) into the bundle so install is offline + reproducible "
        "across machines. --no-vendor-deps installs deps from the CPU index instead (smaller, "
        "needs network, less reproducible)."
    ),
    python_version: Optional[str] = typer.Option(
        None, "--python", help="Pinned interpreter (major.minor) uv provisions on the consumer "
        "(default: derived from the ttnn wheel tag, e.g. cp312 -> 3.12)."
    ),
    repair_wheel: bool = typer.Option(
        True, "--repair/--no-repair", help="Run auditwheel on the ttnn wheel so it is portable "
        "(vendors external libs + $ORIGIN RPATH). --no-repair ships the raw wheel (likely fails "
        "on another machine)."
    ),
    manylinux: Optional[str] = typer.Option(
        None, "--manylinux", help="auditwheel target policy, e.g. manylinux_2_28_x86_64. Asserts a "
        "glibc floor: repair fails if the wheel needs a newer glibc. To serve older hosts (Ubuntu "
        "22.04 = glibc 2.35 as well as 24.04), build the ttnn wheel on the OLDEST target OS. Default: "
        "auditwheel auto-picks the lowest glibc the wheel allows (= the build host's glibc)."
    ),
    out: Optional[str] = typer.Option(
        None, "--out", help="Stage the running folder here (kept even without a push target)."
    ),
    private: bool = typer.Option(True, "--private/--public", help="Repo visibility when pushing."),
    publish: bool = typer.Option(False, "--publish", help="List in the community catalog (requires --public)."),
) -> None:
    """Package what's on your box into ONE self-contained (v5) bundle and (optionally) push it.

    Snapshots your *built* artifacts — your ttnn wheel (custom C++/LLK kernels compiled in), the
    empty-target base vLLM wheel, the vLLM plugin wheel — plus your modified tt-metal-community
    tree, and writes a generated ``install.sh``/``run.sh`` + a v5 manifest. Weights are a POINTER
    (the HF repo id in ``--weights``), never embedded. A consumer then needs only a TT card +
    firmware: ``tt-model pull`` installs the wheels + weights, ``tt-model serve`` runs it.
    """
    # --- container (v5.1): a wholly separate path. Everything below is v5 and untouched.
    if container_manifest:
        from . import container_cli

        try:
            container_cli.package_container(container_manifest, out_root=out)
        except container_cli.ContainerCliError as e:
            raise _err(str(e))
        return

    if not from_metal:
        raise _err(
            "--from-metal is required to package a v5 bundle "
            "(or pass --container <tt-model.yaml> to build a container package instead)."
        )

    if publish and private:
        raise _err("--publish requires --public (a catalog listing is public by definition).")
    if repo_id is None and not out:
        raise _err("Nothing to do: pass a target repo_id to push, or --out to stage locally.")

    metal_dir = Path(from_metal).expanduser()
    if not metal_dir.is_dir():
        raise _err(f"--from-metal {from_metal!r} is not a directory.")

    # Resolve wheels: --wheels-dir auto-classify, then explicit flags override.
    picked = {"ttnn": None, "vllm": None, "plugin": None, "extra": []}
    if wheels_dir:
        wd = Path(wheels_dir).expanduser()
        if not wd.is_dir():
            raise _err(f"--wheels-dir {wheels_dir!r} is not a directory.")
        picked = _classify_wheels(wd)
    if ttnn_wheel:
        picked["ttnn"] = Path(ttnn_wheel).expanduser()
    if vllm_wheel:
        picked["vllm"] = Path(vllm_wheel).expanduser()
    if plugin_wheel:
        picked["plugin"] = Path(plugin_wheel).expanduser()
    if extra_wheel:
        picked["extra"] = [Path(p).expanduser() for p in extra_wheel]

    if not picked["ttnn"]:
        raise _err("No ttnn wheel found. Pass --ttnn-wheel <path> (your built engine wheel with "
                   "your kernels), or --wheels-dir <dir> containing a ttnn-*.whl.")
    for label, p in (("ttnn", picked["ttnn"]), ("vllm", picked["vllm"]), ("plugin", picked["plugin"])):
        if p is not None and not p.is_file():
            raise _err(f"{label} wheel {str(p)!r} does not exist.")
    picked["ttnn"] = _ensure_portable_wheel(picked["ttnn"], repair=repair_wheel, plat=manylinux)

    # vllm_metadata: an authored file, or synthesized from --arch-name/--main-class.
    if metadata:
        vmeta = json.loads(Path(metadata).expanduser().read_text())
        if not vmeta.get("arch") or not vmeta.get("main_class"):
            raise _err(f"{metadata} must set both 'arch' and 'main_class'.")
    elif arch_name and main_class:
        vmeta = {"arch": arch_name, "main_class": main_class}
    else:
        raise _err("Provide the serving entrypoint: --metadata <vllm_metadata.json>, or both "
                   "--arch-name and --main-class.")

    # TT arch (for the compiled-kernel gate) — explicit or detected.
    resolved_arch = arch or metal.detect_device(arch_override=arch).arch
    if not resolved_arch:
        raise _err("Could not detect arch. Pass --arch (blackhole | wormhole_b0 | ...).")

    weights_block = WeightsRef(repo_id=weights, revision=weights_revision) if weights else None
    env_map: dict = {}
    for kv in env or []:
        if "=" not in kv:
            raise _err(f"--env expects KEY=VALUE, got {kv!r}.")
        k, v = kv.split("=", 1)
        env_map[k] = v
    mesh = Mesh(devices=device_count, topology=mesh_topology) if mesh_topology else None
    resources = Resources(
        max_num_seqs=max_num_seqs, block_size=block_size, max_model_len=max_model_len
    ) if (max_num_seqs or block_size or max_model_len) else None
    bundle_name = name or (repo_id.split("/")[-1] if repo_id else metal_dir.name)

    def _stage(staged: Path) -> Manifest:
        return packaging.stage_package(
            staged,
            name=bundle_name,
            arch=resolved_arch,
            ttnn_wheel=picked["ttnn"],
            vllm_wheel=picked["vllm"],
            plugin_wheel=picked["plugin"],
            extra_wheels=picked["extra"],
            metal_dir=metal_dir,
            vllm_metadata=vmeta,
            tt_kernel_version=__version__,
            weights=weights_block,
            device_count=device_count,
            mesh=mesh,
            env=env_map,
            resources=resources,
            tt_metal_version=metal.resolve_version() or "unknown",
            firmware_min=firmware_min,
            python_version=python_version,
            deps_vendored=vendor_deps,
        )

    def _report(m: Manifest, where: Path) -> None:
        typer.secho(f"✓ Staged self-contained bundle {m.name} at {where}", fg=typer.colors.GREEN)
        typer.echo(f"  wheels: {', '.join(Path(w.path).name for w in m.bundled.wheels)}")
        typer.echo(f"  arch registration: {m.entrypoint.arch_name}  ->  {m.entrypoint.cls}")
        if m.weights:
            typer.echo(f"  weights (pointer): {m.weights.repo_id}")

    if out:
        upload_from = Path(out).expanduser()
        if upload_from.exists():
            shutil.rmtree(upload_from)
    else:
        upload_from = Path(tempfile.mkdtemp(prefix="tt-model-pkg-")) / "bundle"
    manifest = _stage(upload_from)
    if vendor_deps:
        _vendor_dependencies(upload_from, manifest)
    _report(manifest, upload_from)
    if repo_id is None:
        typer.secho("  (no push target — staged only)", fg=typer.colors.CYAN)
        return

    # Push (git-LFS handles the large wheels automatically).
    tags = [TT_MODEL_TAG, manifest.arch, "vllm", "self-contained"]
    if mesh_topology:
        tags.append(mesh_topology.lower())
    if publish:
        tags.append(TT_MODEL_CATALOG_TAG)
    typer.echo(f"Creating repo {repo_id} (private={private})")
    hub.create_repo(repo_id, private=private)
    hub.set_visibility(repo_id, private=private)
    total_mb = sum(w.size for w in manifest.bundled.wheels) / 1e6
    typer.echo(f"Uploading bundle (~{total_mb:.0f} MB of wheels via LFS) ...")
    hub.push_folder(repo_id, upload_from, commit_message=f"tt-model package {manifest.name} (self-contained)")
    try:
        hub.tag_repo(repo_id, tags)
    except Exception as exc:  # tagging is best-effort
        typer.secho(f"  (could not write tags: {exc})", fg=typer.colors.YELLOW)
    typer.secho(f"✓ Pushed self-contained bundle {repo_id}", fg=typer.colors.GREEN)
    typer.secho(f"  Anyone: tt-model pull {repo_id} && tt-model serve {repo_id}", fg=typer.colors.CYAN)


# ------------------------------------------------------------------- package-thin (v6)
@app.command(name="package-thin", rich_help_panel="Publish models")
def package_thin(
    repo_id: Optional[str] = typer.Argument(None, help="HF target namespace/name (omit + --out to stage only)."),
    model_py: str = typer.Option(..., "--model-py", help="Path to the model.py / run.py runner."),
    requirements: Optional[str] = typer.Option(
        None, "--requirements", help="requirements.txt of pip pins (ttnn/TTTv2/models wheel). "
        "Omitted => a #29 template with TODO pins for the not-yet-published wheels."),
    plugin_wheel: Optional[str] = typer.Option(
        None, "--plugin-wheel", help="The vllm-tt-plugin wheel — the vLLM integration (we no longer "
        "ship a custom vLLM fork); shipped in wheels/ and installed by path."),
    ops_wheel: Optional[List[str]] = typer.Option(
        None, "--ops-wheel", help="A generic_op custom-op wheel to ship in wheels/ (repeatable)."),
    vllm_wheel: Optional[str] = typer.Option(
        None, "--vllm-wheel", help="Optional PREBUILT empty-target vLLM wheel (stock vLLM built with "
        "VLLM_TARGET_DEVICE=empty — NOT the CUDA vllm, NOT a fork). Ships in wheels/ for a hermetic "
        "install; omit and install.sh builds vLLM from source per the plugin's install-vllm-tt.sh."),
    vllm_version: str = typer.Option(
        packaging.VLLM_VERSION, "--vllm-version", help="Upstream vLLM tag the plugin builds against "
        "(empty target)."),
    with_vllm: bool = typer.Option(
        True, "--vllm/--no-vllm", help="Install vLLM (empty target) for the vllm-tt-plugin. "
        "--no-vllm packages a non-vLLM model (no vLLM step)."),
    arch: Optional[str] = typer.Option(None, "--arch", help="TT arch (blackhole|wormhole_b0); detected if omitted."),
    arch_name: Optional[str] = typer.Option(None, "--arch-name", help="HF architecture -> vllm_metadata."),
    main_class: Optional[str] = typer.Option(None, "--main-class", help='"module:Class" the plugin loads.'),
    metadata: Optional[str] = typer.Option(None, "--metadata", help="Authored vllm_metadata.json instead of --arch-name/--main-class."),
    weights: Optional[str] = typer.Option(None, "--weights", help="HF weights repo id (pointer, never embedded)."),
    weights_revision: Optional[str] = typer.Option(None, "--weights-revision"),
    mesh_topology: Optional[str] = typer.Option(None, "--mesh", help='Device topology, e.g. "P150" / "1x4".'),
    device_count: int = typer.Option(1, "--device-count"),
    python_version: str = typer.Option("3.12", "--python", help="Pinned interpreter (uv provisions)."),
    max_num_seqs: Optional[int] = typer.Option(None, "--max-num-seqs"),
    block_size: Optional[int] = typer.Option(None, "--block-size"),
    max_model_len: Optional[int] = typer.Option(None, "--max-model-len"),
    env: Optional[List[str]] = typer.Option(None, "--env", help="KEY=VALUE serving env (repeatable)."),
    name: Optional[str] = typer.Option(None, "--name"),
    out: Optional[str] = typer.Option(None, "--out", help="Stage the bundle here (kept even without a push target)."),
    private: bool = typer.Option(True, "--private/--public", help="Repo visibility when pushing."),
    publish: bool = typer.Option(False, "--publish", help="List in the community catalog (requires --public)."),
) -> None:
    """Package a v6 THIN bundle (issue #29): ship ``model.py`` + pip dependency pins
    (ttnn / TTTv2 / models wheel) + optional ``generic_op`` wheels. The per-model venv is built from
    those pins at install — NOT from an embedded ttnn wheel or a metal tree. Weights stay a pointer;
    SFPI is an external box dep.

    DRAFT (reflects the plan): fully installable once TTTv2 + the models wheel publish so the pins are
    real; until then the generated requirements.txt carries TODO pins for those two (ttnn already
    resolves from PyPI).
    """
    if publish and private:
        raise _err("--publish requires --public (a catalog listing is public by definition).")
    if repo_id is None and not out:
        raise _err("Nothing to do: pass a repo_id to push, or --out to stage locally.")
    model_path = Path(model_py).expanduser()
    if not model_path.is_file():
        raise _err(f"--model-py {model_py!r} is not a file.")
    if metadata:
        vmeta = json.loads(Path(metadata).expanduser().read_text())
        if not vmeta.get("arch") or not vmeta.get("main_class"):
            raise _err(f"{metadata} must set both 'arch' and 'main_class'.")
    elif arch_name and main_class:
        vmeta = {"arch": arch_name, "main_class": main_class}
    else:
        raise _err("Provide the serving entrypoint: --metadata, or both --arch-name and --main-class.")
    resolved_arch = arch or metal.detect_device(arch_override=arch).arch
    if not resolved_arch:
        raise _err("Could not detect arch. Pass --arch (blackhole | wormhole_b0 | ...).")
    weights_block = WeightsRef(repo_id=weights, revision=weights_revision) if weights else None
    env_map: dict = {}
    for kv in env or []:
        if "=" not in kv:
            raise _err(f"--env expects KEY=VALUE, got {kv!r}.")
        k, v = kv.split("=", 1)
        env_map[k] = v
    mesh = Mesh(devices=device_count, topology=mesh_topology) if mesh_topology else None
    resources = Resources(
        max_num_seqs=max_num_seqs, block_size=block_size, max_model_len=max_model_len
    ) if (max_num_seqs or block_size or max_model_len) else None
    bundle_name = name or (repo_id.split("/")[-1] if repo_id else model_path.stem)

    if out:
        staged = Path(out).expanduser()
        if staged.exists():
            shutil.rmtree(staged)
    else:
        staged = Path(tempfile.mkdtemp(prefix="tt-model-thin-")) / "bundle"
    manifest = packaging.stage_thin_package(
        staged, name=bundle_name, arch=resolved_arch, model_py=model_path,
        vllm_metadata=vmeta, tt_kernel_version=__version__,
        requirements=Path(requirements).expanduser() if requirements else None,
        plugin_wheel=Path(plugin_wheel).expanduser() if plugin_wheel else None,
        extra_wheels=[Path(w).expanduser() for w in (ops_wheel or [])],
        vllm_wheel=Path(vllm_wheel).expanduser() if vllm_wheel else None,
        vllm_version=vllm_version, with_vllm=with_vllm,
        weights=weights_block, device_count=device_count, mesh=mesh, env=env_map,
        resources=resources, python_version=python_version,
        tt_metal_version=metal.resolve_version() or "unknown",
    )
    typer.secho(f"✓ Staged v6 thin bundle {manifest.name} at {staged}", fg=typer.colors.GREEN)
    typer.echo(f"  runner: {model_path.name}   deps: {manifest.deps.requirements}"
               + (f" + {len(manifest.deps.wheels)} bundled wheel(s)" if manifest.deps.wheels else ""))
    if with_vllm:
        vspec = manifest.deps.vllm
        if vspec and vspec.wheel:
            typer.echo(f"  vLLM: prebuilt empty-target wheel {Path(vspec.wheel).name} (installed by path)")
        else:
            typer.echo(f"  vLLM: stock v{vspec.version if vspec else vllm_version} built empty-target at "
                       "install (--vllm-wheel ships a prebuilt one for a hermetic install)")
    if with_vllm and not plugin_wheel:
        typer.secho("  ! no --plugin-wheel given: the vllm serve path needs vllm-tt-plugin in the "
                    "bundle (the vLLM integration; we no longer ship a custom vLLM fork).",
                    fg=typer.colors.YELLOW)
    typer.echo(f"  arch registration: {manifest.entrypoint.arch_name}  ->  {manifest.entrypoint.cls}")
    if manifest.weights:
        typer.echo(f"  weights (pointer): {manifest.weights.repo_id}")
    if requirements is None:
        typer.secho("  ! requirements.txt has TODO pins for TTTv2 + the models wheel (issue #29 M0) — "
                    "edit them once those wheels publish.", fg=typer.colors.YELLOW)
    if repo_id is None:
        typer.secho("  (no push target — staged only)", fg=typer.colors.CYAN)
        return

    tags = [TT_MODEL_TAG, manifest.arch, "vllm", "thin"]
    if mesh_topology:
        tags.append(mesh_topology.lower())
    if publish:
        tags.append(TT_MODEL_CATALOG_TAG)
    typer.echo(f"Creating repo {repo_id} (private={private})")
    hub.create_repo(repo_id, private=private)
    hub.set_visibility(repo_id, private=private)
    hub.push_folder(repo_id, staged, commit_message=f"tt-model package-thin {manifest.name} (v6 thin)")
    try:
        hub.tag_repo(repo_id, tags)
    except Exception as exc:  # tagging is best-effort
        typer.secho(f"  (could not write tags: {exc})", fg=typer.colors.YELLOW)
    typer.secho(f"✓ Pushed v6 thin bundle {repo_id}", fg=typer.colors.GREEN)
    typer.secho(f"  Anyone: tt-model pull {repo_id} && tt-model serve {repo_id}", fg=typer.colors.CYAN)


# ---------------------------------------------------------------------------- pull
@app.command(rich_help_panel="Get models")
def pull(
    repo_id: str = typer.Argument(..., help="Source repo as namespace/name[@revision]."),
    force: bool = typer.Option(False, "--force", help="Install despite non-fatal mismatches."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
    models_dir: Optional[str] = typer.Option(None, "--models-dir", help="Where to install the bundle / download weights."),
    with_weights: bool = typer.Option(
        False, "--with-weights", help="Also download the HF weights now (default: skip — the "
        "model class fetches them from the HF id at load)."
    ),
) -> None:
    """Download a self-contained bundle and install it into its own venv.

    Every bundle is self-contained (v5 fat or v6 thin): ``pull`` fetches it, then runs its
    ``install.sh`` to build the per-model venv (the engine + serving stack). The box needs only a
    TT card + firmware. Pass ``--with-weights`` to fetch the HF weights now instead of at load.
    """
    repo_id, revision = _split_revision(repo_id)
    # Resolve the requested revision to a concrete sha BEFORE downloading, then fetch exactly that
    # sha — so the install records the commit it actually holds (not one a push may have moved to
    # between the download and a later query). None => Hub unreachable; fall back to plain download.
    resolved = hub.latest_revision(repo_id, revision, timeout=None)
    with tempfile.TemporaryDirectory() as td:
        snapshot = _hub(lambda: hub.download_bundle(repo_id, resolved or revision, dest=td),
                        repo_id, what="Pull",
                        consequence="Nothing was installed.")
        manifest_path = snapshot / MANIFEST_NAME
        if not manifest_path.is_file():
            raise _err(f"{repo_id} is not a tt-model bundle (no {MANIFEST_NAME}).")
        manifest = Manifest.from_json(manifest_path.read_text())

        # A container (v5.1) package: the image goes to docker, the weights to the host
        # HF cache, and none of the venv machinery below applies.
        if manifest.is_container:
            from . import container_cli

            try:
                # `--with-weights` is opt-IN here, matching the surrounding command:
                # the model fetches weights from the HF id at load anyway, into the same
                # host cache the container bind-mounts.
                container_cli.pull_container(repo_id, resolved or revision, manifest,
                                             no_weights=not with_weights)
            except (container_cli.ContainerCliError, container.ContainerError) as e:
                raise _err(str(e))
            return
        if not manifest.has_own_venv:
            raise _err(f"{repo_id} is not a self-contained bundle (schema {manifest.schema_version}).")
        _install_self_contained(
            repo_id, snapshot, manifest, force=force, arch=arch,
            models_dir=models_dir, with_weights=with_weights,
            revision=revision, resolved_revision=resolved,
        )


def _install_self_contained(
    repo_id, snapshot, manifest, *, force, arch, models_dir, with_weights,
    revision=None, resolved_revision=None,
) -> None:
    """Install a bundle that builds its OWN venv — v5 fat (embedded wheels) or v6 thin (pip pins):
    materialize it, run its ``install.sh`` to build the venv, (optionally) weights, and record it so
    ``serve`` runs from that venv. Consumer needs only a TT card + firmware (+ SFPI for v6). No host
    tt-metal/vLLM is required or touched.

    ``resolved_revision`` is the concrete commit sha the caller resolved BEFORE the download and
    then fetched — recorded verbatim so the pin matches exactly what is on disk (querying it here,
    after the download, could record a sha a mid-flight push had already moved past). ``revision``
    is the user's original request, kept only to mark a deliberate ``@revision`` pin.
    """
    env = metal.local_env(arch_override=arch)
    report = compare(manifest, env)
    _print_report(report)
    if report.has_fatal:
        raise _err("Refusing to install: fatal incompatibility (see above).")
    if report.issues and not force:
        raise _err("Refusing to install: re-run with --force to override the warnings above.")

    # v5 fat: the shipped wheels are the author's build (cp312/linux_x86_64), not universal — verify
    # their tags. v6 thin ships no platform wheels (its deps resolve via pip at install), so skip.
    if manifest.bundled is not None:
        bad = packaging.host_incompatible_wheels(manifest.bundled)
        if bad:
            for b in bad:
                typer.secho(f"  ! {b}", fg=typer.colors.YELLOW)
            if not force:
                raise _err("Shipped wheel(s) not built for this interpreter/platform; "
                           "--force to attempt anyway (likely to fail at pip install).")

    dest = runtime.resolve_models_dir(models_dir, repo_id)
    prev = localdb.get(repo_id) or {}
    if dest.exists() and (dest / "venv").exists() and not force:
        # Reuse when up to date — or when we couldn't resolve the current tip (offline): we must
        # not wipe a working install just because the check failed. A KNOWN-newer revision falls
        # through and updates in place, so a plain `pull` updates a stale bundle (no --force, which
        # would also skip the compat/wheel gates above). --force forces a reinstall regardless.
        if resolved_revision is None or prev.get("revision") == resolved_revision:
            typer.secho(
                f"✓ already installed and up to date at {dest} — reusing it (your local edits to "
                f"run.sh etc. are kept). Re-run with --force to reinstall.", fg=typer.colors.GREEN)
            return
        old = (prev.get("revision") or "?")[:8]
        typer.secho(f"↻ updating {repo_id}: {old} → {resolved_revision[:8]}", fg=typer.colors.CYAN)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(snapshot, dest)

    typer.echo("Installing shipped wheels + deps into a fresh venv (downloads torch/deps) ...")
    try:
        venv_python = runtime.install_self_contained(dest, dest / "venv")
    except subprocess.CalledProcessError as exc:
        raise _err(f"install.sh failed (exit {exc.returncode}). See output above.")
    except FileNotFoundError as exc:
        raise _err(str(exc))

    weights_path: Optional[Path] = None
    if with_weights and manifest.weights:
        typer.echo(f"Downloading weights {manifest.weights.repo_id} ...")
        weights_path = runtime.download_weights(manifest.weights, dest / "weights")

    run_script = dest / ((manifest.bundled.run_script if manifest.bundled else None) or "run.sh")
    localdb.record(repo_id, {
        "repo_id": repo_id,
        "self_contained": True,  # "has its own venv" — true for v5 fat and v6 thin; serve runs run.sh
        "install_dir": str(dest),
        "bundle_path": str(dest),  # holds vllm_metadata.json (== EXTRA_MODELS_DIR entry)
        "python": str(venv_python),
        "run_script": str(run_script),
        "arch": manifest.arch,
        "weights": manifest.weights.repo_id if manifest.weights else None,
        "weights_path": str(weights_path) if weights_path else None,
        # The exact commit sha we downloaded (resolved by the caller before the fetch), so `serve`
        # can tell this install apart from a newer published revision. `pinned` means the user
        # asked for a specific @revision — don't nag them to update off a version they chose.
        "revision": resolved_revision,
        "pinned": revision is not None,
    })
    typer.secho(f"✓ installed self-contained bundle -> {dest}", fg=typer.colors.GREEN)
    if not weights_path and manifest.weights:
        typer.secho(f"  (weights fetched at serve time from {manifest.weights.repo_id}; "
                    "pass --with-weights to pre-download)", fg=typer.colors.CYAN)
    typer.secho(f"  Serve:  tt-model serve {repo_id}", fg=typer.colors.CYAN)


def _warn_if_update_available(repo_id: str, entry: dict) -> None:
    """Best-effort: tell the user a newer bundle revision has been published.

    ``serve`` reuses an already-installed self-contained bundle as-is; without this it would
    silently keep serving the installed version even after the author pushed a new one. So we
    compare the recorded install revision against the Hub's current tip and print an advisory
    (never blocking — a serve must not depend on the Hub being reachable).

    Skips when the install was pinned to an explicit ``@revision`` (the user chose that
    version) or when no install revision was recorded (an older install predating this check,
    or an offline install), since there is then no honest baseline to compare against.
    """
    if entry.get("pinned"):
        return
    installed = entry.get("revision")
    if not installed:
        return
    latest = hub.latest_revision(repo_id)  # 3s timeout: advisory, must not hang a serve
    if latest and latest != installed:
        typer.secho(
            f"There is an update to {repo_id} "
            f"(installed {installed[:8]}, latest {latest[:8]}). "
            f"You should consider pulling:  tt-model pull {repo_id}",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _serve_self_contained(entry: dict, *, print_only: bool, extra_args: Optional[List[str]] = None) -> None:
    """Serve a v5 self-contained bundle by running its own ``run.sh`` in its own venv.

    ``run.sh`` wires the engine env (LD_PRELOAD of _ttnncpp.so, TT_METAL_HOME, EXTRA_MODELS_DIR,
    fabric-off) and launches the OpenAI server from the bundle's venv — no host stack involved.
    """
    run_script = entry.get("run_script")
    if not run_script or not Path(run_script).is_file():
        raise _err(f"Self-contained bundle for {entry.get('repo_id')} is missing its run.sh "
                   f"({run_script}). Re-run `tt-model pull {entry.get('repo_id')}`.")
    argv = ["bash", str(run_script), *(extra_args or [])]
    typer.secho(f"[vLLM self-contained: {entry.get('repo_id')} via {run_script}]", fg=typer.colors.CYAN)
    if print_only:
        # run.sh resolves the interpreter/LD_PRELOAD/paths at runtime; ask it to echo the fully
        # resolved command + env instead of the bare `bash run.sh` line.
        subprocess.run(argv, env={**os.environ, "TT_MODEL_PRINT": "1"})
        return
    try:
        raise typer.Exit(code=subprocess.run(argv).returncode)
    except KeyboardInterrupt:
        raise typer.Exit(code=130)


@app.command(rich_help_panel="Environment")
def login(
    token: Optional[str] = typer.Option(None, help="HF token; omit for interactive login."),
) -> None:
    """Log in to Hugging Face (reuses HF's token store)."""
    auth.login(token=token)
    me = auth.whoami()
    if me:
        typer.secho(f"Logged in as {me.get('name')}", fg=typer.colors.GREEN)
    else:
        raise _err("Login did not produce a valid identity.")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, rich_help_panel="Run a model")
def serve(
    ctx: typer.Context,
    repo_id: str = typer.Argument(..., help="Bundle id (namespace/name[@rev]) to serve."),
    print_only: bool = typer.Option(False, "--print", help="Print the launch command instead of running it."),
    local_only: bool = typer.Option(False, "--local-only", help="Do not pull; require an installed bundle."),
    force: bool = typer.Option(False, "--force", help="Install despite non-fatal compatibility "
                               "warnings when the pull happens here."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch/machine detection."),
    no_update_check: bool = typer.Option(
        False, "--no-update-check", help="Skip the best-effort check for a newer published "
        "bundle revision (that check makes one short, timeout-bounded Hub request)."
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", help="For a container package: which serve profile to launch "
        "(default: the author's). See `tt-model profiles <id>`."
    ),
    follow: bool = typer.Option(
        False, "--follow", help="For a container package: wait for the server to report "
        "ready, streaming the boot (a cold boot JIT-compiles kernels, ~10 min)."
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Before serving an already-installed package, re-pull it if "
        "the Hub has a newer revision (so a republished source is not served stale). No-op "
        "when already up to date, --local-only, --print, or the install has no recorded "
        "revision; any failure warns and serves the existing install unchanged."
    ),
    port: Optional[int] = typer.Option(
        None, "--port", help="Serve on this port instead of the bundle's/manifest's. For a "
        "container package it moves BOTH the published mapping and the server's own "
        "--port (which is why it must be a flag, not a passthrough argument); for a v5/v6 "
        "bundle it is appended to the launch command, where argparse last-wins."
    ),
) -> None:
    """Serve a self-contained bundle from its own venv, via its ``run.sh``.

    One command: install the bundle if needed (building its per-model venv), then run ``run.sh``,
    which wires the engine env and launches the OpenAI-compatible server. Repeat invocations skip
    the install and go straight to launch. Anything after the bundle id is passed through to vLLM.
    """
    repo_id, revision = _split_revision(repo_id)
    extra_args = list(ctx.args)  # anything after the bundle id is passed through to vLLM

    # --- container (v5.1) --------------------------------------------------------------
    # A pulled package, or a local manifest path (how an author serves a build before
    # pushing it). Checked FIRST because it is the only path that resolves a filesystem
    # path as a target; everything after this is the venv path and unchanged.
    from . import container_cli

    hint = container_cli.authored_manifest_hint(repo_id)
    if hint:
        raise _err(hint)

    cmani = container_cli.resolve_target(repo_id)
    if cmani is None and not local_only and "/" in repo_id:
        try:
            remote = hub.fetch_manifest(repo_id, revision)
        except Exception:  # noqa: BLE001 — fall through to the normal path
            remote = None
        if remote is not None and remote.is_container:
            resolved = hub.latest_revision(repo_id, revision, timeout=None)
            try:
                container_cli.pull_container(repo_id, resolved or revision, remote)
            except (container_cli.ContainerCliError, container.ContainerError) as e:
                raise _err(str(e))
            cmani = container_cli.load_pulled(repo_id)
    if cmani is not None:
        # Opt-in re-pull, only for a Hub target: a local manifest path has no revision to
        # compare against. Returns None (and warns) on any failure, leaving cmani as-is.
        if refresh and not local_only and not Path(repo_id).is_file():
            refreshed = container_cli.refresh_if_newer(repo_id, print_only=print_only)
            if refreshed is not None:
                cmani = refreshed
        src = Path(repo_id).parent if Path(repo_id).is_file() else \
            container_cli.pull_dir(repo_id)
        try:
            container_cli.serve_container(
                cmani, profile_name=profile, print_only=print_only, follow=follow,
                extra_args=extra_args, source=src, port=port, target=repo_id,
            )
        except (container_cli.ContainerCliError, container.ContainerError) as e:
            raise _err(str(e))
        return

    # Past this point --port has ALWAYS been a passthrough: `serve org/m --port 7009`
    # appended it to the launch command and argparse last-wins gave the user priority.
    # Declaring --port as an option above (which the container path needs, since the port
    # must move in two places at once) would otherwise swallow it and silently drop the
    # override. Put it back, appended, so the ordering that makes last-wins work holds.
    if port is not None:
        extra_args = extra_args + ["--port", str(port)]

    # An already-installed bundle serves from its own venv. The host toolchain (ttnn/vLLM versions)
    # is irrelevant — the bundle ships/builds its own — so nothing about the host is checked.
    entry = localdb.get(repo_id)
    if entry and entry.get("self_contained"):
        if not local_only and not no_update_check:
            _warn_if_update_available(repo_id, entry)
        _serve_self_contained(entry, print_only=print_only, extra_args=extra_args)
        return

    if local_only:
        raise _err(f"{repo_id} is not installed. Run `tt-model pull {repo_id}` first "
                   "(or drop --local-only to install now).")

    # Not installed yet: install then serve.
    resolved = hub.latest_revision(repo_id, revision, timeout=None)  # resolve before fetch
    with tempfile.TemporaryDirectory() as td:
        snapshot = _hub(lambda: hub.download_bundle(repo_id, resolved or revision, dest=td),
                        repo_id, what="Pull",
                        consequence="Nothing was installed.")
        manifest_path = snapshot / MANIFEST_NAME
        if not manifest_path.is_file():
            raise _err(f"{repo_id} is not a tt-model bundle (no {MANIFEST_NAME}).")
        mani = Manifest.from_json(manifest_path.read_text())
        if not mani.has_own_venv:
            raise _err(f"{repo_id} is not a self-contained bundle (schema {mani.schema_version}).")
        _install_self_contained(repo_id, snapshot, mani, force=force, arch=arch,
                                models_dir=None, with_weights=False,
                                revision=revision, resolved_revision=resolved)
    entry = localdb.get(repo_id)
    if entry and entry.get("self_contained"):
        _serve_self_contained(entry, print_only=print_only, extra_args=extra_args)


# ---------------------------------------------------------------------------- curl
def _discover_model(served: List[str]) -> "tuple[Optional[str], Optional[str]]":
    """Which model id to put in the request, and why — ``(model, source)``.

    The running server's own ``/v1/models`` (already probed by the caller) is authoritative:
    it is the id vLLM registered, so the request can't 404 on a stale guess. With nothing
    listening we fall back to the install record's ``weights`` — the same string ``run.sh``
    passes as ``--model`` — because ``--print`` has to work before the server is up.
    Ambiguous (several installs, no server) returns ``(None, None)`` so the caller can ask
    for ``--model``.
    """
    if served:
        return served[0], "the running server"
    installed = [e for e in localdb.all_entries() if e.get("weights")]
    if len(installed) == 1:
        return installed[0]["weights"], "the installed bundle"
    return None, None


@app.command(name="curl", context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
             rich_help_panel="Run a model")
def curl_cmd(
    ctx: typer.Context,
    prompt: str = typer.Argument(runtime.DEFAULT_PROMPT, help="The user message to send."),
    print_only: bool = typer.Option(False, "--print", help="Print the request instead of sending it."),
    model: Optional[str] = typer.Option(None, "--model", help="Model id to name in the "
                                        "request (default: ask the running server)."),
    base_url: Optional[str] = typer.Option(None, "--base-url", envvar=runtime.ENV_BASE_URL,
                                           help=f"Server root (default: {runtime.DEFAULT_BASE_URL})."),
) -> None:
    """Send a chat completion to the model you are serving.

    Fills in the endpoint and the model id — which has to match what the server registered
    or the request 404s — so the last step of a bring-up is one line:
    `tt-model curl "hello"`.

    Any option this command does not reserve is passed straight into the request body, so
    the whole vLLM sampling surface works without new flags:
    `tt-model curl "write a haiku" --temperature 0.7 --max-tokens 200`.

    --print emits the equivalent curl instead of sending it, for a doc or a bug report.
    """
    base = base_url or runtime.DEFAULT_BASE_URL
    try:
        params = runtime.parse_extra_params(list(ctx.args))
    except ValueError as exc:
        raise _err(f"{exc}\n  usage: tt-model curl \"your prompt\" [--key value ...]")

    # One probe, used twice: it names the model AND tells us whether anything is listening,
    # so a down server is reported as such instead of as a bare curl exit code.
    served = runtime.list_models(base)
    resolved, source = (model, "--model") if model else _discover_model(served)
    if not resolved:
        installed = [e for e in localdb.all_entries() if e.get("weights")]
        raise _err(
            f"Can't tell which model to ask for: nothing is serving at {base} and "
            + (f"{len(installed)} bundles are installed. " if installed else "no bundle is installed. ")
            + "Start it with `tt-model serve <id>`, or pass --model <hf-id>."
        )

    argv = runtime.curl_argv(base, runtime.chat_payload(resolved, prompt, params=params))
    if print_only:
        if source != "--model":
            # stderr, not stdout: the command must stay pipeable into a shell and
            # copy-pasteable as a whole block.
            typer.secho(f"○ model id from {source}: {resolved}",
                        fg=typer.colors.CYAN, err=True)
        console.raw(runtime.render_curl(argv))
        return
    if not served:
        raise _err(f"Nothing is serving at {base}. Start it with `tt-model serve <id>`, "
                   "or use --print to just see the request.")
    if shutil.which("curl") is None:
        raise _err("curl is not on PATH. Re-run with --print and paste the command, "
                   "or install curl.")
    raise typer.Exit(code=subprocess.run(argv).returncode)


# ---------------------------------------------------------------------------- info
@app.command(rich_help_panel="Get models")
def info(
    repo_id: str = typer.Argument(..., help="Repo as namespace/name[@revision]."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
) -> None:
    """Print a bundle's manifest and its compatibility verdict vs the local env."""
    repo_id, revision = _split_revision(repo_id)
    manifest = _hub(lambda: hub.fetch_manifest(repo_id, revision), repo_id,
                    what="Inspect")
    console.raw(manifest.to_json())
    typer.echo("")
    report = compare(manifest, metal.local_env(arch_override=arch))
    _print_report(report)


# ---------------------------------------------------------------------------- list
@app.command(name="list", rich_help_panel="Get models")
def list_installed() -> None:
    """List locally installed bundles, and whether each can be served right now.

    The last column is the useful one: a package can be recorded as installed and still not
    be servable — a container package whose image was pruned, or a bundle whose install
    directory was removed — and without this that only surfaces as an error at ``serve``.
    """
    entries = localdb.all_entries()
    if not entries:
        typer.echo("No bundles installed.")
        return

    from . import container_cli

    rows = []
    for e in entries:
        if e.get("container"):
            rows.append(container_cli.describe_pulled(e))
            continue
        install = e.get("install_dir") or e.get("bundle_path")
        ready = bool(install) and Path(install).is_dir()
        rows.append({
            "repo_id": e.get("repo_id", "?"),
            "kind": "thin" if e.get("thin") else "bundle",
            "arch": e.get("arch") or "?",
            "profile": (e.get("revision") or "")[:8] or "-",
            "image": "-",
            "size": "",
            "ready": ready,
            "why": "" if ready else f"install dir missing ({install or 'not recorded'})",
        })

    # Laid out directly rather than through check_table: that helper truncates its columns
    # at a fixed width, which turned "container · blackhole" into "container · black…".
    w_id = max(len(r["repo_id"]) for r in rows)
    w_kind = max(len(r["kind"]) for r in rows)
    w_arch = max(len(r["arch"]) for r in rows)
    for r in rows:
        mark = "✓" if r["ready"] else "✗"
        extra = "  ".join(x for x in (
            f"image {r['image']}" if r["image"] != "-" else "",
            r["size"],
            f"profile {r['profile']}" if r["kind"] == "container" else "",
        ) if x)
        console.raw(
            f"  {mark} {r['repo_id']:<{w_id}}  {r['kind']:<{w_kind}}  "
            f"{r['arch']:<{w_arch}}  {extra}".rstrip()
        )

    unready = [r for r in rows if not r["ready"]]
    if unready:
        console.raw("")
        for r in unready:
            console.note(f"{r['repo_id']}: {r['why']}", marker="!", style="warning")


# -------------------------------------------------------------------------- search
@app.command(rich_help_panel="Get models")
def search(
    query: str = typer.Argument("", help="Free-text query over tt-model cache repos."),
    limit: int = typer.Option(50, help="Max results."),
    catalog: bool = typer.Option(
        False, "--catalog", help="Restrict to repos listed in the community catalog "
        "(the set the web frontend shows), not every pushed bundle."
    ),
    arch: Optional[str] = typer.Option(
        None, "--arch", help="Only bundles tagged for this arch (blackhole | wormhole_b0 | ...)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Search the Hub for published tt-model bundles."""
    extra_tags = [t.lower() for t in (arch,) if t]
    results = hub.search(query, limit=limit, catalog_only=catalog, tags=extra_tags)
    if as_json:
        typer.echo(json.dumps(results, indent=2))
        return
    if not results:
        typer.echo("No matching bundles found.")
        return
    for r in results:
        vis = "private" if r.get("private") else "public"
        typer.echo(f"{r['id']}  [{vis}]  downloads={r.get('downloads')}")


# ----------------------------------------------------------------------- publish
@app.command(rich_help_panel="Publish models")
def publish(
    repo_id: str = typer.Argument(..., help="An already-pushed public bundle as namespace/name."),
) -> None:
    """List an existing public bundle in the community catalog (opt-in).

    Use this to add a bundle you pushed earlier without ``--publish``. The catalog only
    ever holds a pointer to your public HF repo; it stores none of your content, and your
    repo stays entirely under your governance. Delist with ``tt-model unpublish``.
    """
    try:
        if hub.is_private(repo_id):
            raise _err(f"{repo_id} is private; the catalog is public. Make it public first "
                       "(`tt-model push ... --public`) before listing.")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _err(f"Could not read {repo_id} on the Hub: {exc}")
    hub.set_catalog_listing(repo_id, listed=True)
    typer.secho(
        f"✓ Listed {repo_id} in the community catalog (pointer only; content stays yours). "
        f"Delist with `tt-model unpublish {repo_id}`.",
        fg=typer.colors.GREEN,
    )


# --------------------------------------------------------------------- unpublish
@app.command(rich_help_panel="Publish models")
def unpublish(
    repo_id: str = typer.Argument(..., help="A listed bundle as namespace/name."),
) -> None:
    """Remove a bundle from the community catalog. The repo itself is untouched."""
    hub.set_catalog_listing(repo_id, listed=False)
    typer.secho(
        f"✓ Delisted {repo_id} from the community catalog (it drops off on the next crawl). "
        "The repo and its content are unchanged.",
        fg=typer.colors.GREEN,
    )


# ------------------------------------------------------------------------------ rm
@app.command(rich_help_panel="Maintenance")
def rm(
    repo_id: str = typer.Argument(..., help="Installed bundle as namespace/name."),
    keep_cache: bool = typer.Option(
        False, "--keep-cache", help="For a container package: keep the JIT kernel cache, "
        "so a later pull of the same model boots fast instead of recompiling (~10 min)."
    ),
    include_weights: bool = typer.Option(
        False, "--include-weights", help="For a container package: ALSO delete the model "
        "weights from the HF cache. They are shared with anything else that uses them and "
        "can be tens of gigabytes to re-download, so this is off by default."
    ),
) -> None:
    """Remove a locally installed bundle and its index entry.

    For a container (v5.1) package this removes the containers, the docker image, the
    pulled manifest, the kernel cache and the package's own snapshot in the HF cache.
    For a v5/v6 bundle it removes the per-model venv and files.

    Weights are kept unless ``--include-weights``: they are shared with everything else on
    the host and are a pointer rather than part of the package.
    """
    entry = localdb.get(repo_id)
    if not entry:
        raise _err(f"{repo_id} is not recorded as installed.")

    # --- container (v5.1) --------------------------------------------------------------
    # Checked FIRST: a container entry has no install_dir, so the venv branch below would
    # drop the index entry and report success while leaving ~10 GB of image on disk.
    if entry.get("container"):
        from . import container_cli

        cmani = container_cli.load_pulled(repo_id)
        if cmani is None:
            localdb.remove(repo_id)
            console.note("index entry removed; the pulled manifest was already gone",
                         marker="○")
            return
        try:
            container_cli.remove_container(repo_id, cmani, keep_cache=keep_cache,
                                           include_weights=include_weights)
        except (container_cli.ContainerCliError, container.ContainerError) as e:
            raise _err(str(e))
        return

    install_dir = entry.get("install_dir") or entry.get("bundle_path")
    removed = False
    if install_dir:
        p = Path(install_dir)
        if p.is_dir():
            shutil.rmtree(p)
            removed = True
    localdb.remove(repo_id)
    if removed:
        typer.secho(f"✓ Removed {repo_id} ({install_dir})", fg=typer.colors.GREEN)
    else:
        typer.secho("Index entry removed; install folder was already gone.",
                    fg=typer.colors.YELLOW)


def _require_container(target: str):
    """Resolve a stop/logs/profiles target, or fail with the one thing to do next."""
    from . import container_cli

    hint = container_cli.authored_manifest_hint(target)
    if hint:
        raise _err(hint)
    m = container_cli.resolve_target(target)
    if m is None:
        raise _err(
            f"{target} is not a pulled container package (nor a container manifest path). "
            f"Pull it first:  tt-model pull {target}"
        )
    return m


@app.command(rich_help_panel="Run a model")
def stop(
    target: str = typer.Argument(..., help="Container package: org/name, or a manifest path."),
    profile: Optional[str] = typer.Option(None, "--profile", help="Stop only this profile."),
) -> None:
    """Stop a running container package, SIGTERM first.

    A clean SIGTERM lets the server close the mesh on its way out. If the grace period
    expires and docker has to SIGKILL, the devices are left needing a reset — so the mesh
    is reset with tt-smi from a throwaway container, and you are told it happened.
    """
    from . import container_cli

    try:
        container_cli.stop_container(_require_container(target), profile_name=profile)
    except (container_cli.ContainerCliError, container.ContainerError) as e:
        raise _err(str(e))


@app.command(rich_help_panel="Run a model")
def logs(
    target: str = typer.Argument(..., help="Container package: org/name, or a manifest path."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new output."),
    profile: Optional[str] = typer.Option(None, "--profile", help="Logs for this profile."),
) -> None:
    """Show the server logs of a running container package."""
    from . import container_cli

    try:
        code = container_cli.logs_container(_require_container(target),
                                            profile_name=profile, follow=follow,
                                            target=target)
    except (container_cli.ContainerCliError, container.ContainerError) as e:
        raise _err(str(e))
    if code != 0:
        raise typer.Exit(code=code)


@app.command(rich_help_panel="Get models")
def profiles(
    target: str = typer.Argument(
        ..., help="A container package: org/name (pulled), or a manifest path."
    ),
) -> None:
    """Show a container package's serve profiles, and which one is the default.

    One image serves every profile; pick one with `tt-model serve <id> --profile <name>`.
    (`tt-model list` remains the inventory of locally installed bundles.)
    """
    from . import container_cli

    try:
        container_cli.list_containers(_require_container(target))
    except (container_cli.ContainerCliError, container.ContainerError) as e:
        raise _err(str(e))


@app.command(rich_help_panel="Publish models")
def push(
    staged_dir: str = typer.Argument(
        ..., help="The staged directory `tt-model package --container` produced."
    ),
    repo: Optional[str] = typer.Option(
        None, "--repo", help="Target repo, overriding the one recorded in the manifest."
    ),
    private: Optional[bool] = typer.Option(
        None, "--private/--public", help="Repo visibility. Applied when the repo is CREATED. "
        "For a repo that already exists, passing the flag changes its visibility and says "
        "so; omitting it leaves visibility exactly as it is."
    ),
    publish: bool = typer.Option(
        False, "--publish", help="Also list the package in the community catalog "
        "(requires a public repo)."
    ),
) -> None:
    """Publish a CONTAINER (v5.1) package directory to the Hub.

    A container package is built first and published second, because the build is long and
    worth verifying locally before it goes anywhere — so publishing takes the staged
    directory rather than a repo id. For a v5/v6 bundle, `tt-model package <repo>` still
    both builds and pushes.

    ``--publish`` adds the community-catalog tag after the upload, exactly as
    ``tt-model publish`` would. The catalog only ever holds a pointer to your public repo.
    """
    if publish and private:
        raise _err("--publish requires --public (a catalog listing is public by definition).")
    from . import container_cli

    out = Path(staged_dir).expanduser()
    cmani = container_cli.is_package_dir(out)
    if cmani is None:
        raise _err(
            f"{out} is not a staged container package (no {MANIFEST_NAME} with a container "
            "block).\n"
            "  → build one:            tt-model package --container <tt-model.yaml>\n"
            "  → for a v5/v6 bundle:   tt-model package <namespace/name>"
        )
    target = repo or (cmani.container.built or {}).get("repo")
    if not target:
        raise _err(f"{out} records no target repo; pass --repo namespace/name.")
    _ensure_repo(target, private, publish=publish)
    try:
        container_cli.push_container(str(out), cmani, target)
    except container_cli.ContainerCliError as e:
        raise _err(str(e))

    if publish:
        # AFTER the upload, so the tag lands on the model card `package` generated and
        # `push_container` just wrote (set_catalog_listing reloads and unions onto it,
        # rather than clobbering it the way tag_repo would).
        try:
            hub.set_catalog_listing(target, listed=True)
        except Exception as exc:  # noqa: BLE001
            # The bytes are already on the Hub; a failed tag write must not read as a
            # failed push. Say exactly what is left to do instead.
            console.note(f"uploaded {target}, but could not list it in the catalog: {exc}",
                         marker="!", style="warning")
            console.note(f"retry the listing alone with: tt-model publish {target}",
                         marker="→")
        else:
            console.milestone(
                f"listed {target} in the community catalog (pointer only; content stays "
                f"yours) — delist with `tt-model unpublish {target}`"
            )


def _split_revision(repo_id: str) -> "tuple[str, Optional[str]]":
    """Split ``namespace/name@revision`` into (repo_id, revision|None)."""
    if "@" in repo_id:
        rid, rev = repo_id.rsplit("@", 1)
        return rid, rev
    return repo_id, None


@app.command(rich_help_panel="Maintenance")
def version() -> None:
    """Print the tt-model version."""
    typer.echo(__version__)


def main() -> None:
    """Console-script entry point. Both ``tt-model`` and the legacy ``tt-kernel`` name map here;
    invoking via the old name still works but prints a one-line deprecation note to stderr."""
    if compat.invoked_as_legacy():
        typer.secho(
            "note: `tt-kernel` has been renamed to `tt-model`; the old command still works "
            "but is deprecated — please switch to `tt-model`.",
            fg=typer.colors.YELLOW, err=True,
        )
    app()


if __name__ == "__main__":
    main()
