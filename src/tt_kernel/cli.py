# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``tt-model`` command-line interface."""

from __future__ import annotations

import datetime
import importlib.util
import json
import os
import shutil
import socket
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
    auth, bundles, cache, compat, device, devtools, hub, instances, localdb, metal,
    packaging, provision, resolve as resolve_mod, runtime,
    start as start_mod, toolchain,
)
from .manifest import (
    CompatibilityReport,
    FileEntry,
    Manifest,
    Mesh,
    Producer,
    Resources,
    RunnerPayload,
    WeightsRef,
    compare,
    runner_version_advisory,
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
    commands |= {"instances", "dev"}
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
    help="Publish and pull precompiled tt-metal kernel caches over Hugging Face Hub.",
    no_args_is_help=True,
    add_completion=False,
    # Typer enables these by default, which turned every unhandled exception into a
    # ~60-line syntax-highlighted stack with source frames from httpx and
    # huggingface_hub. A traceback is not a user-facing error message; the handlers
    # below render a diagnosis card instead. `--verbose` puts the traceback back.
    pretty_exceptions_enable=False,
    cls=_SuggestingGroup,
)

# Sub-app for the tt-metal instance registry (the supply side of version resolution).
instances_app = typer.Typer(
    name="instances",
    help="Discover, register, and inspect the tt-metal builds on this host.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
app.add_typer(instances_app, rich_help_panel="Environment")

# Developer fixtures. Hidden: these fabricate test data and are not part of any user flow,
# so they should not crowd `--help` for the people running models.
dev_app = typer.Typer(
    name="dev",
    help="Developer fixtures (not part of the user flow).",
    no_args_is_help=True,
    add_completion=False,
    hidden=True,
    pretty_exceptions_enable=False,
)
app.add_typer(dev_app)


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
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, rich_help_panel="Get started")
def start(
    ctx: typer.Context,
    model: Optional[str] = typer.Argument(
        None, help="Model or bundle id to serve. Omit to pick from what is installed."
    ),
    token: Optional[str] = typer.Option(None, "--token", help="Hugging Face token (else $HF_TOKEN, the HF token store, or a prompt)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Never prompt; fail instead if something is missing."),
    force: bool = typer.Option(False, "--force", help="Install despite non-fatal mismatches."),
    port: int = typer.Option(8000, "--port", help="Port to serve on."),
    print_only: bool = typer.Option(False, "--print", help="Print the launch command instead of running it."),
    local_only: bool = typer.Option(False, "--local-only", help="Require an installed bundle; never hit the Hub."),
    instance: Optional[str] = typer.Option(None, "--instance", help="Force a registered tt-metal instance by name."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
) -> None:
    """Get a model serving, in four guided steps.

    Walks account -> environment -> model -> serve, stopping at the first step that cannot succeed rather than failing later with less context. Everything it does is available as an individual command; this is the path that sequences them.
    """
    extra_args = list(ctx.args)

    # `ignore_unknown_options` is on so that flags meant for vLLM can be forwarded, which
    # means click hands an option-shaped token to the `model` ARGUMENT rather than rejecting
    # it: `tt-model start --force` went looking for a Hub repo literally named "--force". No
    # model id starts with a dash, so treat such a token as the flag it is and pick a model
    # the normal way.
    if model and model.startswith("-"):
        extra_args.insert(0, model)
        model = None

    # No model given: offer what is installed rather than erroring. A guided command that
    # answers "Missing argument 'model'." has failed at the one thing it exists to do.
    # The roadmap goes up FIRST — before the model picker, which prompts. Printed after
    # the prompt, it told the user what the run would do only once they had already
    # committed to it, and the menu appeared against an empty screen with no frame.
    console.register_phases(start_mod.PHASES)
    # Claim the screen FIRST: the step list lives at the top and everything below scrolls
    # under it. Pinning after the panel printed the panel into the region and then
    # overprinted it from row 1 down. Released before the vLLM handoff below — and by
    # atexit/SIGTERM on any path that does not reach it.
    console.pin_stepper()
    # The panel carries what the one-line stepper has no room for: what each step actually
    # does. Printed once, below the header, into the scrolling body.
    console.console.print(console.steps_panel_lines(
        "tt-model", [(t, start_mod.PHASE_DETAIL[t]) for t in start_mod.PHASES]))

    picked_note = None
    if not model:
        model, picked_note = _pick_model(
            interactive=not yes and start_mod.stdin_is_interactive())

    # ---- 1. Account. The prompt lives OUTSIDE any step(): a capturing step would swallow
    # it and the CLI would appear to hang waiting on input nobody can see.
    with console.phase("Account") as ph:
        allow_prompt = not yes and start_mod.stdin_is_interactive()
        account = start_mod.resolve_account(token, allow_prompt=allow_prompt)
        if account.logged_in:
            console.note(f"{account.name or 'authenticated'} — via {account.source}",
                         marker="✓", style="success")
        else:
            console.note("not logged in — public bundles only", marker="○", style="muted")
            console.mark_skipped(ph, "no token — public bundles only")

    # ---- 2. Validate. Raised through the phase so the stepper shows ✗, with the card
    # rendered after it collapses rather than above its own phase line.
    try:
        with console.phase("Validate"):
            env = start_mod.validate(port, arch_override=arch)
            tbl = console.check_table()
            for c in env.report.components:
                console.check_row(
                    tbl, "[success]✓[/success]" if c.adequate else "[error]✗[/error]",
                    c.name, console.fmt_version(c.version), f"require >= {c.required}")
            console.check_row(
                tbl, "[success]✓[/success]" if env.port_free else "[error]✗[/error]",
                f"port {env.port}", "free" if env.port_free else "in use")
            console.print_table(tbl)
            for c in env.conflicts:
                console.note(c.message, marker="!", style="warning")
            blockers = env.blockers
            if blockers:
                raise _PreflightFailed()
    except _PreflightFailed:
        raise _start_blocked(env.blockers, env)

    # ---- 3. Hardware. Reuses the probe Validate already did — tt-smi is not free, and
    # asking twice could report two different device counts within one run.
    with console.phase("Hardware") as ph:
        tbl = console.check_table()
        console.check_row(
            tbl, "[success]✓[/success]" if env.arch else "[warning]![/warning]",
            env.arch or "no device", f"{env.device_count} device(s)",
            f"via {env.device_source}" if env.device_source else "")
        console.print_table(tbl)
        if not env.arch:
            # Not a blocker: --print needs no card, and the mesh check that really matters
            # happens against the bundle's manifest at pull time. But it must not read as a
            # passed check either.
            console.note("no Tenstorrent device detected — tt-smi found no card",
                         marker="!", style="warning")
            console.mark_skipped(ph, "no card detected")

    # ---- 4. Model.
    with console.phase("Model"):
        repo_id, how = start_mod.resolve_bundle(model)
        console.note(f"{repo_id} — {picked_note or how}", marker="○", style="muted")
        if not start_mod.is_installed(repo_id):
            if local_only:
                raise _err(f"No installed bundle for {repo_id} (and --local-only forbids a pull).")
            _ensure_vllm_pulled(repo_id, None, arch=arch, bundles_dir=None,
                                force=force, instance=instance)
        console.milestone(f"ready: {repo_id}")

    # ---- 5. Serve. The phase is closed before handing the terminal to vLLM: a still-
    # ticking spinner and a foreground child would fight for the same row.
    # The child owns the terminal from here: repaint the header one last time, then hand
    # the full screen back. A foreground process printing inside our scroll region, under a
    # header that stops updating, is worse than no header at all.
    console.stepper_line_for("Serve")
    console.show_stepper()
    console.pinned.release()
    _serve_vllm(repo_id, None, print_only=print_only, local_only=True, arch=arch,
                bundles_dir=None, do_health=False, instance=instance,
                extra_args=(["--port", str(port)] if port != 8000 else []) + extra_args)


def _pick_model(*, interactive: bool) -> "tuple[str, str]":
    """Resolve which model to serve when the user named none.

    The rule is that we may decline to *choose for* the user, but we must not decline to
    *let them choose*. So a bundle whose adapter is missing is never auto-selected — that
    would walk them through three phases to fail at the fourth — but it is still offered in
    the menu, marked, because they may be about to fix PYTHONPATH or may simply want to see
    the failure. Only a non-interactive caller, which cannot be asked, gets a refusal.
    """
    choices = start_mod.installed_choices()

    if not choices:
        raise _fail_panel("No model to start", [
            "[error]Nothing is installed yet, and no model was named.[/error]",
            "",
            "[info]Try:[/info]",
            "  tt-model search --catalog",
            "[muted]    browse bundles published for tt-model[/muted]",
            "  tt-model start <namespace>/<model>",
            "[muted]    pull and serve one directly[/muted]",
        ], code=2)

    servable = [c for c in choices if c.servable]

    # Interactively we always show the list, even for a single bundle. Silently choosing
    # "the only installed bundle" saves one keystroke and costs the user their sense of what
    # is about to run — and reads as the CLI having a favourite model.
    if not interactive:
        if not servable:
            lines = ["[error]Nothing installed here can serve.[/error]", ""]
            for c in choices:
                lines.append(f"  {c.repo_id}"
                             + (f"  [error]{c.blocked_by}[/error]" if c.blocked_by else ""))
            lines += ["", "[warning]Nothing was pulled or started.[/warning]", "",
                      "[info]Try:[/info]",
                      "  tt-model info <id>",
                      "[muted]    what the bundle needs[/muted]",
                      "  tt-model search --catalog",
                      "[muted]    find a bundle whose adapter is present here[/muted]",
                      f"  tt-model start {choices[0].repo_id}",
                      "[muted]    run it anyway — it stops at the same check, with detail[/muted]"]
            raise _fail_panel("No model to start", lines, code=2)
        if len(servable) == 1:
            # No prompt is possible and there is exactly one runnable candidate, so
            # proceeding is unambiguous rather than a guess.
            note = ("the only installed bundle" if len(choices) == 1
                    else "the only installed bundle that can serve here")
            return servable[0].repo_id, note
        raise _fail_panel("No model to start", [
            "[error]No model was named, and several can serve.[/error]",
            "",
            "[info]Pick one:[/info]",
        ] + [f"  tt-model start {c.repo_id}" for c in servable], code=2)

    # Interactive: offer everything, servable first, unrunnable ones marked so the choice
    # is informed rather than blocked.
    if not servable:
        console.console.print(
            "[warning]![/warning] None of these can serve in this environment yet — "
            "pick one to see what it needs.")
    console.console.print("[bold accent]Which model?[/bold accent]")
    index = console.choose_rows("Serve", [
        (not c.servable, c.repo_id, c.meta, c.blocked_by or "") for c in choices])
    if index is None:
        raise typer.Exit(code=1)
    chosen = choices[index]
    return chosen.repo_id, ("chosen from the installed bundles" if chosen.servable
                            else f"chosen despite {chosen.blocked_by}")

def _start_blocked(blockers: List[str], env) -> "typer.Exit":
    """Stop at Validate, naming the step that actually fixes each blocker.

    "tt-model install" is a pointer, not an instruction — and for a missing ttnn it is not
    even the shortest path. Each blocker gets the concrete command, because the whole point
    of stopping early is to hand back something the user can run.
    """
    lines = [f"[error]{b}[/error]" for b in blockers]
    lines += ["", "[warning]Nothing was pulled or started.[/warning]", ""]

    steps: List[tuple] = []
    if any("tt-metal" in b for b in blockers):
        steps += provision.ttnn_remedy()
    if any("vllm" in b for b in blockers):
        steps.append(("tt-model install",
                      "clones the Tenstorrent vLLM fork and installs it + the TT plugin"))
    if any("port" in b for b in blockers):
        steps.append((f"tt-model start --port <n>",
                      f"serve somewhere other than {env.port}"))
        steps.append((f"lsof -i :{env.port}", "see what is holding it"))

    lines.append("[info]" + ("Try:" if len(steps) < 2 else "Fix it with:") + "[/info]")
    for cmd, why in steps:
        lines += [f"  {cmd}", f"[muted]    {why}[/muted]"]
    lines += ["", "[muted]  tt-model doctor          # re-check when you are done[/muted]"]
    return _fail_panel("Validate", lines, code=1)

# ------------------------------------------------------------------------- install
@app.command(rich_help_panel="Get started")
def install(
    venv: Optional[str] = typer.Option(
        None, "--venv", help="Python venv to install into. Default: $VIRTUAL_ENV, else an "
        "auto-detected tt-metal venv with importable ttnn, else this interpreter."
    ),
    vllm_dir: str = typer.Option(
        provision.DEFAULT_VLLM_DIR, "--vllm-dir", help="Where to clone the vLLM fork."
    ),
    vllm_ref: str = typer.Option(
        provision.DEFAULT_VLLM_REF, "--vllm-ref",
        help="Branch/ref of the fork (default: dev — the plugin work does not live on main)."
    ),
    allow_no_ttnn: bool = typer.Option(
        False, "--allow-no-ttnn", envvar="TT_MODEL_ALLOW_NO_TTNN",
        help="Install the serving layers even without ttnn. The result CANNOT serve a "
        "model — use this only to pre-bake an image."
    ),
) -> None:
    """Install the serving stack (Tenstorrent vLLM fork + plugin) and tt-model, then verify.

    This is the one tt-model command that provisions the host. It expects tt-metal (ttnn) to already be importable in the target environment — building that is out of scope — and stops before installing anything if it is not.

    Exit codes: 0 installed and adequate · 1 preflight failed (nothing installed) · 2 usage error · 3 installed, but the toolchain is still not adequate.
    """
    if vllm_ref == "main":
        # PROTECTED FACT: the TT vLLM plugin work lives on `dev`.
        raise _err("The Tenstorrent vLLM plugin work lives on 'dev', not 'main'. "
                   "Refusing --vllm-ref main.")

    console.register_phases(provision.PHASES)
    console.console.print(console.steps_panel_lines(
        "tt-model install", [(t, provision.PHASE_DETAIL[t]) for t in provision.PHASES]))

    # ---- 1. Preflight: everything below installs, so this is the last free exit.
    # The verdict is rendered *after* the phase collapses. A card printed inside the phase
    # body lands above its own "✗ Phase 1/5" line, which reads backwards.
    try:
        with console.phase("Preflight"):
            pre = provision.check(venv, vllm_ref=vllm_ref, allow_no_ttnn=allow_no_ttnn)
            tbl = console.check_table()
            mark = "[success]✓[/success]" if pre.target.usable else "[error]✗[/error]"
            console.check_row(tbl, mark, "python", _short_path(pre.target.python),
                              pre.target.source)
            if pre.target.usable:
                console.check_row(
                    tbl, "[success]✓[/success]" if pre.ttnn_ok else "[warning]![/warning]",
                    "ttnn", "importable" if pre.ttnn_ok else "not importable",
                    "the tt-metal runtime")
            console.check_row(tbl, "[success]✓[/success]", "vllm ref", vllm_ref,
                              "tenstorrent/vllm")
            console.print_table(tbl)
            if not pre.ok:
                raise _PreflightFailed()
    except _PreflightFailed:
        raise _preflight_failed(pre)
    if not pre.ttnn_ok:
        console.note("continuing without ttnn (--allow-no-ttnn): this environment will "
                     "NOT be able to serve a model", marker="!", style="warning")

    python = pre.target.python
    target_dir = Path(vllm_dir).expanduser()

    # ---- 2. The vLLM fork.
    with console.phase("vLLM fork") as ph:
        try:
            cloned, detail = provision.clone_or_reuse_vllm(target_dir, vllm_ref)
        except RuntimeError as exc:
            ph["status"] = "failed"
            raise _fail_card("vLLM fork", {
                "cause": "could not clone the fork",
                "detail": f"git clone of {provision.VLLM_REPO}@{vllm_ref} failed.",
                "evidence": str(exc).splitlines()[-1] if str(exc) else "",
                "actions": [f"check network access to {provision.VLLM_REPO}",
                            f"clone it yourself, then: tt-model install --vllm-dir {target_dir}"],
            }, consequence="Nothing was installed.")
        console.note(detail, marker="✓" if cloned else "○",
                     style="success" if cloned else "muted")

    # ---- 3. The serving layer. Tenstorrent has NO CUDA: build vLLM with the 'empty'
    # device target and pull CPU torch — all compute runs through the TT out-of-tree
    # platform (device "tt"). A plain pip install would default to VLLM_TARGET_DEVICE=cuda.
    with console.phase("Serving layer"):
        env = {**os.environ, "VLLM_TARGET_DEVICE": "empty"}
        rc, out = provision.pip_install(
            python, ["-e", str(target_dir),
                     "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
            label="Installing vLLM fork", env=env)
        if rc != 0:
            raise _pip_failed("Serving layer", "the vLLM fork", rc, out)
        console.milestone("vLLM fork (editable)")
        rc, out = provision.pip_install(
            python, ["-e", str(target_dir / "plugins" / "vllm-tt-plugin")],
            label="Installing TT plugin")
        if rc != 0:
            raise _pip_failed("Serving layer", "the TT plugin", rc, out)
        console.milestone("TT plugin (editable)")

    # ---- 4. tt-model itself.
    with console.phase("tt-model"):
        repo_root = Path(__file__).resolve().parent.parent.parent
        rc, out = provision.pip_install(python, ["-e", str(repo_root)],
                                        label="Installing tt-model")
        if rc != 0:
            raise _pip_failed("tt-model", "tt-model", rc, out)
        console.milestone(f"tt-model (editable) — {_short_path(str(repo_root))}")

    # ---- 5. Verify. The verdict decides which closing message is honest, so it is
    # captured rather than discarded: the old script ran `doctor || true` and printed
    # "Done. Serve a model with..." over a doctor that had just exited non-zero.
    with console.phase("Verify"):
        verdict = provision.verify(python)

    _install_summary(verdict, python, allow_no_ttnn=allow_no_ttnn)


class _PreflightFailed(Exception):
    """Marks the phase failed so the stepper shows ✗, without the card landing inside it."""


def _short_path(path: Optional[str]) -> str:
    """Collapse $HOME to ~ so a long interpreter path stays readable in a row."""
    if not path:
        return "—"
    home = os.path.expanduser("~")
    return path.replace(home, "~", 1) if path.startswith(home) else path


def _preflight_failed(pre) -> "typer.Exit":
    """The card for a preflight that stopped before installing anything.

    Each suggested command gets its own line with the rationale beneath it, so nothing
    wraps mid-flag — a command the user cannot copy is not a suggestion.
    """
    lines = [f"[error]{pre.blockers[0]}[/error]"]
    if pre.target.python:
        lines.append(f"[muted]{_short_path(pre.target.python)}  ({pre.target.source})[/muted]")
    lines += ["", "[error]Nothing was installed.[/error]", ""]
    lines.append("[info]" + ("Two ways forward:" if len(pre.routes) > 1 else "Try:") + "[/info]")
    for cmd, why in pre.routes:
        lines += [f"  {cmd}", f"[muted]    {why}[/muted]"]
    if pre.escape:
        lines += ["", f"  {pre.escape[0]}", f"[muted]    {pre.escape[1]}[/muted]"]
    return _fail_panel("Preflight", lines, code=provision.EXIT_PREFLIGHT)


def _fail_panel(title: str, lines: List[str], *, code: int = 1) -> "typer.Exit":
    console.console.print(console.notice_panel(f"[error]{title}[/error]", lines,
                                               border_style="error"))
    return typer.Exit(code=code)


def _pip_failed(phase_name: str, what: str, rc: int, out: str) -> "typer.Exit":
    return _fail_card(phase_name, {
        "cause": f"pip install of {what} failed",
        "detail": f"pip exited {rc}. The captured output is below the card.",
        "evidence": provision.pip_error_line(out),
        "actions": ["re-run with --verbose to watch pip live",
                    "check disk space and network, then re-run"],
    }, consequence="Whatever installed before this point is still in place.")


def _install_summary(verdict, python: str, *, allow_no_ttnn: bool) -> None:
    """Close with a claim that matches what doctor actually found."""
    rows = [(c.name, console.fmt_version(c.version), "" if c.adequate else "missing")
            for c in verdict.report.components]
    if verdict.report.ok and not allow_no_ttnn:
        console.console.print(console.ready_panel(
            "Serving stack installed", rows,
            [f"[muted]Serve  ·[/muted]  tt-model serve <namespace>/<model>",
             f"[muted]Check  ·[/muted]  tt-model doctor"]))
        return
    lines = [f"[muted]{n:<10}[/muted] {v}" + (f"  [error]{s}[/error]" if s else "")
             for n, v, s in rows]
    lines += ["", "[warning]Everything this command installs is in place. What is missing "
              "above is built separately and is out of scope here.[/warning]",
              "", "[info]Try:[/info]",
              f"[muted]  {python} -m pip install \"{provision.TTNN_PYPI_SPEC}\"[/muted]",
              "[muted]  tt-model install --venv <tt-metal>/python_env[/muted]",
              "[muted]  tt-model doctor[/muted]"]
    console.console.print(console.notice_panel(
        "[warning]Installed, but the stack cannot serve yet[/warning]", lines,
        border_style="warning"))
    raise typer.Exit(code=provision.EXIT_INADEQUATE)



# ---------------------------------------------------------------------------- push
@app.command(rich_help_panel="Publish models")
def push(
    repo_id: str = typer.Argument(..., help="Target repo as namespace/name."),
    private: Optional[bool] = typer.Option(
        None, "--private/--public", help="Repo visibility. Applied when the repo is CREATED "
        "(new repos default to public). For a repo that already exists, passing the flag "
        "changes its visibility and says so; omitting it leaves visibility exactly as it is."
    ),
    publish: bool = typer.Option(
        False, "--publish", help="List this bundle in the community catalog (requires "
        "--public). Adds an opt-in tag; the catalog indexes a pointer to your repo — it "
        "stores nothing and your repo stays under your governance. Delist with `tt-model unpublish`."
    ),
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
    build_key: Optional[int] = typer.Option(None, help="Which build_key subtree to publish."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
    num_hw_cqs: Optional[int] = typer.Option(None, help="Hardware command queues used (default 1)."),
    name: Optional[str] = typer.Option(None, help="Bundle name (defaults to the repo name)."),
    tt_metal_version: Optional[str] = typer.Option(
        None, "--tt-metal-version", help="Override the detected tt-metal version (e.g. for testing)."
    ),
    python_package: Optional[List[str]] = typer.Option(
        None, "--python-package", help="Path to a prebuilt runner wheel/sdist to ship "
        "(repeatable). Omit for a reference runner (--runner-spec only)."
    ),
    runner_spec: Optional[str] = typer.Option(
        None, "--runner-spec", help="Runner as module:Class for dispatch --runner. With "
        "--python-package it is packaged (shipped); alone it is a reference the consumer resolves."
    ),
    runner_source: Optional[str] = typer.Option(
        None, "--runner-source", help="For a reference runner: where to get it (pip name / git URL)."
    ),
    entry_point: Optional[str] = typer.Option(
        None, "--entry-point", help="Entry-point name the wheel registers under tt_models.runners."
    ),
    capability: Optional[List[str]] = typer.Option(
        None, "--capability", help="Model-capability tag to surface in the catalog "
        "(repeatable), e.g. --capability moe --capability sliding-window-attention. Added as "
        "a repo tag; the catalog renders known ones as badges and filters."
    ),
    weights: Optional[str] = typer.Option(
        None, "--weights", help="HF model repo id whose weights this bundle targets."
    ),
    weights_revision: Optional[str] = typer.Option(None, "--weights-revision"),
    weights_allow: Optional[List[str]] = typer.Option(None, "--weights-allow"),
    weights_ignore: Optional[List[str]] = typer.Option(None, "--weights-ignore"),
    backend: str = typer.Option(
        "dispatch", "--backend", help="Serving backend: 'dispatch' (kernel-cache bundle) or "
        "'vllm' (a kernels-less bundle folder served through the Tenstorrent vLLM plugin)."
    ),
    bundle_dir: Optional[str] = typer.Option(
        None, "--bundle-dir", help="For --backend vllm: local folder holding the adapter class "
        "+ its deps (and, on the legacy path, a hand-written vllm_metadata.json). Laid into "
        "EXTRA_MODELS_DIR on pull. Optional with --manifest when the entrypoint is a built-in."
    ),
    manifest_path: Optional[str] = typer.Option(
        None, "--manifest", help="For --backend vllm: path to an authored v4 unified manifest "
        "(entrypoint/platform/runtime/target/mesh/resources). tt-model renders "
        "vllm_metadata.json from it on pull — the author writes one file, not two."
    ),
) -> None:
    """Package a bundle and publish it.

    With ``--backend dispatch`` (default): package the local kernel cache for one build_key;
    the bundle may also declare a runner (packaged or reference) and a --weights ref so a
    single pull installs kernels + runner + weights.

    With ``--backend vllm``: package the ``--bundle-dir`` folder (vllm_metadata.json + the
    ``VllmGeneratorAdapter`` class + deps) as a **kernels-less** bundle — no precompiled cache
    is shipped; the vLLM plugin JITs at first-run warmup.
    """
    # A catalog listing is public by definition — refuse to list a private repo. (An
    # *unspecified* visibility is resolved later, in _ensure_repo, where we know whether the
    # repo already exists and what it currently is.)
    if publish and private is True:
        raise _err("--publish lists the bundle in the public community catalog and requires "
                   "--public. Re-run with --public, or drop --publish to push privately.")

    if backend not in ("dispatch", "vllm"):
        raise _err(f"--backend must be 'dispatch' or 'vllm', not {backend!r}.")
    if backend == "vllm":
        _push_vllm(
            repo_id, private=private, publish=publish, bundle_dir=bundle_dir,
            manifest_path=manifest_path, arch=arch,
            name=name, tt_metal_version=tt_metal_version, weights=weights,
            weights_revision=weights_revision, weights_allow=weights_allow,
            weights_ignore=weights_ignore, capability=capability,
        )
        return
    if bundle_dir:
        raise _err("--bundle-dir is only valid with --backend vllm.")

    # Validate runtime payload args before any device/cache work or upload.
    wheel_paths: List[Path] = []
    if python_package:
        if not runner_spec:
            raise _err("--python-package requires --runner-spec module:Class (the wheel is "
                       "useless to dispatch without a runner spec).")
        for pkg in python_package:
            p = Path(pkg).expanduser()
            if not p.is_file() or p.suffix not in (".whl",) and not p.name.endswith(".tar.gz"):
                raise _err(f"--python-package {pkg!r} must be an existing .whl or .tar.gz file.")
            wheel_paths.append(p)
    if runner_spec and (":" not in runner_spec and "." not in runner_spec):
        raise _err(f"--runner-spec {runner_spec!r} must be 'module:Class' (or 'module.Class').")
    if entry_point and not runner_spec:
        raise _err("--entry-point requires --runner-spec.")
    if runner_source and not runner_spec:
        raise _err("--runner-source requires --runner-spec (it says where to get the reference runner).")

    out_root = cache.resolve_out_root(cache_dir)
    try:
        key = cache.select_build_key(out_root, build_key)
    except (FileNotFoundError, ValueError) as exc:
        raise _err(str(exc))

    subtree = cache.build_key_path(out_root, key)
    typer.echo(f"Packaging build_key {key} from {subtree}")
    # Isolation feedback + pre-push guard (#2): show what's being shipped and warn if the
    # cache does not look isolated to one model (sibling build_keys / the shared default).
    typer.echo(f"  {cache.count_kernels(subtree)} kernel group(s) in this subtree")
    default_cache = cache_dir is None and not os.environ.get("TT_METAL_CACHE")
    for warning in cache.publish_warnings(out_root, key, default_cache=default_cache):
        typer.secho(f"  ! {warning}", fg=typer.colors.YELLOW)

    dev = metal.detect_device(arch_override=arch)
    version = tt_metal_version or metal.resolve_version()
    if not version:
        raise _err(
            "Could not resolve tt-metal version. Set TT_METAL_HOME, install ttnn, or pass "
            "--tt-metal-version so the consumer can match it."
        )
    if not dev.arch:
        raise _err("Could not detect arch. Pass --arch (blackhole | wormhole_b0 | ...).")

    files = cache.index_subtree(subtree)

    # Runtime payload: build the runner block whenever a spec is given (packaged if wheels
    # were supplied, reference otherwise) and index any shipped wheels under python/.
    runner_block: Optional[RunnerPayload] = None
    if runner_spec:
        runner_block = RunnerPayload(
            spec=runner_spec,
            wheels=[p.name for p in wheel_paths],
            entry_point=entry_point,
            source=runner_source,
        )
        files = files + [
            FileEntry(path=f"python/{p.name}", sha256=cache.sha256_file(p), size=p.stat().st_size)
            for p in wheel_paths
        ]
    weights_block: Optional[WeightsRef] = None
    if weights:
        weights_block = WeightsRef(
            repo_id=weights,
            revision=weights_revision,
            allow_patterns=weights_allow or None,
            ignore_patterns=weights_ignore or None,
        )

    manifest = Manifest(
        schema_version="3",  # legacy dispatch kernel-cache bundle
        name=name or repo_id.split("/")[-1],
        tt_metal_version=version,
        arch=dev.arch,
        device_count=dev.device_count or 1,
        build_key=key,
        build_key_inputs=metal.build_key_inputs(
            num_hw_cqs=num_hw_cqs, harvesting_mask=dev.harvesting_mask
        ),
        kernel_count=cache.count_kernels(subtree),
        fast_path_kernels=cache.detect_fast_path_kernels(subtree),
        files=files,
        producer=Producer(
            tt_kernel_version=__version__,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=socket.gethostname(),
            tt_metal_home=cache.detect_cache_tt_metal_root(subtree),
        ),
        runner=runner_block,
        weights=weights_block,
    )

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        # Mirror the subtree under <staged>/<build_key>/ so it installs cleanly.
        shutil.copytree(subtree, staged / str(key), ignore=cache.ignore_junk)
        # Ship the runner wheel(s) under python/ (uploaded automatically by upload_folder).
        if wheel_paths:
            (staged / "python").mkdir()
            for p in wheel_paths:
                shutil.copy2(p, staged / "python" / p.name)
        (staged / MANIFEST_NAME).write_text(manifest.to_json())

        _ensure_repo(repo_id, private, publish=publish)
        typer.echo(
            f"Uploading {len(files)} files ({manifest.total_size / 1e6:.1f} MB) ..."
        )
        hub.push_folder(repo_id, staged, commit_message=f"tt-model push {manifest.name}")
        tags = [TT_MODEL_TAG, dev.arch]
        if publish:
            tags.append(TT_MODEL_CATALOG_TAG)
        if capability:
            tags.extend(c.strip().lower() for c in capability if c.strip())
        try:
            hub.tag_repo(repo_id, tags)
        except Exception as exc:  # tagging is best-effort
            typer.secho(f"  (could not write tags: {exc})", fg=typer.colors.YELLOW)

    typer.secho(f"✓ Pushed {repo_id} (build_key {key})", fg=typer.colors.GREEN)
    if publish:
        typer.secho(
            "✓ Listed in the community catalog. It indexes a pointer to this public repo — "
            "it stores none of your content, which stays under your governance. "
            "Delist any time with `tt-model unpublish " + repo_id + "`.",
            fg=typer.colors.GREEN,
        )


def _build_v4_manifest(
    repo_id, manifest_path, *, folder, arch, name, tt_metal_version, weights,
    weights_revision, weights_allow, weights_ignore, files,
):
    """Complete an authored v4 manifest into a full, publishable ``Manifest``.

    The authored file is a *partial* — it declares what the model needs (entrypoint, platform,
    runtime, target, mesh, resources, capabilities, weights, env) but not the bookkeeping
    tt-model owns. We fill the required fields (name, arch, tt_metal_version, device_count,
    producer, files, runner) and stamp schema v4. ``--weights`` / ``--arch`` / ``--name`` /
    ``--tt-metal-version`` override the authored values when passed.
    """
    try:
        raw = json.loads(Path(manifest_path).expanduser().read_text())
    except (OSError, ValueError) as exc:
        raise _err(f"--manifest {manifest_path!r}: {exc}")
    if not isinstance(raw, dict):
        raise _err("--manifest must be a JSON object.")
    if "entrypoint" not in raw:
        raise _err("A v4 manifest must declare an 'entrypoint' {\"class\": ..., \"arch_name\": ...}.")

    dev = metal.detect_device(arch_override=arch)
    resolved_arch = arch or raw.get("arch") or dev.arch
    if not resolved_arch:
        raise _err("Could not resolve arch. Set 'arch' in the manifest or pass --arch.")
    # Normalize the arch the same way every other push path does (dev.arch is already
    # normalized), so `--arch bh` publishes "blackhole" — not "bh", which would be a fatal
    # mismatch on every consumer and invisible to `search --arch blackhole`.
    resolved_arch = device.normalize_arch(resolved_arch)

    # The authored mesh/device_count describes the model's real topology; it must win over the
    # PUSHER's box (a 4-card model authored on a 1-card dev host must not publish device_count 1).
    mesh_devices = (raw.get("mesh") or {}).get("devices") if isinstance(raw.get("mesh"), dict) else None
    device_count = raw.get("device_count") or mesh_devices or dev.device_count or 1

    # Merge the --weights-* flags onto the authored weights block. The scoping flags
    # (revision/allow/ignore) apply to the manifest's OWN repo too; only the repo id itself is
    # guarded by --weights (a bare `--weights-allow` must still filter the authored repo).
    wblock = dict(raw.get("weights") or {})
    if weights:
        wblock["repo"] = weights
    if weights_revision:
        wblock["revision"] = weights_revision
    if weights_allow:
        wblock["allow_patterns"] = weights_allow
    if weights_ignore:
        wblock["ignore_patterns"] = weights_ignore
    if wblock:
        raw["weights"] = wblock

    raw.update(
        schema_version="4",
        name=name or raw.get("name") or repo_id.split("/")[-1],
        arch=resolved_arch,
        device_count=device_count,
        tt_metal_version=(tt_metal_version or raw.get("tt_metal_version")
                          or metal.resolve_version() or "unknown"),
        build_key=None,
        kernel_count=0,
        files=[f.model_dump() for f in files],
        producer=Producer(
            tt_kernel_version=__version__,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            hostname=socket.gethostname(),
        ).model_dump(),
        runner=RunnerPayload(backend="vllm", bundle_dir="vllm_bundle").model_dump(),
    )
    try:
        return Manifest.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 — surface a clean authoring error, not a traceback
        raise _err(f"--manifest is not a valid v4 manifest: {exc}")


def _push_vllm(
    repo_id: str,
    *,
    private: Optional[bool],  # tri-state; see _ensure_repo
    publish: bool,
    bundle_dir: Optional[str],
    manifest_path: Optional[str],
    arch: Optional[str],
    name: Optional[str],
    tt_metal_version: Optional[str],
    weights: Optional[str],
    weights_revision: Optional[str],
    weights_allow: Optional[List[str]],
    weights_ignore: Optional[List[str]],
    capability: Optional[List[str]],
) -> None:
    """Package and publish a kernels-less vLLM bundle. No kernel cache is shipped — the vLLM
    plugin JITs at first-run warmup.

    Two authoring modes:

    - ``--manifest`` (v4, recommended): the author writes ONE unified manifest; tt-model
      renders ``vllm_metadata.json`` from it on pull. ``--bundle-dir`` is optional (only needed
      to ship a custom adapter class / extension wheels; omit when the entrypoint is a tt-metal
      built-in).
    - ``--bundle-dir`` alone (legacy): the folder carries a hand-written ``vllm_metadata.json``
      + adapter code, shipped verbatim.
    """
    subdir = "vllm_bundle"
    # The folder (adapter code / wheels) is required for the legacy path, optional for v4.
    folder: Optional[Path] = None
    if bundle_dir:
        folder = Path(bundle_dir).expanduser()
        if not folder.is_dir():
            raise _err(f"--bundle-dir {bundle_dir!r} is not a directory.")

    files: List[FileEntry] = []
    if folder is not None:
        indexed = cache.index_subtree(folder)
        files = [FileEntry(path=f"{subdir}/{e.path}", sha256=e.sha256, size=e.size) for e in indexed]

    if manifest_path:
        # ---- v4: authored unified manifest, rendered on pull ----
        manifest = _build_v4_manifest(
            repo_id, manifest_path, folder=folder, arch=arch, name=name,
            tt_metal_version=tt_metal_version, weights=weights,
            weights_revision=weights_revision, weights_allow=weights_allow,
            weights_ignore=weights_ignore, files=files,
        )
        reg_arch = manifest.entrypoint.arch_name
        reg_class = manifest.entrypoint.cls
        pub_arch = manifest.arch
        tags = [TT_MODEL_TAG, pub_arch, "vllm"]
        if manifest.target:
            tags.append(manifest.target.lower())
        if manifest.capabilities:
            for c in (manifest.capabilities.tool_parser, manifest.capabilities.reasoning_parser):
                if c:
                    tags.append(f"parser:{c.lower()}")
        # Honor --capability on the v4 path too (the recommended authoring mode) so the
        # catalog badges/filters the flag promises actually appear.
        if capability:
            tags.extend(c.strip().lower() for c in capability if c.strip())
    else:
        # ---- legacy: verbatim vllm_metadata.json ----
        if folder is None:
            raise _err("--backend vllm requires --manifest (v4) or --bundle-dir (legacy).")
        try:
            md = bundles.read_vllm_metadata(folder)
        except (FileNotFoundError, ValueError) as exc:
            raise _err(str(exc))
        if not md.arch or not md.main_class:
            raise _err(
                f"{bundles.VLLM_METADATA_NAME} must set both 'arch' (HF architecture name) and "
                "'main_class' (\"module:Class\")."
            )
        dev = metal.detect_device(arch_override=arch)
        if not dev.arch:
            raise _err("Could not detect arch. Pass --arch (blackhole | wormhole_b0 | ...).")
        version = tt_metal_version or metal.resolve_version() or "unknown"
        weights_target = weights or md.hf_weights
        weights_block: Optional[WeightsRef] = None
        if weights_target:
            weights_block = WeightsRef(
                repo_id=weights_target, revision=weights_revision,
                allow_patterns=weights_allow or None, ignore_patterns=weights_ignore or None,
            )
        manifest = Manifest(
            schema_version="3",  # legacy verbatim vLLM bundle (no v4 blocks)
            name=name or repo_id.split("/")[-1],
            tt_metal_version=version,
            arch=dev.arch,
            device_count=dev.device_count or 1,
            build_key=None,  # kernels-less
            kernel_count=0,
            fast_path_kernels=None,
            files=files,
            producer=Producer(
                tt_kernel_version=__version__,
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                hostname=socket.gethostname(),
            ),
            runner=RunnerPayload(backend="vllm", bundle_dir=subdir),
            weights=weights_block,
        )
        reg_arch, reg_class, pub_arch = md.arch, md.main_class, dev.arch
        tags = [TT_MODEL_TAG, pub_arch, "vllm"]
        if capability:
            tags.extend(c.strip().lower() for c in capability if c.strip())

    if publish:
        tags.append(TT_MODEL_CATALOG_TAG)

    typer.echo(f"Packaging vLLM bundle ({len(files)} file(s)"
               + (", rendered metadata" if manifest_path else "") + ")")
    typer.echo(f"  arch registration: {reg_arch}  ->  {reg_class}")
    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        if folder is not None:
            shutil.copytree(folder, staged / subdir, ignore=cache.ignore_junk)
        (staged / MANIFEST_NAME).write_text(manifest.to_json())

        _ensure_repo(repo_id, private, publish=publish)
        typer.echo(f"Uploading {len(files)} files ({manifest.total_size / 1e6:.1f} MB) ...")
        hub.push_folder(repo_id, staged, commit_message=f"tt-model push {manifest.name} (vllm)")
        try:
            hub.tag_repo(repo_id, tags)
        except Exception as exc:  # tagging is best-effort
            typer.secho(f"  (could not write tags: {exc})", fg=typer.colors.YELLOW)

    typer.secho(f"✓ Pushed vLLM bundle {repo_id}", fg=typer.colors.GREEN)
    _routine(f"  Serve it:  tt-model serve {repo_id}", fg=typer.colors.CYAN)


# -------------------------------------------------------------------------- package
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
    from_metal: str = typer.Option(
        ..., "--from-metal", help="Path to your modified tt-metal-community tree (embedded as metal/)."
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
        weights=weights_block, device_count=device_count, mesh=mesh, env=env_map,
        resources=resources, python_version=python_version,
        tt_metal_version=metal.resolve_version() or "unknown",
    )
    typer.secho(f"✓ Staged v6 thin bundle {manifest.name} at {staged}", fg=typer.colors.GREEN)
    typer.echo(f"  runner: {model_path.name}   deps: {manifest.deps.requirements}"
               + (f" + {len(manifest.deps.wheels)} bundled wheel(s)" if manifest.deps.wheels else ""))
    if not plugin_wheel:
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
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
    probe: bool = typer.Option(False, "--probe", help="Open a device to read the true build_key."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
    models_dir: Optional[str] = typer.Option(None, "--models-dir", help="Where to download weights."),
    bundles_dir: Optional[str] = typer.Option(
        None, "--bundles-dir", help="For a vLLM bundle: where to lay the model folder "
        "(== EXTRA_MODELS_DIR). Default: $TT_MODEL_BUNDLES_DIR or ~/.cache/tt-model/bundles."
    ),
    with_weights: bool = typer.Option(
        False, "--with-weights", help="For a vLLM bundle: also download the HF weights now "
        "(default: skip — the model class fetches them from the HF id at load)."
    ),
    no_python: bool = typer.Option(False, "--no-python", help="Skip installing the runner wheel."),
    no_weights: bool = typer.Option(False, "--no-weights", help="Skip downloading weights."),
    kernels_only: bool = typer.Option(
        False, "--kernels-only", help="Install only the kernel cache (implies --no-python and --no-weights)."
    ),
    python_exe: Optional[str] = typer.Option(
        None, "--python", help="Target interpreter for the runner pip install (default: this venv)."
    ),
    instance: Optional[str] = typer.Option(
        None, "--instance", help="For a v4 vLLM bundle: force a registered tt-metal instance by "
        "name instead of auto-selecting the newest that satisfies the manifest's ranges."
    ),
) -> None:
    """Download a bundle and install everything it carries: kernels, runner, weights.

    A single pull installs the kernel cache, sets up the runner (pip-installs a packaged
    wheel, or verifies a reference runner resolves), and downloads the model weights, then
    prints the exact `serve` command. Skip parts with --no-python / --no-weights /
    --kernels-only.
    """
    if kernels_only:
        no_python = no_weights = True

    repo_id, revision = _split_revision(repo_id)
    runner_installed = False  # we pip-installed a packaged wheel
    runner_ready = False  # runner is usable (installed, or reference that resolves)
    weights_path: Optional[Path] = None
    # Resolve the requested revision to a concrete sha BEFORE downloading, then fetch exactly that
    # sha — so a self-contained install records the commit it actually holds (not one a push may
    # have moved to between the download and a later query). None => Hub unreachable; fall back to
    # the plain download and record nothing.
    resolved = hub.latest_revision(repo_id, revision, timeout=None)
    with tempfile.TemporaryDirectory() as td:
        snapshot = _hub(lambda: hub.download_bundle(repo_id, resolved or revision, dest=td),
                        repo_id, what="Pull",
                        consequence="Nothing was installed.")
        manifest_path = snapshot / MANIFEST_NAME
        if not manifest_path.is_file():
            raise _err(f"{repo_id} is not a tt-model bundle (no {MANIFEST_NAME}).")
        manifest = Manifest.from_json(manifest_path.read_text())

        # A bundle that builds its own venv — v5 fat (embedded wheels) or v6 thin (pip pins) —
        # installs into that venv; no host tt-metal/vLLM needed. Then return.
        if manifest.has_own_venv:
            _install_self_contained(
                repo_id, snapshot, manifest, force=force, arch=arch,
                models_dir=models_dir, with_weights=with_weights and not no_weights,
                revision=revision, resolved_revision=resolved,
            )
            return

        # vLLM bundles carry no kernel cache: install the model folder into bundles_dir
        # instead of the tt-metal cache, then return.
        if manifest.runner and manifest.runner.is_vllm:
            _install_vllm_bundle(
                repo_id, snapshot, manifest, force=force, arch=arch,
                models_dir=models_dir, bundles_dir=bundles_dir,
                with_weights=with_weights and not no_weights, instance=instance,
            )
            return

        env = metal.local_env(arch_override=arch, probe=probe)
        report = compare(manifest, env)
        _print_report(report)
        _warn_toolchain()  # complements the kernel compat check with the serving-stack versions

        if report.has_fatal:
            raise _err("Refusing to install: fatal incompatibility (see above).")
        if report.issues and not force:
            raise _err("Refusing to install: re-run with --force to override the warnings above.")

        staged = snapshot / str(manifest.build_key)
        if not staged.is_dir():
            raise _err(f"Bundle is missing its build_key subtree {manifest.build_key}/.")

        # Partition the file index: kernels live under the build_key subtree; runner
        # wheels under python/ (verified relative to the snapshot root).
        wheel_entries = [f for f in manifest.files if f.path.startswith("python/")]
        kernel_entries = [f for f in manifest.files if not f.path.startswith("python/")]

        problems = cache.verify_files(staged, kernel_entries)
        if not problems:
            console.milestone(f"verified {len(kernel_entries)} kernel file(s)")
        if problems:
            for p in problems[:20]:
                typer.secho(f"  {p}", fg=typer.colors.RED)
            raise _err(f"Integrity check failed ({len(problems)} problem(s)).")

        out_root = cache.resolve_out_root(cache_dir)
        target = cache.install_subtree(staged, out_root, manifest.build_key)
        typer.secho(f"✓ kernels -> {target}", fg=typer.colors.GREEN)
        if manifest.fast_path_kernels is False:
            typer.secho(
                "  ! baseline-only bundle: it lacks the traced-decode / on-device-lm_head "
                "kernels, so serving on the fast path (DISPATCH_TRACE / "
                "DISPATCH_ONDEVICE_LMHEAD) will re-JIT them. Produce a fast-path bundle by "
                "warming with those flags enabled.",
                fg=typer.colors.YELLOW,
            )

        # Cross-host dep relocation: if this bundle was built against a tt-metal at a
        # different path than ours, rewrite the tree-dep prefix so the cache hits here too
        # (in-cache paths were already relocated by install_subtree).
        producer_home = manifest.producer.tt_metal_home if manifest.producer else None
        if producer_home:
            consumer_home = metal.detect_tt_metal_home()
            if (consumer_home and os.path.isdir(consumer_home)
                    and os.path.normpath(consumer_home) != os.path.normpath(producer_home)):
                n = cache.relocate_tt_metal_tree(target, producer_home, consumer_home)
                if n:
                    typer.secho(
                        f"  ↻ relocated tt-metal tree deps in {n} dephash file(s): "
                        f"{producer_home} -> {consumer_home}",
                        fg=typer.colors.CYAN,
                    )

        # ---- runtime payload ----
        advisory = runner_version_advisory(manifest, env)
        if advisory is not None and (manifest.runner or manifest.weights):
            typer.secho(
                f"  ! runner/weights target tt-metal {advisory.expected!r}; you have "
                f"{advisory.detected!r}. Installing anyway — it will NOT run until the "
                "serving environment matches.",
                fg=typer.colors.YELLOW,
            )

        # Runner: packaged => verify + pip install the shipped wheel(s); reference => the
        # runner is not shipped, so just verify it resolves in the target env (install nothing).
        if manifest.runner and not no_python:
            if manifest.runner.is_packaged:
                wp = cache.verify_files(snapshot, wheel_entries)
                if wp:
                    for p in wp[:20]:
                        typer.secho(f"  {p}", fg=typer.colors.RED)
                    raise _err(f"Runner wheel integrity check failed ({len(wp)} problem(s)).")
                if not runtime.ttnn_importable(python_exe):
                    tgt = python_exe or "this interpreter"
                    typer.secho(
                        f"  ! ttnn is not importable from {tgt}; the runner will install but "
                        "not run there. Use --python to target the tt-metal venv.",
                        fg=typer.colors.YELLOW,
                    )
                wheels = [snapshot / e.path for e in wheel_entries]
                try:
                    typer.echo(f"Installing runner: {manifest.runner.spec} ({len(wheels)} wheel(s)) ...")
                    runtime.pip_install_wheels(wheels, python=python_exe)
                    runner_installed = True
                    runner_ready = True
                    typer.secho("✓ runner installed", fg=typer.colors.GREEN)
                except Exception as exc:  # noqa: BLE001 — record partial progress, don't roll back kernels
                    _record_pull(repo_id, manifest, out_root, runner_installed=False,
                                 weights_path=None, last_error=f"pip install failed: {exc}")
                    raise _err(
                        f"Kernels are installed, but the runner pip install failed: {exc}\n"
                        f"  Re-run `tt-model pull {repo_id} --no-weights` to retry just the runner."
                    )
            else:
                # Reference runner: nothing ships in the bundle; confirm it's importable.
                if runtime.runner_spec_importable(manifest.runner.spec, python_exe):
                    runner_ready = True
                    typer.secho(
                        f"✓ runner {manifest.runner.spec} resolved (reference; not shipped)",
                        fg=typer.colors.GREEN,
                    )
                else:
                    src = f" Install it from {manifest.runner.source}." if manifest.runner.source else ""
                    typer.secho(
                        f"  ! runner {manifest.runner.spec} is a reference (not shipped) and is "
                        f"not importable in the target env.{src}",
                        fg=typer.colors.YELLOW,
                    )

        # Weights: download into a resolvable models dir (resumable).
        if manifest.weights and not no_weights:
            dest = runtime.resolve_models_dir(models_dir, manifest.weights.repo_id)
            try:
                typer.echo(f"Downloading weights {manifest.weights.repo_id} -> {dest} ...")
                weights_path = runtime.download_weights(manifest.weights, dest)
                typer.secho(f"✓ weights -> {weights_path}", fg=typer.colors.GREEN)
            except Exception as exc:  # noqa: BLE001 — kernels+runner remain usable
                _record_pull(repo_id, manifest, out_root, runner_installed=runner_installed,
                             weights_path=None, last_error=f"weights download failed: {exc}")
                raise _err(
                    f"Kernels{' and runner' if runner_installed else ''} installed, but the "
                    f"weights download failed: {exc}\n"
                    f"  Re-run `tt-model pull {repo_id}` to resume the download."
                )

    _record_pull(repo_id, manifest, out_root, runner_installed=runner_installed,
                 weights_path=weights_path, last_error=None)

    # Ready-to-run guidance.
    _routine(f"✓ Installed {repo_id}")
    if manifest.runner and runner_ready and weights_path is not None:
        typer.echo("\nRun it:")
        typer.secho("  " + runtime.serve_command(manifest.runner.spec, weights_path),
                    fg=typer.colors.CYAN)
        if not runtime.legacy_serve_available():
            typer.secho("  (the legacy-runner server needs fastapi + uvicorn: "
                        "pip install 'tt-model[serve]')",
                        fg=typer.colors.YELLOW)
    elif manifest.runner:
        missing = []
        if not runner_ready:
            if manifest.runner.is_packaged:
                missing.append("runner (re-run without --no-python)")
            else:
                src = f" — install from {manifest.runner.source}" if manifest.runner.source else ""
                missing.append(f"runner {manifest.runner.spec} (reference{src})")
        if weights_path is None and manifest.weights:
            missing.append("weights (re-run without --no-weights)")
        if missing:
            typer.secho(f"  pending: {', '.join(missing)}", fg=typer.colors.YELLOW)


def _record_pull(repo_id, manifest, out_root, *, runner_installed, weights_path, last_error,
                 bundle_path=None, instance=None) -> None:
    """Write the install binding to the local index (overwrites on re-pull).

    For a v4 vLLM bundle, ``instance`` is the selected tt-metal activation, pinned here so
    ``serve`` replays exactly the build ``pull`` resolved against. The declared ranges are
    stored too, so a stale pin (build removed) can be re-resolved gracefully at serve time.
    """
    ttnn_range, vllm_range, plugin_range = _manifest_ranges(manifest)
    entry = {
        "name": manifest.name,
        "build_key": manifest.build_key,
        "arch": manifest.arch,
        "tt_metal_version": manifest.tt_metal_version,
        "out_root": out_root,
        "schema_version": manifest.schema_version,
        "runner_spec": manifest.runner.spec if manifest.runner else None,
        "entry_point": manifest.runner.entry_point if manifest.runner else None,
        "backend": manifest.runner.backend if manifest.runner else None,
        "bundle_path": str(bundle_path) if bundle_path else None,
        "weights_repo": manifest.weights.repo_id if manifest.weights else None,
        "weights_path": str(weights_path) if weights_path else None,
        "python_installed": runner_installed,
        "weights_installed": weights_path is not None,
        "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # Pinned tt-metal instance (v4) + the ranges, for serve replay / re-resolve.
        "instance_name": instance.name if instance else None,
        "instance_python": instance.python if instance else None,
        "instance_tt_metal_home": instance.tt_metal_home if instance else None,
        "instance_env": dict(instance.env) if instance else None,
        "platform_ttnn": ttnn_range,
        "runtime_version": vllm_range,
        "runtime_plugin_version": plugin_range,
    }
    if last_error:
        entry["last_error"] = last_error
    localdb.record(repo_id, entry)


def _manifest_ranges(manifest):
    """The (ttnn, vllm, plugin) ranges a v4 manifest declares (any may be None)."""
    ttnn = manifest.platform.ttnn if manifest.platform else None
    vllm = manifest.runtime.version if manifest.runtime else None
    plugin = manifest.runtime.plugin_version if manifest.runtime else None
    return ttnn, vllm, plugin


def _select_instance(manifest, *, arch, instance_override, force):
    """Resolve which tt-metal instance a v4 bundle should link to, and the LocalEnv to gate on.

    Returns ``(instance | None, LocalEnv)``. The env carries hardware facts (arch/device_count
    from tt-smi, build-independent) with the ttnn/vLLM/plugin versions overridden to the
    **selected** instance's, so ``compare()`` gates on what will actually run. When no instance
    satisfies the ranges: with ``--force`` we fall back to the active env (loud warning); without
    it the caller blocks. A v3 / range-less bundle selects nothing (active env, as before).
    """
    base = metal.local_env(arch_override=arch, probe=False)
    if not (manifest.is_v4 and (manifest.platform or manifest.runtime)):
        return None, base

    ttnn, vllm, plugin = _manifest_ranges(manifest)
    if instance_override:
        match = next((i for i in instances.all_instances() if i.name == instance_override), None)
        if match is None:
            raise _err(f"--instance {instance_override!r} not found. See `tt-model instances list`.")
        chosen, v = match, instances.probe_versions(match)
    else:
        result = instances.select(ttnn=ttnn, vllm=vllm, plugin=plugin)
        if result.chosen is None:
            typer.secho(f"  ! {result.reason}", fg=typer.colors.YELLOW)
            for c in result.candidates:
                typer.secho(f"      - {c.instance.name}: ttnn={c.versions.ttnn} "
                            f"vllm={c.versions.vllm} plugin={c.versions.plugin}",
                            fg=typer.colors.YELLOW)
            if not force:
                raise _err("No installed tt-metal instance satisfies this model's ranges. "
                           "Install/register one (`tt-model instances add`), or --force to "
                           "install against the active environment anyway.")
            chosen = instances.active_instance()
            v = instances.probe_versions(chosen)
            typer.secho(f"  ! --force: linking to the active environment ({chosen.python})",
                        fg=typer.colors.YELLOW)
        else:
            chosen, v = result.chosen, next(c.versions for c in result.candidates
                                            if c.instance is result.chosen)
        _routine(f"  tt-metal instance: {chosen.name} "
                 f"(ttnn={v.ttnn}, vllm={v.vllm}, plugin={v.plugin})", fg=typer.colors.CYAN)

    # A probe that fails (missing .so, NFS timeout — all swallowed) yields None; do NOT let it
    # clobber the really-installed version, or the range gate silently becomes a no-op and an
    # incompatible bundle installs clean. Fall back to the detected local value.
    base.tt_metal_version = v.ttnn or base.tt_metal_version
    base.vllm_version = v.vllm or base.vllm_version
    base.vllm_plugin_version = v.plugin or base.vllm_plugin_version
    return chosen, base


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
    env = metal.local_env(arch_override=arch, probe=False)
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


def _install_vllm_bundle(
    repo_id, snapshot, manifest, *, force, arch, models_dir, bundles_dir, with_weights,
    instance=None,
) -> None:
    """Install a kernels-less vLLM bundle: verify + lay the model folder into bundles_dir.

    No tt-metal cache is touched. Optionally downloads weights (default: skip — the model
    class fetches them from the HF id at load). Records the install for `run`/`serve`/`rm`.
    """
    # For a v4 bundle, resolve which installed tt-metal instance satisfies its ranges and gate
    # on THAT instance's versions; arch stays the only fatal gate (see manifest.compare).
    selected, env = _select_instance(manifest, arch=arch, instance_override=instance, force=force)
    report = compare(manifest, env)
    _print_report(report)
    _warn_toolchain()
    if report.has_fatal:
        raise _err("Refusing to install: fatal incompatibility (see above).")
    if report.issues and not force:
        raise _err("Refusing to install: re-run with --force to override the warnings above.")

    subdir = manifest.runner.bundle_dir or "vllm_bundle"
    staged = snapshot / subdir

    # Integrity-verify whatever source the bundle actually ships (adapter code + wheels, and
    # for the legacy path the author-written vllm_metadata.json). For a v4 bundle the plugin
    # file is rendered by tt-model, not shipped, so it is simply absent from the index.
    bundle_entries = [f for f in manifest.files if f.path.startswith(f"{subdir}/")]
    # Fail closed: a shipped folder that the manifest doesn't index would otherwise be copied
    # into EXTRA_MODELS_DIR (which the plugin imports as Python) with NO integrity check — the
    # exact hole a tampered/emptied `files` list opens. If code is present, it must be indexed.
    if staged.is_dir() and any(staged.iterdir()) and not bundle_entries:
        raise _err(
            f"Bundle ships a {subdir}/ folder but the manifest indexes no files for it — "
            "refusing to install unverified code. Re-publish with a current tt-model."
        )
    if bundle_entries:
        # verify_files takes paths relative to a root; the entry paths carry the subdir prefix,
        # so verify against the snapshot root.
        problems = cache.verify_files(snapshot, bundle_entries)
        if not problems:
            console.milestone(f"verified {len(bundle_entries)} bundle file(s)")
        if problems:
            for p in problems[:20]:
                typer.secho(f"  {p}", fg=typer.colors.RED)
            raise _err(f"Integrity check failed ({len(problems)} problem(s)).")

    bdir = bundles.resolve_bundles_dir(bundles_dir)
    key = bundles.model_key(repo_id)

    if manifest.is_v4:
        # v4: tt-model is the source of truth — lay down any shipped code, then RENDER
        # vllm_metadata.json from the one authoritative manifest. `is_v4` is true for a
        # platform-only manifest too, but rendering needs an entrypoint — guard it with a
        # clean CLI error instead of letting render_vllm_metadata raise a raw ValueError.
        if manifest.entrypoint is None:
            raise _err(
                "v4 bundle manifest has no 'entrypoint' — cannot render vllm_metadata.json. "
                "The bundle is malformed; re-publish it with a current tt-model."
            )
        if staged.is_dir():
            dest = bundles.install_bundle(staged, bdir, key)
        else:  # entrypoint is a tt-metal built-in; no code shipped — just make the folder.
            dest = bdir / key
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
        bundles.write_vllm_metadata(dest, bundles.render_vllm_metadata(manifest))
        _routine(f"✓ vLLM bundle (rendered vllm_metadata.json) -> {dest}")
    else:
        # Legacy: ship the author-written vllm_metadata.json verbatim.
        if not (staged / bundles.VLLM_METADATA_NAME).is_file():
            raise _err(f"Bundle is missing its folder {subdir}/{bundles.VLLM_METADATA_NAME}.")
        dest = bundles.install_bundle(staged, bdir, key)
        _routine(f"✓ vLLM bundle -> {dest}")

    md = bundles.read_vllm_metadata(dest)
    _routine(f"  registers {md.arch} -> {md.main_class}", fg=typer.colors.CYAN)

    weights_path = None
    if with_weights and manifest.weights:
        wdest = runtime.resolve_models_dir(models_dir, manifest.weights.repo_id)
        try:
            typer.echo(f"Downloading weights {manifest.weights.repo_id} -> {wdest} ...")
            weights_path = runtime.download_weights(manifest.weights, wdest)
            typer.secho(f"✓ weights -> {weights_path}", fg=typer.colors.GREEN)
        except Exception as exc:  # noqa: BLE001 — bundle is still usable (model self-fetches)
            typer.secho(f"  ! weights download failed (model will fetch at load): {exc}",
                        fg=typer.colors.YELLOW)

    _record_pull(repo_id, manifest, out_root="", runner_installed=False,
                 weights_path=weights_path, last_error=None, bundle_path=str(dest),
                 instance=selected)
    _routine(f"✓ Installed {repo_id}")
    _routine(f"  Serve it:  tt-model serve {repo_id}", fg=typer.colors.CYAN)


# --------------------------------------------------------------------- instances
@instances_app.command("list")
def instances_list(
    for_repo: Optional[str] = typer.Option(
        None, "--for", help="Mark which instances satisfy this bundle's manifest ranges."
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Re-probe versions (ignore the cache)."),
) -> None:
    """List the tt-metal instances discovered on this host (active + registry + scan)."""
    ranges = (None, None, None)
    show_compat = False  # only mark ✓/✗ when we actually retrieved the bundle's requirements
    if for_repo:
        try:
            m = hub.fetch_manifest(*_split_revision(for_repo))
            ranges = _manifest_ranges(m)
            show_compat = True
        except Exception as exc:  # noqa: BLE001
            # Don't fall through and green-check every row against empty requirements we never
            # fetched — just list the instances and say the comparison is unavailable.
            typer.secho(f"  ! could not fetch {for_repo}: {exc} (skipping compatibility column)",
                        fg=typer.colors.YELLOW)
    insts = instances.all_instances()
    if not insts:
        typer.echo("No tt-metal instances found.")
        return
    for inst in insts:
        v = instances.probe_versions(inst, use_cache=not refresh)
        line = (f"[{inst.source}] {inst.name}: ttnn={v.ttnn or '—'} vllm={v.vllm or '—'} "
                f"plugin={v.plugin or '—'}  ({inst.python})")
        color = None
        if show_compat:
            ok = (toolchain.version_satisfies(v.ttnn, ranges[0]) is not False
                  and toolchain.version_satisfies(v.vllm, ranges[1]) is not False
                  and toolchain.version_satisfies(v.plugin, ranges[2]) is not False)
            line = ("✓ " if ok else "✗ ") + line
            color = typer.colors.GREEN if ok else typer.colors.RED
        typer.secho(line, fg=color)
    # Surface checkouts found by scan that can't be launched (no interpreter), for visibility.
    for home, py in instances.scan_checkouts():
        if py is None:
            typer.secho(f"[scan] {home}: found but no interpreter (build/python_env missing) — "
                        "register manually with `tt-model instances add`", fg=typer.colors.YELLOW)


@instances_app.command("add")
def instances_add(
    name: str = typer.Option(..., "--name", help="A short name for this instance."),
    python: str = typer.Option(..., "--python", help="Path to the instance's Python interpreter."),
    tt_metal_home: Optional[str] = typer.Option(None, "--tt-metal-home", help="TT_METAL_HOME for it."),
    env: Optional[List[str]] = typer.Option(
        None, "--env", help="Extra activation env as KEY=VALUE (repeatable), e.g. --env LD_LIBRARY_PATH=..."
    ),
) -> None:
    """Register a tt-metal instance the manager will consider for selection."""
    env_map = {}
    for item in env or []:
        if "=" not in item:
            raise _err(f"--env {item!r} must be KEY=VALUE.")
        k, val = item.split("=", 1)
        env_map[k.strip()] = val
    inst = instances.add_instance(name, python, tt_metal_home=tt_metal_home, env=env_map)
    typer.secho(f"✓ Registered instance {inst.name} -> {inst.python}", fg=typer.colors.GREEN)


@instances_app.command("remove")
def instances_remove(
    name: str = typer.Argument(..., help="Name of the registered instance to remove."),
) -> None:
    """Remove a registered tt-metal instance (scan/active instances can't be removed)."""
    if instances.remove_instance(name):
        typer.secho(f"✓ Removed instance {name}", fg=typer.colors.GREEN)
    else:
        raise _err(f"No registered instance named {name!r}.")


@instances_app.command("scan")
def instances_scan(
    refresh: bool = typer.Option(True, "--refresh/--no-refresh", help="Re-probe versions."),
) -> None:
    """Auto-discover tt-metal checkouts under the scan roots and report them."""
    pairs = instances.scan_checkouts()
    if not pairs:
        typer.echo("No tt-metal checkouts found under the scan roots.")
        return
    for home, py in pairs:
        if py is None:
            typer.secho(f"  {home}: found (no interpreter — register manually)", fg=typer.colors.YELLOW)
            continue
        inst = instances.Instance(name=f"scan:{Path(home).name}", python=py,
                                  tt_metal_home=home, source="scan")
        v = instances.probe_versions(inst, use_cache=not refresh)
        typer.secho(f"  {home}: ttnn={v.ttnn or '—'} vllm={v.vllm or '—'} plugin={v.plugin or '—'} "
                    f"({py})", fg=typer.colors.GREEN)



# ---------------------------------------------------------------------------- dev
@dev_app.command("make-test-cache")
def dev_make_test_cache(
    root: str = typer.Argument("/tmp/ttk-test-cache", help="Cache root to create."),
    build_key: int = typer.Argument(4242, help="Numeric build_key directory name."),
    with_runner: bool = typer.Option(
        False, "--with-runner", help="Also emit a fake runner wheel, for a v2 round-trip."
    ),
) -> None:
    """Generate a synthetic tt-metal kernel cache for testing without hardware."""
    cache = devtools.make_test_cache(root, build_key, with_runner=with_runner)
    console.console.print(f"[success]✓[/success] synthetic cache at {cache.base}")
    tbl = console.check_table()
    console.check_row(tbl, "", "build_key", str(cache.build_key))
    console.check_row(tbl, "", "kernels", str(cache.kernel_count))
    console.check_row(tbl, "", "files", str(cache.file_count))
    console.print_table(tbl)
    if cache.wheel:
        # A filename in the fixed-width "found" column gets ellipsised, which is useless
        # for a path you need to copy.
        console.console.print(f"[success]✓[/success] runner wheel {cache.wheel}")
    for description, command in devtools.push_recipe(cache):
        console.console.print(f"\n[muted]{description}:[/muted]")
        console.raw(f"  {command}")

# --------------------------------------------------------------------------- login
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


# ------------------------------------------------------------------------- doctor
def _warn_toolchain(python: Optional[str] = None) -> None:
    """Warn (never abort) about an inadequate surrounding toolchain. Called by run/pull
    so a version skew is surfaced without blocking the user's action.

    *python* is the interpreter that will actually serve (a selected instance's, when one is
    resolved). Without it the warning describes tt-model's own venv, which for a pipx or
    manager install is not where vLLM lives — the plugin then reads as missing on a host
    that serves fine. `serve`'s hard error already checks the instance interpreter, so
    omitting this made the warning contradict the error a few lines below it."""
    for c in toolchain.check_toolchain(python).problems:
        typer.secho(f"  ! {c.name}: {c.message}", fg=typer.colors.YELLOW)


def _report_bundle_requirements(repo_id: str, arch: Optional[str]) -> bool:
    """Declaratively resolve a v4 bundle's platform/runtime ranges against the local env.

    Prints required-vs-installed for each declared range and, when unsatisfied, the exact
    hint to get compatible — but NEVER installs anything (provisioning is out of scope; that
    is a future tt-cli concern). A v3 bundle (no ranges) just echoes its recorded versions.

    Returns ``True`` when every declared range is satisfied (or can't be assessed) and
    ``False`` when at least one installed version is decisively out of range — so ``doctor``
    can honor its documented "exits non-zero if below the required version" contract instead
    of printing a ✗ and exiting 0.
    """
    try:
        manifest = hub.fetch_manifest(repo_id, None)
    except Exception as exc:  # noqa: BLE001 — can't assess; don't fail the gate on a fetch error
        typer.secho(f"  ! could not fetch manifest for {repo_id}: {exc}", fg=typer.colors.YELLOW)
        return True
    env = metal.local_env(arch_override=arch, probe=False)
    typer.secho(f"\nBundle requirements — {repo_id}:", bold=True)
    unmet = False

    def _line(label: str, spec: Optional[str], installed: Optional[str], hint: str) -> None:
        nonlocal unmet
        if not spec:
            return
        ok = toolchain.version_satisfies(installed, spec)
        if ok is None:
            mark, color, note = "?", typer.colors.YELLOW, "(installed version unresolvable — assuming ok)"
        elif ok:
            mark, color, note = "✓", typer.colors.GREEN, ""
        else:
            mark, color, note = "✗", typer.colors.RED, f"-> {hint}"
            unmet = True
        typer.secho(f"  {mark} {label}: require {spec}, installed {installed or '—'} {note}", fg=color)

    if manifest.platform and manifest.platform.ttnn:
        _line("ttnn (tt-metal)", manifest.platform.ttnn, env.tt_metal_version,
              "install a tt-metal/ttnn in range (see scripts/install.sh)")
    if manifest.runtime and manifest.runtime.version:
        _line(f"{manifest.runtime.kind}", manifest.runtime.version, env.vllm_version,
              "install the Tenstorrent vLLM fork+plugin in range (see scripts/install.sh)")
    if manifest.runtime and manifest.runtime.plugin_version:
        _line(f"{manifest.runtime.kind} plugin", manifest.runtime.plugin_version,
              env.vllm_plugin_version,
              "install a matching vllm_tt_plugin (see scripts/install.sh)")
    if manifest.target:
        typer.secho(f"  · target: {manifest.target}  (detected arch={env.arch or '—'}, "
                    f"devices={env.device_count})", fg=typer.colors.CYAN)
    if not (manifest.platform or manifest.runtime):
        typer.secho(f"  · authored against tt-metal {manifest.tt_metal_version} "
                    "(v3 bundle — no version ranges declared)", fg=typer.colors.CYAN)
    # Which installed tt-metal instance would serve this model. This block sat *after* the
    # `return` below and so had never run: `doctor <id>` has never printed the
    # instance-selection line it was written to print.
    if manifest.platform or manifest.runtime:
        ttnn, vllm, plugin = _manifest_ranges(manifest)
        result = instances.select(ttnn=ttnn, vllm=vllm, plugin=plugin)
        if result.chosen:
            typer.secho(f"  → would link to instance: {result.reason}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"  → no instance selectable: {result.reason} "
                        "(register one with `tt-model instances add`)", fg=typer.colors.YELLOW)

    return not unmet


@app.command(rich_help_panel="Environment")
def doctor(
    repo_id: Optional[str] = typer.Argument(
        None, help="Optional bundle id: also report its declared platform/runtime ranges "
        "against what's installed (declarative — never installs)."
    ),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
) -> None:
    """Report whether the surrounding toolchain (tt-metal, vLLM) and hardware are adequate.

    Purely declarative: `doctor` only ever checks and reports, which is what makes its verdict a reliable description of the machine. To actually provision the stack, use `tt-model install`.

    With a bundle id, also resolves that bundle's v4 version ranges against the local
    environment and prints what (if anything) is out of range.

    Exits non-zero if any component is missing or below the required version.
    """
    report = toolchain.check_toolchain()
    console.console.print("[bold accent]Toolchain[/bold accent]")
    # Aligned columns. This used to stack three em-dashes into one line —
    # "✗ tt-metal: — (require >= 0.72.0) — not found — install tt-metal >= 0.72.0" —
    # where the bare "—" was the *version* and the reader had to parse punctuation to
    # find the verdict.
    table = console.check_table()
    for c in report.components:
        mark = "[success]✓[/success]" if c.adequate else "[error]✗[/error]"
        # Keep the full version under --verbose: the git hash is what you need when you
        # are actually chasing a version skew.
        console.check_row(table, mark, c.name,
                          console.fmt_version(c.version, keep_local=console.is_verbose()),
                          f"require >= {c.required}")
    console.print_table(table)
    for c in report.components:
        if not c.adequate:
            # The row already shows "not found"; keep only the actionable half of the
            # message so the hint says what to do rather than restating the verdict.
            msg = c.message
            for prefix in ("not found — ", "not found - ", "not found: "):
                if msg.startswith(prefix):
                    msg = msg[len(prefix):]
                    break
            console.note(msg, marker="!", style="warning")

    dev = metal.detect_device()
    console.console.print("\n[bold accent]Hardware[/bold accent]")
    if dev.arch:
        hw = console.check_table()
        console.check_row(hw, "[success]✓[/success]", dev.arch,
                          f"{dev.device_count} device(s)", f"via {dev.source}")
        console.print_table(hw)
    else:
        console.note("no Tenstorrent device detected (tt-smi/ARCH_NAME unavailable)",
                     marker="!", style="warning")

    # Environment coherence. Every individual version check can pass on an environment pip
    # already considers broken: ttnn pins numpy<2 while the vLLM fork's opencv wants
    # numpy>=2, so installing both satisfied each check and still printed "adequate" one
    # line after pip printed a hard ERROR. Advisory, not blocking — the conflicting package
    # may be an extra the TT path never imports, and we cannot know that from here.
    conflicts = toolchain.check_environment()
    if conflicts:
        console.console.print("\n[bold accent]Environment[/bold accent]")
        env = console.check_table()
        for c in conflicts:
            console.check_row(env, "[warning]![/warning]", c.package,
                              c.installed or "not installed",
                              f"requires {c.requirement}")
        console.print_table(env)
        console.hint("harmless if the TT serving path never imports these — but if serving "
                     "fails on an import, start here")

    reqs_ok = True
    if repo_id:
        reqs_ok = _report_bundle_requirements(repo_id, arch)

    if not report.ok or not reqs_ok:
        raise typer.Exit(code=1)
    if conflicts:
        # Don't claim "adequate" over a named conflict; don't fail on an advisory either.
        console.console.print(
            f"\n[success]✓ toolchain adequate[/success]"
            f"[muted] — with {len(conflicts)} environment conflict(s) above[/muted]"
        )
        return
    console.console.print("\n[success]✓ toolchain adequate[/success]")


# ----------------------------------------------------------------------------- run
def _handoff(argv: List[str], *, print_only: bool, why: str) -> None:
    """Print or execute the legacy-runner server handoff (``tt_kernel.legacy_serve``).
    Execution replaces this process's foreground with the server (blocks until it exits)."""
    typer.secho(f"[{why}]", fg=typer.colors.CYAN)
    if print_only:
        typer.echo(" ".join(argv))
        return
    if not runtime.legacy_serve_available():
        raise _err(
            "Cannot serve: the legacy-runner server needs fastapi + uvicorn "
            "(pip install 'tt-model[serve]'). Use `--print` to emit the command."
        )
    try:
        raise typer.Exit(code=subprocess.run(argv).returncode)
    except KeyboardInterrupt:  # graceful Ctrl-C of the served process
        raise typer.Exit(code=130)


def _endpoint_from_command(command: List[str]) -> str:
    """Best-effort OpenAI endpoint URL from a launch command's --host/--port (default 8000)."""
    host, port = "localhost", "8000"
    for i, tok in enumerate(command):
        if tok == "--port" and i + 1 < len(command):
            port = command[i + 1]
        elif tok.startswith("--port="):
            port = tok.split("=", 1)[1]
        elif tok in ("--host",) and i + 1 < len(command):
            h = command[i + 1]
            host = "localhost" if h in ("0.0.0.0", "") else h
        elif tok.startswith("--host="):
            h = tok.split("=", 1)[1]
            host = "localhost" if h in ("0.0.0.0", "") else h
    return f"http://{host}:{port}"


def _ensure_vllm_pulled(repo_id: str, revision: Optional[str], *, arch: Optional[str],
                        bundles_dir: Optional[str], force: bool = False,
                        instance: Optional[str] = None) -> dict:
    """Return the local install entry for a vLLM bundle, pulling it first if absent."""
    entry = localdb.get(repo_id)
    if entry and entry.get("bundle_path") and Path(entry["bundle_path"]).is_dir():
        return entry
    with tempfile.TemporaryDirectory() as td:
        snapshot = _hub(lambda: hub.download_bundle(repo_id, revision, dest=td),
                        repo_id, what="Pull",
                        consequence="Nothing was installed.")
        mpath = snapshot / MANIFEST_NAME
        if not mpath.is_file():
            raise _err(f"{repo_id} is not a tt-model bundle (no {MANIFEST_NAME}).")
        manifest = Manifest.from_json(mpath.read_text())
        if not (manifest.runner and manifest.runner.is_vllm):
            raise _err(f"{repo_id} is not a vLLM bundle.")
        _install_vllm_bundle(repo_id, snapshot, manifest, force=force, arch=arch,
                             models_dir=None, bundles_dir=bundles_dir, with_weights=False,
                             instance=instance)
    entry = localdb.get(repo_id)
    if not entry or not entry.get("bundle_path"):
        raise _err(f"Failed to install vLLM bundle {repo_id}.")
    return entry


def _serve_activation(entry: dict, *, instance_override: Optional[str]):
    """Resolve the tt-metal activation to launch under: ``(python|None, activation_env, label)``.

    Replays the instance ``pull`` pinned into ``entry``. Graceful degradation:
    ``--instance`` wins outright; a pin whose interpreter still exists is used as-is; a pin
    that vanished is **re-resolved** from the stored ranges (newest satisfying); and when
    nothing is pinned or resolvable we fall back to the ambient env (``python=None``) — which
    is exactly the pre-registry behavior, so v3 / range-less bundles are unaffected.
    """
    if instance_override:
        match = next((i for i in instances.all_instances() if i.name == instance_override), None)
        if match is None:
            raise _err(f"--instance {instance_override!r} not found. See `tt-model instances list`.")
        return match.python, match.activation_env(), match.name

    py = entry.get("instance_python")
    if py and Path(py).exists():
        env = dict(entry.get("instance_env") or {})
        home = entry.get("instance_tt_metal_home")
        if home:
            env.setdefault("TT_METAL_HOME", home)
        return py, env, entry.get("instance_name") or py
    if py:  # pinned build is gone — re-resolve from the recorded ranges
        typer.secho(f"  ! pinned tt-metal instance missing ({py}); re-resolving from ranges...",
                    fg=typer.colors.YELLOW)
        result = instances.select(ttnn=entry.get("platform_ttnn"),
                                  vllm=entry.get("runtime_version"),
                                  plugin=entry.get("runtime_plugin_version"))
        if result.chosen:
            typer.secho(f"  re-resolved -> {result.chosen.name}", fg=typer.colors.CYAN)
            return result.chosen.python, result.chosen.activation_env(), result.chosen.name
        typer.secho("  ! no instance satisfies the ranges; using the active environment "
                    "(consider `tt-model pull` again).", fg=typer.colors.YELLOW)
    return None, {}, None


def _serve_preflight(endpoint: str, inst_python: Optional[str],
                     main_class: Optional[str] = None,
                     adapter_paths: Optional[List[str]] = None) -> None:
    """Everything knowable before vLLM is launched, checked before anything is promised.

    Two conditions, both of which previously surfaced only after the CLI had already
    announced an endpoint — and the second only after ~18 seconds of plugin and ttnn
    loading, as a traceback from inside vLLM.
    """
    # 1. Can this interpreter even serve? Check the one that will actually run (a pinned
    # instance's, if any) — not the manager's, which may be a pipx venv with no vLLM.
    if not runtime.vllm_available(python=inst_python):
        where = f"the pinned instance ({inst_python})" if inst_python else "this environment"
        raise _fail_card("Preflight", {
            "cause": "the Tenstorrent vLLM stack is not importable",
            "detail": f"vllm + vllm_tt_plugin are not importable in {where}, so there is "
                      "nothing to serve the model with.",
            "evidence": "",
            "actions": [f"{'tt-model install':<32}# set up the serving stack",
                        f"{'tt-model serve <id> --print':<32}# emit the command for another box"],
        }, consequence="vLLM was not started.")

    # 2. Is the model's serving adapter actually present? The bundle names a class like
    # models.autoports.<model>.tt.generator_vllm:Cls. Report the first dotted segment that
    # does not resolve, because "no models tree at all" and "the tree is here but this
    # adapter is not" have different fixes. Without this the failure arrives as an
    # ImportError from inside vLLM after startup.
    missing = runtime.missing_adapter_segment(main_class, inst_python,
                                              search_paths=adapter_paths)
    if missing:
        module = runtime.adapter_module(main_class) or ""
        where = _short_path(inst_python or "this environment")
        if missing == module.split(".", 1)[0]:
            detail = (f"{missing!r} is not importable in {where}. The ttnn PyPI wheel ships "
                      "no models/ tree, so a green toolchain is not enough here.")
            actions = [f"{'export PYTHONPATH=<tt-metal-checkout>':<44}# the tree lives there",
                       f"{'tt-model info <id>':<44}# what this bundle needs"]
        else:
            parent = missing.rsplit(".", 1)[0]
            detail = (f"{parent!r} resolves in {where}, but {missing!r} does not — the "
                      "models tree is present and this particular adapter is not part of it.")
            actions = [f"{'tt-model list':<44}# bundles that ship their own adapter",
                       f"{'check out the branch that adds ' + missing:<44}"]
        raise _fail_card("Preflight", {
            "cause": f"the serving adapter is missing ({missing})",
            "detail": detail,
            "evidence": f"main_class = {module}" if module else "",
            "actions": actions,
        }, consequence="vLLM was not started.")

    # 3. Is the port we are about to claim actually free? One syscall, versus 18 seconds of
    # startup followed by OSError: [Errno 98] from inside vLLM.
    port = runtime.port_of(endpoint)
    if port is not None and runtime.port_in_use(port):
        raise _fail_card("Preflight", {
            "cause": f"port {port} is already in use",
            "detail": f"Something is already listening on {port}.",
            "evidence": "",
            # Naming the holder is the job; killing a process tt-model does not own is not.
            "actions": [f"{'lsof -i :' + str(port):<32}# what is holding it",
                        f"{'tt-model serve <id> --port <n>':<32}# use another port"],
        }, consequence="vLLM was not started.")


def _serve_vllm(repo_id: str, revision: Optional[str], *, print_only: bool, local_only: bool,
                arch: Optional[str], bundles_dir: Optional[str], do_health: bool,
                force: bool = False, instance: Optional[str] = None,
                extra_args: Optional[List[str]] = None) -> None:
    """The vLLM one-command serve flow: pull-if-needed, launch, (optional) health, endpoint."""
    if local_only:
        entry = localdb.get(repo_id)
        if not entry or not entry.get("bundle_path"):
            raise _err(f"No installed vLLM bundle for {repo_id} (and --local-only forbids a pull).")
    else:
        entry = _ensure_vllm_pulled(repo_id, revision, arch=arch, bundles_dir=bundles_dir,
                                    force=force, instance=instance)

    bundle_path = Path(entry["bundle_path"])
    if not bundle_path.is_dir():
        raise _err(f"Installed bundle folder is missing: {bundle_path}. Re-run `tt-model pull {repo_id}`.")
    extra_models_dir = bundle_path.parent  # == EXTRA_MODELS_DIR (holds this model folder)
    md = bundles.read_vllm_metadata(bundle_path)
    mkey, launch = bundles.select_launch(md, arch)
    if launch is None:
        cands = ", ".join(bundles.machine_candidates(arch))
        raise _err(
            f"{bundles.VLLM_METADATA_NAME} has no launch command for this machine "
            f"(tried: {cands}). Add one, or set a 'default'."
        )

    inst_python, activation_env, inst_label = _serve_activation(entry, instance_override=instance)
    # Warn only now: before this line the interpreter that will serve is unknown, and the
    # check aimed at tt-model's own venv reported a missing plugin for instances that have
    # one. Same interpreter the hard error below uses, so the two cannot disagree.
    _warn_toolchain(inst_python)
    # Anything the user typed after the bundle id goes to vLLM, AFTER the bundle's own
    # command so a user --port/--host wins over the bundle default (argparse last-wins).
    argv = runtime.vllm_serve_argv(launch.command, python=inst_python) + list(extra_args or [])
    env = runtime.vllm_serve_env(extra_models_dir, launch.env, activation_env=activation_env)
    # Parse the FINAL argv, not launch.command: otherwise we announce the bundle's port
    # while vLLM binds the user's, and the preflight would check the wrong one.
    endpoint = _endpoint_from_command(argv)

    # If a pin was requested but the launch command's first token isn't a Python interpreter,
    # the interpreter can't be substituted — the process would run under the ambient one while
    # this build's env is exported. Warn loudly rather than silently mis-pin (see review G2).
    if inst_python and launch.command and not runtime.is_python_command(launch.command[0]):
        typer.secho(
            f"  ! instance {inst_label} was selected, but the bundle's launch command starts "
            f"with {launch.command[0]!r} (not a Python interpreter), so the pinned interpreter "
            "could not be applied — the server may run under the wrong tt-metal build.",
            fg=typer.colors.YELLOW,
        )

    # Preflight BEFORE announcing anything. The endpoint line used to print nine lines
    # above the vLLM-importable check, so `serve` promised "OpenAI endpoint (once up):
    # http://localhost:8000" and then said it could not serve at all.
    if not print_only:
        _serve_preflight(endpoint, inst_python, main_class=md.main_class,
                         adapter_paths=[str(bundle_path), str(extra_models_dir)])

    via = f"{mkey}" + (f"; instance={inst_label}" if inst_label else "")
    typer.secho(f"[vLLM: {md.arch} via {via}; EXTRA_MODELS_DIR={extra_models_dir}]",
                fg=typer.colors.CYAN)
    typer.secho(f"  OpenAI endpoint (once up): {endpoint}", fg=typer.colors.CYAN)
    if print_only:
        exports = " ".join(f"{k}={v}" for k, v in
                           {**activation_env, runtime.ENV_EXTRA_MODELS_DIR: str(extra_models_dir),
                            **launch.env}.items())
        console.raw(f"{exports} " + " ".join(argv))
        return
    try:
        raise typer.Exit(code=subprocess.run(argv, env=env).returncode)
    except KeyboardInterrupt:  # graceful Ctrl-C of the served process
        raise typer.Exit(code=130)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, rich_help_panel="Run a model")
def serve(
    ctx: typer.Context,
    repo_id: str = typer.Argument(..., help="vLLM bundle id (namespace/name[@rev]) to serve."),
    print_only: bool = typer.Option(False, "--print", help="Print the launch command instead of running it."),
    local_only: bool = typer.Option(False, "--local-only", help="Do not pull; require an installed bundle."),
    force: bool = typer.Option(False, "--force", help="Install despite non-fatal compatibility "
                               "warnings (e.g. an out-of-range ttnn/vLLM) when the pull happens here."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch/machine detection."),
    bundles_dir: Optional[str] = typer.Option(None, "--bundles-dir", help="Override EXTRA_MODELS_DIR location."),
    instance: Optional[str] = typer.Option(
        None, "--instance", help="Force a registered tt-metal instance by name (see "
        "`tt-model instances list`), overriding the pinned/auto-resolved one."
    ),
    health_check: bool = typer.Option(False, "--health-check", help="(reserved) probe the server after launch."),
    no_update_check: bool = typer.Option(
        False, "--no-update-check", help="Skip the best-effort check for a newer published "
        "bundle revision (that check makes one short, timeout-bounded Hub request)."
    ),
) -> None:
    """Serve a vLLM bundle through the Tenstorrent vLLM plugin (the primary path).

    One command: pull the bundle folder if needed, point EXTRA_MODELS_DIR at it, and launch
    the OpenAI-compatible server with the bundle's per-machine launch command. Repeat
    invocations skip the pull and go straight to launch.
    """
    repo_id, revision = _split_revision(repo_id)
    extra_args = list(ctx.args)  # anything after the bundle id is passed through to vLLM

    # Self-contained (v5) fast path: an already-installed bundle serves from its own venv. The host
    # toolchain (ttnn/vLLM versions) is irrelevant here — the bundle ships its own — so don't warn.
    entry = localdb.get(repo_id)
    if entry and entry.get("self_contained"):
        if not local_only and not no_update_check:
            _warn_if_update_available(repo_id, entry)
        _serve_self_contained(entry, print_only=print_only, extra_args=extra_args)
        return
    # Not installed yet: if the remote bundle is self-contained, install then serve it (unless
    # --local-only). A self-contained bundle never routes through the host-provisioned vLLM path.
    if not local_only:
        try:
            remote = hub.fetch_manifest(repo_id, revision)
        except Exception:  # noqa: BLE001 — fall back to the normal path if we can't peek
            remote = None
        if remote is not None and remote.has_own_venv:
            resolved = hub.latest_revision(repo_id, revision, timeout=None)  # resolve before fetch
            with tempfile.TemporaryDirectory() as td:
                snapshot = _hub(lambda: hub.download_bundle(repo_id, resolved or revision, dest=td),
                                repo_id, what="Pull",
                                consequence="Nothing was installed.")
                mani = Manifest.from_json((snapshot / MANIFEST_NAME).read_text())
                _install_self_contained(repo_id, snapshot, mani, force=force, arch=arch,
                                        models_dir=None, with_weights=False,
                                        revision=revision, resolved_revision=resolved)
            entry = localdb.get(repo_id)
            if entry and entry.get("self_contained"):
                _serve_self_contained(entry, print_only=print_only, extra_args=extra_args)
                return

    # (_serve_vllm warns once it knows which instance's interpreter will serve.)
    _serve_vllm(repo_id, revision, print_only=print_only, local_only=local_only,
                arch=arch, bundles_dir=bundles_dir, do_health=health_check,
                force=force, instance=instance, extra_args=extra_args)


@app.command(rich_help_panel="Run a model")
def run(
    repo_id: str = typer.Argument(
        ..., help="Model to run: a tt-model bundle id (namespace/name[@rev]) or a bare HF model id."
    ),
    print_only: bool = typer.Option(
        False, "--print", help="Print the serve command instead of executing it."
    ),
    local_only: bool = typer.Option(
        False, "--local-only", help="Do not query the Hub; resolve only against installed bundles."
    ),
) -> None:
    """Serve a model through the right path.

    - **vLLM bundle** -> the Tenstorrent vLLM plugin (the default; same as `tt-model serve`).
    - **legacy runner bundle** (a runner following the legacy contract in
      docs/authoring_runners.md), once installed -> tt-model's own OpenAI-compatible
      legacy-runner server (`tt_kernel.legacy_serve`).
    - anything else (kernels-only bundle, or a bare HF repo) -> not servable by tt-model;
      publish a vLLM bundle.

    The old dynamic dispatch path (`tt_api.serve`) is retired.
    """
    _warn_toolchain()
    repo_id, revision = _split_revision(repo_id)
    res = resolve_mod.resolve(repo_id, revision=revision, local_only=local_only)

    # vLLM bundle -> the plugin (default path).
    if res.is_vllm:
        _serve_vllm(repo_id, revision, print_only=print_only, local_only=local_only,
                    arch=None, bundles_dir=None, do_health=False)
        return

    # Legacy runner bundle -> tt-model's legacy-runner server. It needs the runner
    # installed and the weights on disk, so it only works once the bundle is pulled.
    if res.has_runner:
        if res.installed and res.weights_path:
            argv = runtime.serve_argv(res.weights_path, runner_spec=res.runner_spec,
                                      python=sys.executable)
            _handoff(argv, print_only=print_only,
                     why=f"legacy runner {res.runner_spec} via tt_kernel.legacy_serve")
            return
        if res.installed:
            raise _err(
                f"{repo_id} is installed but its weights are not on disk. Re-run "
                f"`tt-model pull {repo_id}` (without --no-weights) so the runner can load."
            )
        typer.secho(
            f"A tt-model bundle exists for {repo_id} (legacy runner {res.runner_spec}).",
            fg=typer.colors.YELLOW,
        )
        typer.secho(f"  Install it first:  tt-model pull {repo_id}", fg=typer.colors.YELLOW)
        typer.secho("  then `tt-model run` serves it via the legacy-runner server.",
                    fg=typer.colors.YELLOW)
        return

    # No runner (kernels-only bundle, or a bare HF repo): the dynamic dispatch path is
    # retired. tt-model serves vLLM bundles and legacy-runner bundles only.
    raise _err(
        f"Nothing to serve for {repo_id}. tt-model serves vLLM bundles "
        f"(`tt-model serve <id>`) and legacy-runner bundles. To serve this model, publish "
        "it as a vLLM bundle — see docs/authoring_runners.md."
    )


# ---------------------------------------------------------------------------- info
@app.command(rich_help_panel="Get models")
def info(
    repo_id: str = typer.Argument(..., help="Repo as namespace/name[@revision]."),
    arch: Optional[str] = typer.Option(None, "--arch", help="Override arch detection."),
    probe: bool = typer.Option(False, "--probe", help="Open a device to read the true build_key."),
) -> None:
    """Print a bundle's manifest and its compatibility verdict vs the local env."""
    repo_id, revision = _split_revision(repo_id)
    manifest = _hub(lambda: hub.fetch_manifest(repo_id, revision), repo_id,
                    what="Inspect")
    console.raw(manifest.to_json())
    typer.echo("")
    report = compare(manifest, metal.local_env(arch_override=arch, probe=probe))
    _print_report(report)


# ---------------------------------------------------------------------------- list
@app.command(name="list", rich_help_panel="Get models")
def list_installed() -> None:
    """List locally installed bundles."""
    entries = localdb.all_entries()
    if not entries:
        typer.echo("No bundles installed.")
        return
    for e in entries:
        backend = e.get("backend") or "dispatch"
        if backend == "vllm":
            typer.echo(
                f"{e['repo_id']}  backend=vllm  arch={e.get('arch')}  "
                f"bundle={e.get('bundle_path')}"
            )
        else:
            typer.echo(
                f"{e['repo_id']}  build_key={e.get('build_key')}  arch={e.get('arch')}  "
                f"tt_metal={e.get('tt_metal_version')}"
            )


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
    target: Optional[str] = typer.Option(
        None, "--target", help="Only bundles tagged for this machine target (e.g. p150x4) — "
        "'what runs on my box'. Matches the v4 manifest's target."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Search the Hub for published tt-model caches."""
    extra_tags = [t.lower() for t in (arch, target) if t]
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
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
) -> None:
    """Remove a locally installed bundle and its index entry.

    For a dispatch bundle this removes the kernel-cache subtree; for a vLLM bundle it
    removes the model folder from bundles_dir (EXTRA_MODELS_DIR).
    """
    entry = localdb.get(repo_id)
    if not entry:
        raise _err(f"{repo_id} is not recorded as installed.")

    # vLLM bundle: no cache subtree — remove the installed model folder instead.
    if (entry.get("backend") == "vllm") or entry.get("build_key") is None:
        bundle_path = entry.get("bundle_path")
        removed = False
        if bundle_path:
            p = Path(bundle_path)
            removed = bundles.remove_bundle(p.parent, p.name)
        localdb.remove(repo_id)
        if removed:
            typer.secho(f"✓ Removed vLLM bundle {repo_id} ({bundle_path})", fg=typer.colors.GREEN)
        else:
            typer.secho("Index entry removed; bundle folder was already gone.",
                        fg=typer.colors.YELLOW)
        return

    # The stored out_root is already a full prefix; only re-resolve if --cache-dir given.
    out_root = cache.resolve_out_root(cache_dir) if cache_dir else (
        entry.get("out_root") or cache.resolve_out_root(None)
    )
    removed = cache.remove_subtree(out_root, int(entry["build_key"]))
    localdb.remove(repo_id)
    if removed:
        typer.secho(f"✓ Removed {repo_id} (build_key {entry['build_key']})", fg=typer.colors.GREEN)
    else:
        typer.secho(
            "Index entry removed; cache subtree was already gone.", fg=typer.colors.YELLOW
        )


# --------------------------------------------------------------------------- clean
@app.command(rich_help_panel="Maintenance")
def clean(
    build_key: Optional[int] = typer.Option(
        None, "--build-key", help="Remove this build_key subtree from the cache."
    ),
    all_keys: bool = typer.Option(
        False, "--all", help="Remove ALL build_key subtrees under the cache root."
    ),
    cache_dir: Optional[str] = typer.Option(None, help="Override the tt-metal cache root."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt for --all."),
) -> None:
    """Clear kernel-cache subtrees to force a clean state before a run/produce.

    The tt-metal JIT cache is keyed by build_key (the build environment), and every model
    run on a build shares one subtree — so to produce a model-specific bundle you must start
    from a clean cache. Use this to wipe a stale subtree (or all of them) first:

      tt-model clean --build-key N            # remove one build_key subtree
      tt-model clean --all                    # remove every build_key subtree
      tt-model clean --all --cache-dir DIR    # ... under a specific cache root

    For removing an *installed bundle* (and its index entry), use `tt-model rm` instead.
    """
    if all_keys and build_key is not None:
        raise _err("Pass either --build-key N or --all, not both.")
    out_root = cache.resolve_out_root(cache_dir)
    keys = cache.list_build_keys(out_root)
    if all_keys:
        if not keys:
            typer.echo(f"No build_key subtrees under {out_root}; nothing to clean.")
            return
        if not yes:
            typer.confirm(
                f"Remove ALL {len(keys)} build_key subtree(s) under {out_root}?", abort=True
            )
        for k in keys:
            cache.remove_subtree(out_root, k)
        typer.secho(
            f"✓ removed {len(keys)} build_key subtree(s) from {out_root}", fg=typer.colors.GREEN
        )
    elif build_key is not None:
        if cache.remove_subtree(out_root, build_key):
            typer.secho(f"✓ removed build_key {build_key} from {out_root}", fg=typer.colors.GREEN)
        else:
            typer.secho(
                f"build_key {build_key} not present under {out_root}.", fg=typer.colors.YELLOW
            )
    else:
        raise _err("Specify --build-key N or --all.")


# ---------------------------------------------------------------------------- utils
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
