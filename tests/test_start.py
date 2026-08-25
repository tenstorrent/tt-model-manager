# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`tt-model start` — the guided flow, and the prompt paths where a wizard rots.

The failure mode of a badly-placed prompt is a *hang*, not an exception, so several of
these assert on completing at all rather than on output. They are cheap; a hang in CI is
not.
"""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from tt_kernel import cli, console, localdb, start, toolchain

runner = CliRunner()
CLI = [sys.executable, "-m", "tt_kernel.cli"]


@pytest.fixture(autouse=True)
def _isolate_bundles_dir(tmp_path, monkeypatch):
    """Point the on-disk bundle scan at an empty dir.

    `installed_choices` reads the real bundles directory to surface folders that have no
    index entry. Without this, whatever is installed on the developer's machine leaks into
    every test's expected choices. Tests that exercise the scan override it themselves.
    """
    import tt_kernel.bundles as _bundles

    empty = tmp_path / "bundles-empty"
    empty.mkdir()
    monkeypatch.setattr(_bundles, "resolve_bundles_dir", lambda *_a, **_k: empty)


# ── prompting policy ─────────────────────────────────────────────────────────
class TestPromptPolicy:
    def test_explicit_token_never_prompts(self, monkeypatch):
        monkeypatch.setattr(start.auth, "login", lambda token=None: None)
        monkeypatch.setattr(start.auth, "whoami", lambda: {"name": "me"})
        monkeypatch.setattr(start.console, "secret", _must_not_be_called)
        acct = start.resolve_account("hf_x")
        assert acct.logged_in and acct.source == "--token"

    def test_existing_identity_never_prompts(self, monkeypatch):
        monkeypatch.setattr(start.auth, "whoami", lambda: {"name": "me"})
        monkeypatch.setattr(start.console, "secret", _must_not_be_called)
        assert start.resolve_account().logged_in

    def test_prompt_suppressed_when_not_allowed(self, monkeypatch):
        """--yes and a non-TTY stdin must both reach this path. A prompt here would read
        EOF and silently take a default the user never saw."""
        monkeypatch.setattr(start.auth, "whoami", lambda: None)
        monkeypatch.setattr(start.console, "secret", _must_not_be_called)
        acct = start.resolve_account(allow_prompt=False)
        assert not acct.logged_in and acct.source == "none"

    def test_empty_prompt_is_not_treated_as_a_token(self, monkeypatch):
        monkeypatch.setattr(start.auth, "whoami", lambda: None)
        monkeypatch.setattr(start.console, "secret", lambda p: "   ")
        monkeypatch.setattr(start.auth, "login", _must_not_be_called)
        assert not start.resolve_account(allow_prompt=True).logged_in

    def test_token_is_never_echoed_into_output(self, monkeypatch):
        """A secret that reaches the renderer reaches scrollback and any log."""
        monkeypatch.setattr(start.auth, "login", lambda token=None: None)
        monkeypatch.setattr(start.auth, "whoami", lambda: {"name": "me"})
        acct = start.resolve_account("hf_SUPERSECRET")
        assert "hf_SUPERSECRET" not in repr(acct)
        assert "hf_SUPERSECRET" not in str(acct.__dict__)


def _must_not_be_called(*a, **k):
    raise AssertionError("prompted when it must not")


def test_stdin_detection_survives_a_closed_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", None)
    assert start.stdin_is_interactive() is False


# ── the guided flow does not hang ────────────────────────────────────────────
class TestNoHang:
    def _env(self):
        return dict(os.environ, COLUMNS="100")

    def test_non_tty_stdin_completes(self):
        """The canonical wizard bug: a prompt inside a capturing step, or on a piped stdin,
        hangs instead of failing. Bounded so a regression shows up as a failure."""
        res = subprocess.run(CLI + ["start", "nope/nope", "--print"],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True,
                             timeout=120, env=self._env())
        assert res.returncode is not None

    def test_yes_flag_completes(self):
        res = subprocess.run(CLI + ["start", "nope/nope", "--print", "--yes"],
                             capture_output=True, text=True, timeout=120, env=self._env())
        assert res.returncode is not None


# ── bundle resolution ────────────────────────────────────────────────────────
class TestResolveBundle:
    def test_an_installed_id_resolves_to_itself(self, monkeypatch):
        monkeypatch.setattr(start.localdb, "get", lambda rid: {"repo_id": rid})
        assert start.resolve_bundle("org/m") == ("org/m", "installed")

    def test_a_bare_model_id_finds_an_installed_bundle(self, monkeypatch):
        """`tt-model start Qwen/Qwen3-32B` should find mando2222/Qwen3-32B-blackhole rather
        than trying to pull a bundle id the user never typed."""
        monkeypatch.setattr(start.localdb, "get", lambda rid: None)
        monkeypatch.setattr(start.localdb, "all_entries",
                            lambda: [{"repo_id": "mando2222/Qwen3-32B-blackhole"}])
        repo, how = start.resolve_bundle("Qwen/Qwen3-32B")
        assert repo == "mando2222/Qwen3-32B-blackhole"
        assert "matching" in how

    def test_an_unknown_id_is_left_to_pull(self, monkeypatch):
        monkeypatch.setattr(start.localdb, "get", lambda rid: None)
        monkeypatch.setattr(start.localdb, "all_entries", lambda: [])
        assert start.resolve_bundle("org/new") == ("org/new", "to pull")


# ── validation gate ──────────────────────────────────────────────────────────
def _report(ok=True):
    return toolchain.ToolchainReport(components=[
        toolchain.ComponentReport("tt-metal", ok, "0.77.0" if ok else None, "0.72.0", ok,
                                  "ok" if ok else "not found"),
    ])


class TestValidate:
    def _env(self, *, ok=True, port_free=True):
        return start.Environment(report=_report(ok), arch="blackhole", device_count=4,
                                 device_source="tt-smi", port=8000, port_free=port_free,
                                 conflicts=[])

    def test_a_busy_port_is_a_blocker(self):
        assert "port 8000 is already in use" in self._env(port_free=False).blockers

    def test_an_inadequate_component_is_a_blocker(self):
        assert any("tt-metal" in b for b in self._env(ok=False).blockers)

    def test_a_healthy_environment_has_no_blockers(self):
        assert self._env().blockers == []

    def test_conflicts_are_reported_but_do_not_block(self):
        """An environment conflict may involve a package the TT path never imports, so it
        is surfaced and not enforced — the same call the doctor change makes."""
        env = self._env()
        env.conflicts = [toolchain.EnvConflict("opencv", "numpy>=2", "numpy 1.26.4")]
        assert env.blockers == []

    def test_blocked_validation_pulls_nothing(self, monkeypatch):
        """The gate exists so a doomed run stops before touching the Hub."""
        called = []
        monkeypatch.setattr(cli, "_ensure_vllm_pulled",
                            lambda *a, **k: called.append(a) or {})
        monkeypatch.setattr(start, "validate", lambda *a, **k: self._env(port_free=False))
        res = runner.invoke(cli.app, ["start", "org/m", "--print", "--yes"])
        assert res.exit_code == 1
        assert called == [], "pulled despite a failed validation"
        assert "Nothing was pulled or started" in res.output


# ── roadmap integrity ────────────────────────────────────────────────────────
def test_every_phase_has_a_description():
    """The upfront panel and the stepper read from the same list; a phase with no detail
    would render a blank row."""
    assert set(start.PHASES) == set(start.PHASE_DETAIL)
    assert all(start.PHASE_DETAIL[p] for p in start.PHASES)


def test_phase_count_is_fixed_and_not_flag_dependent():
    """k/N is only trustworthy if N cannot drift with flags."""
    assert len(start.PHASES) == 5


# ── did-you-mean for a slipped word (F5) ─────────────────────────────────────
class TestDidYouMean:
    def test_two_command_names_in_a_row_suggests_the_later_one(self):
        """`pull serve <id>` reads as "serve <id>" with a stray word; the trailing tokens
        follow the intent, not the typo."""
        assert cli._did_you_mean(["pull", "serve", "x/y"]) == "tt-model serve x/y"

    def test_the_direction_is_not_hardcoded(self):
        assert cli._did_you_mean(["serve", "pull", "x/y"]) == "tt-model pull x/y"

    @pytest.mark.parametrize("argv", [
        ["pull", "x/y"],                 # correct usage
        ["pull", "x/y", "z/w"],          # two repo ids: not a slipped command name
        ["install"],                     # too short
        ["pull", "pull", "x/y"],         # same word twice is not a slip we can resolve
        ["instances", "list"],           # a real two-word command
    ])
    def test_no_suggestion_when_there_is_nothing_to_infer(self, argv):
        """A wrong hint is worse than none: it sends the user somewhere they did not ask
        to go, on a command that may have failed for an unrelated reason."""
        assert cli._did_you_mean(argv) is None

    def test_end_to_end_hint_appears_on_the_usage_error(self):
        res = runner.invoke(cli.app, ["pull", "serve", "x/y"])
        assert res.exit_code != 0
        assert "Did you mean" in res.output
        assert "tt-model serve x/y" in res.output

    def test_valid_commands_never_get_a_hint(self):
        res = runner.invoke(cli.app, ["--help"])
        assert "Did you mean" not in res.output

    def test_usage_error_classes_cover_typers_vendored_fork(self):
        """Typer vendors its own click, so a subcommand parse failure raises
        typer._click.exceptions.UsageError — NOT a subclass of click.UsageError. Catching
        only the latter silently matched nothing."""
        names = {c.__module__ for c in cli._USAGE_ERRORS}
        assert any("typer" in n for n in names), names
        assert any(n.startswith("click") for n in names), names


# ── `tt-model start` with no model named ─────────────────────────────────────
class TestMenu:
    @pytest.mark.parametrize("keys,expected", [
        (["2"], 1),
        (["1"], 0),
        ([""], 0),                      # bare Enter takes the default
        (["q"], None),                  # declining is not an error
        (["9", "abc", "2"], 1),         # re-prompts, does not crash or take a default
    ])
    def test_selection(self, monkeypatch, keys, expected):
        it = iter(keys)
        monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
        assert console.choose("Serve", ["a", "b"]) == expected

    def test_eof_declines_rather_than_looping(self, monkeypatch):
        def boom(prompt=""):
            raise EOFError
        monkeypatch.setattr("builtins.input", boom)
        assert console.choose("Serve", ["a", "b"]) is None


class TestPickModel:
    def _entries(self, monkeypatch, entries):
        monkeypatch.setattr(start.localdb, "all_entries", lambda: entries)

    def test_no_model_and_nothing_installed_explains_instead_of_erroring(self, monkeypatch):
        """`tt-model start` used to answer "Missing argument 'model'." — the one response a
        guided command must not give."""
        self._entries(monkeypatch, [])
        res = runner.invoke(cli.app, ["start"])
        assert res.exit_code == 2
        assert "Nothing is installed yet" in res.output
        assert "tt-model search --catalog" in res.output
        assert "Missing argument" not in res.output

    def test_a_single_bundle_is_still_offered_as_a_list_interactively(self, monkeypatch):
        """Silently taking "the only installed bundle" saves one keystroke and costs the
        user their sense of what is about to run — it reads as the CLI having a favourite
        model. With a TTY, always show the list."""
        self._entries(monkeypatch, [
            {"repo_id": "org/only", "bundle_path": "/tmp/x", "backend": "vllm"}])
        monkeypatch.setattr(cli.console, "choose_rows", lambda *a, **k: 0)
        repo, note = cli._pick_model(interactive=True)
        assert repo == "org/only"

    def test_a_single_bundle_is_taken_without_asking_when_we_cannot_ask(self, monkeypatch):
        """--yes or a piped stdin cannot answer a menu, and one runnable candidate is
        unambiguous rather than a guess."""
        self._entries(monkeypatch, [
            {"repo_id": "org/only", "bundle_path": "/tmp/x", "backend": "vllm"}])
        monkeypatch.setattr(cli.console, "choose_rows", _must_not_be_called)
        repo, note = cli._pick_model(interactive=False)
        assert repo == "org/only"
        assert "only installed" in note

    def test_several_installed_non_interactive_lists_them(self, monkeypatch):
        """No prompt is possible, so name the exact commands rather than failing vaguely."""
        self._entries(monkeypatch, [
            {"repo_id": "org/a", "bundle_path": "/tmp/x"},
            {"repo_id": "org/b", "bundle_path": "/tmp/x"}])
        res = runner.invoke(cli.app, ["start", "--yes"])
        assert res.exit_code == 2
        assert "tt-model start org/a" in res.output
        assert "tt-model start org/b" in res.output

    def test_several_installed_interactive_prompts(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/a", "bundle_path": "/tmp/x"},
            {"repo_id": "org/b", "bundle_path": "/tmp/x"}])
        monkeypatch.setattr(cli.console, "choose_rows", lambda *a, **k: 1)
        repo, note = cli._pick_model(interactive=True)
        assert repo == "org/b"

    def test_declining_the_menu_exits_without_starting_anything(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/a", "bundle_path": "/tmp/x"},
            {"repo_id": "org/b", "bundle_path": "/tmp/x"}])
        monkeypatch.setattr(cli.console, "choose_rows", lambda *a, **k: None)
        with pytest.raises(typer.Exit):
            cli._pick_model(interactive=True)

    def test_entries_without_a_bundle_path_are_not_offered(self, monkeypatch):
        """A recorded-but-not-materialised entry cannot be served, so offering it would
        send the user into a failure."""
        self._entries(monkeypatch, [
            {"repo_id": "org/ghost"},                          # no bundle_path
            {"repo_id": "org/real", "bundle_path": "/tmp/x"}])
        ids = [c.repo_id for c in start.installed_choices()]
        assert ids == ["org/real"]

    def test_labels_carry_enough_to_choose_between(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/a", "bundle_path": "/x", "backend": "vllm", "arch": "blackhole"}])
        assert start.installed_choices()[0].label == "org/a  (vllm · blackhole)"


def test_model_argument_is_optional():
    """The signature is the contract: a required argument makes the guided path impossible."""
    res = runner.invoke(cli.app, ["start", "--help"])
    assert "[{model}]" in res.output or "Omit to pick" in res.output


# ── never auto-pick a bundle that cannot serve ───────────────────────────────
class TestServabilityGate:
    def _entries(self, monkeypatch, entries):
        monkeypatch.setattr(start.localdb, "all_entries", lambda: entries)

    def _servability(self, monkeypatch, mapping):
        monkeypatch.setattr(start, "_servability",
                            lambda path: mapping.get(path, (True, None)))

    def test_the_only_bundle_being_unservable_is_not_auto_picked(self, monkeypatch):
        """Auto-selecting a bundle already known to be unrunnable walked the user through
        three phases to fail at the fourth on something knowable before the first. Only a
        caller that cannot be asked (--yes, or a piped stdin) gets the refusal."""
        self._entries(monkeypatch, [
            {"repo_id": "org/broken", "bundle_path": "/b"}])
        self._servability(monkeypatch, {"/b": (False, "models is not importable")})
        res = runner.invoke(cli.app, ["start", "--yes"])
        assert res.exit_code == 2
        assert "Nothing installed here can serve" in res.output
        assert "models is not importable" in res.output

    def test_the_reason_is_named_per_bundle(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/a", "bundle_path": "/a"},
            {"repo_id": "org/b", "bundle_path": "/b"}])
        self._servability(monkeypatch, {"/a": (False, "models is not importable"),
                                       "/b": (False, "models is not importable")})
        res = runner.invoke(cli.app, ["start", "--yes"])
        assert res.output.count("models is not importable") >= 2

    def test_unservable_bundles_are_still_offered_interactively(self, monkeypatch):
        """Declining to CHOOSE for the user is not the same as declining to LET them
        choose. They may be about to fix PYTHONPATH, or may just want to see the failure —
        so the menu still offers it, marked, rather than refusing outright."""
        self._entries(monkeypatch, [{"repo_id": "org/broken", "bundle_path": "/b"}])
        self._servability(monkeypatch, {"/b": (False, "models is not importable")})
        seen = {}
        monkeypatch.setattr(cli.console, "choose_rows",
                            lambda prompt, rows, **k: (seen.update(rows=rows), 0)[1])
        repo, note = cli._pick_model(interactive=True)
        assert repo == "org/broken"
        assert "despite" in note, note
        # The ✗ is the renderer's job now; the row carries the flag and the reason, each in
        # its own field, so this asserts the meaning rather than the glyph.
        marked, _name, _meta, reason = seen["rows"][0]
        assert marked, "the menu did not flag it unrunnable"
        assert reason == "models is not importable", reason

    def test_a_single_servable_bundle_is_taken_when_we_cannot_ask(self, monkeypatch):
        self._entries(monkeypatch, [{"repo_id": "org/ok", "bundle_path": "/ok"}])
        self._servability(monkeypatch, {})
        repo, note = cli._pick_model(interactive=False)
        assert repo == "org/ok"

    def test_the_one_servable_bundle_wins_over_broken_siblings(self, monkeypatch):
        """With exactly one runnable candidate there is nothing to choose between, even
        though other bundles are installed."""
        self._entries(monkeypatch, [
            {"repo_id": "org/broken", "bundle_path": "/b"},
            {"repo_id": "org/works", "bundle_path": "/w"}])
        self._servability(monkeypatch, {"/b": (False, "models is not importable")})
        repo, note = cli._pick_model(interactive=False)
        assert repo == "org/works"
        assert "can serve here" in note

    def test_servable_bundles_sort_first_so_the_default_is_never_broken(self, monkeypatch):
        self._entries(monkeypatch, [
            {"repo_id": "org/aaa-broken", "bundle_path": "/b"},
            {"repo_id": "org/zzz-works", "bundle_path": "/w"}])
        self._servability(monkeypatch, {"/b": (False, "models is not importable")})
        choices = start.installed_choices()
        assert choices[0].repo_id == "org/zzz-works", "a broken bundle sorted to the default"

    def test_an_explicit_id_is_still_honoured(self, monkeypatch):
        """The gate is about *picking for* the user. Naming a bundle explicitly must still
        work — it stops at the serve preflight, which says the same thing with more detail."""
        self._entries(monkeypatch, [{"repo_id": "org/broken", "bundle_path": "/b"}])
        self._servability(monkeypatch, {"/b": (False, "models is not importable")})
        monkeypatch.setattr(start, "validate", lambda *a, **k: start.Environment(
            report=_report(True), arch="blackhole", device_count=4, device_source="tt-smi",
            port=8000, port_free=True, conflicts=[]))
        res = runner.invoke(cli.app, ["start", "org/broken", "--yes", "--print"])
        assert "Nothing installed here can serve" not in res.output

    def test_unreadable_metadata_does_not_mark_a_bundle_unservable(self, tmp_path):
        """Fail open: a metadata problem is a different failure, and guessing "unservable"
        would hide a bundle that works."""
        servable, reason = start._servability(str(tmp_path / "missing"))
        assert servable is True and reason is None

    def test_servability_check_can_be_skipped(self, monkeypatch):
        """It spawns a subprocess per bundle; callers that only need labels can opt out."""
        self._entries(monkeypatch, [{"repo_id": "org/a", "bundle_path": "/a"}])
        monkeypatch.setattr(start, "_servability", _must_not_be_called)
        assert start.installed_choices(check_servable=False)[0].servable is True


# ── bundles on disk but not indexed ──────────────────────────────────────────
class TestUnregisteredBundles:
    def _bundle(self, root, name, main_class, with_models=False):
        d = root / name
        d.mkdir(parents=True)
        (d / "vllm_metadata.json").write_text(
            '{"arch": "X", "main_class": "%s"}' % main_class)
        if with_models:
            pkg = d / "models" / "mymodel"
            pkg.mkdir(parents=True)
            (d / "models" / "__init__.py").write_text("")
            (pkg / "__init__.py").write_text("")
            (pkg / "generator_vllm.py").write_text("class Cls: pass\n")
        return d

    def test_a_folder_with_no_index_entry_is_still_offered(self, monkeypatch, tmp_path):
        """A bundle can be materialised without an index entry — pulled under a different
        XDG_CACHE_HOME, restored, or copied in. It is servable in every practical sense, so
        leaving it out of the menu hides a working model from its owner."""
        self._bundle(tmp_path, "org__ghost", "models.mymodel.generator_vllm:Cls")
        monkeypatch.setattr(start.localdb, "all_entries", lambda: [])
        import tt_kernel.bundles as b
        monkeypatch.setattr(b, "resolve_bundles_dir", lambda *_a, **_k: tmp_path)
        found = start.unregistered_bundles()
        assert found and found[0][0] == "org/ghost"

    def test_indexed_bundles_are_not_duplicated(self, monkeypatch, tmp_path):
        self._bundle(tmp_path, "org__known", "models.x:Cls")
        monkeypatch.setattr(start.localdb, "all_entries",
                            lambda: [{"repo_id": "org/known", "bundle_path": "/x"}])
        import tt_kernel.bundles as b
        monkeypatch.setattr(b, "resolve_bundles_dir", lambda *_a, **_k: tmp_path)
        assert start.unregistered_bundles() == []

    def test_a_directory_without_metadata_is_not_a_bundle(self, monkeypatch, tmp_path):
        (tmp_path / "org__junk").mkdir(parents=True)
        monkeypatch.setattr(start.localdb, "all_entries", lambda: [])
        import tt_kernel.bundles as b
        monkeypatch.setattr(b, "resolve_bundles_dir", lambda *_a, **_k: tmp_path)
        assert start.unregistered_bundles() == []

    def test_a_bundle_shipping_its_own_adapter_counts_as_servable(self, tmp_path):
        """The TT plugin resolves adapters relative to each EXTRA_MODELS_DIR entry, so a
        bundle carrying its own models/ subtree is servable — checking the bare interpreter
        would have called it broken."""
        d = self._bundle(tmp_path, "org__selfsufficient",
                         "models.mymodel.generator_vllm:Cls", with_models=True)
        servable, reason = start._servability(str(d))
        assert servable is True, reason

    def test_a_bundle_shipping_nothing_is_not_servable(self, tmp_path):
        d = self._bundle(tmp_path, "org__empty", "models.nope.generator_vllm:Cls")
        servable, reason = start._servability(str(d))
        assert servable is False
        assert "models" in reason


# ── skipped phases stay visible ──────────────────────────────────────────────
class TestSkippedPhases:
    """A skipped phase must render as skipped. Silently advancing past it is
    indistinguishable from a quiet failure, and marking it done claims work that never
    happened — both leave k/N describing a run that did not occur."""

    def test_skipped_phase_is_not_pending_in_the_stepper(self):
        console.register_phases(["A", "B", "C"])
        console.skip_phase("B", "nothing to do")
        line = console.stepper_line().plain
        assert "⊘ B" in line, line
        assert "○ B" not in line, "skipped rendered as pending — reads as a stalled run"

    def test_a_body_can_mark_its_own_phase_skipped(self):
        console.register_phases(["A"])
        with console.phase("A") as ph:
            console.mark_skipped(ph, "no token")
        assert "⊘ A" in console.stepper_line().plain

    def test_marking_skipped_survives_the_phase_exit(self):
        """`phase()` used to stamp "done" unconditionally on clean exit."""
        console.register_phases(["A"])
        with console.phase("A") as ph:
            console.mark_skipped(ph)
        assert console._phases[0]["status"] == "skipped"

    def test_a_failure_still_wins_over_a_skip(self):
        console.register_phases(["A"])
        with pytest.raises(RuntimeError):
            with console.phase("A") as ph:
                console.mark_skipped(ph, "decided against")
                raise RuntimeError("boom")
        assert console._phases[0]["status"] == "failed"


def test_the_pin_is_claimed_before_anything_is_printed():
    """Regression: pinning after the panel printed the panel INTO the region, then homed the
    cursor to the region's top row and overprinted it — the panel and the picker interleaved
    on screen. DECSTBM homes the cursor, so the region must be claimed before any output."""
    src = inspect.getsource(cli.start)
    assert src.index("pin_stepper") < src.index("steps_panel_lines"), \
        "the screen is claimed after output was already printed to it"


def test_roadmap_precedes_the_model_prompt():
    """The panel is the frame the picker sits in; printed after it, the menu appears
    against a blank screen and the user commits before seeing what the run will do."""
    src = inspect.getsource(cli.start)
    assert src.index("steps_panel_lines") < src.index("_pick_model"), \
        "roadmap panel is printed after the picker"


# ── option-shaped tokens are flags, not model ids ────────────────────────────
class TestFlagsAreNotModelIds:
    """`start` forwards unknown flags to vLLM, so click is configured with
    ignore_unknown_options — and click then fills the `model` ARGUMENT with the first
    option-shaped token instead of rejecting it. `tt-model start --force` tried to pull a
    Hub repo named "--force", and the advice to re-run with --force came from start itself.
    """

    def test_force_is_a_real_option(self):
        res = runner.invoke(cli.app, ["start", "--help"])
        assert "--force" in res.output, "start advertises --force in failures but rejects it"

    def test_a_leading_flag_does_not_become_the_model(self, monkeypatch):
        picked = {}
        monkeypatch.setattr(cli, "_pick_model",
                            lambda **k: (picked.setdefault("asked", True), ("org/m", "picked"))[1])
        monkeypatch.setattr(cli.start_mod, "validate", lambda *a, **k: None)
        monkeypatch.setattr(cli.start_mod, "resolve_account",
                            lambda *a, **k: start.Account(logged_in=False, name=None,
                                                          source=None))
        runner.invoke(cli.app, ["start", "--force", "--yes"])
        assert picked.get("asked"), "a flag was taken as the model id instead of prompting"

    def test_force_reaches_the_install(self, monkeypatch):
        """Without this the flag parses and is silently dropped, which is worse than the
        crash it replaced: the user is told the override worked when nothing overrode."""
        seen = {}
        monkeypatch.setattr(cli, "_ensure_vllm_pulled", lambda *a, **k: seen.update(k) or {})
        monkeypatch.setattr(cli.start_mod, "is_installed", lambda r: False)
        monkeypatch.setattr(cli.start_mod, "validate", lambda *a, **k: start.Environment(
            report=_report(True), arch="blackhole", device_count=4, device_source="tt-smi",
            port=8000, port_free=True, conflicts=[]))
        monkeypatch.setattr(cli, "_serve_vllm", lambda *a, **k: None)
        res = runner.invoke(cli.app, ["start", "org/m", "--yes", "--force", "--print"])
        assert seen.get("force") is True, (seen, res.output)


def test_is_installed_requires_the_folder_not_just_the_index(monkeypatch, tmp_path):
    """A stale index entry made start skip the pull and fail at serve three steps later,
    where _ensure_vllm_pulled would have re-pulled it."""
    gone = tmp_path / "not-there"
    monkeypatch.setattr(start.localdb, "get", lambda r: {"bundle_path": str(gone)})
    assert start.is_installed("org/m") is False
    gone.mkdir()
    assert start.is_installed("org/m") is True
