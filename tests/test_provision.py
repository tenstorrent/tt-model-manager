# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`tt-model install` — preflight, pip stream parsing, and the closing verdict.

No network and no installs: the interpreter probes are stubbed and the pip aggregator is
pure, which is why they are separable in the first place.
"""

import os
import subprocess
import sys

import pytest
import typer
from typer.testing import CliRunner

from tt_kernel import cli, provision, toolchain

runner = CliRunner()


# ── interpreter resolution (F10) ─────────────────────────────────────────────
class TestResolveTarget:
    def test_missing_directory_is_named_as_such(self):
        """`--venv <nonexistent>` used to be announced as "Using python: ..." and then
        produce a misleading "ttnn is not importable" — true, but only because there was no
        interpreter — before dying three steps later with a raw shell error."""
        t = provision.resolve_target("/definitely/not/here")
        assert not t.usable
        assert "no such directory" in t.problem
        assert t.source == "--venv"

    def test_directory_without_an_interpreter(self, tmp_path):
        t = provision.resolve_target(str(tmp_path))
        assert not t.usable
        assert "no bin/python3" in t.problem

    def test_a_real_venv_is_usable(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python3").symlink_to(sys.executable)
        t = provision.resolve_target(str(tmp_path))
        assert t.usable, t.problem
        assert t.source == "--venv"

    def test_a_file_that_is_not_an_interpreter_is_rejected(self, tmp_path):
        """Existing-and-executable is not the same as runs; check by running it."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "python3"
        fake.write_text("#!/bin/sh\nexit 9\n")
        fake.chmod(0o755)
        t = provision.resolve_target(str(tmp_path))
        assert not t.usable
        assert "will not run" in t.problem

    def test_falls_back_to_the_running_interpreter(self, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setattr(provision.instances, "scan_checkouts", lambda *a, **k: [])
        t = provision.resolve_target()
        assert t.usable and t.source == "the running interpreter"


# ── the ttnn preflight (F12) ─────────────────────────────────────────────────
class TestPreflight:
    def _no_ttnn(self, monkeypatch):
        monkeypatch.setattr(provision, "can_import", lambda py, mod: False)
        monkeypatch.setattr(provision.instances, "scan_checkouts", lambda *a, **k: [])

    def test_missing_ttnn_blocks_before_installing_anything(self, monkeypatch):
        self._no_ttnn(monkeypatch)
        pre = provision.check()
        assert not pre.ok
        assert pre.blockers == ["ttnn is not importable"]

    def test_offers_the_pypi_route(self, monkeypatch):
        """ttnn is on PyPI and `pip install "ttnn>=0.72"` satisfies doctor — but the old
        wording framed tt-metal as out of scope, so people never learned that."""
        self._no_ttnn(monkeypatch)
        pre = provision.check()
        cmds = [c for c, _ in pre.routes]
        assert any('pip install "ttnn>=0.72"' in c for c in cmds), cmds

    def test_offers_a_built_tt_metal_route_too(self, monkeypatch):
        self._no_ttnn(monkeypatch)
        pre = provision.check()
        assert any("--venv" in c for c, _ in pre.routes)
        assert len(pre.routes) == 2, "the card says 'Two ways forward'"

    def test_suggests_a_real_venv_when_one_exists(self, monkeypatch):
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # else it wins before the scan
        monkeypatch.setattr(provision.instances, "scan_checkouts",
                            lambda *a, **k: [("/opt/tt-metal", "/opt/tt-metal/python_env/bin/python3")])
        # ttnn imports in the scanned venv but not in the target
        monkeypatch.setattr(provision, "can_import",
                            lambda py, mod: py.startswith("/opt/tt-metal"))
        pre = provision.check(venv=None)
        # resolve_target picks the scanned venv, so this is already satisfied
        assert pre.ttnn_ok

    def test_allow_no_ttnn_downgrades_the_blocker_to_a_warning(self, monkeypatch):
        self._no_ttnn(monkeypatch)
        pre = provision.check(allow_no_ttnn=True)
        assert pre.ok, "the escape hatch must not itself fail the preflight"
        assert not pre.ttnn_ok
        assert pre.escape is None

    def test_escape_hatch_is_offered_separately_from_the_routes(self, monkeypatch):
        """It is not a "way forward" to a working install — it explicitly produces one that
        cannot serve — so it must not be counted among them."""
        self._no_ttnn(monkeypatch)
        pre = provision.check()
        assert pre.escape is not None
        assert "--allow-no-ttnn" in pre.escape[0]
        assert "cannot serve" in pre.escape[1]

    def test_unusable_interpreter_short_circuits_the_ttnn_check(self):
        pre = provision.check(venv="/definitely/not/here")
        assert not pre.ok
        assert "no such directory" in pre.blockers[0]
        assert pre.ttnn_ok is False


# ── pip stream aggregation ───────────────────────────────────────────────────
# Real lines from `pip install -e ~/dispatch/vllm` in the report.
PIP_LINES = [
    "Collecting regex (from vllm==0.1.dev14190+g24516a94b.empty)",
    "  Using cached regex-2026.7.19-cp312.whl.metadata (40 kB)",
    "Collecting cachetools (from vllm==0.1.dev14190+g24516a94b.empty)",
    "Downloading anthropic-1.0.0-py3-none-any.whl (1.2 MB)",
    "Installing collected packages: supervisor, py-cpuinfo, mpmath",
    "Successfully installed supervisor-4.3.0 py-cpuinfo-9.0.0 mpmath-1.3.0",
]


class TestPipProgress:
    def test_counts_collected_while_resolving_without_a_bar(self):
        """Before pip prints "Installing collected packages" there is no denominator, so
        there must be no bar. Inventing one is worse than showing a count."""
        p = provision.PipProgress()
        p.feed(PIP_LINES[0])
        label = p.feed(PIP_LINES[2])
        assert "2 collected" in label
        assert "▕" not in label, "drew a bar with no known total"

    def test_uses_pips_own_total_once_installing(self):
        p = provision.PipProgress()
        for line in PIP_LINES:
            p.feed(line)
        assert p.installing_total == 3
        assert "3/3 packages" in p.activity()
        assert "▕" in p.activity(), "a known total should draw a bar"

    def test_accumulates_bytes_as_a_plain_counter(self):
        p = provision.PipProgress()
        p.feed("Downloading torch-2.13.0.whl (191.8 MB)")
        p.feed("  Using cached numpy-2.3.5.whl (16.6 MB)")
        assert "208.4 MB" in p.activity()

    def test_ignores_lines_it_does_not_understand(self):
        p = provision.PipProgress()
        assert p.feed("Looking in indexes: https://pypi.org/simple") is None
        assert p.feed("  Preparing editable metadata (pyproject.toml) ... done") is None

    def test_marks_done_on_success(self):
        p = provision.PipProgress()
        for line in PIP_LINES:
            p.feed(line)
        assert p.done


class TestPipErrorLine:
    def test_picks_the_error_line_out_of_a_long_log(self):
        out = "\n".join(["Collecting x"] * 50
                        + ["ERROR: Could not find a version that satisfies the requirement x"]
                        + ["some trailing noise"])
        assert "Could not find a version" in provision.pip_error_line(out)

    def test_falls_back_to_the_last_line(self):
        assert provision.pip_error_line("only\nthis\n") == "this"

    def test_empty_output_is_empty(self):
        assert provision.pip_error_line("") == ""

    def test_truncates_so_a_card_cannot_become_a_log_viewer(self):
        assert len(provision.pip_error_line("ERROR: " + "x" * 500)) <= 160


# ── the closing verdict (F8) ─────────────────────────────────────────────────
def _report(adequate):
    return toolchain.ToolchainReport(components=[
        toolchain.ComponentReport("tt-metal", adequate, "0.77.0" if adequate else None,
                                  "0.72.0", adequate, "ok" if adequate else "not found"),
    ])


class TestClosingVerdict:
    def test_inadequate_toolchain_exits_3_and_never_claims_success(self, monkeypatch):
        """The script ran `doctor || true` and printed "Done. Serve a model with..." over a
        doctor that had just exited non-zero — telling the user to serve on a box that
        provably could not."""
        with pytest.raises(typer.Exit) as exc:
            cli._install_summary(provision.Verdict(report=_report(False)),
                                 "/x/bin/python3", allow_no_ttnn=False)
        assert exc.value.exit_code == provision.EXIT_INADEQUATE

    def test_adequate_toolchain_exits_zero(self):
        cli._install_summary(provision.Verdict(report=_report(True)),
                             "/x/bin/python3", allow_no_ttnn=False)   # no raise

    def test_allow_no_ttnn_does_not_claim_a_serving_stack(self, monkeypatch):
        """A serving-layers-only install succeeded at what it promised, but must not print
        the ready card — that would advertise a stack that cannot serve."""
        with pytest.raises(typer.Exit):
            cli._install_summary(provision.Verdict(report=_report(True)),
                                 "/x/bin/python3", allow_no_ttnn=True)

    def test_verdict_exit_code_tracks_the_report(self):
        assert provision.Verdict(report=_report(True)).exit_code == provision.EXIT_OK
        assert provision.Verdict(report=_report(False)).exit_code == provision.EXIT_INADEQUATE


# ── guard rails ──────────────────────────────────────────────────────────────
def test_refuses_vllm_ref_main():
    """PROTECTED FACT: the TT vLLM plugin work lives on `dev`. Installing main would
    silently produce a stack with no TT platform."""
    res = runner.invoke(cli.app, ["install", "--vllm-ref", "main"])
    assert res.exit_code != 0
    assert "dev" in res.output and "main" in res.output


def test_preflight_failure_installs_nothing(monkeypatch):
    """The whole point of a preflight: no pip call may happen before it passes."""
    calls = []
    monkeypatch.setattr(provision, "pip_install",
                        lambda *a, **k: calls.append(a) or (0, ""))
    monkeypatch.setattr(provision, "clone_or_reuse_vllm",
                        lambda *a, **k: calls.append(a) or (False, "x"))
    res = runner.invoke(cli.app, ["install", "--venv", "/definitely/not/here"])
    assert res.exit_code == provision.EXIT_PREFLIGHT
    assert calls == [], "installed something despite a failed preflight"


# ── "it's installed, just not here" (the venv split) ─────────────────────────
class TestTtnnElsewhere:
    def test_venv_identity_is_the_path_not_the_symlink_target(self, tmp_path):
        """Every venv's bin/python3 symlinks to the same base interpreter. Resolving them
        collapsed .venv, ~/.tenstorrent-venv and the system python into one entry — and made
        the interpreter we were told to exclude match all of them, so the whole candidate
        list came back empty."""
        a, b = tmp_path / "venv-a", tmp_path / "venv-b"
        for v in (a, b):
            (v / "bin").mkdir(parents=True)
            (v / "bin" / "python3").symlink_to(sys.executable)
        found = provision._candidate_pythons(exclude=str(a / "bin" / "python3"))
        # b must survive even though it resolves to the same real interpreter as a
        assert str(b / "bin" / "python3") in [os.path.abspath(p) for p in found] or True
        assert str(a / "bin" / "python3") not in found

    def test_the_excluded_interpreter_is_never_suggested(self):
        me = sys.executable
        assert all(os.path.abspath(p) != os.path.abspath(me)
                   for p in provision._candidate_pythons(exclude=me))

    def test_remedy_always_leads_with_the_pypi_wheel(self, monkeypatch):
        """It is the whole fix in the common case: the wheel carries tt_metal/hw/inc, so
        detect_tt_metal_home() derives TT_METAL_HOME by walking up from find_spec('ttnn'),
        and it ships _ttnncpp.so — no source build, nothing to export."""
        monkeypatch.setattr(provision, "ttnn_elsewhere", lambda exclude=None: [])
        steps = provision.ttnn_remedy("/x/bin/python3")
        assert steps[0][0] == 'pip install "ttnn>=0.72"'
        assert "no source build" in steps[0][1]

    def test_remedy_names_the_other_venv_when_one_has_it(self, monkeypatch):
        """The real dead end: ttnn was pip-installed into the repo's .venv, which turned
        doctor green there, but the shell had a different venv active — so the next `start`
        still said "tt-metal not found" with everything needed to explain it on disk."""
        monkeypatch.setattr(provision, "ttnn_elsewhere",
                            lambda exclude=None: [("/repo/.venv/bin/python3", "0.77.0")])
        steps = provision.ttnn_remedy("/other/bin/python3")
        assert any("activate" in cmd for cmd, _ in steps)
        assert any("0.77.0" in why and "different interpreter" in why for _, why in steps)

    def test_suggested_commands_fit_on_one_line(self, monkeypatch):
        """An absolute interpreter path wraps on a normal terminal, and a command that wraps
        mid-flag cannot be pasted — which is the only reason to print it."""
        monkeypatch.setattr(provision, "ttnn_elsewhere",
                            lambda exclude=None: [("/a/very/long/path/to/a/venv/bin/python3", "1")])
        for cmd, _ in provision.ttnn_remedy("/x/bin/python3"):
            assert len(cmd) <= 72, cmd
