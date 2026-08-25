# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`scripts/install.sh` is a bootstrap shim — verify it stays one.

Its whole job is to put tt-model on PATH and hand off to `tt-model install`. The risk is
drift: logic creeping back into bash where it cannot reuse tt-model's own detection and
cannot render progress. These tests pin the contract rather than the implementation.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "install.sh"


def _source() -> str:
    return SCRIPT.read_text()


def _code() -> str:
    """The script minus comments — the header legitimately *describes* what moved to the
    CLI, so a naive substring search over the whole file matches its own documentation."""
    out = []
    for line in _source().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        out.append(line.split(" #", 1)[0])
    return "\n".join(out)


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & 0o111, "not executable"


def test_it_is_a_shim_not_an_installer():
    """The 104-line version resolved venvs, cloned the fork, ran three pip installs and
    called doctor. All of that moved into `tt-model install`; if it comes back here it will
    drift from the CLI's behaviour silently."""
    body = _code()
    assert len(_source().splitlines()) < 60, "install.sh is growing logic again"
    assert "exec" in body and "tt_kernel.cli install" in body, "no handoff to the CLI"


def test_it_does_not_reimplement_what_the_cli_owns():
    body = _code()
    for banned, why in [
        ("git clone", "cloning the vLLM fork belongs to `tt-model install`"),
        ("VLLM_TARGET_DEVICE", "the empty-device build flag belongs to the CLI"),
        ("import ttnn", "the ttnn preflight belongs to the CLI"),
        # Verification belongs to the CLI. This also covers the original sin here: the old
        # script ran `doctor || true`, discarding the exit code, and then printed "Done.
        # Serve a model with..." over a doctor that had just failed. A blanket ban on
        # `|| true` would be the tighter rule but it false-positives on the legitimate
        # `command -v python3 || true` that `set -e` requires.
        ("doctor", "verification belongs to the CLI"),
    ]:
        assert banned not in body, f"{banned!r} is back in install.sh: {why}"


def test_it_forwards_arguments_verbatim():
    """A shim that eats flags is worse than no shim: `--venv` and `--allow-no-ttnn` have to
    reach the CLI unchanged."""
    body = _code()
    assert '"$@"' in body, "arguments are not forwarded"


def test_it_documents_the_exit_codes_it_propagates():
    body = _source()
    for code in ("0", "1", "2", "3"):
        assert re.search(rf"^#\s+{code}\s+\S", body, re.M), f"exit code {code} undocumented"


def test_help_reaches_the_cli_and_lists_the_real_flags():
    """End-to-end: the shim must produce `tt-model install`'s help, not its own."""
    res = subprocess.run([str(SCRIPT), "--help"], capture_output=True, text=True,
                         cwd=str(REPO), timeout=180,
                         env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable,
                              "HOME": str(Path.home())})
    assert res.returncode == 0, res.stderr[-2000:]
    for flag in ("--venv", "--vllm-dir", "--vllm-ref", "--allow-no-ttnn"):
        assert flag in res.stdout, f"{flag} missing from forwarded --help"


def test_missing_python_is_reported_not_crashed(tmp_path):
    # Keep a PATH that still resolves bash (the shebang needs it) but holds no python3.
    bare = tmp_path / "bin"
    bare.mkdir()
    for tool in ("bash", "env", "dirname", "pwd", "cd", "command"):
        src = Path("/usr/bin") / tool
        if src.exists() and not (bare / tool).exists():
            (bare / tool).symlink_to(src)
    res = subprocess.run([str(SCRIPT)], capture_output=True, text=True, cwd=str(REPO),
                         timeout=60, env={"PATH": str(bare), "PYTHON": "",
                                          "HOME": str(Path.home())})
    assert res.returncode == 1
    assert "no python3" in res.stderr.lower()
