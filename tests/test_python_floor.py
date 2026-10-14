# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The declared Python floor must match the one CI actually enforces (issue #120).

`pyproject.toml` said `>=3.9` while the real floor was 3.10: the CLI tests read
`result.stderr`, which needs click's separated-stderr CliRunner (click >= 8.2), and click
8.2 itself requires Python >= 3.10. On 3.9 `uv --locked` resolved click 8.1 and eight tests
failed with "stderr not separately captured". CI was already restricted to 3.10/3.11/3.12,
so the declaration and the enforcement disagreed and a 3.9 user hit failures CI never saw.

The floor is declared in three places that must agree, and nothing but this test makes them:
`pyproject.toml` (what pip/uv enforce on install), `uv.lock` (what `uv --locked` resolves
against -- a stale value here fails CI loudly, which is the point), and the CI matrix (what
is actually proven to pass).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _declared_floor(text: str) -> tuple[int, int]:
    """The (major, minor) in a ``requires-python = ">=X.Y"`` line."""
    m = re.search(r'requires-python\s*=\s*"\s*>=\s*(\d+)\.(\d+)', text)
    assert m, "no `requires-python = \">=X.Y\"` found"
    return int(m.group(1)), int(m.group(2))


def _ci_matrix_versions() -> list[tuple[int, int]]:
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())
    versions: list[tuple[int, int]] = []
    for job in wf["jobs"].values():
        for raw in ((job.get("strategy") or {}).get("matrix") or {}).get("python-version", []):
            major, _, minor = str(raw).partition(".")
            versions.append((int(major), int(minor)))
    assert versions, "the tests workflow declares no python-version matrix"
    return versions


def test_pyproject_floor_matches_the_lowest_version_ci_proves():
    """Claiming support for a version CI never runs is how #120 happened: the floor has to
    be the lowest version something actually passes on, not the lowest we hope works."""
    declared = _declared_floor((ROOT / "pyproject.toml").read_text())
    lowest_tested = min(_ci_matrix_versions())
    assert declared == lowest_tested, (
        f"pyproject declares >={declared[0]}.{declared[1]} but the lowest version CI runs is "
        f"{lowest_tested[0]}.{lowest_tested[1]}. Either test the version you claim, or claim "
        f"the version you test."
    )


def test_lockfile_floor_matches_pyproject():
    """`uv --locked` resolves against the lock's own `requires-python`. If it drifts below
    pyproject's, the lock can carry resolutions for a version we no longer support."""
    assert _declared_floor((ROOT / "uv.lock").read_text()) == \
        _declared_floor((ROOT / "pyproject.toml").read_text()), \
        "uv.lock's requires-python is stale -- re-run `uv lock` after changing pyproject"


def test_contributing_states_the_same_floor():
    """The number a contributor reads before installing anything."""
    text = (ROOT / "CONTRIBUTING.md").read_text()
    major, minor = _declared_floor((ROOT / "pyproject.toml").read_text())
    assert f"Python {major}.{minor} or newer is required" in text, \
        f"CONTRIBUTING.md does not state the declared floor ({major}.{minor})"


@pytest.mark.parametrize("version", ["3.9"])
def test_superseded_versions_are_not_claimed(version):
    """3.9 went end-of-life 2025-10-31. Kept as an explicit guard because the failure it
    caused was a dependency-resolution artifact, not a syntax error -- `src/` has no
    3.10-only syntax, so nothing else in the suite would notice 3.9 creeping back in."""
    major, minor = _declared_floor((ROOT / "pyproject.toml").read_text())
    dead = tuple(int(p) for p in version.split("."))
    assert (major, minor) > dead, f"pyproject claims support for end-of-life Python {version}"
