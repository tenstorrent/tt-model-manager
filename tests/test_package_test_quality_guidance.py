# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The package-test skill and the container recipe say to score a batch-1 fast path at
concurrency 1.

A 32-way concurrent eval puts almost every decode step on the batched path, so a change
that only the single-request path runs can leave the scores untouched. The guidance has to
survive edits to either file, and the `.codex` copy of the skill must not drift from the
`.claude` one.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude" / "skills" / "tt-model-package-test" / "SKILL.md"
CODEX_SKILL = ROOT / ".codex" / "skills" / "tt-model-package-test" / "SKILL.md"
RECIPE = ROOT / "docs" / "E2E_RECIPE_V5.1.md"


@pytest.mark.parametrize("path", [SKILL, CODEX_SKILL, RECIPE], ids=lambda p: str(p.relative_to(ROOT)))
def test_batch1_fast_path_is_scored_at_concurrency_1(path):
    text = " ".join(path.read_text().split())
    assert "batch-1 fast path" in text
    assert "concurrency 1" in text
    assert "num_concurrent=1" in text


def test_the_codex_skill_copy_matches():
    assert CODEX_SKILL.read_text() == SKILL.read_text()
