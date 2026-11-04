"""Repo skills: each Codex copy mirrors its Claude copy, names its directory, and links to files that exist."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLAUDE = ROOT / ".claude" / "skills"
CODEX = ROOT / ".codex" / "skills"


def _files(skill_dir: Path) -> dict:
    return {p.relative_to(skill_dir): p.read_bytes() for p in skill_dir.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


@pytest.mark.parametrize("name", sorted(p.name for p in CODEX.iterdir() if p.is_dir()))
def test_codex_skill_mirrors_claude_skill(name):
    assert _files(CODEX / name) == _files(CLAUDE / name)


@pytest.mark.parametrize("skill", sorted(CLAUDE.glob("*/SKILL.md")), ids=lambda p: p.parent.name)
def test_skill_names_its_directory_and_links_resolve(skill):
    text = skill.read_text(encoding="utf-8")
    assert yaml.safe_load(text.split("---")[1])["name"] == skill.parent.name
    for target in re.findall(r"\]\(([^)\s#]+)", text):
        if "://" not in target:
            assert (skill.parent / target).exists(), f"{skill}: broken link {target}"
