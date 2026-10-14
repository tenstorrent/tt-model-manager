"""This repository is public. AGENTS.md ("Public GitHub is public") bars two things from
everything committed here: attribution to an AI assistant, and pointers into internal
communication (Jira keys, Atlassian / Slack / Google Docs links). A rule in a doc erodes
one docstring at a time; this scan is what makes it stick."""
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve()
SKIP_DIRS = {".git", ".venv", "build", "dist", ".pytest_cache", "__pycache__", "node_modules"}
SUFFIXES = {".md", ".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".sh", ".cfg", ".ini"}

INTERNAL = re.compile(r"\bDEVSTACK-\d+\b|atlassian\.net|slack\.com|docs\.google\.com", re.I)
ATTRIBUTION = re.compile(
    r"Co-Authored-By:\s*(Claude|Cursor|Copilot|Codex|ChatGPT|Gemini)"
    r"|Generated with \[?(Claude|Cursor|Copilot|Codex)",
    re.I,
)


def _text_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            p = Path(root) / name
            if p.suffix in SUFFIXES and p != THIS:
                yield p


def _hits(pattern):
    out = []
    for p in _text_files():
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if pattern.search(line):
                out.append(f"{p.relative_to(REPO)}:{n}: {line.strip()[:120]}")
    return out


def test_no_internal_ticket_keys_or_links_in_the_tree():
    assert _hits(INTERNAL) == []


def test_no_ai_attribution_in_the_tree():
    assert _hits(ATTRIBUTION) == []


def test_claude_code_attribution_is_switched_off():
    """Left at its default the tool appends a trailer to every commit and PR it makes."""
    cfg = json.loads((REPO / ".claude" / "settings.json").read_text())
    assert cfg["includeCoAuthoredBy"] is False
    assert cfg["attribution"] == {"commit": "", "pr": ""}
