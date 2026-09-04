# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``source.extra_code``: shipping model code that does not live in the tt-metal tree.

``source.code`` is relative to ``source.tt_metal``. That is right for a model whose code
IS a tt-metal file and wrong for one that is not -- and ``tt-dit-server`` made the second
case real, because a diffusion server's ASGI app need not be a tt-metal module at all. The
launcher already only requires that ``runtime.app``'s top-level package is shipped by
*some* allowlist entry; before this field there was no way to ship one from elsewhere.

The assertions below are the ones a wrong answer makes expensive: a package that silently
ships nothing, an allowlist check that consults one root and not the other, and a skip
report that crashes because it assumed a single tree.
"""

import json
from pathlib import Path

import pytest

from tt_kernel.build import BuildError, stage_code
from tt_kernel.container_manifest import ContainerManifest, ContainerManifestError

BASE = {
    "schema": "5.1",
    "repo": "you/my-diffusion-model",
    "name": "my-diffusion-model",
    "weights": "org/Weights",
    "kind": "tt-dit-server",
    "arch": "blackhole",
    "source": {
        "tt_metal": "/path/to/tt-metal",
        "code": ["models/common"],
        "ubuntu": "24.04",
        "python": "3.12",
    },
    "runtime": {"app": "models.tt_dit.server.flux2.app:app"},
    "serve": {"hardware": "p300x2", "mesh_device": "QB2", "port": 8000},
}


def _manifest(source_over=None, runtime_over=None, validate=True) -> ContainerManifest:
    raw = json.loads(json.dumps(BASE))
    if source_over:
        raw["source"].update(source_over)
    if runtime_over:
        raw["runtime"].update(runtime_over)
    m = ContainerManifest.model_validate(raw)
    if validate:
        m.validate_semantics()
    return m


# ---- the model ---------------------------------------------------------------------


def test_extra_code_defaults_empty_so_existing_manifests_are_unaffected():
    """Every manifest written before this field must behave exactly as it did."""
    m = _manifest()
    assert m.source.extra_code == []
    assert m.source.all_code_paths == ["models/common"]


def test_all_code_paths_covers_both_roots():
    """Downstream asks 'does the allowlist ship X?'. Both roots land in one image tree, so
    a check that consults `code` alone is asking about staging while claiming to ask about
    the image."""
    m = _manifest({"extra_code": [{"root": "/src/tt-animatediff",
                                   "paths": ["animatediff_ttnn"]}]})
    assert m.source.all_code_paths == ["models/common", "animatediff_ttnn"]


def test_extra_code_accepts_a_git_root_like_tt_metal_does():
    """The local form alone would not be enough: a published package whose code can only
    be staged from one person's working copy is not reproducible by the consumer reading
    it, which is why tt_metal accepts a GitSource in the first place."""
    m = _manifest({"extra_code": [{
        "root": {"repo": "https://github.com/tenstorrent/tt-animatediff", "ref": "v0.10.0"},
        "paths": ["animatediff_ttnn"]}]})
    assert m.source.extra_code[0].root.ref == "v0.10.0"


@pytest.mark.parametrize("bad", ["/etc/passwd", "../outside", "a/../../b"])
def test_extra_code_paths_may_not_escape_their_root(bad):
    with pytest.raises(Exception, match="relative to their root"):
        _manifest({"extra_code": [{"root": "/src/x", "paths": [bad]}]}, validate=False)


def test_extra_code_requires_at_least_one_path():
    """An entry shipping nothing is a typo, not a configuration."""
    with pytest.raises(Exception):
        _manifest({"extra_code": [{"root": "/src/x", "paths": []}]}, validate=False)


# ---- the launcher gate this field exists to open -------------------------------------


def test_a_dit_app_shipped_only_by_extra_code_is_accepted():
    """THE ENABLING CASE. `animatediff_ttnn` is in no tt-metal tree, so before this field
    the launcher refused the app outright and the model could not be packaged at all."""
    m = _manifest(
        {"extra_code": [{"root": "/src/tt-animatediff", "paths": ["animatediff_ttnn"]}]},
        {"app": "animatediff_ttnn.server.app:app"},
    )
    assert m.runtime["app"] == "animatediff_ttnn.server.app:app"


def test_an_app_shipped_by_no_root_is_still_refused():
    """The allowlist still promises EXACTLY what ships. Widening where code may come from
    must not weaken the check that it comes from somewhere."""
    with pytest.raises(ContainerManifestError, match="no allowlist entry"):
        _manifest(
            {"extra_code": [{"root": "/src/x", "paths": ["something_else"]}]},
            {"app": "animatediff_ttnn.server.app:app"},
        )


# ---- existence, checked on the author's machine ---------------------------------------


def test_a_missing_local_extra_path_is_caught_before_the_build(tmp_path):
    """The alternative is an ImportError deep inside a multi-hour image build."""
    (tmp_path / "animatediff_ttnn").mkdir()
    m = _manifest({"tt_metal": str(tmp_path), "code": ["animatediff_ttnn"],
                   "extra_code": [{"root": str(tmp_path), "paths": ["not_here"]}]},
                  {"app": "animatediff_ttnn.server.app:app"})
    with pytest.raises(ContainerManifestError, match="source.extra_code lists 1 path"):
        m.validate_sources_exist()


def test_a_git_extra_root_is_not_checked_on_disk(tmp_path):
    """Skipped for the same reason tt_metal's GitSource is: it is not cloned yet. Silence
    here is correct; raising would make every CI-shaped manifest unloadable offline."""
    (tmp_path / "animatediff_ttnn").mkdir()
    m = _manifest({"tt_metal": str(tmp_path), "code": ["animatediff_ttnn"],
                   "extra_code": [{"root": {"repo": "https://x/y", "ref": "v1"},
                                   "paths": ["whatever"]}]},
                  {"app": "animatediff_ttnn.server.app:app"})
    m.validate_sources_exist()  # must not raise


# ---- staging, on a real filesystem ----------------------------------------------------


def _tree(root: Path, *files: str) -> Path:
    for f in files:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    return root


def test_stage_code_copies_from_both_roots_into_one_tree(tmp_path):
    """A file's origin stops being observable the moment it is copied -- the image has a
    single code overlay, and that is what makes runtime.app resolvable either way."""
    metal = _tree(tmp_path / "metal", "models/common/__init__.py")
    extra = _tree(tmp_path / "adiff", "animatediff_ttnn/__init__.py",
                  "animatediff_ttnn/server/app.py")
    dest = tmp_path / "code"
    m = _manifest({"tt_metal": str(metal),
                   "extra_code": [{"root": str(extra), "paths": ["animatediff_ttnn"]}]},
                  {"app": "animatediff_ttnn.server.app:app"})
    staged = stage_code(m, metal, dest, [extra])

    assert (dest / "models/common/__init__.py").is_file()
    assert (dest / "animatediff_ttnn/server/app.py").is_file()
    assert sorted(staged.tree) == ["animatediff_ttnn/", "models/common/"]


def test_stage_code_refuses_a_missing_extra_path(tmp_path):
    metal = _tree(tmp_path / "metal", "models/common/__init__.py")
    extra = (tmp_path / "adiff"); extra.mkdir()
    m = _manifest({"tt_metal": str(metal),
                   "extra_code": [{"root": str(extra), "paths": ["animatediff_ttnn"]}]},
                  {"app": "animatediff_ttnn.server.app:app"}, validate=False)
    with pytest.raises(BuildError, match="source.extra_code entry"):
        stage_code(m, metal, tmp_path / "code", [extra])


def test_stage_code_refuses_an_unresolved_extra_root(tmp_path):
    """THE SILENT FAILURE. A root the caller forgot to resolve would ship nothing at all,
    and the image would only complain on a consumer's first import."""
    metal = _tree(tmp_path / "metal", "models/common/__init__.py")
    m = _manifest({"tt_metal": str(metal),
                   "extra_code": [{"root": "/src/x", "paths": ["animatediff_ttnn"]}]},
                  {"app": "animatediff_ttnn.server.app:app"}, validate=False)
    with pytest.raises(BuildError, match="never resolved would silently ship nothing"):
        stage_code(m, metal, tmp_path / "code", [])


def test_a_dropped_file_is_reported_relative_to_its_own_root(tmp_path):
    """relative_to(metal) on a file from an extra root raises ValueError -- which would
    turn a cosmetic skip report into a crash mid-stage."""
    metal = _tree(tmp_path / "metal", "models/common/__init__.py")
    extra = _tree(tmp_path / "adiff", "animatediff_ttnn/__init__.py",
                  "animatediff_ttnn/__pycache__/x.pyc")
    m = _manifest({"tt_metal": str(metal),
                   "extra_code": [{"root": str(extra), "paths": ["animatediff_ttnn"]}]},
                  {"app": "animatediff_ttnn.server.app:app"})
    staged = stage_code(m, metal, tmp_path / "code", [extra])
    assert any("__pycache__" in s for s in staged.skipped)
    assert not (tmp_path / "code/animatediff_ttnn/__pycache__").exists()
