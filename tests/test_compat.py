# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Backward-compat: the tt-kernel -> tt-model rename must not break old installs
(legacy TT_KERNEL_* env vars, legacy ~/.cache|.config/tt-kernel dirs, legacy command name)."""

from pathlib import Path

from tt_kernel import build, compat, container, container_cli, localdb, runtime
from tt_kernel.manifest import Manifest


def _manifest(name: str = "my-model") -> Manifest:
    return Manifest(schema_version="5", name=name, tt_metal_version="v", arch="blackhole",
                    producer={"tt_kernel_version": "0.1.0", "created_at": "now"})


def test_env_prefers_new_then_falls_back_to_legacy(monkeypatch):
    monkeypatch.delenv("TT_MODEL_MODELS_DIR", raising=False)
    monkeypatch.delenv("TT_KERNEL_MODELS_DIR", raising=False)
    assert compat.env("TT_MODEL_MODELS_DIR") is None
    monkeypatch.setenv("TT_KERNEL_MODELS_DIR", "/legacy")      # only old var set
    assert compat.env("TT_MODEL_MODELS_DIR") == "/legacy"       # honored
    monkeypatch.setenv("TT_MODEL_MODELS_DIR", "/new")          # new var wins
    assert compat.env("TT_MODEL_MODELS_DIR") == "/new"


def test_data_dir_prefers_legacy_only_when_new_absent(tmp_path):
    base = tmp_path
    # neither exists -> new default
    assert compat.data_dir(base) == base / "tt-model"
    # only legacy exists -> legacy (a pre-rename install keeps its data)
    (base / "tt-kernel").mkdir()
    assert compat.data_dir(base) == base / "tt-kernel"
    # new exists too -> new wins
    (base / "tt-model").mkdir()
    assert compat.data_dir(base) == base / "tt-model"


def test_resolve_models_dir_honors_legacy_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TT_MODEL_MODELS_DIR", raising=False)
    monkeypatch.setenv("TT_KERNEL_MODELS_DIR", str(tmp_path / "old"))
    got = runtime.resolve_models_dir(None, "org/name")
    assert got == tmp_path / "old" / "org" / "name"


def test_localdb_uses_legacy_cache_dir_if_present(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "tt-kernel").mkdir()          # a pre-rename install's index dir
    assert localdb._index_path() == tmp_path / "tt-kernel" / "installed.json"


def test_cache_dir_honors_the_rename_shim(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert compat.cache_dir() == tmp_path / ".cache" / "tt-model"        # neither -> new
    (tmp_path / ".cache" / "tt-kernel").mkdir(parents=True)              # pre-rename install
    assert compat.cache_dir() == tmp_path / ".cache" / "tt-kernel"       # -> legacy


#: Every helper that derives a path under the per-user cache root, with the segments it must
#: append to that root. Add new helpers HERE — and if you forget,
#: ``test_no_module_names_the_brand_cache_dir_itself`` below is what fails, not review.
def _cache_path_helpers(m):
    return {
        "container.model_cache_dir":
            (lambda: container.model_cache_dir(m), ("my-model", "cache")),
        "container.model_weight_cache_dir":
            (lambda: container.model_weight_cache_dir(m), ("my-model", "weights")),
        "container.model_tensor_cache_dir":
            (lambda: container.model_tensor_cache_dir(m), ("my-model", "tensors")),
        "container_cli.pull_dir":
            (lambda: container_cli.pull_dir("org/model"), ("pulled", "org__model")),
        "build.build_log_path":
            (lambda: build.build_log_path("m").parent, ("build",)),
    }


def test_container_paths_do_not_bypass_the_shim_and_orphan_a_legacy_index(monkeypatch, tmp_path):
    """#62: on a pre-rename box, every container path must resolve UNDER the legacy
    ~/.cache/tt-kernel — not hardcode ~/.cache/tt-model. Creating the new dir flips
    `data_dir` for localdb/runtime too, orphaning the legacy installed.json so
    already-installed bundles vanish from `list` and `rm`."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Unset so localdb lands in ~/.cache too: it honors $XDG_CACHE_HOME and compat.cache_dir
    # does not, so #62 is only reproducible on one shared root. See the KNOWN GAP note on
    # compat.cache_dir -- unifying the base relocates an existing index, so it is separate.
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    legacy = tmp_path / ".cache" / "tt-kernel"
    legacy.mkdir(parents=True)                                          # only the legacy dir
    m = _manifest()

    for label, (call, segments) in _cache_path_helpers(m).items():
        assert call() == legacy.joinpath(*segments), f"{label} bypassed the rename shim"

    # The load-bearing assertion: computing (and, for the build dir, creating) these never
    # brought ~/.cache/tt-model into existence, so the shim stays pointed at the legacy dir.
    assert not (tmp_path / ".cache" / "tt-model").exists()
    assert localdb._index_path() == legacy / "installed.json"


def test_the_per_model_caches_share_one_parent_so_rm_cannot_orphan_one(monkeypatch, tmp_path):
    """`remove_container` rmtrees `model_cache_dir(m).parent` to drop a model's caches in one
    go, and the weight/tensor docstrings both promise "same parent" to justify that. A helper
    that resolves its root differently silently breaks the promise: `rm` deletes one tree and
    leaves the others on disk, unmentioned — the 105 GB-orphan shape those docstrings warn
    about, one directory over."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".cache" / "tt-kernel").mkdir(parents=True)             # pre-rename box
    m = _manifest()
    parents = {
        container.model_cache_dir(m).parent,
        container.model_weight_cache_dir(m).parent,
        container.model_tensor_cache_dir(m).parent,
    }
    assert len(parents) == 1, f"per-model caches split across parents: {sorted(map(str, parents))}"


def test_no_module_names_the_brand_cache_dir_itself():
    """The enumeration above can only check helpers this test knows how to call. This catches
    the rest, including ones that do not exist yet: outside `compat`, no module may build a
    path segment named for either brand dir. Route it through `compat.cache_dir()` instead.

    Matches a `/` path join against the literal, so the Typer app name and the
    `tt-model/<name>:<tag>` image tags — neither of which is a filesystem path — do not trip it.
    """
    import ast

    import tt_kernel

    offenders = []
    for f in sorted(Path(tt_kernel.__file__).parent.glob("*.py")):
        if f.name == "compat.py":       # the shim is where these names are allowed to live
            continue
        for node in ast.walk(ast.parse(f.read_text(), filename=str(f))):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
                continue
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant) and side.value in ("tt-model", "tt-kernel"):
                    offenders.append(f"{f.name}:{node.lineno} joins a path segment {side.value!r}")
    assert not offenders, (
        "these build a brand cache path directly instead of via compat.cache_dir():\n  "
        + "\n  ".join(offenders))


def test_invoked_as_legacy(monkeypatch):
    monkeypatch.setattr("sys.argv", ["/usr/bin/tt-kernel", "serve", "x"])
    assert compat.invoked_as_legacy() is True
    monkeypatch.setattr("sys.argv", ["/usr/bin/tt-model", "serve", "x"])
    assert compat.invoked_as_legacy() is False
