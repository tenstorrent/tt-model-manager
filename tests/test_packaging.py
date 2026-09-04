# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Offline tests for v5 self-contained bundle staging (packaging.stage_package). No hardware,
no network: fake wheel files + a fake metal tree are staged and the running-folder layout +
manifest are asserted.
"""

import json
import os

from typer.testing import CliRunner

from tt_kernel import cli, packaging
from tt_kernel.manifest import Capabilities, Manifest, Mesh, Producer, WeightsRef

_runner = CliRunner()


def test_parse_wheel_tags():
    t = packaging.parse_wheel_tags("ttnn-0.75.0-cp312-cp312-linux_x86_64.whl")
    assert t == {"python_tag": "cp312", "abi_tag": "cp312", "platform_tag": "linux_x86_64"}
    # build-tag variant + none-any plugin wheel
    assert packaging.parse_wheel_tags("vllm_tt_plugin-0.3.0-py3-none-any.whl")["platform_tag"] == "any"
    assert packaging.parse_wheel_tags("not-a-wheel.txt")["python_tag"] is None


def _fake_wheel(dirpath, filename, content=b"PK\x03\x04 fake wheel"):
    p = dirpath / filename
    p.write_bytes(content)
    return p


def _run_sh_manifest(**over):
    base = dict(
        schema_version="5",
        name="m",
        arch="blackhole",
        tt_metal_version="0.75.0",
        producer=Producer(tt_kernel_version="0.0.0", created_at="2026-08-20T00:00:00Z"),
        weights=WeightsRef(repo="org/model"),
        mesh=Mesh(devices=1, topology="P150"),
    )
    base.update(over)
    return Manifest(**base)


def test_render_run_sh_tool_parser_uses_vllm_flag_names():
    """The self-contained run.sh must emit the SAME vLLM flags the compose path does (PR #16).

    vLLM's FlexibleArgumentParser normalizes '_'->'-', so '--tool_parser' becomes the nonexistent
    '--tool-parser'; the real flag is '--tool-call-parser', and vLLM hard-errors on it without
    '--enable-auto-tool-choice'. This is the shipped launch path — before the fix it dropped the
    capability entirely, so a bundle declaring tool_parser silently got no tool calling.
    """
    run = packaging.render_run_sh(
        _run_sh_manifest(capabilities=Capabilities(tool_parser="qwen3_coder",
                                                   reasoning_parser="qwen3_coder"))
    )
    assert "--tool_parser" not in run and "--tool-parser" not in run
    assert "--enable-auto-tool-choice --tool-call-parser qwen3_coder" in run
    assert "--reasoning_parser qwen3_coder" in run


def test_render_run_sh_no_tool_flags_without_capability():
    """No tool_parser declared => neither flag appears (bare --enable-auto-tool-choice is an error)."""
    run = packaging.render_run_sh(_run_sh_manifest())
    assert "--enable-auto-tool-choice" not in run and "--tool-call-parser" not in run


def test_stage_package_layout(tmp_path):
    # fake author artifacts
    wheels = tmp_path / "in_wheels"
    wheels.mkdir()
    ttnn = _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn-bytes")
    plugin = _fake_wheel(wheels, "vllm_tt_plugin-0.3.0-py3-none-any.whl")
    metal = tmp_path / "metal_src"
    (metal / "models" / "tt_transformers" / "tt").mkdir(parents=True)
    (metal / "requirements.txt").write_text("torch==2.11.0\ntransformers==5.12.1\n")
    (metal / "__pycache__").mkdir()
    (metal / "__pycache__" / "junk.pyc").write_bytes(b"junk")  # must be ignored
    (metal / "models" / "tt_transformers" / "tt" / "attention.py").write_text("# blocks\n")

    vmeta = {"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"}
    staged = tmp_path / "staged"

    manifest = packaging.stage_package(
        staged,
        name="llama-3.2-3b-tt",
        arch="blackhole",
        ttnn_wheel=ttnn,
        plugin_wheel=plugin,
        metal_dir=metal,
        vllm_metadata=vmeta,
        tt_kernel_version="0.0.0",
        weights=WeightsRef(repo="unsloth/Llama-3.2-3B-Instruct"),
        mesh=Mesh(devices=1, topology="P150"),
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        tt_metal_version="0.75.0",
    )

    # layout
    assert (staged / "wheels" / "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl").read_bytes() == b"ttnn-bytes"
    assert (staged / "wheels" / "vllm_tt_plugin-0.3.0-py3-none-any.whl").is_file()
    assert (staged / "metal" / "models" / "tt_transformers" / "tt" / "attention.py").is_file()
    assert not (staged / "metal" / "__pycache__").exists()  # ignored
    assert (staged / "requirements.txt").read_text().startswith("torch==2.11.0")
    # vllm_metadata.json lives in a per-model SUBFOLDER under vllm_models/ (EXTRA_MODELS_DIR contract)
    meta = staged / "vllm_models" / "llama-3.2-3b-tt" / "vllm_metadata.json"
    assert meta.is_file()
    assert json.loads(meta.read_text())["arch"] == "LlamaForCausalLM"
    assert not (staged / "vllm_metadata.json").exists()  # NOT at the root (plugin would miss it)
    for s in ("install.sh", "run.sh"):
        # Owner read/write only, no execute bit — run via `bash <script>` (least privilege).
        assert (staged / s).stat().st_mode & 0o777 == 0o600
    # run.sh wired the non-obvious env + serving args the TT backend requires
    run = (staged / "run.sh").read_text()
    assert "_ttnncpp" in run and "TT_VLLM_BUILTIN_MODELS=0" in run
    assert 'EXTRA_MODELS_DIR="$HERE/vllm_models"' in run
    assert "find_spec" in run                      # ttnn located without importing it
    assert 'export HF_MODEL=' in run               # adapter reads HF_MODEL from env
    assert "--max_num_seqs 32" in run and "--block_size 64" in run  # TT backend defaults
    assert "unsloth/Llama-3.2-3B-Instruct" in run and "P150" in run
    # HERMETIC RUNTIME: every cache/home is redirected under the folder wall (overridable).
    assert 'HF_HOME="${HF_HOME:-$HERE/.hf}"' in run
    assert 'TT_CACHE_PATH="${TT_CACHE_PATH:-$HERE/.tt_cache}"' in run
    assert 'TT_CACHE_HOME="${TT_CACHE_HOME:-$HERE/.tt_cache}"' in run  # override upstream /mnt default
    assert 'XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HERE/.cache}"' in run

    # HERMETIC INSTALL: the interpreter lives inside the folder; venv is relocatable + copy-linked.
    inst = (staged / "install.sh").read_text()
    assert 'UV_PYTHON_INSTALL_DIR="$HERE/.python"' in inst
    assert "uv python install" in inst
    assert "uv venv --relocatable" in inst
    assert "--link-mode=copy" in inst

    # manifest
    m2 = Manifest.from_json((staged / "tt_kernel_manifest.json").read_text())
    assert m2.schema_version == "5"
    assert m2.is_self_contained is True
    assert m2.bundled.ttnn_wheel.python_tag == "cp312"
    assert m2.bundled.ttnn_wheel.sha256 == packaging.sha256_file(ttnn)
    assert [w.path for w in m2.bundled.wheels] == [
        "wheels/ttnn-0.75.0-cp312-cp312-linux_x86_64.whl",
        "wheels/vllm_tt_plugin-0.3.0-py3-none-any.whl",
    ]
    assert m2.entrypoint.arch_name == "LlamaForCausalLM"
    assert m2.weights.repo_id == "unsloth/Llama-3.2-3B-Instruct"


def test_stage_package_dangling_symlink_and_cache_excludes(tmp_path):
    """A built metal tree has dangling symlinks, host-absolute links, and multi-GB caches.

    copytree copies links as links (no crash on dangling ones), the excludes are ROOT-anchored
    (a nested dir of the same basename survives), and the staged tree is then normalized so it
    ships NO dangling links and NO links that leak the author's host path.
    """
    wheels = tmp_path / "in_wheels"
    wheels.mkdir()
    ttnn = _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn-bytes")

    # A real directory OUTSIDE the metal tree (the author's box); a link into it must materialize.
    outside = tmp_path / "outside_real"
    outside.mkdir()
    (outside / "data.txt").write_text("outside content\n")

    metal = tmp_path / "metal_src"
    metal.mkdir()
    (metal / "requirements.txt").write_text("torch==2.11.0\n")
    (metal / "real.py").write_text("# real file\n")
    # root-level dangling link (must NOT crash copytree, must be dropped by normalization)
    (metal / "foo").symlink_to("nonexistent")
    # a self-contained relative link INTO the tree — must be kept
    (metal / "alias.py").symlink_to("real.py")
    # `build -> build_Release`, exactly what build_metal.sh creates; target present but ROOT-excluded
    (metal / "build_Release").mkdir()
    (metal / "build_Release" / "libttnn.so").write_bytes(b"\x7fELF" + b"\x00" * 32)
    (metal / "build").symlink_to("build_Release")
    # a link to a real dir OUTSIDE the tree — must be replaced by a real recursive copy
    (metal / "linked_outside").symlink_to(outside)
    # depth-3 ABSOLUTE dangling link (issue #38's real shape) — must be dropped, not shipped broken
    umd = metal / "tt_metal" / "third_party" / "umd"
    umd.mkdir(parents=True)
    (umd / "keep.cpp").write_text("// real\n")
    (umd / "compile_commands.json").symlink_to("/nonexistent/build_Release/compile_commands.json")
    # NESTED python_env — root-anchoring must KEEP this while dropping the root one
    (metal / "tt_metal" / "python_env").mkdir()
    (metal / "tt_metal" / "python_env" / "keep.txt").write_text("tracked nested file\n")
    # root-level regenerable caches / build output — all excluded
    for cache in (".cpmcache", "python_env", "tt_cache", "built", "built_kernels"):
        (metal / cache).mkdir()
        (metal / cache / "blob").write_bytes(b"z" * 16)

    vmeta = {"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"}
    staged = tmp_path / "staged"

    # Must NOT raise on any dangling symlink.
    packaging.stage_package(
        staged,
        name="llama-3.2-3b-tt",
        arch="blackhole",
        ttnn_wheel=ttnn,
        metal_dir=metal,
        vllm_metadata=vmeta,
        tt_kernel_version="0.0.0",
    )

    dst_metal = staged / "metal"
    assert (dst_metal / "real.py").is_file()
    # a self-contained relative link is preserved as a link
    assert (dst_metal / "alias.py").is_symlink()
    assert os.readlink(dst_metal / "alias.py") == "real.py"
    # root-level dangling link dropped
    assert not (dst_metal / "foo").is_symlink() and not (dst_metal / "foo").exists()
    # `build` link dropped (its target was excluded) — never shipped as a dangling link
    assert not (dst_metal / "build").is_symlink()
    assert not (dst_metal / "build_Release").exists()  # excluded build output
    # link to an OUTSIDE dir materialized as a real copy (no leaked host-absolute link)
    assert (dst_metal / "linked_outside").is_dir()
    assert not (dst_metal / "linked_outside").is_symlink()
    assert (dst_metal / "linked_outside" / "data.txt").read_text() == "outside content\n"
    # depth-3 absolute dangling link dropped; its sibling real file survives
    assert (dst_metal / "tt_metal" / "third_party" / "umd" / "keep.cpp").is_file()
    assert not (dst_metal / "tt_metal" / "third_party" / "umd" / "compile_commands.json").is_symlink()
    # ROOT-anchoring: nested python_env kept, root python_env (+ other caches) dropped
    assert (dst_metal / "tt_metal" / "python_env" / "keep.txt").read_text() == "tracked nested file\n"
    for cache in (".cpmcache", "python_env", "tt_cache", "built", "built_kernels"):
        assert not (dst_metal / cache).exists()
    # The invariant that matters for a shipped artifact: no dangling links anywhere.
    assert not any(p.is_symlink() and not p.exists() for p in dst_metal.rglob("*"))


def test_stage_package_materialized_escaping_dir_is_filtered(tmp_path):
    """Regression: materializing a symlink that escapes the tree and points at a real directory
    must still drop junk at every depth (VCS, byte-caches, ...) — empirically, an escaping link
    shipped `.git/HEAD` and `__pycache__/junk.pyc`. (Pre-`symlinks=True`, the followed link was
    filtered too.)

    It must NOT apply the metal-tree's ROOT-ONLY excludes (`python_env`/`build`/`tt_cache`/...) to
    the materialized target: those assume a metal-checkout layout that an arbitrary symlink target
    doesn't share, and applying them anyway silently drops real content — asserted below via a
    `python_env` dir at the materialized root that must ship, not vanish.
    """
    wheels = tmp_path / "in_wheels"
    wheels.mkdir()
    ttnn = _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn-bytes")

    # A real OUTSIDE directory a metal symlink escapes into — full of what the excludes remove.
    outside = tmp_path / "outside_build"
    outside.mkdir()
    (outside / "wanted.txt").write_text("real content the author wants\n")
    (outside / ".git").mkdir()
    (outside / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (outside / "__pycache__").mkdir()
    (outside / "__pycache__" / "junk.pyc").write_bytes(b"\x00junk")
    # `python_env` sits at the materialized root too, but — unlike the metal tree itself — this is
    # NOT a metal checkout, so the root-only exclude must NOT apply: it is real content and ships.
    (outside / "python_env").mkdir()
    (outside / "python_env" / "real_config.txt").write_text("kept: not a metal tree\n")

    metal = tmp_path / "metal_src"
    metal.mkdir()
    (metal / "requirements.txt").write_text("torch==2.11.0\n")
    (metal / "escapes").symlink_to(outside)  # the escaping directory link

    vmeta = {"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"}
    staged = tmp_path / "staged"
    packaging.stage_package(
        staged, name="m", arch="blackhole", ttnn_wheel=ttnn,
        metal_dir=metal, vllm_metadata=vmeta, tt_kernel_version="0.0.0",
    )

    mat = staged / "metal" / "escapes"
    assert mat.is_dir() and not mat.is_symlink()  # materialized, not a leaked link
    assert (mat / "wanted.txt").read_text() == "real content the author wants\n"
    # The junk the exclude list exists to remove must NOT have been re-imported:
    assert not (mat / ".git").exists()          # _METAL_IGNORE_ANYWHERE (VCS)
    assert not (mat / "__pycache__").exists()   # _METAL_IGNORE_ANYWHERE (byte-cache)
    # A materialized target isn't a metal checkout: root-only excludes don't apply, so this ships.
    assert (mat / "python_env" / "real_config.txt").read_text() == "kept: not a metal tree\n"


def test_stage_package_materialized_junk_named_target_is_dropped(tmp_path):
    """Security regression: a symlink under an escaping directory, itself named innocuously but
    resolving DIRECTLY to a junk-named directory (e.g. a `.git`), must not be materialized.

    `copytree`'s `ignore=` only ever filters the *children* of a directory it walks — it never
    checks the root of the copy against the exclude patterns. Without a check at the point of
    materialization, `hist -> /outside/.git` survives the initial filter (its own name doesn't
    match `.git`), and copying its target as the root of a fresh `copytree` would ship the `.git`
    directory's contents whole — verified here with a fake credential in `.git/config`.
    """
    wheels = tmp_path / "in_wheels"
    wheels.mkdir()
    ttnn = _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn-bytes")

    outside = tmp_path / "outside_repo"
    outside.mkdir()
    (outside / ".git").mkdir()
    (outside / ".git" / "config").write_text("[credentials]\ntoken = super-secret\n")
    (outside / "pkg").mkdir()
    # Absolute link, named "hist" (not ".git") -> the real .git dir. Its own name isn't excluded.
    (outside / "pkg" / "hist").symlink_to((outside / ".git").resolve(), target_is_directory=True)
    (outside / "model.py").write_text("# real model code\n")

    metal = tmp_path / "metal_src"
    metal.mkdir()
    (metal / "requirements.txt").write_text("torch==2.11.0\n")
    (metal / "escapes").symlink_to(outside)

    vmeta = {"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"}
    staged = tmp_path / "staged"
    packaging.stage_package(
        staged, name="m", arch="blackhole", ttnn_wheel=ttnn,
        metal_dir=metal, vllm_metadata=vmeta, tt_kernel_version="0.0.0",
    )

    mat = staged / "metal" / "escapes"
    assert (mat / "model.py").is_file()  # real content still ships
    # The .git reached via a non-".git"-named symlink must not have been materialized at all.
    assert not (mat / "pkg" / "hist").exists()
    assert not any(p.name == "config" for p in mat.rglob("*"))


def test_stage_package_materialized_link_into_junk_dir_is_dropped(tmp_path):
    """Security regression, the FILE form of the junk-named-target leak: a symlink under an
    innocuous name resolving to a file *inside* an excluded directory (`gitcfg -> /outside/.git/
    config`) must not be materialized either.

    Checking only the target's own basename closes the directory form (`hist -> .git`) but not
    this one — the basename is `config`, and the `copy2` branch that handles a file target applies
    no filtering at all. The whole escaping path is classified, so the `.git` component is caught
    wherever it sits. Covered at both depths: under an escaping directory, and straight off the
    metal root.
    """
    wheels = tmp_path / "in_wheels"
    wheels.mkdir()
    ttnn = _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn-bytes")

    outside = tmp_path / "outside_repo"
    outside.mkdir()
    (outside / ".git").mkdir()
    (outside / ".git" / "config").write_text("[credentials]\ntoken = super-secret\n")
    (outside / "pkg").mkdir()
    (outside / "pkg" / "model.py").write_text("# real model code\n")
    # Named "gitcfg" (not ".git", not "config"-excluded) -> a single file inside the .git dir.
    (outside / "pkg" / "gitcfg").symlink_to((outside / ".git" / "config").resolve())

    metal = tmp_path / "metal_src"
    metal.mkdir()
    (metal / "requirements.txt").write_text("torch==2.11.0\n")
    (metal / "escapes").symlink_to(outside)
    # The same bypass one level up: straight off the metal root, no escaping directory involved.
    (metal / "hostcfg").symlink_to((outside / ".git" / "config").resolve())

    vmeta = {"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"}
    staged = tmp_path / "staged"
    packaging.stage_package(
        staged, name="m", arch="blackhole", ttnn_wheel=ttnn,
        metal_dir=metal, vllm_metadata=vmeta, tt_kernel_version="0.0.0",
    )

    dst_metal = staged / "metal"
    assert (dst_metal / "escapes" / "pkg" / "model.py").is_file()  # real content still ships
    assert not (dst_metal / "escapes" / "pkg" / "gitcfg").exists()
    assert not (dst_metal / "hostcfg").exists()
    assert not any("super-secret" in p.read_text(errors="ignore")
                   for p in dst_metal.rglob("*") if p.is_file())


def test_stage_package_ambient_junk_named_parent_dir_still_ships(tmp_path):
    """The path-wide junk check must judge only what the link reaches INTO, not the host's ambient
    layout: staging from a workspace that happens to sit under a junk-matching directory name
    (`generated/`, a `build_*` CI dir) must not start dropping ordinary escaping content.

    The shared prefix between the staged tree and the target is skipped for exactly this reason —
    without that, every escaping link in such a workspace would silently vanish.
    """
    workspace = tmp_path / "generated"  # matches _METAL_IGNORE_ANYWHERE, and holds EVERYTHING
    workspace.mkdir()

    wheels = workspace / "in_wheels"
    wheels.mkdir()
    ttnn = _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn-bytes")

    outside = workspace / "outside_lib"
    outside.mkdir()
    (outside / "kernel.so").write_text("compiled\n")

    metal = workspace / "metal_src"
    metal.mkdir()
    (metal / "requirements.txt").write_text("torch==2.11.0\n")
    (metal / "lib.so").symlink_to((outside / "kernel.so").resolve())

    vmeta = {"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"}
    staged = workspace / "staged"
    packaging.stage_package(
        staged, name="m", arch="blackhole", ttnn_wheel=ttnn,
        metal_dir=metal, vllm_metadata=vmeta, tt_kernel_version="0.0.0",
    )

    materialized = staged / "metal" / "lib.so"
    assert materialized.is_file() and not materialized.is_symlink()
    assert materialized.read_text() == "compiled\n"


def test_stage_package_special_file_raises_styled_error(tmp_path):
    """A copytree abort (here a fifo → SpecialFileError) surfaces as StagingError with the path,
    not a raw traceback; the CLI renders it via _err (non-zero exit, no crash)."""
    import pytest

    wheels = tmp_path / "in_wheels"
    wheels.mkdir()
    ttnn = _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn-bytes")
    metal = tmp_path / "metal_src"
    metal.mkdir()
    (metal / "requirements.txt").write_text("torch==2.11.0\n")
    os.mkfifo(metal / "a_socket")  # copytree cannot copy a fifo → shutil.Error at end of walk

    vmeta = {"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"}
    with pytest.raises(packaging.StagingError) as ei:
        packaging.stage_package(
            tmp_path / "staged", name="m", arch="blackhole", ttnn_wheel=ttnn,
            metal_dir=metal, vllm_metadata=vmeta, tt_kernel_version="0.0.0",
        )
    assert any("a_socket" in p for p in ei.value.paths)

    # And through the CLI: styled error, non-zero exit, no traceback.
    res = _runner.invoke(
        cli.app,
        ["package", "--from-metal", str(metal), "--ttnn-wheel", str(ttnn),
         "--arch", "blackhole", "--arch-name", "X", "--main-class", "m:C",
         "--no-repair", "--no-vendor-deps", "--out", str(tmp_path / "s")],
    )
    assert res.exit_code == 1
    assert "metal tree" in res.output


def test_stage_package_normalize_failure_surfaces_as_staging_error(tmp_path, monkeypatch):
    """A failure inside the symlink-normalization pass (its own unlink/copy2/copytree can hit
    EACCES/ENOSPC) must surface as a styled StagingError, not a raw traceback — i.e. the call is
    inside the same try/except as the initial copytree, not after it."""
    import pytest

    wheels = tmp_path / "in_wheels"
    wheels.mkdir()
    ttnn = _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn-bytes")
    metal = tmp_path / "metal_src"
    metal.mkdir()
    (metal / "requirements.txt").write_text("torch==2.11.0\n")

    # copytree succeeds; the normalization pass then blows up (simulate an ENOSPC/EACCES).
    def _boom(root):
        raise OSError("[Errno 28] No space left on device: normalize")

    monkeypatch.setattr(packaging, "_normalize_staged_symlinks", _boom)

    vmeta = {"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"}
    with pytest.raises(packaging.StagingError) as ei:
        packaging.stage_package(
            tmp_path / "staged", name="m", arch="blackhole", ttnn_wheel=ttnn,
            metal_dir=metal, vllm_metadata=vmeta, tt_kernel_version="0.0.0",
        )
    assert "metal tree" in str(ei.value)  # wrapped with staging context, not a bare OSError


def test_cli_package_stage_only(tmp_path):
    """`tt-model package ... --out <dir>` (no repo_id) stages the folder, no network."""
    wheels = tmp_path / "w"
    wheels.mkdir()
    _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn")
    _fake_wheel(wheels, "vllm_tt_plugin-0.3.0-py3-none-any.whl")
    metal = tmp_path / "metal"
    metal.mkdir()
    (metal / "requirements.txt").write_text("torch==2.11.0\n")
    out = tmp_path / "staged"

    res = _runner.invoke(
        cli.app,
        [
            "package",
            "--from-metal", str(metal),
            "--wheels-dir", str(wheels),
            "--arch", "blackhole",
            "--arch-name", "LlamaForCausalLM",
            "--main-class", "generator_vllm:LlamaForCausalLM",
            "--weights", "unsloth/Llama-3.2-3B-Instruct",
            "--mesh", "P150",
            "--no-repair", "--no-vendor-deps",
            "--out", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    m = Manifest.from_json((out / "tt_kernel_manifest.json").read_text())
    assert m.is_self_contained and m.arch == "blackhole"
    # wheels_dir auto-classified ttnn + plugin
    assert m.bundled.ttnn_wheel is not None and m.bundled.plugin_wheel is not None
    assert (out / "wheels" / "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl").is_file()


def test_cli_package_requires_ttnn_wheel(tmp_path):
    metal = tmp_path / "metal"
    metal.mkdir()
    res = _runner.invoke(
        cli.app,
        ["package", "--from-metal", str(metal), "--arch", "blackhole",
         "--arch-name", "X", "--main-class", "m:C", "--out", str(tmp_path / "s")],
    )
    assert res.exit_code != 0
