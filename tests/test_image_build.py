# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The image definition: the generated build scripts and the Dockerfile's invariants.

A cold build is 2.5-4 hours, so every property that can be asserted without one is
asserted here. The Dockerfile tests are deliberately about *invariants that were bugs* —
each maps to a comment in the file explaining what broke when it was absent.
"""

import json
from pathlib import Path

import pytest

from tt_kernel.container_manifest import ContainerManifest
from tt_kernel.launchers import launcher_for

from test_container_manifest import BASE, FORK

DOCKER_DIR = Path(__file__).parent.parent / "src" / "tt_kernel" / "docker"
DOCKERFILE = (DOCKER_DIR / "Dockerfile").read_text()
ENTRYPOINT = (DOCKER_DIR / "entrypoint.sh").read_text()


def _mani(**over) -> ContainerManifest:
    raw = json.loads(json.dumps(BASE))
    raw.update(over)
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    return m


def _install(**over) -> str:
    m = _mani(**over)
    return "\n".join(launcher_for(m.kind).install_lines(m))


def _verify(**over) -> str:
    m = _mani(**over)
    return "\n".join(launcher_for(m.kind).verify_lines(m))


def _verify_payloads(**over) -> str:
    """The python source each verify line runs, recovered from the shell quoting.

    Asserting on the quoted shell line would test shlex.quote's escaping, not ours; this
    also proves every generated line is a well-formed `python -c` invocation.
    """
    import shlex

    out = []
    m = _mani(**over)
    for line in launcher_for(m.kind).verify_lines(m):
        argv = shlex.split(line)
        assert argv[1] == "-c", argv
        out.append(argv[2])
    return "\n".join(out)


# ------------------------------------------------------------------ install_engine.sh


def test_the_plugin_is_cloned_at_the_pinned_sha_and_installed_non_editable():
    """Non-editable: the clone need not survive into the runtime image."""
    s = _install()
    assert "git clone https://github.com/tenstorrent/vllm-tt-plugin /tmp/vllm-tt-plugin" in s
    assert "checkout bc4af2d5" in s
    assert "rm -rf /tmp/vllm-tt-plugin" in s


def test_the_numpy_opencv_conflict_is_resolved_by_an_override():
    """ttnn pins numpy<2; recent vLLM's opencv wants numpy>=2. pip cannot express the
    resolution; uv's --override can."""
    s = _install()
    assert "numpy>=1.24.4,<2" in s and "opencv-python-headless==4.11.0.86" in s
    assert "--override /tmp/tt-overrides.txt" in s


def test_torchaudio_is_removed_after_the_vllm_install():
    """transformers imports it if it is merely INSTALLED, and the wheel riding along with
    CPU torch is unloadable."""
    assert "uv pip uninstall" in _install() and "torchaudio" in _install()


def test_a_pypi_plugin_release_skips_the_clone():
    rt = json.loads(json.dumps(BASE))["runtime"]
    rt["plugin"] = {"version": "1.2.3"}
    s = _install(runtime=rt)
    assert "vllm-tt-plugin==1.2.3" in s and "git clone" not in s


def test_the_fork_is_cloned_and_checked_out_at_the_pinned_ref():
    s = _install(**FORK)
    assert "git clone https://github.com/tenstorrent/vllm /opt/vllm" in s
    assert "checkout bf98d556" in s


def test_a_resolved_sha_wins_over_the_authored_ref():
    """`package` rewrites ref -> sha; the build must use the resolved one."""
    rt = json.loads(json.dumps(BASE))["runtime"]
    rt["plugin"]["sha"] = "deadbeef"
    assert "checkout deadbeef" in _install(runtime=rt)


def test_vllm_builds_with_the_empty_device_target_and_the_cpu_torch_index():
    """There is no CUDA on a TT box, and the published wheel is the CUDA build."""
    s = _install()
    assert "VLLM_TARGET_DEVICE=empty" in s
    assert "https://download.pytorch.org/whl/cpu" in s


def test_the_fork_and_its_in_tree_plugin_are_both_installed_editable():
    s = _install(**FORK)
    assert "-e /opt/vllm " in s or "-e /opt/vllm\n" in s
    assert "-e /opt/vllm/plugins/vllm-tt-plugin" in s


def test_git_metadata_is_dropped_but_the_checkout_survives():
    """Editable installs mean /opt/vllm must exist in the runtime image, .git need not."""
    assert "rm -rf /opt/vllm/.git" in _install(**FORK)


def test_a_lock_file_makes_the_install_no_deps():
    """With a lock, the lock IS the dependency set: nothing resolves at build time."""
    rt = json.loads(json.dumps(BASE))["runtime"]
    rt["lock"] = "requirements.lock"
    s = _install(runtime=rt)
    assert "-r /ctx/requirements.lock" in s
    assert "--no-deps " in s


def test_without_a_lock_nothing_is_no_deps():
    assert "--no-deps" not in _install()


def test_the_model_extension_is_installed_from_the_staged_tree():
    rt = json.loads(json.dumps(BASE))["runtime"]
    rt["extension"] = "models/common/vllm_ext"
    assert "/opt/tt-metal/models/common/vllm_ext" in _install(runtime=rt)


def test_repo_and_ref_are_shell_quoted():
    """These come from a YAML file an author wrote; they end up in a shell script."""
    rt = json.loads(json.dumps(BASE))["runtime"]
    rt["plugin"]["ref"] = "a b; rm -rf /"
    assert "'a b; rm -rf /'" in _install(runtime=rt)


# ------------------------------------------------------------------ verify.sh


def test_verify_checks_the_imports_that_matter():
    s = _verify_payloads()
    assert "import ttnn, vllm, vllm_tt_plugin" in s
    s_fork = _verify_payloads(**FORK)
    assert "models.common.readiness_check.run_vllm_server" in s_fork


def test_verify_asserts_cpu_torch():
    """A CUDA torch would install and import fine here and fail only on device open."""
    assert "'+cpu'" in _verify_payloads()


def test_verify_asserts_vllm_resolves_to_the_fork():
    """If vLLM resolves anywhere else, the editable install silently lost."""
    assert "startswith('/opt/vllm')" in _verify_payloads(**FORK)


def test_the_plugin_kind_asserts_vllm_did_NOT_resolve_into_the_tree():
    """Installed non-editable, so a tree-resolved vLLM means something went wrong."""
    assert "'/tt-metal/' not in vllm.__file__" in _verify_payloads()


def test_verify_checks_extra_models_dir_registers_something():
    """A vllm_metadata.json in the wrong place registers ZERO architectures, silently."""
    rt = json.loads(json.dumps(BASE))["runtime"]
    rt["extra_models_dir"] = "models/common"
    assert "registers no models" in _verify_payloads(runtime=rt)


def test_no_extra_models_check_when_the_model_uses_the_builtin_registry():
    assert "registers no models" not in _verify_payloads()


def test_model_authored_assertions_are_appended_and_quoted():
    s = _verify_payloads(verify=["import models.common as m; assert m"])
    assert "import models.common as m; assert m" in s


# ------------------------------------------------------------------ Dockerfile invariants


def test_models_is_excluded_from_the_metal_tree():
    """The staged code/ allowlist must be the ONLY `models` package in the image, or an
    under-specified allowlist would be masked by the fork's own models/."""
    assert "--exclude=models" in DOCKERFILE


def test_the_ccache_and_cpm_cache_mounts_are_present():
    """Without these, iterating on any later stage re-pays the 1.5-2.5h C++ build."""
    assert "type=cache,target=/root/.cache/ccache" in DOCKERFILE
    assert "type=cache,target=/cpm" in DOCKERFILE


def test_a_git_stub_is_created_for_the_cmake_version_probe():
    """tt-metal's CMake derives PROJECT_VERSION from `git describe` and FAILS on empty."""
    assert "git init -q /opt/tt-metal" in DOCKERFILE
    assert 'tag "${METAL_DESCRIBE}"' in DOCKERFILE


def test_the_runtime_user_is_uid_1000_with_a_real_home():
    """tt-metal derives its JIT cache dir from $HOME; a --user with no passwd entry gets
    HOME=/ and the JIT build dies."""
    assert "useradd --uid 1000 --create-home --home-dir /home/tt" in DOCKERFILE
    assert "USER tt" in DOCKERFILE


def test_the_runtime_dirs_are_writable_by_an_arbitrary_uid():
    """`serve` passes --user <host uid>, which may have no passwd entry and will often not
    be 1000. HOME must be set explicitly (glibc would give an unknown uid HOME=/, which is
    unwritable and kills the JIT) and the runtime dirs must be world-writable."""
    assert "chmod 1777 /home/tt" in DOCKERFILE
    assert "ENV HOME=/home/tt" in DOCKERFILE


def test_the_uid_has_a_name_without_a_passwd_entry():
    """getpass.getuser() falls back to pwd.getpwuid() and raises KeyError for an unknown
    uid. torch's inductor calls it to name its cache dir, so the server would die on
    import. Naming the user in the environment avoids making /etc/passwd writable."""
    assert "ENV USER=tt" in DOCKERFILE
    assert "ENV LOGNAME=tt" in DOCKERFILE


def test_uid_1000_is_freed_before_it_is_claimed():
    """Ubuntu 24.04's base image ships a user at uid 1000 (`ubuntu`); 22.04 did not. An
    unconditional useradd dies with "UID 1000 is not unique" on any 24.04 build."""
    assert "getent passwd 1000" in DOCKERFILE
    assert "userdel" in DOCKERFILE
    # freed by ID, not by assuming the name — base images differ
    assert "userdel -r ubuntu" not in DOCKERFILE


def test_the_metal_tree_is_owned_by_the_runtime_user():
    """tools/tracy/common.py mkdirs inside TT_METAL_HOME on IMPORT — root-owned trees
    turn that into EACCES ten minutes into a boot."""
    assert "--chown=tt:tt /opt/tt-metal/ttnn" in DOCKERFILE
    assert "--chown=tt:tt code/" in DOCKERFILE


def test_the_sfpi_cross_compilers_host_libraries_are_installed():
    """The JIT's RISC-V cc1plus is a HOST binary; without these the first kernel compile
    dies with 'cc1plus: error while loading shared libraries: libmpc.so.3'."""
    for lib in ("libmpc3", "libmpfr6", "libgmp10", "libzstd1"):
        assert lib in DOCKERFILE


def test_ld_library_path_carries_both_metal_libs_and_ulfm_mpi():
    assert "LD_LIBRARY_PATH=/opt/tt-metal/build/lib:${OMPI_DIR}/lib" in DOCKERFILE


def test_home_is_reopened_after_verification_created_cache_dirs():
    """verify.sh runs as uid 1000 and its imports create $HOME cache dirs (matplotlib,
    ttnn, torch, vllm) at 0755. The arbitrary uid `serve` passes cannot write inside them,
    so the tree has to be re-opened AFTER verification, not only before."""
    chmod_after = DOCKERFILE.rindex("chmod -R a+rwX /home/tt")
    verify_run = DOCKERFILE.index("bash /ctx/verify.sh")
    assert chmod_after > verify_run


def test_the_verification_run_is_in_the_runtime_stage_after_the_user_switch():
    """Verifying as root would miss exactly the EACCES class of bug it exists to catch."""
    assert DOCKERFILE.index("USER tt") < DOCKERFILE.index("bash /ctx/verify.sh")


def test_the_entrypoint_and_cmd_close_the_file():
    """ENTRYPOINT was the last word until the image gained a default CMD; the pair is now
    the tail, and their ORDER matters — CMD is the argv that ENTRYPOINT execs."""
    tail = [ln for ln in DOCKERFILE.rstrip().splitlines() if ln and not ln.startswith("#")]
    assert tail[-2] == 'ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]'
    assert tail[-1] == 'CMD ["/usr/local/bin/serve-default.sh"]' 


def test_the_dockerfile_never_mentions_a_specific_serving_stack():
    """Kind-specific work arrives via the generated scripts; this file stays generic."""
    body = "\n".join(
        ln for ln in DOCKERFILE.splitlines() if not ln.strip().startswith("#")
    )
    assert "pip install vllm" not in body
    assert "vllm-tt-plugin" not in body


# ------------------------------------------------------------------ entrypoint.sh


def test_the_entrypoint_execs_so_the_server_is_pid_1():
    """PID 1 receives docker stop's SIGTERM; a clean shutdown closes the mesh."""
    assert ENTRYPOINT.rstrip().endswith('exec "$@"')


def test_the_entrypoint_unsets_vllm_plugins():
    """VLLM_PLUGINS is an ALLOW-list: setting it silently kills the model's parsers."""
    assert "unset VLLM_PLUGINS" in ENTRYPOINT


def test_the_entrypoint_unsets_an_empty_builtin_models_value():
    """Empty would register zero architectures; the plugin needs its own default."""
    assert "unset TT_VLLM_BUILTIN_MODELS" in ENTRYPOINT


def test_the_entrypoint_moves_to_a_writable_cwd():
    """Inspector/watcher/model_cache write RELATIVE to cwd."""
    assert 'cd "$HOME/work"' in ENTRYPOINT


def test_the_entrypoint_is_strict():
    assert "set -euo pipefail" in ENTRYPOINT


# ------------------------------------------------------------------ torch agreement
#
# tt-metal's setup.py has no install_requires and there is no runtime requirements.txt,
# so installing tt-metal brings NO torch — torch arrives only as a transitive dependency
# of vLLM. Nothing makes the two agree unless the image checks.

TORCH_REQ = "tt_metal/python_env/requirements-dev.txt"


def _metal_tree(tmp_path, body: str) -> Path:
    req = tmp_path / TORCH_REQ
    req.parent.mkdir(parents=True)
    req.write_text(body)
    (tmp_path / "models" / "common").mkdir(parents=True)
    return tmp_path


REAL_REQ_SNIPPET = (
    "# CPU-only torch for x86_64 to avoid installing CUDA dependencies on dev machines\n"
    "--extra-index-url https://download.pytorch.org/whl/cpu\n"
    "torch==2.11.0 ; platform_machine == 'x86_64'\n"
    "torchvision==0.26.0 ; platform_machine == 'x86_64'\n"
)


def test_the_torch_pin_is_read_from_tt_metals_own_requirements(tmp_path):
    from tt_kernel.launchers import metal_torch_pin

    assert metal_torch_pin(_metal_tree(tmp_path, REAL_REQ_SNIPPET)) == "2.11.0"


def test_the_aarch64_pin_does_not_win_over_the_x86_64_one(tmp_path):
    """The file carries both; the x86_64 line is the one that applies here."""
    from tt_kernel.launchers import metal_torch_pin

    body = "torch==9.9.9 ; platform_machine == 'aarch64'\n" + REAL_REQ_SNIPPET
    assert metal_torch_pin(_metal_tree(tmp_path, body)) == "2.11.0"


def test_an_unreadable_tree_skips_the_check_rather_than_guessing(tmp_path):
    from tt_kernel.launchers import metal_torch_pin

    assert metal_torch_pin(tmp_path / "nope") is None
    assert metal_torch_pin(None) is None


def test_verify_asserts_torch_matches_the_metal_pin(tmp_path):
    src = dict(BASE["source"], tt_metal=str(_metal_tree(tmp_path, REAL_REQ_SNIPPET)))
    s = _verify_payloads(source=src)
    assert "'2.11.0'" in s
    assert "tt-metal pins 2.11.0" in s


def test_no_torch_assertion_when_the_pin_cannot_be_determined(tmp_path):
    """A git-mode source has no local tree; skip rather than assert a guess."""
    src = dict(BASE["source"], tt_metal={"repo": "https://x/y", "ref": "main"})
    assert "tt-metal pins" not in _verify_payloads(source=src)


def test_the_cpu_suffix_check_survives_alongside_the_version_check(tmp_path):
    """Both matter: +cpu catches the CUDA build, the version catches a torch drift."""
    src = dict(BASE["source"], tt_metal=str(_metal_tree(tmp_path, REAL_REQ_SNIPPET)))
    s = _verify_payloads(source=src)
    assert "'+cpu'" in s and "'2.11.0'" in s


# ------------------------------------------------------------------ generated-script sanity


def test_every_generated_verify_line_is_a_compilable_program():
    """The checks are joined with "; " into one `python -c`, which a snippet containing a
    for-loop cannot survive — such a snippet has to be emitted as its own line. Compiling
    each line catches that class of bug here instead of at the end of a full image build."""
    import shlex

    for kind_over in ({}, FORK):
        m = _mani(**kind_over)
        rt = json.loads(json.dumps(BASE))["runtime"]
        rt["extra_models_dir"] = "models/common"
        for over in (kind_over, {**kind_over, "runtime": rt} if not kind_over else kind_over):
            mm = _mani(**over)
            for i, line in enumerate(launcher_for(mm.kind).verify_lines(mm)):
                argv = shlex.split(line)
                assert argv[1] == "-c", argv
                compile(argv[2], f"<verify {mm.kind} {i}>", "exec")


def test_every_generated_install_line_is_a_single_shell_command():
    """install_engine.sh runs under `set -euxo pipefail`; a line that is accidentally two
    statements would half-execute."""
    import shlex

    for over in ({}, FORK):
        m = _mani(**over)
        for line in launcher_for(m.kind).install_lines(m):
            shlex.split(line)  # raises on unbalanced quoting


def test_extra_models_verification_resolves_the_class_not_just_the_file():
    """Registration is lazy: the plugin stores "module:Class" and vLLM imports it much
    later, in the API server and again in each EngineCore worker. A shim that computes
    the repo root by directory depth resolves on the author's box and fails in the image."""
    rt = json.loads(json.dumps(BASE))["runtime"]
    rt["extra_models_dir"] = "models/common"
    s = _verify_payloads(runtime=rt)
    assert "importlib.import_module" in s
    assert "sys.path.append" in s          # mirrors the plugin: append, never insert(0)
    assert "main_class" in s
    assert "registers no models" in s      # the weaker check still holds too


# ------------------------------------------------------------------ the default CMD
#
# The image had no default command: ENTRYPOINT simply exec'd its arguments, and the
# correct invocation lived only in the manifest, on the host, inside tt-model. A Docker
# image invites `docker run`, and run that way the server started MISCONFIGURED — no
# mesh, no tt additional-config, no builtin-registry suppression — and emitted nonsense
# with no error anywhere.


def _default_serve(**over) -> str:
    from tt_kernel.build import render_default_serve

    m = _mani(**over)
    return render_default_serve(m, launcher_for(m.kind))


def test_the_default_command_carries_the_serve_env():
    """These are what `tt-model serve` passes with --env; without them the server runs
    misconfigured rather than failing."""
    s = _default_serve()
    assert "export MESH_DEVICE=" in s
    assert "export HF_MODEL=" in s


def test_the_default_command_is_the_launcher_argv():
    s = _default_serve()
    assert "exec vllm serve" in s
    assert "--block-size" in s and "--max-num-seqs" in s


def test_the_default_command_is_a_valid_shell_program():
    import subprocess
    import sys

    r = subprocess.run(["bash", "-n"], input=_default_serve(), text=True,
                       capture_output=True)
    assert r.returncode == 0, r.stderr


def test_values_are_shell_quoted():
    """additional-config is JSON with spaces and braces; unquoted it would be mangled."""
    s = _default_serve(serve={"port": 8000, "block_size": 64,
                              "additional_config": {"tt": {"a b": "c d"}}})
    assert "'{\"tt\": {\"a b\": \"c d\"}}'" in s


def test_it_refuses_when_the_devices_are_absent_and_names_the_flags():
    """`docker run <image>` with no --device is the exact mistake this guards."""
    s = _default_serve()
    assert "if [ ! -e /dev/tenstorrent ]" in s
    assert "tt-model serve" in s
    assert "--device /dev/tenstorrent" in s
    assert "/dev/hugepages-1G" in s
    assert "exit 1" in s


def test_the_refusal_runs_before_anything_else():
    s = _default_serve()
    assert s.index("/dev/tenstorrent") < s.index("exec vllm")


def test_the_dockerfile_sets_CMD_and_keeps_the_entrypoint():
    assert 'CMD ["/usr/local/bin/serve-default.sh"]' in DOCKERFILE
    assert 'ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]' in DOCKERFILE
    # ENTRYPOINT execs "$@", so CMD flows through it and the server is still PID 1
    assert DOCKERFILE.index("ENTRYPOINT") < DOCKERFILE.index("CMD [")


def test_the_script_is_copied_executable():
    assert "--chmod=0755 serve-default.sh" in DOCKERFILE
