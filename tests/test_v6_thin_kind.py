# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for a non-vLLM `kind` on v6 "thin" bundles (issue #29): a "tt-dit-server" kind serves
`deps.app` directly with uvicorn instead of vLLM's OpenAI server — the same kind name and shape
`launchers.TtDitServerLauncher` already gives the v5.1 CONTAINER schema, extended here to the
non-container v6 path. No hardware, no network — staging + rendering only, matching
test_v6_thin.py's own style.

Motivating bug (reproduced against the pre-fix code before writing this file): even
`stage_thin_package(with_vllm=False)` — the schema's existing "non-vLLM model" escape hatch — still
rendered a run.sh whose CMD line unconditionally launched `vllm.entrypoints.openai.api_server`.
`render_run_sh` never looked at `with_vllm`/`deps.vllm` at all; only `render_install_sh` did. The
tests below pin down the fix (`deps.kind` now gates run.sh's CMD line, not just install.sh's vLLM
step) and the new validation this kind requires.
"""

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, packaging
from tt_kernel.launchers import TtDitServerLauncher
from tt_kernel.manifest import Manifest, Mesh, WeightsRef

_runner = CliRunner()


def _stage_dit(tmp_path, requirements=None):
    model_py = tmp_path / "app.py"
    model_py.write_text("# ASGI app lives elsewhere; this is just the shipped runner file.\n")
    staged = tmp_path / "thin-dit"
    m = packaging.stage_thin_package(
        staged, name="openvla-thin", arch="blackhole", model_py=model_py,
        tt_kernel_version="0.0.0", kind="tt-dit-server", app="gradio_app.asgi:app",
        with_vllm=False, requirements=requirements,
        weights=WeightsRef(repo="openvla/openvla-7b"), device_count=2,
        mesh=Mesh(devices=2, topology="P300"),
    )
    return staged, m


def test_dit_kind_layout_and_manifest(tmp_path):
    staged, m = _stage_dit(tmp_path)
    assert m.schema_version == "6"
    assert m.is_thin is True and m.has_own_venv is True
    assert m.deps.kind == "tt-dit-server"
    assert m.deps.app == "gradio_app.asgi:app"
    assert m.deps.vllm is None
    # No vLLM plugin registration surface at all for this kind.
    assert m.entrypoint is None
    assert not (staged / "vllm-overrides.txt").exists()
    assert not (staged / "vllm_models").exists()
    m2 = Manifest.from_json((staged / "tt_kernel_manifest.json").read_text())
    assert m2.deps.kind == "tt-dit-server" and m2.deps.app == "gradio_app.asgi:app"


def test_dit_kind_run_sh_serves_app_with_uvicorn_not_vllm(tmp_path):
    staged, _ = _stage_dit(tmp_path)
    run = (staged / "run.sh").read_text()
    assert "vllm.entrypoints.openai.api_server" not in run
    # deps.app is shell-quoted (shlex.quote), not raw-interpolated — a plain "module:attribute"
    # string has no shell-unsafe characters, so shlex.quote leaves it bare (no added quotes).
    assert 'CMD=("$PYBIN" -m uvicorn --host 0.0.0.0 --port "${PORT:-8000}" ' \
           '--lifespan on gradio_app.asgi:app "$@")' in run


def test_dit_kind_install_sh_has_no_vllm_step(tmp_path):
    staged, _ = _stage_dit(tmp_path)
    inst = (staged / "install.sh").read_text()
    assert "VLLM_TARGET_DEVICE" not in inst and "requirements/common.txt" not in inst


def test_dit_kind_default_requirements_has_base_http_stack(tmp_path):
    staged, _ = _stage_dit(tmp_path)
    req = (staged / "requirements.txt").read_text()
    for pkg in TtDitServerLauncher.DEFAULT_PACKAGES:
        assert pkg in req
    # No resolvable vllm pin (a comment mentioning why there's no vLLM step is fine).
    assert "\nvllm==" not in req and "\nvllm>=" not in req and "\nvllm\n" not in req


def test_dit_kind_authored_requirements_gets_defaults_merged_by_name(tmp_path):
    # Author already pins their OWN pydantic version -- it must win, not be duplicated.
    authored = tmp_path / "requirements.txt"
    authored.write_text("pydantic==2.5.0\ntransformers==4.52.4\n")
    staged, _ = _stage_dit(tmp_path, requirements=authored)
    req = (staged / "requirements.txt").read_text()
    assert "pydantic==2.5.0" in req
    assert "pydantic>=2" not in req  # the default, since the author's own pin already covers it
    assert "transformers==4.52.4" in req
    for pkg in ("fastapi", "uvicorn", "pillow"):  # not authored -> appended
        assert pkg in req


def test_dit_kind_run_sh_shell_quotes_app_against_injection(tmp_path):
    # Motivating bug: render_run_sh used to interpolate deps.app raw inside a double-quoted bash
    # array literal. Double quotes don't stop $()/backtick command substitution or an embedded
    # quote breaking out of the word, so a malicious/corrupted app string could inject shell
    # commands the moment run.sh was sourced (reproduced live before this fix: the injected
    # `touch` ran merely from constructing the CMD array, before uvicorn was ever invoked).
    staged, _ = _stage_dit(tmp_path)
    # Bypass the CLI/packaging ":"-format check (that's covered separately below) by mutating the
    # staged manifest directly, mirroring how any non-CLI caller could set deps.app.
    m = Manifest.from_json((staged / "tt_kernel_manifest.json").read_text())
    m.deps.app = "mymodule:app$(touch /tmp/pwned-by-test)"
    run = packaging.render_run_sh(m)
    # The dangerous substring must appear only INSIDE a single-quoted shell word, never bare.
    assert "'mymodule:app$(touch /tmp/pwned-by-test)'" in run
    assert '"mymodule:app$(touch /tmp/pwned-by-test)"' not in run


def test_dit_kind_render_run_sh_rejects_none_app(tmp_path):
    # Defense-in-depth: a Manifest reaching render_run_sh with kind != "vllm" and app=None (e.g.
    # hand-edited JSON, or a Deps built directly rather than through stage_thin_package) must fail
    # loudly here instead of silently emitting the literal string "None" into run.sh.
    staged, _ = _stage_dit(tmp_path)
    m = Manifest.from_json((staged / "tt_kernel_manifest.json").read_text())
    m.deps.app = None
    with pytest.raises(ValueError, match="requires deps.app"):
        packaging.render_run_sh(m)


def test_dit_kind_rejects_unknown_kind(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    with pytest.raises(ValueError, match="is not supported"):
        packaging.stage_thin_package(
            tmp_path / "staged", name="x", arch="blackhole", model_py=model_py,
            tt_kernel_version="0.0.0", kind="tt-dit-servr", app="m:a", with_vllm=False,
        )


def test_dit_kind_rejects_app_missing_colon(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    with pytest.raises(ValueError, match="missing \":\""):
        packaging.stage_thin_package(
            tmp_path / "staged", name="x", arch="blackhole", model_py=model_py,
            tt_kernel_version="0.0.0", kind="tt-dit-server", app="not_a_valid_target",
            with_vllm=False,
        )


def test_dit_kind_requires_app(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    with pytest.raises(ValueError, match="requires app"):
        packaging.stage_thin_package(
            tmp_path / "staged", name="x", arch="blackhole", model_py=model_py,
            tt_kernel_version="0.0.0", kind="tt-dit-server", with_vllm=False,
        )


def test_dit_kind_rejects_vllm_metadata(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    with pytest.raises(ValueError, match="serves no vLLM"):
        packaging.stage_thin_package(
            tmp_path / "staged", name="x", arch="blackhole", model_py=model_py,
            tt_kernel_version="0.0.0", kind="tt-dit-server", app="m:a", with_vllm=False,
            vllm_metadata={"arch": "X", "main_class": "m:X"},
        )


def test_dit_kind_rejects_with_vllm_true(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    with pytest.raises(ValueError, match="with_vllm=False"):
        packaging.stage_thin_package(
            tmp_path / "staged", name="x", arch="blackhole", model_py=model_py,
            tt_kernel_version="0.0.0", kind="tt-dit-server", app="m:a",  # with_vllm defaults True
        )


def test_vllm_kind_rejects_app(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    with pytest.raises(ValueError, match="only used by non-vllm kinds"):
        packaging.stage_thin_package(
            tmp_path / "staged", name="x", arch="blackhole", model_py=model_py,
            tt_kernel_version="0.0.0", app="m:a",
            vllm_metadata={"arch": "X", "main_class": "m:X"},
        )


def test_vllm_kind_still_requires_vllm_metadata(tmp_path):
    # kind="vllm" is the default and the schema's only behavior before this change existed --
    # confirms that path is completely unaffected.
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    with pytest.raises(ValueError, match="requires vllm_metadata"):
        packaging.stage_thin_package(
            tmp_path / "staged", name="x", arch="blackhole", model_py=model_py,
            tt_kernel_version="0.0.0",
        )


def test_cli_package_thin_dit_kind_stage_only(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    out = tmp_path / "staged"
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(model_py), "--arch", "blackhole",
        "--kind", "tt-dit-server", "--app", "gradio_app.asgi:app",
        "--weights", "openvla/openvla-7b", "--mesh", "P300", "--device-count", "2",
        "--out", str(out),
    ])
    assert res.exit_code == 0, res.output
    m = Manifest.from_json((out / "tt_kernel_manifest.json").read_text())
    assert m.deps.kind == "tt-dit-server" and m.deps.app == "gradio_app.asgi:app"
    assert m.entrypoint is None
    run = (out / "run.sh").read_text()
    assert "vllm.entrypoints" not in run


def test_cli_package_thin_dit_kind_requires_app(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(model_py), "--arch", "blackhole",
        "--kind", "tt-dit-server", "--out", str(tmp_path / "staged"),
    ])
    assert res.exit_code != 0
    assert "needs --app" in res.output


def test_cli_package_thin_dit_kind_rejects_vllm_flags(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(model_py), "--arch", "blackhole",
        "--kind", "tt-dit-server", "--app", "gradio_app.asgi:app",
        "--arch-name", "Whatever", "--out", str(tmp_path / "staged"),
    ])
    assert res.exit_code != 0
    assert "serves no vLLM" in res.output


def test_cli_package_thin_rejects_unknown_kind(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(model_py), "--arch", "blackhole",
        "--kind", "tt-dit-servr", "--app", "gradio_app.asgi:app",  # typo
        "--out", str(tmp_path / "staged"),
    ])
    assert res.exit_code != 0
    assert "is not supported" in res.output


def test_cli_package_thin_rejects_app_missing_colon(tmp_path):
    model_py = tmp_path / "app.py"
    model_py.write_text("# runner\n")
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(model_py), "--arch", "blackhole",
        "--kind", "tt-dit-server", "--app", "not_a_valid_target",
        "--out", str(tmp_path / "staged"),
    ])
    assert res.exit_code != 0
    assert "missing ':'" in res.output


def test_cli_package_thin_vllm_kind_explicit_still_requires_entrypoint(tmp_path):
    model_py = tmp_path / "model.py"
    model_py.write_text("class C: pass\n")
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(model_py), "--arch", "blackhole",
        "--kind", "vllm", "--out", str(tmp_path / "staged"),
    ])
    assert res.exit_code != 0
    assert "Provide the serving entrypoint" in res.output
