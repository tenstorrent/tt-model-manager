# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for `tt-model curl` — the one-line "did it actually answer?" step.

Nothing here touches the network or a real server: `list_models` is monkeypatched to stand
in for a live/absent vLLM, and the send path is intercepted so we can assert that what
runs is exactly what `--print` shows.
"""

import json
import shlex

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, container_cli, localdb, runtime
from tt_kernel.container_manifest import ContainerManifest

runner = CliRunner()
MODEL = "unsloth/Llama-3.2-3B-Instruct"


def _body(stdout: str) -> dict:
    """The JSON payload out of a rendered curl command."""
    argv = shlex.split(stdout.replace("\\\n", " "))
    return json.loads(argv[argv.index("-d") + 1])


@pytest.fixture
def no_server(monkeypatch):
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: [])


@pytest.fixture
def live_server(monkeypatch):
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: [MODEL])


# ------------------------------------------------------------------ payload + rendering
def test_payload_defaults():
    payload = runtime.chat_payload(MODEL, "hello")
    assert payload["model"] == MODEL
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["max_tokens"] == runtime.DEFAULT_MAX_TOKENS


@pytest.mark.parametrize("argv,expected", [
    (["--temperature", "0.7"], {"temperature": 0.7}),          # float, not "0.7"
    (["--max-tokens", "200"], {"max_tokens": 200}),            # dashes -> underscores
    (["--stop", '["\\n"]'], {"stop": ["\n"]}),                # JSON structures survive
    (["--echo"], {"echo": True}),                              # bare flag
    (["--model-impl=vllm"], {"model_impl": "vllm"}),           # --key=value
    (["--guided-choice", "yes"], {"guided_choice": "yes"}),    # unparseable stays a string
])
def test_extra_params_are_typed(argv, expected):
    assert runtime.parse_extra_params(argv) == expected


def test_extra_params_reject_a_bare_positional():
    # `tt-model curl hello there` is a quoting mistake; dropping "there" silently would
    # send a different prompt than the user typed.
    with pytest.raises(ValueError):
        runtime.parse_extra_params(["there"])


def test_render_curl_is_the_argv_a_shell_would_run():
    argv = runtime.curl_argv("http://localhost:8000", runtime.chat_payload(MODEL, "hello"))
    rendered = runtime.render_curl(argv)
    assert rendered.endswith("'")                       # the JSON body is quoted as one word
    assert shlex.split(rendered.replace("\\\n", " ")) == argv


# ------------------------------------------------------------------------ model discovery
def test_running_server_wins(live_server):
    res = runner.invoke(cli.app, ["curl", "hello", "--print"])
    assert res.exit_code == 0
    assert _body(res.stdout)["model"] == MODEL


def test_falls_back_to_the_install_record_with_no_server(no_server, monkeypatch):
    # Printing has to work before anything is serving — that's the doc/copy-paste case.
    monkeypatch.setattr(localdb, "all_entries", lambda: [{"repo_id": "a/b", "weights": MODEL}])
    res = runner.invoke(cli.app, ["curl", "hello", "--print"])
    assert res.exit_code == 0
    assert _body(res.stdout)["model"] == MODEL


def test_explicit_model_overrides_discovery(live_server):
    res = runner.invoke(cli.app, ["curl", "hello", "--model", "other/model", "--print"])
    assert _body(res.stdout)["model"] == "other/model"


def test_ambiguous_install_asks_for_model(no_server, monkeypatch):
    monkeypatch.setattr(localdb, "all_entries", lambda: [
        {"repo_id": "a/b", "weights": MODEL}, {"repo_id": "c/d", "weights": "other/model"},
    ])
    res = runner.invoke(cli.app, ["curl", "hello"])
    assert res.exit_code == 1
    assert "--model" in res.stderr


# ------------------------------------------------------------------------------ the command
def test_sampling_params_reach_the_body(live_server):
    res = runner.invoke(cli.app, ["curl", "hi", "--temperature", "0.7", "--max-tokens", "200", "--print"])
    body = _body(res.stdout)
    assert body["temperature"] == 0.7 and body["max_tokens"] == 200
    assert body["messages"][0]["content"] == "hi"


def test_stdout_stays_pipeable(live_server):
    # `tt-model curl --print | bash` must work, so the "model id from ..." note goes to stderr.
    res = runner.invoke(cli.app, ["curl", "hello", "--print"])
    assert res.stdout.startswith("curl -sS ")


def test_sending_runs_exactly_what_print_shows(live_server, monkeypatch):
    printed = runner.invoke(cli.app, ["curl", "hello", "--temperature", "0.2", "--print"]).stdout
    seen = {}

    def fake_run(argv, *a, **k):
        seen["argv"] = argv
        class R:
            returncode = 0
        return R()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/curl")
    res = runner.invoke(cli.app, ["curl", "hello", "--temperature", "0.2"])
    assert res.exit_code == 0
    assert seen["argv"] == shlex.split(printed.replace("\\\n", " "))


def test_sending_without_curl_installed_is_a_clean_error(live_server, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    res = runner.invoke(cli.app, ["curl", "hello"])
    assert res.exit_code == 1
    assert "curl is not on PATH" in res.stderr


def test_sending_with_nothing_serving_says_so(no_server, monkeypatch):
    # A down server must be reported as a down server, not as a bare curl exit code.
    monkeypatch.setattr(localdb, "all_entries", lambda: [{"repo_id": "a/b", "weights": MODEL}])
    res = runner.invoke(cli.app, ["curl", "hello"])
    assert res.exit_code == 1
    assert "Nothing is serving" in res.stderr


def test_print_still_works_with_nothing_serving(no_server, monkeypatch):
    monkeypatch.setattr(localdb, "all_entries", lambda: [{"repo_id": "a/b", "weights": MODEL}])
    res = runner.invoke(cli.app, ["curl", "hello", "--print"])
    assert res.exit_code == 0 and _body(res.stdout)["model"] == MODEL


# ------------------------------------------------------------------ non-chat (tt-dit-server) packages
DIT_REPO = "you/my-diffusion-model"


def _pulled(monkeypatch, kind: str, repo_id: str = DIT_REPO, weights: str = "org/Weights"):
    """A pulled container package of the given kind, as `localdb` + the pulled manifest see it."""
    from test_container_manifest import BASE
    from test_tt_dit_server_kind import DIT_BASE

    raw = json.loads(json.dumps(DIT_BASE if kind == "tt-dit-server" else BASE))
    raw.update({"repo": repo_id, "name": repo_id.split("/")[1], "weights": weights, "kind": kind})
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    wire = m.to_wire(image_tag="tt-model/x:abc", tt_metal_version="0.72.1", tt_kernel_version="0.1.0")
    monkeypatch.setattr(localdb, "all_entries", lambda: [{"repo_id": repo_id, "container": True}])
    monkeypatch.setattr(container_cli, "load_pulled", lambda rid: wire if rid == repo_id else None)


def test_curl_refuses_to_post_a_chat_body_at_a_dit_server(monkeypatch):
    """The dit apps answer /v1/models like vLLM, so discovery 'works' and the chat body
    404s. Name the kind and the endpoint that does exist instead."""
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: ["org/Weights"])
    _pulled(monkeypatch, "tt-dit-server")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("posted a chat body"))
    res = runner.invoke(cli.app, ["curl", "hello"])
    assert res.exit_code == 1
    assert "serves tt-dit-server" in res.stderr
    assert f"curl {runtime.DEFAULT_BASE_URL}/v1/health" in res.stderr
    assert f"huggingface.co/{DIT_REPO}" in res.stderr


def test_curl_names_the_dit_server_even_before_it_is_up(no_server, monkeypatch):
    # Nothing listening and the only pulled package is a dit server: the old message was
    # "no bundle is installed", which is false and points nowhere.
    _pulled(monkeypatch, "tt-dit-server")
    res = runner.invoke(cli.app, ["curl", "hello", "--print"])
    assert res.exit_code == 1
    assert "serves tt-dit-server" in res.stderr and "/v1/health" in res.stderr


def test_curl_still_posts_chat_to_a_vllm_container_package(monkeypatch):
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: ["org/Weights"])
    _pulled(monkeypatch, "vllm-plugin")
    res = runner.invoke(cli.app, ["curl", "hello", "--print"])
    assert res.exit_code == 0
    assert _body(res.stdout)["model"] == "org/Weights"


def test_explicit_model_is_the_escape_hatch_for_a_dit_server(monkeypatch):
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: ["org/Weights"])
    _pulled(monkeypatch, "tt-dit-server")
    res = runner.invoke(cli.app, ["curl", "hello", "--model", "org/Weights", "--print"])
    assert res.exit_code == 0
    assert _body(res.stdout)["model"] == "org/Weights"
