# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for `tt-model curl` — the one-line "did it actually answer?" step.

Nothing here touches the network or a real server: `list_models` is monkeypatched to stand
in for a live/absent server, and the send path is intercepted so we can assert that what
runs is exactly what `--print` shows, and feed it the server's reply.
"""

import base64
import json
import shlex

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, container_cli, localdb, probe, runtime
from tt_kernel.container_manifest import ContainerManifest

runner = CliRunner()
MODEL = "unsloth/Llama-3.2-3B-Instruct"


def _body(stdout: str) -> dict:
    """The JSON payload out of a rendered curl command."""
    argv = shlex.split(stdout.replace("\\\n", " "))
    return json.loads(argv[argv.index("-d") + 1])


class _Completed:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def _chat(content=None, tool_calls=None) -> str:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return json.dumps({"choices": [{"message": message}]})


@pytest.fixture
def send(monkeypatch):
    """Answer the next request with ``reply``; returns the argv curl was run with."""
    seen = {}

    def answer(reply: str, returncode: int = 0, stderr: str = ""):
        def fake_run(argv, *a, **k):
            seen["argv"] = argv
            return _Completed(reply, returncode, stderr)

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/curl")
        return seen

    return answer


def _sent_body(seen: dict) -> dict:
    return json.loads(seen["argv"][seen["argv"].index("-d") + 1])


@pytest.fixture
def no_server(monkeypatch):
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: [])


@pytest.fixture
def live_server(monkeypatch):
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: [MODEL])


# ------------------------------------------------------------------ payload + rendering
def test_payload_defaults():
    payload = probe.build(probe.TEXT_GENERATION, MODEL, "hello").body
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
    body = probe.build(probe.TEXT_GENERATION, MODEL, "hello").body
    argv = runtime.curl_argv("http://localhost:8000", body)
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
        return _Completed(_chat("Hi!"))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/curl")
    res = runner.invoke(cli.app, ["curl", "hello", "--temperature", "0.2"])
    assert res.exit_code == 0
    assert seen["argv"] == shlex.split(printed.replace("\\\n", " "))
    assert res.stdout == "Hi!\n"


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




# ------------------------------------------------------------------ the task decides the request
@pytest.mark.parametrize("task,path,field", [
    ("text-generation", "/v1/chat/completions", "messages"),
    ("image-text-to-text", "/v1/chat/completions", "messages"),
    ("text-to-image", "/v1/images/generations", "prompt"),
    ("feature-extraction", "/v1/embeddings", "input"),
])
def test_task_picks_the_endpoint(live_server, task, path, field):
    res = runner.invoke(cli.app, ["curl", "--task", task, "--print"])
    assert res.exit_code == 0, res.stderr
    assert f"{runtime.DEFAULT_BASE_URL}{path}" in res.stdout
    assert field in _body(res.stdout)


def test_vision_request_carries_a_png(live_server):
    res = runner.invoke(cli.app, ["curl", "--task", "image-text-to-text", "--print"])
    content = _body(res.stdout)["messages"][0]["content"]
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]).startswith(b"\x89PNG")


def test_unknown_task_is_refused(live_server):
    res = runner.invoke(cli.app, ["curl", "--task", "summarization"])
    assert res.exit_code == 1
    assert "unknown --task" in res.stderr


def test_api_key_is_sent_as_a_bearer_token(live_server, monkeypatch):
    monkeypatch.setenv(runtime.ENV_API_KEY, "s3cret")
    res = runner.invoke(cli.app, ["curl", "hi", "--print"])
    argv = shlex.split(res.stdout.replace("\\\n", " "))
    assert argv[argv.index("-H") + 1] == "Authorization: Bearer s3cret"


# ------------------------------------------------------------------------ reading the reply
def test_tool_call_is_shown(live_server, send):
    calls = [{"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
    seen = send(_chat(tool_calls=calls))
    res = runner.invoke(cli.app, ["curl", "--tools"])
    assert res.exit_code == 0, res.stderr
    assert 'tool call: get_weather({"city": "Paris"})' in res.stdout
    assert _sent_body(seen)["tools"][0]["function"]["name"] == "get_weather"
    assert _sent_body(seen)["messages"][0]["content"] == probe.TOOLS_PROMPT


def test_a_tool_offered_but_not_called_is_a_warning(live_server, send):
    send(_chat("It is sunny."))
    res = runner.invoke(cli.app, ["curl", "--tools"])
    assert res.exit_code == 0
    assert "It is sunny." in res.stdout
    assert "without calling the tool" in res.stderr


@pytest.mark.parametrize("reply", [
    json.dumps({"images": [base64.b64encode(b"\xff\xd8jpeg").decode()]}),       # media server
    json.dumps({"data": [{"b64_json": base64.b64encode(b"\xff\xd8jpeg").decode()}]}),  # OpenAI
])
def test_generated_image_is_saved(live_server, send, tmp_path, reply):
    send(reply)
    out = tmp_path / "out.jpg"
    res = runner.invoke(cli.app, ["curl", "a fox", "--task", "text-to-image", "-o", str(out)])
    assert res.exit_code == 0, res.stderr
    assert out.read_bytes() == b"\xff\xd8jpeg"
    assert f"Saved image to {out}" in res.stdout


def test_generated_image_defaults_to_a_named_file_in_the_cwd(live_server, send, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    send(json.dumps({"images": [base64.b64encode(b"\x89PNGx").decode()]}))
    res = runner.invoke(cli.app, ["curl", "--task", "text-to-image"])
    assert res.exit_code == 0, res.stderr
    [saved] = tmp_path.iterdir()
    assert saved.name.startswith("Llama-3.2-3B-Instruct-") and saved.suffix == ".png"


def test_embedding_reports_its_size(live_server, send):
    send(json.dumps({"data": [{"embedding": [0.25] * 8}]}))
    res = runner.invoke(cli.app, ["curl", "--task", "feature-extraction"])
    assert res.exit_code == 0, res.stderr
    assert "8-dimensional embedding" in res.stdout


@pytest.mark.parametrize("reply", [
    json.dumps({"object": "error", "message": "tool_choice requires --enable-auto-tool-choice"}),
    json.dumps({"detail": "Invalid or missing API Key"}),
    "Internal Server Error",
])
def test_a_server_error_is_reported_in_its_own_words(live_server, send, reply):
    send(reply)
    res = runner.invoke(cli.app, ["curl", "hello"])
    assert res.exit_code == 1
    assert "request to" in res.stderr and "failed" in res.stderr


def test_a_transport_failure_keeps_curls_exit_code(live_server, send):
    send("", returncode=28, stderr="curl: (28) Operation timed out")
    res = runner.invoke(cli.app, ["curl", "hello"])
    assert res.exit_code == 28
    assert "timed out" in res.stderr


# ------------------------------------------------------------------ tasks curl cannot exercise
def test_health_only_task_checks_the_server_and_says_so(live_server, send):
    seen = send("unused")
    res = runner.invoke(cli.app, ["curl", "--task", "automatic-speech-recognition"])
    assert res.exit_code == 0
    assert f"{MODEL} is up at {runtime.DEFAULT_BASE_URL}" in res.stdout
    assert "cannot exercise automatic-speech-recognition models yet" in res.stderr
    assert "argv" not in seen


def test_health_only_task_print_shows_the_health_check(live_server):
    res = runner.invoke(cli.app, ["curl", "--task", "text-to-speech", "--print"])
    assert res.exit_code == 0
    assert res.stdout.startswith(f"curl -sS {runtime.DEFAULT_BASE_URL}/v1/models")


def test_health_only_task_with_nothing_serving_is_an_error(no_server):
    res = runner.invoke(cli.app, ["curl", "--task", "text-to-video", "--model", MODEL])
    assert res.exit_code == 1
    assert "Nothing is serving" in res.stderr


# ------------------------------------------------------------- capabilities from the pulled manifest
DIT_REPO = "you/my-diffusion-model"


def _pulled(monkeypatch, kind: str, repo_id: str = DIT_REPO, weights: str = "org/Weights", **over):
    """A pulled container package of the given kind, as `localdb` + the pulled manifest see it."""
    from test_container_manifest import BASE
    from test_tt_dit_server_kind import DIT_BASE

    raw = json.loads(json.dumps(DIT_BASE if kind == "tt-dit-server" else BASE))
    raw.update({"repo": repo_id, "name": repo_id.split("/")[1], "weights": weights, "kind": kind})
    raw.update(over)
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    wire = m.to_wire(image_tag="tt-model/x:abc", tt_metal_version="0.72.1", tt_kernel_version="0.1.0")
    monkeypatch.setattr(localdb, "all_entries", lambda: [{"repo_id": repo_id, "container": True}])
    monkeypatch.setattr(container_cli, "load_pulled", lambda rid: wire if rid == repo_id else None)


def test_a_dit_server_without_a_task_only_gets_a_health_check(monkeypatch, send):
    """The dit apps answer /v1/models like vLLM, so discovery 'works' and a chat body would
    404. Say what was checked, and where its own routes are documented."""
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: ["org/Weights"])
    _pulled(monkeypatch, "tt-dit-server")
    seen = send("unused")
    res = runner.invoke(cli.app, ["curl", "hello"])
    assert res.exit_code == 0
    assert "org/Weights is up" in res.stdout
    assert "tt-dit-server" in res.stderr and "--task" in res.stderr
    assert f"huggingface.co/{DIT_REPO}" in res.stderr
    assert "argv" not in seen


def test_a_dit_server_is_named_even_before_it_is_up(no_server, monkeypatch):
    _pulled(monkeypatch, "tt-dit-server")
    res = runner.invoke(cli.app, ["curl", "hello", "--print"])
    assert res.exit_code == 0
    assert "tt-dit-server" in res.stderr
    assert res.stdout.startswith(f"curl -sS {runtime.DEFAULT_BASE_URL}/v1/models")


def test_the_cards_pipeline_tag_picks_the_task(monkeypatch):
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: ["org/Weights"])
    _pulled(monkeypatch, "tt-dit-server", card={"pipeline_tag": "text-to-image"})
    res = runner.invoke(cli.app, ["curl", "a fox", "--print"])
    assert res.exit_code == 0, res.stderr
    assert "/v1/images/generations" in res.stdout
    assert f"from {DIT_REPO}'s manifest" in res.stderr


def test_task_flag_overrides_the_manifest(monkeypatch):
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: ["org/Weights"])
    _pulled(monkeypatch, "tt-dit-server")
    res = runner.invoke(cli.app, ["curl", "a fox", "--task", "text-to-image", "--print"])
    assert res.exit_code == 0
    assert "/v1/images/generations" in res.stdout


def test_a_vllm_package_with_a_tool_parser_is_offered_a_tool(monkeypatch):
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: ["org/Weights"])
    _pulled(monkeypatch, "vllm-plugin", repo_id="you/my-model",
            serve={"port": 8000, "block_size": 64, "capabilities": {"tool_parser": "hermes"}})
    res = runner.invoke(cli.app, ["curl", "--print"])
    assert res.exit_code == 0, res.stderr
    assert _body(res.stdout)["tools"] == [probe.WEATHER_TOOL]
    assert runner.invoke(cli.app, ["curl", "--no-tools", "--print"]).stdout.count("tools") == 0


def test_a_vllm_package_without_a_tool_parser_just_chats(monkeypatch):
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: ["org/Weights"])
    _pulled(monkeypatch, "vllm-plugin", repo_id="you/my-model")
    res = runner.invoke(cli.app, ["curl", "hello", "--print"])
    assert res.exit_code == 0
    assert _body(res.stdout)["model"] == "org/Weights"
    assert "tools" not in _body(res.stdout)


def test_a_self_contained_bundles_manifest_declares_tools(monkeypatch, tmp_path):
    from tt_kernel import MANIFEST_NAME
    from tt_kernel.manifest import Capabilities, Deps, Manifest, Producer, WeightsRef

    m = Manifest(schema_version="6", name="thin", tt_metal_version="v", arch="blackhole",
                 producer=Producer(tt_kernel_version="0", created_at="t"),
                 weights=WeightsRef(repo=MODEL), deps=Deps(),
                 capabilities=Capabilities(tool_parser="llama3_json"))
    (tmp_path / MANIFEST_NAME).write_text(m.to_json())
    monkeypatch.setattr(localdb, "all_entries", lambda: [
        {"repo_id": "you/thin", "self_contained": True, "install_dir": str(tmp_path), "weights": MODEL},
    ])
    monkeypatch.setattr(runtime, "list_models", lambda *a, **k: [MODEL])
    res = runner.invoke(cli.app, ["curl", "--print"])
    assert res.exit_code == 0, res.stderr
    assert "tools" in _body(res.stdout)
