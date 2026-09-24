# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Unit tests for the runtime helpers (models dir resolution + weights download).

All network side effects are monkeypatched — nothing is actually downloaded.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from tt_kernel import runtime
from tt_kernel.manifest import WeightsRef


# --------------------------------------------------------------------------- models dir

def test_resolve_models_dir_flag_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(runtime.ENV_MODELS_DIR, str(tmp_path / "env"))
    out = runtime.resolve_models_dir(str(tmp_path / "flag"), "org/model")
    assert out == tmp_path / "flag" / "org" / "model"


def test_resolve_models_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv(runtime.ENV_MODELS_DIR, str(tmp_path / "env"))
    out = runtime.resolve_models_dir(None, "org/model")
    assert out == tmp_path / "env" / "org" / "model"


def test_resolve_models_dir_default(monkeypatch):
    monkeypatch.delenv(runtime.ENV_MODELS_DIR, raising=False)
    monkeypatch.setenv("HOME", "/home/someone")
    out = runtime.resolve_models_dir(None, "org/model")
    assert out == Path("/home/someone/.cache/tt-model/models/org/model")


def test_resolve_models_dir_no_org():
    out = runtime.resolve_models_dir("/tmp/x", "just-a-name")
    assert out == Path("/tmp/x/just-a-name")


# --------------------------------------------------------------------------- weights

def test_download_weights_forwards_args(monkeypatch, tmp_path):
    seen = {}

    def fake_snapshot(**kwargs):
        seen.update(kwargs)
        return str(tmp_path / "dl")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    w = WeightsRef(repo_id="org/m", revision="abc", allow_patterns=["*.safetensors"])
    out = runtime.download_weights(w, tmp_path / "dest")
    assert seen["repo_id"] == "org/m"
    assert seen["revision"] == "abc"
    assert seen["allow_patterns"] == ["*.safetensors"]
    assert seen["repo_type"] == "model"
    assert out == tmp_path / "dl"


# ------------------------------------------------------------- local server behind a proxy
# A corporate proxy (http_proxy set, localhost not in no_proxy) must not swallow requests for
# the local server: urllib and curl both send "localhost" to the proxy by default.
SERVED = "Qwen/Qwen3.6-35B-A3B"


def _dead_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def behind_proxy(monkeypatch):
    proxy = f"http://127.0.0.1:{_dead_port()}"
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(var, proxy)
    for var in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def models_server():
    body = json.dumps({"object": "list", "data": [{"id": SERVED}]}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_list_models_reaches_local_server_behind_a_proxy(behind_proxy, models_server, host):
    assert runtime.list_models(f"http://{host}:{models_server}") == [SERVED]


@pytest.mark.parametrize("base", ["http://localhost:20000", "http://127.0.0.5:20000",
                                  "http://[::1]:20000"])
def test_curl_argv_skips_the_proxy_for_a_local_server(base):
    argv = runtime.curl_argv(base, {"model": SERVED})
    assert argv[argv.index("--noproxy") + 1] == "*"


def test_curl_argv_keeps_the_proxy_for_a_remote_server():
    assert "--noproxy" not in runtime.curl_argv("http://gpu-box.example:20000", {"model": SERVED})
