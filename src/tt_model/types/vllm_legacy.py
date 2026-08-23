# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Model type ``vllm-legacy``: the tenstorrent/vllm fork with its in-tree plugin.

- vLLM is the ``tenstorrent/vllm`` fork at ``runtime.vllm.{repo,ref}`` (dev branch
  lineage), built ``VLLM_TARGET_DEVICE=empty`` and installed *editable* — as is its
  in-tree ``plugins/vllm-tt-plugin``. Editable installs mean the fork checkout must
  survive into the runtime image (see ``runtime_copy_lines``); it is ~200 MB and it is
  why this type is called legacy.
- Launched via tt-metal's readiness runner
  (``python -m models.common.readiness_check.run_vllm_server --stages serve``), which
  resolves the plugin's config flag against the installed vLLM, forwards an explicit
  mesh grid, and hands off to a stock ``vllm.entrypoints.openai.api_server``. Models of
  this type are the ones whose model dirs expect that runner's flag names
  (``--tt-config``, ``--additional-server-args``, ``--server-timeout``).

The fork checkout lives at ``/opt/vllm`` in both stages.
"""

from __future__ import annotations

import json
import shlex
from typing import Dict, List

from ..manifest import Manifest, ManifestError, ServeProfile

READY_LINE = "Server ready at"

PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

FORK_DIR = "/opt/vllm"


class VllmLegacyType:
    name = "vllm-legacy"

    # ---- manifest ------------------------------------------------------------------

    def validate(self, m: Manifest) -> None:
        rt = m.runtime
        vllm = rt.get("vllm") or {}
        if not vllm.get("repo") or not vllm.get("ref"):
            raise ManifestError(
                "type vllm-legacy requires runtime.vllm.{repo, ref} (the tenstorrent/vllm "
                "fork; the plugin is in-tree) — for stock vLLM + the standalone plugin "
                "use type: vllm"
            )
        if rt.get("plugin"):
            raise ManifestError(
                "type vllm-legacy takes no runtime.plugin: the plugin comes from the "
                "fork's own plugins/vllm-tt-plugin"
            )
        if not rt.get("model_dir"):
            raise ManifestError(
                "type vllm-legacy requires runtime.model_dir (the model's directory in "
                "the tt-metal tree, e.g. models/autoports/<name>) — the readiness "
                "runner launches by --model-dir"
            )
        if rt["model_dir"] not in m.source.code and not any(
            rt["model_dir"].startswith(c.rstrip("/") + "/") or c == rt["model_dir"]
            for c in m.source.code
        ):
            raise ManifestError(
                f"runtime.model_dir {rt['model_dir']!r} is not covered by source.code — "
                "the launcher would not find the model inside the image"
            )
        for key in rt:
            if key not in ("vllm", "extension", "lock", "model_dir"):
                raise ManifestError(f"type vllm-legacy does not understand runtime.{key}")

    # ---- image build ----------------------------------------------------------------

    def install_lines(self, m: Manifest) -> List[str]:
        rt = m.runtime
        vllm = rt["vllm"]
        ref = vllm.get("sha") or vllm["ref"]
        lines: List[str] = [
            f"git clone {shlex.quote(vllm['repo'])} {FORK_DIR}"
            f" && git -C {FORK_DIR} checkout {shlex.quote(ref)}",
        ]
        if rt.get("lock"):
            lines.append(
                'uv pip install --python "$VENV/bin/python" -r /ctx/requirements.lock '
                f"--extra-index-url {PYTORCH_CPU_INDEX} --index-strategy unsafe-best-match"
            )
            deps = "--no-deps "
        else:
            deps = ""
        lines += [
            # Editable, matching the fork's own documented install; the checkout is
            # COPY'd into the runtime stage so the .pth files stay valid.
            f'VLLM_TARGET_DEVICE=empty uv pip install --python "$VENV/bin/python" {deps}-e {FORK_DIR} '
            f"--extra-index-url {PYTORCH_CPU_INDEX} --index-strategy unsafe-best-match",
            f'uv pip install --python "$VENV/bin/python" -e {FORK_DIR}/plugins/vllm-tt-plugin',
            # The editable installs bake build-stage paths; drop VCS metadata only.
            f"rm -rf {FORK_DIR}/.git",
        ]
        if rt.get("extension"):
            lines.append(
                f'uv pip install --python "$VENV/bin/python" /opt/tt-metal/{rt["extension"]}'
            )
        return lines

    def verify_lines(self, m: Manifest) -> List[str]:
        checks = [
            "import ttnn, vllm, vllm_tt_plugin",
            "import torch; assert torch.__version__.endswith('+cpu'), torch.__version__",
            f"import vllm; assert vllm.__file__.startswith('{FORK_DIR}'), vllm.__file__",
            "import models.common.readiness_check.run_vllm_server",
        ]
        lines = [f'"$VENV/bin/python" -c {shlex.quote("; ".join(checks))}']
        # model-authored assertions from the manifest's verify: list
        lines += [f'"$VENV/bin/python" -c {shlex.quote(v)}' for v in m.verify]
        return lines

    def runtime_copy_lines(self, m: Manifest) -> List[str]:
        # The fork is installed editable: its checkout must exist in the final image or
        # the venv's .pth files dangle.
        return [f"COPY --from=builder {FORK_DIR} {FORK_DIR}"]

    # ---- serve -----------------------------------------------------------------------

    def serve_argv(self, m: Manifest, profile: ServeProfile) -> List[str]:
        from ..manifest import parse_mesh_device  # noqa: PLC0415

        rows, cols = parse_mesh_device(profile.mesh_device)
        argv = [
            "python", "-m", "models.common.readiness_check.run_vllm_server",
            "--stages", "serve",
            "--model-dir", m.runtime["model_dir"],
            "--hf-model", m.weights,
            "--mesh-device", f"({rows}, {cols})",
            "--max-num-seqs", str(profile.max_num_seqs),
        ]
        if profile.max_model_len is not None:
            argv += ["--max-model-len", str(profile.max_model_len)]
        argv += ["--block-size", str(profile.block_size)]
        if profile.server_timeout is not None:
            argv += ["--server-timeout", str(profile.server_timeout)]
        argv += ["--port", str(profile.port or 8000)]
        tt_cfg = profile.additional_config.get("tt")
        if tt_cfg:
            argv += ["--tt-config", json.dumps(tt_cfg)]
        extra = profile.flat_args()
        if extra:
            argv += ["--additional-server-args", " ".join(extra)]
        return argv

    def serve_env(self, m: Manifest, profile: ServeProfile) -> Dict[str, str]:
        # mesh goes through the launcher's --mesh-device flag, not MESH_DEVICE
        env = {"HF_MODEL": m.weights}
        env.update(profile.env)
        return env

    def ready_probe(self, m: Manifest) -> str:
        return READY_LINE
