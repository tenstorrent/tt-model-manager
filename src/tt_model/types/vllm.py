# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Model type ``vllm``: stock vLLM + the standalone Tenstorrent platform plugin.

- vLLM is ``vllm==<runtime.vllm.version>`` built from sdist with
  ``VLLM_TARGET_DEVICE=empty`` against CPU torch (there is no CUDA on a TT box; the
  published wheel is the CUDA build, so it must come from source).
- The TT platform comes from the public ``tenstorrent/vllm-tt-plugin`` repo at
  ``runtime.plugin.{repo,ref}``, installed NON-editable so the clone need not survive
  into the runtime image.
- Launched with plain ``vllm serve <weights>``.

This deliberately reimplements the plugin's ``docs/install-vllm-tt.sh`` rather than
sourcing it: that script is written to be *sourced* (it fails with ``return``), uses
relative paths, ``curl``s vLLM's requirements list live at install time (so two builds a
week apart get different envs), and ends with an editable install whose ``.pth`` would
dangle. With a ``runtime.lock`` present, nothing here resolves at build time.
"""

from __future__ import annotations

import json
import shlex
from typing import Dict, List

from ..manifest import Manifest, ManifestError, ServeProfile

READY_LINE = "Application startup complete"

# vLLM's PyPI metadata is generated on a CUDA machine; without the CPU index a plain
# install resolves the CUDA dependency set (~4 GB of nvidia-* wheels, no device here).
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# ttnn pins numpy<2; recent vLLM asks for opencv-python-headless>=4.13, which requires
# numpy>=2. A hard conflict, not a preference — pip cannot express the resolution, uv's
# --override can. numpy<2 wins (fixed by ttnn); opencv 4.11 is the last release without
# a numpy-2 floor, and vLLM only reaches opencv through its lazy video-IO path, which
# no TT model uses. Mirrors the validated laguna overrides.txt.
DEFAULT_OVERRIDES = ["numpy>=1.24.4,<2", "opencv-python-headless==4.11.0.86"]


class VllmType:
    name = "vllm"

    # ---- manifest ------------------------------------------------------------------

    def validate(self, m: Manifest) -> None:
        rt = m.runtime
        version = (rt.get("vllm") or {}).get("version")
        if not version:
            raise ManifestError(
                "type vllm requires runtime.vllm.version (a released vLLM, e.g. \"0.24.0\") "
                "— for the tenstorrent/vllm fork use type: vllm-legacy"
            )
        plugin = rt.get("plugin") or {}
        if not plugin.get("repo") or not plugin.get("ref"):
            raise ManifestError("type vllm requires runtime.plugin.{repo, ref} (the standalone vllm-tt-plugin)")
        for key in rt:
            if key not in ("vllm", "plugin", "extension", "lock", "overrides"):
                raise ManifestError(f"type vllm does not understand runtime.{key}")

    # ---- image build ----------------------------------------------------------------

    def install_lines(self, m: Manifest) -> List[str]:
        rt = m.runtime
        version = rt["vllm"]["version"]
        plugin = rt["plugin"]
        plugin_ref = plugin.get("sha") or plugin["ref"]
        lines: List[str] = []

        if rt.get("lock"):
            # The lock IS the dependency set: vLLM's own requirements are already in it,
            # so vLLM itself installs --no-deps and nothing resolves at build time.
            lines += [
                'uv pip install --python "$VENV/bin/python" -r /ctx/requirements.lock '
                f"--extra-index-url {PYTORCH_CPU_INDEX} --index-strategy unsafe-best-match",
                f'VLLM_TARGET_DEVICE=empty uv pip install --python "$VENV/bin/python" '
                f"--no-deps --no-binary vllm vllm=={version}",
            ]
        else:
            # First build of a model: resolve live (vLLM's own metadata) under the
            # numpy/opencv override, then `package` freezes the result back out as
            # requirements.lock so every later build is reproducible.
            overrides = DEFAULT_OVERRIDES + list(rt.get("overrides") or [])
            quoted = " ".join(shlex.quote(o) for o in overrides)
            lines += [
                f"printf '%s\n' {quoted} > /tmp/tt-overrides.txt",
                f'VLLM_TARGET_DEVICE=empty uv pip install --python "$VENV/bin/python" '
                f"--no-binary vllm vllm=={version} --override /tmp/tt-overrides.txt "
                f"--extra-index-url {PYTORCH_CPU_INDEX} --index-strategy unsafe-best-match",
            ]
        # transformers imports torchaudio if it is merely installed, and the wheel that
        # rides along with CPU torch is unloadable — the validated recipe removes it.
        lines += ['uv pip uninstall --python "$VENV/bin/python" torchaudio || true']

        lines += [
            # Standalone plugin, NON-editable: the clone does not survive into the image.
            f"git clone {shlex.quote(plugin['repo'])} /tmp/vllm-tt-plugin"
            f" && git -C /tmp/vllm-tt-plugin checkout {shlex.quote(plugin_ref)}"
            f' && uv pip install --python "$VENV/bin/python" /tmp/vllm-tt-plugin'
            f" && rm -rf /tmp/vllm-tt-plugin",
        ]
        if rt.get("extension"):
            # The model's own vLLM extension (general-plugins entry point + extra_models).
            # It ships inside code/, so install it from the staged tree in the image.
            lines.append(
                f'uv pip install --python "$VENV/bin/python" /opt/tt-metal/{rt["extension"]}'
            )
        return lines

    def verify_lines(self, m: Manifest) -> List[str]:
        rt = m.runtime
        checks = [
            "import ttnn, vllm, vllm_tt_plugin",
            "import torch; assert torch.__version__.endswith('+cpu'), torch.__version__",
            "import vllm; assert '/tt-metal/' not in vllm.__file__, vllm.__file__",
        ]
        if rt.get("extension"):
            checks += [
                "import os; md = os.environ['EXTRA_MODELS_DIR']; "
                "entries = [e for e in os.listdir(md) "
                "if os.path.exists(os.path.join(md, e, 'vllm_metadata.json'))]; "
                "assert entries, f'EXTRA_MODELS_DIR {md} registers no models'",
            ]
        lines = [f'"$VENV/bin/python" -c {shlex.quote("; ".join(checks))}']
        # model-authored assertions from the manifest's verify: list
        lines += [f'"$VENV/bin/python" -c {shlex.quote(v)}' for v in m.verify]
        return lines

    def runtime_copy_lines(self, m: Manifest) -> List[str]:
        return []

    # ---- serve -----------------------------------------------------------------------

    def serve_argv(self, m: Manifest, profile: ServeProfile) -> List[str]:
        argv = ["vllm", "serve", m.weights]
        if profile.max_model_len is not None:
            argv += ["--max-model-len", str(profile.max_model_len)]
        argv += ["--max-num-seqs", str(profile.max_num_seqs)]
        argv += ["--block-size", str(profile.block_size)]
        if profile.additional_config:
            argv += ["--additional-config", json.dumps(profile.additional_config)]
        argv += profile.flat_args()
        argv += ["--port", str(profile.port or 8000)]
        return argv

    def serve_env(self, m: Manifest, profile: ServeProfile) -> Dict[str, str]:
        env = {
            "MESH_DEVICE": profile.mesh_device,
            # The tt_transformers-style adapters read the model id from env, not from
            # vLLM's --model.
            "HF_MODEL": m.weights,
        }
        env.update(profile.env)
        return env

    def ready_probe(self, m: Manifest) -> str:
        return READY_LINE
