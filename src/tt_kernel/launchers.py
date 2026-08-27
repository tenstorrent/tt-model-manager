# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""How a container package is launched — one class per ``kind``.

A *kind* is the serving stack inside the image and the command that starts it. It is the
only stack-specific knowledge in the container path: the Dockerfile is kind-agnostic and
everything that varies arrives through the hooks below, so adding a kind is a new class
here plus a registration in :data:`KINDS` — no change to the manifest schema, the image
build, or any command.

Two kinds, named for what they ARE rather than for their age:

``vllm-plugin``
    Stock ``vllm==X.Y.Z`` from PyPI (built from sdist with ``VLLM_TARGET_DEVICE=empty``)
    plus the standalone ``tenstorrent/vllm-tt-plugin``. Upstream vLLM grew a *platform
    plugin* API so an out-of-tree hardware backend no longer needs a fork; this is that
    arrangement, and it is where Tenstorrent is heading. Launched with ``vllm serve``.

``vllm-fork``
    The ``tenstorrent/vllm`` fork with the plugin in-tree at ``plugins/vllm-tt-plugin``,
    both installed *editable* — so the ~200 MB checkout has to survive into the runtime
    image. Launched through tt-metal's readiness runner. This is the older arrangement,
    and it is what this repo's own ``install``/``provision`` still set up.

Neither is called plain ``vllm``: a v4 manifest's ``runtime.kind = "vllm"`` already means
the fork, so reusing the bare word here would give one field two meanings depending on
which schema you were reading.

Two different inputs on purpose:

* ``validate`` runs at AUTHORING time against the YAML the author wrote, because that is
  where a bad ``runtime:`` block should be caught — before a multi-hour build, not after.
* ``serve_argv`` / ``serve_env`` / ``ready_probe`` run at CONSUME time against the
  published manifest, which is all a consumer has.

Argv composition is pure and total: no environment, no filesystem, no clock. That is what
lets ``serve --print`` be the test surface for every flag we claim to pass.

``install_lines`` and ``verify_lines`` are the BUILD side: they render the two generated
scripts (``install_engine.sh``, ``verify.sh``) that the Dockerfile runs, which is why the
Dockerfile itself never mentions vLLM and never changes per kind.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from .manifest import Manifest, ServeProfile

if TYPE_CHECKING:
    from .container_manifest import ContainerManifest


#: Where `package` stages a local plugin checkout inside the build context, and where
#: the Dockerfile puts it in the builder stage.
PLUGIN_CTX_DIR = "/ctx/plugin-src"
#: A local vLLM source tree, staged the same way.
VLLM_CTX_DIR = "/ctx/vllm-src"
#: The author's own vLLM wheel (v5 shipped one via --vllm-wheel).
VLLM_CTX_WHEEL = "/ctx/wheels/vllm-*.whl"
#: Extra local wheels to install alongside the engine (v5: --extra-wheel).
WHEELS_CTX_DIR = "/ctx/wheels"


class LauncherError(ValueError):
    """A launch configuration that must not proceed. The message is user-facing."""


def _weights_id(m: Manifest) -> str:
    if m.weights is None:
        raise LauncherError("a container package must reference weights by HF repo id")
    return m.weights.repo_id


class VllmPluginLauncher:
    """``kind: vllm-plugin`` — stock vLLM plus the standalone Tenstorrent platform plugin.

    The published vLLM wheel is the CUDA build, so vLLM is always built from sdist with
    ``VLLM_TARGET_DEVICE=empty``; the TT platform then arrives at runtime through the
    plugin, which activates only when ``ttnn`` is importable. The plugin installs
    NON-editable, so its clone does not have to survive into the runtime image.

    This deliberately reimplements the plugin's ``docs/install-vllm-tt.sh`` rather than
    sourcing it: that script is written to be *sourced* (it exits with ``return``), uses
    relative paths, and ``curl``s vLLM's requirements list live at install time — so two
    builds a week apart get different environments. With a ``runtime.lock`` present,
    nothing here resolves at build time at all.
    """

    name = "vllm-plugin"

    #: keys the manifest's ``runtime:`` block may contain for this kind
    RUNTIME_KEYS = ("vllm", "plugin", "extension", "extra_models_dir", "lock",
                    "overrides", "wheels")

    #: the log line whose appearance means the OpenAI server is accepting requests
    READY_LINE = "Application startup complete"

    #: vLLM's PyPI metadata is generated on a CUDA machine; without the CPU index a plain
    #: install resolves the CUDA dependency set (~4 GB of nvidia-* wheels, no device here)
    PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

    # ttnn pins numpy<2, while recent vLLM's opencv-python-headless wants numpy>=2. A hard
    # conflict, not a preference: pip cannot express the resolution, uv's --override can.
    # numpy<2 wins (fixed by ttnn); opencv 4.11 is the last release without a numpy-2
    # floor, and vLLM only reaches opencv through a lazy video-IO path no TT model uses.
    DEFAULT_OVERRIDES = ("numpy>=1.24.4,<2", "opencv-python-headless==4.11.0.86")

    def validate(self, m: "ContainerManifest") -> None:
        from .container_manifest import ContainerManifestError

        rt = m.runtime
        vllm = rt.get("vllm") or {}
        if vllm.get("repo"):
            raise ContainerManifestError(
                "runtime.vllm.repo describes the tenstorrent/vllm fork — that is kind "
                "vllm-fork, not vllm-plugin. Either set kind: vllm-fork, or give "
                "runtime.vllm.version / .wheel / .path."
            )
        vllm_sources = [k for k in ("version", "wheel", "path") if vllm.get(k)]
        if not vllm_sources:
            raise ContainerManifestError(
                "kind vllm-plugin requires runtime.vllm as one of:\n"
                '  {version: "0.24.0"}   a released vLLM, built from sdist in the image\n'
                "  {wheel: /path/*.whl}  the empty-target wheel YOU built — fastest, and\n"
                "                        exactly the binary you validated against\n"
                "  {path: /path/to/vllm} a local vLLM source tree, built in the image\n"
                "The plugin monkeypatches vLLM internals, so whichever you choose is "
                "load-bearing, not cosmetic."
            )
        if len(vllm_sources) > 1:
            raise ContainerManifestError(
                "runtime.vllm: give exactly one of version / wheel / path, got "
                + " and ".join(vllm_sources)
            )
        plugin = rt.get("plugin") or {}
        sources = {
            "path": bool(plugin.get("path")),
            "{repo, ref}": bool(plugin.get("repo") and plugin.get("ref")),
            "version": bool(plugin.get("version")),
        }
        chosen = [k for k, v in sources.items() if v]
        if not chosen:
            raise ContainerManifestError(
                "kind vllm-plugin requires runtime.plugin. The normal choice is your own\n"
                "checkout, packaged as-is — uncommitted work included, nothing fetched:\n"
                "  plugin: {path: /path/to/vllm-tt-plugin}\n"
                "Alternatives:\n"
                "  {repo, ref}                      — cloned during the build; the ref must "
                "be PUSHED, or the build cannot fetch it\n"
                '  {version: "X.Y.Z"}               — a PyPI release that already registers '
                "this model"
            )
        if len(chosen) > 1:
            raise ContainerManifestError(
                f"runtime.plugin: give exactly one of path / {{repo, ref}} / version, "
                f"got {' and '.join(chosen)}"
            )
        emd = rt.get("extra_models_dir")
        if emd and not any(
            emd == c or emd.startswith(c.rstrip("/") + "/") for c in m.source.code
        ):
            raise ContainerManifestError(
                f"runtime.extra_models_dir {emd!r} is not covered by source.code — the "
                "plugin would scan a directory that never entered the image"
            )
        for key in rt:
            if key not in self.RUNTIME_KEYS:
                raise ContainerManifestError(
                    f"kind vllm-plugin does not understand runtime.{key}; expected one of "
                    + ", ".join(self.RUNTIME_KEYS)
                )

    # ---- image build ---------------------------------------------------------------

    def install_lines(self, m: "ContainerManifest") -> List[str]:
        rt = m.runtime
        vllm = rt["vllm"]
        plugin = rt.get("plugin") or {}
        lines: List[str] = []

        # A wheel or a local tree the author staged: install it directly. A wheel is what
        # v5 shipped (`--vllm-wheel`) and is both the fastest route and the most faithful
        # — it is the binary the author actually ran, not a rebuild that may resolve
        # differently. Neither needs the sdist build or the override file below.
        if vllm.get("wheel"):
            return (
                [f'uv pip install --python "$VENV/bin/python" {VLLM_CTX_WHEEL} '
                 f"--extra-index-url {self.PYTORCH_CPU_INDEX} "
                 f"--index-strategy unsafe-best-match"]
                + self._post_engine_lines(m, plugin)
            )
        if vllm.get("path"):
            return (
                [f'VLLM_TARGET_DEVICE=empty uv pip install --python "$VENV/bin/python" '
                 f"{VLLM_CTX_DIR} --extra-index-url {self.PYTORCH_CPU_INDEX} "
                 f"--index-strategy unsafe-best-match"]
                + self._post_engine_lines(m, plugin)
            )

        version = vllm["version"]
        if rt.get("lock"):
            # The lock IS the dependency set: vLLM's own requirements are already in it,
            # so vLLM installs --no-deps and nothing resolves at build time.
            lines += [
                'uv pip install --python "$VENV/bin/python" -r /ctx/requirements.lock '
                f"--extra-index-url {self.PYTORCH_CPU_INDEX} --index-strategy unsafe-best-match",
                'VLLM_TARGET_DEVICE=empty uv pip install --python "$VENV/bin/python" '
                f"--no-deps --no-binary vllm vllm=={version}",
            ]
        else:
            # First build of a model: resolve live under the numpy/opencv override, then
            # `package` freezes the result out as requirements.lock for every later build.
            overrides = list(self.DEFAULT_OVERRIDES) + list(rt.get("overrides") or [])
            quoted = " ".join(shlex.quote(o) for o in overrides)
            lines += [
                f"printf '%s\n' {quoted} > /tmp/tt-overrides.txt",
                'VLLM_TARGET_DEVICE=empty uv pip install --python "$VENV/bin/python" '
                f"--no-binary vllm vllm=={version} --override /tmp/tt-overrides.txt "
                f"--extra-index-url {self.PYTORCH_CPU_INDEX} --index-strategy unsafe-best-match",
            ]
        return lines + self._post_engine_lines(m, plugin)

    def _post_engine_lines(self, m: "ContainerManifest", plugin: Dict) -> List[str]:
        """Everything after vLLM itself is installed, whichever route it came by."""
        rt = m.runtime
        lines: List[str] = []
        # transformers imports torchaudio if it is merely INSTALLED, and the wheel that
        # rides along with CPU torch is unloadable — the validated recipe removes it.
        lines.append('uv pip uninstall --python "$VENV/bin/python" torchaudio || true')

        if plugin.get("version"):
            lines.append(
                f'uv pip install --python "$VENV/bin/python" '
                f"vllm-tt-plugin=={plugin['version']}"
            )
        elif plugin.get("path"):
            # The author's own checkout, staged into the build context by `package` —
            # the same hermetic treatment source.tt_metal gets, so plugin changes that
            # are not committed (or not pushed) still ship. Non-editable, so nothing
            # from /ctx has to survive into the runtime image.
            lines.append(
                f'uv pip install --python "$VENV/bin/python" {PLUGIN_CTX_DIR}'
            )
        else:
            ref = plugin.get("sha") or plugin["ref"]
            lines.append(
                f"git clone {shlex.quote(plugin['repo'])} /tmp/vllm-tt-plugin"
                f" && git -C /tmp/vllm-tt-plugin checkout {shlex.quote(ref)}"
                f' && uv pip install --python "$VENV/bin/python" /tmp/vllm-tt-plugin'
                f" && rm -rf /tmp/vllm-tt-plugin"
            )
        if rt.get("extension"):
            lines.append(
                f'uv pip install --python "$VENV/bin/python" /opt/tt-metal/{rt["extension"]}'
            )
        if rt.get("wheels"):
            # Extra local wheels the author needs alongside the engine — v5's
            # `--extra-wheel`. Staged into the context by `package`; installed last so
            # they can override anything resolved above.
            lines.append(
                f'uv pip install --python "$VENV/bin/python" {WHEELS_CTX_DIR}/*.whl'
            )
        return lines

    def verify_lines(self, m: "ContainerManifest") -> List[str]:
        checks = [
            "import ttnn, vllm, vllm_tt_plugin",
            "import torch; assert torch.__version__.endswith('+cpu'), torch.__version__",
            # the plugin is installed non-editable, so vLLM must NOT resolve into the tree
            "import vllm; assert '/tt-metal/' not in vllm.__file__, vllm.__file__",
        ]
        pin = metal_torch_pin(_local_metal_tree(m))
        if pin:
            checks.append(
                f"import torch; v = torch.__version__.split('+')[0]; "
                f"assert v == {pin!r}, "
                f"f'torch {{v}} was resolved by vLLM but tt-metal pins {pin}; "
                f"ttnn extension modules were built against {pin} — pin it via "
                f"runtime.lock'"
            )
        if m.runtime.get("extra_models_dir") or m.runtime.get("extension"):
            # Do what the plugin does, and what vLLM does after it: find each bundle,
            # append its folder to sys.path, and RESOLVE the main_class string.
            #
            # Emitted as its OWN line: `checks` are joined with "; " into one `python -c`,
            # which a multi-statement snippet with a for-loop cannot survive.
            extra_check = RESOLVE_EXTRA_MODELS
        else:
            extra_check = None

        lines = [f'"$VENV/bin/python" -c {shlex.quote("; ".join(checks))}']
        if extra_check:
            lines.append(f'"$VENV/bin/python" -c {shlex.quote(extra_check)}')
        lines += [f'"$VENV/bin/python" -c {shlex.quote(v)}' for v in m.verify]
        return lines

    # ---- serve -----------------------------------------------------------------------

    def serve_argv(self, m: Manifest, profile: ServeProfile) -> List[str]:
        argv = ["vllm", "serve", _weights_id(m)]
        if profile.max_model_len is not None:
            argv += ["--max-model-len", str(profile.max_model_len)]
        argv += ["--max-num-seqs", str(profile.max_num_seqs)]
        argv += ["--block-size", str(profile.block_size)]
        if profile.additional_config:
            argv += ["--additional-config", json.dumps(profile.additional_config)]
        argv += _capability_argv(profile)
        argv += profile.flat_args()
        argv += ["--port", str(profile.port or 8000)]
        return argv

    def serve_env(self, m: Manifest, profile: ServeProfile) -> Dict[str, str]:
        env = {
            # the standalone plugin reads the mesh from the environment
            "MESH_DEVICE": profile.mesh_device or "",
            # tt_transformers-style adapters read the model id from HF_MODEL, not from
            # vLLM's --model. Both are set; they must agree.
            "HF_MODEL": _weights_id(m),
        }
        env.update(profile.env)
        return env

    def ready_probe(self, m: Manifest) -> str:
        return self.READY_LINE


class VllmForkLauncher:
    """``kind: vllm-fork`` — the ``tenstorrent/vllm`` fork with its in-tree plugin.

    Launched through tt-metal's readiness runner rather than ``vllm serve``: models on
    this stack expect that runner's flag names (``--tt-config``,
    ``--additional-server-args``, ``--server-timeout``), and it resolves the plugin's
    config flag against the installed vLLM, forwards an explicit mesh grid, and hands off
    to a stock ``vllm.entrypoints.openai.api_server``.

    Both the fork and its in-tree plugin install *editable* (the fork's own documented
    install), so the fork checkout must survive into the runtime image — about 200 MB.
    """

    name = "vllm-fork"

    #: keys the manifest's ``runtime:`` block may contain for this kind
    RUNTIME_KEYS = ("vllm", "extension", "lock", "model_dir")

    #: the log line whose appearance means the OpenAI server is accepting requests
    READY_LINE = "Server ready at"

    #: where the fork checkout lives in both build stages
    FORK_DIR = "/opt/vllm"

    #: vLLM's PyPI metadata is generated on a CUDA machine; without the CPU index a plain
    #: install resolves the CUDA dependency set (~4 GB of nvidia-* wheels, no device here)
    PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

    def validate(self, m: "ContainerManifest") -> None:
        from .container_manifest import ContainerManifestError

        rt = m.runtime
        vllm = rt.get("vllm") or {}

        if vllm.get("version") and not vllm.get("repo"):
            # The stock-vLLM shape. Say so precisely instead of "missing repo/ref": an
            # author who wrote this was describing a real stack, just not one we serve.
            raise ContainerManifestError(
                "runtime.vllm.version describes stock vLLM from PyPI — that is kind "
                "vllm-plugin, not vllm-fork. Either set kind: vllm-plugin, or give "
                "runtime.vllm.{repo, ref} for the fork."
            )
        if not (vllm.get("repo") and vllm.get("ref")):
            raise ContainerManifestError(
                "kind vllm-fork requires runtime.vllm.{repo, ref} — the tenstorrent/vllm fork, "
                "pinned to the sha the model was VALIDATED with (the plugin is in-tree)"
            )
        if rt.get("plugin"):
            raise ContainerManifestError(
                "kind vllm-fork takes no runtime.plugin: the plugin comes from the fork's own "
                "plugins/vllm-tt-plugin"
            )

        model_dir = rt.get("model_dir")
        if not model_dir:
            raise ContainerManifestError(
                "kind vllm-fork requires runtime.model_dir (the model's directory in the "
                "tt-metal tree, e.g. models/autoports/<name>) — the readiness runner "
                "launches by --model-dir"
            )
        # The launcher resolves model_dir INSIDE the image, so it must be covered by the
        # allowlist that decides what gets into the image at all.
        covered = any(
            model_dir == c or model_dir.startswith(c.rstrip("/") + "/") for c in m.source.code
        )
        if not covered:
            raise ContainerManifestError(
                f"runtime.model_dir {model_dir!r} is not covered by source.code — the "
                "launcher would not find the model inside the image"
            )

        for key in rt:
            if key not in self.RUNTIME_KEYS:
                raise ContainerManifestError(
                    f"kind vllm-fork does not understand runtime.{key}; expected one of "
                    + ", ".join(self.RUNTIME_KEYS)
                )

    # ---- image build ---------------------------------------------------------------

    def install_lines(self, m: "ContainerManifest") -> List[str]:
        """Shell lines that install the serving stack into the image's venv, after
        tt-metal itself is built and installed."""
        rt = m.runtime
        vllm = rt["vllm"]
        ref = vllm.get("sha") or vllm["ref"]
        lines: List[str] = [
            f"git clone {shlex.quote(vllm['repo'])} {self.FORK_DIR}"
            f" && git -C {self.FORK_DIR} checkout {shlex.quote(ref)}",
        ]
        if rt.get("lock"):
            # The lock IS the dependency set, so nothing resolves at build time and two
            # builds a week apart produce the same environment.
            lines.append(
                'uv pip install --python "$VENV/bin/python" -r /ctx/requirements.lock '
                f"--extra-index-url {self.PYTORCH_CPU_INDEX} --index-strategy unsafe-best-match"
            )
            deps = "--no-deps "
        else:
            deps = ""
        lines += [
            # Editable, matching the fork's own documented install; the checkout is COPY'd
            # into the runtime stage so the .pth files stay valid.
            f'VLLM_TARGET_DEVICE=empty uv pip install --python "$VENV/bin/python" {deps}'
            f"-e {self.FORK_DIR} --extra-index-url {self.PYTORCH_CPU_INDEX} "
            f"--index-strategy unsafe-best-match",
            f'uv pip install --python "$VENV/bin/python" -e {self.FORK_DIR}/plugins/vllm-tt-plugin',
            # The editable installs bake build-stage paths; drop VCS metadata only.
            f"rm -rf {self.FORK_DIR}/.git",
        ]
        if rt.get("extension"):
            # The model's own vLLM extension ships inside code/, so install it from the
            # staged tree already present in the image.
            lines.append(
                f'uv pip install --python "$VENV/bin/python" /opt/tt-metal/{rt["extension"]}'
            )
        return lines

    def verify_lines(self, m: "ContainerManifest") -> List[str]:
        """Assertions run INSIDE the finished image. These are what make the prune and
        the code/ allowlist safe: an under-shipped image fails here, on the author's
        machine, not on a consumer's first boot."""
        checks = [
            "import ttnn, vllm, vllm_tt_plugin",
            "import torch; assert torch.__version__.endswith('+cpu'), torch.__version__",
            f"import vllm; assert vllm.__file__.startswith('{self.FORK_DIR}'), vllm.__file__",
            "import models.common.readiness_check.run_vllm_server",
        ]
        # torch is a TRANSITIVE dependency of vLLM here (tt-metal installs none), so
        # nothing makes it agree with what ttnn was built against unless we check.
        pin = metal_torch_pin(_local_metal_tree(m))
        if pin:
            checks.append(
                f"import torch; v = torch.__version__.split('+')[0]; "
                f"assert v == {pin!r}, "
                f"f'torch {{v}} was resolved by vLLM but tt-metal pins {pin}; "
                f"ttnn extension modules were built against {pin} — pin it via "
                f"runtime.lock, or align runtime.vllm.ref'"
            )
        if m.runtime.get("extension"):
            checks.append(
                "import os; md = os.environ['EXTRA_MODELS_DIR']; "
                "entries = [e for e in os.listdir(md) "
                "if os.path.exists(os.path.join(md, e, 'vllm_metadata.json'))]; "
                "assert entries, f'EXTRA_MODELS_DIR {md} registers no models'"
            )
        lines = [f'"$VENV/bin/python" -c {shlex.quote("; ".join(checks))}']
        # the model author's own assertions, from the manifest's verify: list
        lines += [f'"$VENV/bin/python" -c {shlex.quote(v)}' for v in m.verify]
        return lines

    # ---- serve -----------------------------------------------------------------------

    def serve_argv(self, m: Manifest, profile: ServeProfile) -> List[str]:
        from .container_manifest import parse_mesh_device

        rows, cols = parse_mesh_device(profile.mesh_device or "")
        model_dir = (m.container.runtime if m.container else {}).get("model_dir")
        if not model_dir:
            raise LauncherError("kind vllm-fork requires runtime.model_dir")
        argv = [
            "python", "-m", "models.common.readiness_check.run_vllm_server",
            "--stages", "serve",
            "--model-dir", str(model_dir),
            "--hf-model", _weights_id(m),
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
        # This runner takes extra server flags as ONE joined string, not as loose argv.
        extra = profile.flat_args() + _capability_argv(profile)
        if extra:
            argv += ["--additional-server-args", " ".join(extra)]
        return argv

    def serve_env(self, m: Manifest, profile: ServeProfile) -> Dict[str, str]:
        # The mesh goes through --mesh-device here, NOT through a MESH_DEVICE env var.
        # tt_transformers-style adapters read the model id from HF_MODEL, not from
        # vLLM's --model.
        env = {"HF_MODEL": _weights_id(m)}
        env.update(profile.env)
        return env

    def ready_probe(self, m: Manifest) -> str:
        return self.READY_LINE


# tt-metal declares the torch it expects in its dev requirements, e.g.
#   --extra-index-url https://download.pytorch.org/whl/cpu
#   torch==2.11.0 ; platform_machine == 'x86_64'
_TORCH_PIN_RE = re.compile(
    r"^torch==(?P<version>[^\s;#]+)\s*(;.*platform_machine\s*==\s*'x86_64')?\s*$"
)

METAL_TORCH_REQUIREMENTS = "tt_metal/python_env/requirements-dev.txt"


def _local_metal_tree(m: "ContainerManifest") -> Optional[Path]:
    """The author's tt-metal checkout, or None when the manifest names a git source."""
    src = m.source.tt_metal
    return Path(src) if isinstance(src, str) else None


def metal_torch_pin(metal_tree: Optional[Path]) -> Optional[str]:
    """The torch version tt-metal declares it wants, or None if it cannot be read.

    tt-metal's ``setup.py`` has no ``install_requires``, so installing it brings NO
    torch: torch arrives only as a transitive dependency of vLLM. Nothing therefore
    forces the two to agree, and a vLLM release that resolves a different torch than
    ttnn's extension modules were built against breaks at import or, worse, at device
    open. Reading the pin lets the image assert the agreement at BUILD time.

    Returns None for a git-mode source (no local tree yet) or an unreadable/changed
    requirements file — the check is then skipped rather than guessed at.
    """
    if metal_tree is None:
        return None
    req = Path(metal_tree) / METAL_TORCH_REQUIREMENTS
    try:
        text = req.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        m = _TORCH_PIN_RE.match(line.strip())
        if m:
            return m.group("version")
    return None


#: Resolve every model registered through EXTRA_MODELS_DIR, mirroring
#: ``vllm_tt_plugin.platform._register_models_from_extra_dir`` and vLLM's later lazy
#: import of the ``"module:Class"`` string it stores.
RESOLVE_EXTRA_MODELS = (
    "import os, sys, json, importlib; "
    "md = os.environ['EXTRA_MODELS_DIR']; "
    "found = 0\n"
    "for e in sorted(os.listdir(md)):\n"
    "    d = os.path.join(md, e)\n"
    "    mp = os.path.join(d, 'vllm_metadata.json')\n"
    "    if not os.path.exists(mp):\n"
    "        continue\n"
    "    found += 1\n"
    "    meta = json.load(open(mp))\n"
    "    spec = meta['main_class']\n"
    # the plugin appends (never inserts) so an installed package of the same name wins
    "    if d not in sys.path:\n"
    "        sys.path.append(d)\n"
    "    mod, sep, cls = spec.rpartition(':')\n"
    "    if not sep:\n"
    "        mod, _, cls = spec.rpartition('.')\n"
    "    obj = getattr(importlib.import_module(mod), cls)\n"
    "    assert obj is not None, spec\n"
    "    print('resolved', meta.get('arch'), '->', spec)\n"
    "assert found, f'EXTRA_MODELS_DIR {md} registers no models'\n"
)


def _capability_argv(profile: ServeProfile) -> List[str]:
    """Render the tool/reasoning parsers, matching the v4 path's rules exactly.

    ``--tool-call-parser`` is emitted WITH ``--enable-auto-tool-choice`` because vLLM
    hard-errors on the former without the latter, and ``--reasoning_parser`` keeps its
    underscore: typer normalises '_'->'-' so '--tool_parser' would become the nonexistent
    '--tool-parser', while '--reasoning_parser' normalises to the valid spelling.
    See ``bundles._compose_launch_vllm``.
    """
    cap = profile.capabilities
    if cap is None:
        return []
    argv: List[str] = []
    if cap.tool_parser:
        argv += ["--enable-auto-tool-choice", "--tool-call-parser", cap.tool_parser]
    if cap.reasoning_parser:
        argv += ["--reasoning_parser", cap.reasoning_parser]
    return argv


KINDS: Dict[str, object] = {
    VllmPluginLauncher.name: VllmPluginLauncher(),
    VllmForkLauncher.name: VllmForkLauncher(),
}


def launcher_for(kind: str):
    try:
        return KINDS[kind]
    except KeyError:
        raise LauncherError(
            f"unsupported kind {kind!r}; supported: " + ", ".join(sorted(KINDS))
        ) from None
