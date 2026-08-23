# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The ModelType contract.

A *type* is how a model's serving environment is built and launched. The enumeration of
types lives in ``tt_model/types/__init__.py`` (and, in prose, ``docs/model_types.md``).
Adding a future type (diffusion, CNN, a non-vLLM engine) is a new module here plus a row
in that doc — no changes to the Dockerfile, the manifest schema, or any command: the
Dockerfile is type-agnostic, and everything type-specific arrives through the hooks
below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..manifest import Manifest, ServeProfile


@runtime_checkable
class ModelType(Protocol):
    #: the manifest's `type:` value
    name: ClassVar[str]

    def validate(self, m: "Manifest") -> None:
        """Type-specific manifest checks (shape of `runtime:`, launcher needs).
        Raise ManifestError on anything that must not proceed."""
        ...

    def install_lines(self, m: "Manifest") -> List[str]:
        """Shell lines for the builder stage that install the serving stack into the
        venv, after tt-metal itself is built and installed."""
        ...

    def verify_lines(self, m: "Manifest") -> List[str]:
        """Shell lines for the final stage's build-time verification RUN. These are
        what make pruning safe: an under-shipped image fails HERE, on the author's
        machine, not on a consumer's pull."""
        ...

    def runtime_copy_lines(self, m: "Manifest") -> List[str]:
        """Extra COPY --from=builder lines the final stage needs beyond the shared set
        (e.g. vllm-legacy's editable fork checkout)."""
        ...

    def serve_argv(self, m: "Manifest", profile: "ServeProfile") -> List[str]:
        """The command the container runs (exec'd by entrypoint.sh, so it is PID 1 and
        receives docker stop's SIGTERM — a clean shutdown closes the mesh; a SIGKILL
        leaves the devices needing a reset)."""
        ...

    def serve_env(self, m: "Manifest", profile: "ServeProfile") -> Dict[str, str]:
        """Environment for the serve process, on top of the image's baked ENV."""
        ...

    def ready_probe(self, m: "Manifest") -> str:
        """A log line whose appearance means the server is up (used by serve --follow)."""
        ...
