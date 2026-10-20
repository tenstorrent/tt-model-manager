# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Co-host runtime: several models served from ONE process that owns the chips.

Only one process can open a Tenstorrent chip, so models that share chips must share a
process. This package runs inside a co-host container image. It must import nothing beyond
the standard library at module level, because it runs next to vLLM and ttnn, not tt-model's
own CLI dependencies.

- ``host``: the device host. It opens one mesh, runs every member through a phased boot,
  and serializes their device work on one device thread.
- ``vllm_remote``: a vLLM executor whose worker lives in the device host.
- ``launch``: the container entrypoint. It starts the host, then one vLLM server per member.
"""

#: Version of the JSON config contract between tt-model (which writes it) and this runtime.
CONFIG_VERSION = 1
