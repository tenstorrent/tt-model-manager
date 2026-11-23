<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc. -->
# Co-host packages: several models on the same chips

A Tenstorrent chip can be opened by one process at a time, so models that must share chips
have to share a process. That process is the
[tt-cohost](https://github.com/jashansinghTT/tt-cohost) runtime: it opens one mesh, boots
every member in phases and serializes their device work, while each vLLM member keeps its
own stock OpenAI server on its own port. Its README covers the group config, the member
contract, the boot phases, the status endpoint and the limits.

`tt-model` only packages and launches a group (`kind: cohost`):

- **`package`** builds the image FROM an existing package image (`runtime.base_image`,
  whose tt-metal, vLLM and plugin every member shares) plus tt-cohost at a pinned commit.
  There is no tt-metal build, so it takes minutes, and nothing installed on top may move
  the base image's packages (everything is installed with `--no-deps`).
- **`serve`** starts the container with one published port per member plus the status
  port, passes the group config for the chosen profile in `TT_COHOST_CONFIG`, and runs
  `python -m tt_cohost.launch`. The boot is done when the log says
  `cohost: all members ready`.
- **`stop`, `logs`, `pull`, `push`** work as for any container package.
- **`deploy up`** runs tt-cohost's fit check before serving a group and serves its members on
  separate chips when it does not fit ([multi_model.md](multi_model.md#models-that-share-chips)).

## The manifest

```yaml
schema: "5.1"
repo: you/cohost-llm
name: cohost-llm
weights: {repo: openai/gpt-oss-20b, revision: 6cee5e81ee83917806bbde320786a8fb61efebee}
auxiliary_weights:              # every other member's weights, pulled and mounted the same way
  - {repo: meta-llama/Llama-3.1-8B-Instruct, revision: 0e9e39f249a16976918f6564b8830bc894c89659}
kind: cohost
arch: blackhole
runtime:
  base_image: tt-model/gpt-oss-20b-blackhole-fast:<tag>
  tt_cohost: {repo: https://github.com/jashansinghTT/tt-cohost, ref: <sha>}
  packages: []                  # optional pip requirements, installed --no-deps
  code:                         # optional, copied onto /opt/tt-metal
    - {root: ~/my-fit-hooks, paths: [models/cohost_llm]}
  members:
    - {name: gptoss, kind: vllm, model: openai/gpt-oss-20b, port_offset: 0}
    - {name: llama8b, kind: vllm, model: meta-llama/Llama-3.1-8B-Instruct, port_offset: 1,
       kv_tokens: 65536, preload: "models.cohost_llm.fit:tt_transformers"}
  profiles:                     # optional: profile name -> the members it serves
    gptoss-llama: [gptoss, llama8b]
  tt_config: {fabric_config: FABRIC_1D_RING, trace_region_size: 950000000}
serve: {port: 8000}
serve_profiles:
  - {name: gptoss-llama, hardware: p150x4, mesh_device: P150x4}
```

| field | meaning |
|---|---|
| `runtime.base_image` | the package image the members run in; it must be on this machine (`tt-model pull` it first) |
| `runtime.tt_cohost` | `{repo, ref}`; `package` pins `ref` to a sha and records it in the manifest, the card and an image label |
| `runtime.packages` | extra requirements the members need on top of the base, installed with `--no-deps` |
| `runtime.code` | extra code (`root` is a local path or `{repo, ref}`), staged into `code/` like `source.extra_code` |
| `runtime.members` | `name`, `kind` (`vllm` or `app`; a mixed group lists its vLLM members first), `model`, `port_offset`, and optional `args`, `env`, `kv_tokens`, `app`, `preload`, and `standalone` (`{model, profile}`: the package `deploy` serves it with on its own chips when the group does not fit) |
| `runtime.profiles` | which members each serve profile runs; a profile not listed runs them all |
| `runtime.tt_config` | the device settings the one mesh opens with |

A member's port is the serve port plus its `port_offset`; the status port is the one after
the highest member port. `serve` looks for a free run of ports wide enough for all of them.
