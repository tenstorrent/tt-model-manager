# The co-host manifest and what `package` builds

## Manifest (`tt-model.yaml`, `kind: cohost`)

```yaml
schema: "5.1"
repo: you/cohost-llm-blackhole
name: cohost-llm-blackhole
weights: {repo: openai/gpt-oss-20b, revision: <sha>}
auxiliary_weights:
  - {repo: meta-llama/Llama-3.1-8B-Instruct, revision: <sha>}
kind: cohost
arch: blackhole
runtime:
  base_image: tt-model/gpt-oss-20b-fast:<tag>        # the image every member runs in
  tt_cohost: {repo: https://github.com/jashansinghTT/tt-cohost, ref: <sha>}   # or a branch or tag
  packages:                                           # optional, installed --no-deps
    - "torchvision @ https://download.pytorch.org/whl/cpu/torchvision-0.26.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl"
  code:                                               # optional, copied onto /opt/tt-metal
    - {root: ~/my-fit-hooks, paths: [models/cohost_llm]}
  members:
    - name: gptoss
      kind: vllm
      model: /hf/hub/models--openai--gpt-oss-20b/snapshots/<sha>
      port_offset: 0
      args: [--served-model-name, openai/gpt-oss-20b, --max-num-seqs, "32", --max-model-len, "131072"]
      env: {GPT_OSS_20B_FUSED_B1: "0"}
    - name: llama8b
      kind: vllm
      model: /hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<sha>
      port_offset: 1
      args: [--served-model-name, meta-llama/Llama-3.1-8B-Instruct, --max-model-len, "131072"]
      kv_tokens: 131072
      preload: "models.cohost_llm.fit:tt_transformers"
      env: {HF_MODEL: /hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<sha>}
  profiles: {gptoss-llama: [gptoss, llama8b]}
  tt_config: {fabric_config: FABRIC_1D_RING, trace_region_size: 950000000}
serve: {port: 20000}
serve_profiles:
  - {name: gptoss-llama, hardware: p150x4, mesh_device: P150x4, env: {GPT_OSS_20B_KV_POOL_BLOCKS: "16512"}}
default_profile: gptoss-llama
```

| member key | meaning |
| --- | --- |
| `name` | unique in the group; logs and the status endpoint use it |
| `kind` | `vllm` (its own stock vLLM OpenAI server) or `app` (`app: module:factory`). A mixed group lists its vLLM members first |
| `model`, `args` | vllm: what `vllm serve` gets |
| `port_offset` | the member's port is the serve port + `port_offset`, the same in every profile |
| `env` | set while this member loads, and in its vLLM frontend |
| `kv_tokens` | vllm: cap on this member's KV pool, so the members' DRAM adds up |
| `preload` | `module:function` called before the member's model is built: an image-side fit hook. `verify.sh` imports its module |
| `app` | app: `module:factory` |

- `runtime.profiles` maps a serve profile to its members; a profile not listed serves them all,
  and a key that names no serve profile is refused.
- The status endpoint is the port after the highest member port of the profile; `serve` reserves
  the whole range.

## What `package` builds

A rendered Dockerfile: FROM `base_image`; tt-cohost fetched at the pinned commit and installed with
uv (`--no-deps`) together with `packages`; `code` copied onto `/opt/tt-metal`; tt-model's
entrypoint and default serve script; `verify.sh` run as the image's user. The manifest, the card
and an image label record the base image and the tt-cohost commit. `serve` passes the profile's
group config in `TT_COHOST_CONFIG` and runs `python -m tt_cohost.launch`; the ready line is
`cohost: all members ready`.
