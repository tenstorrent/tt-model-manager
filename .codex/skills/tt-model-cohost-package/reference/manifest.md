# The co-host manifest and image

## Manifest (`kind: cohost`)

```json
{
 "container": {
  "kind": "cohost",
  "image": {"registry": "hf", "repository": "cohost-llm-blackhole", "tag": "tt-model/cohost-llm:r4", "digest": "sha256:..."},
  "runtime": {
   "members": [
    {"name": "gptoss", "kind": "vllm", "model": "openai/gpt-oss-20b", "port_offset": 0,
     "args": ["--max-num-seqs", "32", "--max-model-len", "131072"],
     "env": {"GPT_OSS_20B_FUSED_B1": "0"}},
    {"name": "llama8b", "kind": "vllm", "port_offset": 1,
     "model": "/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/<revision>",
     "args": ["--served-model-name", "meta-llama/Llama-3.1-8B-Instruct", "--max-model-len", "131072"],
     "kv_tokens": 131072, "preload": "models.cohost_llm.fit:tt_transformers",
     "env": {"HF_MODEL": "<the same snapshot path>", "MAX_PREFILL_CHUNK_SIZE": "2",
             "TT_COHOST_DISTRIBUTED_PREFILL_NORM": "1"}},
    {"name": "qwen32b", "kind": "vllm", "port_offset": 2, "model": "...", "kv_tokens": 65536,
     "preload": "models.cohost_llm.fit:tt_transformers"}
   ],
   "profiles": {"gptoss-llama": ["gptoss", "llama8b"], "gptoss-qwen": ["gptoss", "qwen32b"]},
   "tt_config": {"fabric_config": "FABRIC_1D_RING", "trace_region_size": 950000000}
  },
  "serve": {"port": 20000},
  "serve_profiles": [
   {"name": "gptoss-llama", "hardware": "p150x4", "mesh_device": "P150x4", "env": {"GPT_OSS_20B_KV_POOL_BLOCKS": "16512"}},
   {"name": "gptoss-qwen",  "hardware": "p150x4", "mesh_device": "P150x4", "env": {"GPT_OSS_20B_KV_POOL_BLOCKS": "4128"}}
  ]
 }
}
```

| member key | meaning |
| --- | --- |
| `name` | unique in the group; logs and the status endpoint use it |
| `kind` | `vllm` (stock vLLM OpenAI server per member; its worker runs in the device host) or `app` (`module:factory` in `app` returns the member's ASGI app). One kind per group |
| `model`, `args` | vllm: what `vllm serve` gets. A snapshot path plus `--served-model-name <hf id>` keeps the API name and works offline |
| `port_offset` | the member's port is `serve.port + port_offset`, the same in every profile; distinct, >= 0 |
| `env` | set while this member loads and warms up, and in its vLLM frontend; put back afterwards |
| `kv_tokens` | vllm: cap on this member's KV pool, so the members' DRAM adds up |
| `preload` | `module:function` called before the member's model is built: an image-side fit hook |
| `app` | app: `module:factory` |

- `runtime.profiles` maps each serve profile to its members; without it every profile serves every
  member. A key that names no serve profile is refused (a typo would serve everything).
- The status endpoint is the port after the highest published member port of the profile.
  Leave the whole block free (deploy reserves it).
- Only the keys a member declares in `env` are restored after its load. A model that writes
  `os.environ` itself while loading leaks into the next member (S2 Pro set `HF_MODEL`, and
  Music 3 then loaded S2 Pro's config): its adapter must restore what it changes.

## Image

```dockerfile
FROM <the image every member already runs on>
USER root
COPY tt-cohost /opt/tt-cohost                 # tt_kernel/__init__.py + tt_kernel/cohost from tt-model-manager
COPY models/<your_fit_hooks_or_adapters> /opt/tt-metal/models/<...>
RUN chmod -R a+rX /opt/tt-cohost /opt/tt-metal/models/<...>
ENV PYTHONPATH=/opt/tt-cohost:/opt/tt-metal HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
USER 1000:1000
RUN python -c "import tt_kernel.cohost.launch, <each member's module>"
ENTRYPOINT []
CMD ["python", "-m", "tt_kernel.cohost.launch"]
```

The server runs as the host user's uid, not the image's: check that every copied file is world
readable. The ready line is `cohost: all members ready`.

## What the host does

- Opens one mesh over the group's chips; its device loop runs on the main thread (tt-metal closes
  the chips on the main thread at exit, and UMD's lock can only be released by the thread that
  took it).
- Boots in phases: load every member, compile every member with tracing off, then capture traces
  member by member. App members: warm up all, release every trace, warm up all again.
- A member owns the device for a whole step (vLLM's `execute_model` through `sample_tokens`; an
  app's whole request); requests take turns first come, first served.
- One member down (a vLLM frontend exits, an app's HTTP thread ends) stops the whole group.
- The frontends reach the host over a Unix socket in a directory private to the host's user; a
  peer running as another user is dropped.
