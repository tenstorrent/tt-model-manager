# Co-hosting: several models on the same chips

A Tenstorrent chip can be opened by one process at a time. UMD takes a per-chip lock when the
device starts and holds it until the process exits. So two models that must share chips have
to share a process. `tt_kernel.cohost` is that process.

- **The device host** opens one mesh over the group's chips and owns every device call. Its
  device loop runs on the process's main thread: tt-metal closes the chips on the main thread
  at exit, and UMD's per-chip lock can only be released by the thread that took it.
- **vLLM members.** Each one keeps its own stock vLLM OpenAI server on its own port. Its
  worker runs inside the device host, behind
  `--distributed-executor-backend tt_kernel.cohost.vllm_remote.RemoteExecutor`. The
  scheduler, the tool and reasoning parsers, and the whole API stay vLLM's own.
- **App members** (speech, music: models with their own HTTP API) run their ASGI app in
  the device host, on their own port.
- **Status.** The host serves `GET /` on `status_port`, with each member's readiness and its
  measured DRAM and L1.

Members take turns on the device. Nothing runs at the same time; see "What co-hosting does not
do".

## The group config

The host reads one JSON document, from `--config <file>` or the `TT_COHOST_CONFIG`
environment variable:

```json
{
  "cohost_config_version": 1,
  "mesh_device": "P150x4",
  "tt_config": {"fabric_config": "FABRIC_1D_RING", "trace_region_size": 950000000},
  "status_port": 21099,
  "members": [
    {"name": "gptoss", "kind": "vllm", "port": 21000, "model": "openai/gpt-oss-20b",
     "args": ["--max-num-seqs", "32"], "env": {"GPT_OSS_20B_FUSED_B1": "0"}},
    {"name": "llama8b", "kind": "vllm", "port": 21001, "model": "meta-llama/Llama-3.1-8B-Instruct",
     "kv_tokens": 65536, "preload": "models.cohost_llm.fit:tt_transformers"}
  ]
}
```

| member key | meaning |
|---|---|
| `kind` | `vllm` (served by vLLM) or `app` (its own ASGI app). One group holds one kind. |
| `model`, `args` | vllm: what `vllm serve` gets |
| `env` | applied while this member loads, and in its vLLM frontend |
| `kv_tokens` | vllm: cap on this member's KV pool, so the members' DRAM adds up |
| `preload` | `module:function` called before this member is built (an image-side fit patch) |
| `app` | app: `module:factory` returning an `AppMember` |

`python -m tt_kernel.cohost.launch` boots the group, prints `cohost: all members ready`, and
exits non-zero if any member fails. One failure stops the whole group.

A member's `env` is set around its own load and warmup calls and put back afterwards. Only
those keys are put back: if a model writes `os.environ` itself while it loads, the members
after it see that value. (Fish Audio S2 Pro points `HF_MODEL` at its own weights, and
MiniMax Music 3 then loaded S2 Pro's config.) An adapter for such a model should restore what
it changes.

## Why the boot is phased

A trace replays the exact addresses it used at capture. Any buffer or program binary
allocated after some trace was captured may sit in that trace's scratch memory, and is
overwritten when the trace replays. So the host boots ALL members together:

1. **Load:** every member's weights and KV cache.
2. **Compile:** every member warms up with tracing off, so every program is compiled and
   every lazily allocated buffer exists.
3. **Capture:** traces are captured member by member, in config order. Nothing new is
   compiled or allocated from here on.

App members get the same guarantee another way: warm up all of them once, release every
trace, then warm up all of them again. The second round compiles and allocates nothing new.

While serving:

- **Step ownership.** A member owns the device for a whole step (vLLM's `execute_model`
  through its `sample_tokens`; an app's whole request). Another member's replay therefore
  never lands between a step's forward and its readback.
- **Late compiles.** A program compiled after capture is counted per member
  (`compiles_after_capture` in the status) and logged, because it can corrupt a parked trace.
  They do happen: a new batch size or prompt length can need a program the warmup did not
  build. Over one benchmark run (1, 8 and 32 users, plus 3000-token prompts) GPT-OSS-20B
  compiled 91 programs after capture and Llama-3.1-8B 30. GPT-OSS's own generator releases and
  recaptures its decode traces around such compiles; tt_transformers does not. Greedy output
  of both members after that whole run was still identical to their baselines (16 of 16 each),
  so no corruption was seen, but the risk is not ruled out.

## The two budgets: DRAM and L1

**DRAM per chip** is the easy one. The host measures it per member and phase (status
`dram_gib_per_chip`). Measured on a QuietBox 2 (4 chips):

| member | weights | KV cache |
|---|---|---|
| GPT-OSS-20B, fused batch-1 path | 12.78 GiB/chip | 3.29 GiB/chip (16,512 blocks) |
| GPT-OSS-20B, `GPT_OSS_20B_FUSED_B1=0` | 7.10 GiB/chip | 3.29 GiB/chip |
| Llama-3.1-8B-Instruct | 1.73 GiB/chip | 1.06 GiB/chip (65,536 tokens) |

Cap a member's KV pool with `kv_tokens`, so the members add up.

**L1 per core** is the binding one. Some models keep buffers resident in L1 between steps:
CCL semaphores, persistent CCL buffers, fused-kernel receive buffers. Those addresses are
then gone for every other member's programs. Each member's program configs were tuned
assuming the whole L1, so the next member's program fails with `Statically allocated dataflow
buffers ... clash with L1 buffers`. Measured:

- GPT-OSS-20B on 4 chips holds about 370 KiB/core after load, and more after warmup. Its fused
  batch-1 path holds more still.
- Llama-3.1-8B's default 4-chip prefill programs need circular buffers up to about 1.2 MiB.

What follows from this:

- Members must not keep persistent L1 state that overlaps what the others need. Run GPT-OSS
  with `GPT_OSS_20B_FUSED_B1=0` in a co-host group.
- Size the other members' programs to fit. Llama-3.1-8B's offender is its prefill RMSNorm:
  tt_transformers distributes it across chips only for models wider than 4096, and Llama-8B
  is exactly 4096. A `preload` hook that turns the distributed prefill norm on makes it fit.
  It changes Llama's numerics slightly (a different reduction order), so take its baseline
  with the same hook. Measured quality with and without it: GSM8K 0.784 both; IFEval
  prompt-strict 0.724 vs 0.740 on 250 prompts (within the +/-0.027 standard error).
- The boot fails loudly (the clash is a hard error at compile time), never silently.

Measured with GPT-OSS-20B and Llama-3.1-8B in one group on 4 chips (QuietBox 2): both members
gave the same greedy output as their own baselines (16 of 16 prompts each), at the same latency,
and still 16 of 16 when both were answering at once (their steps interleaved). With 32 users
each, output was 1444 tok/s (GPT-OSS) and 1150 tok/s (Llama) when each was busy alone, and
825 + 790 = 1615 tok/s with both busy.

Measured with Qwen3-32B instead of Llama, also on 4 chips: same result (16 of 16 for both,
sequential, interleaved and after a full benchmark run). Both busy at 32 users each, GPT-OSS
gave 465 and Qwen 435 tok/s: a slow member's steps set the pace for both.

Measured with three app members on ONE chip (Whisper large-v3, Fish Audio S2 Pro and MiniMax
Music 3): S2 Pro's and Music 3's seeded audio was byte-identical to each model served alone,
Whisper's text was identical, also with all three requests sent at once. Together they need
29.5 GiB of the chip's 32 GB, which only fits with Music 3's context cut from 16K to 8K tokens.

## What co-hosting does not do

- **It does not run models at the same time.** On one mesh, every member's work is one ordered
  stream. Two members each get roughly half the device when both are busy. Spatially splitting
  a chip into sub-devices would let them overlap, but every model op would have to be confined
  to its cores (about 150-200 grid sites per model today).
- **It does not mix tt-metal builds.** One process means one ttnn. Every member must run on the
  image's tt-metal tree.
- **It does not share chips across packages.** Two separately published packages cannot share
  chips. Put the models in one co-host package.
- **It costs the fastest single-model paths.** A model that keeps state in L1 between steps
  must turn that off. GPT-OSS-20B with `GPT_OSS_20B_FUSED_B1=0` gives 1444 tok/s at 32 users,
  against 1753 tok/s for the published 4-chip throughput profile, which uses the fused path.
