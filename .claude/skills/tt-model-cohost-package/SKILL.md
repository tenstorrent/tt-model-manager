---
name: tt-model-cohost-package
description: Build, validate and publish a `kind: cohost` container package — several models served from ONE process on the SAME chips, each on its own port (for example two LLMs on all four chips of a QuietBox 2, or speech-to-text + text-to-speech + music on one chip). Covers when chips must be shared, the members/profiles/port_offset manifest, the phased boot, the DRAM and L1 budgets and how to measure them, fit hooks, the validation protocol, a failure-signature table, and staging/publishing. Use for "co-host these models", "put two models on the same chips", "one chip for TTS and STT", "the second model does not fit next to the first". Do NOT use when the models fit on separate chips — use the tt-model-multi-model skill (`tt-model deploy`).
---

# Build a co-host package and prove it

Input: two or more models with working single-model packages (or at least working serve recipes)
on the same tt-metal tree. Output: one `kind: cohost` package, every model checked against its
own baseline on hardware, and a card that states the measured budgets and limits.

References: [reference/manifest.md](reference/manifest.md) (the manifest and the image),
[reference/budgets.md](reference/budgets.md) (DRAM, L1, fit hooks, measured numbers),
[reference/validation.md](reference/validation.md) (the protocol, step by step),
[reference/failures.md](reference/failures.md) (failure signature → fix).

## Step 1 — Confirm the chips really must be shared

A chip can be opened by one process at a time, so co-hosting is only for models that cannot get
chips of their own. If the models fit on separate chips, stop here and use `tt-model deploy`
(tt-model-multi-model skill): separate processes run at the same time; co-hosted models take
turns and each gets roughly half the device when both are busy.

Hard limits of one process:

- **One tt-metal build.** Every member must run on the image's tt-metal tree and Python env.
- **One mesh.** Every member uses the group's mesh (all its chips); there is no "this member on
  chips 0,1 only" inside a group today.
- **One kind per group:** all members `vllm` (each keeps its own vLLM OpenAI server) or all `app`
  (each its own ASGI app, for speech and music servers).

## Step 2 — Write the group and the manifest

The package is a v5.1 manifest with `kind: cohost`: a `runtime.members` list (each `name`,
`kind`, `model`, `port_offset`, optional `args`, `env`, `kv_tokens`, `preload`, `app`), optional
`runtime.profiles` (profile name → the members it serves) and `runtime.tt_config` (the one
mesh's device settings). Member ports are `serve.port + port_offset`; the status endpoint is the
port after the last. Details and an example: [reference/manifest.md](reference/manifest.md).

The image is built `FROM` the image the members already share (a published package's image),
plus the co-host runtime (`tt_kernel.cohost`) and any adapters or fit hooks, not by
`tt-model package`. Verify imports as the uid the server runs as.

## Step 3 — Fit the budgets

Boot each member alone in the group runtime first, then the group. Read the status endpoint
(`GET /` on the status port): `dram_gib_per_chip` per member and phase, `l1_kib_per_core_after_phase`,
`compiles_after_capture`. Set `TT_COHOST_L1_REPORT=1` to log every live L1 buffer after each phase.

- **DRAM** adds up per member: weights + KV pool + trace region. Cap each vLLM member's KV pool
  with `kv_tokens`; context length is a DRAM cost, not an L1 one.
- **L1 is the binding budget.** A member that keeps buffers resident in L1 between steps (CCL
  state, fused-kernel buffers) takes those addresses from every other member's programs; the
  next member then fails to compile with an L1 clash. Turn such state off for the co-hosted
  member and size the others' programs to fit, with `preload` fit hooks.

Known fits (measured; numbers in [reference/budgets.md](reference/budgets.md)): GPT-OSS-20B with
`GPT_OSS_20B_FUSED_B1=0`; tt_transformers Llama-3.1-8B with the distributed prefill RMSNorm (and
its chunked/prefix-cached prefill kwargs fix); MiniMax Music 3 at 8K context next to Whisper and
S2 Pro on one chip. Take every baseline with the same fit hooks the group uses.

## Step 4 — Validate

Follow [reference/validation.md](reference/validation.md). In short: solo baselines → group
sequential (same outputs) → group interleaved (all members at once, same outputs) → ladder
benchmark alone and all busy → greedy again after the load → an eval subset for any member whose
numerics a fit hook changed → a soak → clean stop → a clean-host consumer check of the staged
package. **Do not publish on "it booted".**

## Step 5 — Stage and publish

- Stage the manifest, the image (OCI under `image/`), README and SHA256SUMS; pin each member's
  weights revision (a member loaded from a snapshot path stays offline-safe after `tt-model pull`).
- The manifest needs a `card:` block with `performance` and `limitations`, or a listing (and a
  re-push of a listed repo) is refused. The card must say: the models take turns; the measured
  budgets and the fit hooks' effect; each model's own license (and any gated weights, which need
  an HF token with access); and that serving needs a tt-model with `kind: cohost`.
- `tt-model push <staged-dir>` (private), then `--publish` once the clean-host check passed.
- Clean-host check: run tt-model with a fresh `HOME` (tt-model keeps its state and compiled caches
  under `~/.cache/tt-model` and does not follow `XDG_CACHE_HOME`), remove the local image tag so
  the image comes from the Hub, then `pull` + `serve` and compare outputs with the experiment.

## Report

Per profile: members, ready time (warm and clean host), DRAM per chip, L1 held per core, greedy
and seeded comparisons (sequential, interleaved, after load), throughput alone and all busy,
late compiles while serving, eval deltas, soak result. Say what was not run.
