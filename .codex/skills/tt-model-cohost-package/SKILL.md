---
name: tt-model-cohost-package
description: Build, validate and publish a `kind: cohost` container package — several models served from ONE process on the SAME chips, each on its own port, with the tt-cohost runtime (for example two LLMs on all four chips of a QuietBox 2, or speech-to-text + text-to-speech + music on one chip). Covers when chips must be shared, the authored manifest (base image, pinned tt-cohost, members, profiles), the build, the fit check and conformance test, the hardware validation, and publishing. Use for "co-host these models", "put two models on the same chips", "one chip for TTS and STT", "package this group". Do NOT use when the models fit on separate chips — serve them as separate packages (`tt-model serve` each, or `tt-model deploy`).
---

# Build a co-host package and prove it

Input: two or more models that already serve from ONE image (a published package's image whose
tt-metal, vLLM and plugin every member can run on). Output: one `kind: cohost` package, every
model checked against its own baseline on hardware, and a card that states the measured budgets
and limits.

The runtime is the separate [tt-cohost](https://github.com/jashansinghTT/tt-cohost) package: read
its README for the group config, the member contract (what a member must and must not do), the
boot phases, the status endpoint, the DRAM and L1 budgets and the failure signatures. tt-model
only packages and launches the group. References here:
[reference/manifest.md](reference/manifest.md) (the authored manifest and what `package` builds),
[reference/validation.md](reference/validation.md) (the protocol, step by step).

## Step 1 — Confirm the chips really must be shared

A chip can be opened by one process at a time, so co-hosting is for models that cannot get chips
of their own. If the models fit on separate chips, stop here and serve them separately: separate
processes run at the same time, while co-hosted models take turns and each gets roughly half the
device when all are busy.

Hard limits of one process: one tt-metal build and one Python env (the base image's), one mesh
that every member uses, and the members take turns on the device.

## Step 2 — Write the manifest

`kind: cohost`, no `source:` block. `runtime.base_image` is the image the members share;
`runtime.tt_cohost` pins the runtime (`ref` is a branch, a tag or a full sha; `package` resolves it
to a commit and records it). Members, profiles and `tt_config` are the group. Details and an
example: [reference/manifest.md](reference/manifest.md).

Put every member's weights in `weights`/`auxiliary_weights` with a pinned `revision`, and load
tt_transformers members from their snapshot path (`/hf/hub/models--<org>--<name>/snapshots/<rev>`)
with `--served-model-name <hf id>`, so a pulled package serves offline.

## Step 3 — Build

```bash
tt-model -v package --container tt-model.yaml --out build
```

No tt-metal build: the image is the base image plus tt-cohost (and `packages`, `code`), all
installed with `--no-deps`. It takes minutes. The base image must be on the machine. The build
runs `verify.sh` as the image's user: it imports `tt_cohost.launch`, every member's `preload`
module and your `verify:` statements. Check `docker history` (a few MB on top of the base) and the
card's provenance rows (base image, tt-cohost commit).

## Step 4 — Fit, then validate

1. **Fit check.** `python -m tt_cohost.fit --config <group.json>` inside the image boots the group
   and reports, per member and phase, DRAM and L1 in use, programs compiled after trace capture,
   and on failure the member, the phase and the error. Fix every failure before going further.
2. **Validate** with [reference/validation.md](reference/validation.md): solo baselines → group
   sequential → group interleaved → load → greedy after load → quality subset for any member a fit
   hook changed → soak → clean stop → package check → clean-host check. **Do not publish on "it
   booted".**

## Step 5 — Publish

- The manifest needs a `card:` block with `performance` and `limitations`, or a listing is
  refused. The card says: the models take turns; the measured budgets and the fit hooks' effect;
  each model's own license (and gated weights, which need an HF token with access); and that
  serving needs a tt-model with `kind: cohost`.
- `tt-model push build/<name>` (private), then `--publish` once the clean-host check passed.

## Report

Per profile: members, tt-cohost commit, base image, ready time (warm and clean host), DRAM per
chip, L1 held per core, greedy comparisons (sequential, interleaved, after load), throughput alone
and all busy, programs compiled after capture, eval deltas, soak result. Say what was not run.
