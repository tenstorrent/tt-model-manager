---
name: tt-model-fast-path
description: Ship a model's Blackhole fast paths in a CONTAINER (v5.1) package — fused batch-1 decode per mesh, multi-user decode, opt-in deterministic prefill, opt-in speculative decoding — with one image, a serve profile per trade-off with explicit env switches, the KV capacity stated per profile, a speculative drafter as a pinned pointer fetched on first boot, and a release-vs-release ladder and quality gate before a new revision is pushed. Use for "package the fast path", "add a speculative profile", "ship a faster revision of this package". The kernels come from the fast-path skill of the tt-model-bringup plugin (tenstorrent/skills); do NOT use this skill to build them, and do NOT use it for v6 thin bundles.
---

# Ship a model's fast paths

Input: a model whose fast paths are built and gated in its tt-metal checkout, and the previous
release of its package (a published `<org>/<name>` and the `tt-model.yaml` it was built from).
The `fast-path` skill of the `tt-model-bringup` plugin in
[tenstorrent/skills](https://github.com/tenstorrent/skills) leaves that evidence in the model's
`doc/fast_path/`: seven evidence files, a condition table, and each path's switch. Output: a new
package revision whose profiles expose those paths, measured on hardware against the previous
release, and a report that says what was verified.

This skill decides profiles, switches and release checks. The manifest fields belong to the
[tt-model-yaml](../tt-model-yaml/SKILL.md) skill; build, serve, stop and push belong to the
[tt-model-package-test](../tt-model-package-test/SKILL.md) skill. A v6 thin bundle has one launch
configuration and no profiles, so opt-in modes need a container package.

## Rules

1. **One image serves every profile.** A profile selects a path through its `env`. Never build a
   second image for a faster mode.
2. **Every switch is explicit in every profile.** Write each fast-path variable in each profile's
   `env`, on or off, even where it matches the code's default, so a default changed later in the
   model code cannot silently change a published profile.
3. **Defaults never get slower.** A path that costs another workload (a two-chip decode that slows
   eight users, speculation, deterministic prefill) ships as its own named profile, opt-in by
   name. Say the trade-off in that profile's `description`.
4. **State the capacity of every profile.** Decode-only weight copies cost DRAM. Write each
   profile's context limit, maximum sequences and KV capacity into its `description`, for example
   `KV cache 1,056,768 tokens (16,512 blocks of 64)`. A profile whose KV pool no longer admits its
   `max_model_len` does not ship.
5. **A second model is a pointer.** A drafter, or any other model a profile needs, is a Hub repo
   pinned to a commit. The package's own entrypoint code downloads it into the mounted HF cache on
   that profile's first boot, before the server starts; later boots reuse it and other profiles
   never touch it. Never copy it into the image, the bundle or `weights`. If the server runs with
   `HF_HUB_OFFLINE=1`, run that one download in a child process with offline mode cleared. When
   it fails, the error names the repo, the revision, a host command
   (`hf download <repo> --revision <sha>`) and the plain profile that needs no drafter.
6. **Compare releases on the same day.** Measure the previous release and the candidate on the same
   box, with the same client, settings and prompts. A chat template that writes the date into the
   prompt moves quality subsets by several points from one day to the next, so only same-day
   pairs count.

## Step 1 — Read what the fast path ships

From the model's `doc/fast_path/` (or the equivalent evidence when the work was done outside the
bring-up track), list:

- each switch: its variable, its values, the meshes it covers, and its default;
- which paths are opt-in and the measured reason (the trade-off);
- the KV blocks each mesh allocates with the fast path on, and the hard-limit evidence for any
  reduction;
- the drafter repo and pinned revision, the conditions under which speculation runs, and its
  measured speedup;
- the condition table: which request shapes take each path and what they fall back to.

If the acceptance evidence is missing or its check fails, stop: the fast path is not ready to ship.

## Step 2 — Plan the profiles

One profile per mesh and purpose. The shape gpt-oss-20b shipped on a QuietBox 2:

| Profile | Purpose | Switches in `env` |
|---|---|---|
| `p150x4-throughput` (default) | many users: larger prefill chunks | fused decode on |
| `p150x4` | interactive: shorter prefill chunks between decode steps | fused decode on |
| `p150x4-spec` | one user, opt-in | fused decode on, speculation on |
| `p150x2`, `p150` | interactive on two chips, one chip | that mesh's fused decode on |
| `p150x2-throughput`, `p150-throughput` | many users on that mesh | fused decode off where it cost multi-user throughput |

Each `description` states the purpose, the context limit, the maximum sequences, which decode path
runs, the KV capacity in tokens and blocks, and, for an opt-in profile, the trade-off and when the
fast path turns itself off. Keep `default_profile` on the profile that is best for most users.

## Step 3 — Build and prove every profile

Follow the tt-model-package-test skill, Steps 1 to 5, for every profile this box can run. Then add
the fast-path checks:

1. `tt-model profiles <id>` lists every planned profile and the default.
   `tt-model serve <id> --profile <p> --print` shows every switch in that profile's environment.
2. **KV capacity:** after each boot, read `KV cache size: <n> tokens` from `tt-model logs <id>` and
   write it into that profile's `description` (blocks = tokens / `block_size`). Compare it with
   the previous release's profile; a reduction needs the hard-limit evidence from Step 1.
3. **The path engaged:** for one request, the log or metrics show that the fast path served it,
   not the fallback. A server that answers is not proof that the fast path ran.
4. **Fallbacks:** on a fast-path profile, a request with log-probs, a repetition penalty, and
   structured output each succeed through the stock path.
5. **Speculation:** the drafter's files are in neither the image nor the package repo. The first
   boot of the speculative profile downloads the drafter and the second does not. The model code's
   fetch has an offline unit test for its Rule 5 failure message. At temperature 0, the
   speculative and plain profiles return identical text for at least 12 prompts of 256 tokens.
6. **Deterministic prefill (when it ships):** the same prompt three times on a fresh server gives
   identical output with the switch on.

## Step 4 — Release against release

Serve the previous release pinned (`tt-model serve <org>/<name>@<previous sha> --profile <p>`)
and the candidate one at a time on the same box, the same day. Compare a profile that is new in
this release with the previous release's profile for the same mesh and purpose.

- **Ladder**, per profile: input lengths 128 and 4096 (or the model's benchmark lengths) at
  concurrency 1, 8 and 32, with the same benchmark client and settings for both. Decode tokens per
  second per user and output tokens per second must be at least 97% of the previous release, and
  TTFT at most 103% of it plus 5 ms.
- **Quality:** full GSM8K and IFEval on the default profile, within 1.0 and 1.5 points of the
  previous release on identical prompts. Use a paired per-question test (McNemar) before calling a
  difference real. GSM8K-250 and IFEval-100 at concurrency 1 on each fast-path profile: a run at 32
  users barely exercises a batch-1 path. These must be within 5 points of the full-set score.
- **Smoke,** per profile: chat; the context limit (the longest prompt is accepted, one token over
  is rejected cleanly); a tool call when the manifest declares a `tool_parser`; reasoning
  separated when it declares a `reasoning_parser`.

Keep every number with the path of its raw output. A row that misses its bar blocks the release
until it is fixed or the user accepts the regression by name.

## Step 5 — Card and push

- The card shows each profile's description. Put a short table of the measured results against the
  previous release in the card description: one user, 32 users and TTFT per mesh. When comparing
  with another implementation, use matched weight formats or name the formats that differ.
- Pin every source the fast path needs in the manifest (the tt-metal commit, the plugin commit).
  Model code that ships in the package's `code/` says where the fast path lives.
- Push only when the user asks, as a new revision of the same repo (tt-model-package-test,
  Step 6). Users update with a plain `tt-model pull <org>/<name>`, never `--force`.

## Report

**Verified:** each profile served on which hardware; KV capacity per profile; the ladder and
quality tables against the previous release, with raw output paths; the drafter download, offline
error and identical-output checks. **Not verified:** profiles this box cannot run, and every check
that did not run. Never report a speedup without the matching previous-release row from the same
day.
