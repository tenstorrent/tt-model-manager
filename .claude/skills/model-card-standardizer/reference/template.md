<!--
Status: PROPOSED, not finalized. Scoped in a comment on DEVSTACK-447
(https://tenstorrent.atlassian.net/browse/DEVSTACK-447, comment id 670207),
derived from surveying 10 of the ~50 live tt-model catalog cards plus the
GA freeze doc. Four open questions were logged on that ticket (tt-cli
readiness, license-as-manifest-field, where whitelist status lives, who
retrofits the existing catalog) — re-check the ticket before treating any
section below as settled. Do not silently drift this file out of sync with
the ticket; if the ticket's template changes, update this file to match.

Update: the "is tt-cli ready" question has since been investigated by
cloning tenstorrent/tt-cli. It's real, published on PyPI as the `tenstorrent`
package (binary `tt`) at v1.0.1, and `tt model pull`/`tt serve` already
install and drive `tt-model pull`/`tt-model serve` themselves — so the
Quickstart block below is safe to document as-is. Note `tt model pull` takes
no `--with-weights` flag (only --bundle/--weights-only/--offline); weights
come down by default for a bundle. Also: `tt report issue` is implemented,
but `tt report feedback` is still a stub that exits UNSUPPORTED, so the
Feedback section must not point at it.
-->

---
tags:
  - <hardware-tag>          # e.g. blackhole
  - <topology-tag>          # e.g. p150, p300x2 — one per supported profile
  - tt-model-cache
  - tt-model-catalog
  - tt-whitelisted           # add only once a DX-team reviewer has approved this model (DEVSTACK-423); omit for plain community packages
  - <runtime-tag>           # e.g. vllm-plugin, tt-dit-server — as applicable
  - <domain-tags>           # e.g. robotics, vla, tts, 3d-reconstruction — optional, aids discovery
license: <spdx-id-or-"other">          # omit only if the weights are genuinely unrestricted
license_name: <name>                    # required if license: other
license_link: <url-to-upstream-license>
pipeline_tag: <hf-pipeline-tag>         # e.g. text-generation, image-to-3d, robotics, text-to-speech
base_model:
  - <upstream-hf-repo-id>
tt_perf_summary:                        # structured mirror of the headline row(s) in "Expected performance" below —
  - metric: <e.g. tokens_per_sec_per_user>   # this is what DevHub/CLI actually parse for whitelist perf-gate
    value: <number>                          # evaluation; the markdown table further down is for humans only.
    unit: <e.g. tok/s>
    profile: <serve profile this was measured on>
---

# <repo-slug>

<One dense paragraph: what the model is, what it does, hardware/mesh it runs on, and any bring-up-maturity caveat (e.g. "experimental community bring-up").>

Packaged and published with [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) <version> (manifest schema <N>).

## At a glance

| | |
| --- | --- |
| Architecture | <e.g. 30B-parameter MoE decoder-only> |
| Hardware | <e.g. p300x2 — 4 Blackhole chips> |
| Context / input limits | <e.g. 262,144 tokens, or resolution/duration for non-text modalities> |
| License | <short name; see Licensing below for the full breakdown> |
| Status | <e.g. Experimental community bring-up / TT-whitelisted / Production> |
| Model CI v0 | <passed <date> — link to run, or "not yet run"> |

## Intended use

**Direct use:** <what this package is meant for.>
**Out-of-scope use:** <what it should not be used for, or what's explicitly unsupported — e.g. multimodal input on a text-only port.>

## Quickstart

```bash
tt model pull <repo-slug>
tt serve <repo-slug>
```

(`tt` is the `tenstorrent` PyPI package — confirmed live on PyPI at v1.0.1. `tt model pull`/`tt serve` already install and drive `tt-model` themselves, so this is safe to document today, not aspirational. There is no `--with-weights` flag on `tt model pull` — for a bundle it pulls the weights by default; that flag belongs to the lower-level `tt-model pull`.)

<Prose: what `tt model pull` downloads (image + weights, into the shared HF cache) — note there is no `--with-weights` flag on `tt`, weights come down by default for a bundle; default port; how long first-boot compile/conversion takes; the exact log line that signals readiness.>

## Serve profiles

| profile | hardware | mesh | <modality-specific capacity column(s), e.g. max_num_seqs / max_model_len for LLMs — omit the column entirely for non-LLM modalities rather than leaving it blank> |
| --- | --- | --- | --- |
| <name> | <e.g. p150> | <e.g. P150> | <value> |

## Using it

<For a chat/LLM model: OpenAI-compatible curl example against `/v1/chat/completions`, plus tool-calling and sampling-defaults notes if applicable.>

<For any non-chat modality: state explicitly "Not an OpenAI-compatible API" if a stub `/v1/models` exists for tooling, then document the real endpoint(s), exact request schema, and an example response — include a small demo image/asset if the output is visual.>

## Expected performance

<Mandatory. A table with accuracy-vs-reference (e.g. PCC, exact-match, or equivalent for the modality) AND throughput/latency numbers, plus the benchmark methodology (hardware, concurrency, dataset/prompt set) in one sentence. The headline numbers here must match the `tt_perf_summary` frontmatter field above — that's the copy tooling reads.>

## Limitations

<Mandatory. What's not implemented, known defects and their status, validated hardware/topology scope, anything declared but unvalidated (e.g. "profile X is declared but not validated").>

## Risks and safety considerations

<Recommended. Known failure modes with user-facing consequences (e.g. hallucination patterns, repetition loops), content or use cases this model should not be trusted for, and any TT-specific numerical/precision deviation from the reference that could silently affect correctness.>

## Licensing

<For weights carrying any restriction beyond the port code's own license: break out weights license, vendored upstream code license, and port/serving code license separately, each with non-commercial/redistribution terms stated plainly. Skip this section only when the base model's license is truly unrestricted.>

## Changelog

| date | change |
| --- | --- |
| <date> | <what changed and why, e.g. "fixed repetition-collapse under load; temperature>0 requests now sample host-side"> |

## Related packages

<Optional. Link sibling packages for the same base model on other hardware/precision/latency tradeoffs, and note explicitly if one supersedes another.>

## Feedback

Run `tt report issue` to file a problem with this package — it auto-collects environment details and opens a prefilled GitHub issue. For broader product feedback, <support@tenstorrent.com> gets tagged into the internal Jira board.

## Provenance

| component | built from |
| --- | --- |
| tt-metal | <commit link, or "local checkout — commit not published" plus a "dirty tree" flag if uncommitted changes are included> |
| <runtime, e.g. vLLM> | <version/tag or commit> |
| weights | <upstream repo id> @ <commit sha> |
| `code/` digest | <sha256, first 16 hex chars> |
| Model CI v0 | <passed/failed, date, link to run> |
| build | <timestamp> · tt-model-manager <version> |

<!-- Optional: ## Shutting down — include only for hardware where a hard kill (docker kill/rm -f) leaves the device unable to load a model until a host reboot. State the safe stop command and the risk explicitly. -->
