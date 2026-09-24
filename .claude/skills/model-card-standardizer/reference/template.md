<!--
Status: PROPOSED, not finalized. Scoped by the DX team, derived from surveying 10 of the ~50 live tt-model catalog cards plus the
GA freeze doc. Four open questions were logged internally (tt-cli
readiness, license-as-manifest-field, where whitelist status lives, who
retrofits the existing catalog) — re-check with the team before treating any
section below as settled. Do not silently drift this file out of sync with
the ticket; if the ticket's template changes, update this file to match.

Update: the "is tt-cli ready" question has since been investigated by
cloning tenstorrent/tt-cli. It's real, published on PyPI as the `tenstorrent`
package (binary `tt`) at v1.0.1, and `tt model pull`/`tt serve` already
install and drive `tt-model pull`/`tt-model serve` themselves. Note
`tt model pull` takes no `--with-weights` flag (only
--bundle/--weights-only/--offline); weights come down by default for a
bundle. Also: `tt report issue` is implemented but files against
tenstorrent/tt-cli (the TOOLING tracker, not the bundle), and
`tt report feedback` is still a stub that exits UNSUPPORTED.

Two decisions from code review (2026-09-18):
- The Quickstart shows BOTH flows, `tt` first. tt-model-manager's AGENTS.md
  invariant 1 ("tt-model alone must do the whole job; never introduce a step
  needing tt-cli") is binding, so a card that names only `tt` would violate
  it; the second fence is what keeps that invariant literally true.
- Whitelist status is NOT a repo tag, and it is not a field on this card at
  all. A tag lives in the author's own README frontmatter, which any author can
  edit on the Hub, so it could never be a review signal. Reviewing a bundle
  copies it into the Tenstorrent HF organisation instead: a model under
  `Tenstorrent/` has been reviewed by definition, because only the DX team can
  write there. Your own repo is never touched, and the copy credits you in its
  first line.
-->

---
tags:
  - <hardware-tag>          # e.g. blackhole
  - <topology-tag>          # e.g. p150, p300x2 — one per supported profile
  - tt-model-cache
  - tt-model-catalog
  #                          NO whitelist tag here: a reviewed model is one Tenstorrent has
  #                          copied into its own org, never something an author marks themselves.
  - <runtime-tag>           # e.g. vllm-plugin, tt-dit-server — as applicable
  - <domain-tags>           # e.g. robotics, vla, tts, 3d-reconstruction — optional, aids discovery
# NOTE: this block is the RENDERED card's frontmatter, which is flat because that is
# what the Hub indexes. In your tt-model.yaml you write it NESTED and the renderer
# flattens it:  card: {license: {id: apache-2.0, name: ..., link: ...}}
# `name`/`link` are emitted only beside `id: other`; the authoring model rejects an
# unknown key, so copying these three lines into the yaml verbatim will not load.
license: <spdx-id-or-"other">          # omit only if the weights are genuinely unrestricted
license_name: <name>                    # emitted only when license is "other"
license_link: <url-to-upstream-license> # emitted only when license is "other"
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
uv tool install tenstorrent   # once — the Tenstorrent CLI, `tt`
tt model pull <repo-slug>
tt serve <repo-slug>
```

<Prose: what `tt model pull` downloads (image + weights, into the shared HF cache) — there is no `--with-weights` flag on `tt`, weights come down by default for a bundle; default port; how long first-boot compile/conversion takes; the exact log line that signals readiness.>

Without tt-cli — tt-model alone does the whole job:

```bash
tt-model pull  <repo-slug> --with-weights
tt-model serve <repo-slug>
```

(Both fences are required. `tt` is the consumer path the GA doc asks for; the `tt-model` fence keeps tt-model-manager's AGENTS.md invariant 1 true — nothing on the card may *require* tt-cli.)

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

Questions or problems with this package: open a discussion at `https://huggingface.co/<repo-slug>/discussions` — the one channel that reaches the bundle's author. A problem with the `tt` tooling itself: `tt report issue` (collects your environment and opens a prefilled issue against tenstorrent/tt-cli — it does not reach this package's author). Product feedback: <support@tenstorrent.com> (tagged into the internal Jira board per the GA doc). Never `tt report feedback` — still a stub.

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
