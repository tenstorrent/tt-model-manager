# DRAM and L1 budgets, fit hooks, measured numbers

All measured on a QuietBox 2 (4x Blackhole p300c) or one of its chips.

## How to measure

- `GET /` on the status port: `dram_gib_per_chip.members.<name>.<phase>` (what each member's load,
  warmup and capture added), `dram_gib_per_chip.total`, `l1_kib_per_core_after_phase.<name>`
  (L1 held after that member's phases), `compiles_after_capture.<name>`.
- The host log prints the same per phase: `cohost: <member> <phase>: +X GiB/chip DRAM (total Y);
  L1 held Z KiB/core`.
- `TT_COHOST_L1_REPORT=1`: after every phase, log each live L1 buffer on the first chip
  (address + size per bank): which buffers hold the addresses another member's programs need.
- Boot each member alone in the group runtime first. A member that fits alone and fails in the
  group is an L1 or DRAM budget problem, not a model bug.

## DRAM per chip (4 chips)

| member | weights | KV cache |
| --- | --- | --- |
| GPT-OSS-20B, fused batch-1 decode | 12.78 GiB | 3.29 GiB (16,512 blocks of 64) |
| GPT-OSS-20B, `GPT_OSS_20B_FUSED_B1=0` | 7.10 GiB | 3.29 GiB |
| Llama-3.1-8B-Instruct | 1.73-1.85 GiB | 1.06 GiB (65,536 tokens); 2.13 GiB (131,072) |

Group totals: GPT-OSS + Llama (131,072-token pool) 14.5 GiB/chip; GPT-OSS + Qwen3-32B (65,536-token
pool, GPT-OSS pool 4,128 blocks) 16.6 GiB/chip. Context length is a DRAM cost only: Llama at
131,072 tokens held the same L1 as at 4,096 and answered a 64,000-token prompt (2,048-token
prefill chunks).

One chip (speech group): Whisper large-v3 3.9 GiB, S2 Pro 7.3 GiB, MiniMax Music 3 18.3 GiB at 8K
context = 29.5 GiB + a 512 MiB trace region. At Music 3's 16K context the three need 31.9 GiB and
the group runs out of DRAM in the last member's warmup.

## L1 per core (the binding budget)

- GPT-OSS-20B on 4 chips keeps about 366-381 KiB/core resident (CCL semaphores and persistent
  buffers); its fused batch-1 decode keeps more. Two GPT-OSS replicas on the same chips hold about
  732 KiB/core and the second one's programs fail.
- tt_transformers Llama-3.1-8B's default 4-chip prefill needs circular buffers up to about
  1.2 MiB: its prefill RMSNorm gathers the full hidden size, because tt_transformers distributes
  the prefill norm only for models wider than 4096 (Llama-8B is exactly 4096).
- The speech group on one chip holds about 33 KiB/core between requests.

## Fit hooks that worked

| member | change | effect |
| --- | --- | --- |
| GPT-OSS-20B | `GPT_OSS_20B_FUSED_B1=0` in its `env` | frees the fused path's resident L1; 1444 vs 1753 tok/s at 32 users, about 10.6 ms per token for one user |
| Llama-3.1-8B (tt_transformers) | `preload` hook that turns on the distributed prefill RMSNorm for the member | fits next to GPT-OSS. Numerics change slightly (another reduction order): GSM8K-chat 0.784 vs 0.784, IFEval prompt-strict 0.724 vs 0.740 (250 prompts, standard error about 0.028). Take its baseline with the hook on |
| Llama / Qwen (tt_transformers) | the same hook drops the plugin kwargs (`new_request_rows`) the model's chunked or prefix-cached prefill does not take | fixes a `TypeError` on the second chunk or a prefix-cache hit |
| any vLLM member | `kv_tokens` | caps the member's KV pool so the members' DRAM adds up |
| MiniMax Music 3 | `max_seq_len` 16384 → 8192, max frames 9000 → 6000 | fits next to Whisper and S2 Pro on one chip; songs up to 4 min; a seeded clip is byte-identical at 8K and 16K |
| tt_transformers KV caches (Music 3) | do not file-cache the all-zero KV tensors | avoids a cold-boot hang (see failures.md) |

## Throughput when members share the device

Members take turns, so when both are busy each gets roughly half the device and the slower
member's steps set the pace:

| group, 32 users each, ISL/OSL 128 | each alone | both busy |
| --- | --- | --- |
| GPT-OSS + Llama-3.1-8B | 1444 / 1128 tok/s | 813 + 778 |
| GPT-OSS + Qwen3-32B | 1455 / 500 tok/s | 461 + 431 |
