# Layouts measured on a QuietBox 2

A QuietBox 2 has two P300 boards with two Blackhole chips each: chips `0,1` and `2,3`. Every
number below is from one run on such a box; greedy LLM checks use 16 fixed prompts, audio checks
a fixed seed.

## 64 users of one 4-chip LLM (GPT-OSS-20B)

One server decodes at most 32 sequences at once; vLLM queues the rest. Three ways to serve 64
users, measured on the same box (ISL 128, OSL 128, greedy, exact input length, one warmup):

| layout | endpoints | 64 users | pick it when |
| --- | --- | --- | --- |
| one `p150x4-throughput` instance (32 decode, 32 queue) | 1 | **1795 tok/s**; time to first token median 2.7 s, whole answer median 4.6 s | total throughput and median answer time matter most; requests 33-64 wait for a free slot |
| one `p150x4-dp2` instance (two 2-chip lanes, 64 decode at once) | 1 | 1497 tok/s; first token median 1.1 s, whole answer median 5.5 s | every request must start streaming at once on one endpoint |
| two `p150x2-throughput` instances, one per board, via `tt-model deploy` | 2 | 1594 tok/s (797 + 797); both 16/16 greedy vs a plain `serve`; up in 3 min 12 s | two independent servers (one restarts without the other); the client splits the users |
| two 4-chip replicas on the same chips (one co-host process) | - | does not fit: each keeps about 366 KiB/core of L1 resident (`L1 buffer allocated at ... and static circular buffer region ends at ...`) | never |

With long answers (ISL 128, OSL 1024) at 64 users the gap in waiting grows: `p150x4-throughput`
1938 tok/s but first token median 17.5 s (half the users wait for a whole answer first);
`p150x4-dp2` 1593 tok/s with first token median 1.1 s. The two `p150x2-throughput` instances
together are 7% below twice one alone (854 tok/s at 32 users); the shared part is the host CPU.

One endpoint: `tt-model serve jashansinghTT/gpt-oss-20b-blackhole-fast --profile p150x4-throughput`
(or `--profile p150x4-dp2`). Two endpoints:

```yaml
deployment: gptoss-64u
instances:
  - {name: gptoss-a, model: jashansinghTT/gpt-oss-20b-blackhole-fast, profile: p150x2-throughput}
  - {name: gptoss-b, model: jashansinghTT/gpt-oss-20b-blackhole-fast, profile: p150x2-throughput}
```

## Three packages on separate chips

S2 Pro (TTS) on chip 0, MiniMax Music 3 on chip 1, GPT-OSS-20B `p150x2` on chips 2,3: all three up
in 209 s and answering at the same time; TTS and music byte-identical and the LLM 16/16 identical
to each instance deployed alone on the same chips.

## Seven endpoints from one deploy file

The speech co-host package (Whisper + S2 Pro + Music 3 + status, four ports) on chip 0, S2 Pro on
chip 1, GPT-OSS-20B `p150x2` on chips 2,3: up in 197 s, every output identical to the same model
alone, clean `down`. The co-host instance needs its whole port block free (base port + one per
member + status).

## Two models that both need all four chips

GPT-OSS-20B with Llama-3.1-8B, or with Qwen3-32B: one co-host package (both on chips 0-3, one
process). Each model 16/16 vs alone, also with both answering at once. At 32 users each:

| pair | each alone | both busy |
| --- | --- | --- |
| GPT-OSS + Llama-3.1-8B | 1444 / 1128 tok/s | 813 + 778 = 1591 tok/s |
| GPT-OSS + Qwen3-32B | 1455 / 500 tok/s | 461 + 431 = 892 tok/s |

The models take turns, so a slow model sets the pace for both.

## Board neighbours

Two one-chip S2 Pro servers on chips 0 and 1 (one board): `docker kill` of chip 0's server left
chip 1's answering, byte-identical before and after. The killed chip may need a reset before its
next use; a reset of the whole box stops every server on it.

## Not possible today

- A second model "in the leftover capacity" of chips already used by a 4-chip model, as a separate
  process: one process per chip. Inside one process, a submesh on the shared chips has its own
  allocator (the models' buffers collide); the 4-chip model's resident L1 is on every chip.
- Splitting chips into sub-devices so two models run at the same time: measured 0.88-1.17x for a
  realistic op pair against good whole-chip configs, and every op of a model would have to be
  confined to its cores.
