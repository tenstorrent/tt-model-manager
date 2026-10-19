---
name: tt-model-multi-model
description: Serve several models on one Tenstorrent box at the same time, each on its own endpoint, without collisions — choose the layout (separate chips with `tt-model deploy`, or one co-host package when models must share chips), write the deploy file, place chips and ports, start, verify every endpoint against the model served alone, and stop cleanly. Use for "run these models together", "two models on one QuietBox", "serve 64 users", "TTS + STT on one box", "deploy several packages", or when two serves collide on chips or ports. Do NOT use to build a co-host package — use the tt-model-cohost-package skill.
---

# Serve several models on one box

Input: the models (published container packages, or manifest paths) and the box. Output: every
model answering on its own port, each verified against the same model served alone, and a
report of what was measured. **Never claim a layout works without the checks in Step 5.**

Details and measured numbers: [reference/layouts.md](reference/layouts.md) (QuietBox 2 layouts
and results) and [reference/troubleshooting.md](reference/troubleshooting.md) (failure → fix).

## The rule that decides everything

A Tenstorrent chip can be opened by **one process** at a time. UMD takes a per-chip lock when
the device starts and holds it until the process exits; there is no shared mode. So:

- models on **different chips** are separate servers → `tt-model deploy` (this skill);
- models that **must share chips** must share one process → a `kind: cohost` package
  (the `tt-model-cohost-package` skill). Two separately published packages can never share a chip.

Inside one co-host process the models take turns on the device; nothing runs at the same time.

## Step 1 — Pick the layout

| situation | layout |
| --- | --- |
| the models fit on separate chips (count each profile's chips) | one deploy file, one instance per model |
| many users of one model, more than one instance decodes at once | first measure one instance with the extra users queued (it can beat the others on throughput); a profile that decodes them all at once on one endpoint, if the package has one (`tt-model profiles <id>`); or several instances, one per board, each its own endpoint |
| two models that each need every chip, or more models than chips | a co-host package that already holds them (or build one) |
| a co-host package plus other models | one deploy file: the co-host package is one instance with several ports |

Check each package's profiles first: `tt-model profiles <org/name>` lists the chips (mesh) and
port each needs. Per-model limits that decide the layout live in the package's card (for example
GPT-OSS-20B decodes at most 32 sequences at once; for 64 users one 4-chip server that queues the
rest gave the most throughput, see reference/layouts.md).

## Step 2 — Place chips

- **A multi-chip instance gets whole boards.** On a QuietBox 2 (two P300 boards) that is chips
  `0,1` or `2,3` for 2 chips, `0,1,2,3` for 4; never `1,2`. The planner does this for `auto`.
- **One-chip instances may share a board.** Killing one board neighbour hard left the other
  answering, byte-identical (measured; see the reference).
- **Pin `chips:` on a shared box.** Under rootless Docker a chip held by another user's process
  can look free to the planner (their `/proc/<pid>/fd` is unreadable). Pin the chips you own.
- Leave room for 1G hugepages: one per chip in use.

## Step 3 — Write the deploy file

```yaml
deployment: speech-music-llm
instances:
  - {name: tts,   model: jashansinghTT/s2-pro-blackhole,         profile: p150,   chips: [0],    port: 20010}
  - {name: music, model: jashansinghTT/minimax-music3-blackhole, profile: p150,   chips: [1],    port: 20011}
  - {name: llm,   model: jashansinghTT/gpt-oss-20b-blackhole-fast, profile: p150x2, chips: [2, 3], port: 20012}
```

| key | meaning |
| --- | --- |
| `name` | instance name; what `--name` takes in `stop`, `logs`, `curl` |
| `model` | a pulled container package (`org/name`) or a manifest path |
| `profile` | serve profile; default: the package's default |
| `chips` | list of chip ids, or `auto` (default) |
| `port` | host port, or `auto` (default, from 20000 up) |
| `args` | extra server arguments, as a list, as for `tt-model serve` |

A co-host instance publishes several ports (its members' ports plus a status port): give it a
base port with room for the whole block.

## Step 4 — Preview, then start

```bash
tt-model pull <each package>                       # image and weights first, so `up` only boots
tt-model deploy up deploy.yaml --print             # placement only: chips, ports, container names
tt-model deploy up deploy.yaml                     # start all, wait until every one is ready
```

`up` refuses a plan before anything starts: two instances on one chip, a chip a running server
holds, a busy or duplicate port, a duplicate name, too few free hugepages. If any instance fails
to start or to become ready, `up` stops the whole deployment (also on Ctrl-C); `--keep-going`
keeps what came up, stops the rest and exits 1.

## Step 5 — Verify (every endpoint, every time)

1. `tt-model ps` and `tt-model deploy status deploy.yaml`: every instance up, on the chips and
   ports you expect.
2. **Same output as alone.** For each instance, send fixed requests and compare with the same
   model served alone on the same profile: LLMs greedy (`temperature 0`) on a fixed prompt set,
   identical text (16 of 16 prompts is the bar used so far); audio models with a fixed `seed`,
   byte-identical files. Then send all instances their requests at the same time and compare
   again.
3. **Throughput with the package's own benchmark flags** (greedy, exact input length, warmup).
   An unseeded or sampled benchmark can read several times too slow. Measure each instance
   alone, then all busy at once; report both.
4. `tt-model curl "hello" --name <instance>` for a quick check of one instance.

## Step 6 — Stop

```bash
tt-model deploy down deploy.yaml        # or: tt-model stop <org/name> --name <instance>
```

If a server does not exit on SIGTERM, `stop` kills it and resets its chips. A reset of only some
chips can leave the next mesh that spans the others unable to start (`Fabric Router Sync:
Timeout`); a current `tt-model stop` also resets the chips nothing holds and names any it left
out. Before a 4-chip serve after such a warning, reset every chip once nothing else uses them.

## Report

Per instance: package, profile, chips, port, ready time, the Step 5 comparison (for example
"16/16 identical", "byte-identical"), throughput alone and all busy. Say which checks were not run.
