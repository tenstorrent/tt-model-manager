# Running several models on one box

`tt-model deploy` serves several container packages side by side from one file. Each
instance gets its own chips and its own port, so every model has its own endpoint.

## The rule: one process per chip

A Tenstorrent chip can be opened by one process at a time. UMD takes a per-chip lock when
the device starts and holds it until the process exits. So the instances in a deployment
never share a chip. The planner gives each one:

- **its own chips.** A multi-chip profile gets whole boards. On a QuietBox 2 that means
  chips 0,1 or 2,3, never 1,2.
  One-chip instances may share a board. Measured on a QuietBox 2 with two S2 Pro servers on
  chips 0 and 1: killing chip 0's server hard (`docker kill`) left chip 1's server answering,
  with byte-identical output before and after. A chip whose server was killed may need a
  `tt-smi -r` before its next use, and a reset of the whole box stops every server on it.
- **its own port**, skipping ports that are in use and ports other services on the box
  use (tt-studio, telemetry).
- **its own container name**, `tt-model-<name>-<profile>.<instance>`.

It refuses a plan that cannot run before anything starts:

- two instances on the same chip;
- a chip a running server already holds;
- a busy or duplicate port;
- a duplicate name;
- fewer free 1G hugepages than chips.

Models that must share chips have to share a process. That is a co-host package, not two
instances.

## The file

```yaml
deployment: gptoss-64u
instances:
  - {name: gptoss-a, model: jashansinghTT/gpt-oss-20b-blackhole-fast, profile: p150x2-throughput}
  - {name: gptoss-b, model: jashansinghTT/gpt-oss-20b-blackhole-fast, profile: p150x2-throughput}
```

| key | meaning |
|---|---|
| `name` | instance name; also what `--name` takes in `stop`, `logs`, `check` |
| `model` | a pulled container package (`org/name`) or a manifest path |
| `profile` | serve profile; default: the package's default |
| `chips` | list of chip ids, or `auto` (default) |
| `port` | host port, or `auto` (default, from 20000 up) |
| `args` | extra server arguments, as for `tt-model serve` |

More examples: [examples/deploy/](../examples/deploy).

## Example: 64 users of GPT-OSS-20B on a QuietBox 2

The file above serves 64 users as two endpoints of 32: one `p150x2-throughput` instance per
board. Measured (ISL 128, OSL 128, greedy, the package's own benchmark flags):

| layout | users | output tok/s | TPOT |
|---|---|---|---|
| one `p150x2-throughput` instance alone | 32 | 854 | 33.0 ms |
| two instances, both busy | 64 (32 each) | 1594 (797 + 797) | 34.4 ms |

Both instances gave the same greedy output as a normal `serve` of the package (16 of 16
prompts each). The two together are 7% below twice one alone; the host is the shared part
(both servers' CPU work runs on the same machine).

Two other layouts were tried and do not work today:

- **One instance with 64 users.** The model's vLLM class accepts at most 32 concurrent
  sequences and does not support TT data parallelism, so one endpoint cannot take 64 users.
- **Two 4-chip instances on the same chips** (in one co-host process, since two processes
  cannot share a chip). Each GPT-OSS-20B instance keeps about 366 KiB of L1 per core resident,
  so two hold about 732 KiB per core and the second one's programs no longer fit in L1.

## Commands

```bash
tt-model deploy up examples/deploy/gptoss-2x-p150x2.yaml --print   # show the placement only
tt-model deploy up examples/deploy/gptoss-2x-p150x2.yaml           # start all, wait until every one is ready
tt-model deploy status examples/deploy/gptoss-2x-p150x2.yaml       # which instances are up
tt-model ps                                                         # every server on the host
tt-model check "hello" --name gptoss-b                              # talk to one instance
tt-model deploy down examples/deploy/gptoss-2x-p150x2.yaml         # stop all
```

`up` starts the instances one after another, then waits for all of them at once. If one fails to
start or to become ready, `up` stops the whole deployment, so you never get half of it.
This holds for any start error and for Ctrl-C while it waits. `--keep-going` keeps the instances
that did come up, stops the ones that did not (even one still booting at the deadline), and exits 1.
