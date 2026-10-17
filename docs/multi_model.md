# Running several models on one box

`tt-model deploy` serves several container packages side by side from one file. Each
instance gets its own chips and its own port, so every model has its own endpoint.

## The rule: one process per chip

A Tenstorrent chip can be opened by one process at a time. UMD takes a per-chip lock when
the device starts and holds it until the process exits. So the instances in a deployment
never share a chip. The planner gives each one:

- **its own chips.** A multi-chip profile gets whole boards. On a QuietBox 2 that means
  chips 0,1 or 2,3, never 1,2.
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
| `name` | instance name; also what `--name` takes in `stop`, `logs`, `curl` |
| `model` | a pulled container package (`org/name`) or a manifest path |
| `profile` | serve profile; default: the package's default |
| `chips` | list of chip ids, or `auto` (default) |
| `port` | host port, or `auto` (default, from 20000 up) |
| `args` | extra server arguments, as for `tt-model serve` |

More examples: [examples/deploy/](../examples/deploy).

## Commands

```bash
tt-model deploy up examples/deploy/gptoss-2x-p150x2.yaml --print   # show the placement only
tt-model deploy up examples/deploy/gptoss-2x-p150x2.yaml           # start all, wait until every one is ready
tt-model deploy status examples/deploy/gptoss-2x-p150x2.yaml       # which instances are up
tt-model ps                                                         # every server on the host
tt-model curl "hello" --name gptoss-b                               # talk to one instance
tt-model deploy down examples/deploy/gptoss-2x-p150x2.yaml         # stop all
```

`up` starts the instances one after another, then waits for all of them. If one fails to
start or to become ready, `up` stops the whole deployment, so you never get half of it.
`--keep-going` keeps the instances that did come up.
