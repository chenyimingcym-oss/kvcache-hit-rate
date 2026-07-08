# KV Cache Simulator

`kvcache-simulator` runs the KVCache.AI hit-rate simulator locally on JSONL traces. It uses the same model accounting formulas as the KV Cache Size Calculator and the same prefix-aware hit-rate semantics as the KV Cache Hit Rate Simulator.

## Installation

```bash
pip install kvcache-simulator
```

The default simulation backend is Python so the command can report cold-start
global hit rate from an empty cache. Use `--backend cpp --exclude-underfilled`
when you want the bundled C++ replay core; it compiles on first use and needs a
local C++ compiler such as `c++` or `clang++`.

## Quick Start

```bash
kvcache-simulator run \
  --trace trace.jsonl.gz \
  --model glm-5.2 \
  --kv-precision fp8_int8 \
  --indexer-precision fp4_int4
```

The default output is a readable table. Use `--format json` when another script needs to consume the result.

```bash
kvcache-simulator run \
  --trace trace.jsonl.gz \
  --model deepseek-v4-pro \
  --kv-precision fp8_int8 \
  --indexer-precision fp4_int4 \
  --format json \
  --output result.json
```

List supported model ids:

```bash
kvcache-simulator list-models
```

`python -m kvcache_sim ...` also works, but the installed CLI command is preferred.

`run` is the main command. It evaluates the selected trace across a set of KV cache memory budgets. `sweep` is kept as an alias for users who prefer benchmark terminology.

## Input Trace Format

Input is JSONL or JSONL.GZ, one request per line. The minimal accepted format is:

```json
{"block_size":64,"hash_ids":[2001,2002],"input_length":128}
```

Required fields:

- `hash_ids`: cache block identities in request-prefix order.
- `input_length`: prefill input token count for this request.
- `block_size`: source-native block size. It can be omitted only when `--block-size` is provided.

Optional fields:

- `timestamp`: ignored by the simulator. Requests are replayed in file order, so sort production traces by timestamp before running the command.
- `output_length`: ignored by the hit-rate denominator. Generated output matters only if it appears later in another request's `hash_ids`.
- `block_tokens`: advanced field for exact per-block token weights. If present, it must be a positive integer list with the same length as `hash_ids`.

`--block-size` is only a fallback for records that omit `block_size`. If any record declares `block_size`, the trace-declared value is used and overrides the CLI fallback for the whole trace.

## Options

| Option | Meaning |
| --- | --- |
| `--trace PATH` | JSONL/JSONL.GZ trace path, or `-` for stdin. |
| `--model ID` | Model id from the bundled KV Cache Size Calculator model catalog. Use `kvcache-simulator list-models` to list ids. |
| `--kv-precision ID` | KV cache precision: usually `bf16_fp16`, `fp8_int8`, or `fp4_int4`. Defaults follow the web calculator. |
| `--indexer-precision ID` | Indexer cache precision for models with an indexer cache, such as DeepSeek V4 / GLM / MiniMax M3. |
| `--include-draft-kv-cache` | Include draft/MTP KV layers when the selected model defines them. Default is off. |
| `--block-size N` | Fallback block size when trace records omit `block_size`; trace-declared `block_size` overrides it. |
| `--estimate-tokens N` | Override the token count used for token-dependent bytes/token formulas. By default the trace average input length is used. |
| `--budgets-gib A,B,C` | Comma-separated KV cache memory budgets in GiB. Default matches the web budget sweep: `1,2,4,...,16384`. |
| `--policies fifo,lru,optimal` | Eviction policies to simulate. Defaults to all three. |
| `--backend cpp\|python` | Simulation backend. Default is `python`; use `cpp` with `--exclude-underfilled` for faster fixed-window sweeps. |
| `--jobs N` | Number of worker processes for the Python backend. The C++ backend runs one batch process and ignores this option. |
| `--warmup-fraction X` | Fraction of requests to skip before measuring hit rate. Default is `0`, which measures all requests from a cold cache. |
| `--exclude-underfilled` | Omit budgets where the cache was not full before measurement. Use with `--warmup-fraction 0.5` to match the old web-style measurement window. |
| `--no-progress` | Disable terminal progress output. Progress is written to stderr only when stderr is interactive, so JSON stdout stays valid. |
| `--format table\|json` | Output format. Default is `table`. |
| `--output PATH` | Write output to a file. Default `-` prints to stdout. |
| `--max-records N` | Debug/testing limit: stop after N valid requests. |
| `--max-events N` | Debug/testing limit: stop after N trace blocks. |

## Output Semantics

- Hit rate is measured over all requests by default (`--warmup-fraction 0`).
- Underfilled budget points are included by default, so cold-start/global hit
  rate is reported even when the cache is not full before the first request.
- Every result includes `hitRateCeiling`, the theoretical highest prefix-cache
  hit rate with unlimited KV cache capacity.
- Hit tokens count only the longest continuous cached prefix of each request. If a middle block misses, later blocks in that same request do not count as prefill hits even if their ids are already cached.
- `speedup` is an ideal prefill-only upper bound: `1 / (1 - hit_rate)`. `1.0x` means no-cache prefill throughput where every prefill input token is computed. It does not include decode, KV lookup, network, batching, scheduling, or memory bandwidth overhead.

## Performance Notes

The C++ backend runs all cache-budget points in one batch after loading the
trace and building the prefix trie once. This is usually faster than the Python
backend, especially on large traces.

`--jobs` applies only to the Python backend. It parallelizes independent `(policy, cache budget)` simulation tasks. More jobs are not always faster: the default budget sweep has only a small number of budget points, tasks have uneven runtimes, and large traces can become memory-bandwidth limited.

## Bundled C++ Core

Most users should call `kvcache-simulator run` instead of invoking the bundled
C++ binary directly. The Python wrapper writes compact binary trace files,
compiles the bundled source when needed, and calls the C++ core in batch mode:

```bash
kv-cache-lab-native-sim \
  --policy batch \
  --ids ids.u32 \
  --tokens tokens.u16 \
  --request-ends request_ends.u32 \
  --request-count 1000 \
  --total-blocks 64000 \
  --warmup-requests 0 \
  --capacities 1024,2048,4096 \
  --policies fifo,lru,optimal \
  --progress
```

Supported direct C++ policies are `fifo`, `lru`, `optimal`, `all`, `ceiling`,
`batch`, and the maintenance command `build-next`. In batch mode:

- `--capacities A,B,C` is a comma-separated list of cache capacities in blocks.
- `--policies fifo,lru,optimal` selects which eviction policies to emit.
- `--progress` writes `KV_PROGRESS done total label` lines to stderr.
- The JSON output contains `ceiling` plus one entry in `points` for each
  non-negative capacity.

For single-policy C++ runs, omit `--capacity` or pass `--capacity -1` to report
the unlimited-capacity theoretical ceiling directly:

```bash
kv-cache-lab-native-sim \
  --policy lru \
  --ids ids.u32 \
  --tokens tokens.u16 \
  --request-ends request_ends.u32 \
  --request-count 1000 \
  --total-blocks 64000 \
  --warmup-requests 0
```

That command returns a `policy: "ceiling"` JSON object with `cacheBlocks: -1`.
Passing `--policy ceiling` is equivalent.

## Limits

The browser version caps uploads to protect the UI. This local package does not use that browser cap, but the C++ backend currently stores trace events and prefix node indexes in 32-bit arrays, so it supports at most `2^32 - 1` block events. In practice, memory and runtime are usually the real limits before that.
