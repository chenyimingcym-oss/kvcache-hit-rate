# CLI Parameter Reference

This document summarizes the command-line parameters for the repository's two
entry points:

- `kvcache-hit-rate`: dataset normalization, trace generation, simulation, and plotting.
- `kvcache-simulator`: simulation and plotting for an existing trace.

Use `--help` on any command for the exact parser output.

## Command Overview

| Command | Purpose |
| --- | --- |
| `kvcache-hit-rate normalize` | Convert a dataset into normalized prompt JSONL. |
| `kvcache-hit-rate convert` | Convert prompts or chat records into simulator trace JSONL. |
| `kvcache-hit-rate simulate` | Run hit-rate simulation over an existing trace. |
| `kvcache-hit-rate run` | Convert a dataset and run simulation in one command. |
| `kvcache-hit-rate plot` | Plot hit-rate curves from a simulation JSON result. |
| `kvcache-simulator run` | Run hit-rate simulation over an existing trace. |
| `kvcache-simulator sweep` | Alias for `kvcache-simulator run`. |
| `kvcache-simulator plot` | Plot hit-rate curves from a sweep JSON result. |
| `kvcache-simulator list-models` | List supported model ids. |

## Dataset And Trace Input

| Parameter | Commands | Meaning |
| --- | --- | --- |
| `--input PATH` | `kvcache-hit-rate normalize`, `run`, `plot` | Input dataset path for `normalize`/`run`; simulation JSON path for `plot`. |
| `--trace PATH` | `kvcache-hit-rate simulate`, `kvcache-simulator run/sweep` | Existing trace path. Supports `.jsonl`, `.jsonl.gz`, or `-` for stdin in `kvcache-simulator`. |
| `--trace-output PATH` | `kvcache-hit-rate run` | Optional path for saving the intermediate trace JSONL created before simulation. |
| `--output PATH` | Most commands | Output path. For simulator `run/sweep`, `-` means stdout. |
| `--dataset-format FORMAT` | `kvcache-hit-rate normalize`, `convert`, `run` | Input dataset format: `auto`, `jsonl`, `openai`, `sharegpt`, or `text`. |
| `--drop-messages` | `kvcache-hit-rate normalize` | Omit normalized `messages` fields and keep only prompt text. |
| `--strict` | `kvcache-hit-rate normalize`, `convert`, `run` | Stop on the first invalid record instead of skipping unsupported records. |

Supported dataset formats:

- `auto`: detect each record.
- `jsonl`: records with `prompt`, `text`, or `messages`.
- `openai`: records with top-level `messages` or OpenAI Batch-style `body.messages`.
- `sharegpt`: records with `conversations`, mapping `human` to `user` and `gpt` to `assistant`.
- `text`: JSON string records or arbitrary JSON rendered as text.

## Tokenization And Trace Generation

| Parameter | Commands | Meaning |
| --- | --- | --- |
| `--tokenizer NAME_OR_PATH` | `kvcache-hit-rate convert`, `run` | Hugging Face tokenizer id or local tokenizer path. Required for `convert`. For `run`, omitted values default to the tokenizer repo inferred from required `--model`; explicit values override that inference. |
| `--block-size N` | `kvcache-hit-rate convert`, `run`; `kvcache-simulator run/sweep` | Token block size used for trace generation or as a fallback when trace records omit `block_size`. Trace-declared `block_size` overrides the fallback. |
| `--use-chat-template` | `kvcache-hit-rate convert`, `run` | Render `messages` records with the tokenizer chat template before tokenization. |
| `--trust-remote-code` | `kvcache-hit-rate convert`, `run` | Pass `trust_remote_code=True` when loading the tokenizer. Use only with trusted tokenizer repositories. |
| `--local-files-only` | `kvcache-hit-rate convert`, `run` | Load tokenizer files only from local cache or local paths. |

## Simulation

| Parameter | Commands | Meaning |
| --- | --- | --- |
| `--model ID` | Simulation commands | Required model id from the bundled model catalog, or a known alias such as a catalog label or Hugging Face repo id. Use `kvcache-simulator list-models` to inspect supported ids. |
| `--kv-precision ID` | Simulation commands | KV cache precision, usually `bf16_fp16`, `fp8_int8`, or `fp4_int4`. |
| `--indexer-precision ID` | Simulation commands | Indexer cache precision for models that define an indexer cache. |
| `--include-draft-kv-cache` | Simulation commands | Include draft/MTP KV cache layers when supported by the selected model. |
| `--estimate-tokens N` | Simulation commands | Override the token count used for token-dependent bytes/token accounting. Defaults to the trace average input length. |
| `--budgets-gib A,B,C` | Simulation commands | Comma-separated finite KV cache memory budgets in GiB. |
| `--policies fifo,lru,optimal` | Simulation commands | Eviction policies to simulate. Defaults to all supported policies. |
| `--backend python\|cpp` | Simulation commands | Simulation backend. Python is the default; C++ is faster for fixed-window sweeps. |
| `--jobs N` | Simulation commands | Worker process count for the Python backend. Ignored by the C++ backend. |
| `--warmup-fraction X` | Simulation commands | Fraction of requests to skip before measuring hit rate. `0` measures all requests from a cold cache. |
| `--exclude-underfilled` | Simulation commands | Omit budgets where the cache was not full before the measurement window. |
| `--models-yaml PATH` | `kvcache-simulator run/sweep`, `list-models`; `kvcache-hit-rate simulate/run` | Override the bundled model catalog. `kvcache-hit-rate run` also uses this file when inferring the tokenizer repo from `--model`. |

`--warmup-fraction 0` with underfilled budgets included is the default global
cold-start measurement. Use `--warmup-fraction 0.5 --exclude-underfilled` to
match the older last-half fixed-window behavior.

## Plotting

| Parameter | Commands | Meaning |
| --- | --- | --- |
| `--plot-output PATH` | Simulation commands | Write a PNG/SVG/PDF hit-rate plot while running simulation. |
| `--input PATH` | Plot commands | Simulation JSON result to plot. |
| `--output PATH` | Plot commands | Plot image path. The format follows the file extension. |
| `--title TEXT` | Plot commands | Optional chart title. |

The plot x-axis is finite KV cache budget in GiB. The y-axis is KV Cache Hit
Rate formatted as a percentage. Unlimited KV cache capacity is drawn as a
horizontal `Unlimited / ceiling` line using `hitRateCeiling`, not as a regular
budget point.

Install plotting support before using plot commands:

```bash
pip install ".[plot]"
```

or, for the simulator package:

```bash
pip install "kvcache-simulator[plot]"
```

## Output And Debug Limits

| Parameter | Commands | Meaning |
| --- | --- | --- |
| `--format table\|json` | `kvcache-simulator run/sweep` | Render a readable table or raw JSON. |
| `--no-progress` | `kvcache-simulator run/sweep` | Disable terminal progress output. |
| `--max-records N` | Most data and simulation commands | Stop after N valid records. `0` means no limit. |
| `--max-events N` | Simulation commands | Stop after N trace block events. `0` means no limit. |

## Common Examples

Convert a ShareGPT dataset and run simulation:

```bash
kvcache-hit-rate run \
  --input sharegpt.json \
  --dataset-format sharegpt \
  --trace-output /tmp/kvcache_trace.jsonl \
  --model Qwen/Qwen3.6-27B \
  --models-yaml /path/to/custom-models.yaml \
  --kv-precision fp8_int8 \
  --budgets-gib 1,2,4,8 \
  --output /tmp/kvcache_hit_rate.json
```

Run simulation and write a percentage hit-rate plot:

```bash
kvcache-simulator run \
  --trace trace.jsonl.gz \
  --model qwen3-32b \
  --kv-precision fp8_int8 \
  --plot-output hit-rate.png
```

Plot from an existing result:

```bash
kvcache-simulator plot \
  --input result.json \
  --output hit-rate.png
```
