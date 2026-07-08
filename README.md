# KV Cache Hit Rate

Standalone repository for KV cache trace generation and hit-rate simulation.

Note: this repository uses part of the code from
[kvcache-ai/kvcache-blog](https://github.com/kvcache-ai/kvcache-blog).

This repository contains:

- `plugins/`: dataset normalization, prompt-to-trace conversion, and a unified hit-rate plugin CLI.
- `packages/kvcache-simulator/`: local KV cache simulator package used by the plugin.

## Install

With pip:

```bash
python3 -m pip install -e .
```

With uv:

```bash
uv pip install -e .
```

This installs both the unified hit-rate CLI and the bundled local simulator package.
Prompt tokenization uses `transformers` and `sentencepiece`; PyTorch is not
installed by default to avoid pulling large CUDA packages.

## Quick Checks

```bash
python3 -c "from plugins.kvcache_trace_plugin import block_hashes; print(block_hashes([1, 2, 3], block_size=64))"
kvcache-hit-rate --help
kvcache-simulator --help
```

## Example

Convert a supported dataset directly into KV cache hit-rate results:

```bash
kvcache-hit-rate run \
  --input path/to/sharegpt.json \
  --dataset-format sharegpt \
  --trace-output /tmp/kvcache_trace_blksz64.jsonl \
  --model qwen3-32b \
  --kv-precision fp8_int8 \
  --budgets-gib 1,2,4,8 \
  --output /tmp/kvcache_hit_rate.json
```

`--dataset-format auto` can detect prompt JSONL, OpenAI `messages` /
`body.messages`, and ShareGPT `conversations`. If the input has no supported
records, the command exits with an unsupported dataset format message listing
the accepted shapes.

Hit-rate commands default to `--warmup-fraction 0`, so results are measured
from a cold cache over all requests. Use `--warmup-fraction 0.5
--exclude-underfilled` to reproduce the old last-half measurement window.
The JSON result also includes `hitRateCeiling`, the theoretical highest hit
rate when KV cache capacity is unlimited.

See `plugins/README.md` and `packages/kvcache-simulator/README.md` for more details.
