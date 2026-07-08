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
`transformers` and `torch` are needed when converting prompts with a tokenizer.

## Quick Checks

```bash
python3 -c "from plugins.kvcache_trace_plugin import block_hashes; print(block_hashes([1, 2, 3], block_size=64))"
kvcache-hit-rate --help
kvcache-simulator --help
```

## Example

```bash
kvcache-hit-rate simulate \
  --trace path/to/kvcache_trace.jsonl \
  --model qwen3-32b \
  --kv-precision fp8_int8 \
  --budgets-gib 1,2,4,8
```

See `plugins/README.md` and `packages/kvcache-simulator/README.md` for more details.
