# KV Cache Hit Rate

Standalone repository for KV cache trace generation and hit-rate simulation.

This repository contains:

- `plugins/`: dataset normalization, prompt-to-trace conversion, and a unified hit-rate plugin CLI.
- `packages/kvcache-simulator/`: local KV cache simulator package used by the plugin.

## Install

```bash
python3 -m pip install -e packages/kvcache-simulator
python3 -m pip install -r plugins/requirements.txt
```

`transformers` and `torch` are only needed when converting prompts with a tokenizer.
Trace-only simulation can use the simulator package directly.

## Quick Checks

```bash
python3 -c "from plugins.kvcache_trace_plugin import block_hashes; print(block_hashes([1, 2, 3], block_size=64))"
python3 plugins/kv_cache_hit_rate_plugin.py --help
python3 -m kvcache_sim --help
```

## Example

```bash
python3 plugins/kv_cache_hit_rate_plugin.py simulate \
  --trace path/to/kvcache_trace.jsonl \
  --model qwen3-32b \
  --kv-precision fp8_int8 \
  --budgets-gib 1,2,4,8
```

See `plugins/README.md` and `packages/kvcache-simulator/README.md` for more details.
