# KV Cache Hit Rate Plugin

This directory is a standalone plugin wrapper for the current KV cache hit-rate
workflow:

1. request prompt -> token ids
2. token ids -> prefix-aware block hashes
3. JSONL/OpenAI/ShareGPT datasets -> normalized prompt JSONL
4. prompt JSONL -> simulator trace JSONL
5. trace JSONL -> KV cache hit-rate sweep

See `../PARAMETERS.md` for the consolidated CLI parameter reference.

The block hash algorithm matches `../test/prompts_to_kvcache_trace.py`.

## Python API

```python
from kv_cache_hit_rate_plugin import KVCacheHitRatePlugin

plugin = KVCacheHitRatePlugin(local_files_only=True)
token_ids = plugin.prompt_to_token_ids("hello")
hash_ids = plugin.token_ids_to_block_hashes(token_ids)
result = plugin.simulate_trace(
    "../test/kvcache_trace_blksz64.sample.jsonl",
    model_id="qwen3-32b",
    precision="fp8_int8",
)
```

For direct token-id input, `block_hashes` has no tokenizer dependency:

```python
from kvcache_trace_plugin import block_hashes

hash_ids = block_hashes([1, 2, 3, 4], block_size=64)
```

Normalize OpenAI or ShareGPT style datasets before tokenization:

```python
from dataset_converters import convert_dataset

convert_dataset(
    "sharegpt.json",
    "/tmp/prompts.jsonl",
    dataset_format="sharegpt",
)
```

## CLI

Normalize a dataset to prompt JSONL:

```bash
python3 kv_cache_hit_rate_plugin.py normalize \
  --input sharegpt.json \
  --output /tmp/prompts.jsonl \
  --dataset-format sharegpt
```

Convert prompt JSONL into trace JSONL:

```bash
python3 kv_cache_hit_rate_plugin.py convert \
  --input ../test/prompts.jsonl \
  --output /tmp/kvcache_trace_blksz64.jsonl \
  --tokenizer Qwen/Qwen3.6-27B \
  --block-size 64 \
  --local-files-only
```

Convert OpenAI chat/messages or ShareGPT data directly into trace JSONL:

```bash
python3 kv_cache_hit_rate_plugin.py convert \
  --input openai_messages.jsonl \
  --output /tmp/kvcache_trace_blksz64.jsonl \
  --dataset-format openai \
  --tokenizer Qwen/Qwen3.6-27B \
  --use-chat-template \
  --block-size 64 \
  --local-files-only
```

Run hit-rate simulation over an existing trace:

```bash
python3 kv_cache_hit_rate_plugin.py simulate \
  --trace ../test/kvcache_trace_blksz64.sample.jsonl \
  --model qwen3-32b \
  --kv-precision fp8_int8 \
  --budgets-gib 1,2,4,8 \
  --output /tmp/kvcache_hit_rate.json
```

Convert prompts and simulate in one command:

```bash
python3 kv_cache_hit_rate_plugin.py run \
  --input ../test/prompts.jsonl \
  --trace-output /tmp/kvcache_trace_blksz64.jsonl \
  --model qwen3-32b \
  --kv-precision fp8_int8 \
  --budgets-gib 1,2,4,8 \
  --local-files-only \
  --output /tmp/kvcache_hit_rate.json
```

Write a hit-rate plot during simulation:

```bash
python3 kv_cache_hit_rate_plugin.py simulate \
  --trace ../test/kvcache_trace_blksz64.sample.jsonl \
  --model qwen3-32b \
  --kv-precision fp8_int8 \
  --budgets-gib 1,2,4,8 \
  --plot-output /tmp/kvcache_hit_rate.png
```

Or plot from an existing simulation JSON:

```bash
python3 kv_cache_hit_rate_plugin.py plot \
  --input /tmp/kvcache_hit_rate.json \
  --output /tmp/kvcache_hit_rate.png
```

The plot uses finite KV cache budget as the x-axis and formats KV Cache Hit
Rate as percentages on the y-axis. Unlimited capacity is drawn as a horizontal
ceiling/asymptote line instead of a regular budget point.

The unified plugin defaults to cold-start/global hit-rate semantics:
`--warmup-fraction 0` and underfilled budgets included. Use
`--warmup-fraction 0.5 --exclude-underfilled` to match the web simulator's
default measurement window. Simulation output includes `hitRateCeiling`, the
theoretical highest prefix-cache hit rate with unlimited KV cache capacity.

Input prompt JSONL records should contain either a `prompt` string, or a
`messages` list when `--use-chat-template` is set.

Supported `--dataset-format` values:

- `auto`: detect each record.
- `jsonl`: records with `prompt`, `text`, or `messages`.
- `openai`: records with top-level `messages` or OpenAI Batch-style
  `body.messages`.
- `sharegpt`: records with `conversations`, mapping `human` to `user` and
  `gpt` to `assistant`.
- `text`: JSON string records or arbitrary JSON rendered as text.
