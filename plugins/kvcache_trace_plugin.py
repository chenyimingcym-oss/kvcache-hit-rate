#!/usr/bin/env python3
"""KV cache trace plugin.

The block hash implementation intentionally matches
test/prompts_to_kvcache_trace.py: prefix-aware 64-bit blake2b hashes over
little-endian uint32 token ids.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

try:
    from .dataset_converters import (
        DATASET_FORMATS,
        iter_dataset_records,
        normalize_dataset_record,
    )
except ImportError:
    from dataset_converters import (
        DATASET_FORMATS,
        iter_dataset_records,
        normalize_dataset_record,
    )


DEFAULT_BLOCK_SIZE = 64
DEFAULT_TOKENIZER = "moonshotai/Kimi-K2.6"


def block_hashes(token_ids, block_size=DEFAULT_BLOCK_SIZE):
    """Return prefix-aware 64-bit block hashes for token ids."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    running = hashlib.blake2b(digest_size=8)
    ids = []
    for start in range(0, len(token_ids), block_size):
        block = token_ids[start : start + block_size]
        running.update(struct.pack(f"<{len(block)}I", *(t & 0xFFFFFFFF for t in block)))
        ids.append(int.from_bytes(running.copy().digest(), "little"))
    return ids


def load_tokenizer(
    name_or_path=DEFAULT_TOKENIZER,
    trust_remote_code=False,
    local_files_only=False,
):
    """Load a Hugging Face tokenizer lazily."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required. Install it with: python3 -m pip install transformers"
        ) from exc

    return AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )


def tokenize_record(record, tokenizer, use_chat_template=False):
    """Convert one prompt record into token ids."""
    if use_chat_template:
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError("record does not contain a messages list")
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

    prompt = record.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("record does not contain a prompt string")
    return tokenizer.encode(prompt, add_special_tokens=False)


def trace_record(record, token_ids, block_size, timestamp):
    """Build a simulator trace record from token ids."""
    return {
        "id": record.get("request_id") or record.get("id") or str(timestamp),
        "timestamp": timestamp,
        "block_size": block_size,
        "hash_ids": block_hashes(token_ids, block_size),
        "input_length": len(token_ids),
        "output_length": 0,
    }


class KVCacheTracePlugin:
    """Plugin facade for prompt tokenization and block hash generation."""

    def __init__(
        self,
        tokenizer_name_or_path=DEFAULT_TOKENIZER,
        block_size=DEFAULT_BLOCK_SIZE,
        trust_remote_code=False,
        local_files_only=False,
        tokenizer=None,
    ):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.tokenizer = tokenizer or load_tokenizer(
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )

    def prompt_to_token_ids(self, prompt, add_special_tokens=False):
        """Convert a prompt string into token ids."""
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        return self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    def record_to_token_ids(self, record, use_chat_template=False):
        """Convert a prompt JSONL-style record into token ids."""
        return tokenize_record(record, self.tokenizer, use_chat_template)

    def token_ids_to_block_hashes(self, token_ids, block_size=None):
        """Convert token ids into prefix-aware block hashes."""
        return block_hashes(token_ids, block_size or self.block_size)

    def prompt_to_block_hashes(self, prompt, block_size=None, add_special_tokens=False):
        """Convert a prompt string directly into block hashes."""
        token_ids = self.prompt_to_token_ids(prompt, add_special_tokens=add_special_tokens)
        return self.token_ids_to_block_hashes(token_ids, block_size)

    def record_to_trace(self, record, timestamp=0, use_chat_template=False):
        """Convert one prompt record into one KV cache simulator trace record."""
        token_ids = self.record_to_token_ids(record, use_chat_template)
        if not token_ids:
            raise ValueError("record produced no tokens")
        return trace_record(record, token_ids, self.block_size, timestamp)


def convert_jsonl(
    input_path,
    output_path,
    tokenizer_name_or_path=DEFAULT_TOKENIZER,
    block_size=DEFAULT_BLOCK_SIZE,
    dataset_format="auto",
    use_chat_template=False,
    trust_remote_code=False,
    local_files_only=False,
    max_records=0,
    strict=False,
):
    """Convert a supported dataset into KV cache simulator trace JSONL."""
    plugin = KVCacheTracePlugin(
        tokenizer_name_or_path=tokenizer_name_or_path,
        block_size=block_size,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    total_tokens = 0
    total_blocks = 0

    with output_path.open("w", encoding="utf-8") as target:
        for index, record in iter_dataset_records(input_path):
            try:
                record = normalize_dataset_record(
                    record,
                    dataset_format=dataset_format,
                    index=index,
                    keep_messages=True,
                )
                output = plugin.record_to_trace(
                    record,
                    timestamp=index,
                    use_chat_template=use_chat_template,
                )
            except Exception as exc:
                skipped += 1
                if strict:
                    raise
                print(f"skip line {index + 1}: {exc}", file=sys.stderr)
                continue

            target.write(json.dumps(output, ensure_ascii=False) + "\n")
            total += 1
            total_tokens += output["input_length"]
            total_blocks += len(output["hash_ids"])

            if max_records and total >= max_records:
                break

    return {
        "total": total,
        "skipped": skipped,
        "total_tokens": total_tokens,
        "total_blocks": total_blocks,
        "avg_tokens": total_tokens / total if total else 0,
        "avg_blocks": total_blocks / total if total else 0,
        "output": output_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="prompt JSONL path")
    parser.add_argument("--output", type=Path, required=True, help="trace JSONL path")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="tokenizer name or path")
    parser.add_argument(
        "--dataset-format",
        choices=DATASET_FORMATS,
        default="auto",
        help="input dataset format; auto detects prompt JSONL, OpenAI messages, and ShareGPT",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="tokenize messages with tokenizer.apply_chat_template instead of prompt text",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="pass trust_remote_code=True to AutoTokenizer.from_pretrained",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="load tokenizer from the local Hugging Face cache only",
    )
    parser.add_argument("--max-records", type=int, default=0, help="debug limit")
    parser.add_argument("--strict", action="store_true", help="fail on the first bad row")
    args = parser.parse_args()

    stats = convert_jsonl(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name_or_path=args.tokenizer,
        block_size=args.block_size,
        dataset_format=args.dataset_format,
        use_chat_template=args.use_chat_template,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        max_records=args.max_records,
        strict=args.strict,
    )

    print(
        f"wrote {stats['total']} requests to {stats['output']}; "
        f"{stats['total_tokens']} input tokens, {stats['total_blocks']} blocks, "
        f"avg {stats['avg_tokens']:.0f} tokens/request, "
        f"avg {stats['avg_blocks']:.1f} blocks/request"
    )
    if stats["skipped"]:
        print(f"skipped {stats['skipped']} rows")


if __name__ == "__main__":
    main()
