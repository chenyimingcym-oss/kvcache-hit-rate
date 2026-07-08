#!/usr/bin/env python3
"""Dataset format converters for KV cache trace generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


DATASET_FORMATS = ("auto", "jsonl", "openai", "sharegpt", "text")
SUPPORTED_DATASET_FORMATS = (
    "Supported dataset formats: prompt JSONL records with `prompt`, `text`, "
    "or `messages`; OpenAI chat records with `messages` or `body.messages`; "
    "ShareGPT records with `conversations`; JSON string records when "
    "`--dataset-format text` is selected."
)


class UnsupportedDatasetFormatError(ValueError):
    """Raised when the input dataset has no records in a supported shape."""

_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "chatgpt": "assistant",
    "system": "system",
}


def content_to_text(content: Any) -> str:
    """Convert common multimodal/text content shapes into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("value")
                if text is not None:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def messages_to_prompt(messages: Iterable[dict[str, Any]]) -> str:
    """Render chat messages into the repository's plain prompt convention."""
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        content = content_to_text(message.get("content"))
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def _record_id(record: dict[str, Any], fallback: int) -> str:
    for key in ("request_id", "custom_id", "id", "conversation_id", "uid"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return str(fallback)


def _normalize_openai_message(message: Any) -> Optional[dict[str, str]]:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str):
        return None
    return {
        "role": role,
        "content": content_to_text(message.get("content")),
    }


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    normalized = []
    for message in messages:
        normalized_message = _normalize_openai_message(message)
        if normalized_message is not None:
            normalized.append(normalized_message)
    if not normalized:
        raise ValueError("messages list contains no valid chat messages")
    return normalized


def _openai_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if messages is None and isinstance(record.get("body"), dict):
        messages = record["body"].get("messages")
    return _normalize_messages(messages)


def _sharegpt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    conversations = record.get("conversations")
    if conversations is None:
        conversations = record.get("conversation")
    if not isinstance(conversations, list):
        raise ValueError("ShareGPT record does not contain a conversations list")

    messages = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        raw_role = turn.get("from") or turn.get("role")
        role = _SHAREGPT_ROLE_MAP.get(str(raw_role).lower(), str(raw_role or "unknown"))
        content = content_to_text(turn.get("value") if "value" in turn else turn.get("content"))
        messages.append({"role": role, "content": content})

    if not messages:
        raise ValueError("ShareGPT conversations list contains no valid turns")
    return messages


def detect_dataset_format(record: Any) -> str:
    """Infer the dataset format for one parsed record."""
    if isinstance(record, str):
        return "text"
    if not isinstance(record, dict):
        raise UnsupportedDatasetFormatError(
            f"cannot auto-detect dataset format for {type(record).__name__} record. "
            f"{SUPPORTED_DATASET_FORMATS}"
        )
    if "conversations" in record or "conversation" in record:
        return "sharegpt"
    if isinstance(record.get("messages"), list):
        return "openai"
    if isinstance(record.get("body"), dict) and isinstance(record["body"].get("messages"), list):
        return "openai"
    return "jsonl"


def normalize_dataset_record(
    record: Any,
    *,
    dataset_format: str = "auto",
    index: int = 0,
    keep_messages: bool = True,
) -> dict[str, Any]:
    """Normalize one dataset record to `{request_id, prompt, messages?}`."""
    if dataset_format not in DATASET_FORMATS:
        raise ValueError(f"unsupported dataset format: {dataset_format}")

    selected_format = detect_dataset_format(record) if dataset_format == "auto" else dataset_format

    if selected_format == "text":
        prompt = record if isinstance(record, str) else json.dumps(record, ensure_ascii=False)
        return {"request_id": str(index), "prompt": prompt}

    if not isinstance(record, dict):
        raise ValueError(f"{selected_format} record must be a JSON object")

    request_id = _record_id(record, index)

    if selected_format == "sharegpt":
        messages = _sharegpt_messages(record)
        output = {"request_id": request_id, "prompt": messages_to_prompt(messages)}
        if keep_messages:
            output["messages"] = messages
        return output

    if selected_format == "openai":
        messages = _openai_messages(record)
        output = {"request_id": request_id, "prompt": messages_to_prompt(messages)}
        if keep_messages:
            output["messages"] = messages
        return output

    prompt = record.get("prompt")
    if not isinstance(prompt, str):
        text = record.get("text")
        if isinstance(text, str):
            prompt = text
        elif isinstance(record.get("messages"), list):
            messages = _normalize_messages(record["messages"])
            output = {"request_id": request_id, "prompt": messages_to_prompt(messages)}
            if keep_messages:
                output["messages"] = messages
            return output
        else:
            raise UnsupportedDatasetFormatError(
                "JSONL record does not contain prompt, text, or messages. "
                f"{SUPPORTED_DATASET_FORMATS}"
            )

    output = {"request_id": request_id, "prompt": prompt}
    if keep_messages and isinstance(record.get("messages"), list):
        output["messages"] = _normalize_messages(record["messages"])
    return output


def iter_dataset_records(input_path: str | Path) -> Iterator[tuple[int, Any]]:
    """Yield parsed records from JSONL, a JSON array, or a single JSON value."""
    input_path = Path(input_path)
    with input_path.open(encoding="utf-8") as source:
        first = ""
        while True:
            char = source.read(1)
            if not char or not char.isspace():
                first = char
                break
        source.seek(0)
        if first == "[":
            try:
                payload = json.load(source)
            except json.JSONDecodeError as exc:
                raise UnsupportedDatasetFormatError(
                    f"input is not valid JSON: {exc}. {SUPPORTED_DATASET_FORMATS}"
                ) from exc
            if not isinstance(payload, list):
                raise ValueError("top-level JSON array expected")
            for index, record in enumerate(payload):
                yield index, record
            return

        for index, raw in enumerate(source):
            line = raw.strip()
            if line:
                try:
                    yield index, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise UnsupportedDatasetFormatError(
                        f"line {index + 1} is not valid JSONL: {exc}. "
                        f"{SUPPORTED_DATASET_FORMATS}"
                    ) from exc


def convert_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dataset_format: str = "auto",
    keep_messages: bool = True,
    max_records: int = 0,
    strict: bool = False,
) -> dict[str, Any]:
    """Convert an input dataset into normalized prompt JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as target:
        for index, record in iter_dataset_records(input_path):
            try:
                output = normalize_dataset_record(
                    record,
                    dataset_format=dataset_format,
                    index=index,
                    keep_messages=keep_messages,
                )
            except Exception as exc:
                skipped += 1
                if strict:
                    raise
                print(f"skip line {index + 1}: {exc}", file=sys.stderr)
                continue

            target.write(json.dumps(output, ensure_ascii=False) + "\n")
            total += 1
            if max_records and total >= max_records:
                break

    if total == 0:
        raise UnsupportedDatasetFormatError(
            f"no supported dataset records were found in {input_path}; skipped {skipped} record(s). "
            f"{SUPPORTED_DATASET_FORMATS}"
        )

    return {"total": total, "skipped": skipped, "output": output_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-format",
        choices=DATASET_FORMATS,
        default="auto",
        help="input format; auto detects per record",
    )
    parser.add_argument(
        "--drop-messages",
        action="store_true",
        help="omit normalized messages and keep only prompt text",
    )
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stats = convert_dataset(
            args.input,
            args.output,
            dataset_format=args.dataset_format,
            keep_messages=not args.drop_messages,
            max_records=args.max_records,
            strict=args.strict,
        )
    except UnsupportedDatasetFormatError as exc:
        print(f"unsupported dataset format: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
