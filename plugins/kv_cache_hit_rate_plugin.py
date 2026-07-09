#!/usr/bin/env python3
"""Standalone KV cache hit-rate plugin.

This module integrates prompt-to-trace conversion with the local
`kvcache-simulator` package so a caller can:

1. convert prompts to token ids,
2. convert token ids to prefix-aware block hashes,
3. normalize JSONL/OpenAI/ShareGPT datasets,
4. convert normalized prompts to simulator trace JSONL,
5. run KV cache hit-rate sweeps over that trace.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

try:
    from .kvcache_trace_plugin import (
        DEFAULT_BLOCK_SIZE,
        DEFAULT_TOKENIZER,
        KVCacheTracePlugin,
        block_hashes,
    )
    from .dataset_converters import (
        DATASET_FORMATS,
        SUPPORTED_DATASET_FORMATS,
        UnsupportedDatasetFormatError,
        convert_dataset,
        iter_dataset_records,
        normalize_dataset_record,
    )
except ImportError:
    from kvcache_trace_plugin import (
        DEFAULT_BLOCK_SIZE,
        DEFAULT_TOKENIZER,
        KVCacheTracePlugin,
        block_hashes,
    )
    from dataset_converters import (
        DATASET_FORMATS,
        SUPPORTED_DATASET_FORMATS,
        UnsupportedDatasetFormatError,
        convert_dataset,
        iter_dataset_records,
        normalize_dataset_record,
    )


DEFAULT_MODEL = "qwen3-32b"
DEFAULT_PRECISION = "fp8_int8"
DEFAULT_BUDGETS_GIB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_POLICIES = ["fifo", "lru", "optimal"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_simulator_importable() -> None:
    simulator_src = _repo_root() / "packages" / "kvcache-simulator" / "src"
    if simulator_src.exists():
        simulator_src_text = str(simulator_src)
        if simulator_src_text not in sys.path:
            sys.path.insert(0, simulator_src_text)


def _plugin_models_data(models_yaml: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    _ensure_simulator_importable()
    from kvcache_sim.calculator import load_models_data

    if models_yaml:
        return load_models_data(models_yaml)
    repo_models_yaml = _repo_root() / "data" / "kv_cache_calculator" / "models.yaml"
    return load_models_data(repo_models_yaml) if repo_models_yaml.exists() else load_models_data()


def _resolve_model_arg(model: str, models_yaml: Optional[Union[str, Path]] = None):
    _ensure_simulator_importable()
    from kvcache_sim.model_aliases import resolve_model_alias

    return resolve_model_alias(model, _plugin_models_data(models_yaml))


def tokenizer_name_from_args(args: argparse.Namespace) -> str:
    explicit_tokenizer = getattr(args, "tokenizer", None)
    if explicit_tokenizer:
        return explicit_tokenizer
    model = getattr(args, "model", None)
    if model:
        return _resolve_model_arg(model, getattr(args, "models_yaml", None)).tokenizer or DEFAULT_TOKENIZER
    return DEFAULT_TOKENIZER


def _parse_csv_numbers(value: Optional[str], fallback: list[float]) -> list[float]:
    if not value:
        return fallback
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: Optional[str], fallback: list[str]) -> list[str]:
    if not value:
        return fallback
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def _write_json(payload: dict[str, Any], output_path: Optional[Path]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


class KVCacheHitRatePlugin:
    """Unified plugin for trace generation and KV cache hit-rate simulation."""

    def __init__(
        self,
        tokenizer_name_or_path: str = DEFAULT_TOKENIZER,
        block_size: int = DEFAULT_BLOCK_SIZE,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.trace_plugin = KVCacheTracePlugin(
            tokenizer_name_or_path=tokenizer_name_or_path,
            block_size=block_size,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            tokenizer=tokenizer,
        )

    def prompt_to_token_ids(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        return self.trace_plugin.prompt_to_token_ids(
            prompt,
            add_special_tokens=add_special_tokens,
        )

    def token_ids_to_block_hashes(
        self,
        token_ids: list[int],
        block_size: Optional[int] = None,
    ) -> list[int]:
        return self.trace_plugin.token_ids_to_block_hashes(token_ids, block_size)

    def prompt_to_block_hashes(
        self,
        prompt: str,
        block_size: Optional[int] = None,
        add_special_tokens: bool = False,
    ) -> list[int]:
        return self.trace_plugin.prompt_to_block_hashes(
            prompt,
            block_size=block_size,
            add_special_tokens=add_special_tokens,
        )

    def record_to_trace(
        self,
        record: dict[str, Any],
        timestamp: int = 0,
        use_chat_template: bool = False,
    ) -> dict[str, Any]:
        return self.trace_plugin.record_to_trace(
            record,
            timestamp=timestamp,
            use_chat_template=use_chat_template,
        )

    def prompts_jsonl_to_trace(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        *,
        dataset_format: str = "auto",
        use_chat_template: bool = False,
        max_records: int = 0,
        strict: bool = False,
    ) -> dict[str, Any]:
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
                    output = self.trace_plugin.record_to_trace(
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

        if total == 0:
            raise UnsupportedDatasetFormatError(
                f"no supported dataset records were found in {input_path}; skipped {skipped} record(s). "
                f"{SUPPORTED_DATASET_FORMATS}"
            )

        return {
            "total": total,
            "skipped": skipped,
            "total_tokens": total_tokens,
            "total_blocks": total_blocks,
            "avg_tokens": total_tokens / total if total else 0,
            "avg_blocks": total_blocks / total if total else 0,
            "output": output_path,
        }

    def simulate_trace(
        self,
        trace_path: Union[str, Path],
        *,
        model_id: str = DEFAULT_MODEL,
        precision: Optional[str] = DEFAULT_PRECISION,
        indexer_precision: Optional[str] = None,
        budgets_gib: Optional[list[float]] = None,
        policies: Optional[list[str]] = None,
        backend: str = "python",
        jobs: int = 1,
        warmup_fraction: float = 0,
        include_underfilled: bool = True,
        estimate_tokens: Optional[int] = None,
        include_draft_kv_cache: bool = False,
        max_records: int = 0,
        max_events: int = 0,
        models_yaml: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        _ensure_simulator_importable()
        from kvcache_sim.simulator import run_sweep
        from kvcache_sim.trace import parse_trace_file

        models_data = _plugin_models_data(models_yaml)
        trace = parse_trace_file(
            trace_path,
            block_size=self.trace_plugin.block_size,
            max_records=max_records,
            max_events=max_events,
        )
        return run_sweep(
            trace,
            model_id=model_id,
            precision=precision,
            indexer_precision=indexer_precision,
            budgets_gib=budgets_gib or DEFAULT_BUDGETS_GIB,
            policies=policies or DEFAULT_POLICIES,
            backend=backend,
            jobs=jobs,
            warmup_fraction=warmup_fraction,
            include_underfilled=include_underfilled,
            estimate_tokens=estimate_tokens,
            include_draft_kv_cache=include_draft_kv_cache,
            models_data=models_data,
        )

    def prompts_jsonl_to_hit_rate(
        self,
        input_path: Union[str, Path],
        *,
        trace_output_path: Optional[Union[str, Path]] = None,
        dataset_format: str = "auto",
        use_chat_template: bool = False,
        max_records: int = 0,
        strict: bool = False,
        **simulate_kwargs: Any,
    ) -> dict[str, Any]:
        if trace_output_path:
            trace_path = Path(trace_output_path)
            self.prompts_jsonl_to_trace(
                input_path,
                trace_path,
                dataset_format=dataset_format,
                use_chat_template=use_chat_template,
                max_records=max_records,
                strict=strict,
            )
            return self.simulate_trace(trace_path, **simulate_kwargs)

        with tempfile.TemporaryDirectory(prefix="kvcache-hit-rate-") as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            self.prompts_jsonl_to_trace(
                input_path,
                trace_path,
                dataset_format=dataset_format,
                use_chat_template=use_chat_template,
                max_records=max_records,
                strict=strict,
            )
            return self.simulate_trace(trace_path, **simulate_kwargs)

    def plot_result(
        self,
        result: dict[str, Any],
        output_path: Union[str, Path],
        *,
        title: Optional[str] = None,
    ) -> Path:
        _ensure_simulator_importable()
        from kvcache_sim.plotting import plot_hit_rate_sweep

        return plot_hit_rate_sweep(result, output_path, title=title)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    convert = subparsers.add_parser(
        "convert",
        help="Convert a supported dataset to trace JSONL",
    )
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    add_tokenizer_args(convert)
    convert.add_argument("--max-records", type=int, default=0)
    convert.add_argument("--strict", action="store_true")

    normalize = subparsers.add_parser(
        "normalize",
        help="Normalize JSONL, OpenAI messages, or ShareGPT data to prompt JSONL",
    )
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    add_dataset_args(normalize)
    normalize.add_argument(
        "--drop-messages",
        action="store_true",
        help="omit normalized messages and keep only prompt text",
    )
    normalize.add_argument("--max-records", type=int, default=0)
    normalize.add_argument("--strict", action="store_true")

    simulate = subparsers.add_parser("simulate", help="Run hit-rate simulation on trace JSONL")
    simulate.add_argument("--trace", type=Path, required=True)
    add_simulator_args(simulate)
    simulate.add_argument("--max-records", type=int, default=0)

    plot = subparsers.add_parser("plot", help="Plot hit-rate curves from a simulation JSON result")
    plot.add_argument("--input", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument("--title", default=None)

    run = subparsers.add_parser(
        "run",
        help="Convert a supported dataset and run hit-rate simulation",
    )
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--trace-output", type=Path, default=None)
    add_tokenizer_args(run)
    add_simulator_args(run)
    run.add_argument("--max-records", type=int, default=0)
    run.add_argument("--strict", action="store_true")

    return parser


def add_tokenizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--tokenizer", default=None)
    add_dataset_args(parser)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-format",
        choices=DATASET_FORMATS,
        default="auto",
        help="input format; auto detects prompt JSONL, OpenAI messages, and ShareGPT",
    )


def add_simulator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--kv-precision", default=DEFAULT_PRECISION)
    parser.add_argument("--indexer-precision", default=None)
    parser.add_argument("--budgets-gib", default=",".join(str(v) for v in DEFAULT_BUDGETS_GIB))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--backend", choices=["python", "cpp"], default="python")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0,
        help="0 means cold-start/global hit rate; use 0.5 to match web default.",
    )
    parser.add_argument(
        "--exclude-underfilled",
        action="store_true",
        help="Omit budgets where cache was not full before measurement.",
    )
    parser.add_argument("--estimate-tokens", type=int, default=None)
    parser.add_argument("--include-draft-kv-cache", action="store_true")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--models-yaml", type=Path, default=None)


def plugin_from_args(args: argparse.Namespace) -> KVCacheHitRatePlugin:
    tokenizer = object() if getattr(args, "command", "") == "simulate" else None
    return KVCacheHitRatePlugin(
        tokenizer_name_or_path=tokenizer_name_from_args(args),
        block_size=getattr(args, "block_size", DEFAULT_BLOCK_SIZE),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        local_files_only=getattr(args, "local_files_only", False),
        tokenizer=tokenizer,
    )


def simulator_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    resolved_model = _resolve_model_arg(args.model, getattr(args, "models_yaml", None))
    kwargs: dict[str, Any] = {
        "model_id": resolved_model.model_id,
        "precision": args.kv_precision,
        "indexer_precision": args.indexer_precision,
        "budgets_gib": _parse_csv_numbers(args.budgets_gib, DEFAULT_BUDGETS_GIB),
        "policies": _parse_csv_strings(args.policies, DEFAULT_POLICIES),
        "backend": args.backend,
        "jobs": args.jobs,
        "warmup_fraction": args.warmup_fraction,
        "include_underfilled": not args.exclude_underfilled,
        "estimate_tokens": args.estimate_tokens,
        "include_draft_kv_cache": args.include_draft_kv_cache,
        "max_events": args.max_events,
        "models_yaml": args.models_yaml,
    }
    if args.command == "simulate":
        kwargs["max_records"] = getattr(args, "max_records", 0)
    return kwargs


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help(sys.stderr)
        return 2

    try:
        if args.command == "normalize":
            stats = convert_dataset(
                args.input,
                args.output,
                dataset_format=args.dataset_format,
                keep_messages=not args.drop_messages,
                max_records=args.max_records,
                strict=args.strict,
            )
            _write_json(stats, None)
            return 0

        if args.command == "plot":
            _ensure_simulator_importable()
            from kvcache_sim.plotting import plot_hit_rate_sweep

            payload = json.loads(args.input.read_text(encoding="utf-8"))
            plot_hit_rate_sweep(payload, args.output, title=args.title)
            return 0

        plugin = plugin_from_args(args)
        if args.command == "convert":
            stats = plugin.prompts_jsonl_to_trace(
                args.input,
                args.output,
                dataset_format=args.dataset_format,
                use_chat_template=args.use_chat_template,
                max_records=args.max_records,
                strict=args.strict,
            )
            _write_json(stats, None)
            return 0

        if args.command == "simulate":
            result = plugin.simulate_trace(args.trace, **simulator_kwargs(args))
            if args.plot_output:
                plugin.plot_result(result, args.plot_output)
            _ensure_simulator_importable()
            from kvcache_sim.formatting import render_summary

            _write_json(result, args.output)
            print(render_summary(result, plot_output=str(args.plot_output) if args.plot_output else None), file=sys.stderr)
            return 0

        if args.command == "run":
            result = plugin.prompts_jsonl_to_hit_rate(
                args.input,
                trace_output_path=args.trace_output,
                dataset_format=args.dataset_format,
                use_chat_template=args.use_chat_template,
                max_records=args.max_records,
                strict=args.strict,
                **simulator_kwargs(args),
            )
            if args.plot_output:
                plugin.plot_result(result, args.plot_output)
            _ensure_simulator_importable()
            from kvcache_sim.formatting import render_summary

            _write_json(result, args.output)
            print(render_summary(result, plot_output=str(args.plot_output) if args.plot_output else None), file=sys.stderr)
            return 0
    except UnsupportedDatasetFormatError as exc:
        print(f"unsupported dataset format: {exc}", file=sys.stderr)
        return 1

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
