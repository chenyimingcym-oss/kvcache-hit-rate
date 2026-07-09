from __future__ import annotations

import contextlib
import gzip
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kvcache_sim.calculator import calculate_cache_size, load_models_data, models_by_id
from kvcache_sim.cpp_backend import _build_path_for_source, _write_binary_trace, ensure_cpp_simulator
from kvcache_sim.model_aliases import resolve_model_alias
from kvcache_sim.plan import build_execution_plan
from kvcache_sim.policies import simulate_policy
from kvcache_sim.plotting import plot_hit_rate_sweep
from kvcache_sim._resources import package_resource_path, user_temp_suffix
from kvcache_sim.simulator import run_sweep
from kvcache_sim.trace import parse_trace_file, parse_trace_lines


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]


def make_trace(*hash_paths: list[str], block_size: int = 1):
    lines = [
        json.dumps({
            "block_size": block_size,
            "hash_ids": path,
            "input_length": len(path) * block_size,
        })
        for path in hash_paths
    ]
    return parse_trace_lines(lines)


class CalculatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models_data = load_models_data()
        cls.models = models_by_id(cls.models_data)

    def test_standard_gqa_matches_web_calculator_constant(self) -> None:
        result = calculate_cache_size(
            self.models["qwen3-32b"],
            tokens=128000,
            precision="bf16_fp16",
            models_data=self.models_data,
        )

        self.assertEqual(result.bytes_per_token, 262144)
        self.assertAlmostEqual(result.total_gib, 31.25)

    def test_glm_5_2_indexer_precision_matches_web_calculator_constant(self) -> None:
        result = calculate_cache_size(
            self.models["glm-5.2"],
            tokens=128000,
            precision="fp8_int8",
            indexer_precision="fp4_int4",
            models_data=self.models_data,
        )

        self.assertEqual(result.indexer_precision, "fp4_int4")
        self.assertEqual(result.bytes_per_token, 46272)
        self.assertAlmostEqual(result.total_gib, 5.51605224609375)

    def test_minimax_m3_keeps_fixed_bf16_indexer_precision(self) -> None:
        result = calculate_cache_size(
            self.models["minimax-m3"],
            tokens=1048576,
            precision="fp8_int8",
            indexer_precision="fp4_int4",
            models_data=self.models_data,
        )

        self.assertEqual(result.indexer_precision, "bf16_fp16")
        self.assertEqual(result.indexer_precision_label, "BF16 / FP16")
        self.assertAlmostEqual(result.total_gib, 74.25)

    def test_bundled_models_match_web_calculator_catalog_when_in_repo(self) -> None:
        web_catalog = REPO_ROOT / "data" / "kv_cache_calculator" / "models.yaml"
        bundled_catalog = PACKAGE_ROOT / "src" / "kvcache_sim" / "resources" / "models.yaml"
        if not web_catalog.exists():
            self.skipTest("web calculator catalog is not present")

        self.assertEqual(bundled_catalog.read_text(encoding="utf-8"), web_catalog.read_text(encoding="utf-8"))

    def test_model_alias_resolves_huggingface_repo_and_label(self) -> None:
        from_repo = resolve_model_alias("Qwen/Qwen3.6-27B", self.models_data)
        from_label = resolve_model_alias("Qwen3.6-27B", self.models_data)

        self.assertEqual(from_repo.model_id, "qwen3.6-27b")
        self.assertEqual(from_label.model_id, "qwen3.6-27b")
        self.assertEqual(from_repo.tokenizer, "Qwen/Qwen3.6-27B")


class TraceParserTests(unittest.TestCase):
    def test_jsonl_trace_parses_required_fields(self) -> None:
        trace = make_trace(["A", "B"], ["A", "C"], block_size=64)

        self.assertEqual(trace.request_count, 2)
        self.assertEqual(trace.block_size, 64)
        self.assertEqual(trace.total_input_tokens, 256)
        self.assertEqual(trace.unique_raw_blocks, 3)

    def test_missing_block_size_needs_cli_fallback(self) -> None:
        line = json.dumps({"hash_ids": ["A"], "input_length": 1})

        with self.assertRaisesRegex(ValueError, "without block_size need --block-size"):
            parse_trace_lines([line])

        trace = parse_trace_lines([line], block_size=64)
        self.assertEqual(trace.block_size, 64)


class DatasetConverterTests(unittest.TestCase):
    def test_sharegpt_pair_turns_with_human_and_assistant_keys(self) -> None:
        from plugins.dataset_converters import normalize_dataset_record

        record = {
            "conversation_id": "conv-1",
            "conversation": [
                {
                    "human": "写一个 landing page",
                    "assistant": "下面是一个 HTML 示例",
                }
            ],
        }

        normalized = normalize_dataset_record(record, dataset_format="sharegpt")

        self.assertEqual(normalized["request_id"], "conv-1")
        self.assertEqual(
            normalized["messages"],
            [
                {"role": "user", "content": "写一个 landing page"},
                {"role": "assistant", "content": "下面是一个 HTML 示例"},
            ],
        )
        self.assertIn("写一个 landing page", normalized["prompt"])
        self.assertIn("下面是一个 HTML 示例", normalized["prompt"])

    def test_trace_block_size_overrides_cli_fallback(self) -> None:
        lines = [
            json.dumps({"hash_ids": ["A", "B"], "input_length": 32}),
            json.dumps({"block_size": 16, "hash_ids": ["A", "B"], "input_length": 32}),
        ]

        trace = parse_trace_lines(lines, block_size=64)

        self.assertEqual(trace.block_size, 16)
        self.assertEqual(trace.tokens, [16, 16, 16, 16])

    def test_explicit_block_tokens_preserve_per_block_weights(self) -> None:
        line = json.dumps({
            "block_size": 64,
            "hash_ids": ["A", "B", "C"],
            "input_length": 130,
            "block_tokens": [64, 32, 34],
        })

        trace = parse_trace_lines([line])

        self.assertEqual(trace.tokens, [64, 32, 34])
        self.assertEqual(trace.total_input_tokens, 130)

    def test_gzip_jsonl_trace_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}) + "\n")

            trace = parse_trace_file(path)

        self.assertEqual(trace.request_count, 1)
        self.assertEqual(trace.ids, [0])


class PrefixPolicyTests(unittest.TestCase):
    def test_context_aware_identity_does_not_reuse_same_raw_block_under_different_parent(self) -> None:
        trace = make_trace(["A", "B"], ["C", "B"])
        plan = build_execution_plan(trace)

        self.assertEqual(trace.unique_raw_blocks, 3)
        self.assertEqual(plan.unique_blocks, 4)
        self.assertEqual(len(set(plan.node_for_event)), 4)

    def test_middle_miss_stops_prefix_hit_but_later_blocks_still_enter_cache(self) -> None:
        trace = make_trace(["A", "B"], ["A", "C", "B"])
        plan = build_execution_plan(trace)

        lru = simulate_policy(plan, "lru", 2)
        fifo = simulate_policy(plan, "fifo", 2)

        self.assertEqual(lru.hitTokens, 1)
        self.assertEqual(lru.totalTokens, 3)
        self.assertEqual(lru.hitRate, 1 / 3)
        self.assertEqual(fifo.hitTokens, 1)
        self.assertEqual(fifo.totalTokens, 3)

    def test_repeated_prefix_can_hit_after_warmup(self) -> None:
        trace = make_trace(["A", "B"], ["A", "B"], ["A", "B"])
        plan = build_execution_plan(trace)

        result = simulate_policy(plan, "lru", 2)

        self.assertEqual(result.hitTokens, 4)
        self.assertEqual(result.totalTokens, 4)
        self.assertEqual(result.hitRate, 1)

    def test_optimal_bypasses_polluting_leaf(self) -> None:
        trace = make_trace(["A"], ["B"], ["A"])
        plan = build_execution_plan(trace)

        result = simulate_policy(plan, "optimal", 1)

        self.assertEqual(result.hitTokens, 1)
        self.assertEqual(result.totalTokens, 2)
        self.assertEqual(result.hitRate, 0.5)

    def test_underfilled_capacity_is_reported_when_not_full_before_measurement(self) -> None:
        trace = make_trace(["A"], ["A"], ["A"], ["A"])
        plan = build_execution_plan(trace)

        result = simulate_policy(plan, "lru", 2)

        self.assertEqual(result.measurementMode, "underfilled_at_window")

    def test_global_mode_counts_from_empty_cache_without_underfilled_skip(self) -> None:
        trace = make_trace(["A"], ["A"], ["A"], ["A"])
        plan = build_execution_plan(trace, warmup_fraction=0)

        result = simulate_policy(plan, "lru", 2, require_full_before_measurement=False)

        self.assertEqual(result.measurementMode, "fixed_window")
        self.assertEqual(result.warmupRequests, 0)
        self.assertEqual(result.totalTokens, 4)
        self.assertEqual(result.hitTokens, 3)
        self.assertEqual(result.hitRate, 0.75)


class SweepAndCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models_data = load_models_data()

    def test_run_sweep_defaults_to_global_hit_rate_and_includes_underfilled_budgets(self) -> None:
        trace = make_trace(["A"], ["A"], ["A"], ["A"])
        result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025, 0.0005],
            policies=["lru"],
            backend="python",
            models_data=self.models_data,
        )

        self.assertEqual(result["metadata"]["warmupFraction"], 0)
        self.assertEqual(result["metadata"]["warmupRequests"], 0)
        self.assertEqual(result["metadata"]["measurementWindow"], "all_requests")
        self.assertEqual(result["metadata"]["underfilledBudgetPolicy"], "include_from_empty_cache")
        self.assertEqual([point["cacheBlocks"] for point in result["points"]], [1])

    def test_run_sweep_can_use_fixed_50_percent_window_and_omit_underfilled_larger_budgets(self) -> None:
        trace = make_trace(["A"], ["A"], ["B"], ["C"])
        result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025, 0.0005, 0.001],
            policies=["fifo", "lru", "optimal"],
            backend="python",
            warmup_fraction=0.5,
            include_underfilled=False,
            models_data=self.models_data,
        )

        self.assertEqual(result["metadata"]["warmupFraction"], 0.5)
        self.assertEqual(result["metadata"]["totalMeasuredTokens"], 2)
        self.assertEqual([point["cacheBlocks"] for point in result["points"]], [1])
        self.assertEqual(result["points"][0]["results"]["fifo"]["totalTokens"], 2)

    def test_run_sweep_global_hit_rate_includes_underfilled_budgets(self) -> None:
        trace = make_trace(["A"], ["A"], ["A"], ["A"])
        result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025, 0.0005],
            policies=["lru"],
            backend="python",
            warmup_fraction=0,
            include_underfilled=True,
            models_data=self.models_data,
        )

        self.assertEqual(result["metadata"]["warmupFraction"], 0)
        self.assertEqual(result["metadata"]["warmupRequests"], 0)
        self.assertEqual(result["metadata"]["measurementWindow"], "all_requests")
        self.assertEqual(result["metadata"]["underfilledBudgetPolicy"], "include_from_empty_cache")
        self.assertEqual([point["cacheBlocks"] for point in result["points"]], [1])
        self.assertEqual(result["points"][0]["results"]["lru"]["totalTokens"], 4)
        self.assertEqual(result["points"][0]["results"]["lru"]["hitTokens"], 3)

    def test_run_sweep_sorts_budgets_before_early_breaks(self) -> None:
        trace = make_trace(["A"], ["B"], ["A"], ["B"])
        result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.001, 0.00025, 0.00025],
            policies=["fifo"],
            backend="python",
            models_data=self.models_data,
        )

        self.assertEqual([point["cacheBlocks"] for point in result["points"]], [1])
        self.assertEqual(result["points"][0]["gib"], 0.00025)

    def test_run_sweep_accepts_huggingface_model_alias(self) -> None:
        trace = make_trace(["A"], ["B"], ["A"], ["B"])
        result = run_sweep(
            trace,
            model_id="Qwen/Qwen3.6-27B",
            precision="fp8_int8",
            budgets_gib=[0.00025],
            policies=["fifo"],
            backend="python",
            models_data=self.models_data,
        )

        self.assertEqual(result["metadata"]["modelId"], "qwen3.6-27b")
        self.assertEqual(result["metadata"]["modelLabel"], "Qwen3.6-27B")

    def test_plot_hit_rate_sweep_writes_image(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        trace = make_trace(["A"], ["B"], ["A"], ["B"])
        result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025],
            policies=["fifo", "lru", "optimal"],
            backend="python",
            models_data=self.models_data,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = plot_hit_rate_sweep(result, Path(tmpdir) / "hit-rate.png")

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_multiprocess_sweep_matches_single_process(self) -> None:
        trace = make_trace(["A"], ["B"], ["A"], ["B"])
        serial = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025],
            jobs=1,
            backend="python",
            models_data=self.models_data,
        )
        parallel = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025],
            jobs=2,
            backend="python",
            models_data=self.models_data,
        )

        self.assertEqual(serial["points"], parallel["points"])

    def test_cpp_backend_matches_python_backend_when_compiler_is_available(self) -> None:
        if not shutil.which("c++"):
            self.skipTest("c++ compiler is not available")
        trace = make_trace(["A"], ["B"], ["A"], ["B"])
        python_result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025],
            jobs=1,
            backend="python",
            warmup_fraction=0.5,
            include_underfilled=False,
            models_data=self.models_data,
        )
        cpp_result = run_sweep(
            trace,
            model_id="qwen3-32b",
            precision="bf16_fp16",
            budgets_gib=[0.00025],
            backend="cpp",
            warmup_fraction=0.5,
            include_underfilled=False,
            models_data=self.models_data,
        )

        self.assertEqual(cpp_result["metadata"]["backend"], "cpp")
        self.assertEqual(python_result["points"], cpp_result["points"])
        self.assertEqual(python_result["hitRateCeiling"], cpp_result["hitRateCeiling"])

    def test_cpp_backend_reports_ceiling_for_unlimited_capacity(self) -> None:
        if not shutil.which("c++"):
            self.skipTest("c++ compiler is not available")
        trace = make_trace(["A"], ["A"], ["B"], ["A"])
        binary = ensure_cpp_simulator()
        with tempfile.TemporaryDirectory() as tmpdir:
            files = _write_binary_trace(trace, Path(tmpdir))
            base = [
                str(binary),
                "--ids",
                str(files.ids),
                "--tokens",
                str(files.tokens),
                "--request-ends",
                str(files.request_ends),
                "--request-count",
                str(trace.request_count),
                "--total-blocks",
                str(len(trace.ids)),
                "--warmup-requests",
                "0",
            ]
            omitted = subprocess.run([*base, "--policy", "lru"], check=True, text=True, capture_output=True)
            explicit = subprocess.run([*base, "--policy", "lru", "--capacity", "-1"], check=True, text=True, capture_output=True)

        omitted_payload = json.loads(omitted.stdout)
        explicit_payload = json.loads(explicit.stdout)
        self.assertEqual(omitted_payload["policy"], "ceiling")
        self.assertEqual(omitted_payload["cacheBlocks"], -1)
        self.assertEqual(omitted_payload["hitRate"], explicit_payload["hitRate"])
        self.assertEqual(omitted_payload["hitTokens"], 2)

    def test_cli_outputs_table_by_default_and_json_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            trace_path.write_text(
                "\n".join([
                    json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["B"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["B"], "input_length": 1}),
                ])
                + "\n",
                encoding="utf-8",
            )
            base = [
                sys.executable,
                "-m",
                "kvcache_sim",
                "sweep",
                "--trace",
                str(trace_path),
                "--model",
                "qwen3-32b",
                "--kv-precision",
                "bf16_fp16",
                "--budgets-gib",
                "0.00025",
                "--backend",
                "python",
            ]

            table = subprocess.run(base, cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)
            as_json = subprocess.run(base + ["--format", "json"], cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)

        self.assertIn("Budget", table.stdout)
        self.assertIn("FIFO hit", table.stdout)
        parsed = json.loads(as_json.stdout)
        self.assertEqual(parsed["metadata"]["modelId"], "qwen3-32b")
        self.assertEqual(parsed["metadata"]["warmupFraction"], 0)
        self.assertEqual(parsed["metadata"]["measurementWindow"], "all_requests")
        self.assertEqual(parsed["points"][0]["cacheBlocks"], 1)
        self.assertIn("Measurement: hit rates use all requests", table.stdout)
        self.assertIn("Speedup: 1.0x means no-cache prefill throughput", table.stdout)
        self.assertIn("Summary:", as_json.stderr)
        self.assertIn("Hit rate ceiling: 50.00%", as_json.stderr)

    def test_cli_plot_command_writes_image_from_json(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.json"
            plot_path = Path(tmpdir) / "hit-rate.png"
            result_path.write_text(
                json.dumps({
                    "metadata": {
                        "modelId": "test-model",
                        "modelLabel": "Test Model",
                        "precision": "bf16_fp16",
                        "precisionLabel": "BF16 / FP16",
                    },
                    "hitRateCeiling": 0.75,
                    "policies": ["lru"],
                    "points": [
                        {
                            "gib": 1,
                            "cacheBlocks": 1,
                            "results": {"lru": {"hitRate": 0.5}},
                        }
                    ],
                }),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kvcache_sim",
                    "plot",
                    "--input",
                    str(result_path),
                    "--output",
                    str(plot_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_cli_run_command_matches_sweep_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            trace_path.write_text(
                "\n".join([
                    json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["B"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["B"], "input_length": 1}),
                ])
                + "\n",
                encoding="utf-8",
            )
            args = [
                "--trace",
                str(trace_path),
                "--model",
                "qwen3-32b",
                "--kv-precision",
                "bf16_fp16",
                "--budgets-gib",
                "0.00025",
                "--backend",
                "python",
                "--format",
                "json",
            ]

            run_result = subprocess.run([sys.executable, "-m", "kvcache_sim", "run", *args], cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)
            sweep_result = subprocess.run([sys.executable, "-m", "kvcache_sim", "sweep", *args], cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)

        self.assertEqual(json.loads(run_result.stdout)["points"], json.loads(sweep_result.stdout)["points"])

    def test_plugin_cli_reports_unsupported_dataset_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "bad.jsonl"
            output_path = Path(tmpdir) / "prompts.jsonl"
            input_path.write_text(json.dumps({"title": "missing prompt shape"}) + "\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "plugins" / "kv_cache_hit_rate_plugin.py"),
                    "normalize",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported dataset format", result.stderr)
        self.assertIn("Supported dataset formats", result.stderr)

    def test_plugin_run_derives_tokenizer_from_model_alias(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        from plugins.kv_cache_hit_rate_plugin import build_parser, simulator_kwargs, tokenizer_name_from_args

        args = build_parser().parse_args([
            "run",
            "--input",
            "input.jsonl",
            "--model",
            "Qwen/Qwen3.6-27B",
        ])

        self.assertEqual(tokenizer_name_from_args(args), "Qwen/Qwen3.6-27B")
        self.assertEqual(simulator_kwargs(args)["model_id"], "qwen3.6-27b")

    def test_plugin_run_requires_model(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        from plugins.kv_cache_hit_rate_plugin import build_parser

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args([
                "run",
                "--input",
                "input.jsonl",
            ])

    def test_plugin_convert_requires_tokenizer(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        from plugins.kv_cache_hit_rate_plugin import build_parser

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args([
                "convert",
                "--input",
                "input.jsonl",
                "--output",
                "trace.jsonl",
            ])

    def test_plugin_run_keeps_explicit_tokenizer(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        from plugins.kv_cache_hit_rate_plugin import build_parser, tokenizer_name_from_args

        args = build_parser().parse_args([
            "run",
            "--input",
            "input.jsonl",
            "--model",
            "Qwen/Qwen3.6-27B",
            "--tokenizer",
            "moonshotai/Kimi-K2.6",
        ])

        self.assertEqual(tokenizer_name_from_args(args), "moonshotai/Kimi-K2.6")

    def test_plugin_run_uses_custom_models_yaml_for_alias_and_simulation(self) -> None:
        sys.path.insert(0, str(REPO_ROOT))
        from plugins.kv_cache_hit_rate_plugin import KVCacheHitRatePlugin, build_parser, simulator_kwargs, tokenizer_name_from_args

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            models_yaml = tmpdir_path / "models.yaml"
            models_yaml.write_text(
                "\n".join([
                    "precision_options:",
                    "  - id: fp8_int8",
                    "    label: \"FP8 / INT8\"",
                    "    bytes_per_element: 1",
                    "indexer_precision_options:",
                    "  - id: fp8_int8",
                    "    label: \"FP8 / INT8\"",
                    "    bytes_per_element: 1",
                    "models:",
                    "  - id: custom-1b",
                    "    label: \"Custom-1B\"",
                    "    family: \"Custom\"",
                    "    formula: \"standard_gqa\"",
                    "    default_tokens: 1024",
                    "    source_url: \"https://huggingface.co/Custom/Custom-1B/raw/main/config.json\"",
                    "    fields:",
                    "      num_hidden_layers: 1",
                    "      num_key_value_heads: 1",
                    "      head_dim: 1",
                ])
                + "\n",
                encoding="utf-8",
            )
            trace_path = tmpdir_path / "trace.jsonl"
            trace_path.write_text(
                "\n".join([
                    json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["B"], "input_length": 1}),
                    json.dumps({"block_size": 1, "hash_ids": ["A"], "input_length": 1}),
                ])
                + "\n",
                encoding="utf-8",
            )

            args = build_parser().parse_args([
                "run",
                "--input",
                "input.jsonl",
                "--model",
                "Custom/Custom-1B",
                "--models-yaml",
                str(models_yaml),
                "--budgets-gib",
                "0.000000002",
            ])

            self.assertEqual(tokenizer_name_from_args(args), "Custom/Custom-1B")
            kwargs = simulator_kwargs(args)
            self.assertEqual(kwargs["model_id"], "custom-1b")
            plugin = KVCacheHitRatePlugin(tokenizer=object(), block_size=1)
            result = plugin.simulate_trace(trace_path, **kwargs)

        self.assertEqual(result["metadata"]["modelId"], "custom-1b")
        self.assertEqual(result["metadata"]["modelLabel"], "Custom-1B")
        self.assertEqual(result["metadata"]["bytesPerBlock"], 2)
        self.assertEqual(result["points"][0]["cacheBlocks"], 1)


class TempPathTests(unittest.TestCase):
    def test_temp_paths_include_user_suffix_when_available(self) -> None:
        suffix = user_temp_suffix()
        if not suffix:
            self.skipTest("platform does not expose os.getuid")

        resource_path = package_resource_path("models.yaml")
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "sim.cc"
            source.write_text("int main() { return 0; }\n", encoding="utf-8")
            binary_path = _build_path_for_source(source)

        if tempfile.gettempdir() in str(resource_path):
            self.assertIn(suffix, resource_path.name)
        self.assertIn(suffix, binary_path.name)


if __name__ == "__main__":
    unittest.main()
