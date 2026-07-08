from __future__ import annotations

import json
from typing import Any


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_speedup(value: float) -> str:
    return f"{value:.1f}x"


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def measurement_description(meta: dict[str, Any]) -> str:
    window = meta.get("measurementWindow")
    if window == "all_requests":
        measurement = "all requests"
    elif window == "last_50_percent_requests":
        measurement = "the last 50% of requests"
    elif isinstance(window, str) and window.startswith("last_") and window.endswith("_fraction_requests"):
        fraction = window.removeprefix("last_").removesuffix("_fraction_requests")
        try:
            measurement = f"the last {float(fraction) * 100:.1f}% of requests"
        except ValueError:
            measurement = window.replace("_", " ")
    else:
        measurement = str(window or "the configured request window").replace("_", " ")

    if meta.get("underfilledBudgetPolicy") == "include_from_empty_cache":
        pressure = "underfilled budgets are included."
    else:
        pressure = "budget points that do not fill the cache before this window are omitted."
    return f"Measurement: hit rates use {measurement}; {pressure}"


def render_table(result: dict[str, Any]) -> str:
    meta = result["metadata"]
    policies = result.get("policies") or ["fifo", "lru", "optimal"]
    lines = [
        f"Model: {meta['modelLabel']} ({meta['modelId']})",
        f"KV precision: {meta['precisionLabel']}" + (f", Indexer: {meta['indexerPrecisionLabel']}" if meta.get("indexerPrecisionLabel") else ""),
        f"Trace: {format_number(meta['requestCount'])} requests, {format_number(meta['uniqueBlocks'])} prefix blocks, block size {meta['blockSize']} tokens",
        f"Accounting: {meta['bytesPerToken']:.5f} bytes/token, {meta['bytesPerBlock']:.5f} bytes/block, estimate tokens {meta['estimateTokens']}",
        f"Simulation backend: {str(meta.get('backend', 'cpp')).upper()}",
        f"Hit rate ceiling: {format_percent(result['hitRateCeiling'])}",
        measurement_description(meta),
        "Speedup: 1.0x means no-cache prefill throughput where every prefill input token is computed.",
        "",
    ]
    headers = ["Budget", "Cache blocks"]
    for policy in policies:
        headers.append(f"{policy.upper()} hit")
    for policy in policies:
        headers.append(f"{policy.upper()} speedup")
    rows: list[list[str]] = []
    for point in result.get("points", []):
        row = [f"{format_number(point['gib'])} GiB", format_number(point["cacheBlocks"])]
        for policy in policies:
            row.append(format_percent(point["results"][policy]["hitRate"]))
        for policy in policies:
            row.append(format_speedup(point["results"][policy]["idealPrefillSpeedup"]))
        rows.append(row)
    if not rows:
        lines.append("No budget point reached cache pressure before the measurement window.")
        return "\n".join(lines)
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines.append("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.rjust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def render_json(result: dict[str, Any], *, pretty: bool = True) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2 if pretty else None, sort_keys=False)


def render_summary(result: dict[str, Any], *, plot_output: str | None = None) -> str:
    meta = result.get("metadata") or {}
    policies = result.get("policies") or ["fifo", "lru", "optimal"]
    points = result.get("points") or []
    lines = [
        "Summary:",
        f"  Model: {meta.get('modelLabel') or meta.get('modelId') or 'unknown'}",
        f"  Requests: {format_number(float(meta.get('requestCount') or 0))}",
        f"  Hit rate ceiling: {format_percent(float(result.get('hitRateCeiling') or 0))}",
    ]
    if points:
        lines.append(f"  Budget points: {len(points)} ({format_number(points[0]['gib'])} GiB to {format_number(points[-1]['gib'])} GiB)")
        for policy in policies:
            best_point = max(points, key=lambda point: float(point["results"][policy]["hitRate"]))
            best_hit_rate = float(best_point["results"][policy]["hitRate"])
            lines.append(f"  Best {policy.upper()}: {format_percent(best_hit_rate)} at {format_number(best_point['gib'])} GiB")
    else:
        lines.append("  Budget points: 0")
    if plot_output:
        lines.append(f"  Plot: {plot_output}")
    return "\n".join(lines)
