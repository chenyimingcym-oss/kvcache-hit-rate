from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_hit_rate_sweep(
    result: dict[str, Any],
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot KV cache hit-rate curves from a sweep result JSON payload."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as exc:
        raise RuntimeError('plotting requires matplotlib; install with "kvcache-simulator[plot]"') from exc

    points = result.get("points") or []
    if not points:
        raise ValueError("cannot plot a sweep result with no budget points")

    policies = result.get("policies") or sorted(points[0].get("results", {}))
    budgets = [float(point["gib"]) for point in points]
    ceiling = result.get("hitRateCeiling")
    metadata = result.get("metadata") or {}

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for policy in policies:
        hit_rates = [float(point["results"][policy]["hitRate"]) for point in points]
        ax.plot(budgets, hit_rates, marker="o", linewidth=2, label=policy.upper())

    if ceiling is not None:
        ax.axhline(
            float(ceiling),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Unlimited / ceiling",
        )

    if all(budget > 0 for budget in budgets):
        ax.set_xscale("log", base=2)
    ax.set_xlabel("KV cache budget (GiB)")
    ax.set_ylabel("KV Cache Hit Rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.65)
    ax.legend()
    ax.set_title(title or _default_title(metadata))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _default_title(metadata: dict[str, Any]) -> str:
    model = metadata.get("modelLabel") or metadata.get("modelId") or "KV cache"
    precision = metadata.get("precisionLabel") or metadata.get("precision")
    if precision:
        return f"{model} KV cache hit rate ({precision})"
    return f"{model} KV cache hit rate"
