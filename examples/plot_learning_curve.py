"""Plot a training run's learning curves from its TensorBoard logs.

    uv run --no-sync python examples/plot_learning_curve.py runs/maskable_ppo/<run> \
        --baselines examples/results/eval_results.json --out examples/results/learning_curve

Writes ``<out>.png`` (light), ``<out>_dark.png`` and ``<out>.csv``, the plotted values as a
table with one row per evaluation.  The baselines file is evaluate_agents.py's JSON: its random
and smart agents are drawn as reference lines.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

#: (panel title, training-rollout tag, greedy-eval tag, sampled-eval tag, baseline key)
PANELS = [
    ("Blinds beaten per episode", "balatro/blinds_beaten", "eval/blinds_beaten",
     "eval_sampled/blinds_beaten", "blinds_beaten"),
    ("Ante reached", "balatro/ante_reached", "eval/ante_reached",
     "eval_sampled/ante_reached", "ante_reached"),
    ("Episode reward", "rollout/ep_rew_mean", "eval/reward", "eval_sampled/reward", "reward"),
    ("Episode length (steps, log scale)", "rollout/ep_len_mean", "eval/length",
     "eval_sampled/length", "length"),
]  # fmt: skip

SERIES = ("Training rollouts", "Evaluation, sampled", "Evaluation, greedy")

#: Chart tokens (the dataviz reference palette): surfaces, ink and the first three
#: categorical slots, stepped separately for each mode.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70"),
    },
}


def read_scalars(run: Path) -> dict[str, list[tuple[int, float]]]:
    """Every scalar of the run's TensorBoard event files, as ``tag -> [(step, value)]``."""
    out: dict[str, dict[int, float]] = {}
    for path in sorted(run.rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            out.setdefault(tag, {}).update({e.step: e.value for e in acc.Scalars(tag)})
    return {tag: sorted(points.items()) for tag, points in out.items()}


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over up to ``window`` points."""
    cum = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(1, len(values) + 1)
    lo = np.maximum(idx - window, 0)
    return (cum[idx] - cum[lo]) / (idx - lo)


def plot(
    scalars: dict[str, list[tuple[int, float]]],
    baselines: dict[str, dict[str, float]],
    theme: dict,
    path: Path,
    window: int,
) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
        }
    )
    fig = Figure(figsize=(11, 7.2), facecolor=theme["surface"])
    axes = fig.subplots(2, 2)
    for ax, (title, train_tag, greedy_tag, sampled_tag, key) in zip(axes.flat, PANELS, strict=True):
        ax.set_facecolor(theme["surface"])
        # Marker shapes back up the colors: none, circles, squares.
        for name, tag, color, marker in zip(
            SERIES, (train_tag, sampled_tag, greedy_tag), theme["series"], (None, "o", "s"),
            strict=True,
        ):  # fmt: skip
            points = scalars.get(tag, [])
            if not points:
                continue
            x = np.array([p[0] for p in points]) / 1e6
            y = np.array([p[1] for p in points])
            if tag == train_tag:
                y = smooth(y, window)
            ax.plot(
                x, y, color=color, lw=1.6, label=name, marker=marker, ms=5,
                mec=theme["surface"], mew=1.2, solid_capstyle="round", solid_joinstyle="round",
            )  # fmt: skip
        # The baselines: reference lines labelled in the right margin.
        for agent, label in (("smart", "smart_agent"), ("random", "random")):
            if agent in baselines and key in baselines[agent]:
                value = baselines[agent][key]
                ax.axhline(value, color=theme["muted"], lw=1.0, zorder=1)
                ax.annotate(
                    label, (1.0, value), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5,
                    color=theme["ink2"], annotation_clip=False,
                )  # fmt: skip
        ax.set_title(title, loc="left", fontsize=11, color=theme["ink"], pad=8)
        ax.grid(True, color=theme["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(theme["axis"])
        ax.tick_params(colors=theme["muted"], labelcolor=theme["ink2"], length=0, pad=4)
        ax.set_xlabel("Environment steps (millions)", color=theme["ink2"], fontsize=9)
        if key == "length":
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 500, 1000, 2000]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.yaxis.set_minor_locator(NullLocator())
        ax.margins(x=0.03)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.01, 0.995),
        fontsize=9.5, labelcolor=theme["ink"], handlelength=1.6,
    )  # fmt: skip
    legend.set_zorder(5)
    fig.text(
        0.99, 0.985,
        f"Training: rolling mean over {window} rollouts.  Evaluations: 64 validation seeds.  "
        "Gray lines: the baselines on the test seeds.",
        ha="right", va="top", fontsize=8.5, color=theme["muted"],
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2, w_pad=2.0)
    fig.savefig(path, dpi=150, facecolor=theme["surface"])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", help="the run directory (holds tb/)")
    p.add_argument("--baselines", default=None, help="evaluate_agents.py's JSON")
    p.add_argument("--out", default="examples/results/learning_curve", help="output stem")
    p.add_argument("--window", type=int, default=50, help="rollouts in the training mean")
    args = p.parse_args(argv)

    scalars = read_scalars(Path(args.run))
    baselines = {}
    if args.baselines:
        baselines = json.loads(Path(args.baselines).read_text())["agents"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(scalars, baselines, THEMES["light"], out.with_suffix(".png"), args.window)
    plot(scalars, baselines, THEMES["dark"], out.with_name(out.name + "_dark.png"), args.window)

    write_table(scalars, out.with_suffix(".csv"))
    print(f"wrote {out}.png, {out}_dark.png, {out}.csv")


def write_table(scalars: dict[str, list[tuple[int, float]]], path: Path) -> None:
    """The plotted values as a table, one row per evaluation: the two evaluations' values, and
    each training-rollout metric averaged over the rollouts since the previous evaluation."""
    eval_steps = [step for step, _ in scalars.get(PANELS[0][2], [])]
    columns = []
    for _, train_tag, greedy_tag, sampled_tag, key in PANELS:
        columns += [(f"train_{key}", train_tag), (f"greedy_{key}", greedy_tag)]
        columns.append((f"sampled_{key}", sampled_tag))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *(name for name, _ in columns)])
        previous = -1
        for step in eval_steps:
            row: list[str] = [str(step)]
            for name, tag in columns:
                points = scalars.get(tag, [])
                if name.startswith("train_"):
                    values = [v for s, v in points if previous < s <= step]
                else:
                    values = [v for s, v in points if s == step]
                row.append(f"{np.mean(values):.4g}" if values else "")
            writer.writerow(row)
            previous = step


if __name__ == "__main__":
    main()
