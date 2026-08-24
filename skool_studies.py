#!/usr/bin/env python3
"""Run additional tiny scaling studies for the Skool research track."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import scaling_lab as core


ROOT = Path(__file__).resolve().parent
DEFAULT_SEEDS = (7, 19, 31)


@dataclass
class Measurement:
    seed: int
    scale: int
    value: float
    exact_accuracy: float | None
    seconds: float


@dataclass
class Summary:
    scale: int
    mean: float
    standard_deviation: float


@dataclass
class Panel:
    title: str
    x_label: str
    y_label: str
    measurements: list[Measurement]
    value_format: str
    color: str


def parse_positive_list(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from error
    if len(values) < 3 or len(set(values)) != len(values) or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("provide at least three unique positive integers")
    return tuple(sorted(values))


def summarize(measurements: list[Measurement]) -> list[Summary]:
    output: list[Summary] = []
    for scale in sorted({measurement.scale for measurement in measurements}):
        values = np.asarray(
            [measurement.value for measurement in measurements if measurement.scale == scale],
            dtype=float,
        )
        output.append(
            Summary(
                scale=scale,
                mean=float(values.mean()),
                standard_deviation=float(values.std()),
            )
        )
    return output


def run_budget_study(arguments: argparse.Namespace) -> tuple[Panel, Panel, dict[str, object]]:
    training_data = core.encode(core.make_language_corpus(2026, 900))
    validation_data = core.encode(core.make_language_corpus(2027, 180))
    train_prompts, train_answers, eval_prompts, eval_answers = core.copy_prompt_split(2028, 512, 256)
    pretraining_runs: list[Measurement] = []
    rl_runs: list[Measurement] = []

    for seed in arguments.seeds:
        batches = core.fixed_batches(
            training_data,
            steps=max(max(arguments.pretrain_budgets), arguments.base_pretrain_steps),
            batch_size=arguments.batch_size,
            seed=seed,
        )
        base_model: core.TinyTransformer | None = None
        for steps in arguments.pretrain_budgets:
            model, loss, elapsed = core.train_language_model(
                arguments.width,
                batches[:steps],
                validation_data,
                seed=seed,
                learning_rate=arguments.pretrain_learning_rate,
            )
            pretraining_runs.append(Measurement(seed, steps, loss, None, elapsed))
            if steps == arguments.base_pretrain_steps:
                base_model = model

        if base_model is None:
            base_model, _, _ = core.train_language_model(
                arguments.width,
                batches[: arguments.base_pretrain_steps],
                validation_data,
                seed=seed,
                learning_rate=arguments.pretrain_learning_rate,
            )

        for steps in arguments.rl_budgets:
            _, exact_after, _, reward_after, elapsed = core.rl_post_train(
                base_model,
                train_prompts,
                train_answers,
                eval_prompts,
                eval_answers,
                steps=steps,
                batch_size=arguments.batch_size,
                seed=seed + 1000,
                learning_rate=arguments.rl_learning_rate,
            )
            rl_runs.append(Measurement(seed, steps, 1.0 - reward_after, exact_after, elapsed))

    controls = {
        "study": "training_budget",
        "controlled_variable": "number of optimizer updates",
        "fixed_width": arguments.width,
        "fixed_pretraining_data_characters": len(training_data),
        "fixed_rl_training_prompts": len(train_prompts),
        "batch_size": arguments.batch_size,
        "replicate_seeds": list(arguments.seeds),
    }
    return (
        Panel(
            "1. Pretraining update sweep",
            "pretraining updates (log scale)",
            "validation cross-entropy ↓",
            pretraining_runs,
            "{:.3f}",
            "#2563eb",
        ),
        Panel(
            "2. RL update sweep",
            "RL updates (log scale)",
            "reward gap (1 - mean reward) ↓",
            rl_runs,
            "{:.1%}",
            "#dc2626",
        ),
        controls,
    )


def run_data_study(arguments: argparse.Namespace) -> tuple[Panel, Panel, dict[str, object]]:
    validation_data = core.encode(core.make_language_corpus(2027, 180))
    full_training_prompts, full_training_answers, eval_prompts, eval_answers = core.copy_prompt_split(
        2028, max(arguments.rl_data_sizes), 256
    )
    pretraining_runs: list[Measurement] = []
    rl_runs: list[Measurement] = []
    character_counts: dict[int, int] = {}

    for seed in arguments.seeds:
        base_model: core.TinyTransformer | None = None
        for line_count in arguments.pretrain_data_lines:
            training_data = core.encode(core.make_language_corpus(2026, line_count))
            character_counts[line_count] = len(training_data)
            batches = core.fixed_batches(
                training_data,
                steps=arguments.base_pretrain_steps,
                batch_size=arguments.batch_size,
                seed=seed,
            )
            model, loss, elapsed = core.train_language_model(
                arguments.width,
                batches,
                validation_data,
                seed=seed,
                learning_rate=arguments.pretrain_learning_rate,
            )
            pretraining_runs.append(Measurement(seed, len(training_data), loss, None, elapsed))
            if line_count == max(arguments.pretrain_data_lines):
                base_model = model

        assert base_model is not None
        for prompt_count in arguments.rl_data_sizes:
            _, exact_after, _, reward_after, elapsed = core.rl_post_train(
                base_model,
                full_training_prompts[:prompt_count],
                full_training_answers[:prompt_count],
                eval_prompts,
                eval_answers,
                steps=arguments.base_rl_steps,
                batch_size=arguments.batch_size,
                seed=seed + 1000,
                learning_rate=arguments.rl_learning_rate,
            )
            rl_runs.append(Measurement(seed, prompt_count, 1.0 - reward_after, exact_after, elapsed))

    controls = {
        "study": "data_size",
        "controlled_variable": "available unique training data",
        "fixed_width": arguments.width,
        "fixed_pretraining_steps": arguments.base_pretrain_steps,
        "fixed_rl_steps": arguments.base_rl_steps,
        "batch_size": arguments.batch_size,
        "pretraining_line_to_character_counts": {
            str(lines): character_counts[lines] for lines in arguments.pretrain_data_lines
        },
        "replicate_seeds": list(arguments.seeds),
    }
    return (
        Panel(
            "1. Pretraining data sweep",
            "training corpus size in characters (log scale)",
            "validation cross-entropy ↓",
            pretraining_runs,
            "{:.3f}",
            "#2563eb",
        ),
        Panel(
            "2. RL data sweep",
            "unique rewarded training prompts (log scale)",
            "reward gap (1 - mean reward) ↓",
            rl_runs,
            "{:.1%}",
            "#dc2626",
        ),
        controls,
    )


def _format_scale(value: int) -> str:
    if value >= 10_000:
        return f"{value / 1000:.0f}k"
    return f"{value:,}"


def _plot_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    panel: Panel,
    seed_count: int,
) -> tuple[list[Summary], float]:
    left, top, right, bottom = box
    plot_left, plot_top, plot_right, plot_bottom = left + 105, top + 125, right - 35, bottom - 90
    summaries = summarize(panel.measurements)
    scales = [summary.scale for summary in summaries]
    means = [summary.mean for summary in summaries]
    deviations = [summary.standard_deviation for summary in summaries]
    slope, _ = core.descriptive_power_slope(scales, means)

    core._text(draw, (left + 8, top + 8), panel.title, size=30, fill="#111827", bold=True)
    core._text(draw, (left + 8, top + 49), f"mean ± 1 SD over {seed_count} seeds", size=18, fill="#64748b")
    core._text(
        draw,
        (left + 8, top + 76),
        f"descriptive log-log slope: {slope:+.2f}",
        size=16,
        fill="#64748b",
    )
    core._text(draw, (plot_left, plot_top - 12), panel.y_label, size=17, fill="#334155", bold=True, anchor="ls")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#94a3b8", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#94a3b8", width=2)

    log_x = np.log10(np.asarray(scales, dtype=float))
    x_min, x_max = float(log_x.min()), float(log_x.max())
    observed_low = min(mean - deviation for mean, deviation in zip(means, deviations))
    observed_high = max(mean + deviation for mean, deviation in zip(means, deviations))
    padding = max((observed_high - observed_low) * 0.15, 0.015)
    lower = max(0.0, observed_low - padding)
    upper = observed_high + padding
    if panel.y_label.startswith("reward gap"):
        upper = min(1.0, upper)

    def x_pixel(value: float) -> float:
        fraction = (math.log10(value) - x_min) / max(x_max - x_min, 1e-9)
        return plot_left + fraction * (plot_right - plot_left)

    def y_pixel(value: float) -> float:
        fraction = (value - lower) / max(upper - lower, 1e-9)
        return plot_bottom - fraction * (plot_bottom - plot_top)

    for tick in np.linspace(lower, upper, 5):
        y = y_pixel(float(tick))
        draw.line((plot_left, y, plot_right, y), fill="#e2e8f0", width=1)
        core._text(draw, (plot_left - 14, y), f"{tick:.2f}", size=17, fill="#475569", anchor="ra")

    points = [(x_pixel(scale), y_pixel(mean)) for scale, mean in zip(scales, means)]
    for (x, _), mean, deviation in zip(points, means, deviations):
        error_top = y_pixel(mean + deviation)
        error_bottom = y_pixel(max(0.0, mean - deviation))
        draw.line((x, error_top, x, error_bottom), fill="#ffffff", width=11)
        draw.line((x - 16, error_top, x + 16, error_top), fill="#ffffff", width=11)
        draw.line((x - 16, error_bottom, x + 16, error_bottom), fill="#ffffff", width=11)
        draw.line((x, error_top, x, error_bottom), fill=panel.color, width=6)
        draw.line((x - 14, error_top, x + 14, error_top), fill=panel.color, width=6)
        draw.line((x - 14, error_bottom, x + 14, error_bottom), fill=panel.color, width=6)

    draw.line(points, fill=panel.color, width=5, joint="curve")
    for (x, y), scale, mean in zip(points, scales, means):
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#ffffff", outline=panel.color, width=5)
        core._text(draw, (x, y - 20), panel.value_format.format(mean), size=17, fill="#0f172a", bold=True, anchor="ms")
        core._text(draw, (x, plot_bottom + 15), _format_scale(scale), size=16, fill="#475569", anchor="ma")
    core._text(
        draw,
        ((plot_left + plot_right) / 2, bottom - 38),
        panel.x_label,
        size=18,
        fill="#334155",
        anchor="mm",
    )
    return summaries, slope


def save_chart(
    study: str,
    first: Panel,
    second: Panel,
    image_path: Path,
    seed_count: int,
) -> tuple[list[Summary], list[Summary], float, float]:
    canvas = Image.new("RGB", (1600, 940), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    title = "Tiny scaling study: training budget" if study == "budget" else "Tiny scaling study: data size"
    subtitle = (
        "Fixed model and data. Only the optimizer-update budget changes."
        if study == "budget"
        else "Fixed model and update budget. Only available training data changes."
    )
    core._text(draw, (70, 55), title, size=48, fill="#0f172a", bold=True)
    core._text(draw, (70, 120), subtitle, size=23, fill="#475569")
    first_summary, first_slope = _plot_panel(draw, (55, 185, 800, 790), first, seed_count)
    second_summary, second_slope = _plot_panel(draw, (800, 185, 1545, 790), second, seed_count)
    draw.rounded_rectangle((70, 825, 1530, 900), radius=18, fill="#e2e8f0")
    core._text(
        draw,
        (800, 862),
        "Vertical bars = ±1 SD. A small controlled sweep, not a universal scaling law.",
        size=23,
        fill="#334155",
        bold=True,
        anchor="mm",
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(image_path)
    return first_summary, second_summary, first_slope, second_slope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("budget", "data"))
    parser.add_argument("--seeds", type=core.parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pretrain-budgets", type=parse_positive_list, default=(5, 15, 30, 60))
    parser.add_argument("--rl-budgets", type=parse_positive_list, default=(10, 25, 50, 100))
    parser.add_argument("--pretrain-data-lines", type=parse_positive_list, default=(3, 10, 50, 900))
    parser.add_argument("--rl-data-sizes", type=parse_positive_list, default=(16, 64, 256, 512))
    parser.add_argument("--base-pretrain-steps", type=int, default=30)
    parser.add_argument("--base-rl-steps", type=int, default=50)
    parser.add_argument("--pretrain-learning-rate", type=float, default=0.008)
    parser.add_argument("--rl-learning-rate", type=float, default=0.008)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--no-open", action="store_true")
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not 4 <= arguments.width <= 128:
        raise SystemExit("--width must be from 4 to 128")
    if arguments.batch_size < 1 or arguments.base_pretrain_steps < 1 or arguments.base_rl_steps < 1:
        raise SystemExit("batch size and step counts must be positive")
    if max(arguments.rl_data_sizes) > 4096:
        raise SystemExit("--rl-data-sizes values must not exceed 4096")
    for value in (arguments.pretrain_learning_rate, arguments.rl_learning_rate):
        if not 0 < value <= 0.2:
            raise SystemExit("learning rates must be in (0, 0.2]")


def main() -> int:
    arguments = build_parser().parse_args()
    validate_arguments(arguments)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()

    if arguments.study == "budget":
        first, second, controls = run_budget_study(arguments)
        default_stem = "my_training_budget_scaling"
    else:
        first, second, controls = run_data_study(arguments)
        default_stem = "my_data_scaling"

    image_path = (arguments.image or ROOT / f"{default_stem}.png").resolve()
    receipt_path = (arguments.receipt or ROOT / f"{default_stem}.json").resolve()
    first_summary, second_summary, first_slope, second_slope = save_chart(
        arguments.study,
        first,
        second,
        image_path,
        len(arguments.seeds),
    )
    elapsed = time.perf_counter() - started
    receipt = {
        "claim_scope": "small multi-seed educational sweep; not a universal scaling law",
        "controls": controls,
        "first_panel": {
            "title": first.title,
            "metric": first.y_label,
            "descriptive_log_log_slope": first_slope,
            "summary": [asdict(item) for item in first_summary],
            "per_seed": [asdict(item) for item in first.measurements],
        },
        "second_panel": {
            "title": second.title,
            "metric": second.y_label,
            "descriptive_log_log_slope": second_slope,
            "summary": [asdict(item) for item in second_summary],
            "per_seed": [asdict(item) for item in second.measurements],
        },
        "total_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(f"\n{first.title}")
    for item in first_summary:
        print(f"  {_format_scale(item.scale):>7}: {item.mean:.3f} ± {item.standard_deviation:.3f}")
    print(f"\n{second.title}")
    for item in second_summary:
        print(f"  {_format_scale(item.scale):>7}: {item.mean:.1%} ± {item.standard_deviation:.1%}")
    print(f"\nFinished in {elapsed:.2f} seconds")
    print(f"Image:   {image_path}")
    print(f"Receipt: {receipt_path}")
    if not arguments.no_open:
        core.open_image(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
