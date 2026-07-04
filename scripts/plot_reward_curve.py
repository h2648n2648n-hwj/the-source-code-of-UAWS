import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def remove_iqr_outliers(values, factor=1.5):
    series = pd.Series(values, dtype="float64")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    cleaned = series.mask((series < lower) | (series > upper))
    cleaned = cleaned.interpolate(limit_direction="both")
    return cleaned.to_numpy(), lower, upper


def smooth_reward(values, median_window=7, ema_span=18):
    series = pd.Series(values, dtype="float64")
    median = series.rolling(
        window=median_window,
        center=True,
        min_periods=1,
    ).median()
    smoothed = median.ewm(span=ema_span, adjust=False).mean()
    return smoothed.to_numpy()


def plot_reward_curve(
    input_csv,
    output_dir=None,
    reward_column="episode_mean_reward",
    remove_outliers=False,
    raw_only=False,
    resample_points=0,
    x_label=None,
):
    input_csv = Path(input_csv)
    if output_dir is None:
        output_dir = input_csv.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if reward_column not in df.columns:
        raise ValueError(f"Column '{reward_column}' was not found in {input_csv}")

    rewards = pd.to_numeric(df[reward_column], errors="coerce").dropna().to_numpy()
    if rewards.size == 0:
        raise ValueError(f"Column '{reward_column}' has no numeric reward values")

    episodes = np.arange(1, rewards.size + 1, dtype="float64")
    plot_rewards = rewards
    plot_x = episodes
    if resample_points and resample_points > 0 and resample_points != rewards.size:
        source_x = np.linspace(1, rewards.size, rewards.size)
        plot_x = np.arange(1, resample_points + 1, dtype="float64")
        target_x = np.linspace(1, rewards.size, resample_points)
        plot_rewards = np.interp(target_x, source_x, rewards)

    if remove_outliers:
        cleaned, lower, upper = remove_iqr_outliers(plot_rewards)
    else:
        cleaned = plot_rewards.copy()
        lower = float(np.nanmin(plot_rewards))
        upper = float(np.nanmax(plot_rewards))
    smoothed = smooth_reward(cleaned)
    rolling_std = pd.Series(cleaned).rolling(window=11, center=True, min_periods=1).std()
    rolling_std = rolling_std.fillna(0).to_numpy()

    stem = input_csv.stem
    processed_csv = output_dir / f"{stem}_episode_reward_smoothed.csv"
    with open(processed_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "raw_reward", "cleaned_reward", "smoothed_reward"])
        for row in zip(plot_x, plot_rewards, cleaned, smoothed):
            writer.writerow(row)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4.3, 4.0), dpi=300)
    if raw_only:
        ax.plot(
            plot_x,
            plot_rewards,
            color="#c44724",
            linewidth=1.05,
            alpha=0.98,
            label="Raw Reward",
        )
    else:
        ax.plot(
            plot_x,
            plot_rewards,
            color="#9aa0a6",
            linewidth=0.7,
            alpha=0.28,
            label="Raw reward",
        )
        ax.fill_between(
            plot_x,
            smoothed - 0.35 * rolling_std,
            smoothed + 0.35 * rolling_std,
            color="#1f77b4",
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            plot_x,
            smoothed,
            color="#1f77b4",
            linewidth=2.3,
            label="Smoothed reward",
        )

    if x_label is None:
        x_label = "Resampled training points" if resample_points else "Number of episodes"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Reward")
    ax.set_xlim(1, int(plot_x[-1]))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    suffix = "raw_reward" if raw_only else "episode_reward_smoothed"
    if resample_points:
        suffix += f"_{resample_points}points"
    png_path = output_dir / f"{stem}_{suffix}.png"
    pdf_path = output_dir / f"{stem}_{suffix}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "reward_count": int(rewards.size),
        "plot_points": int(plot_rewards.size),
        "outlier_lower": float(lower),
        "outlier_upper": float(upper),
        "remove_outliers": bool(remove_outliers),
        "processed_csv": str(processed_csv),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Plot paper-style episode reward curve.")
    parser.add_argument(
        "--input",
        default="logs/2026-06-24_16-31-41_training_curves.csv",
        help="Training curves CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated figures. Defaults to the input CSV directory.",
    )
    parser.add_argument(
        "--reward-column",
        default="episode_mean_reward",
        help="Reward column to plot. Use episode_mean_reward for episode-level curve.",
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help="Apply IQR-based outlier removal before smoothing. Disabled by default.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Plot only the selected raw reward column without smoothing.",
    )
    parser.add_argument(
        "--resample-points",
        type=int,
        default=0,
        help="Interpolate the reward curve to this many plotting points. Use only for visual resampling.",
    )
    parser.add_argument(
        "--x-label",
        default=None,
        help="Custom x-axis label.",
    )
    args = parser.parse_args()

    result = plot_reward_curve(
        args.input,
        args.output_dir,
        args.reward_column,
        remove_outliers=args.remove_outliers,
        raw_only=args.raw_only,
        resample_points=args.resample_points,
        x_label=args.x_label,
    )
    print("Reward curve generated.")
    print(f"Reward points: {result['reward_count']}")
    print(f"Plot points: {result['plot_points']}")
    if result["remove_outliers"]:
        print(f"IQR outlier range: [{result['outlier_lower']:.6f}, {result['outlier_upper']:.6f}]")
    else:
        print("Outlier removal: disabled")
    print(f"Processed CSV: {result['processed_csv']}")
    print(f"PNG: {result['png']}")
    print(f"PDF: {result['pdf']}")


if __name__ == "__main__":
    main()
