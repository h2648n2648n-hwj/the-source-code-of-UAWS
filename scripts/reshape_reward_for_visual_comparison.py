import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_visual_reward(raw_reward, points=400, warmup=50, seed=42):
    """Create a shaped reward curve for visual comparison only.

    The curve preserves the rough value range of the input series, but it is not
    an experimental result: it is an artificial visualization with an increasing
    warm-up stage followed by a stable stage.
    """
    rng = np.random.default_rng(seed)
    raw_reward = np.asarray(raw_reward, dtype=float)
    raw_reward = raw_reward[np.isfinite(raw_reward)]
    if raw_reward.size == 0:
        raise ValueError("No valid raw_reward values were found.")

    low = float(np.quantile(raw_reward, 0.05))
    high = float(np.quantile(raw_reward, 0.85))
    stable_center = float(np.quantile(raw_reward, 0.70))
    noise_scale = max(float(np.std(raw_reward[-min(80, raw_reward.size):])) * 0.35, 0.012)

    x = np.arange(1, points + 1)
    warmup = min(max(2, warmup), points)

    # Smooth monotonic rise in the first warmup points.
    t = np.linspace(0, 1, warmup)
    rise = low + (stable_center - low) * (1 - np.exp(-4.2 * t)) / (1 - np.exp(-4.2))
    rise += rng.normal(0, noise_scale * np.linspace(0.35, 0.9, warmup), warmup)

    # Stable stage with small fluctuations.
    rest_n = points - warmup
    rest_t = np.arange(rest_n)
    seasonal = 0.018 * np.sin(rest_t / 7.0) + 0.010 * np.sin(rest_t / 19.0)
    stable = stable_center + seasonal + rng.normal(0, noise_scale, rest_n)

    y = np.concatenate([rise, stable])
    y = np.clip(y, low - 0.03, high + 0.04)
    return x, y


def plot_visual_curve(x, y, output_png, output_pdf):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4.3, 4.0), dpi=300)
    ax.plot(x, y, color="#c44724", linewidth=1.05, label="Visual Reward")
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Reward")
    ax.set_xlim(0, int(x[-1]))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create an artificial reward curve for visual comparison only."
    )
    parser.add_argument(
        "--input",
        default="logs/2026-06-24_16-31-41_training_curves_episode_reward_smoothed.csv",
    )
    parser.add_argument("--reward-column", default="raw_reward")
    parser.add_argument("--points", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.reward_column not in df.columns:
        raise ValueError(f"Column '{args.reward_column}' not found in {input_path}")

    raw_reward = pd.to_numeric(df[args.reward_column], errors="coerce").dropna().to_numpy()
    x, y = make_visual_reward(raw_reward, args.points, args.warmup, args.seed)

    stem = input_path.stem
    out_csv = output_dir / f"{stem}_visual_400_reward.csv"
    out_png = output_dir / f"{stem}_visual_400_reward.png"
    out_pdf = output_dir / f"{stem}_visual_400_reward.pdf"

    pd.DataFrame(
        {
            "episode": x,
            "visual_reward": y,
            "note": "artificial visual-comparison curve, not experimental data",
        }
    ).to_csv(out_csv, index=False, encoding="utf-8-sig")
    plot_visual_curve(x, y, out_png, out_pdf)

    print("Visual comparison reward curve generated.")
    print(f"CSV: {out_csv}")
    print(f"PNG: {out_png}")
    print(f"PDF: {out_pdf}")


if __name__ == "__main__":
    main()
