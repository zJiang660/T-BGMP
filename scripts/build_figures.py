from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "paper_tables"
FIG_DIR = ROOT / "figures" / "paper"

MODELS = ("Qwen3-4B", "Qwen2.5-3B", "Qwen2.5-14B", "Llama3.2-3B")
DOMAINS = ("code", "literature", "math", "science")
STYLE = {
    "code": ("Code", "#2C9BE8", "-", "o"),
    "literature": ("Literature", "#4CAF68", "--", "s"),
    "math": ("Math", "#F39C12", "-.", "^"),
    "science": ("Science", "#F2385A", ":", "D"),
}


def main() -> None:
    data = pd.read_csv(DATA_DIR / "figure_2_domain_recovery.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.68))
    budgets = list(range(1, 13))

    for ax, model in zip(axes.flat, MODELS):
        model_rows = data[data["model"] == model].set_index("domain")
        for domain in DOMAINS:
            label, color, linestyle, marker = STYLE[domain]
            values = [float(model_rows.loc[domain, f"k{k}"]) for k in budgets]
            ax.plot(
                budgets,
                values,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=4.5,
                markeredgewidth=0.8,
                markerfacecolor="white" if domain in {"literature", "science"} else color,
            )
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Top-k budget")
        ax.set_ylabel("Cumulative recovery rate (%)")
        ax.set_xlim(1, 12.2)
        ax.set_ylim(0, 104)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, color="#D8D8D8", linewidth=0.6, alpha=0.75)

    axes[0, 0].legend(
        loc="lower right", frameon=True, framealpha=0.96, edgecolor="#D0D0D0"
    )
    fig.subplots_adjust(
        left=0.095, right=0.985, bottom=0.09, top=0.95, wspace=0.20, hspace=0.25
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "figure_2_domain_recovery.png"
    pdf = FIG_DIR / "figure_2_domain_recovery.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Generated: {png.relative_to(ROOT)}")
    print(f"Generated: {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
