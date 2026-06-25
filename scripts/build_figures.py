from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "paper_tables"
FIG_DIR = ROOT / "figures" / "paper"


def percent(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.rstrip("%").astype(float).to_numpy()


def main() -> None:
    df = pd.read_csv(DATA_DIR / "table_main_evidence.csv")
    x = np.arange(len(df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(x - width, percent(df["top_rate"]), width, label="Top-risk")
    ax.bar(x, percent(df["random_rate"]), width, label="Random")
    ax.bar(x + width, percent(df["bottom_rate"]), width, label="Bottom-risk")
    ax.set_xticks(x, df["model"])
    ax.set_ylabel("Found rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Same-budget layer-selection controls")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output = FIG_DIR / "same_budget_controls_reproduced.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
