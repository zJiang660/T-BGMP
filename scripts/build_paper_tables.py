from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "paper_tables"


def main() -> None:
    main_df = pd.read_csv(DATA_DIR / "table_main_evidence.csv")
    controls = pd.read_csv(DATA_DIR / "table_control_statistics.csv")
    supporting = pd.read_csv(DATA_DIR / "table_supporting_models.csv")

    restored = main_df["recovered"].str.split("/").str[0].astype(int).sum()
    sensitive = main_df["sensitive"].astype(int).sum()

    print(f"Main evidence models: {len(main_df)}")
    print(f"Conditional restoration: {restored}/{sensitive}")
    print(f"Control-statistics rows: {len(controls)}")
    print(f"Supporting/boundary-supporting rows: {len(supporting)}")
    print("Available paper-ready tables:")
    for path in sorted(DATA_DIR.glob("*.csv")):
        print(f"  - {path.name}: {len(pd.read_csv(path))} rows")


if __name__ == "__main__":
    main()
